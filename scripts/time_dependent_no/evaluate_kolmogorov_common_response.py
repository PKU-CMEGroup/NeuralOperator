"""Solver-free common-input assay of the two frozen Kolmogorov Clean maps.

Run ``python -m scripts.time_dependent_no.evaluate_kolmogorov_common_response
--help``. Validation makes the shared bank without constructing a model. Assay
replays and evaluates both immutable terminals; it does not compose new paths.
The target is the archived clean successor, NOT an off-state solver successor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_coverage_rollout as expanded
from utility.time_dependent_no.path_conditioned_tube import (
    path_conditioned_tube_metrics,
)

clean, baseline, fit = expanded.clean, expanded.baseline, expanded.coverage
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "CM_NEXT_KF_COMMON_RESPONSE_20260910C"
MODELS = ("clean8", "clean32")
STEPS = (1, 8, 32, 64)  # Output steps; recurrent inputs are at k-1.
WALL_SECONDS = 1800
REPLAY_LIMIT = 1e-6
FLOAT32_RESOLUTION_LIMIT = 1e-4
ROLLOUT_PINS = {
    "clean8": (
        baseline.RUN_ID,
        "43fef61913e62650169bc8d0a298eb89046f44783c09f492bd89eb75719f985b",
    ),
    "clean32": (
        expanded.RUN_ID,
        "63ed02808bd7ce68126e685c7e51e9ce18156aae606ce1c695bfacd57883cd9f",
    ),
}
SOURCE_PATHS = (
    "scripts/time_dependent_no/evaluate_kolmogorov_common_response.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_common_response.py",
    "utility/time_dependent_no/path_conditioned_tube.py",
    "tests/time_dependent_no/test_path_conditioned_tube.py",
    *expanded.SOURCE_PATHS,
)


def array_hash(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def calibrate(teacher_file, index, *, steps, nodes, train_scale):
    """One physical RMS from all expanded training pairs; validation is excluded."""
    expected = fit.extension.population_index(fit.extension.NEW_SEEDS)
    if index != expected or steps != 512 or nodes != 256**2:
        raise ValueError("calibration requires the frozen 32/4 population")
    with np.load(teacher_file, allow_pickle=False) as saved:
        if not np.array_equal(
            saved["trajectory_index"], np.repeat(np.arange(36), steps)
        ) or not np.array_equal(saved["input_step"], np.tile(np.arange(steps), 36)):
            raise ValueError("calibration teacher index/time mismatch")
        errors = saved["learned_sse"]
        if (
            errors.shape != (36 * steps,)
            or errors.dtype != np.float64
            or not np.isfinite(errors).all()
            or np.any(errors < 0)
        ):
            raise ValueError("invalid calibration squared errors")
        selected = errors.reshape(36, steps)[:32]
        rms = float(np.sqrt(selected.sum() / (32 * steps * nodes)))
    if (
        not np.isfinite(rms)
        or rms <= 0
        or not np.isfinite(train_scale)
        or train_scale <= 0
    ):
        raise ValueError("calibration RMS and train scale must be finite positive")
    return {
        "physical_rms": rms,
        "rms_over_train_scale": rms / train_scale,
        "training_pairs": 32 * steps,
        "validation_pairs": 0,
        "rule": "sqrt(sum expanded terminal training SSE / (32*512*256^2))",
        "teacher_sha256": clean._hash(teacher_file),
    }


def read_rollout(packet, model_name, model_evidence, parent_evidence, scale):
    """Validate the closed packet before decoding any donor snapshots."""
    run_id, digest = ROLLOUT_PINS[model_name]
    if clean._hash(packet / "artifact_manifest.json") != digest:
        raise ValueError("rollout manifest differs from the frozen donor")
    manifest = clean._read(packet / "artifact_manifest.json")
    evidence = {
        "manifest_sha256": digest,
        "artifacts": {**manifest["artifacts"], "artifact_manifest.json": digest},
        "sources": manifest["sources"],
    }
    if not fit.evidence_stable(packet, REPO_ROOT, evidence):
        raise ValueError("rollout artifacts/live source mismatch")
    record = clean._read(packet / "result.json")
    if (
        record["run_id"] != run_id
        or manifest["run_id"] != run_id
        or record["status"] != "completed"
        or manifest["status"] != "completed"
        or record["sources_before"] != record["sources_after"]
        or record["sources_before"] != manifest["sources"]
        or record["source_stable"] is not True
        or manifest["source_stable"] is not True
        or record["train_scale"] != scale
        or record["horizon"] != 512
        or [(c["seed"], c["role"]) for c in record["cases"]] != list(clean.SEED_ROLES)
        or record["checkpoint_replay"]["passed"] is not True
        or record["solver_calls"] != 0
        or record["optimization_steps"] != 0
        or any(
            record[k] is not False
            for k in (
                "protected_access",
                "network_access",
                "geometry_fitted",
                "corrections_added",
            )
        )
    ):
        raise ValueError("rollout identity, completion, split or access mismatch")
    if model_name == "clean8":
        source_model = record["clean_evidence"]
        source_parent = record["population_evidence"]
        stable = record["parent_evidence_stable"]
    else:
        source_model = record["coverage_evidence"]
        source_parent = record["data"]["evidence"]
        stable = record["input_evidence_stable"]
    if (
        source_model != model_evidence
        or source_parent != parent_evidence
        or stable is not True
    ):
        raise ValueError("rollout donor is not bound to the selected model/population")
    return record, evidence


def read_snapshots(packet, case, reference, *, steps=STEPS, deadline=float("inf")):
    fit.check_deadline(deadline, "snapshot hash validation")
    path = clean._bound(packet, case["snapshot_file"])
    if clean._hash(path) != case["snapshot_sha256"]:
        raise ValueError("snapshot hash mismatch")
    fit.check_deadline(deadline, "snapshot decoding")
    with np.load(path, allow_pickle=False) as saved:
        times = saved["step"]
        if (
            times.ndim != 1
            or times.dtype.kind not in "iu"
            or np.any(times[1:] <= times[:-1])
        ):
            raise ValueError("snapshot steps must be unique ordered integers")
        indices = []
        for step in steps:
            found = np.flatnonzero(times == step)
            if found.size != 1 or step > case["completed_steps"]:
                raise ValueError("missing common pre-guard snapshot")
            indices.append(int(found[0]))
        fields = {}
        for name in ("previous_input", "raw_output", "next_state", "truth"):
            fit.check_deadline(deadline, "snapshot field decoding")
            values = saved[name]
            if (
                values.dtype != np.float32
                or values.shape != (len(times), *reference.shape[1:])
                or not np.isfinite(values).all()
            ):
                raise ValueError("nonfinite or mistyped snapshot field")
            fields[name] = values[indices].copy()
            fit.check_deadline(deadline, "snapshot field validation")
    if not np.array_equal(fields["truth"], reference[list(steps)]):
        raise ValueError("snapshot truth differs from the actual C reference")
    if steps[0] == 1 and not np.array_equal(fields["previous_input"][0], reference[0]):
        raise ValueError("first snapshot input is not the reference initial state")
    return fields


def make_bank(reference, donors, physical_rms, *, steps=STEPS, deadline=float("inf")):
    """Preserve natural inputs bitwise; rescale directions in float64 then cast."""
    if not np.isfinite(physical_rms) or physical_rms <= 0:
        raise ValueError("matched RMS must be finite positive")
    reference_input = reference[np.asarray(steps) - 1].copy()
    reference_next = reference[list(steps)].copy()
    if (
        reference_input.dtype != np.float32
        or reference_input.ndim != 3
        or reference_input.shape[-1] != reference_input.shape[-2]
        or not np.isfinite(reference_input).all()
        or not np.isfinite(reference_next).all()
        or set(donors) != set(MODELS)
    ):
        raise ValueError("bank requires aligned finite float32 reference/donors")
    probes, queries, skipped = [], [], []
    for anchor, step in enumerate(steps):
        u = reference_input[anchor]
        for donor in MODELS:
            fit.check_deadline(deadline, "common-bank query construction")
            x = donors[donor]["previous_input"][anchor]
            if x.shape != u.shape or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError("donor input shape/dtype/finiteness mismatch")
            delta = x.astype(np.float64) - u
            rms = float(np.sqrt(np.mean(delta**2)))
            values = [("natural", x.copy())]
            vector_error = None
            if rms > 0:
                intended = delta * (physical_rms / rms)
                matched = (u.astype(np.float64) + intended).astype(np.float32)
                realized = matched.astype(np.float64) - u
                actual = float(np.sqrt(np.mean(realized**2)))
                vector_error = float(
                    np.sqrt(np.mean((realized - intended) ** 2)) / physical_rms
                )
                if (
                    not np.isfinite(matched).all()
                    or not np.isfinite(vector_error)
                    or abs(actual / physical_rms - 1) > FLOAT32_RESOLUTION_LIMIT
                    or vector_error > FLOAT32_RESOLUTION_LIMIT
                ):
                    raise ValueError("matched displacement is unresolved in float32")
                values.append(("matched_rms", matched))
            else:
                skipped.append(
                    {"step": step, "donor": donor, "reason": "zero_direction"}
                )
            for view, value in values:
                queries.append(
                    {
                        "query_index": len(probes),
                        "anchor": anchor,
                        "output_step": step,
                        "input_step": step - 1,
                        "donor": donor,
                        "view": view,
                        "input_sha256": array_hash(value),
                        "natural_physical_rms": rms,
                        "matched_relative_vector_error": vector_error
                        if view == "matched_rms"
                        else None,
                    }
                )
                probes.append(value)
    fit.check_deadline(deadline, "common-bank construction completion")
    return {
        "arrays": {
            "reference_input": reference_input,
            "reference_next": reference_next,
            "probe_input": np.stack(probes),
        },
        "queries": queries,
        "skipped_matched_rms": skipped,
    }


def measure(u, x, clean_prediction, displaced_prediction, truth, scale, modes):
    fields = [
        np.asarray(a, dtype=np.float64)
        for a in (u, x, clean_prediction, displaced_prediction, truth)
    ]
    u, x, a, b, truth = fields
    metrics = path_conditioned_tube_metrics(
        reference_input=u.reshape(-1, 1),
        displaced_input=x.reshape(-1, 1),
        reference_prediction=a.reshape(-1, 1),
        displaced_prediction=b.reshape(-1, 1),
        reference_next=truth.reshape(-1, 1),
        node_weights=np.ones(u.size),
        component_scale=[scale],
    )
    metrics["path_recovery_error_scaled_rms"] = metrics.pop(
        "displaced_defect_scaled_rms"
    )
    metrics["path_error_excess_slope"] = metrics.pop("tube_escape_slope")
    if (
        metrics["energy_identity_relative"] > 1e-10
        or (metrics["secant_bound_relative_violation"] or 0) > 1e-10
    ):
        raise ArithmeticError("response algebra closure failed")
    metrics["fourier_band_scaled_energy"] = {
        name: (baseline.spectral_error_sse(value, modes) / (u.size * scale**2)).tolist()
        for name, value in (
            ("input_displacement", x - u),
            ("clean_forcing", a - truth),
            ("learned_response", b - a),
            ("path_recovery_error", b - truth),
        )
    }
    metrics["prediction_structure"] = baseline.field_diagnostics(b)
    return metrics


def replay_error(actual, expected):
    numerator = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)))
    denominator = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    relative = (
        numerator / denominator
        if denominator
        else (0.0 if numerator == 0 else float("inf"))
    )
    if not np.isfinite(relative) or relative > REPLAY_LIMIT:
        raise RuntimeError("natural donor snapshot replay failed")
    return relative


def evaluate_bank(model, bank, own_snapshots, recipient, scale, modes, deadline):
    arrays = bank["arrays"]
    device = next(model.parameters()).device
    outputs = {
        key: [] for key in ("clean_raw", "clean_next", "probe_raw", "probe_next")
    }
    calls = 0

    def predict(x):
        nonlocal calls
        fit.check_deadline(deadline, "common-input prediction")
        tensor = torch.from_numpy(x.copy()).unsqueeze(0).to(device)
        fit.check_deadline(deadline, "common-input transfer completion")
        calls += 1
        with torch.no_grad():
            result = model(tensor)
        values = [
            result[key][0].detach().cpu().numpy().copy()
            for key in ("raw_next", "next_state")
        ]
        if any(
            v.shape != x.shape or v.dtype != np.float32 or not np.isfinite(v).all()
            for v in values
        ):
            raise ValueError("invalid common-input model output")
        fit.check_deadline(deadline, "common-input prediction completion")
        return values

    for u in arrays["reference_input"]:
        raw, projected = predict(u)
        outputs["clean_raw"].append(raw)
        outputs["clean_next"].append(projected)
    rows, replays = [], []
    for query, x in zip(bank["queries"], arrays["probe_input"], strict=True):
        anchor = query["anchor"]
        u, truth = arrays["reference_input"][anchor], arrays["reference_next"][anchor]
        # Deterministic zero displacement shares the identical clean evaluation.
        if np.array_equal(x, u):
            raw, projected = outputs["clean_raw"][anchor], outputs["clean_next"][anchor]
        else:
            raw, projected = predict(x)
        outputs["probe_raw"].append(raw)
        outputs["probe_next"].append(projected)
        if query["donor"] == recipient and query["view"] == "natural":
            replays.append(
                {
                    "output_step": query["output_step"],
                    "raw_relative_rms": replay_error(
                        raw, own_snapshots["raw_output"][anchor]
                    ),
                    "next_relative_rms": replay_error(
                        projected, own_snapshots["next_state"][anchor]
                    ),
                }
            )
        restriction = float(
            np.sqrt(np.mean((raw.astype(np.float64) - projected) ** 2)) / scale
        )
        for kind, output in (("raw", raw), ("next", projected)):
            rows.append(
                {
                    **query,
                    "recipient": recipient,
                    "output_kind": kind,
                    "restriction_scaled_rms": restriction,
                    **measure(
                        u,
                        x,
                        outputs["clean_" + kind][anchor],
                        output,
                        truth,
                        scale,
                        modes,
                    ),
                }
            )
    return {k: np.stack(v) for k, v in outputs.items()}, rows, replays, calls


def load_inputs(paths, record, bindings, deadline):
    fit.check_deadline(deadline, "original fit validation")
    old, old_e = baseline.validate_clean(paths["clean"], paths["clean_source"])
    bindings["clean8"] = (paths["clean"], paths["clean_source"], old_e)
    fit.check_deadline(deadline, "expanded fit validation")
    new, new_e = expanded.validate_coverage(paths["coverage"], paths["coverage_source"])
    bindings["clean32"] = (paths["coverage"], paths["coverage_source"], new_e)
    fit.check_deadline(deadline, "population loading")
    receipt = record.setdefault("population", {})
    try:
        population = clean.load_population(
            paths["parent"], paths["parent_source"], receipt=receipt
        )
    finally:
        if receipt.get("evidence") is not None:
            bindings["parent"] = (
                paths["parent"],
                paths["parent_source"],
                receipt["evidence"],
            )
    fit.check_deadline(deadline, "population loading completion")
    scale = float(np.float32(population.train_scale))
    for previous, data in ((old, old["data"]), (new, new["data"]["original"])):
        if (
            previous["model_config"] != clean.MODEL_CONFIG
            or previous["checkpoint_identity"]["train_scale_model_float32"] != scale
            or data["evidence"] != receipt["evidence"]
            or data["state_store_sha256"] != population.metadata["state_store_sha256"]
        ):
            raise ValueError("fit/reference configuration, normalizer or data mismatch")
    fit.check_deadline(deadline, "teacher mapping validation")
    _, record["teacher_mapping"] = expanded.matched_teacher(
        paths["coverage"], new_e, population, new
    )
    fit.check_deadline(deadline, "train-only RMS calibration")
    calibration = calibrate(
        paths["coverage"] / new_e["teacher_file"],
        new["population_index"],
        steps=512,
        nodes=256**2,
        train_scale=scale,
    )
    records = {}
    for name, evidence, key in (
        ("clean8", old_e, "old_rollout"),
        ("clean32", new_e, "new_rollout"),
    ):
        fit.check_deadline(deadline, "donor rollout validation")
        records[name], bound = read_rollout(
            paths[key], name, evidence, receipt["evidence"], scale
        )
        bindings[key] = (paths[key], REPO_ROOT, bound)
        fit.check_deadline(deadline, "donor rollout validation completion")
    cases = []
    for i, (seed, role) in enumerate(clean.SEED_ROLES):
        donors = {
            name: read_snapshots(
                paths[key],
                records[name]["cases"][i],
                population.states[i],
                deadline=deadline,
            )
            for name, key in (("clean8", "old_rollout"), ("clean32", "new_rollout"))
        }
        bank = make_bank(
            population.states[i], donors, calibration["physical_rms"], deadline=deadline
        )
        cases.append({"seed": seed, "role": role, "bank": bank, "donors": donors})
    record.update(
        train_scale=scale, calibration=calibration, model_config=clean.MODEL_CONFIG
    )
    return cases, {"clean8": old, "clean32": new}


def load_model(
    packet, previous, evidence, device, *, record=None, deadline=float("inf")
):
    if record is None:
        record = {"model_construction_attempts": 0, "models_constructed": 0}
    fit.check_deadline(deadline, "checkpoint byte capture")
    captured = (packet / evidence["checkpoint"]).read_bytes()
    if hashlib.sha256(captured).hexdigest() != evidence["checkpoint_sha256"]:
        raise ValueError("captured terminal checkpoint hash mismatch")
    fit.check_deadline(deadline, "checkpoint deserialization")
    checkpoint = torch.load(
        io.BytesIO(captured), map_location="cpu", weights_only=False
    )
    if (
        checkpoint["identity"] != previous["checkpoint_identity"]
        or checkpoint["update"] != 49152
        or checkpoint["schedule_position"] != checkpoint["update"]
    ):
        raise ValueError("terminal checkpoint identity/schedule mismatch")
    fit.check_deadline(deadline, "model construction")
    record["model_construction_attempts"] += 1
    model = clean.PeriodicVorticityPCNO(
        **previous["model_config"],
        train_scale=previous["checkpoint_identity"]["train_scale_float64"],
    )
    record["models_constructed"] += 1
    record["model_created"] = True
    fit.check_deadline(deadline, "model construction completion")
    model.load_state_dict(checkpoint["model"], strict=True)
    if (
        float(model.train_scale)
        != previous["checkpoint_identity"]["train_scale_model_float32"]
    ):
        raise ValueError("terminal model normalizer mismatch")
    fit.check_deadline(deadline, "model state loading completion")
    model.to(device).eval()
    fit.check_deadline(deadline, "model device transfer completion")
    return model


def terminal_replay(model, evaluation_path, deadline):
    """Deadline-aware equivalent of the immutable clean._terminal_replay helper."""
    device = next(model.parameters()).device
    model.eval()
    fit.check_deadline(deadline, "sentinel archive opening")
    with np.load(evaluation_path, allow_pickle=False) as saved, torch.no_grad():
        fit.check_deadline(deadline, "sentinel input decoding")
        inputs = saved["sentinel_input"]
        fit.check_deadline(deadline, "sentinel input transfer")
        tensor = torch.from_numpy(inputs).to(device)
        fit.check_deadline(deadline, "sentinel model inference")
        result = model(tensor)
        fit.check_deadline(deadline, "sentinel inference completion")
        errors = {}
        for name, key in (("raw", "raw_next"), ("next", "next_state")):
            fit.check_deadline(deadline, "sentinel output transfer")
            values = result[key].cpu().numpy().astype(np.float64)
            fit.check_deadline(deadline, "sentinel expected-output decoding")
            expected = saved["sentinel_" + name].astype(np.float64)
            fit.check_deadline(deadline, "sentinel comparison")
            numerator = np.sqrt(np.mean((values - expected) ** 2, axis=(-2, -1)))
            denominator = np.sqrt(np.mean(expected**2, axis=(-2, -1)))
            relative = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.inf),
                where=denominator > 0,
            )
            relative[(denominator == 0) & (numerator == 0)] = 0
            if not np.isfinite(relative).all() or np.any(relative > REPLAY_LIMIT):
                raise RuntimeError("terminal checkpoint sentinel replay failed")
            errors[name] = relative.tolist()
        fit.check_deadline(deadline, "sentinel replay completion")
    return {"relative_rms_limit": REPLAY_LIMIT, "errors": errors, "passed": True}


def save_arrays(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"file": path.name, "sha256": clean._hash(path)}


def run(paths, output, *, phase, device=None):
    if not __debug__:
        raise RuntimeError("optimized Python is forbidden")
    if phase not in ("validate", "assay") or device != (
        "cuda" if phase == "assay" else None
    ):
        raise ValueError("only assay uses --device cuda; validate creates no model")
    output = Path(output).resolve()
    paths = {k: Path(v).resolve() for k, v in paths.items()}
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in paths.values()
    ):
        raise ValueError("output must be disjoint from every input/source root")
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    deadline = start + WALL_SECONDS
    sources, bindings = {}, {}
    record = {
        "run_id": RUN_ID,
        "phase": phase,
        "status": "running",
        "stage": "input_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": sources,
        "banks": [],
        "cases": [],
        "completed_recipients": [],
        "checkpoint_replay": {},
        "model_created": False,
        "model_construction_attempts": 0,
        "models_constructed": 0,
        "model_forward_calls": 0,
        "execution_accounting": "construction attempts increment before constructor; models_constructed and model_created after constructor returns, before state loading; model_forward_calls counts top-level forward pre-hook invocations including failures and sentinel batches",
        "solver_calls": 0,
        "optimization_steps": 0,
        "protected_access": False,
        "network_access": False,
        "geometry_fitted": False,
        "corrections_added": False,
        "new_rollouts": False,
        "steps": list(STEPS),
        "wall_seconds": WALL_SECONDS,
        "float32_resolution_limit": FLOAT32_RESOLUTION_LIMIT,
        "interpretation": "Retrospective path recovery/learned response, not solver-relative fidelity, measured normal dynamics, OOD detection, causal attribution or prospective validation. Fourier bands are not tangent/normal. Forced dissipative energy/enstrophy are not conserved.",
    }
    try:
        for name in SOURCE_PATHS:
            fit.check_deadline(deadline, "source validation")
            sources[name] = clean._hash(REPO_ROOT / name)
        clean._write(output / "launch.json", record)
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError("less than 1 GiB free; no automatic cleanup")
        fit.check_deadline(deadline, "input validation")
        cases, models = load_inputs(paths, record, bindings, deadline)
        fit.check_deadline(deadline, "input validation completion")
        if [(c["seed"], c["role"]) for c in cases] != list(clean.SEED_ROLES):
            raise ValueError("common bank must contain all twelve original paths")
        for case in cases:
            fit.check_deadline(deadline, "bank serialization")
            bank = case["bank"]
            record["banks"].append(
                {
                    "seed": case["seed"],
                    "role": case["role"],
                    "queries": bank["queries"],
                    "skipped_matched_rms": bank["skipped_matched_rms"],
                    "array_hashes": {
                        k: array_hash(v) for k, v in bank["arrays"].items()
                    },
                    **save_arrays(output / f"bank_{case['seed']}.npz", bank["arrays"]),
                }
            )
            fit.check_deadline(deadline, "bank serialization completion")
        record["expected_metric_rows"] = 4 * sum(
            len(c["bank"]["queries"]) for c in cases
        )
        fit.check_deadline(deadline, "bank validation and serialization")
        if phase == "assay":
            record["stage"] = "model_assay"
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            def count_forward(_module, _inputs):
                record["model_forward_calls"] += 1

            for recipient in MODELS:
                packet, _, evidence = bindings[recipient]
                fit.check_deadline(deadline, "model loading")
                model = load_model(
                    packet,
                    models[recipient],
                    evidence,
                    device,
                    record=record,
                    deadline=deadline,
                )
                fit.check_deadline(deadline, "model loading completion")
                hook = model.register_forward_pre_hook(count_forward)
                try:
                    fit.check_deadline(deadline, "terminal sentinel replay")
                    record["checkpoint_replay"][recipient] = terminal_replay(
                        model, packet / evidence["teacher_file"], deadline
                    )
                    fit.check_deadline(deadline, "terminal sentinel replay completion")
                    for case in cases:
                        arrays, rows, replays, calls = evaluate_bank(
                            model,
                            case["bank"],
                            case["donors"][recipient],
                            recipient,
                            record["train_scale"],
                            record["model_config"]["modes"],
                            deadline,
                        )
                        fit.check_deadline(deadline, "case serialization")
                        saved = save_arrays(
                            output / f"outputs_{recipient}_{case['seed']}.npz", arrays
                        )
                        record["cases"].append(
                            {
                                "seed": case["seed"],
                                "role": case["role"],
                                "recipient": recipient,
                                "rows": rows,
                                "natural_replays": replays,
                                "forward_calls": calls,
                                **saved,
                            }
                        )
                        fit.check_deadline(deadline, "case serialization completion")
                        clean._write(output / "progress.json", record)
                        fit.check_deadline(deadline, "case progress serialization")
                finally:
                    hook.remove()
                record["completed_recipients"].append(recipient)
                del model
            if (
                len(record["cases"]) != 24
                or sum(len(c["rows"]) for c in record["cases"])
                != record["expected_metric_rows"]
            ):
                raise RuntimeError("common-input assay completion accounting failed")
        fit.check_deadline(deadline, "work completion")
        record["status"] = "completed"
    except Exception as error:
        # Decoder/model failures must leave a failed packet; no silent partial success.
        record.update(
            status="incomplete_budget" if isinstance(error, TimeoutError) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    work_seconds = perf_counter() - start
    after = {}
    for name in SOURCE_PATHS:
        try:
            after[name] = clean._hash(REPO_ROOT / name)
        except OSError:
            after[name] = None
    checks = {name: None for name in ("parent", *MODELS, "old_rollout", "new_rollout")}
    for name, (packet, source, evidence) in bindings.items():
        checks[name] = fit.evidence_stable(packet, source, evidence)
    stable = (
        False
        if False in checks.values()
        else True
        if all(v is True for v in checks.values())
        else None
    )
    source_stable = len(sources) == len(SOURCE_PATHS) and sources == after
    if stable is False or not source_stable:
        record["status"] = "invalid_provenance"
    if record["status"] == "completed" and stable is not True:
        record["status"] = "invalid_provenance"
    record.update(
        sources_after=after,
        source_stable=source_stable,
        input_evidence={k: e for k, (_, _, e) in bindings.items()},
        input_evidence_checks=checks,
        input_evidence_stable=stable,
        work_seconds=work_seconds,
        provenance_check_seconds=perf_counter() - start - work_seconds,
        seconds=perf_counter() - start,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
        },
    )
    clean._write(output / "result.json", record)
    clean._write(
        output / "artifact_manifest.json",
        {
            "run_id": RUN_ID,
            "phase": phase,
            "status": record["status"],
            "sources": sources,
            "source_stable": source_stable,
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = (
        "parent",
        "parent-source",
        "clean",
        "clean-source",
        "coverage",
        "coverage-source",
        "old-rollout",
        "new-rollout",
    )
    for name in (*names, "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "assay"), required=True)
    parser.add_argument("--device", choices=("cuda",))
    args = parser.parse_args()
    if (args.phase == "assay") != (args.device == "cuda"):
        parser.error("--device cuda is required only for assay")
    paths = {
        name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in names
    }
    result = run(paths, args.output, phase=args.phase, device=args.device)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
