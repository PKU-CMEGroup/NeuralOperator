"""Train-only, 53-call solver qualification of a frozen common-input bank.

Run ``python -m scripts.time_dependent_no.screen_kolmogorov_common_solver
--common <C-packet> --population <population-C-packet> --output <fresh>
--phase validate|assay``. No model is constructed or checkpoint deserialized.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness import (
    BudgetedReferenceStepper,
    BudgetExceeded,
    _json,
    _sha256,
)
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
    resize_dealiased_vorticity,
)
from utility.time_dependent_no.path_conditioned_tube import (
    paired_solver_response_metrics,
)

RUN_ID = "CM_NEXT_KF_COMMON_SOLVER_20260911A"
COMMON_RUN_ID = "CM_NEXT_KF_COMMON_RESPONSE_20260910C"
COMMON_RESULT_SHA256 = (
    "7721c52862154292376e1c061b3bc9a5be7ff332f410de542989037f310605ff"
)
COMMON_MANIFEST_SHA256 = (
    "0a5d4cb2563f6d873a0c315bc6e0469ad71b8cd68796616cfedf74190a5e033f"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2026090611, 2026090612)
STEPS = (8, 32)
MODELS = ("clean8", "clean32")
LEVELS = ("A", "B", "C", "D")
WALL_SECONDS = 90 * 60
SOURCE_PATHS = (
    "scripts/time_dependent_no/screen_kolmogorov_common_solver.py",
    "tests/time_dependent_no/test_screen_kolmogorov_common_solver.py",
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
    "utility/time_dependent_no/path_conditioned_tube.py",
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
LIMITS = {
    "projection_over_displacement": 1e-4,
    "projected_direction_change": 1e-4,
    "lift_roundtrip_relative_l2": 1e-11,
    "native_replay_relative_l2": 1e-10,
    "coarse_temporal_response_over_displacement": 1e-3,
    "fine_temporal_response_over_displacement": 1e-3,
    "spatial_response_over_displacement": 1e-2,
    "discarded_fine_response_over_displacement": 1e-2,
    "spatial_state_relative_l2": 1e-3,
    "discarded_fine_state_relative_l2": 1e-3,
}


def check_deadline(deadline, stage):
    if perf_counter() >= deadline:
        raise BudgetExceeded(f"work budget expired before {stage}")


def array_hash(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def rms(values):
    array = np.asarray(values, dtype=np.float64)
    value = float(np.sqrt(np.mean(array * array)))
    if not math.isfinite(value):
        raise ArithmeticError("nonfinite RMS")
    return value


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def relative(values, reference):
    numerator, denominator = rms(values - reference), rms(reference)
    return (
        ratio(numerator, denominator)
        if denominator
        else (0.0 if not numerator else None)
    )


def one(items, label):
    if len(items) != 1:
        raise ValueError(f"expected exactly one {label}")
    return items[0]


def bound_bytes(root, name, digest, bindings, label, deadline):
    check_deadline(deadline, "input capture")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("input file must remain inside its packet")
    captured = path.read_bytes()
    if hashlib.sha256(captured).hexdigest() != digest:
        raise ValueError(f"input hash mismatch: {label}/{name}")
    bindings[(label, name)] = (path, digest)
    check_deadline(deadline, "input decoding")
    return captured


def arrays_from_bytes(captured, deadline):
    with np.load(io.BytesIO(captured), allow_pickle=False) as archive:
        arrays = {}
        for name in archive.files:
            check_deadline(deadline, "array decoding")
            arrays[name] = archive[name]
            if not np.isfinite(arrays[name]).all():
                raise ValueError("nonfinite input array")
    check_deadline(deadline, "array decoding completion")
    return arrays


def numerical_configs(config):
    refined = replace(config, dt_max=config.dt_max / 2, cfl=config.cfl / 2)
    return {
        "A": config,
        "B": refined,
        "C": replace(refined, resolution=2 * config.resolution),
        "D": replace(
            config,
            resolution=2 * config.resolution,
            dt_max=config.dt_max / 4,
            cfl=config.cfl / 4,
        ),
    }


def load_inputs(common, population, bindings, deadline, sources, *, unit_fixture=False):
    """Decode selected train files only, from immutable parent manifests."""
    manifest_hash = (
        _sha256(common / "artifact_manifest.json")
        if unit_fixture
        else COMMON_MANIFEST_SHA256
    )
    manifest = json.loads(
        bound_bytes(
            common,
            "artifact_manifest.json",
            manifest_hash,
            bindings,
            "common",
            deadline,
        )
    )
    result_hash = manifest["artifacts"]["result.json"]
    if not unit_fixture and result_hash != COMMON_RESULT_SHA256:
        raise ValueError("wrong common-input result binding")
    previous = json.loads(
        bound_bytes(common, "result.json", result_hash, bindings, "common", deadline)
    )
    expected_id = COMMON_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if (
        previous["run_id"] != expected_id
        or previous["status"] != "completed"
        or previous["steps"] != [1, 8, 32, 64]
        or previous["solver_calls"] != 0
        or previous["model_forward_calls"] != 386
        or not previous["source_stable"]
        or not previous["input_evidence_stable"]
        or previous["calibration"]["validation_pairs"] != 0
        or previous["calibration"]["training_pairs"] != 16384
    ):
        raise ValueError("common-input completion/recipe mismatch")
    if not unit_fixture:
        for name in SOURCE_PATHS[2:]:
            if sources[name] != manifest["sources"][name]:
                raise ValueError(f"closed source changed: {name}")
        origins = (
            (BudgetedReferenceStepper, SOURCE_PATHS[2]),
            (KolmogorovReferenceStepper, SOURCE_PATHS[3]),
            (paired_solver_response_metrics, SOURCE_PATHS[4]),
        )
        if any(
            Path(inspect.getfile(obj)).resolve() != (REPO_ROOT / name).resolve()
            for obj, name in origins
        ):
            raise ValueError("imported implementation is outside the source closure")
    parent = json.loads(
        bound_bytes(
            population,
            "artifact_manifest.json",
            previous["input_evidence"]["parent"]["manifest_sha256"],
            bindings,
            "population",
            deadline,
        )
    )
    if parent["run_id"] != "CM_NEXT_KF_POP_20260907C" or not parent["source_stable"]:
        raise ValueError("wrong population parent")
    scale = float(previous["train_scale"])
    displacement_scale = float(previous["calibration"]["physical_rms"])
    if not all(math.isfinite(x) and x > 0 for x in (scale, displacement_scale)):
        raise ValueError("invalid frozen scale")
    n = int(previous["model_config"]["resolution"])
    if not unit_fixture and n != 256:
        raise ValueError("scientific resolution must be 256")
    selected = []
    for seed in SEEDS:
        bank_record = one([b for b in previous["banks"] if b["seed"] == seed], "bank")
        if bank_record["role"] != "train":
            raise ValueError("selected bank is not training")
        name = bank_record["file"]
        if manifest["artifacts"][name] != bank_record["sha256"]:
            raise ValueError("bank manifest disagreement")
        bank = arrays_from_bytes(
            bound_bytes(
                common, name, bank_record["sha256"], bindings, "common", deadline
            ),
            deadline,
        )
        if set(bank) != {"reference_input", "reference_next", "probe_input"}:
            raise ValueError("wrong bank fields")
        for name, value in bank.items():
            count = len(bank_record["queries"]) if name == "probe_input" else 4
            if value.dtype != np.float32 or value.shape != (count, n, n):
                raise ValueError("wrong bank dtype/shape")
            if array_hash(value) != bank_record["array_hashes"][name]:
                raise ValueError("bank array identity mismatch")
        predictions = {}
        cases = {}
        for model in MODELS:
            case = one(
                [
                    c
                    for c in previous["cases"]
                    if c["seed"] == seed and c["recipient"] == model
                ],
                "model case",
            )
            if (
                case["role"] != "train"
                or manifest["artifacts"][case["file"]] != case["sha256"]
            ):
                raise ValueError("wrong prediction role/hash")
            predictions[model] = arrays_from_bytes(
                bound_bytes(
                    common, case["file"], case["sha256"], bindings, "common", deadline
                ),
                deadline,
            )
            cases[model] = case
            for prefix, count in (("clean", 4), ("probe", len(bank_record["queries"]))):
                for kind in ("raw", "next"):
                    values = predictions[model][prefix + "_" + kind]
                    if values.dtype != np.float32 or values.shape != (count, n, n):
                        raise ValueError("wrong prediction dtype/shape")
        name = f"case_{seed}.json"
        population_case = json.loads(
            bound_bytes(
                population,
                name,
                parent["artifacts"][name],
                bindings,
                "population",
                deadline,
            )
        )
        if (
            population_case["seed"],
            population_case["role"],
            population_case["status"],
        ) != (seed, "train", "completed"):
            raise ValueError("wrong population case")
        config = KolmogorovReferenceConfig(**population_case["config"]).validated()
        expected = KolmogorovReferenceConfig(
            resolution=n, viscosity=0.01, macro_dt=0.05
        )
        if config != expected:
            raise ValueError("population numerical/physical recipe mismatch")
        reference = {}
        block_cache = {}
        for step in (7, 8, 31, 32):
            block = one(
                [
                    b
                    for b in population_case["blocks"]
                    if b["first_step"] <= step <= b["last_step"]
                ],
                "population block",
            )
            name = block["file"]
            if parent["artifacts"][name] != block["sha256"]:
                raise ValueError("population block manifest disagreement")
            if name not in block_cache:
                block_cache[name] = arrays_from_bytes(
                    bound_bytes(
                        population,
                        name,
                        block["sha256"],
                        bindings,
                        "population",
                        deadline,
                    ),
                    deadline,
                )
            values = block_cache[name]
            indices = values["steps"]
            states = values["states"]
            if (
                indices.dtype != np.int64
                or indices.ndim != 1
                or states.dtype != np.float64
                or states.shape != (len(indices), n, n)
                or not np.array_equal(
                    indices, np.arange(block["first_step"], block["last_step"] + 1)
                )
            ):
                raise ValueError("invalid population block ordering/dtype/shape")
            reference[step] = states[
                one(np.flatnonzero(indices == step).tolist(), "step")
            ].copy()
        for step in STEPS:
            anchor = previous["steps"].index(step)
            queries = []
            for donor in MODELS:
                query = one(
                    [
                        q
                        for q in bank_record["queries"]
                        if q["output_step"] == step
                        and q["view"] == "matched_rms"
                        and q["donor"] == donor
                    ],
                    "matched donor query",
                )
                j = query["query_index"]
                if (
                    query["anchor"] != anchor
                    or query["input_step"] != step - 1
                    or not 0 <= j < len(bank_record["queries"])
                    or array_hash(bank["probe_input"][j]) != query["input_sha256"]
                ):
                    raise ValueError("query time/input identity mismatch")
                for model in MODELS:
                    for kind in ("raw", "next"):
                        row = one(
                            [
                                r
                                for r in cases[model]["rows"]
                                if r["query_index"] == j and r["output_kind"] == kind
                            ],
                            "metric row",
                        )
                        if any(row[key] != value for key, value in query.items()):
                            raise ValueError("recipient query pairing mismatch")
                queries.append(query)
            u = bank["reference_input"][anchor]
            target = bank["reference_next"][anchor]
            if not np.array_equal(
                reference[step - 1].astype(np.float32), u
            ) or not np.array_equal(reference[step].astype(np.float32), target):
                raise ValueError("FP64 population and float32 bank mismatch")
            raw = np.stack(
                [u, *(bank["probe_input"][q["query_index"]] for q in queries)]
            )
            for x in raw[1:]:
                amplitude = rms(x.astype(np.float64) - u.astype(np.float64))
                if amplitude == 0 or abs(amplitude / displacement_scale - 1) > 1e-4:
                    raise ValueError("matched direction collapsed or changed scale")
            selected.append(
                {
                    "seed": seed,
                    "output_step": step,
                    "queries": queries,
                    "config": config,
                    "raw_inputs": raw,
                    "reference_input64": reference[step - 1],
                    "reference_next64": reference[step],
                    "reference_next32": target,
                    "predictions": {
                        model: {
                            kind: np.stack(
                                [
                                    predictions[model]["clean_" + kind][anchor],
                                    *(
                                        predictions[model]["probe_" + kind][
                                            q["query_index"]
                                        ]
                                        for q in queries
                                    ),
                                ]
                            )
                            for kind in ("raw", "next")
                        }
                        for model in MODELS
                    },
                }
            )
    check_deadline(deadline, "input validation completion")
    return selected, scale, int(previous["model_config"]["modes"])


def prepare_anchor(data):
    raw = data["raw_inputs"].astype(np.float64)
    stepper = KolmogorovReferenceStepper(data["config"])
    canonical = np.stack([stepper.canonicalize(x) for x in raw])
    lifted = np.stack([resize_dealiased_vorticity(x, 2 * len(x)) for x in canonical])
    geometry = []
    for j in (1, 2):
        eta = raw[j] - raw[0]
        size = rms(eta)
        projected_eta = canonical[j] - canonical[0]
        geometry.append(
            {
                "donor": MODELS[j - 1],
                "input_displacement_rms": size,
                "projected_displacement_rms": rms(projected_eta),
                "direction_resolved": size > 0 and rms(projected_eta) > 0,
                "projection_over_displacement": ratio(
                    max(rms(canonical[0] - raw[0]), rms(canonical[j] - raw[j])), size
                ),
                "projected_direction_change": ratio(rms(projected_eta - eta), size),
                "lift_roundtrip_relative_l2": max(
                    relative(
                        resize_dealiased_vorticity(lifted[i], len(raw[i])), canonical[i]
                    )
                    for i in (0, j)
                ),
                "rounded_clean_vs_original_over_displacement": ratio(
                    rms(canonical[0] - data["reference_input64"]), size
                ),
            }
        )
    return canonical, lifted, geometry


def band_energies(field, scale, modes):
    n = len(field)
    k = np.fft.fftfreq(n) * n
    radius = np.maximum(abs(k[:, None]), abs(k[None, :]))
    power = np.abs(np.fft.fft2(field) / field.size) ** 2 / scale**2
    return [
        float(power[mask].sum())
        for mask in (
            radius <= modes,
            (radius > modes) & (radius <= n // 3),
            radius > n // 3,
        )
    ]


def measure_anchor(data, outputs, geometry, scale, modes):
    """All trusted responses use a clean/displaced pair from the SAME map."""
    n = data["config"].resolution
    coarse = {
        level: (
            values
            if level in ("A", "B")
            else np.stack([resize_dealiased_vorticity(x, n) for x in values])
        )
        for level, values in outputs.items()
    }
    numeric, metrics = [], []
    for j, geo in enumerate(geometry, start=1):
        size = geo["input_displacement_rms"]
        responses = {level: values[j] - values[0] for level, values in coarse.items()}
        temporal = rms(responses["A"] - responses["B"])
        spatial = rms(responses["B"] - responses["C"])
        fine_temporal = rms(responses["C"] - responses["D"])
        sensitivity = temporal + spatial + fine_temporal
        full_fine_response = outputs["D"][j] - outputs["D"][0]
        row = {
            **geo,
            "coarse_temporal_response_over_displacement": ratio(temporal, size),
            "fine_temporal_response_over_displacement": ratio(fine_temporal, size),
            "spatial_response_over_displacement": ratio(spatial, size),
            "discarded_fine_response_over_displacement": ratio(
                rms(
                    full_fine_response
                    - resize_dealiased_vorticity(responses["D"], 2 * n)
                ),
                size,
            ),
            "spatial_state_relative_l2": max(
                relative(coarse["B"][i], coarse["C"][i]) for i in (0, j)
            ),
            "discarded_fine_state_relative_l2": max(
                relative(
                    resize_dealiased_vorticity(coarse["D"][i], 2 * n), outputs["D"][i]
                )
                for i in (0, j)
            ),
            "response_refinement_sensitivity_rms": sensitivity,
            "response_refinement_sensitivity_over_displacement": ratio(
                sensitivity, size
            ),
            "archived_clean_successor_difference_scaled_rms": {
                level: rms(values[0] - data["reference_next32"].astype(np.float64))
                / scale
                for level, values in coarse.items()
            },
            "refinement_resolution": {},
        }
        for kind in ("raw", "next"):
            learned = {
                model: (
                    pred[kind][j].astype(np.float64) - pred[kind][0].astype(np.float64)
                )
                for model, pred in data["predictions"].items()
            }
            defects = {
                model: rms(value - responses["D"]) for model, value in learned.items()
            }
            gap = abs(defects["clean32"] - defects["clean8"])
            row["refinement_resolution"][kind] = {
                "indicator_not_error_bound": True,
                "response_defect_rms": defects,
                "response_defect_exceeds_indicator": {
                    model: value > sensitivity for model, value in defects.items()
                },
                "recipient_defect_gap_rms": gap,
                "recipient_gap_exceeds_twice_indicator": gap > 2 * sensitivity,
            }
        numeric.append(row)
        for level, trusted in coarse.items():
            for model, predictions in data["predictions"].items():
                for kind, values in predictions.items():
                    u, x = data["raw_inputs"][[0, j]].astype(np.float64)
                    a, b = values[[0, j]].astype(np.float64)
                    measured = paired_solver_response_metrics(
                        reference_input=u.reshape(-1, 1),
                        displaced_input=x.reshape(-1, 1),
                        reference_prediction=a.reshape(-1, 1),
                        displaced_prediction=b.reshape(-1, 1),
                        reference_next=trusted[0].reshape(-1, 1),
                        trusted_displaced_next=trusted[j].reshape(-1, 1),
                        node_weights=np.ones(n * n),
                        component_scale=[scale],
                    )
                    if any(
                        measured[key] > 1e-10
                        for key in (
                            "recovery_closure_scaled_rms",
                            "dynamics_closure_scaled_rms",
                            "response_defect_closure_scaled_rms",
                        )
                    ):
                        raise ArithmeticError("paired-response closure failed")
                    fields = {
                        "input_displacement": x - u,
                        "learned_response": b - a,
                        "trusted_response": trusted[j] - trusted[0],
                        "response_defect": (b - a) - (trusted[j] - trusted[0]),
                    }
                    metrics.append(
                        {
                            "donor": MODELS[j - 1],
                            "recipient": model,
                            "output_kind": kind,
                            "reference_level": level,
                            **measured,
                            "fourier_band_scaled_energy": {
                                name: band_energies(value, scale, modes)
                                for name, value in fields.items()
                            },
                        }
                    )
    return numeric, metrics


def qualification_checks(rows, *, include_solver):
    names = (
        tuple(LIMITS)
        if include_solver
        else (
            "projection_over_displacement",
            "projected_direction_change",
            "lift_roundtrip_relative_l2",
        )
    )
    checks = []
    for row in rows:
        checks.append(bool(row["direction_resolved"]))
        for name in names:
            value = row[name]
            checks.append(
                value is not None
                and math.isfinite(value)
                and 0 <= value <= LIMITS[name]
            )
    return all(checks) if checks else False


def save_arrays(output, row, arrays):
    path = output / row["file"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    row["sha256"] = _sha256(path)
    row["array_hashes"] = {name: array_hash(value) for name, value in arrays.items()}


def run(common, population, output, *, phase, unit_fixture=False):
    if not __debug__:
        raise RuntimeError("optimized Python is not supported")
    if phase not in ("validate", "assay"):
        raise ValueError("unknown phase")
    common, population, output = map(
        lambda x: Path(x).resolve(), (common, population, output)
    )
    if any(
        output.is_relative_to(root) or root.is_relative_to(output)
        for root in (common, population)
    ):
        raise ValueError("output overlaps an input packet")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + WALL_SECONDS
    bindings, sources, selected = {}, {}, []
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "phase": phase,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "output_steps": list(STEPS),
        "donors": list(MODELS),
        "wall_seconds": WALL_SECONDS,
        "gate_limits": LIMITS,
        "solver_call_attempts": 0,
        "solver_calls_completed": 0,
        "expected_solver_calls": 53 if phase == "assay" else 0,
        "expected_metric_rows": 128 if phase == "assay" else 0,
        "model_calls": 0,
        "checkpoint_loads": 0,
        "optimization_steps": 0,
        "validation_arrays_read": False,
        "protected_access": False,
        "new_rollouts": False,
        "geometry_fitted": False,
        "corrections_added": False,
        "inputs_validated": False,
        "qualification_passed": None,
        "anchors": [],
        "interpretation": "Retrospective projected finite-grid solver qualification on four training anchors only; no continuum bound, manifold label, correction or rollout claim.",
        "reference_maps": "A/B: S_N256(P_N256(q)); C/D: R_N256 S_N512(I_N512 P_N256(q)); raw model inputs/outputs unchanged.",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
    }

    def advance(config, state, row, label):
        check_deadline(deadline, "stepper construction")
        stepper = BudgetedReferenceStepper(config, deadline)
        check_deadline(deadline, "solver invocation")
        call = {
            "label": label,
            "status": "prepared",
            "config": asdict(config),
            "input_sha256": array_hash(state),
        }
        row["calls"].append(call)
        check_deadline(deadline, "counted solver invocation")
        call["status"] = "running"
        record["solver_call_attempts"] += 1
        call_started = perf_counter()
        result = stepper.advance_canonical(state)
        record["solver_calls_completed"] += 1
        call.update(
            status="completed",
            seconds=perf_counter() - call_started,
            output_sha256=array_hash(result.state),
            diagnostics=asdict(result.diagnostics),
        )
        return result.state

    try:
        for name in SOURCE_PATHS:
            check_deadline(deadline, "source hashing")
            sources[name] = _sha256(REPO_ROOT / name)
        selected, scale, modes = load_inputs(
            common, population, bindings, deadline, sources, unit_fixture=unit_fixture
        )
        record.update(inputs_validated=True, train_scale=scale, modes=modes)
        if len(selected) != 4:
            raise ValueError("wrong selected anchor count")
        for index, data in enumerate(selected):
            check_deadline(deadline, "anchor preparation")
            canonical, lifted, geometry = prepare_anchor(data)
            check_deadline(deadline, "anchor preparation completion")
            row = {
                "seed": data["seed"],
                "role": "train",
                "output_step": data["output_step"],
                "input_step": data["output_step"] - 1,
                "status": "running",
                "calls": [],
                "file": f"anchor_{data['seed']}_{data['output_step']:03d}.npz",
                "queries": data["queries"],
                "geometry": geometry,
                "numeric_rows": [],
                "metrics": [],
                "repeat_exact": None,
            }
            record["anchors"].append(row)
            arrays = {
                "raw_inputs": data["raw_inputs"],
                "canonical_inputs": canonical,
                "lifted_inputs": lifted,
                "reference_input64": data["reference_input64"],
                "reference_next64": data["reference_next64"],
                "reference_next32": data["reference_next32"],
            }
            save_arrays(output, row, arrays)
            _json(output / "progress.json", record)
            check_deadline(deadline, "input qualification")
            if not qualification_checks(geometry, include_solver=False):
                raise ValueError("input projection/roundtrip qualification failed")
            if phase == "assay":
                configs = numerical_configs(data["config"])
                arrays["native_replay"] = advance(
                    configs["A"], data["reference_input64"], row, "native_fp64_replay"
                )
                save_arrays(output, row, arrays)
                check_deadline(deadline, "native replay comparison")
                replay_error = relative(
                    arrays["native_replay"], data["reference_next64"]
                )
                row["native_replay_relative_l2"] = replay_error
                if (
                    replay_error is None
                    or replay_error > LIMITS["native_replay_relative_l2"]
                ):
                    raise ValueError(
                        "native FP64 replay failed before displaced solves"
                    )
                outputs = {}
                for level, config in configs.items():
                    for j in range(3):
                        check_deadline(deadline, "numerical level")
                        key = f"{level}_{j}"
                        state = canonical[j] if level in ("A", "B") else lifted[j]
                        arrays[key] = advance(config, state, row, key)
                        save_arrays(output, row, arrays)
                        _json(output / "progress.json", record)
                        check_deadline(
                            deadline, "solver answer serialization completion"
                        )
                    outputs[level] = np.stack(
                        [arrays[f"{level}_{j}"] for j in range(3)]
                    )
                if index == 0:
                    arrays["repeat_A_clean"] = advance(
                        configs["A"], canonical[0].copy(), row, "repeat_A_clean"
                    )
                    save_arrays(output, row, arrays)
                    check_deadline(deadline, "repeat comparison")
                    row["repeat_exact"] = bool(
                        np.array_equal(arrays["repeat_A_clean"], arrays["A_0"])
                    )
                    if not row["repeat_exact"]:
                        raise ValueError("same-process solver repeat failed")
                check_deadline(deadline, "response metrics")
                row["numeric_rows"], row["metrics"] = measure_anchor(
                    data, outputs, geometry, scale, modes
                )
                for numeric in row["numeric_rows"]:
                    numeric["native_replay_relative_l2"] = replay_error
                row["qualified"] = qualification_checks(
                    row["numeric_rows"], include_solver=True
                )
                row["reference_structure"] = {
                    level: [
                        asdict(
                            KolmogorovReferenceStepper(
                                configs[level]
                            ).diagnostics_canonical(x)
                        )
                        for x in values
                    ]
                    for level, values in outputs.items()
                }
            row["status"] = "completed"
            _json(output / "progress.json", record)
            check_deadline(deadline, "anchor completion")
            print(
                json.dumps(
                    {
                        "seed": row["seed"],
                        "output_step": row["output_step"],
                        "solver_calls": record["solver_calls_completed"],
                        "phase": phase,
                    }
                ),
                flush=True,
            )
        metric_count = sum(len(row["metrics"]) for row in record["anchors"])
        if (
            record["solver_call_attempts"] != record["expected_solver_calls"]
            or record["solver_calls_completed"] != record["expected_solver_calls"]
            or metric_count != record["expected_metric_rows"]
        ):
            raise RuntimeError("pilot completion accounting mismatch")
        check_deadline(deadline, "work completion")
        if phase == "assay":
            record["qualification_passed"] = (
                all(row["qualified"] for row in record["anchors"])
                and record["anchors"][0]["repeat_exact"] is True
            )
        record["status"] = "completed"
    except Exception as error:
        record.update(
            status="incomplete_budget"
            if isinstance(error, BudgetExceeded)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
            qualification_passed=None,
        )
    record["work_seconds"] = perf_counter() - started
    final_started = perf_counter()
    after = {}
    for name in SOURCE_PATHS:
        try:
            after[name] = _sha256(REPO_ROOT / name)
        except OSError:
            after[name] = None
    input_checks = {}
    for (label, name), (path, expected) in bindings.items():
        try:
            input_checks[label + "/" + name] = _sha256(path) == expected
        except OSError:
            input_checks[label + "/" + name] = False
    record.update(
        sources_before=sources,
        sources_after=after,
        source_stable=len(sources) == len(SOURCE_PATHS) and sources == after,
        input_files={
            label + "/" + name: digest
            for (label, name), (_, digest) in bindings.items()
        },
        input_checks=input_checks,
        inputs_stable=bool(input_checks)
        and all(input_checks.values())
        and record["inputs_validated"],
        provenance_seconds=perf_counter() - final_started,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    if record["status"] == "completed" and not (
        record["source_stable"] and record["inputs_stable"]
    ):
        record.update(status="invalid_provenance", qualification_passed=None)
    _json(output / "result.json", record)
    manifest = {
        "run_id": record["run_id"],
        "phase": phase,
        "status": record["status"],
        "sources": sources,
        "source_stable": record["source_stable"],
        "artifacts": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _json(output / "artifact_manifest.json", manifest)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "assay"), required=True)
    args = parser.parse_args()
    result = run(args.common, args.population, args.output, phase=args.phase)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "status",
                    "qualification_passed",
                    "solver_call_attempts",
                    "seconds",
                )
            }
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
