"""Matched-start rollout of the fixed 32-path Clean terminal, not a new fit.

Run ``python -m scripts.time_dependent_no.evaluate_kolmogorov_coverage_rollout
--help``. Validation decodes only C's original twelve paths and creates no
model. Scientific rollout uses CUDA and preserves the closed evaluator's
recurrence, diagnostics, snapshots and failure guard without modifying it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import platform
import shutil
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_clean_rollout as baseline
from scripts.time_dependent_no import fit_kolmogorov_coverage as coverage

clean = coverage.clean
RUN_ID = "CM_NEXT_KF_CLEAN32_ROLLOUT_20260910B"
REPO_ROOT = Path(__file__).resolve().parents[2]
COVERAGE_MANIFEST_SHA256 = (
    "437f826a273afc331ce5a1441ffd57f230b28bb27d94bc198080dca64f45fa76"
)
CHECKPOINT_SHA256 = "6f8b106eb3758ac4cdb08ece4313969835bdc9c77afdaffab3f055810c9d0dbf"
TEACHER_SHA256 = "b33aa703aafecc2e6f85889e5873dc65896dcf9aaa7c6a0e2b663c26bdf1a5a0"
SOURCE_PATHS = (
    "scripts/time_dependent_no/evaluate_kolmogorov_coverage_rollout.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_coverage_rollout.py",
    *coverage.SOURCE_PATHS,
)
WALL_SECONDS = baseline.WALL_SECONDS
MINIMUM_FREE_BYTES = 1024**3


def validate_coverage(packet, source, *, unit_fixture=False):
    """Verify the fixed completed fit and all its bytes before loading tensors."""
    packet, source = Path(packet), Path(source)
    digest = clean._hash(packet / "artifact_manifest.json")
    if not unit_fixture and digest != COVERAGE_MANIFEST_SHA256:
        raise ValueError("coverage packet differs from the pinned completed fit")
    manifest = clean._read(packet / "artifact_manifest.json")
    expected = coverage.RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if (
        manifest["run_id"] != expected
        or manifest["status"] != "completed"
        or manifest["phase"] != "fit"
        or manifest["source_stable"] is not True
        or set(manifest["sources"]) != set(coverage.SOURCE_PATHS)
    ):
        raise ValueError("coverage identity/source closure mismatch")
    if {p.name for p in packet.iterdir()} != set(manifest["artifacts"]) | {
        "artifact_manifest.json"
    }:
        raise ValueError("coverage artifact inventory mismatch")
    for root, values in (
        (packet, manifest["artifacts"]),
        (source, manifest["sources"]),
        (REPO_ROOT, manifest["sources"]),
    ):
        if any(
            clean._hash(clean._bound(root, name)) != sha for name, sha in values.items()
        ):
            raise ValueError("coverage artifact or archived/live source hash mismatch")
    previous = clean._read(packet / "result.json")
    total = 12 if unit_fixture else coverage.TOTAL_UPDATES
    if (
        previous["run_id"] != expected
        or previous["status"] != "completed"
        or previous["phase"] != "fit"
        or previous["updates_completed"] != total
        or previous["sources_before"] != manifest["sources"]
        or previous["sources_after"] != manifest["sources"]
        or previous["input_evidence_stable"] is not True
        or previous["input_evidence_checks"]
        != {"baseline": True, "original": True, "extension": True}
        or any(
            previous[k] is not False
            for k in (
                "rollouts_evaluated",
                "development_used_for_updates",
                "protected_access",
                "network_access",
            )
        )
        or previous["solver_calls"] != 0
    ):
        raise ValueError("coverage completion or access contract mismatch")
    terminal, evaluation = previous["terminal_checkpoint"], previous["evaluations"][-1]
    selected = f"terminal_{total:06d}.pt"
    if (
        terminal["file"] != selected
        or terminal["update"] != total
        or terminal["sha256"] != manifest["artifacts"][selected]
        or terminal["replay"]["passed"] is not True
        or evaluation["update"] != total
        or evaluation["file"] != f"teacher_{total:06d}.npz"
        or evaluation["sha256"] != manifest["artifacts"][evaluation["file"]]
        or (
            not unit_fixture
            and (
                terminal["sha256"] != CHECKPOINT_SHA256
                or evaluation["sha256"] != TEACHER_SHA256
            )
        )
    ):
        raise ValueError("fixed terminal checkpoint/teacher identity mismatch")
    identity = previous["checkpoint_identity"]
    for key, value in {
        "run_id": expected,
        "sources": manifest["sources"],
        "input_manifests": manifest["input_manifests"],
        "model_config": previous["model_config"],
        "recipe": previous["recipe"],
        "population_index": previous["population_index"],
        "state_store_sha256": previous["data"]["state_store_sha256"],
        "train_scale_float64": previous["data"]["original"]["train_input_rms_float64"],
        "train_scale_model_float32": float(
            np.float32(previous["data"]["original"]["train_input_rms_float64"])
        ),
    }.items():
        if identity[key] != value:
            raise ValueError(f"coverage checkpoint identity mismatch: {key}")
    return previous, {
        "manifest_sha256": digest,
        "checkpoint": selected,
        "checkpoint_sha256": terminal["sha256"],
        "teacher_file": evaluation["file"],
        "teacher_sha256": evaluation["sha256"],
        "artifacts": {**manifest["artifacts"], "artifact_manifest.json": digest},
        "sources": manifest["sources"],
    }


def matched_teacher(packet, evidence, population, previous):
    """Match by declared seed/role, then check ordering and actual reference norms."""
    index = previous["population_index"]
    steps = population.states.shape[1] - 1
    training = len(index) - 4
    expected_index = coverage.extension.population_index(
        coverage.extension.NEW_SEEDS[: training - 8]
    )
    if training not in (10, 32) or index != expected_index:
        raise ValueError("coverage population index/role order mismatch")
    mapping = []
    for i, (seed, role) in enumerate(clean.SEED_ROLES):
        j = i if i < 8 else training + i - 8
        if (index[j]["seed"], index[j]["role"]) != (seed, role):
            raise ValueError("matched seed/role mismatch")
        mapping.append(
            {"original_index": i, "coverage_index": j, "seed": seed, "role": role}
        )
    selected = [entry["coverage_index"] for entry in mapping]
    with np.load(Path(packet) / evidence["teacher_file"], allow_pickle=False) as saved:
        if not np.array_equal(
            saved["trajectory_index"], np.repeat(np.arange(len(index)), steps)
        ) or not np.array_equal(
            saved["input_step"], np.tile(np.arange(steps), len(index))
        ):
            raise ValueError("teacher pair order/shape mismatch")
        rows = {}
        for key in ("learned_sse", "input_sse", "target_sse", "persistence_sse"):
            values = saved[key]
            if (
                values.shape != (len(index) * steps,)
                or values.dtype != np.float64
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(
                    "teacher squared norms must be finite nonnegative float64"
                )
            rows[key] = values.reshape(len(index), steps)[selected].copy()
    for i in range(12):
        for start in range(0, steps, 16):
            stop = min(start + 16, steps)
            x = population.states[i, start:stop].astype(np.float64)
            y = population.states[i, start + 1 : stop + 1].astype(np.float64)
            for key, field in (
                ("input_sse", x),
                ("target_sse", y),
                ("persistence_sse", x - y),
            ):
                norms = np.sum(field**2, axis=(-2, -1))
                if not np.allclose(
                    rows[key][i, start:stop], norms, rtol=1e-12, atol=1e-20
                ):
                    raise ValueError(f"teacher/reference norm mismatch: {key}")
    return rows["learned_sse"], mapping


def run(
    parent,
    parent_source,
    packet,
    source,
    output,
    device_name,
    *,
    phase="rollout",
    unit_fixture=False,
):
    """Fresh evidence packet; all numerical rollout operations reuse the baseline."""
    if not __debug__:
        raise RuntimeError("optimized Python is not allowed for this evaluator")
    if phase not in ("validate", "rollout"):
        raise ValueError("unknown phase")
    if phase == "rollout" and device_name != ("cpu" if unit_fixture else "cuda"):
        raise ValueError("real rollout uses CUDA; CPU is fixture-only")
    if phase == "validate" and device_name not in (None, "cpu"):
        raise ValueError("validate phase creates no model and uses no CUDA")
    parent, parent_source, packet, source, output = (
        Path(p).resolve() for p in (parent, parent_source, packet, source, output)
    )
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (parent, packet)
    ):
        raise ValueError("output must be disjoint from input packets")
    if any(
        output.is_relative_to(s) or s.is_relative_to(output)
        for s in (parent_source, source)
    ):
        raise ValueError("output must be disjoint from input source roots")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + WALL_SECONDS
    sources, bindings = {}, []
    receipt = {}
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "phase": phase,
        "status": "running",
        "stage": "source_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": sources,
        "data": receipt,
        "cases": [],
        "started_seeds": [],
        "model_created": False,
        "checkpoint_replay": None,
        "optimization_steps": 0,
        "solver_calls": 0,
        "protected_access": False,
        "network_access": False,
        "geometry_fitted": False,
        "corrections_added": False,
        "added_trajectory_arrays_loaded": False,
        "wall_seconds": WALL_SECONDS,
        "snapshot_steps": list(baseline.SNAPSHOT_STEPS),
        "amplitude_limit_rms_over_train_scale": baseline.MAXIMUM_RMS_OVER_TRAIN_SCALE,
        "interpretation": (
            "Exploratory matched-start comparison after expanded clean training. "
            "Only next_state is recurrent feedback; raw output is diagnostic. "
            "Shared mean-zero/dealias restriction is not measured data geometry. "
            "No causal drift, corrective necessity or prospective confirmation claim. "
            "Energy/enstrophy are diagnostics, not conserved in this forced dissipative system."
        ),
    }
    try:
        for name in SOURCE_PATHS:
            sources[name] = clean._hash(REPO_ROOT / name)
        clean._write(output / "launch.json", record)
        record["stage"] = "input_validation"
        previous, evidence = validate_coverage(
            packet, source, unit_fixture=unit_fixture
        )
        record["coverage_evidence"] = evidence
        bindings.append(("coverage", packet, source, evidence))
        try:
            population = clean.load_population(
                parent, parent_source, unit_fixture=unit_fixture, receipt=receipt
            )
        finally:
            if receipt.get("evidence") is not None:
                bindings.append(
                    ("original", parent, parent_source, receipt["evidence"])
                )
        original = previous["data"]["original"]
        if original["evidence"] != receipt["evidence"]:
            raise ValueError(
                "original population evidence differs from coverage training"
            )
        for key in (
            "state_store_sha256",
            "training_input_identity_sha256",
            "train_input_rms_float64",
        ):
            if original[key] != population.metadata[key]:
                raise ValueError(f"original data/normalization mismatch: {key}")
        if not unit_fixture and (
            population.states.shape != (12, 513, 256, 256)
            or previous["model_config"] != clean.MODEL_CONFIG
        ):
            raise ValueError("scientific population/model configuration mismatch")
        teacher, mapping = matched_teacher(packet, evidence, population, previous)
        record.update(
            teacher_mapping=mapping,
            horizon=population.states.shape[1] - 1,
            train_scale=previous["checkpoint_identity"]["train_scale_model_float32"],
        )
        coverage.check_deadline(deadline, "data and teacher validation")
        if phase == "rollout":
            record["stage"] = "checkpoint_replay"
            if shutil.disk_usage(output).free < (
                1024**2 if unit_fixture else MINIMUM_FREE_BYTES
            ):
                raise RuntimeError("insufficient free storage; no automatic deletion")
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = clean.PeriodicVorticityPCNO(
                **previous["model_config"], train_scale=population.train_scale
            )
            record["model_created"] = True
            # Hash and deserialize the same captured bytes, not a reopened path.
            checkpoint_bytes = (packet / evidence["checkpoint"]).read_bytes()
            if (
                hashlib.sha256(checkpoint_bytes).hexdigest()
                != evidence["checkpoint_sha256"]
            ):
                raise ValueError("captured checkpoint hash mismatch")
            checkpoint = torch.load(
                io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
            )
            del checkpoint_bytes
            if (
                checkpoint["identity"] != previous["checkpoint_identity"]
                or checkpoint["update"] != previous["updates_completed"]
                or checkpoint["schedule_position"] != checkpoint["update"]
            ):
                raise ValueError("checkpoint identity/schedule mismatch")
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            if float(model.train_scale) != record["train_scale"]:
                raise ValueError(
                    "checkpoint model normalizer differs from the fixed normalizer"
                )
            model.to(device_name).eval()
            record["checkpoint_replay"] = clean._terminal_replay(
                model, packet / evidence["teacher_file"]
            )
            coverage.check_deadline(deadline, "terminal replay")
            record["stage"] = "rollout"
            for i, (seed, role) in enumerate(clean.SEED_ROLES):
                with (output / f"progress_{seed}.jsonl").open(
                    "x", encoding="utf-8"
                ) as log:

                    def progress(row):
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                        log.flush()

                    record["started_seeds"].append(seed)
                    case, snapshots = baseline.rollout_one(
                        model,
                        population.states[i],
                        record["train_scale"],
                        previous["model_config"]["modes"],
                        deadline,
                        progress=progress,
                    )
                case.update(seed=seed, role=role, teacher_sse=teacher[i].tolist())
                record["cases"].append(case)
                case["summaries"] = baseline.summarize(
                    case,
                    teacher[i],
                    record["train_scale"],
                    population.states.shape[-1] ** 2,
                    (1, 2, 4) if unit_fixture else baseline.HORIZONS,
                )
                snapshot_path = output / f"snapshots_{seed}.npz"
                with snapshot_path.open("xb") as stream:
                    np.savez_compressed(stream, **snapshots)
                case.update(
                    snapshot_file=snapshot_path.name,
                    snapshot_sha256=clean._hash(snapshot_path),
                )
                clean._write(output / f"case_{seed}.json", case)
                clean._write(output / "progress.json", record)
                coverage.check_deadline(deadline, "case serialization")
        record["status"] = (
            "failed"
            if any(c["status"] == "model_error" for c in record["cases"])
            else "completed"
        )
        if any(c["status"] == "incomplete_budget" for c in record["cases"]):
            record["status"] = "incomplete_budget"
        coverage.check_deadline(deadline, "work completion")
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        TypeError,
        IndexError,
        FloatingPointError,
        MemoryError,
        zipfile.BadZipFile,
        pickle.UnpicklingError,
        EOFError,
        zlib.error,
    ) as error:
        record.update(
            status="incomplete_budget"
            if isinstance(error, TimeoutError)
            else "invalid_input"
            if record["stage"] in ("source_validation", "input_validation")
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    work_seconds = perf_counter() - started
    after = {}
    for name in SOURCE_PATHS:
        try:
            after[name] = clean._hash(REPO_ROOT / name)
        except OSError:
            after[name] = None
    checks = {"coverage": None, "original": None}
    for role, bound_packet, bound_source, bound_evidence in bindings:
        checks[role] = coverage.evidence_stable(
            bound_packet, bound_source, bound_evidence
        )
    stable = (
        False
        if False in checks.values()
        else True
        if all(v is True for v in checks.values())
        else None
    )
    source_stable = len(sources) == len(SOURCE_PATHS) and sources == after
    if not source_stable or stable is False:
        record["status"] = "invalid_provenance"
    record.update(
        sources_after=after,
        source_stable=source_stable,
        input_evidence_stable=stable,
        input_evidence_checks=checks,
        work_seconds=work_seconds,
        provenance_check_seconds=perf_counter() - started - work_seconds,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        all_rollouts_completed=len(record["cases"]) == 12
        and all(c["status"] == "completed" for c in record["cases"]),
        unstarted_seeds=[
            seed for seed, _ in clean.SEED_ROLES if seed not in record["started_seeds"]
        ],
        failure_counts={
            role: sum(
                c["status"] != "completed" for c in record["cases"] if c["role"] == role
            )
            for role in ("train", "development")
        },
        process_peak_rss_bytes=clean._peak_rss_bytes(),
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device_name,
            "cuda": torch.version.cuda,
        },
    )
    clean._write(output / "result.json", record)
    clean._write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "phase": phase,
            "status": record["status"],
            "sources": sources,
            "source_stable": source_stable,
            "input_manifests": [e["manifest_sha256"] for _, _, _, e in bindings],
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-source", "coverage", "coverage-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "rollout"), required=True)
    parser.add_argument("--device", choices=("cuda",))
    args = parser.parse_args()
    if (args.phase == "rollout") != (args.device == "cuda"):
        parser.error("--device cuda is required only for rollout")
    result = run(
        args.parent,
        args.parent_source,
        args.coverage,
        args.coverage_source,
        args.output,
        args.device,
        phase=args.phase,
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
