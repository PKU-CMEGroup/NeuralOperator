"""Frozen train-only Gaussian bank with paired recovery/dynamics targets.

Run with --population, --extension, --common, --qualification, --output and
--phase validate|generate. Validation constructs inputs but advances no solver.
The API-only unit fixture is never selectable from the scientific CLI.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no import screen_kolmogorov_common_solver as core

RUN_ID = "CM_NEXT_KF_PAIRED_BANK_20260911A"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "scripts/time_dependent_no/generate_kolmogorov_paired_bank.py",
    "tests/time_dependent_no/test_generate_kolmogorov_paired_bank.py",
    *core.SOURCE_PATHS,
)
ORIGINAL_SEEDS = tuple(range(2026090611, 2026090619))
ADDED_SEEDS = tuple(range(2026090801, 2026090825))
INPUT_STEPS = tuple(15 + 32 * j for j in range(16))
SENTINEL_SEEDS = (2026090611, 2026090612, 2026090801, 2026090802)
SENTINEL_STEPS = (15, 239, 495)
NOISE_SEED = 2026091101
WALL_SECONDS = 3 * 60 * 60
MINIMUM_FREE_BYTES = 2 * 1024**3
ROUNDING_LIMIT = 1e-4
CLEAN_LABEL_LIMIT = 1e-3
PINS = {
    "population": (
        "CM_NEXT_KF_POP_20260907C",
        "9ae1c66f2d7b8a2cdb105d7d57c71ee7811e64cc7f17203db7bcb51441c57f9c",
        "824bd16c259030abe054a352d0ac1aa7d336a89749d6d5ef228550e451b4291d",
    ),
    "extension": (
        "CM_NEXT_KF_TRAIN32_POP_20260908A",
        "a0f446389160974f29ad0cec07ce9ac752442b5b7d78f118a8a7e83ffd01a0cc",
        "b5aed8a70cc436b0c7d4dafdaca9555111a8dc307e51954c23e7f5428beac377",
    ),
    "common": (
        core.COMMON_RUN_ID,
        core.COMMON_MANIFEST_SHA256,
        core.COMMON_RESULT_SHA256,
    ),
    "qualification": (
        core.RUN_ID,
        "1620aafac5a8d27e8690dbe2c901a27b09942afabe7eaeca289a03b39e297051",
        "5b61f36b7b5a8e811b37474d2a7b5b1577ce98e5d082b8c8270d57de358d2fd0",
    ),
}


def selection(*, unit_fixture=False):
    if unit_fixture:
        return (ORIGINAL_SEEDS[0],), (ADDED_SEEDS[0],), (1, 3), (1,)
    return ORIGINAL_SEEDS, ADDED_SEEDS, INPUT_STEPS, SENTINEL_STEPS


def expected_calls(*, unit_fixture=False):
    old, added, steps, sentinel_steps = selection(unit_fixture=unit_fixture)
    seeds = old + added
    sentinels = sum(seed in SENTINEL_SEEDS for seed in seeds) * len(sentinel_steps)
    return 3 * len(seeds) * len(steps) + 10 * sentinels + 1


def prepare_inputs(reference, config, sigma, seed, step):
    """Gaussian in the retained real subspace; no per-draw RMS normalization."""
    n = config.resolution
    if reference.shape != (n, n) or reference.dtype != np.float64:
        raise ValueError("reference must be a native FP64 state")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid frozen Gaussian scale")
    solver = core.KolmogorovReferenceStepper(config)
    rank = (2 * (n // 3) + 1) ** 2 - 1
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([NOISE_SEED, seed, step]))
    )
    eta = solver.canonicalize(rng.standard_normal((n, n))) * (
        sigma * n / math.sqrt(rank)
    )
    clean = reference.astype(np.float32)
    raw = np.stack(
        [
            clean,
            (clean.astype(np.float64) + eta).astype(np.float32),
            (clean.astype(np.float64) - eta).astype(np.float32),
        ]
    )
    canonical = np.stack([solver.canonicalize(x.astype(np.float64)) for x in raw])
    lifted = np.stack([core.resize_dealiased_vorticity(x, 2 * n) for x in canonical])
    geometry = []
    for j, sign in ((1, 1), (2, -1)):
        actual = raw[j].astype(np.float64) - raw[0].astype(np.float64)
        size = core.rms(actual)
        projected = canonical[j] - canonical[0]
        geometry.append(
            {
                "sign": sign,
                "input_displacement_rms": size,
                "input_displacement_over_expected_rms": size / sigma,
                "intended_displacement_rms": core.rms(eta),
                "rounding_over_intended_displacement": core.ratio(
                    core.rms(actual - sign * eta), core.rms(eta)
                ),
                "direction_resolved": size > 0 and core.rms(projected) > 0,
                "projection_over_displacement": core.ratio(
                    max(core.rms(canonical[i] - raw[i]) for i in (0, j)), size
                ),
                "projected_direction_change": core.ratio(
                    core.rms(projected - actual), size
                ),
                "lift_roundtrip_relative_l2": max(
                    core.relative(
                        core.resize_dealiased_vorticity(lifted[i], n), canonical[i]
                    )
                    for i in (0, j)
                ),
            }
        )
    return (
        raw,
        canonical,
        lifted,
        geometry,
        {
            "seed_sequence": [NOISE_SEED, seed, step],
            "retained_real_dimension": rank,
            "expected_rms": sigma,
            "intended_sha256": core.array_hash(eta),
            "antithetic_residual_over_expected_rms": core.rms(
                raw[1].astype(np.float64)
                + raw[2].astype(np.float64)
                - 2 * raw[0].astype(np.float64)
            )
            / sigma,
        },
    )


def refinement_rows(outputs, geometry, replay):
    """Same-map response checks, with full fine answers retained in outputs."""
    n = outputs["A"].shape[-1]
    restricted = {
        level: values
        if level in ("A", "B")
        else np.stack([core.resize_dealiased_vorticity(x, n) for x in values])
        for level, values in outputs.items()
    }
    rows = []
    for j, geo in enumerate(geometry, 1):
        size = geo["input_displacement_rms"]
        response = {
            level: values[j] - values[0] for level, values in restricted.items()
        }
        errors = [
            core.rms(response[a] - response[b])
            for a, b in (("A", "B"), ("B", "C"), ("C", "D"))
        ]
        rows.append(
            {
                **geo,
                "native_replay_relative_l2": replay,
                "coarse_temporal_response_over_displacement": core.ratio(
                    errors[0], size
                ),
                "fine_temporal_response_over_displacement": core.ratio(errors[2], size),
                "spatial_response_over_displacement": core.ratio(errors[1], size),
                "discarded_fine_response_over_displacement": core.ratio(
                    core.rms(
                        outputs["D"][j]
                        - outputs["D"][0]
                        - core.resize_dealiased_vorticity(response["D"], 2 * n)
                    ),
                    size,
                ),
                "spatial_state_relative_l2": max(
                    core.relative(restricted["B"][i], restricted["C"][i])
                    for i in (0, j)
                ),
                "discarded_fine_state_relative_l2": max(
                    core.relative(
                        core.resize_dealiased_vorticity(restricted["D"][i], 2 * n),
                        outputs["D"][i],
                    )
                    for i in (0, j)
                ),
                "response_refinement_sensitivity_over_displacement": core.ratio(
                    sum(errors), size
                ),
                "trusted_A_response_gain": core.ratio(core.rms(response["A"]), size),
                "trusted_D_response_gain": core.ratio(core.rms(response["D"]), size),
            }
        )
    return rows


def load_metadata(roots, bindings, deadline, sources, *, unit_fixture):
    manifests, results = {}, {}
    suffix = "__UNIT_FIXTURE" if unit_fixture else ""
    for role, root in roots.items():
        run_id, manifest_pin, result_pin = PINS[role]
        if unit_fixture:
            manifest_pin = core._sha256(root / "artifact_manifest.json")
        manifest = json.loads(
            core.bound_bytes(
                root, "artifact_manifest.json", manifest_pin, bindings, role, deadline
            )
        )
        if unit_fixture:
            result_pin = manifest["artifacts"]["result.json"]
        if manifest["artifacts"]["result.json"] != result_pin:
            raise ValueError("parent result binding differs")
        result = json.loads(
            core.bound_bytes(root, "result.json", result_pin, bindings, role, deadline)
        )
        for value in (manifest, result):
            if value["run_id"] != run_id + suffix or value["source_stable"] is not True:
                raise ValueError("parent completion or identity differs")
        # The immutable C-population manifest predates the manifest status field.
        manifest_status = (
            manifest.get("status", "completed")
            if role == "population"
            else manifest["status"]
        )
        if result["status"] != "completed" or manifest_status != "completed":
            raise ValueError("parent completion or identity differs")
        manifests[role], results[role] = manifest, result
    population_sha = bindings[("population", "artifact_manifest.json")][1]
    common_sha = bindings[("common", "artifact_manifest.json")][1]
    if (
        results["extension"]["parent_manifest_sha256"] != population_sha
        or manifests["extension"]["parent_manifest_sha256"] != population_sha
    ):
        raise ValueError("extension parent differs")
    if results["extension"]["engineering_gates_pass"] is not True:
        raise ValueError("extension is not numerically qualified")
    common, qualified = results["common"], results["qualification"]
    if (
        common["input_evidence"]["parent"]["manifest_sha256"] != population_sha
        or common["input_evidence_stable"] is not True
        or common["calibration"]["validation_pairs"] != 0
        or common["calibration"]["training_pairs"] != 16384
        or qualified["qualification_passed"] is not True
        or qualified["inputs_stable"] is not True
        or qualified["solver_calls_completed"] != 53
        or qualified["input_files"]["population/artifact_manifest.json"]
        != population_sha
        or qualified["input_files"]["common/artifact_manifest.json"] != common_sha
    ):
        raise ValueError("calibration or completed pilot binding differs")
    if not unit_fixture:
        for name in core.SOURCE_PATHS:
            if sources[name] != manifests["qualification"]["sources"][name]:
                raise ValueError(f"closed dependency changed: {name}")
        if (
            Path(inspect.getfile(core.run)).resolve()
            != REPO_ROOT / core.SOURCE_PATHS[0]
        ):
            raise ValueError("imported pilot is outside source closure")
    old, added, _, _ = selection(unit_fixture=unit_fixture)
    expected_index = [
        {
            "seed": seed,
            "role": "train",
            "packet": "parent_C" if seed in old else "this_packet",
            "case_file": f"case_{seed}.json",
        }
        for seed in old + added
    ]
    if not unit_fixture:
        expected_index += [
            {
                "seed": seed,
                "role": "development",
                "packet": "parent_C",
                "case_file": f"case_{seed}.json",
            }
            for seed in range(2026090621, 2026090625)
        ]
    if results["extension"]["population_index"] != expected_index:
        raise ValueError("population index roles or order differ")
    n = 16 if unit_fixture else 256
    if common["model_config"]["resolution"] != n:
        raise ValueError("calibration resolution differs")
    sigma, scale = (
        float(common["calibration"]["physical_rms"]),
        float(common["train_scale"]),
    )
    if not all(math.isfinite(x) and x > 0 for x in (sigma, scale)):
        raise ValueError("invalid calibration")
    return (
        manifests,
        core.KolmogorovReferenceConfig(resolution=n, viscosity=0.01, macro_dt=0.05),
        sigma,
        scale,
    )


def selected_anchors(roots, manifests, bindings, deadline, config, *, unit_fixture):
    """Only named training files are decoded; cache at most one trajectory block."""
    old, added, steps, sentinel_steps = selection(unit_fixture=unit_fixture)
    for seed in old + added:
        role = "population" if seed in old else "extension"
        root, manifest = roots[role], manifests[role]
        name = f"case_{seed}.json"
        case = json.loads(
            core.bound_bytes(
                root, name, manifest["artifacts"][name], bindings, role, deadline
            )
        )
        if (case["seed"], case["role"], case["status"]) != (
            seed,
            "train",
            "completed",
        ) or case["config"] != asdict(config):
            raise ValueError("training case role or numerical recipe differs")
        cached_name, cached = None, None
        for step in steps:
            values, identities = [], []
            for t in (step, step + 1):
                block = core.one(
                    [
                        b
                        for b in case["blocks"]
                        if b["first_step"] <= t <= b["last_step"]
                    ],
                    "training block",
                )
                if block["sha256"] != manifest["artifacts"][block["file"]]:
                    raise ValueError("training block hash differs")
                if block["file"] != cached_name:
                    captured = core.bound_bytes(
                        root, block["file"], block["sha256"], bindings, role, deadline
                    )
                    cached = core.arrays_from_bytes(captured, deadline)
                    del captured
                    cached_name = block["file"]
                    count = block["last_step"] - block["first_step"] + 1
                    if (
                        set(cached) != {"steps", "states"}
                        or cached["steps"].dtype != np.int64
                        or not np.array_equal(
                            cached["steps"],
                            np.arange(block["first_step"], block["last_step"] + 1),
                        )
                        or cached["states"].dtype != np.float64
                        or cached["states"].shape
                        != (count, config.resolution, config.resolution)
                    ):
                        raise ValueError("training block order, dtype or shape differs")
                value = cached["states"][t - block["first_step"]].copy()
                if (
                    core.relative(
                        core.KolmogorovReferenceStepper(config).canonicalize(value),
                        value,
                    )
                    > config.canonical_tolerance
                ):
                    raise ValueError("native training state is not canonical")
                values.append(value)
                identities.append(
                    {
                        "packet": role,
                        "file": block["file"],
                        "sha256": block["sha256"],
                        "step": t,
                        "array_sha256": core.array_hash(value),
                    }
                )
            yield (
                seed,
                step,
                values,
                identities,
                seed in SENTINEL_SEEDS and step in sentinel_steps,
            )


def run(
    population, extension, common, qualification, output, *, phase, unit_fixture=False
):
    if not __debug__:
        raise RuntimeError("optimized Python is not supported")
    if phase not in ("validate", "generate"):
        raise ValueError("unknown phase")
    roots = {
        role: Path(path).resolve()
        for role, path in zip(PINS, (population, extension, common, qualification))
    }
    output = Path(output).resolve()
    if (
        output == REPO_ROOT
        or REPO_ROOT.is_relative_to(output)
        or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in roots.values()
        )
    ):
        raise ValueError("output overlaps source or input packet")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + WALL_SECONDS
    old, added, steps, sentinel_steps = selection(unit_fixture=unit_fixture)
    sources, bindings = {}, {}
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "phase": phase,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(old + added),
        "input_steps": list(steps),
        "signs": [1, -1],
        "expected_anchors": len(old + added) * len(steps),
        "expected_displaced_rows": 2 * len(old + added) * len(steps),
        "expected_sentinels": sum(seed in SENTINEL_SEEDS for seed in old + added)
        * len(sentinel_steps),
        "expected_solver_calls": expected_calls(unit_fixture=unit_fixture)
        if phase == "generate"
        else 0,
        "solver_call_attempts": 0,
        "solver_calls_completed": 0,
        "wall_seconds": WALL_SECONDS,
        "gate_limits": core.LIMITS,
        "rounding_limit": ROUNDING_LIMIT,
        "clean_label_limit": CLEAN_LABEL_LIMIT,
        "model_calls": 0,
        "checkpoint_loads": 0,
        "optimization_steps": 0,
        "validation_arrays_read": False,
        "protected_access": False,
        "new_rollouts": False,
        "metadata_validated": False,
        "inputs_validated": False,
        "qualification_passed": None,
        "anchors": [],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "target_map": "Every label is A=S_N256(P_N256(raw_float32_input)); B/C/D are diagnostics only.",
        "interpretation": "Fixed band-limited Gaussian augmentation bank. Fine-grid checks are sampled at twelve anchors, not a uniform label certificate or manifold-normal identification. No learned result or rollout claim.",
    }

    def advance(config, value, row, arrays, key):
        core.check_deadline(deadline, "stepper construction")
        solver = core.BudgetedReferenceStepper(config, deadline)
        call = {
            "label": key,
            "status": "prepared",
            "config": asdict(config),
            "input_sha256": core.array_hash(value),
        }
        row["calls"].append(call)
        core.check_deadline(deadline, "solver invocation")
        call["status"] = "running"
        record["solver_call_attempts"] += 1
        before = perf_counter()
        answer = solver.advance_canonical(value)
        record["solver_calls_completed"] += 1
        call.update(
            status="completed",
            seconds=perf_counter() - before,
            output_sha256=core.array_hash(answer.state),
            diagnostics=asdict(answer.diagnostics),
        )
        arrays[key] = answer.state
        core.save_arrays(output, row, arrays)
        core._json(output / "progress.json", record)
        core.check_deadline(deadline, "saved solver answer")
        return answer.state

    try:
        for name in SOURCE_PATHS:
            core.check_deadline(deadline, "source hashing")
            sources[name] = core._sha256(REPO_ROOT / name)
        if (
            phase == "generate"
            and not unit_fixture
            and shutil.disk_usage(output).free < MINIMUM_FREE_BYTES
        ):
            raise RuntimeError("insufficient free storage for the bounded bank")
        manifests, config, sigma, scale = load_metadata(
            roots, bindings, deadline, sources, unit_fixture=unit_fixture
        )
        record.update(
            metadata_validated=True,
            noise_expected_physical_rms=sigma,
            train_scale=scale,
            noise_expected_scaled_rms=sigma / scale,
            config=asdict(config),
        )
        configs = core.numerical_configs(config)
        for index, (seed, step, references, identities, sentinel) in enumerate(
            selected_anchors(
                roots, manifests, bindings, deadline, config, unit_fixture=unit_fixture
            )
        ):
            core.check_deadline(deadline, "input preparation")
            raw, canonical, lifted, geometry, noise = prepare_inputs(
                references[0], config, sigma, seed, step
            )
            row = {
                "seed": seed,
                "role": "train",
                "input_step": step,
                "output_step": step + 1,
                "status": "running",
                "sentinel": sentinel,
                "references": identities,
                "file": f"bank_{seed}_{step:03d}.npz",
                "calls": [],
                "geometry": geometry,
                "noise": noise,
                "numeric_rows": [],
                "repeat_exact": None,
            }
            record["anchors"].append(row)
            arrays = {"raw_inputs": raw}
            core.save_arrays(output, row, arrays)
            core._json(output / "progress.json", record)
            if not core.qualification_checks(geometry, include_solver=False) or any(
                g["rounding_over_intended_displacement"] is None
                or g["rounding_over_intended_displacement"] > ROUNDING_LIMIT
                for g in geometry
            ):
                raise ValueError(
                    "Gaussian draw failed input projection or FP32 rounding gate"
                )
            if phase == "generate":
                replay = None
                if sentinel:
                    native = advance(
                        config, references[0], row, arrays, "native_replay"
                    )
                    replay = core.relative(native, references[1])
                    row["native_replay_relative_l2"] = replay
                    if (
                        replay is None
                        or replay > core.LIMITS["native_replay_relative_l2"]
                    ):
                        raise ValueError("native FP64 replay failed")
                for j in range(3):
                    advance(config, canonical[j], row, arrays, f"A_{j}")
                row["clean_A_vs_archived_over_displacement"] = [
                    core.rms(arrays["A_0"] - references[1])
                    / g["input_displacement_rms"]
                    for g in geometry
                ]
                if (
                    max(row["clean_A_vs_archived_over_displacement"])
                    > CLEAN_LABEL_LIMIT
                ):
                    raise ValueError(
                        "fresh clean target differs from archived successor"
                    )
                row["A_structure"] = [
                    asdict(
                        core.KolmogorovReferenceStepper(config).diagnostics_canonical(
                            arrays[f"A_{j}"]
                        )
                    )
                    for j in range(3)
                ]
                if sentinel:
                    for level in ("B", "C", "D"):
                        for j in range(3):
                            advance(
                                configs[level],
                                canonical[j] if level == "B" else lifted[j],
                                row,
                                arrays,
                                f"{level}_{j}",
                            )
                    outputs = {
                        level: np.stack([arrays[f"{level}_{j}"] for j in range(3)])
                        for level in core.LEVELS
                    }
                    row["numeric_rows"] = refinement_rows(outputs, geometry, replay)
                    row["qualified"] = core.qualification_checks(
                        row["numeric_rows"], include_solver=True
                    )
                    row["fine_structure"] = {
                        level: [
                            asdict(
                                core.KolmogorovReferenceStepper(
                                    configs[level]
                                ).diagnostics_canonical(v)
                            )
                            for v in outputs[level]
                        ]
                        for level in ("B", "C", "D")
                    }
                if index == 0:
                    repeated = advance(
                        config, canonical[0].copy(), row, arrays, "repeat_A_clean"
                    )
                    row["repeat_exact"] = bool(np.array_equal(repeated, arrays["A_0"]))
                    if not row["repeat_exact"]:
                        raise ValueError("same-process solver repeat differs")
                row["training_columns"] = {
                    "clean_input": "raw_inputs[0]",
                    "recovery_target": "A_0",
                    "signed_rows": [
                        {
                            "sign": sign,
                            "input": f"raw_inputs[{j}]",
                            "dynamics_target": f"A_{j}",
                        }
                        for j, sign in ((1, 1), (2, -1))
                    ],
                }
            row["status"] = "completed"
            core._json(output / "progress.json", record)
            core.check_deadline(deadline, "completed anchor")
            if (index + 1) % 16 == 0 or unit_fixture:
                print(
                    json.dumps(
                        {
                            "completed_anchors": index + 1,
                            "solver_calls": record["solver_calls_completed"],
                        }
                    ),
                    flush=True,
                )
        record["inputs_validated"] = True
        if (
            len(record["anchors"]) != record["expected_anchors"]
            or sum(row["sentinel"] for row in record["anchors"])
            != record["expected_sentinels"]
            or record["solver_calls_completed"] != record["expected_solver_calls"]
            or record["solver_call_attempts"] != record["expected_solver_calls"]
        ):
            raise RuntimeError("bank completion accounting differs")
        if phase == "generate":
            record["qualification_passed"] = (
                all(row["qualified"] for row in record["anchors"] if row["sentinel"])
                and record["anchors"][0]["repeat_exact"] is True
            )
        core.check_deadline(deadline, "work completion")
        record["status"] = "completed"
    except Exception as error:
        record.update(
            status="incomplete_budget"
            if isinstance(error, core.BudgetExceeded)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
            qualification_passed=None,
        )
    record["work_seconds"] = perf_counter() - started
    final_started = perf_counter()
    after, checks = {}, {}
    for name in SOURCE_PATHS:
        try:
            after[name] = core._sha256(REPO_ROOT / name)
        except OSError:
            after[name] = None
    for (role, name), (path, expected) in bindings.items():
        try:
            checks[f"{role}/{name}"] = core._sha256(path) == expected
        except OSError:
            checks[f"{role}/{name}"] = False
    record.update(
        sources_before=sources,
        sources_after=after,
        source_stable=len(sources) == len(SOURCE_PATHS) and sources == after,
        input_files={
            f"{role}/{name}": sha for (role, name), (_, sha) in bindings.items()
        },
        input_checks=checks,
        inputs_stable=record["inputs_validated"]
        and bool(checks)
        and all(checks.values()),
        provenance_seconds=perf_counter() - final_started,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    if record["status"] == "completed" and not (
        record["source_stable"] and record["inputs_stable"]
    ):
        record.update(status="invalid_provenance", qualification_passed=None)
    core._json(output / "result.json", record)
    core._json(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "phase": phase,
            "status": record["status"],
            "sources": sources,
            "source_stable": record["source_stable"],
            "artifacts": {
                path.name: core._sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PINS, "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "generate"), required=True)
    args = parser.parse_args()
    result = run(
        args.population,
        args.extension,
        args.common,
        args.qualification,
        args.output,
        phase=args.phase,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "status",
                    "qualification_passed",
                    "solver_calls_completed",
                    "seconds",
                )
            }
        )
    )
    if result["status"] != "completed" or result["qualification_passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
