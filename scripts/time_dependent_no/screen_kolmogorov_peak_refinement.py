"""Fixed native-grid early-transient refinement; no fitting or population claim.

Run ``python -m scripts.time_dependent_no.screen_kolmogorov_peak_refinement
--parent <completed-population-packet> --output <fresh-directory>``.
Parent metadata select two numerical stress cases; parent state arrays are
never opened. Full native query/continuation arrays permit independent audits.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness import (
    INITIAL_MODES,
    BudgetedReferenceStepper,
    BudgetExceeded,
    KolmogorovReferenceConfig,
    _advance,
    _flush_block,
    _initial,
    _json,
    _relative,
    _sha256,
    _state_hash,
    resize_dealiased_vorticity,
)

RUN_ID = "CM_NEXT_KF_PEAK_20260906A"
PARENT_RUN_ID = "CM_NEXT_KF_POP_20260906A"
PARENT_MANIFEST_SHA256 = (
    "10103ba067a5e07cbe6bd1a2faa7f36c70392170da8400e1e4503a0d0c21ed5b"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SOURCES = {
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py": "0cff649a4d80bd9140d819e284697fdeace5f622e4bbd0161f0bac17975bfa02",
    "tests/time_dependent_no/test_screen_kolmogorov_trajectory_readiness.py": "575fb8805c7b13d4100af77b2f855c58451c948e07fa3fc5a1ebaa4895151f3e",
    "utility/time_dependent_no/kolmogorov_reference.py": "6c3c1938318deb52c8873243692bd06daed4f9958e010556fee3c8666fe3dbda",
}
SOURCE_PATHS = (
    "scripts/time_dependent_no/screen_kolmogorov_peak_refinement.py",
    "tests/time_dependent_no/test_screen_kolmogorov_peak_refinement.py",
    *FROZEN_SOURCES,
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
PARENT_ROLES = tuple((seed, "train") for seed in range(2026090611, 2026090619)) + tuple(
    (seed, "development") for seed in range(2026090621, 2026090625)
)
SELECTED_ROLES = ((2026090617, "train"), (2026090621, "development"))
CONTINUATION_SEED = 2026090617
MINIMUM_FREE_BYTES = 512 * 1024**2
GATE_LIMITS = {
    "clean_spatial_relative_l2": 1e-3,
    "fine_discarded_state_relative_l2": 1e-3,
    "continuation_endpoint_relative_l2": 2e-3,
}


@dataclass(frozen=True)
class PeakProtocol:
    resolution: int = 256
    steps: int = 64
    anchor_steps: tuple[int, ...] = (16, 24, 64)
    continuation_steps: int = 8
    block_steps: int = 16
    macro_dt: float = 0.05
    dt_max: float = 0.002
    wall_seconds: float = 90 * 60


PROTOCOL = PeakProtocol()


def _bound_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("parent evidence path must be a file inside the packet")
    return path


def validate_parent(parent: Path, *, unit_fixture: bool = False) -> dict:
    """Bind selection to the failed, completed parent without opening its arrays."""
    parent = Path(parent)
    manifest_path = parent / "artifact_manifest.json"
    digest = _sha256(manifest_path)
    if not unit_fixture and digest != PARENT_MANIFEST_SHA256:
        raise ValueError("parent manifest does not match the pinned population")
    manifest = json.loads(manifest_path.read_text())
    expected_id = PARENT_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if manifest["run_id"] != expected_id or manifest["source_stable"] is not True:
        raise ValueError("parent identity or source stability mismatch")
    evidence = {"artifact_manifest.json": digest}

    def read_bound(name):
        path = _bound_file(parent, name)
        if manifest["artifacts"].get(name) != _sha256(path):
            raise ValueError(f"parent evidence hash mismatch: {name}")
        evidence[name] = manifest["artifacts"][name]
        return json.loads(path.read_text())

    result, launch = read_bound("result.json"), read_bound("launch.json")
    if (
        result["run_id"] != expected_id
        or launch["run_id"] != expected_id
        or result["status"] != "completed"
        or result["engineering_gates_pass"] is not False
        or result["source_stable"] is not True
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or launch["sources"] != manifest["sources"]
        or result["protocol"] != launch["protocol"]
        or result["scope"]
        != (
            "unit_test_only"
            if unit_fixture
            else "fresh_training_development_population"
        )
    ):
        raise ValueError("parent completion, scope or provenance mismatch")
    expected_roles = [{"seed": seed, "role": role} for seed, role in PARENT_ROLES]
    if result["seed_roles"] != expected_roles or launch["seed_roles"] != expected_roles:
        raise ValueError("parent trajectory roles changed")
    if [(c["seed"], c["role"]) for c in result["cases"]] != list(PARENT_ROLES):
        raise ValueError("parent case population changed")
    candidates = []
    for case in result["cases"]:
        if (
            case != read_bound(f"case_{case['seed']}.json")
            or case["status"] != "completed"
        ):
            raise ValueError("parent case evidence is inconsistent or incomplete")
        peak = case["maximum_palinstrophy"]
        diagnostics = case["trajectory_diagnostics"]
        if (
            peak["complete_trajectory"] is not True
            or [row["step"] for row in diagnostics]
            != list(range(result["protocol"]["steps"] + 1))
            or peak["step"]
            != max(diagnostics, key=lambda row: row["structure"]["palinstrophy"])[
                "step"
            ]
        ):
            raise ValueError(
                "parent peak is not the earliest complete-trajectory maximum"
            )
        rows = [row for row in case["queries"] if row["anchor_step"] == peak["step"]]
        if len(rows) != 1 or rows[0]["query_sha256"] != peak["state_sha256"]:
            raise ValueError("parent peak query is not uniquely state-aligned")
        row = rows[0]
        if row["probe"] != "clean" or row["sign"] != 0:
            raise ValueError("parent selection must use a clean-state check")
        value = row["fine_discarded_state_relative_l2"]
        if not math.isfinite(value) or value < 0:
            raise ValueError("parent discarded-state measurement is invalid")
        candidates.append(
            {
                "seed": case["seed"],
                "parent_role": case["role"],
                "parent_peak_step": peak["step"],
                "parent_peak_state_sha256": peak["state_sha256"],
                "parent_peak_discarded_state_relative_l2": value,
            }
        )
    selected = [
        max(
            (row for row in candidates if row["parent_role"] == role),
            key=lambda row: (
                row["parent_peak_discarded_state_relative_l2"],
                -row["seed"],
            ),
        )
        for _, role in SELECTED_ROLES
    ]
    if [(row["seed"], row["parent_role"]) for row in selected] != list(SELECTED_ROLES):
        raise ValueError("numerical stress selection differs from the frozen two seeds")
    return {
        "manifest_sha256": digest,
        "evidence_hashes": evidence,
        "selected": selected,
    }


def _save_arrays(output: Path, row: dict, arrays: dict) -> None:
    """Atomically update this run's named partial query/continuation packet."""
    if any(
        value.dtype != np.float64 or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise FloatingPointError("query arrays must remain finite float64")
    path = output / row["file"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    row["sha256"] = _sha256(path)
    row["array_hashes"] = {name: _state_hash(value) for name, value in arrays.items()}


def _clean_query(output, record, base, half, fine, anchor, step, deadline):
    row = {
        "anchor_step": step,
        "status": "running",
        "file": f"query_{record['seed']}_{step:05d}.npz",
        "query_sha256": _state_hash(anchor),
        "calls": {},
    }
    record["queries"].append(row)
    arrays = {"input": anchor.copy()}
    _save_arrays(output, row, arrays)
    case_path = output / f"case_{record['seed']}.json"
    _json(case_path, record)
    for name, stepper in (
        ("base_next", base),
        ("half_next", half),
        ("fine_next", fine),
    ):
        state = (
            resize_dealiased_vorticity(anchor, fine.config.resolution)
            if name == "fine_next"
            else anchor
        )
        arrays[name], row["calls"][name] = _advance(stepper, state, deadline)
        _save_arrays(output, row, arrays)
        _json(case_path, record)
    restricted = resize_dealiased_vorticity(arrays["fine_next"], base.config.resolution)
    row.update(
        status="completed",
        temporal_relative_l2=_relative(arrays["base_next"], arrays["half_next"]),
        spatial_relative_l2=_relative(arrays["half_next"], restricted),
        fine_discarded_state_relative_l2=_relative(
            resize_dealiased_vorticity(restricted, fine.config.resolution),
            arrays["fine_next"],
        ),
    )
    _json(case_path, record)
    print(
        json.dumps(
            {
                "seed": record["seed"],
                "query_step": step,
                "fine_seconds": row["calls"]["fine_next"]["seconds"],
            }
        ),
        flush=True,
    )


def _continuation(output, record, half, fine, anchor, protocol, deadline):
    coarse_state = anchor.copy()
    fine_state = resize_dealiased_vorticity(anchor, fine.config.resolution)
    case_path = output / f"case_{record['seed']}.json"
    record["continuation_anchor_step"] = record["maximum_palinstrophy"]["step"]
    for step in range(1, protocol.continuation_steps + 1):
        row = {
            "step": step,
            "status": "running",
            "file": f"continuation_{record['seed']}_{step:02d}.npz",
            "calls": {},
        }
        record["continuation_rows"].append(row)
        arrays = {"coarse_input": coarse_state.copy(), "fine_input": fine_state.copy()}
        _save_arrays(output, row, arrays)
        _json(case_path, record)
        for name, stepper, state in (
            ("coarse_next", half, coarse_state),
            ("fine_next", fine, fine_state),
        ):
            arrays[name], row["calls"][name] = _advance(stepper, state, deadline)
            _save_arrays(output, row, arrays)
            _json(case_path, record)
        coarse_state, fine_state = arrays["coarse_next"], arrays["fine_next"]
        restricted = resize_dealiased_vorticity(fine_state, half.config.resolution)
        row.update(
            status="completed",
            restricted_state_relative_l2=_relative(coarse_state, restricted),
            fine_discarded_state_relative_l2=_relative(
                resize_dealiased_vorticity(restricted, fine.config.resolution),
                fine_state,
            ),
        )
        _json(case_path, record)


def _run_case(output, selection, protocol, deadline):
    seed = selection["seed"]
    config = KolmogorovReferenceConfig(
        resolution=protocol.resolution,
        viscosity=0.01,
        linear_drag=0.1,
        forcing_amplitude=1.0,
        forcing_wavenumber=4,
        macro_dt=protocol.macro_dt,
        dt_max=protocol.dt_max,
    )
    base = BudgetedReferenceStepper(config, deadline)
    half = BudgetedReferenceStepper(
        replace(config, dt_max=config.dt_max / 2, cfl=config.cfl / 2), deadline
    )
    fine = BudgetedReferenceStepper(
        replace(half.config, resolution=2 * config.resolution), deadline
    )
    record = {
        **selection,
        "status": "running",
        "started_generation": False,
        "config": asdict(config),
        "comparison_configs": {
            "half": asdict(half.config),
            "fine": asdict(fine.config),
        },
        "blocks": [],
        "last_retained_step": None,
        "trajectory_diagnostics": [],
        "maximum_palinstrophy": None,
        "expected_anchor_steps": None,
        "queries": [],
        "continuation_rows": [],
    }
    indices, states = [], []
    peak_state = None
    started = perf_counter()
    case_path = output / f"case_{seed}.json"
    try:
        if perf_counter() >= deadline:
            raise BudgetExceeded(
                "budget expired before this assigned native seed started"
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            current = _initial(base, seed)
            record["started_generation"] = True
            indices.append(0)
            states.append(current)
            _flush_block(output, seed, indices, states, record)
            with (output / f"progress_{seed}.jsonl").open(
                "x", encoding="utf-8"
            ) as progress:
                for step in range(protocol.steps + 1):
                    if step:
                        current, timing = _advance(base, current, deadline)
                        indices.append(step)
                        states.append(current)
                    else:
                        timing = None
                    diagnostic = {
                        "step": step,
                        "structure": asdict(base.diagnostics_canonical(current)),
                        "state_sha256": _state_hash(current),
                        "call": timing,
                    }
                    encoded = json.dumps(diagnostic, allow_nan=False)
                    record["trajectory_diagnostics"].append(diagnostic)
                    progress.write(encoded + "\n")
                    progress.flush()
                    value = diagnostic["structure"]["palinstrophy"]
                    if (
                        peak_state is None
                        or value > record["maximum_palinstrophy"]["value"]
                    ):
                        peak_state = current.copy()
                        record["maximum_palinstrophy"] = {
                            "step": step,
                            "value": value,
                            "state_sha256": diagnostic["state_sha256"],
                            "complete_trajectory": False,
                        }
                    if (
                        len(states) >= protocol.block_steps
                        or step in protocol.anchor_steps
                        or step == protocol.steps
                    ):
                        _flush_block(output, seed, indices, states, record)
                        _json(case_path, record)
                    if step in protocol.anchor_steps:
                        _clean_query(
                            output, record, base, half, fine, current, step, deadline
                        )
                peak = record["maximum_palinstrophy"]
                peak["complete_trajectory"] = True
                record["expected_anchor_steps"] = sorted(
                    set(protocol.anchor_steps) | {peak["step"]}
                )
                if peak["step"] not in protocol.anchor_steps:
                    _clean_query(
                        output,
                        record,
                        base,
                        half,
                        fine,
                        peak_state,
                        peak["step"],
                        deadline,
                    )
                if seed == CONTINUATION_SEED:
                    _continuation(
                        output, record, half, fine, peak_state, protocol, deadline
                    )
                record["status"] = "completed"
    except BudgetExceeded as error:
        record.update(status="incomplete_budget", error=str(error))
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        record.update(
            status="solver_failure", error_type=type(error).__name__, error=str(error)
        )
    finally:
        _flush_block(output, seed, indices, states, record)
        for row in record["queries"] + record["continuation_rows"]:
            if row["status"] == "running":
                row["status"] = record["status"]
        record["seconds"] = perf_counter() - started
        _json(case_path, record)
    return record


def numeric_gates(records, protocol):
    complete = [case["seed"] for case in records] == [
        seed for seed, _ in SELECTED_ROLES
    ] and all(
        case["status"] == "completed" and case["last_retained_step"] == protocol.steps
        for case in records
    )
    known_anchors = len(records) == len(SELECTED_ROLES) and all(
        case["expected_anchor_steps"] is not None for case in records
    )
    expected_clean = (
        sum(len(case["expected_anchor_steps"]) for case in records)
        if known_anchors
        else None
    )
    clean_complete = complete and all(
        sorted(
            row["anchor_step"]
            for row in case["queries"]
            if row["status"] == "completed"
        )
        == case["expected_anchor_steps"]
        and all(row["status"] == "completed" for row in case["queries"])
        for case in records
    )
    continuation_complete = complete and all(
        [
            row["step"]
            for row in case["continuation_rows"]
            if row["status"] == "completed"
        ]
        == (
            list(range(1, protocol.continuation_steps + 1))
            if case["seed"] == CONTINUATION_SEED
            else []
        )
        and all(row["status"] == "completed" for row in case["continuation_rows"])
        for case in records
    )
    queries = [
        row
        for case in records
        for row in case["queries"]
        if row["status"] == "completed"
    ]
    values = {
        "clean_spatial_relative_l2": [row["spatial_relative_l2"] for row in queries],
        "fine_discarded_state_relative_l2": [
            row["fine_discarded_state_relative_l2"] for row in queries
        ],
        "continuation_endpoint_relative_l2": [
            row["restricted_state_relative_l2"]
            for case in records
            for row in case["continuation_rows"]
            if row["status"] == "completed"
            and row["step"] == protocol.continuation_steps
        ],
    }
    gates = {}
    for name, samples in values.items():
        endpoint = name == "continuation_endpoint_relative_l2"
        expected = 1 if endpoint else expected_clean
        coverage = (continuation_complete if endpoint else clean_complete) and len(
            samples
        ) == expected
        gates[name] = {
            "maximum": max(samples) if samples else None,
            "limit": GATE_LIMITS[name],
            "sample_count": len(samples),
            "expected_sample_count": expected,
            "complete": coverage,
            "pass": all(
                math.isfinite(value) and 0 <= value <= GATE_LIMITS[name]
                for value in samples
            )
            if coverage
            else None,
        }
    return gates


def run_screen(parent: Path, output: Path, *, unit_fixture: bool = False):
    output, parent = Path(output), Path(parent)
    if output.exists():
        raise FileExistsError(output)
    evidence = validate_parent(parent, unit_fixture=unit_fixture)
    protocol = (
        replace(
            PROTOCOL,
            resolution=16,
            steps=4,
            anchor_steps=(1, 2, 4),
            continuation_steps=2,
            block_steps=2,
            macro_dt=0.004,
            wall_seconds=60,
        )
        if unit_fixture
        else PROTOCOL
    )
    sources_before = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    if any(sources_before[name] != digest for name, digest in FROZEN_SOURCES.items()):
        raise ValueError("closed trajectory/reference source mismatch")
    if (
        Path(_initial.__code__.co_filename).resolve()
        != (REPO_ROOT / next(iter(FROZEN_SOURCES))).resolve()
    ):
        raise ValueError(
            "imported initial-state helper is outside the bound source root"
        )
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError("at least 512 MiB free output space is required")
    output.mkdir()
    started = perf_counter()
    deadline = started + protocol.wall_seconds
    run_id = RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    launch = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "scope": "unit_test_only"
        if unit_fixture
        else "numerically_selected_native_early_refinement",
        "parent_evidence": evidence,
        "selection_rule": "largest parent peak discarded-state fraction within each role; lowest seed breaks exact ties",
        "initial_recipe": {
            "modes": INITIAL_MODES,
            "phase_law": "NumPy default_rng(seed), uniform[0,2pi) per cosine",
            "perturbation_rms_over_laminar_rms": 0.25,
            "zero_mean_velocity": True,
            "native_grid_generation": True,
        },
        "gate_limits": GATE_LIMITS,
        "sources": sources_before,
        "free_disk_bytes_at_launch": free_bytes,
    }
    _json(output / "launch.json", launch)
    records = [
        _run_case(output, selection, protocol, deadline)
        for selection in evidence["selected"]
    ]
    status = (
        "incomplete_budget"
        if any(case["status"] == "incomplete_budget" for case in records)
        else "solver_failure"
        if any(case["status"] != "completed" for case in records)
        else "completed"
    )
    sources_after = {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}
    parent_stable = all(
        _sha256(_bound_file(parent, name)) == digest
        for name, digest in evidence["evidence_hashes"].items()
    )
    if not parent_stable:
        status = "invalid_parent"
    if sources_after != sources_before:
        status = "invalid_source"
    gates = numeric_gates(records, protocol)
    result = {
        **launch,
        "status": status,
        "cases": records,
        "numeric_gates": gates,
        "engineering_gates_pass": all(gate["pass"] for gate in gates.values())
        if status == "completed"
        else None,
        "parent_evidence_stable": parent_stable,
        "sources_before": sources_before,
        "sources_after": sources_after,
        "source_stable": sources_before == sources_after,
        "parent_array_access": False,
        "generated_solver_states": True,
        "model_access": False,
        "network_access": False,
        "displaced_input_qualification": False,
        "population_qualification": False,
        "stationarity_claim": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "seconds": perf_counter() - started,
        "interpretation": "Numerically selected native early-window stress screen only. No lifted parent trajectories, displacement qualification or replacement-population qualification. Fine states are retained without intermediate restriction/reset. Temporal differences are reported without a new threshold. Deadline checks precede solver calls/substeps; final flush/hash work follows compute shutdown.",
    }
    _json(output / "result.json", result)
    _json(
        output / "artifact_manifest.json",
        {
            "run_id": run_id,
            "sources": sources_before,
            "source_stable": result["source_stable"],
            "parent_manifest_sha256": evidence["manifest_sha256"],
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_screen(args.parent, args.output)
    if result["status"] != "completed" or result["engineering_gates_pass"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
