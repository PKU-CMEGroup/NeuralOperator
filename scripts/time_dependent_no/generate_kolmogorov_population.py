"""Recover the fixed interrupted population into a fresh, parent-bound packet.

Run ``python -m scripts.time_dependent_no.generate_kolmogorov_population
--parent <B-packet> --parent-source <B-launch/source> --output <fresh-C>``.
Five complete cases are inherited bytewise; one native prefix is resumed only
after a fixed cross-runtime replay gate. No parent file is modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no import screen_kolmogorov_peak_refinement as peak
from scripts.time_dependent_no import screen_kolmogorov_trajectory_readiness as long
from scripts.time_dependent_no.screen_kolmogorov_peak_refinement import (
    BudgetedReferenceStepper,
    _advance,
    _clean_query,
    _initial,
    _save_arrays,
)

RUN_ID = "CM_NEXT_KF_POP_20260907C"
PARENT_RUN_ID = "CM_NEXT_KF_POP_20260906B"
REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_MANIFEST_SHA256 = (
    "27ad081d6b2875c9d613b2978ce5d0aee0f698e3ba380775d26c0dccce281815"
)
PARENT_RESULT_SHA256 = (
    "067a5706cc1f954766c345b1cac51c43c17b306e4c86665a178da7a2f8864b1c"
)
PARENT_AUDIT_SHA256 = "e31afb7cf11024f53f566bdab6bdda7cb52a51f00e89a05652c91454c3fa7d14"
PARENT_PEAK_SHA256 = "da4991f1e1d7c5f5b93f551aaf77d38f73c41dc08a61a878b53893ca358edfe6"
REPLAY_LIMIT = 1e-10
RESUME_SEED = 2026090616
FROZEN_SOURCES = {
    "scripts/time_dependent_no/screen_kolmogorov_peak_refinement.py": "cde2e942da8a05398c57ca4f944e6db2ecc8b78c9d9916aa7caf478130ca0fe4",
    "tests/time_dependent_no/test_screen_kolmogorov_peak_refinement.py": "bf6f1c6797cd66ff84471c3362c3379e10f929eb8acf845367ee92c4a8a6740a",
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py": "0cff649a4d80bd9140d819e284697fdeace5f622e4bbd0161f0bac17975bfa02",
    "tests/time_dependent_no/test_screen_kolmogorov_trajectory_readiness.py": "575fb8805c7b13d4100af77b2f855c58451c948e07fa3fc5a1ebaa4895151f3e",
    "utility/time_dependent_no/kolmogorov_reference.py": "6c3c1938318deb52c8873243692bd06daed4f9958e010556fee3c8666fe3dbda",
}
SOURCE_PATHS = (
    "scripts/time_dependent_no/generate_kolmogorov_population.py",
    "tests/time_dependent_no/test_generate_kolmogorov_population.py",
    *FROZEN_SOURCES,
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
SEED_ROLES = tuple((seed, "train") for seed in range(2026090611, 2026090619)) + tuple(
    (seed, "development") for seed in range(2026090621, 2026090625)
)
LATE_SEEDS = (2026090617, 2026090621)
PROTOCOL = replace(
    long.PROTOCOL,
    resolution=256,
    seeds=tuple(seed for seed, _ in SEED_ROLES),
    anchor_steps=(0, 16, 24, 64, 256, 512),
    wall_seconds=8 * 3600,
    fixture_label="fresh_training_development_population",
)
MINIMUM_FREE_BYTES = 6 * 1024**3
GATE_LIMITS = {
    "clean_spatial_relative_l2": 1e-3,
    "fine_discarded_state_relative_l2": 1e-3,
    "continuation_endpoint_relative_l2": 2e-3,
}


def validate_parent(parent: Path, parent_source: Path, *, unit_fixture: bool = False):
    """Verify B and its archived source/audit, without requiring PEAK arrays."""
    manifest_path = parent / "artifact_manifest.json"
    digest = long._sha256(manifest_path)
    if not unit_fixture and digest != PARENT_MANIFEST_SHA256:
        raise ValueError("parent manifest differs from the pinned interrupted B")
    manifest = json.loads(manifest_path.read_text())
    expected_id = PARENT_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if manifest["run_id"] != expected_id or manifest["source_stable"] is not True:
        raise ValueError("parent identity or source stability mismatch")
    if set(manifest["sources"]) != set(SOURCE_PATHS) or any(
        long._sha256(peak._bound_file(parent_source, name)) != digest
        for name, digest in manifest["sources"].items()
    ):
        raise ValueError("parent source bindings do not match the frozen helpers")
    evidence = {"artifact_manifest.json": digest}
    if {path.name for path in parent.iterdir()} != set(manifest["artifacts"]) | {
        "artifact_manifest.json"
    }:
        raise ValueError("parent payload inventory differs from its manifest")
    for name, expected in manifest["artifacts"].items():
        if (
            Path(name).name != name
            or long._sha256(peak._bound_file(parent, name)) != expected
        ):
            raise ValueError(f"parent payload hash mismatch: {name}")
        evidence[name] = expected
    result = json.loads((parent / "result.json").read_text())
    launch = json.loads((parent / "launch.json").read_text())
    audit_path = parent_source.parent / "independent_population_audit.json"
    audit_digest = long._sha256(audit_path)
    audit = json.loads(audit_path.read_text())
    protocol = long.TrajectoryProtocol(**result["protocol"])
    expected_states = [protocol.steps] * 5 + [4 if unit_fixture else 16] + [None] * 6
    expected_status = ["completed"] * 5 + ["incomplete_budget"] * 7
    if (
        result["run_id"] != expected_id
        or launch["run_id"] != expected_id
        or result["status"] != "incomplete_budget"
        or result["engineering_gates_pass"] is not None
        or result["source_stable"] is not True
        or result["parent_evidence_stable"] is not True
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or launch["sources"] != manifest["sources"]
        or result["protocol"] != launch["protocol"]
        or (
            not unit_fixture
            and result["protocol"] != json.loads(json.dumps(asdict(PROTOCOL)))
        )
        or result["numeric_gates"] != _numeric_gates(result["cases"], protocol)
        or [(c["seed"], c["role"]) for c in result["cases"]] != list(SEED_ROLES)
        or [c["status"] for c in result["cases"]] != expected_status
        or [c["last_retained_step"] for c in result["cases"]] != expected_states
        or result["seed_roles"]
        != [{"seed": seed, "role": role} for seed, role in SEED_ROLES]
        or launch["seed_roles"] != result["seed_roles"]
        or audit["audit_status"] != "PASS"
        or audit["population_verdict"] != "INCOMPLETE_BUDGET_NOT_QUALIFIED"
        or audit["packet_manifest_sha256"] != digest
        or audit["result_sha256"] != evidence["result.json"]
        or (
            not unit_fixture
            and (
                evidence["result.json"] != PARENT_RESULT_SHA256
                or audit_digest != PARENT_AUDIT_SHA256
                or result["parent_manifest_sha256"] != PARENT_PEAK_SHA256
            )
        )
        or any(
            case != json.loads((parent / f"case_{case['seed']}.json").read_text())
            for case in result["cases"]
        )
    ):
        raise ValueError("parent interruption, roles, gates or provenance mismatch")
    return {
        "manifest_sha256": digest,
        "evidence_hashes": evidence,
        "source_hashes": manifest["sources"],
        "audit_sha256": audit_digest,
    }, result


def _retained_states(root, record):
    states = {}
    for block in record["blocks"]:
        with np.load(root / block["file"], allow_pickle=False) as data:
            for index, state in zip(data["steps"], data["states"], strict=True):
                if (
                    int(index) in states
                    or state.dtype != np.float64
                    or not np.isfinite(state).all()
                ):
                    raise ValueError("invalid retained prefix state")
                states[int(index)] = state.copy()
    if sorted(states) != list(range(record["last_retained_step"] + 1)):
        raise ValueError("retained prefix has missing or repeated indices")
    for row in record["trajectory_diagnostics"]:
        if long._state_hash(states[row["step"]]) != row["state_sha256"]:
            raise ValueError("retained prefix state/diagnostic hash mismatch")
    return states


def _runtime_replay(parent, output, parent_result, protocol, deadline, *, unit_fixture):
    """Fixed base/half/fine and restart checks, with durable per-call evidence."""
    cases = {case["seed"]: case for case in parent_result["cases"]}
    resume = 4 if unit_fixture else 16
    restart = 2 if unit_fixture else 8
    fine_anchor = cases[2026090611]["maximum_palinstrophy"]["step"]
    if not unit_fixture and fine_anchor != 14:
        raise ValueError("pinned fine replay anchor changed")
    record = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "relative_l2_limit": REPLAY_LIMIT,
        "calls": [],
        "checks": [],
        "expected_solver_calls": 10 + 2 * (resume - restart),
        "parent_state_arrays_decoded": False,
        "generated_solver_states": False,
    }
    path = output / "runtime_replay.json"
    long._json(path, record)
    started = perf_counter()

    def checked(label, config, inputs, references):
        previous = None
        for repetition in range(2):
            stepper = BudgetedReferenceStepper(config, deadline)
            current = inputs.copy()
            for index, reference in enumerate(references):
                row = {
                    "label": label,
                    "repetition": repetition,
                    "step": index + 1,
                    "status": "running",
                    "file": f"replay_{label}_{repetition}_{index + 1:02d}.npz",
                    "calls": {},
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                record["calls"].append(row)
                arrays = {"input": current.copy(), "reference": reference}
                _save_arrays(output, row, arrays)
                long._json(path, record)
                current, row["calls"]["output"] = _advance(stepper, current, deadline)
                arrays["output"] = current
                _save_arrays(output, row, arrays)
                error = long._relative(current, reference)
                row.update(
                    status="completed",
                    relative_l2=error,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                )
                if repetition == 0:
                    if index == 0:
                        previous = []
                    previous.append(current.copy())
                else:
                    row["same_process_bitwise"] = bool(
                        long._state_hash(current) == long._state_hash(previous[index])
                    )
                long._json(path, record)
                if (
                    not np.isfinite(error)
                    or error > REPLAY_LIMIT
                    or row.get("same_process_bitwise") is False
                ):
                    raise ValueError(
                        f"runtime replay disagreement: {label}/{index + 1}"
                    )

    try:
        if perf_counter() >= deadline:
            raise long.BudgetExceeded("budget expired before runtime replay")
        first = cases[2026090611]
        config = long.KolmogorovReferenceConfig(**first["config"])
        query0 = next(row for row in first["queries"] if row["anchor_step"] == 0)
        record["parent_state_arrays_decoded"] = True
        with np.load(parent / query0["file"], allow_pickle=False) as data:
            expected = data["input"].copy()
        initial_a = _initial(BudgetedReferenceStepper(config, deadline), 2026090611)
        record["generated_solver_states"] = True
        initial_b = _initial(BudgetedReferenceStepper(config, deadline), 2026090611)
        initial_error = long._relative(initial_a, expected)
        initial_check = {
            "label": "initial611",
            "file": "replay_initial611.npz",
            "relative_l2": initial_error,
            "same_process_bitwise": bool(
                long._state_hash(initial_a) == long._state_hash(initial_b)
            ),
        }
        _save_arrays(
            output,
            initial_check,
            {"reference": expected, "first": initial_a, "repeat": initial_b},
        )
        record["checks"].append(initial_check)
        long._json(path, record)
        if initial_error > REPLAY_LIMIT or not initial_check["same_process_bitwise"]:
            raise ValueError("native initial-law replay disagreement")
        for seed, anchor in ((2026090611, 0), (RESUME_SEED, resume)):
            case = cases[seed]
            row = next(q for q in case["queries"] if q["anchor_step"] == anchor)
            with np.load(parent / row["file"], allow_pickle=False) as data:
                inputs = data["input"].copy()
                for name, conf in (
                    ("base", case["config"]),
                    ("half", case["comparison_configs"]["half_dt"]),
                ):
                    checked(
                        f"{seed}_{anchor}_{name}",
                        long.KolmogorovReferenceConfig(**conf),
                        inputs,
                        [data[name + "_next"].copy()],
                    )
        states = _retained_states(parent, cases[RESUME_SEED])
        checked(
            "restart616",
            config,
            states[restart],
            [states[i] for i in range(restart + 1, resume + 1)],
        )
        fine_row = next(q for q in first["queries"] if q["anchor_step"] == fine_anchor)
        fine_config = long.KolmogorovReferenceConfig(
            **first["comparison_configs"]["fine_half_dt"]
        )
        with np.load(parent / fine_row["file"], allow_pickle=False) as data:
            checked(
                "fine611",
                fine_config,
                long.resize_dealiased_vorticity(data["input"], fine_config.resolution),
                [data["fine_next"].copy()],
            )
        if len(record["calls"]) != record["expected_solver_calls"]:
            raise ValueError("runtime replay call coverage mismatch")
        record["status"] = "completed"
    except long.BudgetExceeded as error:
        record.update(status="incomplete_budget", error=str(error))
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        record.update(
            status="replay_failed", error_type=type(error).__name__, error=str(error)
        )
    finally:
        for row in record["calls"]:
            if row["status"] == "running":
                row.update(
                    status=record["status"],
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                )
        record.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            seconds=perf_counter() - started,
        )
        long._json(path, record)
    return record


def _copy_case_evidence(parent, output, cases):
    copied = {}
    for case in cases:
        names = [f"case_{case['seed']}.json"]
        if case["started_generation"]:
            names += [
                f"checkpoint_{case['seed']}.json",
                f"progress_{case['seed']}.jsonl",
            ]
        names += [
            row["file"]
            for key in ("blocks", "queries", "continuation_rows")
            for row in case[key]
        ]
        for name in names:
            if (output / name).exists():
                raise FileExistsError(output / name)
            shutil.copy2(parent / name, output / name)
            digest = long._sha256(parent / name)
            if long._sha256(output / name) != digest:
                raise ValueError("copied parent case evidence differs")
            copied[name] = digest
    return copied


def _continuation(output, record, half, fine, anchor, anchor_step, protocol, deadline):
    """Compose both complete states independently; persist each partial call."""
    coarse_state = anchor.copy()
    fine_state = long.resize_dealiased_vorticity(anchor, fine.config.resolution)
    for step in range(1, protocol.late_horizon + 1):
        row = {
            "anchor_step": anchor_step,
            "step": step,
            "status": "running",
            "file": f"continuation_{record['seed']}_{anchor_step:05d}_{step:02d}.npz",
            "calls": {},
        }
        record["continuation_rows"].append(row)
        arrays = {"coarse_input": coarse_state.copy(), "fine_input": fine_state.copy()}
        _save_arrays(output, row, arrays)
        case_path = output / f"case_{record['seed']}.json"
        long._json(case_path, record)
        for name, stepper, state in (
            ("coarse_next", half, coarse_state),
            ("fine_next", fine, fine_state),
        ):
            arrays[name], row["calls"][name] = _advance(stepper, state, deadline)
            _save_arrays(output, row, arrays)
            long._json(case_path, record)
        coarse_state, fine_state = arrays["coarse_next"], arrays["fine_next"]
        restricted = long.resize_dealiased_vorticity(fine_state, half.config.resolution)
        row.update(
            status="completed",
            restricted_state_relative_l2=long._relative(coarse_state, restricted),
            fine_discarded_state_relative_l2=long._relative(
                long.resize_dealiased_vorticity(restricted, fine.config.resolution),
                fine_state,
            ),
        )
        long._json(case_path, record)


def _run_case(output, protocol, seed, role, deadline, resume_record=None):
    config = long.KolmogorovReferenceConfig(
        resolution=protocol.resolution,
        viscosity=0.01,
        linear_drag=0.1,
        forcing_amplitude=1.0,
        forcing_wavenumber=4,
        macro_dt=protocol.macro_dt,
        dt_max=protocol.dt_max,
    )
    stepper = BudgetedReferenceStepper(config, deadline)
    refined = replace(config, dt_max=config.dt_max / 2, cfl=config.cfl / 2)
    half = BudgetedReferenceStepper(refined, deadline)
    fine = BudgetedReferenceStepper(
        replace(refined, resolution=2 * config.resolution), deadline
    )
    record = {
        "seed": seed,
        "role": role,
        "status": "running",
        "started_generation": False,
        "config": asdict(config),
        "comparison_configs": {
            "half_dt": asdict(half.config),
            "fine_half_dt": asdict(fine.config),
        },
        "blocks": [],
        "last_retained_step": None,
        "trajectory_diagnostics": [],
        "queries": [],
        "continuation_rows": [],
        "expected_continuation_anchors": None if seed in LATE_SEEDS else [],
        "maximum_palinstrophy": None,
        "expected_anchor_steps": None,
    }
    case_path = output / f"case_{seed}.json"
    indices, states = [], []
    peak_state = None
    started = perf_counter()
    start_index = 0
    prefix = None
    if resume_record is not None:
        if seed != RESUME_SEED:
            raise ValueError("only the pinned partial seed may resume")
        record = copy.deepcopy(resume_record)
        for name in ("error", "error_type", "seconds"):
            record.pop(name, None)
        record["status"] = "running"
        prefix = _retained_states(output, record)
        start_index = record["last_retained_step"] + 1
        current = prefix[start_index - 1]
        peak_index = max(
            record["trajectory_diagnostics"],
            key=lambda row: row["structure"]["palinstrophy"],
        )["step"]
        if record["maximum_palinstrophy"]["step"] != peak_index:
            raise ValueError("resume prefix peak metadata mismatch")
        peak_state = prefix[peak_index]

    try:
        if perf_counter() >= deadline:
            raise long.BudgetExceeded(
                "budget expired before this assigned seed started"
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if prefix is None:
                current = _initial(stepper, seed)
                record["started_generation"] = True
                indices.append(0)
                states.append(current)
                long._flush_block(output, seed, indices, states, record)
            else:
                # Replace only C's partial query; every new comparison uses
                # one runtime for base, half and fine answers.
                incomplete = [
                    row for row in record["queries"] if row["status"] != "completed"
                ]
                if (
                    len(incomplete) != 1
                    or incomplete[0]["anchor_step"] != start_index - 1
                ):
                    raise ValueError("resume query does not match retained checkpoint")
                record["queries"].remove(incomplete[0])
                _clean_query(
                    output,
                    record,
                    stepper,
                    half,
                    fine,
                    current,
                    start_index - 1,
                    deadline,
                )
            with (output / f"progress_{seed}.jsonl").open(
                "a" if prefix is not None else "x", encoding="utf-8"
            ) as progress:
                for index in range(start_index, protocol.steps + 1):
                    if index:
                        current, timing = _advance(stepper, current, deadline)
                        indices.append(index)
                        states.append(current)
                    else:
                        timing = None
                    diagnostic = {
                        "step": index,
                        "structure": asdict(stepper.diagnostics_canonical(current)),
                        "state_sha256": long._state_hash(current),
                        "call": timing,
                    }
                    encoded = json.dumps(diagnostic, allow_nan=False)
                    record["trajectory_diagnostics"].append(diagnostic)
                    progress.write(encoded + "\n")
                    progress.flush()
                    palinstrophy = diagnostic["structure"]["palinstrophy"]
                    # Strict comparison retains the earliest index in an exact tie.
                    if (
                        peak_state is None
                        or palinstrophy > record["maximum_palinstrophy"]["value"]
                    ):
                        peak_state = current.copy()
                        record["maximum_palinstrophy"] = {
                            "step": index,
                            "value": palinstrophy,
                            "state_sha256": diagnostic["state_sha256"],
                            "complete_trajectory": False,
                        }
                    if (
                        len(states) >= protocol.block_steps
                        or index in protocol.anchor_steps
                        or index == protocol.steps
                    ):
                        long._flush_block(output, seed, indices, states, record)
                        long._json(case_path, record)
                    if index in protocol.anchor_steps:
                        _clean_query(
                            output,
                            record,
                            stepper,
                            half,
                            fine,
                            current,
                            index,
                            deadline,
                        )
                    if seed in LATE_SEEDS and index == protocol.late_anchor:
                        _continuation(
                            output,
                            record,
                            half,
                            fine,
                            current,
                            index,
                            protocol,
                            deadline,
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
                        stepper,
                        half,
                        fine,
                        peak_state,
                        peak["step"],
                        deadline,
                    )
                if seed in LATE_SEEDS:
                    record["expected_continuation_anchors"] = sorted(
                        {protocol.late_anchor, peak["step"]}
                    )
                    if peak["step"] != protocol.late_anchor:
                        _continuation(
                            output,
                            record,
                            half,
                            fine,
                            peak_state,
                            peak["step"],
                            protocol,
                            deadline,
                        )
                record["status"] = "completed"
    except long.BudgetExceeded as error:
        record.update(status="incomplete_budget", error=str(error))
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        record.update(
            status="solver_failure", error_type=type(error).__name__, error=str(error)
        )
    finally:
        for row in record["queries"] + record["continuation_rows"]:
            if row["status"] == "running":
                row["status"] = record["status"]
        long._flush_block(output, seed, indices, states, record)
        record["seconds"] = perf_counter() - started
        long._json(case_path, record)
    return record


def _numeric_gates(records, protocol):
    complete = [case["seed"] for case in records] == list(protocol.seeds) and all(
        case["status"] == "completed"
        and case["last_retained_step"] == protocol.steps
        and case["maximum_palinstrophy"]["complete_trajectory"] is True
        and case["expected_anchor_steps"]
        == sorted(set(protocol.anchor_steps) | {case["maximum_palinstrophy"]["step"]})
        and case["expected_continuation_anchors"]
        == (
            sorted({protocol.late_anchor, case["maximum_palinstrophy"]["step"]})
            if case["seed"] in LATE_SEEDS
            else []
        )
        for case in records
    )
    expected_clean = (
        sum(len(case["expected_anchor_steps"]) for case in records)
        if len(records) == len(protocol.seeds)
        and all(case["expected_anchor_steps"] is not None for case in records)
        else None
    )
    clean_complete = complete and all(
        sorted(row["anchor_step"] for row in case["queries"])
        == case["expected_anchor_steps"]
        and all(row["status"] == "completed" for row in case["queries"])
        for case in records
    )
    expected_continuations = (
        sum(len(case["expected_continuation_anchors"]) for case in records)
        if len(records) == len(protocol.seeds)
        and all(case["expected_continuation_anchors"] is not None for case in records)
        else None
    )
    continuation_complete = complete and all(
        sorted((row["anchor_step"], row["step"]) for row in case["continuation_rows"])
        == [
            (anchor, step)
            for anchor in case["expected_continuation_anchors"]
            for step in range(1, protocol.late_horizon + 1)
        ]
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
            if row["step"] == protocol.late_horizon and row["status"] == "completed"
        ],
    }
    gates = {}
    for name, samples in values.items():
        expected = (
            expected_continuations
            if name == "continuation_endpoint_relative_l2"
            else expected_clean
        )
        coverage = (
            continuation_complete
            if name == "continuation_endpoint_relative_l2"
            else clean_complete
        )
        covered = coverage and len(samples) == expected
        gates[name] = {
            "maximum": max(samples) if samples else None,
            "limit": GATE_LIMITS[name],
            "sample_count": len(samples),
            "expected_sample_count": expected,
            "complete": covered,
            "pass": bool(
                all(
                    np.isfinite(value) and 0 <= value <= GATE_LIMITS[name]
                    for value in samples
                )
            )
            if covered
            else None,
        }
    return gates


def _sources():
    return {name: long._sha256(REPO_ROOT / name) for name in SOURCE_PATHS}


def run_population(
    parent: Path, parent_source: Path, output: Path, *, unit_fixture: bool = False
):
    """The Python-only tiny fixture cannot produce the real population identity."""
    output, parent, parent_source = Path(output), Path(parent), Path(parent_source)
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    roles = SEED_ROLES
    protocol = (
        replace(
            PROTOCOL,
            resolution=16,
            seeds=tuple(seed for seed, _ in roles),
            steps=8,
            anchor_steps=(0, 4, 6, 8),
            late_anchor=6,
            late_horizon=2,
            block_steps=2,
            macro_dt=0.004,
            high_mode=(4, 3),
            wall_seconds=60,
            fixture_label="unit_test_only",
        )
        if unit_fixture
        else PROTOCOL
    )
    sources_before = _sources()
    if any(sources_before[name] != digest for name, digest in FROZEN_SOURCES.items()):
        raise ValueError("frozen readiness/reference source mismatch")
    if any(
        Path(module.__file__).resolve() != (REPO_ROOT / name).resolve()
        for module, name in (
            (
                long,
                "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
            ),
            (peak, "scripts/time_dependent_no/screen_kolmogorov_peak_refinement.py"),
        )
    ):
        raise ValueError(
            "imported trajectory helpers are outside the bound source root"
        )
    deadline = started + protocol.wall_seconds
    evidence, parent_result = validate_parent(
        parent, parent_source, unit_fixture=unit_fixture
    )
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError("at least 6 GiB free output space is required")
    output.mkdir()
    run_id = RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    launch = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "seed_roles": [{"seed": seed, "role": role} for seed, role in roles],
        "late_seeds": list(LATE_SEEDS),
        "gate_limits": GATE_LIMITS,
        "parent_manifest_sha256": evidence["manifest_sha256"],
        "parent_evidence": evidence,
        "initial_recipe": {
            "modes": long.INITIAL_MODES,
            "phase_law": "NumPy default_rng(seed), uniform[0,2pi) per cosine",
            "perturbation_rms_over_laminar_rms": 0.25,
            "zero_mean_velocity": True,
            "native_grid_generation": True,
        },
        "sources": sources_before,
        "free_disk_bytes_at_launch": free_bytes,
    }
    long._json(output / "launch.json", launch)
    replay = _runtime_replay(
        parent, output, parent_result, protocol, deadline, unit_fixture=unit_fixture
    )
    records = []
    origin = {
        "parent_runtime": parent_result["runtime"],
        "parent_source_hashes": evidence["source_hashes"],
        "inherited_complete_seeds": [
            case["seed"]
            for case in parent_result["cases"]
            if case["status"] == "completed"
        ],
        "resume_seed": RESUME_SEED,
        "resume_prefix_last_step": 4 if unit_fixture else 16,
        "parent_case_seconds": {
            str(case["seed"]): case["seconds"] for case in parent_result["cases"]
        },
        "parent_resume_error": parent_result["cases"][5].get("error"),
        "copied_case_owned_hashes": {},
        "timing_meaning": "Inherited complete-case seconds and resumed prefix call times belong to B's runtime. C seconds measure only this successor; never sum all case seconds as C compute.",
    }
    if replay["status"] == "completed":
        origin["copied_case_owned_hashes"] = _copy_case_evidence(
            parent, output, parent_result["cases"]
        )
        long._json(output / "origin.json", origin)
        for previous in parent_result["cases"]:
            seed, role = previous["seed"], previous["role"]
            if previous["status"] == "completed":
                record = previous
            else:
                record = _run_case(
                    output,
                    protocol,
                    seed,
                    role,
                    deadline,
                    resume_record=previous if seed == RESUME_SEED else None,
                )
            records.append(record)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "role": role,
                        "status": record["status"],
                        "last_retained_step": record["last_retained_step"],
                    }
                ),
                flush=True,
            )
    else:
        long._json(output / "origin.json", origin)
    sources_after = _sources()
    parent_stable = all(
        (parent / name).is_file() and long._sha256(parent / name) == digest
        for name, digest in evidence["evidence_hashes"].items()
    )
    parent_stable = (
        parent_stable
        and all(
            long._sha256(parent_source / name) == digest
            for name, digest in evidence["source_hashes"].items()
        )
        and long._sha256(parent_source.parent / "independent_population_audit.json")
        == evidence["audit_sha256"]
    )
    status = "completed" if replay["status"] == "completed" else replay["status"]
    if any(case["status"] == "incomplete_budget" for case in records):
        status = "incomplete_budget"
    elif any(case["status"] != "completed" for case in records):
        status = "solver_failure"
    if sources_after != sources_before:
        status = "invalid_source"
    elif not parent_stable:
        status = "invalid_parent"
    gates = _numeric_gates(records, protocol)
    result = {
        "run_id": run_id,
        "status": status,
        "scope": protocol.fixture_label,
        "parent_manifest_sha256": evidence["manifest_sha256"],
        "parent_evidence": evidence,
        "parent_evidence_stable": parent_stable,
        "initial_recipe": launch["initial_recipe"],
        "protocol": asdict(protocol),
        "seed_roles": launch["seed_roles"],
        "cases": records,
        "runtime_replay": replay,
        "origin": origin,
        "generation_started": replay["status"] == "completed",
        "numeric_gates": gates,
        "engineering_gates_pass": all(gate["pass"] for gate in gates.values())
        if status == "completed"
        else None,
        "retained_transitions_by_role": {
            role: sum(
                case["last_retained_step"] or 0
                for case in records
                if case["role"] == role
            )
            for role in ("train", "development")
        },
        "stationarity_claim": False,
        "uniform_numerical_guarantee": False,
        "displaced_input_qualification": False,
        "confirmation_access": False,
        "existing_scientific_data_access": not unit_fixture,
        "parent_payload_bytes_read_for_hashing": True,
        "parent_state_arrays_decoded": replay["parent_state_arrays_decoded"],
        "generated_solver_states": replay["generated_solver_states"],
        "network_access": False,
        "model_access": False,
        "sources_before": sources_before,
        "sources_after": sources_after,
        "source_stable": sources_before == sources_after,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "seconds": perf_counter() - started,
        "interpretation": "Intended parent-bound recovery: after the fixed runtime-equivalence gate, inherit complete B cases byte-identically, retain seed616's prefix, recompute its partial query entirely in C, then resume at17. Observed completion is reported by runtime_replay, generation_started and per-case status, not implied by this recipe. Origin metadata distinguishes inherited runtime/cost from new work. The unchanged sampled checks do not imply a uniform or displaced-input guarantee. No seed replacement or confirmation access. Budget starts before replay/copy/new work; final flush and hashes follow compute shutdown.",
    }
    long._json(output / "result.json", result)
    long._json(
        output / "artifact_manifest.json",
        {
            "run_id": run_id,
            "sources": sources_before,
            "source_stable": result["source_stable"],
            "parent_manifest_sha256": evidence["manifest_sha256"],
            "artifacts": {
                path.name: long._sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_population(args.parent, args.parent_source, args.output)
    if result["status"] != "completed" or result["engineering_gates_pass"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
