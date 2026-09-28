"""Add 24 train-only paths to qualified C; never copy or change its 12 paths.

Run ``python -m scripts.time_dependent_no.extend_kolmogorov_training_population
--parent <C-packet> --parent-source <frozen-C-source-root> --output <fresh>``.
The resulting index references C plus this packet. No model is evaluated or fit.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.time_dependent_no import fit_kolmogorov_clean as clean
from scripts.time_dependent_no import generate_kolmogorov_population as parent_code

RUN_ID = "CM_NEXT_KF_TRAIN32_POP_20260908A"
REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_SEEDS = tuple(range(2026090801, 2026090825))
CONTINUATION_SEEDS = NEW_SEEDS[:2]
MINIMUM_FREE_BYTES = 9 * 1024**3
REPLAY_LIMIT = 1e-10
PROTOCOL = replace(
    parent_code.PROTOCOL,
    seeds=NEW_SEEDS,
    fixture_label="train_only_same_law_coverage_expansion",
)
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/time_dependent_no/extend_kolmogorov_training_population.py",
            "tests/time_dependent_no/test_extend_kolmogorov_training_population.py",
            *clean.SOURCE_PATHS,
            *parent_code.SOURCE_PATHS,
        )
    )
)


def population_index(seeds):
    """Two explicit roots; existing validation identities cannot enter training."""
    original_train = [s for s, role in clean.SEED_ROLES if role == "train"]
    validation = [s for s, role in clean.SEED_ROLES if role == "development"]
    if len(seeds) != len(set(seeds)) or set(seeds) & {s for s, _ in clean.SEED_ROLES}:
        raise ValueError("new seeds must be unique and disjoint from every C role")
    return [
        {"seed": seed, "role": role, "packet": packet, "case_file": f"case_{seed}.json"}
        for group, role, packet in (
            (original_train, "train", "parent_C"),
            (seeds, "train", "this_packet"),
            (validation, "development", "parent_C"),
        )
        for seed in group
    ]


def runtime_replay(parent, result, output, deadline):
    """Repeat the existing train-611 clean query on base, half and fine grids."""
    case = result["cases"][0]
    query = next(row for row in case["queries"] if row["anchor_step"] == 0)
    record = {"status": "running", "relative_l2_limit": REPLAY_LIMIT, "calls": []}
    path = output / "runtime_replay.json"
    clean._write(path, record)
    try:
        with np.load(parent / query["file"], allow_pickle=False) as saved:
            anchor = saved["input"].copy()
            for key, config in (
                ("base_next", case["config"]),
                ("half_next", case["comparison_configs"]["half_dt"]),
                ("fine_next", case["comparison_configs"]["fine_half_dt"]),
            ):
                state = parent_code.long.resize_dealiased_vorticity(
                    anchor, config["resolution"]
                )
                previous = None
                for repetition in range(2):
                    stepper = parent_code.BudgetedReferenceStepper(
                        parent_code.long.KolmogorovReferenceConfig(**config), deadline
                    )
                    row = {
                        "kind": key,
                        "repetition": repetition,
                        "status": "running",
                        "file": f"replay_{key}_{repetition}.npz",
                    }
                    record["calls"].append(row)
                    clean._write(path, record)
                    predicted, row["timing"] = parent_code._advance(
                        stepper, state, deadline
                    )
                    parent_code._save_arrays(
                        output,
                        row,
                        {
                            "input": state,
                            "reference": saved[key],
                            "output": predicted,
                        },
                    )
                    error = parent_code.long._relative(predicted, saved[key])
                    same = previous is None or np.array_equal(previous, predicted)
                    row.update(
                        status="completed", relative_l2=error, repeat_equal=bool(same)
                    )
                    clean._write(path, record)
                    if not np.isfinite(error) or error > REPLAY_LIMIT or not same:
                        raise ValueError("frozen C solver query does not replay")
                    previous = predicted
        record["status"] = "completed"
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        record.update(
            status="replay_failed", error_type=type(error).__name__, error=str(error)
        )
    clean._write(path, record)
    return record


def add_continuations(output, case, protocol, deadline):
    """Check the peak and late anchor using independently recurrent N/N*2 states."""
    states = parent_code._retained_states(output, case)
    anchors = sorted({case["maximum_palinstrophy"]["step"], protocol.late_anchor})
    case.update(status="running", expected_continuation_anchors=anchors)
    path = output / f"case_{case['seed']}.json"
    clean._write(path, case)
    try:
        configs = case["comparison_configs"]
        half, fine = (
            parent_code.BudgetedReferenceStepper(
                parent_code.long.KolmogorovReferenceConfig(**configs[key]), deadline
            )
            for key in ("half_dt", "fine_half_dt")
        )
        for anchor in anchors:
            parent_code._continuation(
                output, case, half, fine, states[anchor], anchor, protocol, deadline
            )
        case["status"] = "completed"
    except parent_code.long.BudgetExceeded as error:
        case.update(status="incomplete_budget", error=str(error))
    except (
        OSError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        OverflowError,
    ) as error:
        case.update(
            status="solver_failure", error_type=type(error).__name__, error=str(error)
        )
    finally:
        for row in case["continuation_rows"]:
            if row["status"] == "running":
                row["status"] = case["status"]
        clean._write(path, case)


def numerical_gates(cases, protocol, continuation_seeds):
    complete = [case["seed"] for case in cases] == list(protocol.seeds) and all(
        case["role"] == "train"
        and case["status"] == "completed"
        and case["last_retained_step"] == protocol.steps
        and case["maximum_palinstrophy"]["complete_trajectory"] is True
        for case in cases
    )
    clean_rows, continuation_rows = [], []
    expected_clean = expected_continuations = 0
    for case in cases:
        peak = case["maximum_palinstrophy"]
        if peak is None:
            complete = False
            continue
        anchors = sorted(set(protocol.anchor_steps) | {peak["step"]})
        cont = (
            sorted({protocol.late_anchor, peak["step"]})
            if case["seed"] in continuation_seeds
            else []
        )
        expected_clean += len(anchors)
        expected_continuations += len(cont)
        complete = complete and (
            case["expected_anchor_steps"] == anchors
            and sorted(row["anchor_step"] for row in case["queries"]) == anchors
            and case["expected_continuation_anchors"] == cont
            and sorted((r["anchor_step"], r["step"]) for r in case["continuation_rows"])
            == [(a, t) for a in cont for t in range(1, protocol.late_horizon + 1)]
            and all(
                r["status"] == "completed"
                for r in case["queries"] + case["continuation_rows"]
            )
        )
        clean_rows.extend(r for r in case["queries"] if r["status"] == "completed")
        continuation_rows.extend(
            r
            for r in case["continuation_rows"]
            if r["status"] == "completed" and r["step"] == protocol.late_horizon
        )
    gates = {}
    for name, key, rows, count in (
        (
            "clean_spatial_relative_l2",
            "spatial_relative_l2",
            clean_rows,
            expected_clean,
        ),
        (
            "fine_discarded_state_relative_l2",
            "fine_discarded_state_relative_l2",
            clean_rows,
            expected_clean,
        ),
        (
            "continuation_endpoint_relative_l2",
            "restricted_state_relative_l2",
            continuation_rows,
            expected_continuations,
        ),
    ):
        values = [r[key] for r in rows]
        covered = complete and count > 0 and len(values) == count
        limit = parent_code.GATE_LIMITS[name]
        gates[name] = {
            "maximum": max(values) if values else None,
            "limit": limit,
            "sample_count": len(values),
            "expected_sample_count": count,
            "complete": covered,
            "pass": all(np.isfinite(v) and 0 <= v <= limit for v in values)
            if covered
            else None,
        }
    return gates


def run(parent, parent_source, output, *, unit_fixture=False):
    parent, parent_source, output = map(Path, (parent, parent_source, output))
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    seeds = NEW_SEEDS[:2] if unit_fixture else NEW_SEEDS
    protocol = (
        replace(
            PROTOCOL,
            resolution=16,
            steps=4,
            anchor_steps=(0, 1, 4),
            late_anchor=2,
            late_horizon=2,
            block_steps=2,
            wall_seconds=60,
            seeds=seeds,
            fixture_label="unit_test_only",
        )
        if unit_fixture
        else PROTOCOL
    )
    sources = {name: clean._hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    previous, evidence = clean.validate_parent(
        parent, parent_source, unit_fixture=unit_fixture
    )
    if not unit_fixture and any(
        sources[name] != digest for name, digest in evidence["sources"].items()
    ):
        raise ValueError("live solver/population dependencies differ from C")
    if shutil.disk_usage(output.parent).free < (
        1024**2 if unit_fixture else MINIMUM_FREE_BYTES
    ):
        raise RuntimeError("insufficient free space for the bounded population packet")
    index = population_index(seeds)
    output.mkdir()
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "parent_manifest_sha256": evidence["manifest_sha256"],
        "parent_evidence": evidence,
        "initial_recipe": previous["initial_recipe"],
        "population_index": index,
        "continuation_seeds": list(CONTINUATION_SEEDS),
        "sources_before": sources,
        "cases": [],
        "model_evaluated": False,
        "optimization_steps": 0,
        "new_development_trajectories": 0,
        "protected_access": False,
        "network_access": False,
        "interpretation": (
            "Same-law clean training coverage only. C is read-only; no seed "
            "replacement, new regime, model training, geometry claim or "
            "displaced-input qualification."
        ),
    }
    clean._write(output / "launch.json", record)
    deadline = started + protocol.wall_seconds
    replay = runtime_replay(parent, previous, output, deadline)
    record["runtime_replay"] = replay
    if replay["status"] == "completed":
        for seed in seeds:
            case_started = perf_counter()
            case = parent_code._run_case(output, protocol, seed, "train", deadline)
            if seed in CONTINUATION_SEEDS and case["status"] == "completed":
                add_continuations(output, case, protocol, deadline)
            case["seconds"] = perf_counter() - case_started
            clean._write(output / f"case_{seed}.json", case)
            record["cases"].append(case)
            clean._write(output / "progress.json", record)
            print(
                json.dumps(
                    {"seed": seed, "status": case["status"], "seconds": case["seconds"]}
                ),
                flush=True,
            )
    statuses = [c["status"] for c in record["cases"]]
    record["status"] = (
        "replay_failed"
        if replay["status"] != "completed"
        else "incomplete_budget"
        if "incomplete_budget" in statuses
        else "solver_failure"
        if any(s != "completed" for s in statuses)
        else "completed"
    )
    gates = numerical_gates(record["cases"], protocol, CONTINUATION_SEEDS)
    after = {name: clean._hash(REPO_ROOT / name) for name in SOURCE_PATHS}
    parent_stable = all(
        clean._hash(clean._bound(root, name)) == digest
        for root, values in (
            (parent, evidence["artifacts"]),
            (parent_source, evidence["sources"]),
        )
        for name, digest in values.items()
    )
    if after != sources or not parent_stable:
        record["status"] = "invalid_provenance"
    record.update(
        numeric_gates=gates,
        engineering_gates_pass=all(g["pass"] is True for g in gates.values())
        if record["status"] == "completed"
        else None,
        sources_after=after,
        source_stable=after == sources,
        parent_evidence_stable=parent_stable,
        retained_new_training_transitions=sum(
            c["last_retained_step"] or 0 for c in record["cases"]
        ),
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
    )
    clean._write(output / "result.json", record)
    clean._write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "status": record["status"],
            "sources": sources,
            "source_stable": record["source_stable"],
            "parent_manifest_sha256": evidence["manifest_sha256"],
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.parent, args.parent_source, args.output)
    if result["status"] != "completed" or result["engineering_gates_pass"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
