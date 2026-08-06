#!/usr/bin/env python3
"""Audit and aggregate the matched D072 N0/G1/S1 training matrices."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    jsonable_args,
    sha256_file,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)

SCHEMA = "pcno_boundary_field_training_analysis_v1"
ARMS = ("N0", "G1", "S1")
SEEDS = (20260718, 20260719, 20260720)
RUN_PATTERN = re.compile(r"^(?P<family>dynamic|bump)_s(?P<seed>\d+)_(?P<arm>N0|G1|S1)$")

FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "dynamic_fv": {
        "directory_prefix": "dynamic",
        "horizons": (20, 30),
        "primary_horizon": 30,
        "epochs": 50,
        "optimizer_steps": 13_400,
        "presentations": 51_200,
        "train_trajectories": 84,
        "validation_trajectories": 24,
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "input_dims": {"N0": 8, "G1": 9, "S1": 10},
        "field_modes": {
            "N0": "none",
            "G1": "geometry_collar",
            "S1": "semantic_collar",
        },
    },
    "bump": {
        "directory_prefix": "bump",
        "horizons": (20, 40, 60, 79),
        "primary_horizon": 79,
        "epochs": 40,
        "optimizer_steps": 216_000,
        "presentations": 853_200,
        "train_trajectories": 270,
        "validation_trajectories": 30,
        "boundary_mode": "causal_nodal_physical",
        "raw_recurrence": False,
        "input_dims": {"N0": 8, "G1": 9, "S1": 11},
        "field_modes": {
            "N0": "none",
            "G1": "geometry_collar",
            "S1": "semantic_collar",
        },
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILY_CONTRACTS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Exploratory compact-mirror mode only; scientific audits must omit this.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(value)


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"history row {line_number} is not a mapping: {path}")
        rows.append(dict(value))
    if not rows:
        raise ValueError(f"empty training history: {path}")
    return rows


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs: {actual!r} != {expected!r}")


def _finite_number(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} is nonfinite: {value!r}")
    return numeric


def _run_directories(root: Path, family: str) -> dict[tuple[int, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_prefix = FAMILY_CONTRACTS[family]["directory_prefix"]
    result = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None or match.group("family") != expected_prefix:
            continue
        key = (int(match.group("seed")), match.group("arm"))
        if key in result:
            raise ValueError(f"duplicate run for {key}: {path}")
        result[key] = path
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(result) != expected:
        raise ValueError(
            f"run matrix differs; missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    return result


def _selected_row(
    summary: Mapping[str, Any], history: Sequence[Mapping[str, Any]], run_name: str
) -> Mapping[str, Any]:
    best_epoch = int(summary["best_epoch"])
    selected = [row for row in history if int(row["epoch"]) == best_epoch]
    if len(selected) != 1 or not isinstance(selected[0].get("rollout"), Mapping):
        raise ValueError(f"{run_name} lacks one selected rollout row")
    row = selected[0]
    if list(row["best_selection"]) != list(summary["best_selection"]):
        raise ValueError(f"{run_name} selected-row and summary rankings differ")
    return row


def _trajectory_endpoints(rollout: Mapping[str, Any], horizon: int) -> dict[str, float]:
    result = {}
    for row in rollout.get("trajectories", []):
        values = row.get("endpoint_relative_l2", {})
        value = values.get(str(horizon))
        if value is not None:
            result[str(row["trajectory"])] = _finite_number(
                value, f"trajectory {row['trajectory']} H{horizon}"
            )
    return result


def _verify_checkpoint_hashes(
    run_dir: Path, summary: Mapping[str, Any], *, skip: bool
) -> dict[str, Any]:
    expected = summary["artifact_sha256"]
    rows = {}
    for name, filename in (
        ("best_checkpoint", "best.pt"),
        ("last_checkpoint", "last.pt"),
    ):
        path = run_dir / filename
        if skip:
            rows[name] = {
                "status": "skipped_compact_mirror",
                "expected": expected[name],
            }
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        _require_equal(actual, expected[name], f"{run_dir.name} {name} SHA-256")
        rows[name] = {
            "status": "verified",
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return rows


def _extract_run(
    run_dir: Path,
    *,
    family: str,
    seed: int,
    arm: str,
    skip_checkpoint_hash: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    contract = FAMILY_CONTRACTS[family]
    summary = _read_json(run_dir / "summary.json")
    run_contract = _read_json(run_dir / "run_contract.json")
    split = _read_json(run_dir / "split.json")
    history = _read_history(run_dir / "metrics.jsonl")
    selected = _selected_row(summary, history, run_dir.name)

    checks = {
        "completed_epochs": (summary["completed_epochs"], contract["epochs"]),
        "actual_optimizer_steps": (
            summary["actual_optimizer_steps"],
            contract["optimizer_steps"],
        ),
        "actual_presentations": (
            summary["actual_presentations"],
            contract["presentations"],
        ),
        "train_trajectories": (
            summary["train_trajectories"],
            contract["train_trajectories"],
        ),
        "validation_trajectories": (
            summary["validation_trajectories"],
            contract["validation_trajectories"],
        ),
        "test_trajectories": (summary["test_trajectories"], 0),
        "boundary_mode": (summary["boundary_mode"], contract["boundary_mode"]),
        "raw_recurrence": (summary["raw_recurrence"], contract["raw_recurrence"]),
        "field_mode": (
            summary["boundary_field_mode"],
            contract["field_modes"][arm],
        ),
        "input_dim": (summary["model_config"]["in_dim"], contract["input_dims"][arm]),
    }
    for name, (actual, expected) in checks.items():
        _require_equal(actual, expected, f"{run_dir.name} {name}")
    _require_equal(
        len(split["train_keys"]), contract["train_trajectories"], "train split"
    )
    _require_equal(
        len(split["val_keys"]), contract["validation_trajectories"], "validation split"
    )
    _require_equal(split.get("test_keys", []), [], "sealed test split")
    _require_equal(
        split["data_manifest_digest"],
        summary["data_manifest_digest"],
        f"{run_dir.name} split manifest binding",
    )
    initialization = summary["initialization_control"]
    if arm == "N0":
        _require_equal(
            initialization["ordinary_initialization"], True, "N0 initialization"
        )
    else:
        for field in (
            "mathematical_initial_function_match",
            "lifting_copy_exact",
            "all_non_lifting_state_copied",
            "cpu_rng_state_matches_no_boundary_arm",
        ):
            _require_equal(initialization[field], True, f"{run_dir.name} {field}")

    rollout = selected["rollout"]
    validation = selected["validation"]
    horizons = contract["horizons"]
    primary_horizon = contract["primary_horizon"]
    record: dict[str, Any] = {
        "family": family,
        "run": run_dir.name,
        "seed": seed,
        "arm": arm,
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "completion_rate": _finite_number(rollout["completion_rate"], "completion"),
        "mean_survival_fraction": _finite_number(
            rollout["mean_survival_fraction"], "survival"
        ),
        "selected_one_step_relative_l2": _finite_number(
            validation["all_relative_l2"], "one-step error"
        ),
        "selected_one_step_boundary_relative_l2": _finite_number(
            validation["boundary_relative_l2"], "boundary one-step error"
        ),
        "selected_one_step_normal_relative_l2": _finite_number(
            validation["normal_relative_l2"], "normal one-step error"
        ),
        "elapsed_seconds": _finite_number(summary["elapsed_seconds"], "elapsed time"),
        "gpu_max_memory_bytes": int(summary["gpu_max_memory_bytes"]),
        "parameter_count": int(summary["parameter_count"]),
        "checkpoint_sha256": summary["artifact_sha256"]["best_checkpoint"],
        "data_manifest_digest": summary["data_manifest_digest"],
        "normalization_digest": summary["normalization_digest"],
        "source_set_digest": summary["source_snapshot"]["source_set_digest"],
        "provenance_set_digest": summary["source_snapshot"]["provenance_set_digest"],
        "presentation_stream_sha256": run_contract["exposure_contract"][
            "presentation_stream_sha256"
        ],
    }
    for horizon in horizons:
        value = rollout["mean_endpoint_relative_l2"].get(str(horizon))
        count = int(rollout["endpoint_population_count"].get(str(horizon), 0))
        record[f"H{horizon}_relative_l2"] = (
            None if value is None else _finite_number(value, f"H{horizon} error")
        )
        record[f"H{horizon}_population"] = count
    selected_primary = record[f"H{primary_horizon}_relative_l2"]
    if selected_primary is not None:
        ranking_value = abs(float(summary["best_selection"][3]))
        if not math.isclose(
            selected_primary, ranking_value, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"{run_dir.name} selected metric and ranking differ")

    curve_rows = []
    for row in history:
        rollout_row = row.get("rollout")
        primary = None
        if isinstance(rollout_row, Mapping):
            primary = rollout_row["mean_endpoint_relative_l2"].get(str(primary_horizon))
        curve_rows.append(
            {
                "family": family,
                "run": run_dir.name,
                "seed": seed,
                "arm": arm,
                "epoch": int(row["epoch"]),
                "learning_rate": _finite_number(row["learning_rate"], "learning rate"),
                "train_relative_l2": _finite_number(
                    row["train"]["relative_l2"], "train error"
                ),
                "validation_relative_l2": _finite_number(
                    row["validation"]["all_relative_l2"], "validation error"
                ),
                f"H{primary_horizon}_relative_l2": (
                    None
                    if primary is None
                    else _finite_number(primary, "rollout error")
                ),
                "selected": int(row["epoch"]) == int(summary["best_epoch"]),
            }
        )
    endpoint_map = _trajectory_endpoints(rollout, primary_horizon)
    hashes = _verify_checkpoint_hashes(run_dir, summary, skip=skip_checkpoint_hash)
    return record, curve_rows, endpoint_map, hashes


def _sample_std(values: Sequence[float]) -> float:
    return 0.0 if len(values) < 2 else float(np.std(values, ddof=1))


def _aggregate(
    records: Sequence[Mapping[str, Any]], primary_horizon: int
) -> list[dict[str, Any]]:
    result = []
    fields = (
        f"H{primary_horizon}_relative_l2",
        "selected_one_step_relative_l2",
        "selected_one_step_boundary_relative_l2",
        "selected_one_step_normal_relative_l2",
        "completion_rate",
        "mean_survival_fraction",
        "elapsed_seconds",
    )
    for arm in ARMS:
        selected = [row for row in records if row["arm"] == arm]
        for field in fields:
            values = [float(row[field]) for row in selected if row[field] is not None]
            result.append(
                {
                    "arm": arm,
                    "metric": field,
                    "n": len(values),
                    "mean": None if not values else float(np.mean(values)),
                    "sample_std": None if not values else _sample_std(values),
                    "minimum": None if not values else min(values),
                    "maximum": None if not values else max(values),
                }
            )
    return result


def _lexicographic_win(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], horizon: int
) -> bool:
    for field in ("completion_rate", "mean_survival_fraction"):
        left, right = float(candidate[field]), float(baseline[field])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15):
            return left > right
    left = candidate[f"H{horizon}_relative_l2"]
    right = baseline[f"H{horizon}_relative_l2"]
    return left is not None and (right is None or float(left) < float(right))


def _paired_rows(
    records: Sequence[Mapping[str, Any]], primary_horizon: int
) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in records}
    result = []
    for seed in SEEDS:
        for candidate_arm, baseline_arm in (("G1", "N0"), ("S1", "N0"), ("S1", "G1")):
            candidate = by_key[(seed, candidate_arm)]
            baseline = by_key[(seed, baseline_arm)]
            candidate_error = candidate[f"H{primary_horizon}_relative_l2"]
            baseline_error = baseline[f"H{primary_horizon}_relative_l2"]
            result.append(
                {
                    "seed": seed,
                    "comparison": f"{candidate_arm}_vs_{baseline_arm}",
                    "candidate": candidate_arm,
                    "baseline": baseline_arm,
                    "candidate_completion": candidate["completion_rate"],
                    "baseline_completion": baseline["completion_rate"],
                    "candidate_survival": candidate["mean_survival_fraction"],
                    "baseline_survival": baseline["mean_survival_fraction"],
                    "candidate_error": candidate_error,
                    "baseline_error": baseline_error,
                    "error_delta": (
                        None
                        if candidate_error is None or baseline_error is None
                        else float(candidate_error) - float(baseline_error)
                    ),
                    "error_ratio": (
                        None
                        if candidate_error is None or baseline_error is None
                        else float(candidate_error) / float(baseline_error)
                    ),
                    "lexicographic_win": _lexicographic_win(
                        candidate, baseline, primary_horizon
                    ),
                }
            )
    return result


def _common_endpoint_rows(
    endpoints: Mapping[tuple[int, str], Mapping[str, float]], primary_horizon: int
) -> list[dict[str, Any]]:
    result = []
    for seed in SEEDS:
        for candidate_arm, baseline_arm in (("G1", "N0"), ("S1", "N0"), ("S1", "G1")):
            candidate = endpoints[(seed, candidate_arm)]
            baseline = endpoints[(seed, baseline_arm)]
            common = sorted(set(candidate) & set(baseline))
            candidate_values = np.asarray(
                [candidate[key] for key in common], dtype=np.float64
            )
            baseline_values = np.asarray(
                [baseline[key] for key in common], dtype=np.float64
            )
            result.append(
                {
                    "seed": seed,
                    "comparison": f"{candidate_arm}_vs_{baseline_arm}",
                    "horizon": primary_horizon,
                    "common_case_count": len(common),
                    "candidate_mean": (
                        None if not common else float(np.mean(candidate_values))
                    ),
                    "baseline_mean": (
                        None if not common else float(np.mean(baseline_values))
                    ),
                    "mean_error_ratio": (
                        None
                        if not common
                        else float(np.mean(candidate_values) / np.mean(baseline_values))
                    ),
                    "case_wins": (
                        0
                        if not common
                        else int(np.sum(candidate_values < baseline_values))
                    ),
                }
            )
    return result


def _bootstrap_rows(
    endpoints: Mapping[tuple[int, str], Mapping[str, float]],
    *,
    primary_horizon: int,
    replicates: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(72_072)
    rows = []
    for common in _common_endpoint_rows(endpoints, primary_horizon):
        seed = int(common["seed"])
        candidate_arm, baseline_arm = str(common["comparison"]).split("_vs_")
        candidate = endpoints[(seed, candidate_arm)]
        baseline = endpoints[(seed, baseline_arm)]
        keys = sorted(set(candidate) & set(baseline))
        if not keys:
            rows.append({**common, "delta_ci95_low": None, "delta_ci95_high": None})
            continue
        deltas = np.asarray([candidate[key] - baseline[key] for key in keys])
        indices = rng.integers(0, len(keys), size=(replicates, len(keys)))
        samples = np.mean(deltas[indices], axis=1)
        rows.append(
            {
                **common,
                "paired_mean_delta": float(np.mean(deltas)),
                "delta_ci95_low": float(np.quantile(samples, 0.025)),
                "delta_ci95_high": float(np.quantile(samples, 0.975)),
                "interval_role": "supporting_case_bootstrap_not_seed_replication",
            }
        )
    return rows


def _optimization_rows(
    records: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    primary_horizon: int,
) -> list[dict[str, Any]]:
    by_run: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in curves:
        by_run[str(row["run"])].append(row)
    records_by_run = {str(row["run"]): row for row in records}
    result = []
    field = f"H{primary_horizon}_relative_l2"
    for run, values in sorted(by_run.items()):
        ordered = sorted(values, key=lambda row: int(row["epoch"]))
        evaluations = [row for row in ordered if row[field] is not None]
        tail = evaluations[-3:]
        slope = None
        if len(tail) >= 2:
            slope = float(
                np.polyfit(
                    [float(row["epoch"]) for row in tail],
                    [float(row[field]) for row in tail],
                    1,
                )[0]
            )
        selected = records_by_run[run]
        selected_error = selected[field]
        terminal_error = evaluations[-1][field]
        result.append(
            {
                "run": run,
                "seed": selected["seed"],
                "arm": selected["arm"],
                "best_epoch": selected["best_epoch"],
                "last_rollout_epoch": evaluations[-1]["epoch"],
                "best_is_last_rollout": selected["best_epoch"]
                == evaluations[-1]["epoch"],
                "best_in_last_three_rollouts": selected["best_epoch"]
                in {row["epoch"] for row in tail},
                "selected_error": selected_error,
                "last_rollout_error": terminal_error,
                "last_to_selected_ratio": (
                    None
                    if selected_error is None or terminal_error is None
                    else float(terminal_error) / float(selected_error)
                ),
                "last_three_rollout_linear_slope_per_epoch": slope,
                "first_five_train_mean": float(
                    np.mean([float(row["train_relative_l2"]) for row in ordered[:5]])
                ),
                "last_five_train_mean": float(
                    np.mean([float(row["train_relative_l2"]) for row in ordered[-5:]])
                ),
            }
        )
    return result


def _mean_metric(
    aggregate: Sequence[Mapping[str, Any]], arm: str, metric: str
) -> float | None:
    selected = [
        row for row in aggregate if row["arm"] == arm and row["metric"] == metric
    ]
    if len(selected) != 1:
        raise ValueError(f"missing aggregate row for {arm}/{metric}")
    value = selected[0]["mean"]
    return None if value is None else float(value)


def _decision(
    *,
    family: str,
    records: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    common_endpoints: Sequence[Mapping[str, Any]],
    primary_horizon: int,
) -> dict[str, Any]:
    metric = f"H{primary_horizon}_relative_l2"
    n0_mean = _mean_metric(aggregate, "N0", metric)
    comparisons = {}
    for candidate, baseline in (("G1", "N0"), ("S1", "N0"), ("S1", "G1")):
        key = f"{candidate}_vs_{baseline}"
        selected = [row for row in paired if row["comparison"] == key]
        wins = sum(bool(row["lexicographic_win"]) for row in selected)
        candidate_mean = _mean_metric(aggregate, candidate, metric)
        baseline_mean = _mean_metric(aggregate, baseline, metric)
        mean_ratio = (
            None
            if candidate_mean is None or baseline_mean is None
            else candidate_mean / baseline_mean
        )
        common = [row for row in common_endpoints if row["comparison"] == key]
        common_ratios = [
            float(row["mean_error_ratio"])
            for row in common
            if row["mean_error_ratio"] is not None
        ]
        comparisons[key] = {
            "seed_wins": wins,
            "mean_error_ratio": mean_ratio,
            "mean_common_population_error_ratio": (
                None if not common_ratios else float(np.mean(common_ratios))
            ),
        }

    all_complete = all(float(row["completion_rate"]) == 1.0 for row in records)
    if family == "dynamic_fv":
        for key, value in comparisons.items():
            threshold = 0.95 if key == "S1_vs_G1" else 0.90
            value["scalar_gate_threshold"] = threshold
            value["provisional_scalar_gate_passed"] = (
                all_complete
                and value["seed_wins"] >= 2
                and value["mean_error_ratio"] is not None
                and value["mean_error_ratio"] <= threshold
            )
    else:
        completion_by_arm = {
            arm: _mean_metric(aggregate, arm, "completion_rate") for arm in ARMS
        }
        for key, value in comparisons.items():
            candidate, baseline = key.split("_vs_")
            completion_delta = (
                completion_by_arm[candidate] - completion_by_arm[baseline]
            )
            tied = abs(completion_delta) <= 0.05
            value["mean_completion_delta"] = completion_delta
            value["completion_tied_within_five_points"] = tied
            value["provisional_scalar_gate_passed"] = value["seed_wins"] >= 2 and (
                completion_delta > 0.0
                or (
                    tied
                    and value["mean_common_population_error_ratio"] is not None
                    and value["mean_common_population_error_ratio"] <= 0.90
                )
            )
    return {
        "primary_horizon": primary_horizon,
        "n0_mean_error": n0_mean,
        "all_runs_complete": all_complete,
        "comparisons": comparisons,
        "front_and_anti_smearing_controls": "pending_dedicated_evaluator",
        "claim_status": (
            "pending_required_controls_even_if_a_provisional_scalar_gate_passes"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = FAMILY_CONTRACTS[args.family]
    run_dirs = _run_directories(args.run_root, args.family)
    records = []
    curves = []
    endpoints = {}
    checkpoint_hashes = {}
    for seed in SEEDS:
        for arm in ARMS:
            record, run_curves, run_endpoints, hashes = _extract_run(
                run_dirs[(seed, arm)],
                family=args.family,
                seed=seed,
                arm=arm,
                skip_checkpoint_hash=args.skip_checkpoint_hash,
            )
            records.append(record)
            curves.extend(run_curves)
            endpoints[(seed, arm)] = run_endpoints
            checkpoint_hashes[record["run"]] = hashes

    invariant_fields = (
        "data_manifest_digest",
        "normalization_digest",
        "source_set_digest",
        "provenance_set_digest",
    )
    invariants = {}
    for field in invariant_fields:
        values = {str(row[field]) for row in records}
        if len(values) != 1:
            raise ValueError(f"{field} drifted across the matrix: {sorted(values)}")
        invariants[field] = next(iter(values))
    streams = {}
    for seed in SEEDS:
        values = {
            str(row["presentation_stream_sha256"])
            for row in records
            if int(row["seed"]) == seed
        }
        if len(values) != 1:
            raise ValueError(f"presentation stream drifted within seed {seed}")
        streams[str(seed)] = next(iter(values))
    invariants["presentation_stream_sha256_by_seed"] = streams
    invariants["checkpoint_hashes"] = checkpoint_hashes
    invariants["checkpoint_hash_verification_required"] = not args.skip_checkpoint_hash

    primary_horizon = int(contract["primary_horizon"])
    aggregate = _aggregate(records, primary_horizon)
    paired = _paired_rows(records, primary_horizon)
    common_endpoints = _common_endpoint_rows(endpoints, primary_horizon)
    bootstrap = _bootstrap_rows(
        endpoints,
        primary_horizon=primary_horizon,
        replicates=args.bootstrap_replicates,
    )
    optimization = _optimization_rows(records, curves, primary_horizon)
    decision = _decision(
        family=args.family,
        records=records,
        aggregate=aggregate,
        paired=paired,
        common_endpoints=common_endpoints,
        primary_horizon=primary_horizon,
    )
    optimization_summary = {
        "best_is_last_rollout_count": sum(
            bool(row["best_is_last_rollout"]) for row in optimization
        ),
        "best_in_last_three_rollouts_count": sum(
            bool(row["best_in_last_three_rollouts"]) for row in optimization
        ),
        "run_count": len(optimization),
        "interpretation": (
            "late selected checkpoints and descending tail slopes indicate that "
            "the fixed matched budget may not establish per-arm optimization "
            "saturation; this does not invalidate matched-budget contrasts"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "records.csv", records)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_csv(args.output_dir / "paired.csv", paired)
    write_csv(args.output_dir / "common_endpoint_pairs.csv", common_endpoints)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap)
    write_csv(args.output_dir / "optimization.csv", optimization)
    write_csv(args.output_dir / "training_curves.csv", curves)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "family": args.family,
        "scope": "open-validation only; no sealed population accessed",
        "args": jsonable_args(args),
        "contract": contract,
        "invariants": invariants,
        "records": records,
        "aggregate": aggregate,
        "paired": paired,
        "common_endpoint_pairs": common_endpoints,
        "paired_bootstrap": bootstrap,
        "optimization": optimization,
        "optimization_summary": optimization_summary,
        "decision": decision,
        "claim_boundary": {
            "training_difference": (
                "matched-training contrasts estimate the effect of access to the "
                "representation under this recipe; they do not prove optimality"
            ),
            "seed_unit": "training seeds are the replicate unit",
            "case_bootstrap": "supporting only; not a substitute for seed replication",
            "missing_controls": (
                "front, region, frequency, completion, and hook-qualified mechanism "
                "evidence remain required before a D072 claim"
            ),
        },
    }
    atomic_write_json(args.output_dir / "analysis_summary.json", summary)
    print(f"wrote {args.output_dir / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
