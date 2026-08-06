#!/usr/bin/env python3
"""Aggregate D072 frozen-checkpoint region, frequency, and structure evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)

SCHEMA = "pcno_boundary_field_evaluation_analysis_v1"
EVALUATION_SCHEMA = "pcno_boundary_field_evaluation_v1"
TRAINING_SCHEMA = "pcno_boundary_field_training_analysis_v1"
ARMS = ("N0", "G1", "S1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-analysis", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation table: {path}")
    return rows


def _float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _std(values: Sequence[float]) -> float | None:
    return (
        None
        if not values
        else (0.0 if len(values) < 2 else float(np.std(values, ddof=1)))
    )


def _group_metric(
    rows: Sequence[Mapping[str, str]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        record: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        record["row_count"] = len(selected)
        for field in metric_fields:
            values = [
                value
                for row in selected
                if (value := _float(row.get(field))) is not None
            ]
            record[f"{field}_count"] = len(values)
            record[f"{field}_mean"] = _mean(values)
            record[f"{field}_sample_std"] = _std(values)
        result.append(record)
    return result


def _endpoint_rows(
    outcomes: Sequence[Mapping[str, str]], *, primary_horizon: int
) -> list[Mapping[str, str]]:
    return [
        row
        for row in outcomes
        if row["mode"] == "free_rollout"
        and int(row["call"]) == primary_horizon
        and row["region"] == "all"
    ]


def _paired_case_rows(
    endpoints: Sequence[Mapping[str, str]], *, replicates: int
) -> list[dict[str, Any]]:
    by_key = {
        (row["case_id"], row["arm"], row["intervention"]): float(
            row["prediction_relative_l2"]
        )
        for row in endpoints
    }
    cases = sorted({row["case_id"] for row in endpoints})
    comparisons = (
        (("G1", "correct"), ("N0", "correct")),
        (("S1", "correct"), ("N0", "correct")),
        (("S1", "correct"), ("G1", "correct")),
        (("G1", "zero_all"), ("G1", "correct")),
        (("S1", "zero_all"), ("S1", "correct")),
    )
    comparisons += tuple(
        (("S1", intervention), ("S1", "correct"))
        for intervention in sorted(
            {
                row["intervention"]
                for row in endpoints
                if row["arm"] == "S1"
                and row["intervention"].startswith("zero_")
                and row["intervention"] != "zero_all"
            }
        )
    )
    rng = np.random.default_rng(72_073)
    result = []
    for candidate, baseline in comparisons:
        common = [
            case
            for case in cases
            if (case, *candidate) in by_key and (case, *baseline) in by_key
        ]
        candidate_values = np.asarray([by_key[(case, *candidate)] for case in common])
        baseline_values = np.asarray([by_key[(case, *baseline)] for case in common])
        deltas = candidate_values - baseline_values
        indices = rng.integers(0, len(common), size=(replicates, len(common)))
        samples = np.mean(deltas[indices], axis=1)
        result.append(
            {
                "comparison": f"{candidate[0]}_{candidate[1]}_vs_{baseline[0]}_{baseline[1]}",
                "candidate_arm": candidate[0],
                "candidate_intervention": candidate[1],
                "baseline_arm": baseline[0],
                "baseline_intervention": baseline[1],
                "common_case_count": len(common),
                "candidate_mean": float(candidate_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "ratio_of_means": float(
                    candidate_values.mean() / baseline_values.mean()
                ),
                "paired_mean_delta": float(deltas.mean()),
                "case_wins": int(np.sum(candidate_values < baseline_values)),
                "delta_ci95_low": float(np.quantile(samples, 0.025)),
                "delta_ci95_high": float(np.quantile(samples, 0.975)),
                "interval_role": "supporting_case_bootstrap_not_seed_replication",
            }
        )
    return result


def _structure_no_harm(
    structure: Sequence[Mapping[str, str]], *, primary_horizon: int
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in structure
        if row["mode"] == "free_rollout"
        and int(row["call"]) == primary_horizon
        and row["intervention"] == "correct"
    ]
    fields = {
        "front_iou": "higher_is_better",
        "front_symmetric_chamfer": "lower_is_better",
        "shock_thickness_log_error": "lower_is_better",
        "shock_strength_log_error": "lower_is_better",
        "smooth_highpass_energy_equal_node_proxy": "lower_is_better",
    }
    means: dict[tuple[str, str], float | None] = {}
    for arm in ARMS:
        arm_rows = [row for row in selected if row["arm"] == arm]
        for field in fields:
            values = [
                value
                for row in arm_rows
                if (value := _float(row.get(field))) is not None
            ]
            means[(arm, field)] = _mean(values)
    result = []
    for candidate, baseline in (("G1", "N0"), ("S1", "N0"), ("S1", "G1")):
        for field, direction in fields.items():
            candidate_mean = means[(candidate, field)]
            baseline_mean = means[(baseline, field)]
            if candidate_mean is None or baseline_mean is None:
                no_harm_ratio = None
            elif direction == "higher_is_better":
                no_harm_ratio = baseline_mean / max(candidate_mean, 1.0e-30)
            else:
                no_harm_ratio = candidate_mean / max(baseline_mean, 1.0e-30)
            result.append(
                {
                    "comparison": f"{candidate}_vs_{baseline}",
                    "field": field,
                    "direction": direction,
                    "candidate_mean": candidate_mean,
                    "baseline_mean": baseline_mean,
                    "no_harm_ratio_above_one_is_worse": no_harm_ratio,
                    "within_1p05": no_harm_ratio is not None and no_harm_ratio <= 1.05,
                }
            )
    return result


def _decision(
    training: Mapping[str, Any], no_harm: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    training_comparisons = training["decision"]["comparisons"]
    result = {}
    for comparison, training_row in training_comparisons.items():
        front_rows = [row for row in no_harm if row["comparison"] == comparison]
        ratios = [
            float(row["no_harm_ratio_above_one_is_worse"])
            for row in front_rows
            if row["no_harm_ratio_above_one_is_worse"] is not None
        ]
        maximum = None if not ratios else max(ratios)
        result[comparison] = {
            "training_scalar_gate_passed": bool(
                training_row["provisional_scalar_gate_passed"]
            ),
            "maximum_front_no_harm_ratio": maximum,
            "front_no_harm_gate_passed": maximum is not None and maximum <= 1.05,
            "combined_gate_passed": bool(
                training_row["provisional_scalar_gate_passed"]
                and maximum is not None
                and maximum <= 1.05
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    training = _read_json(args.training_analysis)
    evaluation = _read_json(args.evaluation_dir / "summary.json")
    if training.get("schema") != TRAINING_SCHEMA:
        raise ValueError("unexpected training-analysis schema")
    if (
        evaluation.get("schema") != EVALUATION_SCHEMA
        or evaluation.get("status") != "complete"
    ):
        raise ValueError("evaluation is not a complete D072 artifact")
    if training["family"] != evaluation["family"]:
        raise ValueError("training and evaluation families differ")
    if (
        training["invariants"]["data_manifest_digest"]
        != evaluation["data_manifest_digest"]
    ):
        raise ValueError("training and evaluation data manifests differ")

    inputs = {
        "training_analysis": args.training_analysis,
        "evaluation_summary": args.evaluation_dir / "summary.json",
        "outcomes": args.evaluation_dir / "outcomes.csv",
        "completion": args.evaluation_dir / "completion.csv",
        "frequency": args.evaluation_dir / "frequency.csv",
        "structure": args.evaluation_dir / "structure.csv",
        "activations": args.evaluation_dir / "activations.csv",
        "hook_equivalence": args.evaluation_dir / "hook_equivalence.json",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    outcomes = _read_csv(inputs["outcomes"])
    frequency = _read_csv(inputs["frequency"])
    structure = _read_csv(inputs["structure"])
    activations = _read_csv(inputs["activations"])
    primary_horizon = int(evaluation["contract"]["endpoints"][-1])

    endpoints = _endpoint_rows(outcomes, primary_horizon=primary_horizon)
    paired = _paired_case_rows(endpoints, replicates=args.bootstrap_replicates)
    regions = _group_metric(
        [
            row
            for row in outcomes
            if row["mode"] == "free_rollout" and int(row["call"]) == primary_horizon
        ],
        group_fields=("arm", "intervention", "region"),
        metric_fields=(
            "prediction_relative_l2",
            "prediction_error_rms",
            "state_gap_to_arm_correct_rms",
            "increment_gap_to_arm_correct_rms",
            "normalized_volume_weighted_total_error_rms",
        ),
    )
    teacher = _group_metric(
        [
            row
            for row in outcomes
            if row["mode"] == "teacher_forced" and row["region"] == "all"
        ],
        group_fields=("arm", "intervention"),
        metric_fields=(
            "prediction_relative_l2",
            "prediction_error_rms",
            "state_gap_to_arm_correct_rms",
            "increment_gap_to_arm_correct_rms",
        ),
    )
    frequency_summary = _group_metric(
        [
            row
            for row in frequency
            if row["mode"] == "free_rollout" and int(row["call"]) == primary_horizon
        ],
        group_fields=(
            "arm",
            "intervention",
            "metric_kind",
            "wavelength_min",
            "wavelength_max",
        ),
        metric_fields=("relative_spectral_l2", "error_highpass_rms"),
    )
    structure_summary = _group_metric(
        [
            row
            for row in structure
            if row["mode"] == "free_rollout" and int(row["call"]) == primary_horizon
        ],
        group_fields=("arm", "intervention"),
        metric_fields=(
            "front_iou",
            "front_symmetric_chamfer",
            "front_centroid_distance",
            "shock_thickness_log_error",
            "shock_strength_log_error",
            "smooth_highpass_energy_equal_node_proxy",
        ),
    )
    no_harm = _structure_no_harm(structure, primary_horizon=primary_horizon)
    activation_summary = _group_metric(
        [
            row
            for row in activations
            if row["field"]
            not in (
                "analytical_full_post_lift_check",
                "analytical_boundary_only_post_lift_check",
            )
        ],
        group_fields=("mode", "call", "field", "region"),
        metric_fields=("difference_rms", "difference_to_correct_rms"),
    )
    full_checks = [
        row for row in activations if row["field"] == "analytical_full_post_lift_check"
    ]
    boundary_checks = [
        row
        for row in activations
        if row["field"] == "analytical_boundary_only_post_lift_check"
    ]
    evaluated_boundary = [
        row
        for row in boundary_checks
        if _float(row.get("maximum_absolute_error")) is not None
    ]
    lift_checks = {
        "full_check_count": len(full_checks),
        "full_maximum_absolute_error": max(
            float(row["maximum_absolute_error"]) for row in full_checks
        ),
        "boundary_only_row_count": len(boundary_checks),
        "boundary_only_evaluated_count": len(evaluated_boundary),
        "boundary_only_maximum_absolute_error": max(
            float(row["maximum_absolute_error"]) for row in evaluated_boundary
        ),
        "boundary_only_unevaluated_reason": (
            "later free-rollout states differ, so total lift difference is not boundary-only"
        ),
    }

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "paired_cases.csv", paired)
    write_csv(args.output_dir / "regions.csv", regions)
    write_csv(args.output_dir / "teacher_forced.csv", teacher)
    write_csv(args.output_dir / "frequency.csv", frequency_summary)
    write_csv(args.output_dir / "structure.csv", structure_summary)
    write_csv(args.output_dir / "front_no_harm.csv", no_harm)
    write_csv(args.output_dir / "activation_propagation.csv", activation_summary)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "frozen open-validation population only; no sealed population accessed",
        "family": evaluation["family"],
        "seed": evaluation["seed"],
        "primary_horizon": primary_horizon,
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "training_source_set_digest": training["invariants"]["source_set_digest"],
        "evaluation_source_verification": evaluation["source_verification"],
        "hook_equivalence": evaluation["hook_equivalence"],
        "completion_summary": evaluation["aggregate"]["completion_summary"],
        "paired_cases": paired,
        "lift_checks": lift_checks,
        "decision": _decision(training, no_harm),
        "claim_boundary": {
            "training": "matched-training association under one fixed recipe",
            "frozen_intervention": "within-checkpoint causal effect on this open cohort",
            "activation": "diagnostic propagation norm, not causal branch share",
            "case_bootstrap": "supporting only; training seed is the replicate unit",
            "resolution": "native training mesh only; no resolution-transfer claim",
            "bump": "graph quadrature weights are proxy-only",
        },
    }
    atomic_write_json(args.output_dir / "analysis_summary.json", summary)
    print(f"wrote {args.output_dir / 'analysis_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
