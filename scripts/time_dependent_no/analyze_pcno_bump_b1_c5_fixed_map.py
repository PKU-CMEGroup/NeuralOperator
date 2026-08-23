#!/usr/bin/env python3
"""Analyze the paired FP32 D094 B1-C5-B fixed-map evaluations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map import (
    ARTIFACT_SCHEMA as EVALUATOR_ARTIFACT_SCHEMA,
    CHECKPOINT_ROLES,
    EXPECTED_EXECUTION_IDS,
    EXPECTED_STEPS,
    INPUT_VIEWS,
    MAX_EFFECT_FLOOR_FRACTION,
    MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
    ROW_SCHEMA,
    SCHEMA as EVALUATOR_SCHEMA,
    TRAJECTORY_COUNTS,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)

SCHEMA = "d094_b1_c5_fixed_map_analysis_v1"
ARTIFACT_SCHEMA = "d094_b1_c5_fixed_map_analysis_artifacts_v1"
PRIMARY_ROLE = "terminal"
PRIMARY_VIEW = "common_selected_path"
PRIMARY_CALL_SIGNS = {20: -1, 79: 1}
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_fixed_map.py",
    "scripts/time_dependent_no/analyze_pcno_bump_b1_c5_fixed_map.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        action="append",
        required=True,
        help="Pass exactly the fp32_1 and fp32_2 summary.json files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verify_evaluator_artifacts(summary_path: Path) -> dict[str, Any]:
    if summary_path.name != "summary.json" or not summary_path.is_file():
        raise ValueError("each evaluator input must be a summary.json file")
    manifest_path = summary_path.parent / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != EVALUATOR_ARTIFACT_SCHEMA:
        raise ValueError("fixed-map evaluator artifact schema changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("fixed-map evaluator artifact manifest lacks files")
    expected = {"summary.json", "map_response.jsonl", "aggregate_summary.csv"}
    if set(files) != expected:
        raise ValueError("fixed-map evaluator artifact inventory changed")
    for name, record in files.items():
        path = summary_path.parent / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"fixed-map evaluator artifact failed rehash: {name}")
    summary = _load_json(summary_path)
    if str(manifest.get("execution_id")) != str(summary.get("execution_id")):
        raise ValueError("fixed-map summary and manifest execution IDs differ")
    return summary


def _load_rows(summary_path: Path, execution_id: str) -> list[dict[str, Any]]:
    rows = []
    with (summary_path.parent / "map_response.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} is not a JSON object")
            if (
                row.get("schema") != ROW_SCHEMA
                or row.get("execution_id") != execution_id
            ):
                raise ValueError(f"row {line_number} identity changed")
            rows.append(row)
    return rows


def _binding_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    source = summary.get("evaluator_source_snapshot")
    if not isinstance(source, Mapping) or not source.get("source_set_digest"):
        raise ValueError("fixed-map evaluator lacks a source-set digest")
    return {
        "precision": summary["precision"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "split_partition_digest": summary["split_partition_digest"],
        "data_manifest_sha256": summary["data_manifest_sha256"],
        "data_manifest_digest": summary["data_manifest_digest"],
        "checkpoint_source_set_digest": summary["checkpoint_source_set_digest"],
        "evaluator_source_set_digest": source["source_set_digest"],
        "checkpoint_bindings": summary["checkpoint_bindings"],
        "validation_keys": summary["validation_keys"],
        "selection_keys": summary["selection_keys"],
        "outside_selection_keys": summary["outside_selection_keys"],
        "boundary_policy_digests": summary["boundary_policy_digests"],
        "rollout_horizon": summary["rollout_horizon"],
        "input_views": summary["input_views"],
        "learned_residual_definition": summary["learned_residual_definition"],
        "functional_drift_definition": summary["functional_drift_definition"],
        "node_measure": summary["node_measure"],
        "primary_directional_rule": summary["primary_directional_rule"],
        "evaluation_numerics": summary["evaluation_numerics"],
        "runtime_environment": summary["runtime_environment"],
    }


def _validate_execution(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    execution_id = str(summary.get("execution_id"))
    if (
        summary.get("schema") != EVALUATOR_SCHEMA
        or summary.get("status") != "complete"
        or execution_id not in EXPECTED_EXECUTION_IDS
        or summary.get("precision") != "float32_no_autocast"
    ):
        raise ValueError("input is not a completed registered FP32 evaluation")
    if (
        summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_performed") is not False
        or summary.get("new_training_performed") is not False
        or summary.get("contract_complete") is not True
        or not all(summary.get("contract_checks", {}).values())
    ):
        raise ValueError("fixed-map evaluator contract did not close")
    outside_keys = [str(key) for key in summary["outside_selection_keys"]]
    if len(outside_keys) != 28 or len(set(outside_keys)) != 28:
        raise ValueError("fixed-map outside population changed")
    if len(rows) != int(summary["row_count"]):
        raise ValueError("fixed-map row count differs from the summary")
    identities = set()
    for row in rows:
        identity = (
            int(row["trajectory_count"]),
            str(row["checkpoint_role"]),
            str(row["trajectory"]),
            int(row["call_index"]),
            str(row["input_view"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate fixed-map row: {identity}")
        identities.add(identity)
        count, role, trajectory, call, view = identity
        if (
            count not in TRAJECTORY_COUNTS
            or role not in CHECKPOINT_ROLES
            or trajectory not in outside_keys
            or call not in range(1, 80)
            or view not in INPUT_VIEWS
            or int(row["optimizer_step"]) != EXPECTED_STEPS[role]
            or not bool(row["candidate_admissibility"]["all_finite"])
        ):
            raise ValueError(f"invalid fixed-map row identity: {identity}")
    expected = {
        (count, role, trajectory, call, view)
        for count in TRAJECTORY_COUNTS
        for role in CHECKPOINT_ROLES
        for trajectory in outside_keys
        for call in range(1, 80)
        for view in INPUT_VIEWS
    }
    if identities != expected:
        raise ValueError("fixed-map row inventory is incomplete")


def paired_floor_record(
    values: Sequence[float], *, expected_sign: int | None = None
) -> dict[str, Any]:
    """Resolve one two-process effect against its registered range floor."""

    if len(values) != len(EXPECTED_EXECUTION_IDS):
        raise ValueError("paired floor requires exactly two executions")
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("paired floor values must be finite")
    value_range = max(finite) - min(finite)
    smaller_magnitude = min(abs(value) for value in finite)
    floor_fraction = (
        math.inf if smaller_magnitude == 0.0 else value_range / smaller_magnitude
    )
    positive = all(value > 0.0 for value in finite)
    negative = all(value < 0.0 for value in finite)
    sign_agrees = positive or negative
    direction = "positive" if positive else "negative" if negative else "mixed_or_zero"
    floor_pass = floor_fraction <= MAX_EFFECT_FLOOR_FRACTION
    result = {
        "by_execution": dict(zip(EXPECTED_EXECUTION_IDS, finite)),
        "range": value_range,
        "smaller_magnitude": smaller_magnitude,
        "effect_floor_fraction": floor_fraction,
        "effect_floor_fraction_ceiling": MAX_EFFECT_FLOOR_FRACTION,
        "floor_pass": floor_pass,
        "sign_agrees": sign_agrees,
        "direction": direction,
        "resolved": floor_pass and sign_agrees,
    }
    if expected_sign is not None:
        if expected_sign not in (-1, 1):
            raise ValueError("expected_sign must be -1 or +1")
        result["expected_direction"] = "negative" if expected_sign < 0 else "positive"
        result["matches_expected_direction"] = bool(
            result["resolved"] and all(value * expected_sign > 0.0 for value in finite)
        )
    return result


def _endpoint(
    summary: Mapping[str, Any],
    *,
    count: int,
    role: str,
    view: str,
    call: int,
) -> Mapping[str, Any]:
    return summary["aggregates"][str(count)][role][view]["endpoints"][str(call)]


def _effect_record(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    count: int,
    role: str,
    view: str,
    call: int,
    expected_sign: int | None,
) -> dict[str, Any]:
    endpoints = {
        execution: _endpoint(summary, count=count, role=role, view=view, call=call)
        for execution, summary in summaries.items()
    }
    effects = [
        float(endpoints[execution]["candidate_minus_selected_relative_l2_mean"])
        for execution in EXPECTED_EXECUTION_IDS
    ]
    drifts = [
        float(endpoints[execution]["pooled_drift_scaled_rms"])
        for execution in EXPECTED_EXECUTION_IDS
    ]
    effect_floor = paired_floor_record(effects, expected_sign=expected_sign)
    drift_floor = paired_floor_record(drifts, expected_sign=1)
    ratios = {
        execution: float(endpoint["candidate_state_relative_l2_mean"])
        / float(endpoint["selected_state_relative_l2_mean"])
        for execution, endpoint in endpoints.items()
    }
    resolved = bool(effect_floor["resolved"] and drift_floor["resolved"])
    return {
        "trajectory_count": count,
        "checkpoint_role": role,
        "optimizer_step": EXPECTED_STEPS[role],
        "input_view": view,
        "call_index": call,
        "effect_definition": (
            "mean candidate state relative L2 minus mean selected-map state "
            "relative L2 on identical inputs and targets"
        ),
        "effect_floor": effect_floor,
        "functional_drift_floor": drift_floor,
        "candidate_over_selected_error_by_execution": ratios,
        "resolved": resolved,
        "matches_expected_direction": (
            None
            if expected_sign is None
            else bool(resolved and effect_floor["matches_expected_direction"])
        ),
    }


def classify_primary_effects(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_keys = {
        (count, call) for count in TRAJECTORY_COUNTS for call in PRIMARY_CALL_SIGNS
    }
    observed = {
        (int(record["trajectory_count"]), int(record["call_index"]))
        for record in records
    }
    if observed != expected_keys:
        raise ValueError("primary fixed-map effect inventory changed")
    all_resolved = all(bool(record["resolved"]) for record in records)
    all_match = all(bool(record["matches_expected_direction"]) for record in records)
    if not all_resolved:
        classification = "numerically_unresolved"
    elif all_match:
        classification = "resolved_common_input_functional_map_crossover"
    else:
        classification = "resolved_common_input_response_does_not_match_full_crossover"
    return {
        "classification": classification,
        "all_four_primary_cells_resolved": all_resolved,
        "all_four_primary_cells_match_early_help_tail_harm": all_match,
        "interpretation_allowed": all_resolved,
    }


def _case_effect_rows(
    rows_by_execution: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    indices = {
        execution: {
            (
                int(row["trajectory_count"]),
                str(row["checkpoint_role"]),
                str(row["trajectory"]),
                int(row["call_index"]),
                str(row["input_view"]),
            ): row
            for row in rows
        }
        for execution, rows in rows_by_execution.items()
    }
    outside_keys = sorted(
        {str(row["trajectory"]) for row in rows_by_execution[EXPECTED_EXECUTION_IDS[0]]}
    )
    output = []
    for count in TRAJECTORY_COUNTS:
        for call, expected_sign in PRIMARY_CALL_SIGNS.items():
            for trajectory in outside_keys:
                identity = (
                    count,
                    PRIMARY_ROLE,
                    trajectory,
                    call,
                    PRIMARY_VIEW,
                )
                effects = [
                    float(
                        indices[execution][identity][
                            "candidate_minus_selected_relative_l2"
                        ]
                    )
                    for execution in EXPECTED_EXECUTION_IDS
                ]
                floor = paired_floor_record(effects, expected_sign=expected_sign)
                output.append(
                    {
                        "trajectory_count": count,
                        "trajectory": trajectory,
                        "call_index": call,
                        "expected_direction": floor["expected_direction"],
                        "direction": floor["direction"],
                        "resolved": floor["resolved"],
                        "matches_expected_direction": floor[
                            "matches_expected_direction"
                        ],
                        "effect_floor_fraction": floor["effect_floor_fraction"],
                        **{
                            f"{execution}_candidate_minus_selected_relative_l2": value
                            for execution, value in zip(EXPECTED_EXECUTION_IDS, effects)
                        },
                    }
                )
    return output


def _effect_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    effect = record["effect_floor"]
    drift = record["functional_drift_floor"]
    return {
        "trajectory_count": record["trajectory_count"],
        "checkpoint_role": record["checkpoint_role"],
        "optimizer_step": record["optimizer_step"],
        "input_view": record["input_view"],
        "call_index": record["call_index"],
        "resolved": record["resolved"],
        "matches_expected_direction": record["matches_expected_direction"],
        "effect_direction": effect["direction"],
        "effect_floor_fraction": effect["effect_floor_fraction"],
        "drift_floor_fraction": drift["effect_floor_fraction"],
        **{
            f"{execution}_effect": effect["by_execution"][execution]
            for execution in EXPECTED_EXECUTION_IDS
        },
        **{
            f"{execution}_drift_scaled_rms": drift["by_execution"][execution]
            for execution in EXPECTED_EXECUTION_IDS
        },
        **{
            f"{execution}_candidate_over_selected_error": record[
                "candidate_over_selected_error_by_execution"
            ][execution]
            for execution in EXPECTED_EXECUTION_IDS
        },
    }


def analyze_evaluations(
    summaries: Mapping[str, Mapping[str, Any]],
    rows_by_execution: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen B1-C5-B numerical and directional rules."""

    if set(summaries) != set(EXPECTED_EXECUTION_IDS) or set(rows_by_execution) != set(
        EXPECTED_EXECUTION_IDS
    ):
        raise ValueError("B1-C5-B requires exactly fp32_1 and fp32_2")
    primary_records = [
        _effect_record(
            summaries,
            count=count,
            role=PRIMARY_ROLE,
            view=PRIMARY_VIEW,
            call=call,
            expected_sign=expected_sign,
        )
        for count in TRAJECTORY_COUNTS
        for call, expected_sign in PRIMARY_CALL_SIGNS.items()
    ]
    primary_classification = classify_primary_effects(primary_records)
    exact_records = [
        _effect_record(
            summaries,
            count=count,
            role=role,
            view="exact_reference",
            call=call,
            expected_sign=None,
        )
        for count in TRAJECTORY_COUNTS
        for role in ("matched_20480", "terminal")
        for call in PRIMARY_CALL_SIGNS
    ]
    common_context_records = [
        _effect_record(
            summaries,
            count=count,
            role="matched_20480",
            view=PRIMARY_VIEW,
            call=call,
            expected_sign=None,
        )
        for count in TRAJECTORY_COUNTS
        for call in PRIMARY_CALL_SIGNS
    ]
    response_amplification = []
    for count in TRAJECTORY_COUNTS:
        for role in ("matched_20480", "terminal"):
            for call in PRIMARY_CALL_SIGNS:
                ratios = {}
                for execution, summary in summaries.items():
                    common = float(
                        _endpoint(
                            summary,
                            count=count,
                            role=role,
                            view=PRIMARY_VIEW,
                            call=call,
                        )["pooled_drift_scaled_rms"]
                    )
                    exact = float(
                        _endpoint(
                            summary,
                            count=count,
                            role=role,
                            view="exact_reference",
                            call=call,
                        )["pooled_drift_scaled_rms"]
                    )
                    ratios[execution] = None if exact <= 1.0e-30 else common / exact
                response_amplification.append(
                    {
                        "trajectory_count": count,
                        "checkpoint_role": role,
                        "call_index": call,
                        "common_over_exact_functional_drift_by_execution": ratios,
                    }
                )

    case_rows = _case_effect_rows(rows_by_execution)
    case_summary = {}
    for count in TRAJECTORY_COUNTS:
        for call in PRIMARY_CALL_SIGNS:
            selected = [
                row
                for row in case_rows
                if int(row["trajectory_count"]) == count
                and int(row["call_index"]) == call
            ]
            case_summary[f"n{count}_call{call}"] = {
                "case_count": len(selected),
                "resolved_case_count": sum(bool(row["resolved"]) for row in selected),
                "expected_direction_case_count": sum(
                    bool(row["matches_expected_direction"]) for row in selected
                ),
                "opposite_resolved_case_count": sum(
                    bool(row["resolved"])
                    and not bool(row["matches_expected_direction"])
                    for row in selected
                ),
            }
    maximum_replay = max(
        float(summary["selected_path_replay_drift_fraction"])
        for summary in summaries.values()
    )
    return {
        "primary_records": primary_records,
        "primary_classification": primary_classification,
        "exact_input_records": exact_records,
        "matched_20480_common_input_context": common_context_records,
        "response_amplification": response_amplification,
        "maximum_selected_path_replay_drift_fraction_across_executions": (
            maximum_replay
        ),
        "selected_path_replay_floor_pass": maximum_replay
        <= MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
        "case_summary": case_summary,
        "case_rows": case_rows,
    }


def _curve_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for execution, summary in summaries.items():
        for count in TRAJECTORY_COUNTS:
            for role in CHECKPOINT_ROLES:
                for view in INPUT_VIEWS:
                    curve = summary["aggregates"][str(count)][role][view]["curve"]
                    for call in range(1, 80):
                        rows.append(
                            {
                                "execution_id": execution,
                                "trajectory_count": count,
                                "checkpoint_role": role,
                                "optimizer_step": EXPECTED_STEPS[role],
                                "input_view": view,
                                "call_index": call,
                                **curve[str(call)],
                            }
                        )
    return rows


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if len(args.evaluation_summary) != len(EXPECTED_EXECUTION_IDS):
        raise ValueError("pass exactly two fixed-map evaluation summaries")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C5-B analysis output directory must not already exist")
    input_records = []
    summaries = {}
    rows_by_execution = {}
    for path in args.evaluation_summary:
        summary = _verify_evaluator_artifacts(path)
        execution_id = str(summary["execution_id"])
        if execution_id in summaries:
            raise ValueError(f"duplicate evaluation execution: {execution_id}")
        rows = _load_rows(path, execution_id)
        _validate_execution(summary, rows)
        summaries[execution_id] = summary
        rows_by_execution[execution_id] = rows
        input_records.append(
            {
                "execution_id": execution_id,
                "summary_sha256": sha256_file(path),
                "artifact_manifest_sha256": sha256_file(
                    path.parent / "artifact_manifest.json"
                ),
                "map_response_sha256": sha256_file(path.parent / "map_response.jsonl"),
            }
        )
    if set(summaries) != set(EXPECTED_EXECUTION_IDS):
        raise ValueError("evaluation IDs must be exactly fp32_1 and fp32_2")
    reference_contract = _canonical(_binding_contract(summaries["fp32_1"]))
    if _canonical(_binding_contract(summaries["fp32_2"])) != reference_contract:
        raise ValueError(
            "paired fixed-map inputs, source, numerics, or environment differ"
        )

    analysis = analyze_evaluations(summaries, rows_by_execution)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    effect_records = [
        *analysis["primary_records"],
        *analysis["matched_20480_common_input_context"],
        *analysis["exact_input_records"],
    ]
    write_csv(
        args.output_dir / "primary_and_context_effects.csv",
        [_effect_csv_row(record) for record in effect_records],
    )
    write_csv(args.output_dir / "case_effects.csv", analysis["case_rows"])
    write_csv(args.output_dir / "map_response_curves.csv", _curve_rows(summaries))

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "input_records": sorted(input_records, key=lambda row: row["execution_id"]),
        "binding_contract": _binding_contract(summaries["fp32_1"]),
        "analysis_source_snapshot": source_snapshot,
        "numerical_floor_contract": {
            "fresh_processes": list(EXPECTED_EXECUTION_IDS),
            "effect_and_drift_range_over_smaller_magnitude_ceiling": (
                MAX_EFFECT_FLOOR_FRACTION
            ),
            "selected_path_same_map_replay_ceiling": (
                MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION
            ),
        },
        "primary_question": (
            "does terminal-versus-selected map response on one common selected "
            "path reproduce the H20-help/H79-harm crossover?"
        ),
        **{key: value for key, value in analysis.items() if key != "case_rows"},
        "historical_test_population_accessed": False,
        "new_training_performed": False,
        "claim_boundary": {
            "verified_if_resolved": (
                "paired functional response of retained maps on exact inputs and "
                "one selected-checkpoint path under this source, population, and "
                "FP32 evaluator"
            ),
            "not_identified": (
                "hidden representations, optimizer cause, data-count cause, "
                "architecture cause, capacity, convergence, conservation, or test "
                "performance"
            ),
        },
        "runtime_environment": runtime_environment(torch.device("cpu")),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    artifact_files = (
        "summary.json",
        "primary_and_context_effects.csv",
        "case_effects.csv",
        "map_response_curves.csv",
    )
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_files
        },
    }
    atomic_write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(argv)
    print(
        json.dumps(
            {
                "status": summary["status"],
                **summary["primary_classification"],
                "selected_path_replay_floor_pass": summary[
                    "selected_path_replay_floor_pass"
                ],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
