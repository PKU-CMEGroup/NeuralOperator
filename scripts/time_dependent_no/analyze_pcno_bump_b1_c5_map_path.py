#!/usr/bin/env python3
"""Analyze the paired FP32 D094 B1-C5-C symmetric map--path evaluations."""

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

from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_fixed_map import (
    paired_floor_record,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map import (
    EXPECTED_EXECUTION_IDS,
    MAX_EFFECT_FLOOR_FRACTION,
    MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
    TRAJECTORY_COUNTS,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_map_path import (
    ARTIFACT_SCHEMA as EVALUATOR_ARTIFACT_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_map_path import (
    MAX_OUTPUT_CLOSURE_RELATIVE,
    MAX_SCALAR_CLOSURE_RELATIVE,
    OUTPUT_LABELS,
    ROW_SCHEMA,
    aggregate_map_path_rows,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_map_path import (
    SCHEMA as EVALUATOR_SCHEMA,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)

SCHEMA = "d094_b1_c5_map_path_analysis_v1"
ARTIFACT_SCHEMA = "d094_b1_c5_map_path_analysis_artifacts_v1"
PRIMARY_CALL = 79
PRIMARY_FIELDS = {
    "map_effect_selected_path": -1,
    "path_effect_selected_map": None,
    "map_path_interaction": None,
    "autonomous_total_effect": 1,
    "zero_interaction_counterfactual_effect": None,
}
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_fixed_map.py",
    "scripts/time_dependent_no/analyze_pcno_bump_b1_c5_fixed_map.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_map_path.py",
    "scripts/time_dependent_no/analyze_pcno_bump_b1_c5_map_path.py",
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
        raise ValueError("map--path evaluator artifact schema changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("map--path evaluator artifact manifest lacks files")
    expected = {"summary.json", "map_path_response.jsonl", "aggregate_summary.csv"}
    if set(files) != expected:
        raise ValueError("map--path evaluator artifact inventory changed")
    for name, record in files.items():
        path = summary_path.parent / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"map--path evaluator artifact failed rehash: {name}")
    summary = _load_json(summary_path)
    if str(manifest.get("execution_id")) != str(summary.get("execution_id")):
        raise ValueError("map--path summary and manifest execution IDs differ")
    return summary


def _load_rows(summary_path: Path, execution_id: str) -> list[dict[str, Any]]:
    rows = []
    with (summary_path.parent / "map_path_response.jsonl").open(
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
        raise ValueError("map--path evaluator lacks a source-set digest")
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
        "map_path_labels": summary["map_path_labels"],
        "recurrence_contract": summary["recurrence_contract"],
        "effect_identity": summary["effect_identity"],
        "node_measure": summary["node_measure"],
        "numerical_floor_contract": summary["numerical_floor_contract"],
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
        raise ValueError("map--path evaluator contract did not close")
    outside_keys = [str(key) for key in summary["outside_selection_keys"]]
    if len(outside_keys) != 28 or len(set(outside_keys)) != 28:
        raise ValueError("map--path outside population changed")
    if len(rows) != int(summary["row_count"]):
        raise ValueError("map--path row count differs from the summary")
    identities = set()
    for row in rows:
        identity = (
            int(row["trajectory_count"]),
            str(row["trajectory"]),
            int(row["call_index"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate map--path row: {identity}")
        identities.add(identity)
        count, trajectory, call = identity
        if (
            count not in TRAJECTORY_COUNTS
            or trajectory not in outside_keys
            or call not in range(1, 80)
            or int(row["selected_optimizer_step"]) != 38_400
            or int(row["terminal_optimizer_step"]) != 40_960
            or any(
                not bool(row["admissibility"][label]["all_finite"])
                for label in OUTPUT_LABELS
            )
            or row.get("replay_all_finite") is not True
            or any(
                not math.isfinite(float(row["replay"][label][metric]))
                for label in OUTPUT_LABELS
                for metric in (
                    "replay_drift_scaled_mse",
                    "owner_residual_scaled_mse",
                )
            )
            or float(row["maximum_scalar_closure_relative"])
            > MAX_SCALAR_CLOSURE_RELATIVE
            or float(row["output_closure_relative"]) > MAX_OUTPUT_CLOSURE_RELATIVE
        ):
            raise ValueError(f"invalid map--path row identity or contract: {identity}")
    expected = {
        (count, trajectory, call)
        for count in TRAJECTORY_COUNTS
        for trajectory in outside_keys
        for call in range(1, 80)
    }
    if identities != expected:
        raise ValueError("map--path row inventory is incomplete")
    recomputed = aggregate_map_path_rows(rows)
    if _canonical(recomputed) != _canonical(summary["aggregates"]):
        raise ValueError("map--path aggregates do not recompute from rows")


def _endpoint(
    summary: Mapping[str, Any], *, count: int, call: int
) -> Mapping[str, Any]:
    return summary["aggregates"][str(count)]["endpoints"][str(call)]


def _effect_record(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    count: int,
    field: str,
    expected_sign: int | None,
) -> dict[str, Any]:
    values = [
        float(
            _endpoint(summaries[execution], count=count, call=PRIMARY_CALL)[
                f"{field}_mean"
            ]
        )
        for execution in EXPECTED_EXECUTION_IDS
    ]
    floor = paired_floor_record(values, expected_sign=expected_sign)
    return {
        "trajectory_count": count,
        "call_index": PRIMARY_CALL,
        "effect": field,
        "floor": floor,
    }


def classify_mechanism(
    records_by_count: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen parent-replay and path-sufficiency decision tree."""

    count_branches: dict[str, str] = {}
    parent_resolved = True
    parent_matches = True
    for count in TRAJECTORY_COUNTS:
        records = records_by_count[count]
        map_floor = records["map_effect_selected_path"]["floor"]
        total_floor = records["autonomous_total_effect"]["floor"]
        if not bool(map_floor["resolved"] and total_floor["resolved"]):
            parent_resolved = False
            count_branches[str(count)] = "numerically_unresolved"
            continue
        if not bool(
            map_floor["matches_expected_direction"]
            and total_floor["matches_expected_direction"]
        ):
            parent_matches = False
            count_branches[str(count)] = "parent_crossover_not_reproduced"
            continue
        counter = records["zero_interaction_counterfactual_effect"]["floor"]
        interaction = records["map_path_interaction"]["floor"]
        if bool(counter["resolved"]) and counter["direction"] == "positive":
            count_branches[str(count)] = "path_displacement_sufficient"
        elif (
            bool(counter["resolved"])
            and counter["direction"] == "negative"
            and bool(interaction["resolved"])
            and interaction["direction"] == "positive"
        ):
            count_branches[str(count)] = "map_path_interaction_required"
        else:
            count_branches[str(count)] = "numerically_unresolved"

    branches = set(count_branches.values())
    if not parent_resolved or "numerically_unresolved" in branches:
        classification = "numerically_unresolved"
    elif not parent_matches or "parent_crossover_not_reproduced" in branches:
        classification = "parent_crossover_not_reproduced"
    elif branches == {"path_displacement_sufficient"}:
        classification = "resolved_path_displacement_sufficient"
    elif branches == {"map_path_interaction_required"}:
        classification = "resolved_map_path_interaction_required"
    else:
        classification = "mixed_count_mechanism"
    return {
        "classification": classification,
        "parent_crossover_numerically_resolved": parent_resolved,
        "parent_crossover_direction_reproduced": parent_resolved and parent_matches,
        "count_branches": count_branches,
        "interpretation_allowed": classification
        not in {
            "numerically_unresolved",
            "parent_crossover_not_reproduced",
        },
    }


def _case_effect_rows(
    rows_by_execution: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    indices = {
        execution: {
            (
                int(row["trajectory_count"]),
                str(row["trajectory"]),
                int(row["call_index"]),
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
        for trajectory in outside_keys:
            identity = (count, trajectory, PRIMARY_CALL)
            floors = {
                field: paired_floor_record(
                    [
                        float(indices[execution][identity][field])
                        for execution in EXPECTED_EXECUTION_IDS
                    ],
                    expected_sign=PRIMARY_FIELDS[field],
                )
                for field in PRIMARY_FIELDS
            }
            map_floor = floors["map_effect_selected_path"]
            total_floor = floors["autonomous_total_effect"]
            parent_sign_flip = bool(
                map_floor["resolved"]
                and map_floor["matches_expected_direction"]
                and total_floor["resolved"]
                and total_floor["matches_expected_direction"]
            )
            counter = floors["zero_interaction_counterfactual_effect"]
            interaction = floors["map_path_interaction"]
            if not parent_sign_flip:
                branch = "not_stable_parent_sign_flip"
            elif bool(counter["resolved"]) and counter["direction"] == "positive":
                branch = "path_displacement_sufficient"
            elif (
                bool(counter["resolved"])
                and counter["direction"] == "negative"
                and bool(interaction["resolved"])
                and interaction["direction"] == "positive"
            ):
                branch = "map_path_interaction_required"
            else:
                branch = "numerically_unresolved"
            output.append(
                {
                    "trajectory_count": count,
                    "trajectory": trajectory,
                    "call_index": PRIMARY_CALL,
                    "stable_parent_sign_flip": parent_sign_flip,
                    "mechanism_branch": branch,
                    **{
                        f"{field}_{execution}": floor["by_execution"][execution]
                        for field, floor in floors.items()
                        for execution in EXPECTED_EXECUTION_IDS
                    },
                    **{
                        f"{field}_floor_fraction": floor["effect_floor_fraction"]
                        for field, floor in floors.items()
                    },
                }
            )
    return output


def analyze_evaluations(
    summaries: Mapping[str, Mapping[str, Any]],
    rows_by_execution: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(summaries) != set(EXPECTED_EXECUTION_IDS) or set(rows_by_execution) != set(
        EXPECTED_EXECUTION_IDS
    ):
        raise ValueError("B1-C5-C requires exactly fp32_1 and fp32_2")
    records_by_count = {
        count: {
            field: _effect_record(
                summaries,
                count=count,
                field=field,
                expected_sign=expected_sign,
            )
            for field, expected_sign in PRIMARY_FIELDS.items()
        }
        for count in TRAJECTORY_COUNTS
    }
    classification = classify_mechanism(records_by_count)
    case_rows = _case_effect_rows(rows_by_execution)
    case_summary = {}
    for count in TRAJECTORY_COUNTS:
        selected = [row for row in case_rows if int(row["trajectory_count"]) == count]
        case_summary[str(count)] = {
            "case_count": len(selected),
            "stable_parent_sign_flip_count": sum(
                bool(row["stable_parent_sign_flip"]) for row in selected
            ),
            "path_displacement_sufficient_count": sum(
                row["mechanism_branch"] == "path_displacement_sufficient"
                for row in selected
            ),
            "map_path_interaction_required_count": sum(
                row["mechanism_branch"] == "map_path_interaction_required"
                for row in selected
            ),
            "numerically_unresolved_sign_flip_count": sum(
                bool(row["stable_parent_sign_flip"])
                and row["mechanism_branch"] == "numerically_unresolved"
                for row in selected
            ),
        }
    maximum_owner_replay = max(
        float(value)
        for summary in summaries.values()
        for value in summary["owner_replay_drift_fractions"].values()
    )
    return {
        "primary_records_by_count": records_by_count,
        "primary_classification": classification,
        "case_summary": case_summary,
        "case_rows": case_rows,
        "maximum_owner_replay_drift_fraction_across_executions": maximum_owner_replay,
        "owner_replay_floor_pass": maximum_owner_replay
        <= MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
    }


def _primary_csv_rows(
    records_by_count: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for count in TRAJECTORY_COUNTS:
        for field in PRIMARY_FIELDS:
            record = records_by_count[count][field]
            floor = record["floor"]
            rows.append(
                {
                    "trajectory_count": count,
                    "call_index": PRIMARY_CALL,
                    "effect": field,
                    "direction": floor["direction"],
                    "resolved": floor["resolved"],
                    "effect_floor_fraction": floor["effect_floor_fraction"],
                    **{
                        f"{execution}_value": floor["by_execution"][execution]
                        for execution in EXPECTED_EXECUTION_IDS
                    },
                }
            )
    return rows


def _curve_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for execution, summary in summaries.items():
        for count in TRAJECTORY_COUNTS:
            curve = summary["aggregates"][str(count)]["curve"]
            for call in range(1, 80):
                rows.append(
                    {
                        "execution_id": execution,
                        "trajectory_count": count,
                        "call_index": call,
                        **curve[str(call)],
                    }
                )
    return rows


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if len(args.evaluation_summary) != len(EXPECTED_EXECUTION_IDS):
        raise ValueError("pass exactly two map--path evaluation summaries")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C5-C analysis output directory must not already exist")
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
                "map_path_response_sha256": sha256_file(
                    path.parent / "map_path_response.jsonl"
                ),
            }
        )
    if set(summaries) != set(EXPECTED_EXECUTION_IDS):
        raise ValueError("evaluation IDs must be exactly fp32_1 and fp32_2")
    reference_contract = _canonical(_binding_contract(summaries["fp32_1"]))
    if _canonical(_binding_contract(summaries["fp32_2"])) != reference_contract:
        raise ValueError(
            "paired map--path inputs, source, numerics, or environment differ"
        )

    analysis = analyze_evaluations(summaries, rows_by_execution)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    write_csv(
        args.output_dir / "primary_effects.csv",
        _primary_csv_rows(analysis["primary_records_by_count"]),
    )
    write_csv(args.output_dir / "case_effects.csv", analysis["case_rows"])
    write_csv(args.output_dir / "map_path_curves.csv", _curve_rows(summaries))
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "input_records": sorted(input_records, key=lambda row: row["execution_id"]),
        "binding_contract": _binding_contract(summaries["fp32_1"]),
        "analysis_source_snapshot": source_snapshot,
        "numerical_floor_contract": {
            "fresh_processes": list(EXPECTED_EXECUTION_IDS),
            "effect_range_over_smaller_magnitude_ceiling": MAX_EFFECT_FLOOR_FRACTION,
            "owner_same_map_replay_ceiling": MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
            "scalar_closure_relative_ceiling": MAX_SCALAR_CLOSURE_RELATIVE,
            "output_closure_relative_ceiling": MAX_OUTPUT_CLOSURE_RELATIVE,
        },
        "primary_question": (
            "is selected-map response to terminal-path displacement sufficient to "
            "reverse the favorable selected-path terminal-map effect at H79, or is "
            "map--path interaction required?"
        ),
        **{key: value for key, value in analysis.items() if key != "case_rows"},
        "historical_test_population_accessed": False,
        "new_training_performed": False,
        "claim_boundary": {
            "verified_if_resolved": (
                "paired scalar-error and output-space decomposition of retained "
                "selected/terminal maps and paths under this one-seed development "
                "population and FP32 evaluator"
            ),
            "not_identified": (
                "optimizer, hidden representation, data-count, architecture, "
                "capacity, gradient, convergence, conservation, or test cause"
            ),
        },
        "runtime_environment": runtime_environment(torch.device("cpu")),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    artifact_files = (
        "summary.json",
        "primary_effects.csv",
        "case_effects.csv",
        "map_path_curves.csv",
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
                "owner_replay_floor_pass": summary["owner_replay_floor_pass"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
