#!/usr/bin/env python3
"""Analyze the preregistered D094 B1-C5 evaluator repeatability matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    SCHEMA as EVALUATOR_SCHEMA,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)

SCHEMA = "d094_b1_c5_repeatability_analysis_v1"
ARTIFACT_SCHEMA = "d094_b1_c5_repeatability_artifacts_v1"
TRAJECTORY_COUNTS = (128, 256)
PRIMARY_ROLES = ("selected", "terminal")
EXPECTED_EXECUTIONS = ("bf16_1", "bf16_2", "bf16_3", "fp32")
MAX_EFFECT_FLOOR_FRACTION = 0.25
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c4.py",
    "scripts/time_dependent_no/analyze_pcno_bump_b1_c5_repeatability.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-summary",
        type=Path,
        action="append",
        required=True,
        help="Pass exactly three independent-process BF16 summary.json files.",
    )
    parser.add_argument("--fp32-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cell_key(cell: Mapping[str, Any]) -> tuple[int, str]:
    return int(cell["trajectory_count"]), str(cell["checkpoint_role"])


def _cells(summary: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for cell in summary["cells"]:
        key = _cell_key(cell)
        if key in result:
            raise ValueError(f"duplicate evaluator cell: {key}")
        result[key] = cell
    expected = {(count, role) for count in TRAJECTORY_COUNTS for role in PRIMARY_ROLES}
    if not expected <= set(result):
        raise ValueError("evaluator summary lacks a selected or terminal primary cell")
    return result


def _h79(cell: Mapping[str, Any]) -> float:
    value = float(
        cell["outside_selection_rollout"]["mean_endpoint_relative_l2"]["79"]
    )
    if not math.isfinite(value):
        raise ValueError("nonfinite aggregate H79 metric")
    return value


def _all_call(cell: Mapping[str, Any]) -> float:
    value = float(
        cell["outside_selection_rollout"]["mean_full_horizon_relative_l2"]
    )
    if not math.isfinite(value):
        raise ValueError("nonfinite aggregate all-call metric")
    return value


def _trajectory_map(
    cell: Mapping[str, Any], outside_keys: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    result = {
        str(row["trajectory"]): row
        for row in cell["outside_selection_rollout"]["trajectories"]
    }
    if set(result) != set(outside_keys):
        raise ValueError("per-trajectory rollout rows differ from the outside population")
    for row in result.values():
        value = float(row["final_relative_l2"])
        if not math.isfinite(value):
            raise ValueError("nonfinite per-trajectory H79 metric")
    return result


def _binding_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    cells = _cells(summary)
    numerics = summary.get("evaluation_numerics")
    if not isinstance(numerics, Mapping) or "amp" not in numerics:
        raise ValueError("evaluator summary lacks the B1-C5 numerical contract")
    source = summary.get("evaluator_source_snapshot")
    if not isinstance(source, Mapping) or not source.get("source_set_digest"):
        raise ValueError("evaluator summary lacks its source-set digest")
    cell_bindings = [
        {
            "trajectory_count": key[0],
            "checkpoint_role": key[1],
            "optimizer_step": int(cell["optimizer_step"]),
            "checkpoint_sha256": str(cell["checkpoint_sha256"]),
            "config_digest": str(cell["config_digest"]),
            "normalization_digest": str(cell["normalization_digest"]),
        }
        for key, cell in sorted(cells.items())
    ]
    return {
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "split_partition_digest": summary["split_partition_digest"],
        "data_manifest_sha256": summary["data_manifest_sha256"],
        "checkpoint_source_set_digest": summary["checkpoint_source_set_digest"],
        "trajectory_counts": summary["trajectory_counts"],
        "checkpoint_steps": summary["checkpoint_steps"],
        "selection_keys": summary["selection_keys"],
        "outside_selection_keys": summary["outside_selection_keys"],
        "rollout_horizon": summary["rollout_horizon"],
        "rollout_checkpoints": summary["rollout_checkpoints"],
        "rollout_failure_policy": summary["rollout_failure_policy"],
        "shock_quantile": summary["shock_quantile"],
        "boundary_policy_digests": summary["boundary_policy_digests"],
        "runtime_environment": summary["runtime_environment"],
        "evaluator_source_set_digest": source["source_set_digest"],
        "numerical_flags_except_amp": {
            key: value for key, value in numerics.items() if key != "amp"
        },
        "cells": cell_bindings,
    }


def _validate_execution(summary: Mapping[str, Any], expected_amp: str) -> None:
    if summary.get("schema") != EVALUATOR_SCHEMA or summary.get("status") != "complete":
        raise ValueError("input is not a completed B1-C4 outside audit")
    if summary.get("historical_test_population_accessed") is not False:
        raise ValueError("an input summary accessed the historical test population")
    if summary.get("checkpoint_reselection_on_outside_cases") is not False:
        raise ValueError("an input summary reselected on outside cases")
    if summary.get("evaluation_numerics", {}).get("amp") != expected_amp:
        raise ValueError(f"expected amp={expected_amp!r}")
    for key, cell in _cells(summary).items():
        if key[1] not in PRIMARY_ROLES:
            continue
        rollout = cell["outside_selection_rollout"]
        if (
            float(rollout["completion_rate"]) != 1.0
            or int(rollout["hard_failure_count"]) != 0
        ):
            raise ValueError(f"primary cell {key} has an incomplete or failed rollout")


def _direction(values: Sequence[float]) -> tuple[bool, str]:
    if all(value > 0.0 for value in values):
        return True, "positive"
    if all(value < 0.0 for value in values):
        return True, "negative"
    return False, "mixed_or_zero"


def _first_event(row: Mapping[str, Any]) -> tuple[str | None, int | None]:
    event = row.get("first_physical_violation")
    if event is None:
        return None, None
    if not isinstance(event, Mapping):
        raise TypeError("first_physical_violation must be null or an object")
    return str(event["cause"]), int(event["call"])


def analyze_summaries(
    bf16_summaries: Sequence[Mapping[str, Any]], fp32_summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and analyze exactly three BF16 runs and one FP32 control."""

    if len(bf16_summaries) != 3:
        raise ValueError("B1-C5-A requires exactly three BF16 summaries")
    executions = [
        (label, summary, "bf16")
        for label, summary in zip(EXPECTED_EXECUTIONS[:3], bf16_summaries)
    ] + [("fp32", fp32_summary, "none")]
    for _, summary, amp in executions:
        _validate_execution(summary, amp)
    reference_contract = _canonical(_binding_contract(executions[0][1]))
    if any(
        _canonical(_binding_contract(summary)) != reference_contract
        for _, summary, _ in executions[1:]
    ):
        raise ValueError("B1-C5 evaluator inputs or environment differ")

    outside_keys = [str(key) for key in executions[0][1]["outside_selection_keys"]]
    if len(outside_keys) < 2 or len(set(outside_keys)) != len(outside_keys):
        raise ValueError("outside-selection trajectory inventory is invalid")
    cells_by_execution = {
        label: _cells(summary) for label, summary, _ in executions
    }
    trajectories = {
        label: {
            key: _trajectory_map(cell, outside_keys)
            for key, cell in cells.items()
            if key[1] in PRIMARY_ROLES
        }
        for label, cells in cells_by_execution.items()
    }

    cell_rows = []
    cell_summaries: dict[str, Any] = {}
    for count in TRAJECTORY_COUNTS:
        for role in PRIMARY_ROLES:
            key = (count, role)
            h79_values = [
                _h79(cells_by_execution[label][key]) for label in EXPECTED_EXECUTIONS
            ]
            all_call_values = [
                _all_call(cells_by_execution[label][key])
                for label in EXPECTED_EXECUTIONS
            ]
            admissible_counts = [
                int(
                    cells_by_execution[label][key]["outside_selection_rollout"][
                        "physically_admissible_count"
                    ]
                )
                for label in EXPECTED_EXECUTIONS
            ]
            bf16_h79 = h79_values[:3]
            per_case_ranges = np.asarray(
                [
                    max(
                        float(trajectories[label][key][case]["final_relative_l2"])
                        for label in EXPECTED_EXECUTIONS
                    )
                    - min(
                        float(trajectories[label][key][case]["final_relative_l2"])
                        for label in EXPECTED_EXECUTIONS
                    )
                    for case in outside_keys
                ],
                dtype=np.float64,
            )
            row = {
                "trajectory_count": count,
                "checkpoint_role": role,
                "optimizer_step": int(
                    cells_by_execution["bf16_1"][key]["optimizer_step"]
                ),
                "bf16_h79_mean": float(np.mean(bf16_h79)),
                "bf16_h79_range": max(bf16_h79) - min(bf16_h79),
                "fp32_h79": h79_values[3],
                "fp32_minus_bf16_h79_mean": h79_values[3]
                - float(np.mean(bf16_h79)),
                "all_execution_h79_range": max(h79_values) - min(h79_values),
                "all_execution_all_call_range": max(all_call_values)
                - min(all_call_values),
                "per_case_h79_range_median": float(np.median(per_case_ranges)),
                "per_case_h79_range_p95": float(np.quantile(per_case_ranges, 0.95)),
                "per_case_h79_range_max": float(np.max(per_case_ranges)),
                "admissible_counts": admissible_counts,
                "admissible_count_stable": len(set(admissible_counts)) == 1,
                **{
                    f"{label}_h79": value
                    for label, value in zip(EXPECTED_EXECUTIONS, h79_values)
                },
                **{
                    f"{label}_all_call": value
                    for label, value in zip(EXPECTED_EXECUTIONS, all_call_values)
                },
            }
            cell_rows.append(row)
            cell_summaries[f"n{count}_{role}"] = row

    selected_gaps = [
        _h79(cells_by_execution[label][(256, "selected")])
        - _h79(cells_by_execution[label][(128, "selected")])
        for label in EXPECTED_EXECUTIONS
    ]
    largest_selected_cell_range = max(
        float(cell_summaries[f"n{count}_selected"]["all_execution_h79_range"])
        for count in TRAJECTORY_COUNTS
    )
    smallest_count_gap = min(abs(value) for value in selected_gaps)
    floor_fraction = (
        math.inf
        if smallest_count_gap == 0.0
        else largest_selected_cell_range / smallest_count_gap
    )
    selected_ordering = {
        "definition": "positive gap means n=128 has lower selected outside H79",
        "n256_minus_n128_h79_by_execution": dict(
            zip(EXPECTED_EXECUTIONS, selected_gaps)
        ),
        "largest_selected_within_cell_h79_range": largest_selected_cell_range,
        "smallest_cross_count_h79_gap": smallest_count_gap,
        "effect_floor_fraction": floor_fraction,
        "effect_floor_fraction_ceiling": MAX_EFFECT_FLOOR_FRACTION,
        "ordering_agrees": all(value > 0.0 for value in selected_gaps),
        "effect_floor_pass": floor_fraction <= MAX_EFFECT_FLOOR_FRACTION,
    }
    selected_ordering["robust"] = bool(
        selected_ordering["ordering_agrees"]
        and selected_ordering["effect_floor_pass"]
    )

    terminal_directions: dict[str, Any] = {}
    for count in TRAJECTORY_COUNTS:
        differences = [
            _h79(cells_by_execution[label][(count, "terminal")])
            - _h79(cells_by_execution[label][(count, "selected")])
            for label in EXPECTED_EXECUTIONS
        ]
        robust, sign = _direction(differences)
        terminal_directions[str(count)] = {
            "definition": "positive means terminal H79 is worse than selected",
            "terminal_minus_selected_h79_by_execution": dict(
                zip(EXPECTED_EXECUTIONS, differences)
            ),
            "robust": robust,
            "direction": (
                "worse" if sign == "positive" else "better" if sign == "negative" else sign
            ),
        }

    case_rows = []
    for case in outside_keys:
        gaps = [
            float(
                trajectories[label][(256, "selected")][case]["final_relative_l2"]
            )
            - float(
                trajectories[label][(128, "selected")][case]["final_relative_l2"]
            )
            for label in EXPECTED_EXECUTIONS
        ]
        stable, sign = _direction(gaps)
        case_rows.append(
            {
                "trajectory": case,
                "stable_winner": stable,
                "winner": (
                    "n128" if sign == "positive" else "n256" if sign == "negative" else None
                ),
                **{
                    f"{label}_n256_minus_n128_h79": value
                    for label, value in zip(EXPECTED_EXECUTIONS, gaps)
                },
                "n128_h79_range": max(
                    float(
                        trajectories[label][(128, "selected")][case][
                            "final_relative_l2"
                        ]
                    )
                    for label in EXPECTED_EXECUTIONS
                )
                - min(
                    float(
                        trajectories[label][(128, "selected")][case][
                            "final_relative_l2"
                        ]
                    )
                    for label in EXPECTED_EXECUTIONS
                ),
                "n256_h79_range": max(
                    float(
                        trajectories[label][(256, "selected")][case][
                            "final_relative_l2"
                        ]
                    )
                    for label in EXPECTED_EXECUTIONS
                )
                - min(
                    float(
                        trajectories[label][(256, "selected")][case][
                            "final_relative_l2"
                        ]
                    )
                    for label in EXPECTED_EXECUTIONS
                ),
            }
        )

    candidate_cases = outside_keys[1:]
    maximum_cases = {}
    for label in EXPECTED_EXECUTIONS:
        maximum_cases[label] = max(
            candidate_cases,
            key=lambda case: abs(
                float(
                    trajectories[label][(256, "selected")][case]["final_relative_l2"]
                )
                - float(
                    trajectories[label][(128, "selected")][case]["final_relative_l2"]
                )
            ),
        )

    event_rows = []
    for count in TRAJECTORY_COUNTS:
        for role in PRIMARY_ROLES:
            key = (count, role)
            for case in outside_keys:
                events = [
                    _first_event(trajectories[label][key][case])
                    for label in EXPECTED_EXECUTIONS
                ]
                causes = [event[0] for event in events]
                calls = [event[1] for event in events]
                cause_stable = len(set(causes)) == 1
                finite_calls = [call for call in calls if call is not None]
                call_range = (
                    0
                    if not finite_calls
                    else max(finite_calls) - min(finite_calls)
                )
                call_stable = (
                    len(finite_calls) in (0, len(EXPECTED_EXECUTIONS))
                    and call_range <= 1
                )
                event_rows.append(
                    {
                        "trajectory_count": count,
                        "checkpoint_role": role,
                        "trajectory": case,
                        "event_stable": cause_stable and call_stable,
                        "cause_stable": cause_stable,
                        "first_call_range": call_range,
                        **{
                            f"{label}_cause": event[0]
                            for label, event in zip(EXPECTED_EXECUTIONS, events)
                        },
                        **{
                            f"{label}_call": event[1]
                            for label, event in zip(EXPECTED_EXECUTIONS, events)
                        },
                    }
                )

    stable_case_rows = [row for row in case_rows if row["stable_winner"]]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": "b1_c5_a_evaluator_repeatability_and_precision_control",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "execution_count": len(EXPECTED_EXECUTIONS),
        "bf16_execution_count": 3,
        "fp32_execution_count": 1,
        "binding_contract": json.loads(reference_contract),
        "selected_count_ordering": selected_ordering,
        "selected_to_terminal_direction": terminal_directions,
        "per_case_selected_ordering": {
            "case_count": len(case_rows),
            "stable_winner_count": len(stable_case_rows),
            "stable_n128_win_count": sum(
                row["winner"] == "n128" for row in stable_case_rows
            ),
            "stable_n256_win_count": sum(
                row["winner"] == "n256" for row in stable_case_rows
            ),
            "unstable_winner_count": len(case_rows) - len(stable_case_rows),
        },
        "maximum_disagreement_case": {
            "by_execution": maximum_cases,
            "stable": len(set(maximum_cases.values())) == 1,
        },
        "physical_event_repeatability": {
            "event_count": len(event_rows),
            "stable_event_count": sum(row["event_stable"] for row in event_rows),
            "unstable_event_count": sum(not row["event_stable"] for row in event_rows),
        },
        "cell_rows": cell_rows,
        "case_rows": case_rows,
        "event_rows": event_rows,
        "claims_not_supported": [
            "float32 as numerical ground truth",
            "optimization or representation convergence",
            "multi-seed data scaling",
            "causal attribution to data count, schedule, or architecture",
            "historical test performance",
        ],
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if len(args.bf16_summary) != 3:
        raise ValueError("pass exactly three --bf16-summary paths")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C5 analysis output directory must not already exist")
    input_paths = [*args.bf16_summary, args.fp32_summary]
    if len({path.resolve() for path in input_paths}) != len(input_paths):
        raise ValueError("B1-C5 requires four distinct input summary paths")
    for path in input_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"input summary must be a regular file: {path}")
    summaries = [_load_json(path) for path in input_paths]
    analysis = analyze_summaries(summaries[:3], summaries[3])

    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    analysis["analysis_source_snapshot"] = source_snapshot
    analysis["input_summaries"] = [
        {
            "execution": label,
            "path_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in zip(EXPECTED_EXECUTIONS, input_paths)
    ]
    analysis["runtime_environment"] = runtime_environment(torch.device("cpu"))
    atomic_write_json(args.output_dir / "summary.json", analysis)
    write_csv(args.output_dir / "cell_repeatability.csv", analysis["cell_rows"])
    write_csv(
        args.output_dir / "per_case_selected_ordering.csv", analysis["case_rows"]
    )
    write_csv(
        args.output_dir / "physical_event_repeatability.csv", analysis["event_rows"]
    )
    artifact_files = (
        "summary.json",
        "cell_repeatability.csv",
        "per_case_selected_ordering.csv",
        "physical_event_repeatability.csv",
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
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    result = run(argv)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_count_ordering_robust": result[
                    "selected_count_ordering"
                ]["robust"],
                "maximum_disagreement_case_stable": result[
                    "maximum_disagreement_case"
                ]["stable"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
