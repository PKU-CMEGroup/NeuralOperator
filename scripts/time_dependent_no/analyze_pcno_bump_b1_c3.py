#!/usr/bin/env python3
"""Analyze the paired-precision D094 B1-C3 exact-exposure audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    sha256_file,
    write_csv,
)

AUDIT_SCHEMA = "d094_b1_c3_exact_exposure_audit_v1"
AUDIT_ARTIFACT_SCHEMA = "d094_b1_c3_exact_exposure_artifacts_v1"
ANALYSIS_SCHEMA = "d094_b1_c3_exact_exposure_analysis_v1"
ANALYSIS_ARTIFACT_SCHEMA = "d094_b1_c3_exact_exposure_analysis_artifacts_v1"
B1_C2_ANALYSIS_SCHEMA = "d094_b1_c2_three_seed_analysis_v1"
B1_C2_TOTAL_STEPS = 20_480

SEEDS = (20_260_812, 20_260_813)
COUNTS = (8, 16, 32, 64, 128, 256)
ARCHITECTURES = ("pcno", "pcfno")
PRECISIONS = ("bf16", "none")

ERROR_METRICS = (
    "fixed_seen_one_step_relative_l2",
    "fixed_validation_one_step_relative_l2",
    "selection_rollout_all_call_relative_l2",
    "selection_rollout_h79_relative_l2",
    "outside_rollout_all_call_relative_l2",
    "outside_rollout_h79_relative_l2",
    "common_rollout_all_call_relative_l2",
    "common_rollout_h79_relative_l2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--b1-c2-analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_825)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _verify_artifacts(root: Path, expected_amp: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("B1-C3 audit root must be a regular directory")
    summary = _load_json(root / "summary.json")
    manifest = _load_json(root / "artifact_manifest.json")
    if (
        summary.get("schema") != AUDIT_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("schema") != AUDIT_ARTIFACT_SCHEMA
        or summary.get("evaluation_numerics", {}).get("amp") != expected_amp
        or summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_on_outside_cases") is not False
        or summary.get("training_or_resume_performed") is not False
        or summary.get("seeds") != list(SEEDS)
        or summary.get("trajectory_counts") != list(COUNTS)
        or summary.get("architectures") != list(ARCHITECTURES)
        or summary.get("checkpoint_role") != "matched_exposure"
        or summary.get("presentations_per_training_trajectory") != 64
        or summary.get("optimizer_step_by_trajectory_count")
        != {str(count): 64 * count for count in COUNTS}
        or summary.get("common_outside_selection_count") != 9
        or manifest.get("historical_test_population_accessed") is not False
    ):
        raise ValueError(f"B1-C3 {expected_amp} summary contract changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("B1-C3 artifact manifest lacks files")
    for name, record in files.items():
        path = root / str(name)
        if (
            not isinstance(record, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"B1-C3 artifact changed: {expected_amp}/{name}")
    expected_names = {
        "summary.json",
        "checkpoint_summary.csv",
        *{
            f"s{seed}_n{count:03d}_{architecture}_matched_exposure.json"
            for seed in SEEDS
            for count in COUNTS
            for architecture in ARCHITECTURES
        },
    }
    if set(files) != expected_names:
        raise ValueError("B1-C3 artifact inventory changed")
    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 24:
        raise ValueError("B1-C3 audit does not contain exactly 24 cells")
    identities = {
        (
            int(cell["seed"]),
            int(cell["trajectory_count"]),
            str(cell["architecture"]),
            str(cell["checkpoint_role"]),
            int(cell["optimizer_step"]),
        )
        for cell in cells
    }
    expected = {
        (seed, count, architecture, "matched_exposure", 64 * count)
        for seed in SEEDS
        for count in COUNTS
        for architecture in ARCHITECTURES
    }
    if identities != expected:
        raise ValueError("B1-C3 cell identities changed")
    for cell in cells:
        for rollout_name in (
            "selection_rollout",
            "outside_selection_rollout",
            "common_outside_selection_rollout",
        ):
            rollout = cell.get(rollout_name)
            if (
                not isinstance(rollout, Mapping)
                or float(rollout.get("completion_rate", -1.0)) != 1.0
                or int(rollout.get("hard_failure_count", -1)) != 0
            ):
                raise ValueError(
                    f"B1-C3 {expected_amp} contains an incomplete rollout"
                )
    return summary


def _cell_identity(cell: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(cell["seed"]),
        int(cell["trajectory_count"]),
        str(cell["architecture"]),
    )


def _validate_precision_pair(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[tuple[int, int, str], Mapping[str, Any]]]:
    cells = {
        amp: {_cell_identity(cell): cell for cell in summary["cells"]}
        for amp, summary in summaries.items()
    }
    if set(cells["bf16"]) != set(cells["none"]):
        raise ValueError("B1-C3 precision arms expose different cells")
    for identity in cells["bf16"]:
        left = cells["bf16"][identity]
        right = cells["none"][identity]
        for name in (
            "checkpoint_sha256",
            "config_digest",
            "normalization_digest",
            "checkpoint_source_set_digest",
            "optimizer_step",
        ):
            if left.get(name) != right.get(name):
                raise ValueError(f"precision arms differ at {identity}: {name}")
    for name in (
        "split_partition_digest",
        "prior_b1_c2_audit_summary_sha256",
        "outside_selection_keys_by_seed",
        "common_outside_selection_keys",
        "fixed_validation_pair_bank_sha256",
        "fixed_seen_pair_bank_sha256_by_count",
        "boundary_policy_digests",
    ):
        if summaries["bf16"].get(name) != summaries["none"].get(name):
            raise ValueError(f"precision audit contracts differ: {name}")
    return cells


def _cell_row(amp: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cell["checkpoint_training_metrics"]
    selection = cell["selection_rollout"]
    outside = cell["outside_selection_rollout"]
    common = cell["common_outside_selection_rollout"]
    structure = cell["outside_selection_h79_structure_means"]
    row = {
        "precision": amp,
        "seed": int(cell["seed"]),
        "trajectory_count": int(cell["trajectory_count"]),
        "architecture": str(cell["architecture"]),
        "optimizer_step": int(cell["optimizer_step"]),
        "checkpoint_sha256": str(cell["checkpoint_sha256"]),
        "online_train_one_step_relative_l2": float(
            metrics["online_train_one_step_relative_l2"]
        ),
        "stored_fixed_validation_one_step_relative_l2": float(
            metrics["stored_fixed_validation_one_step_relative_l2"]
        ),
        "fixed_seen_one_step_relative_l2": float(
            metrics["fixed_seen_train_one_step_relative_l2"]
        ),
        "fixed_validation_one_step_relative_l2": float(
            metrics["fixed_validation_one_step_relative_l2"]
        ),
        "selection_rollout_all_call_relative_l2": float(
            selection["mean_full_horizon_relative_l2"]
        ),
        "selection_rollout_h79_relative_l2": float(
            selection["mean_endpoint_relative_l2"]["79"]
        ),
        "outside_rollout_all_call_relative_l2": float(
            outside["mean_full_horizon_relative_l2"]
        ),
        "outside_rollout_h79_relative_l2": float(
            outside["mean_endpoint_relative_l2"]["79"]
        ),
        "outside_rollout_physical_admissibility_rate": float(
            outside["physical_admissibility_rate"]
        ),
        "common_rollout_all_call_relative_l2": float(
            common["mean_full_horizon_relative_l2"]
        ),
        "common_rollout_h79_relative_l2": float(
            common["mean_endpoint_relative_l2"]["79"]
        ),
    }
    row["recomputed_over_stored_validation"] = (
        row["fixed_validation_one_step_relative_l2"]
        / row["stored_fixed_validation_one_step_relative_l2"]
    )
    row["fixed_validation_over_seen"] = (
        row["fixed_validation_one_step_relative_l2"]
        / row["fixed_seen_one_step_relative_l2"]
    )
    row["outside_h79_over_fixed_validation"] = (
        row["outside_rollout_h79_relative_l2"]
        / row["fixed_validation_one_step_relative_l2"]
    )
    for name, value in structure.items():
        row[f"h79_structure_{name}"] = None if value is None else float(value)
    finite_values = [
        value
        for value in row.values()
        if isinstance(value, float) and value is not None
    ]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("B1-C3 cell contains a non-finite metric")
    if not all(float(row[metric]) > 0.0 for metric in ERROR_METRICS):
        raise ValueError("B1-C3 error metrics must be positive")
    return row


def _group_summary(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["precision"]),
                int(row["trajectory_count"]),
                str(row["architecture"]),
            )
        ].append(row)
    result = []
    for (amp, count, architecture), group in sorted(grouped.items()):
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            if not values:
                continue
            result.append(
                {
                    "precision": amp,
                    "trajectory_count": count,
                    "architecture": architecture,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "minimum": min(values),
                    "maximum": max(values),
                    "geometric_mean": (
                        math.exp(statistics.mean(math.log(value) for value in values))
                        if all(value > 0.0 for value in values)
                        else None
                    ),
                }
            )
    return result


def _architecture_ratios(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    by_identity = {
        (
            str(row["precision"]),
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
        ): row
        for row in rows
    }
    result = []
    for amp in PRECISIONS:
        for seed in SEEDS:
            for count in COUNTS:
                pcno = by_identity[(amp, seed, count, "pcno")]
                pcfno = by_identity[(amp, seed, count, "pcfno")]
                for metric in metrics:
                    left = float(pcno[metric])
                    right = float(pcfno[metric])
                    result.append(
                        {
                            "precision": amp,
                            "seed": seed,
                            "trajectory_count": count,
                            "metric": metric,
                            "pcno": left,
                            "pcfno": right,
                            "pcno_over_pcfno": left / right,
                            "pcno_lower": left < right,
                        }
                    )
    return result


def _ratio_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["trajectory_count"]), str(row["metric"]))].append(row)
    result = []
    for (count, metric), group in sorted(grouped.items()):
        record: dict[str, Any] = {"trajectory_count": count, "metric": metric}
        for amp in PRECISIONS:
            selected = [row for row in group if row["precision"] == amp]
            ratios = [float(row["pcno_over_pcfno"]) for row in selected]
            record[f"{amp}_geometric_mean_ratio"] = math.exp(
                statistics.mean(math.log(value) for value in ratios)
            )
            record[f"{amp}_pcno_lower_seed_count"] = sum(
                bool(row["pcno_lower"]) for row in selected
            )
        record["replicated_in_both_precisions"] = all(
            int(record[f"{amp}_pcno_lower_seed_count"]) == len(SEEDS)
            for amp in PRECISIONS
        )
        result.append(record)
    return result


def _precision_ratios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {
        (
            str(row["precision"]),
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
        ): row
        for row in rows
    }
    result = []
    for seed in SEEDS:
        for count in COUNTS:
            for architecture in ARCHITECTURES:
                bf16 = by_identity[("bf16", seed, count, architecture)]
                fp32 = by_identity[("none", seed, count, architecture)]
                for metric in ERROR_METRICS:
                    left = float(bf16[metric])
                    right = float(fp32[metric])
                    result.append(
                        {
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            "metric": metric,
                            "fp32_over_bf16": right / left,
                            "absolute_relative_difference": abs(right - left) / left,
                        }
                    )
    return result


def _scaling_summary(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for metric in metrics:
            grouped[
                (str(row["precision"]), str(row["architecture"]), metric)
            ][int(row["trajectory_count"])].append(float(row[metric]))
    result = []
    for (amp, architecture, metric), by_count in sorted(grouped.items()):
        means = {count: statistics.mean(by_count[count]) for count in COUNTS}
        result.append(
            {
                "precision": amp,
                "architecture": architecture,
                "metric": metric,
                "best_trajectory_count": min(means, key=means.get),
                "n128_over_n8": means[128] / means[8],
                "n256_over_n128": means[256] / means[128],
                "monotone_nonincreasing": all(
                    means[right] <= means[left]
                    for left, right in zip(COUNTS[:-1], COUNTS[1:])
                ),
            }
        )
    return result


def _read_b1_c2_rows(
    root: Path,
) -> dict[tuple[int, int, str, str], dict[str, str]]:
    analysis = _load_json(root / "analysis.json")
    manifest = _load_json(root / "artifact_manifest.json")
    if analysis.get("schema") != B1_C2_ANALYSIS_SCHEMA:
        raise ValueError("B1-C2 selected analysis schema changed")
    record = manifest.get("files", {}).get("checkpoint_metrics.csv")
    path = root / "checkpoint_metrics.csv"
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != sha256_file(path)
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise ValueError("B1-C2 selected checkpoint table changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_and_terminal = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            row["architecture"],
            row["checkpoint_role"],
        ): row
        for row in rows
        if int(row["seed"]) in SEEDS
        and row["checkpoint_role"] in {"selected", "terminal"}
    }
    if len(selected_and_terminal) != 48:
        raise ValueError(
            "B1-C2 selected/terminal comparison does not contain 48 cells"
        )
    return selected_and_terminal


def _read_b1_c2_training_ratios(root: Path) -> list[dict[str, str]]:
    manifest = _load_json(root / "artifact_manifest.json")
    record = manifest.get("files", {}).get("training_architecture_ratios.csv")
    path = root / "training_architecture_ratios.csv"
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != sha256_file(path)
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise ValueError("B1-C2 training architecture-ratio table changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if int(row["seed"]) in SEEDS
        and int(row["trajectory_count"]) in COUNTS
    ]
    if not selected:
        raise ValueError("B1-C2 training architecture-ratio table is empty")
    return selected


def _transition_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["trajectory_count"]))].append(row)
    result = []
    for seed in SEEDS:
        for count in COUNTS:
            group = sorted(
                grouped[(seed, count)], key=lambda row: int(row["optimizer_step"])
            )
            rollout = [
                row
                for row in group
                if row.get("pcno_over_pcfno_rollout_h79_relative_l2")
                not in {None, ""}
            ]
            validation = [
                row
                for row in group
                if row.get("pcno_over_pcfno_fixed_validation_one_step_relative_l2")
                not in {None, ""}
            ]
            if not rollout or not validation:
                raise ValueError("B1-C2 transition series is incomplete")
            rollout_values = [
                float(row["pcno_over_pcfno_rollout_h79_relative_l2"])
                for row in rollout
            ]
            crossing_index = next(
                (
                    index
                    for index, value in enumerate(rollout_values)
                    if value < 1.0
                    and all(later < 1.0 for later in rollout_values[index:])
                ),
                None,
            )
            crossing_step = (
                None
                if crossing_index is None
                else int(rollout[crossing_index]["optimizer_step"])
            )
            result.append(
                {
                    "seed": seed,
                    "trajectory_count": count,
                    "rollout_observation_count": len(rollout),
                    "first_h79_ratio": rollout_values[0],
                    "last_h79_ratio": rollout_values[-1],
                    "minimum_h79_ratio": min(rollout_values),
                    "maximum_h79_ratio": max(rollout_values),
                    "permanent_pcno_lower_step": crossing_step,
                    "permanent_pcno_lower_scheduler_fraction": (
                        None
                        if crossing_step is None
                        else crossing_step / B1_C2_TOTAL_STEPS
                    ),
                    "permanent_pcno_lower_presentations_per_trajectory": (
                        None if crossing_step is None else crossing_step / count
                    ),
                    "first_validation_ratio": float(
                        validation[0][
                            "pcno_over_pcfno_fixed_validation_one_step_relative_l2"
                        ]
                    ),
                    "last_validation_ratio": float(
                        validation[-1][
                            "pcno_over_pcfno_fixed_validation_one_step_relative_l2"
                        ]
                    ),
                }
            )
    return result


def _transition_coordinate_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coordinates = (
        "permanent_pcno_lower_step",
        "permanent_pcno_lower_scheduler_fraction",
        "permanent_pcno_lower_presentations_per_trajectory",
    )
    result = []
    for coordinate in coordinates:
        values = [
            float(row[coordinate])
            for row in rows
            if row.get(coordinate) is not None
        ]
        if not values or not all(value > 0.0 for value in values):
            raise ValueError("transition coordinates must be positive and nonempty")
        logs = [math.log(value) for value in values]
        result.append(
            {
                "coordinate": coordinate,
                "n": len(values),
                "geometric_mean": math.exp(statistics.mean(logs)),
                "log_sample_std": (
                    statistics.stdev(logs) if len(logs) > 1 else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
                "maximum_over_minimum": max(values) / min(values),
            }
        )
    return result


def _exact_vs_b1_c2(
    rows: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, int, str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    mapping = {
        "fixed_seen_one_step_relative_l2": "fixed_seen_train_one_step_relative_l2",
        "fixed_validation_one_step_relative_l2": (
            "fixed_validation_one_step_relative_l2"
        ),
        "selection_rollout_h79_relative_l2": "internal_rollout_h79_relative_l2",
        "outside_rollout_all_call_relative_l2": (
            "seed_specific_outside_rollout_all_call_mean_relative_l2"
        ),
        "outside_rollout_h79_relative_l2": (
            "seed_specific_outside_rollout_h79_relative_l2"
        ),
    }
    result = []
    for row in rows:
        if row["precision"] != "bf16":
            continue
        key = (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
        )
        for reference_role in ("selected", "terminal"):
            reference = references[(*key, reference_role)]
            for exact_metric, reference_metric in mapping.items():
                exact = float(row[exact_metric])
                reference_value = float(reference[reference_metric])
                result.append(
                    {
                        "seed": key[0],
                        "trajectory_count": key[1],
                        "architecture": key[2],
                        "metric": exact_metric,
                        "exact_exposure_step": int(row["optimizer_step"]),
                        "reference_role": reference_role,
                        "reference_step": int(reference["optimizer_step"]),
                        "exact_exposure": exact,
                        "reference": reference_value,
                        "exact_over_reference": exact / reference_value,
                    }
                )
    return result


def _bootstrap_ratio(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal nonempty samples")
    generator = random.Random(seed)
    ratios = []
    for _ in range(draws):
        indices = [generator.randrange(len(left)) for _ in left]
        ratios.append(
            statistics.mean(left[index] for index in indices)
            / statistics.mean(right[index] for index in indices)
        )
    ratios.sort()
    return (
        ratios[int(0.025 * (draws - 1))],
        ratios[int(0.975 * (draws - 1))],
    )


def _paired_case_rows(
    cells: Mapping[str, Mapping[tuple[int, int, str], Mapping[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    result = []
    for amp_index, amp in enumerate(PRECISIONS):
        for seed_index, init_seed in enumerate(SEEDS):
            for count_index, count in enumerate(COUNTS):
                pcno = cells[amp][(init_seed, count, "pcno")][
                    "outside_selection_rollout"
                ]["trajectories"]
                pcfno = cells[amp][(init_seed, count, "pcfno")][
                    "outside_selection_rollout"
                ]["trajectories"]
                pcno_by_key = {str(row["trajectory"]): row for row in pcno}
                pcfno_by_key = {str(row["trajectory"]): row for row in pcfno}
                if list(pcno_by_key) != list(pcfno_by_key) or len(pcno_by_key) != 28:
                    raise ValueError("paired outside-case population changed")
                left = [
                    float(row["endpoint_relative_l2"]["79"])
                    for row in pcno_by_key.values()
                ]
                right = [
                    float(pcfno_by_key[key]["endpoint_relative_l2"]["79"])
                    for key in pcno_by_key
                ]
                lower, upper = _bootstrap_ratio(
                    left,
                    right,
                    draws=draws,
                    seed=seed + 10_000 * amp_index + 100 * seed_index + count_index,
                )
                result.append(
                    {
                        "precision": amp,
                        "seed": init_seed,
                        "trajectory_count": count,
                        "case_count": len(left),
                        "pcno_lower_case_count": sum(
                            lvalue < rvalue for lvalue, rvalue in zip(left, right)
                        ),
                        "pcno_over_pcfno_h79": statistics.mean(left)
                        / statistics.mean(right),
                        "bootstrap_95_lower": lower,
                        "bootstrap_95_upper": upper,
                        "bootstrap_draws": draws,
                    }
                )
    return result


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C3 analysis output directory must not already exist")
    summaries = {
        "bf16": _verify_artifacts(args.bf16_root, "bf16"),
        "none": _verify_artifacts(args.fp32_root, "none"),
    }
    cells = _validate_precision_pair(summaries)
    rows = [
        _cell_row(amp, cells[amp][(seed, count, architecture)])
        for amp in PRECISIONS
        for seed in SEEDS
        for count in COUNTS
        for architecture in ARCHITECTURES
    ]
    structure_metrics = sorted(
        {name for row in rows for name in row if name.startswith("h79_structure_")}
    )
    summary_metrics = (
        "online_train_one_step_relative_l2",
        "stored_fixed_validation_one_step_relative_l2",
        *ERROR_METRICS,
        "outside_rollout_physical_admissibility_rate",
        "recomputed_over_stored_validation",
        "fixed_validation_over_seen",
        "outside_h79_over_fixed_validation",
        *structure_metrics,
    )
    ratios = _architecture_ratios(rows, ERROR_METRICS)
    b1_c2_rows = _read_b1_c2_rows(args.b1_c2_analysis_root)
    b1_c2_training_ratios = _read_b1_c2_training_ratios(
        args.b1_c2_analysis_root
    )
    transition_rows = _transition_rows(b1_c2_training_ratios)
    outputs: dict[str, list[dict[str, Any]]] = {
        "exact_exposure_cells.csv": rows,
        "cross_seed_summary.csv": _group_summary(rows, summary_metrics),
        "architecture_ratios.csv": ratios,
        "architecture_ratio_summary.csv": _ratio_summary(ratios),
        "precision_ratios.csv": _precision_ratios(rows),
        "scaling_summary.csv": _scaling_summary(rows, ERROR_METRICS),
        "exact_vs_b1_c2.csv": _exact_vs_b1_c2(rows, b1_c2_rows),
        "b1_c2_transition_alignment.csv": transition_rows,
        "b1_c2_transition_coordinate_summary.csv": (
            _transition_coordinate_summary(transition_rows)
        ),
        "paired_case_bootstrap.csv": _paired_case_rows(
            cells,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        ),
    }
    args.output_dir.mkdir(parents=True)
    for name, table in outputs.items():
        write_csv(args.output_dir / name, table)
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "cell_count_per_precision": 24,
        "precision_arms": list(PRECISIONS),
        "same_checkpoint_hashes_across_precisions": True,
        "presentations_per_training_trajectory": 64,
        "optimizer_step_by_trajectory_count": {
            str(count): 64 * count for count in COUNTS
        },
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "input_summary_sha256": {
            "bf16": sha256_file(args.bf16_root / "summary.json"),
            "none": sha256_file(args.fp32_root / "summary.json"),
        },
        "input_artifact_manifest_sha256": {
            "bf16": sha256_file(args.bf16_root / "artifact_manifest.json"),
            "none": sha256_file(args.fp32_root / "artifact_manifest.json"),
        },
        "b1_c2_analysis_manifest_sha256": sha256_file(
            args.b1_c2_analysis_root / "artifact_manifest.json"
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "claims_not_supported": [
            "three-seed fixed-exposure uncertainty",
            "fixed optimizer updates, compute, or learning-rate phase",
            "causal attribution to data, optimizer, capacity, or gradient",
            "a paper-faithful FFNO comparison",
            "independent test performance",
        ],
    }
    atomic_write_json(args.output_dir / "analysis.json", analysis)
    artifact_names = ["analysis.json", *outputs]
    artifact_manifest = {
        "schema": ANALYSIS_ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_names
        },
    }
    atomic_write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    analysis = run(argv)
    print(json.dumps(analysis, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
