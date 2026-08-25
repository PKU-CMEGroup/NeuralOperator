#!/usr/bin/env python3
"""Combine the D094 B1-C3 exact-exposure matrix with its seed-0 replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import analyze_pcno_bump_b1_c3 as prior
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    sha256_file,
    write_csv,
)

ANALYSIS_SCHEMA = "d094_b1_c3_three_seed_exact_exposure_analysis_v1"
ARTIFACT_SCHEMA = "d094_b1_c3_three_seed_exact_exposure_artifacts_v1"
R1_AUDIT_SCHEMA = "d094_b1_c3_r1_exact_exposure_audit_v1"
R1_ARTIFACT_SCHEMA = "d094_b1_c3_r1_exact_exposure_artifacts_v1"
R1_RETRIEVAL_SCHEMA = "d094_b1_c3_r1_compact_retrieval_v1"

SEED0 = 20_260_718
SEEDS = (SEED0, *prior.SEEDS)
COUNTS = prior.COUNTS
ARCHITECTURES = prior.ARCHITECTURES
PRECISIONS = prior.PRECISIONS

REPLAY_METRICS = (
    "learning_rate",
    "online_train_one_step_relative_l2",
    "fixed_seen_train_one_step_relative_l2",
    "fixed_validation_one_step_relative_l2",
    "rollout_h79_relative_l2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-bf16-root", type=Path, required=True)
    parser.add_argument("--prior-fp32-root", type=Path, required=True)
    parser.add_argument("--r1-bf16-root", type=Path, required=True)
    parser.add_argument("--r1-fp32-root", type=Path, required=True)
    parser.add_argument("--b1-c2-analysis-root", type=Path, required=True)
    parser.add_argument("--r1-packet-root", type=Path, required=True)
    parser.add_argument("--historical-source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_825)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("artifact manifest lacks a file inventory")
    for name, record in files.items():
        path = root / str(name)
        if (
            not isinstance(record, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"manifest-bound artifact changed: {name}")


def _verify_r1_audit(root: Path, expected_amp: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("B1-C3-R1 audit root must be a regular directory")
    summary = _load_json(root / "summary.json")
    manifest = _load_json(root / "artifact_manifest.json")
    if (
        summary.get("schema") != R1_AUDIT_SCHEMA
        or summary.get("status") != "complete"
        or manifest.get("schema") != R1_ARTIFACT_SCHEMA
        or summary.get("evaluation_numerics", {}).get("amp") != expected_amp
        or summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_on_outside_cases") is not False
        or summary.get("training_or_resume_performed") is not False
        or summary.get("seeds") != [SEED0]
        or summary.get("trajectory_counts") != list(COUNTS)
        or summary.get("architectures") != list(ARCHITECTURES)
        or summary.get("checkpoint_role") != "matched_exposure"
        or summary.get("presentations_per_training_trajectory") != 64
        or summary.get("optimizer_step_by_trajectory_count")
        != {str(count): 64 * count for count in COUNTS}
        or summary.get("common_outside_selection_count") != 9
        or manifest.get("historical_test_population_accessed") is not False
    ):
        raise ValueError(f"B1-C3-R1 {expected_amp} summary contract changed")
    _verify_manifest_files(root, manifest)
    expected_names = {
        "summary.json",
        "checkpoint_summary.csv",
        *{
            f"s{SEED0}_n{count:03d}_{architecture}_matched_exposure.json"
            for count in COUNTS
            for architecture in ARCHITECTURES
        },
    }
    if set(manifest["files"]) != expected_names:
        raise ValueError("B1-C3-R1 artifact inventory changed")
    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise ValueError("B1-C3-R1 audit must contain exactly 12 cells")
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
        (SEED0, count, architecture, "matched_exposure", 64 * count)
        for count in COUNTS
        for architecture in ARCHITECTURES
    }
    if identities != expected:
        raise ValueError("B1-C3-R1 cell identities changed")
    for cell in cells:
        for name in (
            "selection_rollout",
            "outside_selection_rollout",
            "common_outside_selection_rollout",
        ):
            rollout = cell.get(name)
            if (
                not isinstance(rollout, Mapping)
                or float(rollout.get("completion_rate", -1.0)) != 1.0
                or int(rollout.get("hard_failure_count", -1)) != 0
            ):
                raise ValueError(f"B1-C3-R1 {expected_amp} rollout is incomplete")
    return summary


def _verify_r1_packet(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "retrieval_manifest.json")
    if (
        manifest.get("schema") != R1_RETRIEVAL_SCHEMA
        or manifest.get("checkpoints_included") is not False
        or manifest.get("historical_test_population_accessed") is not False
    ):
        raise ValueError("B1-C3-R1 compact retrieval contract changed")
    _verify_manifest_files(root, manifest)
    return manifest


def _combined_precision_cells(
    summaries: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, dict[tuple[int, int, str], Mapping[str, Any]]]:
    combined: dict[str, dict[str, Any]] = {}
    for amp, (older, replay) in summaries.items():
        payload = dict(older)
        payload["cells"] = [*older["cells"], *replay["cells"]]
        payload["seeds"] = list(SEEDS)
        combined[amp] = payload
    cells = prior._validate_precision_pair(combined)
    expected = {
        (seed, count, architecture)
        for seed in SEEDS
        for count in COUNTS
        for architecture in ARCHITECTURES
    }
    if set(cells["bf16"]) != expected:
        raise ValueError("combined exact-exposure matrix is incomplete")
    return cells


def _architecture_ratios(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    indexed = {
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
                pcno = indexed[(amp, seed, count, "pcno")]
                pcfno = indexed[(amp, seed, count, "pcfno")]
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
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["precision"]),
                int(row["trajectory_count"]),
                str(row["metric"]),
            )
        ].append(row)
    result = []
    for (amp, count, metric), group in sorted(grouped.items()):
        ratios = [float(row["pcno_over_pcfno"]) for row in group]
        log_ratios = [math.log(value) for value in ratios]
        lower_count = sum(bool(row["pcno_lower"]) for row in group)
        result.append(
            {
                "precision": amp,
                "trajectory_count": count,
                "metric": metric,
                "seed_count": len(group),
                "geometric_mean_ratio": math.exp(statistics.mean(log_ratios)),
                "log_ratio_sample_std": statistics.stdev(log_ratios),
                "minimum_ratio": min(ratios),
                "maximum_ratio": max(ratios),
                "pcno_lower_seed_count": lower_count,
                "direction_consistent": lower_count in {0, len(group)},
            }
        )
    return result


def _precision_ratios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
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
                bf16 = indexed[("bf16", seed, count, architecture)]
                fp32 = indexed[("none", seed, count, architecture)]
                for metric in prior.ERROR_METRICS:
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
                trajectories = {
                    architecture: cells[amp][(init_seed, count, architecture)][
                        "outside_selection_rollout"
                    ]["trajectories"]
                    for architecture in ARCHITECTURES
                }
                pcno = {str(row["trajectory"]): row for row in trajectories["pcno"]}
                pcfno = {
                    str(row["trajectory"]): row for row in trajectories["pcfno"]
                }
                if list(pcno) != list(pcfno) or len(pcno) != 28:
                    raise ValueError("paired outside-case population changed")
                left = [
                    float(row["endpoint_relative_l2"]["79"])
                    for row in pcno.values()
                ]
                right = [
                    float(pcfno[key]["endpoint_relative_l2"]["79"])
                    for key in pcno
                ]
                lower, upper = prior._bootstrap_ratio(
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


def _manifest_bound_csv(root: Path, name: str) -> list[dict[str, str]]:
    manifest = _load_json(root / "artifact_manifest.json")
    record = manifest.get("files", {}).get(name)
    path = root / name
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != sha256_file(path)
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise ValueError(f"B1-C2 analysis input changed: {name}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def _training_surface_rows(
    history_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    selected = []
    for row in history_rows:
        seed = int(row["seed"])
        count = int(row["trajectory_count"])
        architecture = str(row["architecture"])
        if (
            seed not in SEEDS
            or count not in COUNTS
            or architecture not in ARCHITECTURES
        ):
            continue
        step = int(row["optimizer_step"])
        selected.append(
            {
                "seed": seed,
                "trajectory_count": count,
                "architecture": architecture,
                "epoch": int(row["epoch"]),
                "optimizer_step": step,
                "scheduler_fraction": step / prior.B1_C2_TOTAL_STEPS,
                "presentations_per_trajectory": step / count,
                "online_train_one_step_relative_l2": _optional_float(
                    row.get("online_train_one_step_relative_l2")
                ),
                "fixed_seen_train_one_step_relative_l2": _optional_float(
                    row.get("fixed_seen_train_one_step_relative_l2")
                ),
                "fixed_validation_one_step_relative_l2": _optional_float(
                    row.get("fixed_validation_one_step_relative_l2")
                ),
                "rollout_h79_relative_l2": _optional_float(
                    row.get("rollout_h79_relative_l2")
                ),
                "fixed_validation_over_seen_ratio": _optional_float(
                    row.get("fixed_validation_over_seen_ratio")
                ),
                "is_exact_exposure_sentinel": step == 64 * count,
            }
        )
    expected = len(SEEDS) * len(COUNTS) * len(ARCHITECTURES) * 80
    if len(selected) != expected:
        raise ValueError("three-seed training surface is incomplete")
    return selected


def _r1_run_directory(root: Path, count: int, architecture: str) -> Path:
    matches = list(
        (root / "training" / "results").glob(
            f"b1_c3_r1_s{SEED0}_n{count}_{architecture}_stretched_20480_*"
        )
    )
    if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_dir():
        raise ValueError(f"expected one B1-C3-R1 training run for n={count}/{architecture}")
    return matches[0]


def _r1_metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "learning_rate":
        value = row.get("learning_rate")
    elif metric == "online_train_one_step_relative_l2":
        value = row.get("train", {}).get("relative_l2")
    elif metric == "fixed_validation_one_step_relative_l2":
        value = row.get("validation", {}).get("relative_l2")
    elif metric == "fixed_seen_train_one_step_relative_l2":
        comparable_seen = row.get("train", {}).get("comparable_seen")
        value = (
            comparable_seen.get("relative_l2")
            if isinstance(comparable_seen, Mapping)
            else None
        )
    elif metric == "rollout_h79_relative_l2":
        rollout = row.get("rollout")
        endpoint = (
            rollout.get("mean_endpoint_relative_l2")
            if isinstance(rollout, Mapping)
            else None
        )
        value = endpoint.get("79") if isinstance(endpoint, Mapping) else None
    else:
        raise ValueError(f"unsupported replay metric: {metric}")
    return None if value is None else float(value)


def _replay_comparison(
    history_rows: Sequence[Mapping[str, str]], r1_packet_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical = {
        (
            int(row["trajectory_count"]),
            str(row["architecture"]),
            int(row["optimizer_step"]),
        ): row
        for row in history_rows
        if int(row["seed"]) == SEED0
        and int(row["trajectory_count"]) in COUNTS
        and row["architecture"] in ARCHITECTURES
    }
    if len(historical) != len(COUNTS) * len(ARCHITECTURES) * 80:
        raise ValueError("historical seed-0 training history is incomplete")
    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for count in COUNTS:
        for architecture in ARCHITECTURES:
            path = _r1_run_directory(r1_packet_root, count, architecture) / "metrics.jsonl"
            replay_rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            if len(replay_rows) != 80:
                raise ValueError("B1-C3-R1 training history must contain 80 epochs")
            for metric in REPLAY_METRICS:
                comparisons = []
                for row in replay_rows:
                    step = int(row["train"]["completed_optimizer_steps"])
                    old_text = historical[(count, architecture, step)].get(metric, "")
                    new_value = _r1_metric_value(row, metric)
                    if old_text in {None, ""} or new_value is None:
                        if old_text not in {None, ""} or new_value is not None:
                            raise ValueError("replay and historical observation grids differ")
                        continue
                    old_value = float(old_text)
                    relative = abs(new_value - old_value) / max(abs(old_value), 1.0e-30)
                    comparisons.append((step, old_value, new_value, relative))
                    point_rows.append(
                        {
                            "trajectory_count": count,
                            "architecture": architecture,
                            "metric": metric,
                            "optimizer_step": step,
                            "historical": old_value,
                            "replay": new_value,
                            "absolute_relative_difference": relative,
                            "exact_match": old_value == new_value,
                        }
                    )
                if not comparisons:
                    raise ValueError("replay comparison has no common observations")
                relative_differences = [row[3] for row in comparisons]
                divergent = [row[0] for row in comparisons if row[1] != row[2]]
                summary_rows.append(
                    {
                        "trajectory_count": count,
                        "architecture": architecture,
                        "metric": metric,
                        "observation_count": len(comparisons),
                        "exact_match_count": sum(
                            left == right for _, left, right, _ in comparisons
                        ),
                        "all_exact": not divergent,
                        "first_divergent_optimizer_step": (
                            None if not divergent else min(divergent)
                        ),
                        "median_absolute_relative_difference": statistics.median(
                            relative_differences
                        ),
                        "maximum_absolute_relative_difference": max(
                            relative_differences
                        ),
                    }
                )
    return summary_rows, point_rows


def _source_comparison(
    historical_summary: Mapping[str, Any], r1_packet_root: Path
) -> list[dict[str, Any]]:
    old = historical_summary.get("source_snapshot")
    if not isinstance(old, Mapping) or not isinstance(old.get("files"), Mapping):
        raise TypeError("historical seed-0 summary lacks a source snapshot")
    snapshots = []
    for count in COUNTS:
        for architecture in ARCHITECTURES:
            summary = _load_json(
                _r1_run_directory(r1_packet_root, count, architecture) / "summary.json"
            )
            snapshot = summary.get("source_snapshot")
            if not isinstance(snapshot, Mapping):
                raise TypeError("B1-C3-R1 summary lacks a source snapshot")
            snapshots.append(snapshot)
    digests = {str(snapshot.get("source_set_digest")) for snapshot in snapshots}
    if len(digests) != 1:
        raise ValueError("B1-C3-R1 training cells used different source snapshots")
    new = snapshots[0]
    old_files = old["files"]
    new_files = new.get("files")
    if not isinstance(new_files, Mapping):
        raise TypeError("B1-C3-R1 source snapshot lacks files")
    result = []
    for name in sorted(set(old_files) | set(new_files)):
        old_record = old_files.get(name)
        new_record = new_files.get(name)
        old_hash = old_record.get("sha256") if isinstance(old_record, Mapping) else None
        new_hash = new_record.get("sha256") if isinstance(new_record, Mapping) else None
        result.append(
            {
                "repository_relative_file": name,
                "historical_sha256": old_hash,
                "replay_sha256": new_hash,
                "status": (
                    "same"
                    if old_hash == new_hash
                    else "historical_only"
                    if new_hash is None
                    else "replay_only"
                    if old_hash is None
                    else "changed"
                ),
            }
        )
    return result


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("three-seed analysis output directory must not already exist")

    summaries = {
        "bf16": (
            prior._verify_artifacts(args.prior_bf16_root, "bf16"),
            _verify_r1_audit(args.r1_bf16_root, "bf16"),
        ),
        "none": (
            prior._verify_artifacts(args.prior_fp32_root, "none"),
            _verify_r1_audit(args.r1_fp32_root, "none"),
        ),
    }
    _verify_r1_packet(args.r1_packet_root)
    cells = _combined_precision_cells(summaries)
    rows = [
        prior._cell_row(amp, cells[amp][(seed, count, architecture)])
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
        *prior.ERROR_METRICS,
        "outside_rollout_physical_admissibility_rate",
        "recomputed_over_stored_validation",
        "fixed_validation_over_seen",
        "outside_h79_over_fixed_validation",
        *structure_metrics,
    )
    ratios = _architecture_ratios(rows, prior.ERROR_METRICS)
    history_rows = _manifest_bound_csv(args.b1_c2_analysis_root, "training_curves.csv")
    training_surface = _training_surface_rows(history_rows)
    replay_summary, replay_points = _replay_comparison(
        history_rows, args.r1_packet_root
    )
    historical_source_summary = _load_json(args.historical_source_summary)
    source_rows = _source_comparison(historical_source_summary, args.r1_packet_root)

    outputs: dict[str, list[dict[str, Any]]] = {
        "exact_exposure_cells.csv": rows,
        "cross_seed_summary.csv": prior._group_summary(rows, summary_metrics),
        "architecture_ratios.csv": ratios,
        "architecture_ratio_summary.csv": _ratio_summary(ratios),
        "precision_ratios.csv": _precision_ratios(rows),
        "scaling_summary.csv": prior._scaling_summary(rows, prior.ERROR_METRICS),
        "paired_case_bootstrap.csv": _paired_case_rows(
            cells, draws=args.bootstrap_draws, seed=args.bootstrap_seed
        ),
        "training_surface_points.csv": training_surface,
        "replay_trace_summary.csv": replay_summary,
        "replay_trace_points.csv": replay_points,
        "source_snapshot_comparison.csv": source_rows,
    }
    args.output_dir.mkdir(parents=True)
    for name, table in outputs.items():
        write_csv(args.output_dir / name, table)

    source_status_counts = {
        status: sum(row["status"] == status for row in source_rows)
        for status in ("same", "changed", "historical_only", "replay_only")
    }
    all_pcfno_exact = all(
        row["all_exact"]
        for row in replay_summary
        if row["architecture"] == "pcfno"
    )
    any_pcno_divergence = any(
        not row["all_exact"]
        for row in replay_summary
        if row["architecture"] == "pcno"
        and row["metric"] != "learning_rate"
    )
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "cell_count_per_precision": 36,
        "seeds": list(SEEDS),
        "trajectory_counts": list(COUNTS),
        "architectures": list(ARCHITECTURES),
        "precision_arms": list(PRECISIONS),
        "same_checkpoint_hashes_across_precisions": True,
        "presentations_per_training_trajectory": 64,
        "optimizer_step_by_trajectory_count": {
            str(count): 64 * count for count in COUNTS
        },
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "replay_trace_finding": {
            "pcfno_all_metrics_exact": all_pcfno_exact,
            "pcno_non_learning_rate_divergence_observed": any_pcno_divergence,
            "source_status_counts": source_status_counts,
            "causal_attribution_supported": False,
        },
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "input_summary_sha256": {
            "prior_bf16": sha256_file(args.prior_bf16_root / "summary.json"),
            "prior_none": sha256_file(args.prior_fp32_root / "summary.json"),
            "r1_bf16": sha256_file(args.r1_bf16_root / "summary.json"),
            "r1_none": sha256_file(args.r1_fp32_root / "summary.json"),
        },
        "b1_c2_analysis_manifest_sha256": sha256_file(
            args.b1_c2_analysis_root / "artifact_manifest.json"
        ),
        "r1_retrieval_manifest_sha256": sha256_file(
            args.r1_packet_root / "retrieval_manifest.json"
        ),
        "historical_source_summary_sha256": sha256_file(
            args.historical_source_summary
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "claims_not_supported": [
            "causal attribution to data, optimization, capacity, or gradient",
            "bitwise-identical historical and replay executables",
            "deterministic PCNO training failure",
            "fixed optimizer updates, compute, or learning-rate phase across counts",
            "a paper-faithful FFNO comparison",
            "independent test performance",
        ],
    }
    atomic_write_json(args.output_dir / "analysis.json", analysis)
    artifact_names = ["analysis.json", *outputs]
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
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
