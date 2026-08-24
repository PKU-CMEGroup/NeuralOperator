#!/usr/bin/env python3
"""Analyze the frozen D094 B1-C2 three-seed checkpoint audit.

The analysis keeps online train, fixed seen-train, fixed validation,
selection-cohort rollout, seed-specific outside-selection rollout, and the
common-nine rollout view distinct.  Seed-specific 28-case statistics combine
initialization and cohort variation; the common-nine view fixes cases but is a
small cohort.  No checkpoint is selected by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

from scripts.time_dependent_no.analyze_pcno_bump_scaling import (
    _load_json,
    _load_jsonl,
    _metric_snapshot,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c2 import (
    ARCHITECTURES,
    ARTIFACT_SCHEMA as AUDIT_ARTIFACT_SCHEMA,
    CHECKPOINT_ROLES,
    SCHEMA as AUDIT_SCHEMA,
    SEEDS,
    TRAJECTORY_COUNTS,
)

SCHEMA = "d094_b1_c2_three_seed_analysis_v1"
ARTIFACT_SCHEMA = "d094_b1_c2_three_seed_analysis_artifacts_v1"
EPOCHS = 80
STEPS_PER_EPOCH = 256
WINDOWS_PER_TRAJECTORY = 79
ERROR_METRICS = (
    "online_train_one_step_relative_l2",
    "fixed_seen_train_one_step_relative_l2",
    "fixed_validation_one_step_relative_l2",
    "internal_rollout_all_call_mean_relative_l2",
    "internal_rollout_h79_relative_l2",
    "seed_specific_outside_rollout_all_call_mean_relative_l2",
    "seed_specific_outside_rollout_h79_relative_l2",
    "common_outside_rollout_all_call_mean_relative_l2",
    "common_outside_rollout_h79_relative_l2",
)
CURVE_METRICS = (
    "online_train_one_step_relative_l2",
    "fixed_seen_train_one_step_relative_l2",
    "fixed_validation_one_step_relative_l2",
    "rollout_all_call_mean_relative_l2",
    "rollout_h79_relative_l2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-ladder-root", type=Path, required=True)
    parser.add_argument("--seed0-n256-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_825)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV requires rows: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("analysis metric must be finite or null")
    return resolved


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-14)
    return left == right


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("summary statistics require finite values")
    mean = statistics.mean(finite)
    return {
        "n": len(finite),
        "mean": mean,
        "sample_std": statistics.stdev(finite) if len(finite) > 1 else None,
        "minimum": min(finite),
        "maximum": max(finite),
        "coefficient_of_variation": (
            statistics.stdev(finite) / abs(mean)
            if len(finite) > 1 and mean != 0.0
            else None
        ),
    }


def _log_log_slope(counts: Sequence[int], values: Sequence[float]) -> float:
    if len(counts) != len(values) or len(counts) < 2:
        raise ValueError("log-log slope requires paired points")
    x = [math.log2(int(count)) for count in counts]
    y = [math.log10(float(value)) for value in values]
    if any(not math.isfinite(value) for value in y):
        raise ValueError("log-log slope requires positive finite errors")
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return (
        sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y))
        / denominator
    )


def verify_audit_artifacts(audit_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(audit_root / "summary.json")
    if (
        summary.get("schema") != AUDIT_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_on_outside_cases") is not False
        or summary.get("common_outside_selection_count") != 9
        or summary.get("same_audit_cohort_across_all_seeds") is not False
    ):
        raise ValueError("B1-C2 audit is incomplete or violates its contract")
    if set(summary.get("outside_selection_count_by_seed", {}).values()) != {28}:
        raise ValueError("B1-C2 seed-specific audit counts changed")
    manifest_path = audit_root / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != AUDIT_ARTIFACT_SCHEMA:
        raise ValueError("B1-C2 audit artifact schema changed")
    records = manifest.get("files")
    if not isinstance(records, Mapping) or len(records) != 74:
        raise ValueError("B1-C2 audit artifact inventory changed")
    for name, record in records.items():
        path = audit_root / str(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"B1-C2 audit artifact changed: {name}")
    return summary, {
        "manifest_sha256": _sha256_file(manifest_path),
        "checked_files": len(records),
    }


def _cell_identity(cell: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(cell["seed"]),
        int(cell["trajectory_count"]),
        str(cell["architecture"]),
        str(cell["checkpoint_role"]),
    )


def validate_cells(
    summary: Mapping[str, Any],
) -> dict[tuple[int, int, str, str], Mapping[str, Any]]:
    cells = summary.get("cells")
    if not isinstance(cells, list):
        raise TypeError("B1-C2 audit lacks cells")
    indexed: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    common_keys = [str(key) for key in summary["common_outside_selection_keys"]]
    outside_by_seed = summary["outside_selection_keys_by_seed"]
    for cell in cells:
        identity = _cell_identity(cell)
        if identity in indexed:
            raise ValueError(f"duplicate B1-C2 audit cell: {identity}")
        seed = str(identity[0])
        expected_outside = [str(key) for key in outside_by_seed[seed]]
        observed_outside = [
            str(row["trajectory"])
            for row in cell["outside_selection_rollout"]["trajectories"]
        ]
        if (
            observed_outside != expected_outside
            or cell["audit_cohort"]["outside_selection_keys"] != expected_outside
            or cell["common_outside_selection_rollout"]["keys"] != common_keys
        ):
            raise ValueError(f"B1-C2 audit cohort changed: {identity}")
        indexed[identity] = cell
    expected = {
        (seed, count, architecture, role)
        for seed in SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
        for role in CHECKPOINT_ROLES
    }
    if set(indexed) != expected:
        raise ValueError("B1-C2 audit identity matrix changed")
    return indexed


def checkpoint_rows(
    indexed: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in sorted(
        indexed,
        key=lambda value: (
            value[0],
            value[1],
            ARCHITECTURES.index(value[2]),
            CHECKPOINT_ROLES.index(value[3]),
        ),
    ):
        cell = indexed[identity]
        metrics = cell["checkpoint_training_metrics"]
        outside = cell["outside_selection_rollout"]
        common = cell["common_outside_selection_rollout"]
        rows.append(
            {
                "seed": identity[0],
                "trajectory_count": identity[1],
                "architecture": identity[2],
                "checkpoint_role": identity[3],
                "optimizer_step": int(cell["optimizer_step"]),
                "checkpoint_sha256": str(cell["checkpoint_sha256"]),
                "online_train_one_step_relative_l2": _finite_or_none(
                    metrics["online_train_one_step_relative_l2"]
                ),
                "fixed_seen_train_one_step_relative_l2": _finite_or_none(
                    metrics["fixed_seen_train_one_step_relative_l2"]
                ),
                "fixed_validation_one_step_relative_l2": _finite_or_none(
                    metrics["fixed_validation_one_step_relative_l2"]
                ),
                "internal_rollout_all_call_mean_relative_l2": _finite_or_none(
                    metrics["rollout_all_call_mean_relative_l2"]
                ),
                "internal_rollout_h79_relative_l2": _finite_or_none(
                    metrics["rollout_h79_relative_l2"]
                ),
                "seed_specific_outside_rollout_all_call_mean_relative_l2": (
                    _finite_or_none(outside["mean_full_horizon_relative_l2"])
                ),
                "seed_specific_outside_rollout_h79_relative_l2": _finite_or_none(
                    outside["mean_endpoint_relative_l2"]["79"]
                ),
                "seed_specific_outside_rollout_completion_rate": _finite_or_none(
                    outside["completion_rate"]
                ),
                "seed_specific_outside_rollout_hard_failure_count": int(
                    outside["hard_failure_count"]
                ),
                "seed_specific_outside_rollout_physical_admissibility_rate": (
                    _finite_or_none(outside["physical_admissibility_rate"])
                ),
                "common_outside_rollout_all_call_mean_relative_l2": _finite_or_none(
                    common["mean_full_horizon_relative_l2"]
                ),
                "common_outside_rollout_h79_relative_l2": _finite_or_none(
                    common["mean_endpoint_relative_l2"]["79"]
                ),
                "common_outside_rollout_completion_rate": _finite_or_none(
                    common["completion_rate"]
                ),
                "common_outside_rollout_hard_failure_count": int(
                    common["hard_failure_count"]
                ),
                "common_outside_rollout_physical_admissibility_rate": (
                    _finite_or_none(common["physical_admissibility_rate"])
                ),
            }
        )
    return rows


def _find_run_directories(
    roots: Sequence[Path], expected_names: set[str]
) -> dict[str, Path]:
    observed: dict[str, Path] = {}
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"training metadata root is not regular: {root}")
        for metrics_path in root.rglob("metrics.jsonl"):
            run_dir = metrics_path.parent
            if run_dir.name not in expected_names:
                continue
            if run_dir.name in observed or run_dir.is_symlink():
                raise ValueError(f"duplicate or linked training run: {run_dir.name}")
            observed[run_dir.name] = run_dir
    if set(observed) != expected_names:
        missing = sorted(expected_names - set(observed))
        raise ValueError(f"training metadata runs are incomplete: {missing}")
    return observed


def training_input_manifest(
    indexed: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    labelled_roots: Mapping[str, Path],
) -> dict[str, Any]:
    expected_names = {str(cell["run"]) for cell in indexed.values()}
    run_dirs = _find_run_directories(list(labelled_roots.values()), expected_names)
    receipt_records: dict[str, Mapping[str, Any]] = {}
    replication_root = labelled_roots["replication"]
    receipt_path = replication_root / "matrix_receipt.json"
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        receipt_records = {
            str(record["path"]): record for record in receipt.get("files", [])
        }
    records: dict[str, Any] = {}
    resolved_roots = {
        label: root.resolve() for label, root in labelled_roots.items()
    }
    for run, run_dir in sorted(run_dirs.items()):
        metrics_path = run_dir / "metrics.jsonl"
        root_label = None
        relative = None
        for label, root in resolved_roots.items():
            try:
                relative = metrics_path.resolve().relative_to(root).as_posix()
                root_label = label
                break
            except ValueError:
                continue
        if root_label is None or relative is None:
            raise ValueError(f"training history leaves its declared roots: {run}")
        observed = {
            "bytes": metrics_path.stat().st_size,
            "sha256": _sha256_file(metrics_path),
        }
        receipt_match: bool | None = None
        if root_label == "replication":
            retained = receipt_records.get(relative)
            if not isinstance(retained, Mapping):
                raise ValueError(f"replication receipt lacks metrics: {run}")
            receipt_match = (
                observed["bytes"] == int(retained["bytes"])
                and observed["sha256"] == str(retained["sha256"])
            )
            if not receipt_match:
                raise ValueError(f"replication training history changed: {run}")
        records[run] = {
            **observed,
            "root_role": root_label,
            "matrix_receipt_match": receipt_match,
            "selected_and_terminal_anchors_match_audit": True,
        }
    return {
        "schema": "d094_b1_c2_training_history_inputs_v1",
        "run_count": len(records),
        "historical_test_population_accessed": False,
        "records": records,
    }


def _history_scope_matches(
    snapshot: Mapping[str, Any], audit_metrics: Mapping[str, Any]
) -> bool:
    fields = {
        "epoch": "epoch",
        "optimizer_step": "optimizer_step",
        "online_train_one_step_relative_l2": "online_train_one_step_relative_l2",
        "fixed_seen_train_one_step_relative_l2": (
            "fixed_seen_train_one_step_relative_l2"
        ),
        "fixed_validation_one_step_relative_l2": (
            "fixed_validation_one_step_relative_l2"
        ),
        "rollout_all_call_mean_relative_l2": "rollout_all_call_mean_relative_l2",
        "rollout_h79_relative_l2": "rollout_h79_relative_l2",
    }
    return all(
        _same_value(snapshot[left], audit_metrics[right])
        for left, right in fields.items()
    )


def training_curve_rows(
    indexed: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    roots: Sequence[Path],
) -> list[dict[str, Any]]:
    runs_by_identity: dict[tuple[int, int, str], str] = {}
    for (seed, count, architecture, role), cell in indexed.items():
        key = (seed, count, architecture)
        run = str(cell["run"])
        if key in runs_by_identity and runs_by_identity[key] != run:
            raise ValueError(f"selected/terminal run mismatch: {key}")
        runs_by_identity[key] = run
    run_dirs = _find_run_directories(roots, set(runs_by_identity.values()))
    output: list[dict[str, Any]] = []
    for identity in sorted(runs_by_identity):
        seed, count, architecture = identity
        run_dir = run_dirs[runs_by_identity[identity]]
        raw_rows = _load_jsonl(run_dir / "metrics.jsonl")
        snapshots = [_metric_snapshot(row) for row in raw_rows]
        if len(snapshots) != EPOCHS or [row["optimizer_step"] for row in snapshots] != [
            (epoch + 1) * STEPS_PER_EPOCH for epoch in range(EPOCHS)
        ]:
            raise ValueError(f"training history changed: {run_dir.name}")
        selected_cell = indexed[(seed, count, architecture, "selected")]
        terminal_cell = indexed[(seed, count, architecture, "terminal")]
        selected_step = int(selected_cell["optimizer_step"])
        terminal_step = int(terminal_cell["optimizer_step"])
        selected_rows = [
            row for row in snapshots if int(row["optimizer_step"]) == selected_step
        ]
        terminal_rows = [
            row for row in snapshots if int(row["optimizer_step"]) == terminal_step
        ]
        if (
            len(selected_rows) != 1
            or len(terminal_rows) != 1
            or not _history_scope_matches(
                selected_rows[0], selected_cell["checkpoint_training_metrics"]
            )
            or not _history_scope_matches(
                terminal_rows[0], terminal_cell["checkpoint_training_metrics"]
            )
        ):
            raise ValueError(f"audit/history metric anchors changed: {run_dir.name}")
        for row in snapshots:
            seen = row["fixed_seen_train_one_step_relative_l2"]
            validation = row["fixed_validation_one_step_relative_l2"]
            output.append(
                {
                    "seed": seed,
                    "trajectory_count": count,
                    "architecture": architecture,
                    "run": run_dir.name,
                    "epoch": row["epoch"],
                    "optimizer_step": row["optimizer_step"],
                    "window_equivalent_exposure": row["optimizer_step"]
                    / (WINDOWS_PER_TRAJECTORY * count),
                    "learning_rate": row["learning_rate"],
                    **{name: row[name] for name in CURVE_METRICS},
                    "fixed_validation_over_seen_ratio": (
                        None
                        if seen is None or validation is None
                        else float(validation) / float(seen)
                    ),
                    "is_selected_checkpoint": row["optimizer_step"] == selected_step,
                    "is_terminal_checkpoint": row["optimizer_step"] == terminal_step,
                }
            )
    return output


def cross_seed_rows(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for count in TRAJECTORY_COUNTS:
        for architecture in ARCHITECTURES:
            for role in CHECKPOINT_ROLES:
                group = [
                    row
                    for row in checkpoints
                    if row["trajectory_count"] == count
                    and row["architecture"] == architecture
                    and row["checkpoint_role"] == role
                ]
                if len(group) != len(SEEDS):
                    raise ValueError("cross-seed checkpoint group is incomplete")
                for metric in ERROR_METRICS:
                    values = [row[metric] for row in group]
                    if any(value is None for value in values):
                        continue
                    output.append(
                        {
                            "trajectory_count": count,
                            "architecture": architecture,
                            "checkpoint_role": role,
                            "metric": metric,
                            **_summary_stats([float(value) for value in values]),
                        }
                    )
    return output


def training_architecture_ratio_rows(
    curves: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
            int(row["optimizer_step"]),
        ): row
        for row in curves
    }
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for count in TRAJECTORY_COUNTS:
            for step in range(
                STEPS_PER_EPOCH, EPOCHS * STEPS_PER_EPOCH + 1, STEPS_PER_EPOCH
            ):
                pcno = indexed[(seed, count, "pcno", step)]
                pcfno = indexed[(seed, count, "pcfno", step)]
                row: dict[str, Any] = {
                    "seed": seed,
                    "trajectory_count": count,
                    "optimizer_step": step,
                    "window_equivalent_exposure": pcno["window_equivalent_exposure"],
                }
                for metric in CURVE_METRICS:
                    left = pcno[metric]
                    right = pcfno[metric]
                    row[f"pcno_{metric}"] = left
                    row[f"pcfno_{metric}"] = right
                    row[f"pcno_over_pcfno_{metric}"] = (
                        None
                        if left is None or right is None
                        else float(left) / float(right)
                    )
                output.append(row)
    return output


def training_ratio_summary_rows(
    ratios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for count in TRAJECTORY_COUNTS:
            group = [
                row
                for row in ratios
                if row["seed"] == seed and row["trajectory_count"] == count
            ]
            for metric in CURVE_METRICS:
                name = f"pcno_over_pcfno_{metric}"
                values = [float(row[name]) for row in group if row[name] is not None]
                stats = _summary_stats(values)
                output.append(
                    {
                        "seed": seed,
                        "trajectory_count": count,
                        "metric": metric,
                        **stats,
                        "first_ratio": values[0],
                        "last_ratio": values[-1],
                        "relative_range": (max(values) - min(values))
                        / statistics.mean(values),
                    }
                )
    return output


def generalization_ratio_summary_rows(
    curves: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the fixed-validation/fixed-seen ratio over optimization."""
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for count in TRAJECTORY_COUNTS:
            for architecture in ARCHITECTURES:
                group = sorted(
                    (
                        row
                        for row in curves
                        if row["seed"] == seed
                        and row["trajectory_count"] == count
                        and row["architecture"] == architecture
                    ),
                    key=lambda row: int(row["optimizer_step"]),
                )
                values = [
                    float(row["fixed_validation_over_seen_ratio"])
                    for row in group
                    if row["fixed_validation_over_seen_ratio"] is not None
                ]
                if not values:
                    raise ValueError("fixed validation/seen ratio is unavailable")
                output.append(
                    {
                        "seed": seed,
                        "trajectory_count": count,
                        "architecture": architecture,
                        **_summary_stats(values),
                        "first_ratio": values[0],
                        "last_ratio": values[-1],
                        "relative_range": (max(values) - min(values))
                        / statistics.mean(values),
                    }
                )
    return output


def checkpoint_architecture_ratio_rows(
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
            str(row["checkpoint_role"]),
        ): row
        for row in checkpoints
    }
    raw: list[dict[str, Any]] = []
    for seed in SEEDS:
        for count in TRAJECTORY_COUNTS:
            for role in CHECKPOINT_ROLES:
                pcno = indexed[(seed, count, "pcno", role)]
                pcfno = indexed[(seed, count, "pcfno", role)]
                for metric in ERROR_METRICS:
                    left = pcno[metric]
                    right = pcfno[metric]
                    if left is None or right is None:
                        continue
                    raw.append(
                        {
                            "seed": seed,
                            "trajectory_count": count,
                            "checkpoint_role": role,
                            "metric": metric,
                            "pcno": left,
                            "pcfno": right,
                            "pcno_over_pcfno": float(left) / float(right),
                            "pcno_minus_pcfno": float(left) - float(right),
                            "lower_error_architecture": (
                                "pcno" if float(left) < float(right) else "pcfno"
                            ),
                        }
                    )
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[(row["trajectory_count"], row["checkpoint_role"], row["metric"])].append(
            row
        )
    for (count, role, metric), rows in sorted(groups.items()):
        if len(rows) != len(SEEDS):
            raise ValueError("checkpoint architecture ratio lacks three seeds")
        ratios = [float(row["pcno_over_pcfno"]) for row in rows]
        wins = sum(value < 1.0 for value in ratios)
        summary.append(
            {
                "trajectory_count": count,
                "checkpoint_role": role,
                "metric": metric,
                **_summary_stats(ratios),
                "geometric_mean_ratio": math.exp(
                    statistics.mean(math.log(value) for value in ratios)
                ),
                "pcno_lower_seed_count": wins,
                "replicated_direction": (
                    "pcno_lower_2of3_or_more"
                    if wins >= 2
                    else "pcfno_lower_2of3_or_more"
                ),
            }
        )
    return raw, summary


def selected_terminal_rows(
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
            str(row["checkpoint_role"]),
        ): row
        for row in checkpoints
    }
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for count in TRAJECTORY_COUNTS:
            for architecture in ARCHITECTURES:
                selected = indexed[(seed, count, architecture, "selected")]
                terminal = indexed[(seed, count, architecture, "terminal")]
                for metric in ERROR_METRICS:
                    left = selected[metric]
                    right = terminal[metric]
                    if left is None or right is None:
                        continue
                    output.append(
                        {
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            "metric": metric,
                            "selected_step": selected["optimizer_step"],
                            "terminal_step": terminal["optimizer_step"],
                            "selected": left,
                            "terminal": right,
                            "terminal_over_selected": float(right) / float(left),
                            "terminal_minus_selected": float(right) - float(left),
                        }
                    )
    return output


def scaling_rows(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            for role in CHECKPOINT_ROLES:
                group = {
                    int(row["trajectory_count"]): row
                    for row in checkpoints
                    if row["seed"] == seed
                    and row["architecture"] == architecture
                    and row["checkpoint_role"] == role
                }
                if set(group) != set(TRAJECTORY_COUNTS):
                    raise ValueError("scaling row lacks the six counts")
                for metric in ERROR_METRICS:
                    values = [group[count][metric] for count in TRAJECTORY_COUNTS]
                    if any(value is None for value in values):
                        continue
                    finite = [float(value) for value in values]
                    best_index = min(range(len(finite)), key=finite.__getitem__)
                    output.append(
                        {
                            "seed": seed,
                            "architecture": architecture,
                            "checkpoint_role": role,
                            "metric": metric,
                            "best_trajectory_count": TRAJECTORY_COUNTS[best_index],
                            "monotone_nonincreasing": all(
                                right <= left for left, right in zip(finite, finite[1:])
                            ),
                            "n256_over_n128": finite[-1] / finite[-2],
                            "log2_count_log10_error_slope": _log_log_slope(
                                TRAJECTORY_COUNTS, finite
                            ),
                        }
                    )
    return output


def _case_values(
    cell: Mapping[str, Any], keys: Sequence[str], metric: str
) -> dict[str, float]:
    rows = {
        str(row["trajectory"]): row
        for row in cell["outside_selection_rollout"]["trajectories"]
    }
    output: dict[str, float] = {}
    for key in keys:
        row = rows[str(key)]
        value = (
            row["mean_prefix_relative_l2"]
            if metric == "all_call"
            else row["endpoint_relative_l2"]["79"]
        )
        output[str(key)] = float(value)
    return output


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left or draws <= 0:
        raise ValueError("paired bootstrap requires matched nonempty samples")
    rng = random.Random(seed)
    differences: list[float] = []
    ratios: list[float] = []
    for _ in range(draws):
        indices = [rng.randrange(len(left)) for _ in left]
        left_mean = statistics.mean(left[index] for index in indices)
        right_mean = statistics.mean(right[index] for index in indices)
        differences.append(left_mean - right_mean)
        ratios.append(left_mean / right_mean)
    observed_left = statistics.mean(left)
    observed_right = statistics.mean(right)
    return {
        "case_count": len(left),
        "pcno_mean": observed_left,
        "pcfno_mean": observed_right,
        "pcno_minus_pcfno": observed_left - observed_right,
        "difference_ci025": _percentile(differences, 0.025),
        "difference_ci975": _percentile(differences, 0.975),
        "pcno_over_pcfno": observed_left / observed_right,
        "ratio_ci025": _percentile(ratios, 0.025),
        "ratio_ci975": _percentile(ratios, 0.975),
        "pcno_lower_case_count": sum(
            left_value < right_value for left_value, right_value in zip(left, right)
        ),
    }


def paired_case_bootstrap_rows(
    indexed: Mapping[tuple[int, int, str, str], Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    common_keys = [str(key) for key in summary["common_outside_selection_keys"]]
    for seed_index, initialization_seed in enumerate(SEEDS):
        seed_specific_keys = [
            str(key)
            for key in summary["outside_selection_keys_by_seed"][
                str(initialization_seed)
            ]
        ]
        for count_index, count in enumerate(TRAJECTORY_COUNTS):
            for role_index, role in enumerate(CHECKPOINT_ROLES):
                pcno = indexed[(initialization_seed, count, "pcno", role)]
                pcfno = indexed[(initialization_seed, count, "pcfno", role)]
                for cohort_index, (cohort, keys) in enumerate(
                    (
                        ("seed_specific_28", seed_specific_keys),
                        ("common_9", common_keys),
                    )
                ):
                    for metric_index, metric in enumerate(("all_call", "h79")):
                        left_by_key = _case_values(pcno, keys, metric)
                        right_by_key = _case_values(pcfno, keys, metric)
                        if list(left_by_key) != list(right_by_key):
                            raise ValueError("paired architecture case order changed")
                        bootstrap_seed = (
                            seed
                            + seed_index * 10_000
                            + count_index * 1_000
                            + role_index * 100
                            + cohort_index * 10
                            + metric_index
                        )
                        stats = _paired_bootstrap(
                            list(left_by_key.values()),
                            list(right_by_key.values()),
                            draws=draws,
                            seed=bootstrap_seed,
                        )
                        output.append(
                            {
                                "seed": initialization_seed,
                                "trajectory_count": count,
                                "checkpoint_role": role,
                                "cohort": cohort,
                                "metric": metric,
                                "bootstrap_draws": draws,
                                "bootstrap_seed": bootstrap_seed,
                                **stats,
                            }
                        )
    return output


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C2 analysis output already exists")
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap draws must be positive")

    audit_summary, audit_integrity = verify_audit_artifacts(args.audit_root)
    indexed = validate_cells(audit_summary)
    checkpoints = checkpoint_rows(indexed)
    labelled_roots = {
        "seed0_ladder": args.seed0_ladder_root,
        "seed0_n256": args.seed0_n256_root,
        "replication": args.replication_root,
    }
    curves = training_curve_rows(indexed, list(labelled_roots.values()))
    training_inputs = training_input_manifest(indexed, labelled_roots)
    cross_seed = cross_seed_rows(checkpoints)
    training_ratios = training_architecture_ratio_rows(curves)
    training_ratio_summary = training_ratio_summary_rows(training_ratios)
    generalization_ratio_summary = generalization_ratio_summary_rows(curves)
    checkpoint_ratios, checkpoint_ratio_summary = checkpoint_architecture_ratio_rows(
        checkpoints
    )
    selected_terminal = selected_terminal_rows(checkpoints)
    scaling = scaling_rows(checkpoints)
    bootstrap = paired_case_bootstrap_rows(
        indexed,
        audit_summary,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )

    args.output_dir.mkdir(parents=True)
    outputs: dict[str, Sequence[Mapping[str, Any]]] = {
        "checkpoint_metrics.csv": checkpoints,
        "training_curves.csv": curves,
        "cross_seed_summary.csv": cross_seed,
        "training_architecture_ratios.csv": training_ratios,
        "training_ratio_summary.csv": training_ratio_summary,
        "fixed_validation_seen_ratio_summary.csv": generalization_ratio_summary,
        "checkpoint_architecture_ratios.csv": checkpoint_ratios,
        "checkpoint_architecture_ratio_summary.csv": checkpoint_ratio_summary,
        "selected_terminal_dynamics.csv": selected_terminal,
        "scaling_summary.csv": scaling,
        "paired_case_bootstrap.csv": bootstrap,
    }
    for name, rows in outputs.items():
        _write_csv(args.output_dir / name, rows)
    _write_json(args.output_dir / "training_input_manifest.json", training_inputs)

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "audit_integrity": audit_integrity,
        "audit_schema": audit_summary["schema"],
        "checkpoint_count": len(checkpoints),
        "training_curve_row_count": len(curves),
        "training_history_run_count": training_inputs["run_count"],
        "analysis_source_sha256": _sha256_file(Path(__file__)),
        "common_outside_selection_count": 9,
        "seed_specific_outside_selection_count": 28,
        "cross_seed_uncertainty_scope": audit_summary["cross_seed_uncertainty_scope"],
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "claims_not_supported": [
            "pure initialization uncertainty from seed-specific 28-case metrics",
            "a monotone or asymptotic data-scaling law from six counts",
            "causal attribution to optimization, capacity, or the gradient branch",
            "paper-faithful FFNO behavior",
            "historical-test performance",
        ],
    }
    _write_json(args.output_dir / "analysis.json", result)
    artifact_names = ["analysis.json", "training_input_manifest.json", *outputs]
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": _sha256_file(args.output_dir / name),
            }
            for name in artifact_names
        },
    }
    _write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
