#!/usr/bin/env python3
"""Validate and visualize the D094 bump schedule-gate training dynamics.

The analysis deliberately keeps four different objects separate: online
one-step training error, fixed-bank seen-train one-step error, fixed-bank open
validation one-step error, and autonomous rollout error.  In particular, the
selected checkpoint is reconstructed from ``metrics.jsonl`` using
``summary.json::best_epoch``; terminal scalar receipts are never relabelled as
selected-checkpoint measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "d094_bump_schedule_gate_analysis_v1"
EXPECTED_SOURCE_SET_DIGEST = (
    "6c510fbdca8f50d2bfacd40239574e7ac4496bdb0ba575c5ce69bb744568fca5"
)
EXPECTED_SPLIT_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
EXPECTED_OPTIMIZER_STEPS = 20_480
EXPECTED_EPOCHS = 80
EXPECTED_STEPS_PER_EPOCH = 256
EXPECTED_ROLLOUT_STEPS = (
    256,
    1_280,
    2_560,
    3_840,
    5_120,
    6_400,
    7_680,
    8_960,
    10_240,
    11_520,
    12_800,
    14_080,
    15_360,
    16_640,
    17_920,
    19_200,
    20_480,
)
ARCHITECTURE_BY_BRANCH_MODE = {"full": "pcno", "no_gradient": "pcfno"}
SCHEDULE_BY_DECAY_STEPS = {5_120: "prefix_tail", 20_480: "stretched"}
CELL_ORDER = (
    ("pcno", "prefix_tail"),
    ("pcno", "stretched"),
    ("pcfno", "prefix_tail"),
    ("pcfno", "stretched"),
)

METRIC_FIELDS = (
    "online_train_one_step_relative_l2",
    "fixed_seen_train_one_step_relative_l2",
    "fixed_validation_one_step_relative_l2",
    "rollout_all_call_mean_relative_l2",
    "rollout_h79_relative_l2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-root",
        type=Path,
        required=True,
        help="D094 schedule-gate root containing gate.exit and outputs/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-set-digest",
        default=EXPECTED_SOURCE_SET_DIGEST,
    )
    parser.add_argument(
        "--expected-split-partition-digest",
        default=EXPECTED_SPLIT_PARTITION_DIGEST,
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV output requires at least one row")
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _require_finite(value: Any, *, field: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _metric_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    train = row.get("train")
    validation = row.get("validation")
    if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
        raise TypeError("metric row lacks train or validation measurements")
    comparable_seen = train.get("comparable_seen")
    if comparable_seen is not None and not isinstance(comparable_seen, Mapping):
        raise TypeError(
            "fixed seen-train pair-bank measurement must be an object or null"
        )
    rollout = row.get("rollout")
    snapshot: dict[str, Any] = {
        "epoch": int(row["epoch"]),
        "optimizer_step": int(train["completed_optimizer_steps"]),
        "learning_rate": _require_finite(row["learning_rate"], field="learning_rate"),
        "online_train_one_step_relative_l2": _require_finite(
            train["relative_l2"], field="online train one-step relative L2"
        ),
        "fixed_seen_train_one_step_relative_l2": (
            None
            if comparable_seen is None
            else _require_finite(
                comparable_seen["relative_l2"],
                field="fixed seen-train one-step relative L2",
            )
        ),
        "fixed_validation_one_step_relative_l2": _require_finite(
            validation["relative_l2"],
            field="fixed open-validation one-step relative L2",
        ),
        "rollout_all_call_mean_relative_l2": None,
        "rollout_h79_relative_l2": None,
        "rollout_completion_rate": None,
        "rollout_hard_failure_count": None,
        "physical_admissibility_rate": None,
    }
    if rollout is not None:
        if not isinstance(rollout, Mapping):
            raise TypeError("rollout measurement must be an object or null")
        endpoints = rollout.get("mean_endpoint_relative_l2")
        if not isinstance(endpoints, Mapping) or "79" not in endpoints:
            raise ValueError("rollout measurement lacks the H79 endpoint")
        snapshot.update(
            {
                "rollout_all_call_mean_relative_l2": _require_finite(
                    rollout["mean_full_horizon_relative_l2"],
                    field="rollout all-call mean relative L2",
                ),
                "rollout_h79_relative_l2": _require_finite(
                    endpoints["79"], field="rollout H79 relative L2"
                ),
                "rollout_completion_rate": _require_finite(
                    rollout["completion_rate"], field="rollout completion rate"
                ),
                "rollout_hard_failure_count": int(rollout.get("hard_failure_count", 0)),
                "physical_admissibility_rate": _require_finite(
                    rollout["physical_admissibility_rate"],
                    field="rollout physical-admissibility rate",
                ),
            }
        )
    return snapshot


def _selection_tuple(snapshot: Mapping[str, Any]) -> list[float]:
    rollout_mean = snapshot["rollout_all_call_mean_relative_l2"]
    rollout_h79 = snapshot["rollout_h79_relative_l2"]
    if rollout_mean is None or rollout_h79 is None:
        raise ValueError("selected checkpoint must have a rollout measurement")
    completion = float(snapshot["rollout_completion_rate"])
    hard_failures = int(snapshot["rollout_hard_failure_count"])
    numerically_complete = completion == 1.0 and hard_failures == 0
    return [
        float(numerically_complete),
        -float(rollout_mean),
        -float(rollout_h79),
        -float(snapshot["fixed_validation_one_step_relative_l2"]),
        completion,
        completion,
    ]


def _snapshot_integrity(run_dir: Path, manifest: Mapping[str, Any]) -> int:
    if manifest.get("schema") != "pcno_euler2d_source_snapshot_v6":
        raise ValueError(f"unexpected source-snapshot schema in {run_dir.name}")
    inventories = (manifest.get("files"), manifest.get("provenance_files"))
    checked = 0
    for inventory in inventories:
        if not isinstance(inventory, Mapping):
            raise TypeError(f"source snapshot inventory is missing in {run_dir.name}")
        for relative, metadata in inventory.items():
            if not isinstance(relative, str) or not isinstance(metadata, Mapping):
                raise TypeError("source snapshot inventory entry is malformed")
            path = run_dir / "source_snapshot" / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"source snapshot file is absent or linked: {relative}"
                )
            if path.stat().st_size != int(metadata["bytes"]):
                raise ValueError(f"source snapshot byte count changed: {relative}")
            if _sha256_file(path) != str(metadata["sha256"]):
                raise ValueError(f"source snapshot digest changed: {relative}")
            checked += 1
    return checked


def _receipt_scope(
    receipt: Mapping[str, Any],
    selected: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, str]:
    scopes = {}
    for field in METRIC_FIELDS:
        if field not in receipt:
            scopes[field] = "absent"
            continue
        value = float(receipt[field])
        selected_match = math.isclose(
            value, float(selected[field]), rel_tol=0, abs_tol=1e-14
        )
        terminal_match = math.isclose(
            value, float(terminal[field]), rel_tol=0, abs_tol=1e-14
        )
        if selected_match and terminal_match:
            scopes[field] = "selected_and_terminal_identical"
        elif selected_match:
            scopes[field] = "selected"
        elif terminal_match:
            scopes[field] = "terminal"
        else:
            scopes[field] = "neither"
    return scopes


def _overoptimization_event(
    rollout_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the D094 three-checkpoint one-step over-optimization rule."""

    qualifying: list[dict[str, Any]] = []
    prior: list[Mapping[str, Any]] = []
    for snapshot in rollout_snapshots:
        if not prior:
            prior.append(snapshot)
            continue
        baseline = min(
            prior,
            key=lambda item: float(item["fixed_validation_one_step_relative_l2"]),
        )
        validation_ratio = float(
            snapshot["fixed_validation_one_step_relative_l2"]
        ) / float(baseline["fixed_validation_one_step_relative_l2"])
        seen_ratio = float(snapshot["fixed_seen_train_one_step_relative_l2"]) / float(
            baseline["fixed_seen_train_one_step_relative_l2"]
        )
        qualifies = validation_ratio >= 1.25 and seen_ratio <= 0.95
        qualifying.append(
            {
                "optimizer_step": int(snapshot["optimizer_step"]),
                "baseline_optimizer_step": int(baseline["optimizer_step"]),
                "validation_ratio_to_earlier_best": validation_ratio,
                "seen_ratio_to_same_checkpoint": seen_ratio,
                "qualifies": qualifies,
            }
        )
        prior.append(snapshot)
    streak = 0
    first_three_checkpoint_event = None
    for row in qualifying:
        streak = streak + 1 if row["qualifies"] else 0
        if streak == 3:
            first_three_checkpoint_event = int(row["optimizer_step"])
            break
    return {
        "detected": first_three_checkpoint_event is not None,
        "first_three_checkpoint_event_optimizer_step": first_three_checkpoint_event,
        "checkpoint_rows": qualifying,
        "contract": (
            "three consecutive rollout-evaluation checkpoints with fixed open-"
            "validation one-step error >=1.25x its earlier best and fixed seen-train "
            "one-step error <=0.95x the value at that earlier-best checkpoint"
        ),
    }


def _relative_change(terminal: float, selected: float) -> float:
    return terminal / selected - 1.0


def _discover_run_dirs(gate_root: Path) -> list[Path]:
    if gate_root.is_symlink() or not gate_root.is_dir():
        raise ValueError("gate root must be a regular directory")
    exit_path = gate_root / "gate.exit"
    if not exit_path.is_file() or exit_path.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("schedule gate does not have a zero exit receipt")
    if not (gate_root / "gate.completed").is_file():
        raise ValueError("schedule gate lacks its completion marker")
    return sorted((gate_root / "outputs").glob("gate_n256_*"))


def _validate_cell(
    run_dir: Path,
    *,
    expected_source_set_digest: str,
    expected_split_partition_digest: str,
) -> dict[str, Any]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"run directory must be regular: {run_dir}")
    summary = _load_json(run_dir / "summary.json")
    contract = _load_json(run_dir / "bump_scaling_contract.json")
    run_contract = _load_json(run_dir / "run_contract.json")
    receipt = _load_json(run_dir / "validation_receipt.json")
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    manifest = _load_json(run_dir / "source_snapshot" / "manifest.json")

    branch_mode = str(contract.get("differential_branch_mode"))
    if branch_mode not in ARCHITECTURE_BY_BRANCH_MODE:
        raise ValueError(f"unsupported differential branch mode: {branch_mode}")
    architecture = ARCHITECTURE_BY_BRANCH_MODE[branch_mode]
    scheduler = summary.get("optimizer", {}).get("scheduler_contract", {})
    decay_steps = int(scheduler.get("decay_optimizer_steps", -1))
    if decay_steps not in SCHEDULE_BY_DECAY_STEPS:
        raise ValueError(f"unsupported scheduler decay length: {decay_steps}")
    schedule = SCHEDULE_BY_DECAY_STEPS[decay_steps]

    expected_scalars = {
        "actual_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "completed_epochs": EXPECTED_EPOCHS,
        "train_trajectories": 256,
        "validation_trajectories": 44,
        "test_trajectories": 0,
    }
    for field, expected in expected_scalars.items():
        if int(summary.get(field, -1)) != expected:
            raise ValueError(f"{run_dir.name} has unexpected {field}")
    if contract.get("historical_test_population_accessed") is not False:
        raise ValueError("historical test population access is not closed")
    if contract.get("rollout_failure_policy") != "finite_only":
        raise ValueError("schedule gate must use finite-only rollout")
    if contract.get("checkpoint_selection_mode") != "full_horizon_error_first":
        raise ValueError("schedule gate selection mode changed")
    if int(contract.get("rollout_selection_trajectory_count", -1)) != 16:
        raise ValueError("rollout selection cohort must contain 16 trajectories")
    if contract.get("split_partition_digest") != expected_split_partition_digest:
        raise ValueError("split partition digest changed")
    if manifest.get("source_set_digest") != expected_source_set_digest:
        raise ValueError("source-set digest changed")
    checked_source_files = _snapshot_integrity(run_dir, manifest)

    if len(rows) != EXPECTED_EPOCHS:
        raise ValueError(f"{run_dir.name} must have {EXPECTED_EPOCHS} metric rows")
    snapshots = [_metric_snapshot(row) for row in rows]
    expected_epochs = list(range(EXPECTED_EPOCHS))
    expected_steps = [
        (epoch + 1) * EXPECTED_STEPS_PER_EPOCH for epoch in expected_epochs
    ]
    if [row["epoch"] for row in snapshots] != expected_epochs:
        raise ValueError("metric epochs are not contiguous")
    if [row["optimizer_step"] for row in snapshots] != expected_steps:
        raise ValueError("metric optimizer-step accounting changed")
    rollout_snapshots = [
        row for row in snapshots if row["rollout_all_call_mean_relative_l2"] is not None
    ]
    if (
        tuple(row["optimizer_step"] for row in rollout_snapshots)
        != EXPECTED_ROLLOUT_STEPS
    ):
        raise ValueError("rollout evaluation checkpoint inventory changed")

    best_epoch = int(summary["best_epoch"])
    selected_rows = [row for row in snapshots if row["epoch"] == best_epoch]
    if len(selected_rows) != 1:
        raise ValueError("best epoch does not identify one metric row")
    selected = selected_rows[0]
    terminal = snapshots[-1]
    reconstructed_selection = _selection_tuple(selected)
    stored_selection = [float(value) for value in summary["best_selection"]]
    if len(stored_selection) != len(reconstructed_selection) or any(
        not math.isclose(left, right, rel_tol=0, abs_tol=1e-14)
        for left, right in zip(stored_selection, reconstructed_selection, strict=True)
    ):
        raise ValueError("stored best-selection tuple does not match the selected row")
    if selected["rollout_completion_rate"] != 1.0 or int(
        selected["rollout_hard_failure_count"]
    ):
        raise ValueError("selected checkpoint is not numerically complete")

    receipt_scope = _receipt_scope(receipt, selected, terminal)
    receipt_has_terminal_metrics_under_best_epoch = any(
        scope == "terminal" for scope in receipt_scope.values()
    )
    changes = {
        field: _relative_change(float(terminal[field]), float(selected[field]))
        for field in METRIC_FIELDS
    }
    recurrence_divergence = bool(
        changes["fixed_seen_train_one_step_relative_l2"] < 0.0
        and changes["fixed_validation_one_step_relative_l2"] < 0.0
        and changes["rollout_all_call_mean_relative_l2"] > 0.0
    )
    return {
        "run": run_dir.name,
        "architecture": architecture,
        "differential_branch_mode": branch_mode,
        "schedule": schedule,
        "scheduler_decay_optimizer_steps": decay_steps,
        "source_set_digest": str(manifest["source_set_digest"]),
        "split_partition_digest": str(contract["split_partition_digest"]),
        "checked_source_snapshot_files": checked_source_files,
        "best_checkpoint_sha256": str(summary["artifact_sha256"]["best_checkpoint"]),
        "best_checkpoint_payload_retrieved": (run_dir / "best.pt").is_file(),
        "selected": selected,
        "terminal": terminal,
        "selected_to_terminal_relative_change": changes,
        "receipt_metric_scope": receipt_scope,
        "receipt_has_terminal_metrics_under_best_epoch": (
            receipt_has_terminal_metrics_under_best_epoch
        ),
        "one_step_overoptimization": _overoptimization_event(rollout_snapshots),
        "one_step_rollout_objective_divergence_after_selection": recurrence_divergence,
        "curve": snapshots,
        "run_contract_config_digest": str(run_contract["config_digest"]),
        "normalization_digest": str(summary["normalization_digest"]),
    }


def _ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {field: float(left[field]) / float(right[field]) for field in METRIC_FIELDS}


def _paired_comparisons(
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    schedule_effect = {}
    for architecture in ("pcno", "pcfno"):
        stretched = cells[(architecture, "stretched")]["selected"]
        prefix = cells[(architecture, "prefix_tail")]["selected"]
        schedule_effect[architecture] = {
            "numerator": "stretched",
            "denominator": "prefix_tail",
            "selected_metric_ratios": _ratio(stretched, prefix),
        }
    architecture_effect = {}
    for schedule in ("prefix_tail", "stretched"):
        pcno = cells[("pcno", schedule)]["selected"]
        pcfno = cells[("pcfno", schedule)]["selected"]
        architecture_effect[schedule] = {
            "numerator": "pcno",
            "denominator": "pcfno",
            "selected_metric_ratios": _ratio(pcno, pcfno),
        }
    return {
        "schedule_effect_stretched_over_prefix_tail": schedule_effect,
        "architecture_effect_pcno_over_pcfno": architecture_effect,
    }


def analyze_gate(
    gate_root: Path,
    *,
    expected_source_set_digest: str = EXPECTED_SOURCE_SET_DIGEST,
    expected_split_partition_digest: str = EXPECTED_SPLIT_PARTITION_DIGEST,
) -> dict[str, Any]:
    run_dirs = _discover_run_dirs(gate_root)
    if len(run_dirs) != 4:
        raise ValueError(f"expected four schedule-gate cells, found {len(run_dirs)}")
    cell_list = [
        _validate_cell(
            run_dir,
            expected_source_set_digest=expected_source_set_digest,
            expected_split_partition_digest=expected_split_partition_digest,
        )
        for run_dir in run_dirs
    ]
    cells = {(cell["architecture"], cell["schedule"]): cell for cell in cell_list}
    if set(cells) != set(CELL_ORDER) or len(cells) != len(cell_list):
        raise ValueError(
            "schedule gate does not contain the required architecture cross-product"
        )
    ordered = [cells[key] for key in CELL_ORDER]
    return {
        "schema": SCHEMA,
        "status": "validated_complete",
        "scientific_scope": "single_seed_n256_schedule_routing_evidence",
        "historical_test_population_accessed": False,
        "source_set_digest": expected_source_set_digest,
        "split_partition_digest": expected_split_partition_digest,
        "metric_semantics": {
            "online_train_one_step_relative_l2": (
                "changing-parameter online optimization trace; not a fixed-bank estimate"
            ),
            "fixed_seen_train_one_step_relative_l2": (
                "post-epoch frozen seen-trajectory pair bank"
            ),
            "fixed_validation_one_step_relative_l2": (
                "post-epoch frozen 44-trajectory open-validation pair bank"
            ),
            "rollout_all_call_mean_relative_l2": (
                "autonomous 79-call mean over 16 frozen selection trajectories"
            ),
            "rollout_h79_relative_l2": (
                "autonomous H79 endpoint over the same 16 selection trajectories"
            ),
        },
        "cells": ordered,
        "paired_comparisons": _paired_comparisons(cells),
        "provisional_schedule": "stretched",
        "provisional_schedule_basis": (
            "lower selected all-call rollout error for both architectures; outside-"
            "selection 28-trajectory audit is still required before ladder expansion"
        ),
        "outside_selection_audit_required": True,
        "claims_not_supported": [
            "data scaling",
            "architecture superiority independent of schedule",
            "classical overfitting onset",
            "causal mechanism for one-step versus rollout divergence",
        ],
    }


def _curve_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in analysis["cells"]:
        selected_epoch = int(cell["selected"]["epoch"])
        for snapshot in cell["curve"]:
            rows.append(
                {
                    "run": cell["run"],
                    "architecture": cell["architecture"],
                    "schedule": cell["schedule"],
                    **snapshot,
                    "is_selected_checkpoint": int(snapshot["epoch"]) == selected_epoch,
                }
            )
    return rows


def _cell_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in analysis["cells"]:
        selected = cell["selected"]
        terminal = cell["terminal"]
        row = {
            "run": cell["run"],
            "architecture": cell["architecture"],
            "schedule": cell["schedule"],
            "selected_epoch": selected["epoch"],
            "selected_optimizer_step": selected["optimizer_step"],
            "terminal_epoch": terminal["epoch"],
            "terminal_optimizer_step": terminal["optimizer_step"],
            "one_step_overoptimization_detected": cell["one_step_overoptimization"][
                "detected"
            ],
            "one_step_rollout_objective_divergence_after_selection": cell[
                "one_step_rollout_objective_divergence_after_selection"
            ],
        }
        for scope, snapshot in (("selected", selected), ("terminal", terminal)):
            for field in METRIC_FIELDS:
                row[f"{scope}_{field}"] = snapshot[field]
        rows.append(row)
    return rows


def _plot_training_dynamics(
    analysis: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "lines.linewidth": 1.7,
        }
    )
    colors = {
        ("pcno", "prefix_tail"): "#56B4E9",
        ("pcno", "stretched"): "#0072B2",
        ("pcfno", "prefix_tail"): "#E69F00",
        ("pcfno", "stretched"): "#D55E00",
    }
    labels = {
        ("pcno", "prefix_tail"): "PCNO prefix-tail",
        ("pcno", "stretched"): "PCNO stretched",
        ("pcfno", "prefix_tail"): "PCFNO prefix-tail",
        ("pcfno", "stretched"): "PCFNO stretched",
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0), sharex=True)
    online_ax, fixed_ax, rollout_ax, endpoint_ax = axes.flat
    for cell in analysis["cells"]:
        key = (cell["architecture"], cell["schedule"])
        curve = cell["curve"]
        steps = [row["optimizer_step"] for row in curve]
        color = colors[key]
        online_ax.plot(
            steps,
            [row["online_train_one_step_relative_l2"] for row in curve],
            color=color,
            label=labels[key],
        )
        fixed_ax.plot(
            steps,
            [row["fixed_validation_one_step_relative_l2"] for row in curve],
            color=color,
        )
        fixed_ax.plot(
            steps,
            [row["fixed_seen_train_one_step_relative_l2"] for row in curve],
            color=color,
            linestyle=":",
            alpha=0.85,
        )
        rollout_rows = [
            row for row in curve if row["rollout_all_call_mean_relative_l2"] is not None
        ]
        rollout_steps = [row["optimizer_step"] for row in rollout_rows]
        rollout_ax.plot(
            rollout_steps,
            [row["rollout_all_call_mean_relative_l2"] for row in rollout_rows],
            color=color,
            marker="o",
            markersize=3,
        )
        endpoint_ax.plot(
            rollout_steps,
            [row["rollout_h79_relative_l2"] for row in rollout_rows],
            color=color,
            marker="o",
            markersize=3,
        )
        selected = cell["selected"]
        for axis, field in (
            (online_ax, "online_train_one_step_relative_l2"),
            (fixed_ax, "fixed_validation_one_step_relative_l2"),
            (rollout_ax, "rollout_all_call_mean_relative_l2"),
            (endpoint_ax, "rollout_h79_relative_l2"),
        ):
            axis.scatter(
                [selected["optimizer_step"]],
                [selected[field]],
                color=color,
                marker="*",
                s=48,
                edgecolor="black",
                linewidth=0.35,
                zorder=5,
            )
    titles = (
        "Online one-step train (changing parameters)",
        "Fixed one-step banks (solid: validation; dotted: seen)",
        "Autonomous rollout: all-call mean",
        "Autonomous rollout: H79 endpoint",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_yscale("log")
        axis.axvline(5_120, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Relative L2 (lower is better)")
    for axis in axes[1]:
        axis.set_xlabel("Optimizer steps")
    online_ax.legend(ncol=2, loc="upper right")
    fig.suptitle("D094 n=256 schedule gate, seed 20260718", fontsize=11)
    fig.tight_layout()
    pdf = output_dir / "d094_schedule_gate_training_dynamics.pdf"
    png = output_dir / "d094_schedule_gate_training_dynamics.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def write_analysis(
    analysis: Mapping[str, Any], output_dir: Path, *, plots: bool
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("analysis output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis.json"
    curve_path = output_dir / "training_curves.csv"
    cell_path = output_dir / "selected_and_terminal_metrics.csv"
    _write_json(analysis_path, analysis)
    _write_csv(curve_path, _curve_rows(analysis))
    _write_csv(cell_path, _cell_rows(analysis))
    generated = [analysis_path, curve_path, cell_path]
    if plots:
        generated.extend(_plot_training_dynamics(analysis, output_dir))
    manifest = {
        "schema": "d094_bump_schedule_gate_analysis_artifacts_v1",
        "historical_test_population_accessed": False,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in generated
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_gate(
        args.gate_root,
        expected_source_set_digest=args.expected_source_set_digest,
        expected_split_partition_digest=args.expected_split_partition_digest,
    )
    manifest = write_analysis(analysis, args.output_dir, plots=not args.skip_plots)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
