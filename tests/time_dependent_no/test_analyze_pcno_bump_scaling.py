from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.time_dependent_no.analyze_pcno_bump_scaling import (
    CELL_ORDER,
    EXPECTED_ROLLOUT_STEPS,
    _overoptimization_event,
    _selection_tuple,
    analyze_gate,
    build_parser,
    write_analysis,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _metric_row(
    epoch: int,
    *,
    scale: float,
    selected_epoch: int,
    recurrence_divergence: bool,
) -> dict[str, object]:
    step = (epoch + 1) * 256
    one_step = scale / (epoch + 10)
    rollout = None
    if step in EXPECTED_ROLLOUT_STEPS:
        distance = abs(epoch - selected_epoch)
        rollout_value = scale * (0.05 + 0.001 * distance)
        if recurrence_divergence and epoch > selected_epoch:
            rollout_value = scale * (0.05 + 0.002 * (epoch - selected_epoch))
        rollout = {
            "mean_full_horizon_relative_l2": rollout_value,
            "mean_endpoint_relative_l2": {"79": 1.5 * rollout_value},
            "completion_rate": 1.0,
            "mean_survival_fraction": 1.0,
            "hard_failure_count": 0,
            "physical_admissibility_rate": 1.0,
        }
    return {
        "epoch": epoch,
        "learning_rate": 1.0e-3,
        "train": {
            "completed_optimizer_steps": step,
            "relative_l2": 0.9 * one_step,
            "comparable_seen": {"relative_l2": one_step},
        },
        "validation": {"relative_l2": 1.02 * one_step},
        "rollout": rollout,
    }


def _make_cell(
    root: Path,
    architecture: str,
    schedule: str,
    *,
    source_digest: str,
    split_digest: str,
) -> None:
    mode = "full" if architecture == "pcno" else "no_gradient"
    best_epoch = 59 if (architecture, schedule) == ("pcfno", "stretched") else 79
    scale = {
        ("pcno", "prefix_tail"): 1.0,
        ("pcno", "stretched"): 0.55,
        ("pcfno", "prefix_tail"): 0.95,
        ("pcfno", "stretched"): 0.85,
    }[(architecture, schedule)]
    rows = [
        _metric_row(
            epoch,
            scale=scale,
            selected_epoch=best_epoch,
            recurrence_divergence=(architecture, schedule) == ("pcfno", "stretched"),
        )
        for epoch in range(80)
    ]
    snapshots = []
    from scripts.time_dependent_no.analyze_pcno_bump_scaling import _metric_snapshot

    snapshots = [_metric_snapshot(row) for row in rows]
    selected = snapshots[best_epoch]
    terminal = snapshots[-1]
    name = f"gate_n256_{architecture}_{schedule}_synthetic"
    run = root / "outputs" / name
    run.mkdir(parents=True)
    _write_json(
        run / "summary.json",
        {
            "actual_optimizer_steps": 20_480,
            "completed_epochs": 80,
            "train_trajectories": 256,
            "validation_trajectories": 44,
            "test_trajectories": 0,
            "best_epoch": best_epoch,
            "best_selection": _selection_tuple(selected),
            "optimizer": {
                "scheduler_contract": {
                    "decay_optimizer_steps": (
                        5_120 if schedule == "prefix_tail" else 20_480
                    )
                }
            },
            "artifact_sha256": {"best_checkpoint": "a" * 64},
            "normalization_digest": "b" * 64,
        },
    )
    _write_json(
        run / "bump_scaling_contract.json",
        {
            "differential_branch_mode": mode,
            "historical_test_population_accessed": False,
            "rollout_failure_policy": "finite_only",
            "checkpoint_selection_mode": "full_horizon_error_first",
            "rollout_selection_trajectory_count": 16,
            "split_partition_digest": split_digest,
        },
    )
    _write_json(run / "run_contract.json", {"config_digest": "c" * 64})
    _write_json(
        run / "validation_receipt.json",
        {
            "best_epoch": best_epoch,
            **{
                field: terminal[field]
                for field in (
                    "online_train_one_step_relative_l2",
                    "fixed_seen_train_one_step_relative_l2",
                    "fixed_validation_one_step_relative_l2",
                    "rollout_all_call_mean_relative_l2",
                    "rollout_h79_relative_l2",
                )
            },
        },
    )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    source = run / "source_snapshot" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_json(
        run / "source_snapshot" / "manifest.json",
        {
            "schema": "pcno_euler2d_source_snapshot_v6",
            "source_set_digest": source_digest,
            "files": {"model.py": {"bytes": source.stat().st_size, "sha256": digest}},
            "provenance_files": {},
        },
    )


@pytest.fixture
def synthetic_gate(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "gate"
    root.mkdir()
    (root / "gate.exit").write_text("0\n", encoding="utf-8")
    (root / "gate.completed").write_text("done\n", encoding="utf-8")
    source_digest = "1" * 64
    split_digest = "2" * 64
    for architecture, schedule in CELL_ORDER:
        _make_cell(
            root,
            architecture,
            schedule,
            source_digest=source_digest,
            split_digest=split_digest,
        )
    return root, source_digest, split_digest


def test_analysis_uses_best_epoch_row_and_flags_terminal_receipt(
    synthetic_gate: tuple[Path, str, str],
) -> None:
    root, source_digest, split_digest = synthetic_gate

    analysis = analyze_gate(
        root,
        expected_source_set_digest=source_digest,
        expected_split_partition_digest=split_digest,
    )

    cells = {
        (cell["architecture"], cell["schedule"]): cell for cell in analysis["cells"]
    }
    divergent = cells[("pcfno", "stretched")]
    assert divergent["selected"]["epoch"] == 59
    assert divergent["terminal"]["epoch"] == 79
    assert (
        divergent["selected"]["rollout_all_call_mean_relative_l2"]
        < divergent["terminal"]["rollout_all_call_mean_relative_l2"]
    )
    assert divergent["receipt_has_terminal_metrics_under_best_epoch"] is True
    assert set(divergent["receipt_metric_scope"].values()) == {"terminal"}
    assert divergent["one_step_rollout_objective_divergence_after_selection"] is True
    assert divergent["one_step_overoptimization"]["detected"] is False
    assert analysis["historical_test_population_accessed"] is False


def test_overoptimization_rule_requires_three_consecutive_checkpoints() -> None:
    snapshots = [
        {
            "optimizer_step": 1,
            "fixed_validation_one_step_relative_l2": 1.0,
            "fixed_seen_train_one_step_relative_l2": 1.0,
        },
        {
            "optimizer_step": 2,
            "fixed_validation_one_step_relative_l2": 1.3,
            "fixed_seen_train_one_step_relative_l2": 0.9,
        },
        {
            "optimizer_step": 3,
            "fixed_validation_one_step_relative_l2": 1.4,
            "fixed_seen_train_one_step_relative_l2": 0.85,
        },
        {
            "optimizer_step": 4,
            "fixed_validation_one_step_relative_l2": 1.5,
            "fixed_seen_train_one_step_relative_l2": 0.8,
        },
    ]

    event = _overoptimization_event(snapshots)

    assert event["detected"] is True
    assert event["first_three_checkpoint_event_optimizer_step"] == 4


def test_analysis_outputs_are_manifested_without_plots(
    synthetic_gate: tuple[Path, str, str], tmp_path: Path
) -> None:
    root, source_digest, split_digest = synthetic_gate
    analysis = analyze_gate(
        root,
        expected_source_set_digest=source_digest,
        expected_split_partition_digest=split_digest,
    )

    manifest = write_analysis(analysis, tmp_path / "analysis", plots=False)

    assert set(manifest["files"]) == {
        "analysis.json",
        "selected_and_terminal_metrics.csv",
        "training_curves.csv",
    }
    assert manifest["historical_test_population_accessed"] is False


def test_parser_has_no_historical_test_surface() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "test" not in destinations
    assert "test_root" not in destinations
    assert {"gate_root", "output_dir", "skip_plots"} <= destinations
