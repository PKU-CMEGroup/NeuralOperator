from __future__ import annotations

import pytest

from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_SPLIT_PARTITION_DIGEST,
    EXPECTED_TERMINAL_STEP,
    TRAINING_CONTRACT_SCHEMA,
    _checkpoint_metric_anchor,
    _validate_training_contract,
    select_diagnostic_keys,
)


def _metric_row(step: int) -> dict:
    return {
        "epoch": step // 256 - 1,
        "learning_rate": 1.0e-3,
        "train": {
            "completed_optimizer_steps": step,
            "relative_l2": 0.1,
            "comparable_seen": {"relative_l2": 0.2},
        },
        "validation": {"relative_l2": 0.3},
        "rollout": {
            "mean_full_horizon_relative_l2": 0.4,
            "mean_endpoint_relative_l2": {"79": 0.5},
            "completion_rate": 1.0,
            "hard_failure_count": 0,
            "physical_admissibility_rate": 0.75,
        },
    }


def _selected_result(errors: dict[str, float]) -> dict:
    return {
        "outside_selection_rollout": {
            "trajectories": [
                {"trajectory": key, "final_relative_l2": value}
                for key, value in errors.items()
            ]
        }
    }


def test_checkpoint_metric_anchor_preserves_distinct_metric_scopes() -> None:
    snapshot = _checkpoint_metric_anchor([_metric_row(8_192)], 8_192)
    assert snapshot["online_train_one_step_relative_l2"] == pytest.approx(0.1)
    assert snapshot["fixed_seen_train_one_step_relative_l2"] == pytest.approx(0.2)
    assert snapshot["fixed_validation_one_step_relative_l2"] == pytest.approx(0.3)
    assert snapshot["rollout_h79_relative_l2"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="does not identify one"):
        _checkpoint_metric_anchor([_metric_row(8_192)], 20_480)


def test_b1_c4_contract_requires_exact_registered_budget() -> None:
    contract = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "status": "completed_unreplicated_pilot",
        "registered_stage": "b1_c4_pcno_40960",
        "trajectory_count": 128,
        "differential_branch_mode": "full",
        "schedule_arm": "b1_c4_cold_stretched",
        "requested_optimizer_steps": EXPECTED_TERMINAL_STEP,
        "actual_optimizer_steps": EXPECTED_TERMINAL_STEP,
        "requested_epochs": 160,
        "completed_epochs": 160,
        "optimizer_steps_per_epoch": 256,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": "finite_only",
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "historical_test_population_accessed": False,
        "automatic_continuation_authorized": False,
        "source_manifest_sha256": EXPECTED_DATA_MANIFEST_DIGEST,
        "split_partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
        "sentinel_steps": [8_192, 20_480],
    }
    _validate_training_contract(contract, trajectory_count=128)
    contract["actual_optimizer_steps"] = 20_480
    with pytest.raises(ValueError, match="actual_optimizer_steps"):
        _validate_training_contract(contract, trajectory_count=128)


def test_diagnostic_cases_use_fixed_first_and_largest_nonfirst_gap() -> None:
    keys = ["fixed", "small_gap", "large_gap", "medium_gap"]
    selected = {
        128: _selected_result(
            {"fixed": 9.0, "small_gap": 0.2, "large_gap": 0.3, "medium_gap": 0.4}
        ),
        256: _selected_result(
            {"fixed": 1.0, "small_gap": 0.1, "large_gap": 0.9, "medium_gap": 0.7}
        ),
    }
    result = select_diagnostic_keys(keys, selected)
    assert result["checkpoint_reselection"] is False
    assert [row["trajectory"] for row in result["records"]] == [
        "fixed",
        "large_gap",
    ]
    assert result["records"][1]["absolute_h79_difference"] == pytest.approx(0.6)
