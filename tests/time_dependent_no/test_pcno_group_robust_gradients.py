from __future__ import annotations

import pytest
import torch

from scripts.time_dependent_no.diagnose_pcno_euler2d_group_robust_gradients import (
    SCHEMA,
    TRAIN_Y_INDICES,
    group_robust_selector,
    minimum_norm_frank_wolfe,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_joint_objective_gradients import (
    OBJECTIVES,
    VALIDATION_CALLS,
    VALIDATION_CASES,
)


def test_minimum_norm_frank_wolfe_balances_orthogonal_tasks() -> None:
    direction, report = minimum_norm_frank_wolfe(
        {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([0.0, 2.0]),
        }
    )

    torch.testing.assert_close(direction, torch.tensor([0.5, 0.5]), atol=1e-6, rtol=0.0)
    assert report["coefficients"]["a"] == pytest.approx(0.5, abs=1e-8)
    assert report["coefficients"]["b"] == pytest.approx(0.5, abs=1e-8)
    assert report["frank_wolfe_gap"] <= 1.0e-12
    assert report["task_directional_cosines"]["a"] == pytest.approx(2.0**-0.5)


def _mgda(cosine: float = 0.2) -> dict[str, object]:
    names = [f"state_recurrence_y{index:02d}" for index in TRAIN_Y_INDICES]
    names.append("smooth_highpass_global")
    return {
        "direction_norm": 0.5,
        "frank_wolfe_gap": 1.0e-8,
        "task_directional_cosines": {name: cosine for name in names},
    }


def _rows(cosine: float = 0.2) -> list[dict[str, object]]:
    return [
        {
            "schema": SCHEMA,
            "trajectory": trajectory,
            "call_index": call,
            "objective_losses": {name: 1.0 for name in OBJECTIVES},
            "gradient_norms": {name: 1.0 for name in OBJECTIVES},
            "directional_cosines": {name: cosine for name in OBJECTIVES},
            "smooth_physical_mass_fraction": 0.8,
            "generated_input_admissibility": {"all_admissible": True},
        }
        for call in VALIDATION_CALLS
        for trajectory in VALIDATION_CASES
    ]


def test_group_robust_selector_requires_training_tasks_and_repeated_transfer() -> None:
    ledger = {
        "optimizer_created": False,
        "optimizer_steps": 0,
        "parameter_state_unchanged": True,
        "test_trajectory_access": [],
    }
    passed = group_robust_selector(_mgda(), _rows(), ledger)
    assert passed["contract_complete"] is True
    assert passed["promotion_passed"] is True
    assert passed["classification"] == "geometry_group_common_descent_candidate"

    rows = _rows()
    for row in rows:
        if row["call_index"] == 60 and row["trajectory"] in VALIDATION_CASES[:2]:
            row["directional_cosines"]["clean_state"] = -0.1
    failed = group_robust_selector(_mgda(), rows, ledger)
    assert failed["contract_complete"] is True
    assert failed["validation_calls"]["60"]["joint_positive_case_count"] == 4
    assert failed["promotion_passed"] is False
    assert failed["route"] == "stop_joint_objective_route_no_training"
