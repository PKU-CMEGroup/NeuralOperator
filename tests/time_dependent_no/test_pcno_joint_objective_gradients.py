from __future__ import annotations

import pytest
import torch

from scripts.time_dependent_no.diagnose_pcno_euler2d_joint_objective_gradients import (
    OBJECTIVES,
    SCHEMA,
    TRAIN_COSINE_MIN,
    VALIDATION_CALLS,
    VALIDATION_CASES,
    joint_gradient_selector,
    normalized_gradient_sum,
    smooth_highpass_error_mse,
)
from utility.time_dependent_no.pcno_euler2d import (
    primitive_to_conservative_torch,
)


def test_smooth_highpass_error_mse_is_differentiable_and_masks_shock() -> None:
    batch_size = 2
    nodes = 9
    undirected = torch.stack(
        (torch.arange(nodes - 1), torch.arange(1, nodes)), dim=-1
    )
    directed = torch.cat((undirected, undirected.flip(-1)), dim=0)
    directed = directed.unsqueeze(0).expand(batch_size, -1, -1)
    node_mask = torch.ones((batch_size, nodes, 1))
    node_weights = torch.full((batch_size, nodes, 1), 1.0 / nodes)
    node_type = torch.zeros((batch_size, nodes), dtype=torch.int64)
    primitive = torch.zeros((batch_size, nodes, 4))
    primitive[..., 0] = 1.0
    primitive[..., 3] = 1.0
    primitive[:, 4:, 3] = 2.0
    target = primitive_to_conservative_torch(primitive)
    prediction = target.detach().clone()
    prediction[..., 0] += 0.01 * ((-1.0) ** torch.arange(nodes))
    prediction.requires_grad_()

    loss, mass_fraction = smooth_highpass_error_mse(
        prediction,
        target,
        target,
        directed_edges=directed,
        node_weights=node_weights,
        node_mask=node_mask,
        node_type=node_type,
        component_scale=torch.ones(4),
        gamma=1.4,
    )

    assert float(loss.detach()) > 0.0
    assert 0.0 < float(mass_fraction.detach()) < 1.0
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_normalized_gradient_sum_reports_common_descent_geometry() -> None:
    gradients = {
        "clean_state": torch.tensor([1.0, 0.0, 0.0]),
        "smooth_highpass": torch.tensor([0.0, 2.0, 0.0]),
        "generated_state": torch.tensor([1.0, 1.0, 0.0]),
    }
    direction, report = normalized_gradient_sum(gradients)

    assert direction.shape == (3,)
    assert report["pairwise_cosines"]["clean_state"]["smooth_highpass"] == pytest.approx(0.0)
    assert set(report["directional_cosines"]) == set(OBJECTIVES)
    assert min(report["directional_cosines"].values()) > TRAIN_COSINE_MIN
    assert report["gradient_norms"]["smooth_highpass"] == pytest.approx(2.0)


def _selector_rows(cosine: float) -> list[dict[str, object]]:
    return [
        {
            "schema": SCHEMA,
            "trajectory": trajectory,
            "call_index": call,
            "objective_losses": {name: 1.0 for name in OBJECTIVES},
            "gradient_norms": {name: 1.0 for name in OBJECTIVES},
            "directional_cosines": {name: cosine for name in OBJECTIVES},
            "smooth_physical_mass_fraction": 0.7,
            "generated_input_admissibility": {"all_admissible": True},
        }
        for call in VALIDATION_CALLS
        for trajectory in VALIDATION_CASES
    ]


def test_joint_gradient_selector_requires_repeated_three_objective_transfer() -> None:
    training = {
        "all_finite": True,
        "directional_cosines": {name: 0.3 for name in OBJECTIVES},
    }
    ledger = {
        "optimizer_created": False,
        "optimizer_steps": 0,
        "parameter_state_unchanged": True,
        "test_trajectory_access": [],
    }
    passed = joint_gradient_selector(training, _selector_rows(0.2), ledger)
    assert passed["contract_complete"] is True
    assert passed["promotion_passed"] is True
    assert passed["classification"] == "joint_objective_local_descent_compatible"

    rows = _selector_rows(0.2)
    for row in rows:
        if row["call_index"] == 60 and row["trajectory"] in VALIDATION_CASES[:2]:
            row["directional_cosines"]["generated_state"] = -0.1
    failed = joint_gradient_selector(training, rows, ledger)
    assert failed["contract_complete"] is True
    assert failed["validation_calls"]["30"]["passed"] is True
    assert failed["validation_calls"]["60"]["joint_positive_case_count"] == 4
    assert failed["promotion_passed"] is False
    assert failed["route"] == "reject_equal_gradient_continuation_no_training"
