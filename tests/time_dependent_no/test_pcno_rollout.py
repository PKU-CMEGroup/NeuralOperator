from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from utility.time_dependent_no.pcno_rollout import (
    FINITE_ONLY_ROLLOUT_POLICY,
    contract_forward_sample,
    evaluate_rollouts,
    failure_cause,
    rollout_trajectory,
)

ROOT = Path(__file__).resolve().parents[2]
RETAINED_BUMP_MODULES = {
    "scripts.time_dependent_no.train_pcno_euler2d_residual",
    "scripts.time_dependent_no.evaluate_pcno_euler2d_residual",
    "scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol",
    "scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_splice",
    "scripts.time_dependent_no.decompose_pcno_euler2d_rollout_error",
}


class _DifferentiableStep(torch.nn.Module):
    gamma = 1.4
    model_node_type_input = "physical"

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.01))

    def forward(self, current: torch.Tensor, **_kwargs: torch.Tensor) -> torch.Tensor:
        return self.scale * current


class _OneNodeStore:
    def __init__(self, steps: int = 4) -> None:
        state = np.array([1.0, 0.0, 0.0, 2.5], dtype=np.float32)
        self._states = np.repeat(state[None, None, :], steps, axis=0)

    def states(self, _key: str) -> np.ndarray:
        return self._states

    def tensor_sample(
        self, _key: str, frame: int, *, step_stride: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        del step_stride
        return {
            "current": torch.as_tensor(
                self._states[frame], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "node_mask": torch.ones(1, 1, 1, device=device),
            "nodes": torch.zeros(1, 1, 2, device=device),
            "node_weights": torch.ones(1, 1, 1, device=device),
            "node_rhos": torch.ones(1, 1, 1, device=device),
            "directed_edges": torch.empty(1, 0, 2, dtype=torch.long, device=device),
            "edge_gradient_weights": torch.empty(1, 0, 2, device=device),
            "node_type": torch.zeros(1, 1, dtype=torch.long, device=device),
            "mach": torch.ones(1, 1, device=device),
        }


class _NonpositiveDensityStep(torch.nn.Module):
    gamma = 1.4
    model_node_type_input = "physical"

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state_scale", torch.ones(4))

    def forward(self, current: torch.Tensor, **_kwargs: torch.Tensor) -> torch.Tensor:
        proposal = current.clone()
        proposal[..., 0] = -1.0
        return proposal


class _NonfiniteStep(_NonpositiveDensityStep):
    def forward(self, current: torch.Tensor, **_kwargs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(current, float("nan"))


def test_contract_forward_sample_remains_differentiable() -> None:
    model = _DifferentiableStep()
    current = torch.tensor([[[1.0, 0.0, 0.0, 2.5]]])
    sample = {
        "node_mask": torch.ones(1, 1, 1),
        "nodes": torch.zeros(1, 1, 2),
        "node_weights": torch.ones(1, 1, 1),
        "node_rhos": torch.ones(1, 1, 1),
        "directed_edges": torch.empty(1, 0, 2, dtype=torch.long),
        "edge_gradient_weights": torch.empty(1, 0, 2),
        "node_type": torch.zeros(1, 1, dtype=torch.long),
        "mach": torch.ones(1, 1),
    }

    prediction, raw_prediction, model_current = contract_forward_sample(
        model,
        sample,
        current,
        boundary_policy=None,
    )
    prediction.sum().backward()

    assert torch.equal(prediction, raw_prediction)
    assert model_current is current
    assert model.scale.grad is not None
    assert float(model.scale.grad) == pytest.approx(float(current.sum()))


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ([1.0, 0.0, 0.0, float("nan")], "nonfinite_state"),
        ([-1.0, 0.0, 0.0, 2.5], "nonpositive_density"),
        ([1.0, 2.0, 0.0, 1.0], "nonpositive_internal_energy"),
        ([1.0, 0.0, 0.0, 2.5], None),
    ],
)
def test_failure_cause_preserves_admissibility_precedence(
    state: list[float], expected: str | None
) -> None:
    failure, summary = failure_cause(
        torch.tensor([[state]], dtype=torch.float32),
        gamma=1.4,
    )

    assert failure == expected
    assert set(summary) == {"min_density", "min_internal_energy", "min_pressure"}


def test_finite_only_rollout_scores_physically_invalid_states_at_full_horizon() -> None:
    store = _OneNodeStore()
    strict = rollout_trajectory(
        _NonpositiveDensityStep(),
        store,
        "case",
        step_stride=1,
        start_frame=0,
        num_steps=3,
        device=torch.device("cpu"),
        amp="none",
        rollout_checkpoints=(3,),
    )
    assert strict["valid_length"] == 0
    assert strict["failure_cause"] == "nonpositive_density"

    summary = evaluate_rollouts(
        _NonpositiveDensityStep(),
        store,
        ["case"],
        step_stride=1,
        start_frame=0,
        num_steps=3,
        device=torch.device("cpu"),
        amp="none",
        rollout_checkpoints=(3,),
        failure_policy=FINITE_ONLY_ROLLOUT_POLICY,
    )
    row = summary["trajectories"][0]
    assert row["completed"]
    assert row["first_physical_violation"] == {
        "call": 1,
        "cause": "nonpositive_density",
    }
    assert row["physical_violation_counts"] == {"nonpositive_density": 3}
    assert row["first_physical_violation_by_cause"] == {
        "nonpositive_density": 1
    }
    assert not row["physically_admissible"]
    assert row["hard_failure_cause"] is None
    assert summary["completion_rate"] == 1.0
    assert summary["physical_admissibility_rate"] == 0.0
    assert summary["mean_full_horizon_relative_l2"] is not None
    assert summary["mean_endpoint_relative_l2"]["3"] is not None


def test_finite_only_rollout_still_stops_on_nonfinite_state() -> None:
    summary = evaluate_rollouts(
        _NonfiniteStep(),
        _OneNodeStore(),
        ["case"],
        step_stride=1,
        start_frame=0,
        num_steps=3,
        device=torch.device("cpu"),
        amp="none",
        rollout_checkpoints=(3,),
        failure_policy=FINITE_ONLY_ROLLOUT_POLICY,
    )
    row = summary["trajectories"][0]
    assert not row["completed"]
    assert row["hard_failure_cause"] == "nonfinite_state"
    assert row["first_physical_violation"] is None
    assert summary["hard_failure_count"] == 1
    assert summary["mean_full_horizon_relative_l2"] is None


def test_retained_bump_scripts_do_not_import_one_another() -> None:
    violations: list[tuple[str, str]] = []
    for module in sorted(RETAINED_BUMP_MODULES):
        path = ROOT / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module in RETAINED_BUMP_MODULES
            ):
                violations.append((module, str(node.module)))
            elif isinstance(node, ast.Import):
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name in RETAINED_BUMP_MODULES
                )

    assert violations == []
