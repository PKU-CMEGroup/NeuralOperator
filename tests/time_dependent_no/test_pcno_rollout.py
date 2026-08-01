from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from utility.time_dependent_no.pcno_rollout import (
    contract_forward_sample,
    failure_cause,
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
