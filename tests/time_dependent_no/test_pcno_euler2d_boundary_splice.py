from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_splice import (
    BOUNDARY_FROM_TEACHER,
    NORMAL_FROM_TEACHER,
    BoundarySplicePCNO,
    boundary_h20_advance_gate,
    gain_fraction,
    paired_structure_ratios,
    splice_raw_predictions,
    verified_normalization_digest,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol import (
    STRUCTURE_ERROR_FIELDS,
)


def test_splice_raw_predictions_exchanges_only_selected_node_population() -> None:
    parent = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    teacher = parent + 100.0
    node_type = torch.tensor([[0, 1, 0, 2, 3, 0]])
    node_mask = torch.tensor([[[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]])

    boundary_from_teacher = splice_raw_predictions(
        parent,
        teacher,
        node_type,
        node_mask,
        variant=BOUNDARY_FROM_TEACHER,
    )
    normal_from_teacher = splice_raw_predictions(
        parent,
        teacher,
        node_type,
        node_mask,
        variant=NORMAL_FROM_TEACHER,
    )

    assert torch.equal(boundary_from_teacher[0, 0], parent[0, 0])
    assert torch.equal(boundary_from_teacher[0, 1], teacher[0, 1])
    assert torch.equal(boundary_from_teacher[0, 2], parent[0, 2])
    assert torch.equal(boundary_from_teacher[0, 3], teacher[0, 3])
    assert torch.equal(normal_from_teacher[0, 0], teacher[0, 0])
    assert torch.equal(normal_from_teacher[0, 1], parent[0, 1])
    assert torch.equal(normal_from_teacher[0, 2], teacher[0, 2])
    assert torch.equal(normal_from_teacher[0, 3], parent[0, 3])
    assert torch.count_nonzero(boundary_from_teacher[0, 5]) == 0
    assert torch.count_nonzero(normal_from_teacher[0, 5]) == 0


def test_splice_raw_predictions_rejects_shape_and_variant_errors() -> None:
    proposal = torch.zeros(1, 2, 4)
    node_type = torch.zeros(1, 2, dtype=torch.long)
    node_mask = torch.ones(1, 2, 1)
    with pytest.raises(ValueError, match="shapes differ"):
        splice_raw_predictions(
            proposal,
            torch.zeros(1, 3, 4),
            node_type,
            node_mask,
            variant=BOUNDARY_FROM_TEACHER,
        )
    with pytest.raises(ValueError, match="unsupported splice variant"):
        splice_raw_predictions(
            proposal,
            proposal,
            node_type,
            node_mask,
            variant="unknown",
        )


class _OffsetModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.gamma = 1.4
        self.register_buffer("state_mean", torch.zeros(4))
        self.register_buffer("state_scale", torch.ones(4))
        self.register_buffer("residual_scale", torch.ones(4))
        self.register_buffer("mach_mean", torch.tensor(0.0))
        self.register_buffer("mach_scale", torch.tensor(1.0))

    def forward(self, current: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return current + self.offset


def test_boundary_splice_model_calls_both_models_on_the_same_state() -> None:
    parent = _OffsetModel(1.0)
    teacher = _OffsetModel(10.0)
    model = BoundarySplicePCNO(parent, teacher, variant=BOUNDARY_FROM_TEACHER)
    current = torch.zeros(1, 3, 4)
    node_mask = torch.ones(1, 3, 1)
    node_type = torch.tensor([[0, 1, 0]])
    filler = torch.zeros(1, 1)

    prediction = model(
        current,
        node_mask=node_mask,
        nodes=filler,
        node_weights=filler,
        node_rhos=filler,
        directed_edges=filler,
        edge_gradient_weights=filler,
        node_type=node_type,
        mach=filler,
    )

    assert torch.equal(prediction[0, 0], torch.ones(4))
    assert torch.equal(prediction[0, 1], torch.full((4,), 10.0))
    assert torch.equal(prediction[0, 2], torch.ones(4))


def _structure_rows(multiplier: float) -> dict[str, object]:
    rows = []
    for trajectory, baseline in (("a", 1.0), ("b", 2.0)):
        row: dict[str, object] = {"trajectory": trajectory, "call_index": 20}
        for field in STRUCTURE_ERROR_FIELDS:
            row[field] = baseline * multiplier
        rows.append(row)
    return {"rows": rows}


def test_paired_structure_ratios_are_trajectory_matched() -> None:
    ratios = paired_structure_ratios(_structure_rows(1.0), _structure_rows(0.8), [20])
    for field in STRUCTURE_ERROR_FIELDS:
        record = ratios["endpoints"]["20"][field]
        assert record["count"] == 2
        assert record["wins"] == 2
        assert record["mean_ratio"] == pytest.approx(0.8)
        assert record["median_ratio"] == pytest.approx(0.8)


def test_gain_fraction_and_h20_gate_enforce_locality_contract() -> None:
    assert gain_fraction(10.0, 8.0, 9.0) == pytest.approx(0.5)
    assert gain_fraction(10.0, 10.0, 9.0) is None
    endpoint = {field: {"median_ratio": 1.0} for field in STRUCTURE_ERROR_FIELDS}
    comparison = {
        "final_checkpoint": 20,
        "endpoint": {
            "all": {"teacher_parent_gain_fraction_retained": 0.6},
            "normal": {"teacher_parent_gain_fraction_retained": 0.7},
        },
        "structure_ratios": {"endpoints": {"20": endpoint}},
    }
    rollout = {
        "num_trajectories": 30,
        "completed": 30,
        "trajectories": [{"minimum_outflow_normal_mach": 1.2} for _ in range(30)],
    }

    assert boundary_h20_advance_gate(comparison, rollout)["passed"]
    failed = deepcopy(comparison)
    failed["structure_ratios"]["endpoints"]["20"]["front_centroid_distance"][
        "median_ratio"
    ] = 1.051
    assert not boundary_h20_advance_gate(failed, rollout)["passed"]


def test_normalization_digest_accepts_legacy_missing_declaration() -> None:
    normalization = {
        "state_mean": [1.0, 2.0, 3.0, 4.0],
        "state_scale": [4.0, 3.0, 2.0, 1.0],
        "residual_scale": [0.1, 0.2, 0.3, 0.4],
        "mach_mean": 3.0,
        "mach_scale": 0.2,
        "mach_scale_floor": 0.1,
        "gamma": 1.4,
        "weight_provenance": "test",
    }
    legacy = {"normalization": normalization}
    digest = verified_normalization_digest(legacy, name="legacy")
    declared = {
        "normalization": normalization,
        "normalization_digest": digest,
    }

    assert verified_normalization_digest(declared, name="declared") == digest
    declared["normalization_digest"] = "0" * 64
    with pytest.raises(ValueError, match="does not match its payload"):
        verified_normalization_digest(declared, name="declared")
