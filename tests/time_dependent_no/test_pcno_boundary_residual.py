from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    NO_TYPE_CHANNEL_CONTROL,
    build_model,
    parse_args,
    validate_args,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_FIELD_SEMANTIC_COLLAR,
    BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
    NODE_TYPE_FEATURE_OMITTED,
    Euler2DNormalization,
    PCNOEuler2DResidual,
    copy_no_boundary_initialization_to_boundary_residual_model,
)
from utility.time_dependent_no.pcno_runtime import build_checkpoint_model

SEMANTIC_NAMES = ("wall", "outflow", "inflow")


def _unit_normalization() -> Euler2DNormalization:
    return Euler2DNormalization(
        state_mean=np.zeros(4),
        state_scale=np.ones(4),
        residual_scale=np.ones(4),
        mach_mean=0.0,
        mach_scale=1.0,
    )


def _model_common() -> dict[str, object]:
    return {
        "normalization": _unit_normalization(),
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "zero_initialize": False,
        "node_type_feature_mode": NODE_TYPE_FEATURE_OMITTED,
    }


def _sample_fields() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current = torch.tensor([[[1.0, 0.2, 0.1, 2.6], [0.9, 0.3, -0.1, 2.4]]])
    nodes = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    fields = torch.tensor([[[0.7, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    return current, nodes, fields


def _empty_gradient_geometry() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((1, 0, 2), dtype=torch.long),
        torch.empty((1, 0, 2), dtype=torch.float32),
    )


def test_boundary_residual_initialization_exactly_preserves_no_type_pcno() -> None:
    torch.manual_seed(29)
    no_boundary = PCNOEuler2DResidual(**_model_common())
    residual = PCNOEuler2DResidual(
        **_model_common(),
        boundary_residual_mode=BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
        boundary_residual_names=SEMANTIC_NAMES,
        boundary_residual_width=4,
    )
    audit = copy_no_boundary_initialization_to_boundary_residual_model(
        no_boundary, residual
    )

    assert audit["mathematical_initial_function_match"] is True
    assert audit["shared_pcno_lift_receives_boundary_features"] is False
    assert no_boundary.backbone.in_dim == residual.backbone.in_dim == 8
    assert no_boundary.input_feature_names() == residual.input_feature_names()
    source_state = no_boundary.state_dict()
    target_state = residual.state_dict()
    for name, value in source_state.items():
        assert torch.equal(value, target_state[name]), name

    current, nodes, fields = _sample_fields()
    directed_edges, edge_gradient_weights = _empty_gradient_geometry()
    correction = residual.normalized_boundary_residual(
        current,
        mach=torch.tensor([1.1]),
        boundary_features=fields,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
    )
    assert torch.count_nonzero(correction).item() == 0
    side_features = residual.boundary_residual_input_feature_names()
    assert len(side_features) == 14
    assert not any("coordinate" in name for name in side_features)

    geometry = {
        "node_mask": torch.ones((1, 2, 1)),
        "nodes": nodes,
        "node_weights": torch.full((1, 2, 1), 0.5),
        "node_rhos": torch.full((1, 2, 1), 0.5),
        "directed_edges": directed_edges,
        "edge_gradient_weights": edge_gradient_weights,
        "node_type": torch.zeros((1, 2), dtype=torch.int64),
        "mach": torch.tensor([1.1]),
    }
    with torch.no_grad():
        source_prediction = no_boundary(current, **geometry)
        residual_prediction = residual(current, **geometry, boundary_features=fields)
    assert torch.equal(source_prediction, residual_prediction)


def test_boundary_residual_is_semantic_and_exactly_zero_outside_collar() -> None:
    model = PCNOEuler2DResidual(
        **_model_common(),
        boundary_residual_mode=BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
        boundary_residual_names=SEMANTIC_NAMES,
        boundary_residual_width=1,
    )
    assert model.boundary_residual_input is not None
    assert model.boundary_residual_hidden is not None
    assert model.boundary_residual_output is not None
    with torch.no_grad():
        model.boundary_residual_input.weight.zero_()
        model.boundary_residual_input.bias.zero_()
        model.boundary_residual_input.weight[0, 5] = 1.0
        model.boundary_residual_hidden.weight.fill_(1.0)
        model.boundary_residual_hidden.bias.zero_()
        model.boundary_residual_output.weight.fill_(1.0)
        model.boundary_residual_output.bias.zero_()

    current, _, wall_fields = _sample_fields()
    directed_edges, edge_gradient_weights = _empty_gradient_geometry()
    outflow_fields = wall_fields.roll(shifts=1, dims=-1)
    wall = model.normalized_boundary_residual(
        current,
        mach=torch.tensor([1.1]),
        boundary_features=wall_fields,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
    )
    outflow = model.normalized_boundary_residual(
        current,
        mach=torch.tensor([1.1]),
        boundary_features=outflow_fields,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
    )
    assert torch.count_nonzero(wall[:, 0]).item() == 4
    assert torch.count_nonzero(wall[:, 1]).item() == 0
    assert torch.count_nonzero(outflow).item() == 0


def test_trainer_side_path_matches_no_boundary_seed_backbone_and_rng() -> None:
    common = {
        "k_max": 1,
        "domain_lengths": (1.0, 1.0),
        "layers": (8, 8),
        "fc_dim": 8,
        "model_node_type_input": "physical",
        "node_type_channel_control": NO_TYPE_CHANNEL_CONTROL,
        "boundary_field_mode": "none",
        "boundary_field_names": [],
    }
    torch.manual_seed(211)
    no_boundary = build_model(
        SimpleNamespace(**common), _unit_normalization(), zero_initialize=False
    )
    rng_after_no_boundary = torch.get_rng_state().clone()

    torch.manual_seed(211)
    residual = build_model(
        SimpleNamespace(
            **common,
            boundary_residual_mode=BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
            boundary_residual_names=list(SEMANTIC_NAMES),
            boundary_residual_width=4,
        ),
        _unit_normalization(),
        zero_initialize=False,
    )
    assert torch.equal(torch.get_rng_state(), rng_after_no_boundary)
    assert (
        residual.initialization_control["cpu_rng_state_matches_no_boundary_arm"] is True
    )
    source_state = no_boundary.state_dict()
    target_state = residual.state_dict()
    for name, value in source_state.items():
        assert torch.equal(value, target_state[name]), name


def test_trainer_rejects_mixed_boundary_routes_and_categorical_backbone() -> None:
    common = [
        "--data-dir",
        "unused",
        "--output-dir",
        "unused",
        "--amp",
        "none",
        "--boundary-residual-mode",
        BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
    ]
    categorical = parse_args(common)
    with pytest.raises(ValueError, match="no_type_channels"):
        validate_args(categorical, torch.device("cpu"))

    mixed = parse_args(
        [
            *common,
            "--node-type-channel-control",
            NO_TYPE_CHANNEL_CONTROL,
            "--boundary-field-mode",
            BOUNDARY_FIELD_SEMANTIC_COLLAR,
        ]
    )
    with pytest.raises(ValueError, match="separate representation studies"):
        validate_args(mixed, torch.device("cpu"))


def test_checkpoint_runtime_rebuilds_boundary_residual_model_strictly() -> None:
    model = PCNOEuler2DResidual(
        **_model_common(),
        boundary_residual_mode=BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
        boundary_residual_names=SEMANTIC_NAMES,
        boundary_residual_width=4,
    )
    checkpoint = {
        "normalization": _unit_normalization().to_dict(),
        "model_config": model.model_config(),
        "model_state": model.state_dict(),
    }
    rebuilt, _ = build_checkpoint_model(
        checkpoint, torch.device("cpu"), model_node_type_input="physical"
    )
    assert rebuilt.model_config() == model.model_config()
    for name, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[name]), name
