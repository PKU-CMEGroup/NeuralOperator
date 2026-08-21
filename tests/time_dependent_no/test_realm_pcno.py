from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from pcno.pcno import compute_gradient
from utility.time_dependent_no.realm_benchmark import predict_one_call
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
    estimate_realm_pcno_static_bytes,
)


def _released_coordinates(height: int = 4, width: int = 5) -> np.ndarray:
    y = np.linspace(4.5, 0.5, height, dtype=np.float32)
    x = np.linspace(11.0, 1.0, width, dtype=np.float32)
    return np.stack(
        (
            np.broadcast_to(y[:, None], (height, width)),
            np.broadcast_to(x[None, :], (height, width)),
        )
    )


def _geometry(height: int = 4, width: int = 5):
    return build_realm_regular_grid_geometry(
        _released_coordinates(height, width),
        domain_lengths_xy=(12.0, 5.0),
        released_coordinate_order=("y", "x"),
    )


def _config(
    channels: int,
    *,
    zero_initialize_head: bool = False,
    use_gradient: bool = True,
):
    return RealmPCNOConfig(
        channels=channels,
        mode_counts_xy=(1, 1),
        layers=(4, 4),
        fc_dim=4,
        zero_initialize_head=zero_initialize_head,
        use_gradient=use_gradient,
    )


def _normalized_coordinates(height: int = 4, width: int = 5) -> torch.Tensor:
    coordinates = torch.as_tensor(_released_coordinates(height, width)).unsqueeze(0)
    minima = coordinates.amin(dim=(2, 3), keepdim=True)
    spans = coordinates.amax(dim=(2, 3), keepdim=True) - minima
    return (coordinates - minima) / spans


def test_descending_yx_geometry_has_xy_nodes_open_edges_and_exact_gradients() -> None:
    geometry = _geometry()

    assert geometry.nodes.shape == (20, 2)
    np.testing.assert_allclose(geometry.nodes[0], [10.0, 4.0])
    np.testing.assert_allclose(geometry.nodes[-1], [0.0, 0.0])
    assert geometry.contract["released_coordinate_order"] == ["y", "x"]
    assert geometry.contract["model_node_order"] == ["x", "y"]
    assert geometry.contract["axis_direction_xy"] == ["descending", "descending"]
    assert geometry.contract["physical_volume_claim"] is False
    assert geometry.directed_edges.shape == (4 * 20 - 2 * 4 - 2 * 5, 2)
    assert np.isclose(geometry.node_weights.sum(), 1.0)
    np.testing.assert_allclose(geometry.node_rhos, np.ones((20, 1)))

    first_neighbors = set(
        geometry.directed_edges[geometry.directed_edges[:, 0] == 0, 1].tolist()
    )
    assert first_neighbors == {1, 5}

    field = 3.0 * geometry.nodes[:, 0] + 2.0 * geometry.nodes[:, 1]
    gradient = compute_gradient(
        torch.as_tensor(field).reshape(1, 1, -1),
        torch.as_tensor(geometry.directed_edges).unsqueeze(0),
        torch.as_tensor(geometry.edge_gradient_weights).unsqueeze(0),
    )
    expected = torch.tensor([3.0, 2.0]).reshape(1, 2, 1).expand_as(gradient)
    torch.testing.assert_close(gradient, expected, atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize("channels", [12, 13])
def test_tiny_model_forward_backward_and_residual_reconstruction(channels: int) -> None:
    torch.manual_seed(11)
    geometry = _geometry()
    model = RealmRegularGridPCNO(
        config=_config(channels),
        geometry=geometry,
    )
    state = torch.randn(
        2, channels, geometry.height, geometry.width, requires_grad=True
    )
    coordinates = _normalized_coordinates()
    truth = torch.randn_like(state)

    raw = model(state, coordinates)
    prediction = predict_one_call(
        model,
        state,
        coordinates,
        parameterization="residual",
    )
    loss = (prediction - truth).square().mean()
    loss.backward()

    assert raw.shape == prediction.shape == state.shape
    torch.testing.assert_close(prediction, state + raw)
    assert torch.isfinite(prediction).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_zero_initialized_head_is_exact_identity_under_residual_parameterization() -> (
    None
):
    geometry = _geometry()
    model = RealmRegularGridPCNO(
        config=_config(13, zero_initialize_head=True),
        geometry=geometry,
    )
    state = torch.randn(2, 13, geometry.height, geometry.width)
    prediction = predict_one_call(
        model,
        state,
        _normalized_coordinates(),
        parameterization="residual",
    )

    torch.testing.assert_close(prediction, state, atol=0.0, rtol=0.0)
    assert model.model_contract()["parameterization"].startswith("external")
    assert model.model_contract()["pde_residual_claim"] is False


def test_pcfno_removes_only_gradient_parameters_and_never_calls_gradient_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    pcno = RealmRegularGridPCNO(config=_config(3), geometry=geometry)
    pcfno = RealmRegularGridPCNO(
        config=_config(3, use_gradient=False),
        geometry=geometry,
    )
    pcno_count = sum(parameter.numel() for parameter in pcno.parameters())
    pcfno_count = sum(parameter.numel() for parameter in pcfno.parameters())
    expected_removed = sum(
        1 + out_size * 2 * in_size for in_size, out_size in zip((4,), (4,), strict=True)
    )
    assert pcno_count - pcfno_count == expected_removed
    assert not any("backbone.gws" in key for key in pcfno.state_dict())
    assert pcfno.model_contract()["gradient_branch"] is False

    def forbidden_gradient(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("PCFNO must not evaluate compute_gradient")

    monkeypatch.setattr("pcno.pcno.compute_gradient", forbidden_gradient)
    state = torch.randn(2, 3, geometry.height, geometry.width, requires_grad=True)
    output = pcfno(state, _normalized_coordinates())
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pcfno.parameters()
    )


class _CaptureBackbone(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.features: torch.Tensor | None = None

    def forward(self, values, aux, *, fourier_tensors=None):
        self.features = values.detach().clone()
        return values.new_zeros(values.shape[0], values.shape[1], self.channels)


def test_feature_order_is_state_then_xy_then_quadrature_density() -> None:
    geometry = _geometry()
    model = RealmRegularGridPCNO(config=_config(2), geometry=geometry)
    capture = _CaptureBackbone(channels=2)
    model.backbone = capture
    state = torch.zeros(1, 2, geometry.height, geometry.width)
    state[:, 0] = 3.0
    state[:, 1] = -2.0
    coordinates = _normalized_coordinates()

    model(state, coordinates)

    assert capture.features is not None
    features = capture.features
    torch.testing.assert_close(features[..., 0], torch.full_like(features[..., 0], 3.0))
    torch.testing.assert_close(
        features[..., 1], torch.full_like(features[..., 1], -2.0)
    )
    expected_x = coordinates[:, 1].reshape(1, -1)
    expected_y = coordinates[:, 0].reshape(1, -1)
    torch.testing.assert_close(features[..., 2], expected_x)
    torch.testing.assert_close(features[..., 3], expected_y)
    torch.testing.assert_close(features[..., 4], torch.ones_like(features[..., 4]))


def test_cached_core_matches_ordinary_core_and_checkpoint_rebuilds_exactly() -> None:
    torch.manual_seed(3)
    geometry = _geometry()
    config = _config(3)
    model = RealmRegularGridPCNO(config=config, geometry=geometry)
    features = torch.randn(2, geometry.height * geometry.width, 6)

    ordinary = model.backbone(features, model._expanded_geometry(2))
    cached = model.backbone(
        features,
        model._expanded_geometry(2),
        fourier_tensors=model._expanded_fourier(2),
    )
    torch.testing.assert_close(cached, ordinary)

    state_keys = set(model.state_dict())
    assert "nodes" not in state_keys
    assert "basis_cos" not in state_keys
    rebuilt = RealmRegularGridPCNO(config=config, geometry=geometry)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    state = torch.randn(2, 3, geometry.height, geometry.width)
    coordinates = _normalized_coordinates()
    torch.testing.assert_close(
        rebuilt(state, coordinates),
        model(state, coordinates),
    )


def test_full_planardet_memory_estimate_exposes_dense_basis_cost() -> None:
    estimate = estimate_realm_pcno_static_bytes(832, 384, (8, 8))

    assert estimate["node_count"] == 319_488
    assert estimate["directed_edge_count"] == 1_275_520
    assert estimate["fourier_mode_count"] == 144
    assert estimate["fourier_tensors"] == 738_656_256
    assert estimate["total_bytes"] > estimate["fourier_tensors"]


def test_geometry_and_model_contracts_fail_closed() -> None:
    coordinates = _released_coordinates()
    with pytest.raises(ValueError, match="permutation"):
        build_realm_regular_grid_geometry(
            coordinates,
            domain_lengths_xy=(12.0, 5.0),
            released_coordinate_order=("x", "x"),
        )
    with pytest.raises(ValueError, match="smaller than coordinate point extents"):
        build_realm_regular_grid_geometry(
            coordinates,
            domain_lengths_xy=(1.0, 1.0),
            released_coordinate_order=("y", "x"),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        build_realm_regular_grid_geometry(
            coordinates,
            domain_lengths_xy=(12.0, 5.0),
            released_coordinate_order=("y", "x"),
            node_weights_yx=np.zeros((4, 5)),
        )

    model = RealmRegularGridPCNO(config=_config(13), geometry=_geometry())
    valid_coordinates = _normalized_coordinates()
    with pytest.raises(ValueError, match="state must have shape"):
        model(torch.zeros(1, 12, 4, 5), valid_coordinates)
    with pytest.raises(ValueError, match="coordinate case count"):
        model(torch.zeros(3, 13, 4, 5), valid_coordinates.expand(2, -1, -1, -1))
    with pytest.raises(ValueError, match="finite floating"):
        model(torch.full((1, 13, 4, 5), torch.nan), valid_coordinates)
    with pytest.raises(ValueError, match="share dtype"):
        model(
            torch.zeros(1, 13, 4, 5, dtype=torch.float64),
            valid_coordinates.to(torch.float64),
        )


def test_forward_surface_matches_generic_realm_step_model() -> None:
    assert tuple(inspect.signature(RealmRegularGridPCNO.forward).parameters) == (
        "self",
        "state",
        "static_coordinates",
    )
