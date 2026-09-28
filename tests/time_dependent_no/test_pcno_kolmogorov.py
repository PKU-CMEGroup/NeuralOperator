from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pcno.pcno import compute_gradient
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
)
from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO,
    canonicalize_vorticity,
    periodic_grid_geometry,
)


def _grid(n: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    coordinate = torch.arange(n, dtype=dtype) * (2 * math.pi / n)
    return torch.meshgrid(coordinate, coordinate, indexing="ij")


def _model(*, train_scale: float = 2.0) -> PeriodicVorticityPCNO:
    return PeriodicVorticityPCNO(
        16, train_scale=train_scale, modes=2, width=4, depth=1, fc_dim=8
    )


@pytest.mark.parametrize("n", (16, 32))
def test_periodic_geometry_gradient_matches_centered_sine_stencil_at_seams(
    n: int,
) -> None:
    geometry = periodic_grid_geometry(n, dtype=torch.float64)
    x, y = _grid(n)
    field = torch.sin(2 * x) + 0.7 * torch.sin(3 * y)
    gradient = compute_gradient(
        field.reshape(1, 1, -1),
        geometry["directed_edges"].unsqueeze(0),
        geometry["edge_gradient_weights"].unsqueeze(0),
    ).reshape(2, n, n)
    spacing = 2 * math.pi / n
    expected = torch.stack(
        (
            math.sin(2 * spacing) / spacing * torch.cos(2 * x),
            0.7 * math.sin(3 * spacing) / spacing * torch.cos(3 * y),
        )
    )
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        geometry["nodes"], torch.stack((x, y), -1).reshape(-1, 2)
    )
    assert geometry["node_weights"].sum().item() == pytest.approx(1.0)
    assert geometry["directed_edges"].shape == (4 * n * n, 2)
    offsets = geometry["edge_offsets"]
    torch.testing.assert_close(
        offsets.norm(dim=-1), torch.full((4 * n * n,), spacing, dtype=torch.float64)
    )
    edges = geometry["directed_edges"]
    raw = geometry["nodes"][edges[:, 1]] - geometry["nodes"][edges[:, 0]]
    assert torch.any(raw.abs() > math.pi)  # The test really includes wraparound edges.
    assert torch.all(offsets.abs() < math.pi)
    torch.testing.assert_close(gradient[:, [0, -1], :], expected[:, [0, -1], :])
    torch.testing.assert_close(gradient[:, :, [0, -1]], expected[:, :, [0, -1]])


@pytest.mark.parametrize("n", (16, 32))
def test_restriction_matches_reference_and_removes_mean_and_outside_band(
    n: int,
) -> None:
    x, y = _grid(n)
    inside = torch.sin(2 * x) + torch.cos(4 * y)
    raw = (inside + 2.0 + 0.5 * torch.cos((n // 3 + 1) * x)).unsqueeze(0)
    before = raw.clone()
    restricted = canonicalize_vorticity(raw)
    torch.testing.assert_close(restricted, inside.unsqueeze(0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        canonicalize_vorticity(restricted), restricted, rtol=1e-12, atol=1e-12
    )
    assert restricted.mean().abs().item() < 1e-14
    solver = KolmogorovReferenceStepper(KolmogorovReferenceConfig(resolution=n))
    expected = solver.canonicalize(raw[0].numpy())
    np.testing.assert_allclose(restricted[0].numpy(), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_restriction_backward_is_the_same_self_adjoint_projection() -> None:
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(
        2, 16, 16, generator=generator, dtype=torch.float64, requires_grad=True
    )
    probe = torch.randn(2, 16, 16, generator=generator, dtype=torch.float64)
    projected = canonicalize_vorticity(raw)
    (projected * probe).sum().backward()
    torch.testing.assert_close(
        raw.grad, canonicalize_vorticity(probe), rtol=1e-12, atol=1e-12
    )


def test_zero_head_is_residual_identity_on_canonical_states_and_keeps_raw_output() -> (
    None
):
    model = _model()
    with torch.no_grad():
        model.pcno.fc2.weight.zero_()
        model.pcno.fc2.bias.zero_()
    x, y = _grid(16, torch.float32)
    state = canonicalize_vorticity((torch.sin(x) + torch.cos(4 * y)).unsqueeze(0))
    output = model(state)
    torch.testing.assert_close(output["raw_next"], state, rtol=0, atol=0)
    torch.testing.assert_close(output["next_state"], state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        output["projection_residual"], output["raw_next"] - output["next_state"]
    )
    with torch.no_grad():
        model.pcno.fc2.bias.fill_(1.0)
    shifted = model(state)
    torch.testing.assert_close(shifted["raw_next"], state + 2.0)
    torch.testing.assert_close(shifted["next_state"], state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        shifted["projection_residual"], torch.full_like(state, 2.0)
    )


def test_forcing_feature_is_explicit_and_scaling_is_caller_supplied() -> None:
    model = _model(train_scale=2.0)
    x, y = _grid(16, torch.float32)
    state = (2.0 * torch.sin(x)).unsqueeze(0)
    captured: list[torch.Tensor] = []
    hook = model.pcno.fc0.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    output = model(state)
    hook.remove()
    assert model.pcno.in_dim == 2
    assert captured[0].shape == (1, 16 * 16, 2)
    torch.testing.assert_close(captured[0][0, :, 0], torch.sin(x).reshape(-1))
    torch.testing.assert_close(captured[0][0, :, 1], torch.cos(4 * y).reshape(-1))
    scaled_model = _model(train_scale=4.0)
    scaled_model.pcno.load_state_dict(model.pcno.state_dict())
    scaled = scaled_model(2.0 * state)
    for key in output:
        torch.testing.assert_close(scaled[key], 2.0 * output[key], rtol=1e-5, atol=1e-6)
    assert model.train_scale.item() == 2.0


def test_backpropagation_and_lazy_fourier_cache_survive_dtype_transfer() -> None:
    model = _model()
    assert model._fourier_cache is None
    state = torch.randn(2, 16, 16, requires_grad=True)
    output = model(state)
    assert model._fourier_cache is not None
    assert all(
        not value.requires_grad and value.grad_fn is None
        for value in model._fourier_cache
    )
    cache = model._fourier_cache
    output["next_state"].square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert model.pcno.fc2.weight.grad is not None
    assert torch.isfinite(model.pcno.fc2.weight.grad).all()
    model(state.detach()[:1])
    assert model._fourier_cache is cache
    assert not any("cache" in name for name in model.state_dict())
    model.double()
    assert model._fourier_cache is None
    converted = model(state.detach().double())
    assert converted["next_state"].dtype == torch.float64
    assert all(value.dtype == torch.float64 for value in model._fourier_cache)


@pytest.mark.parametrize("resolution", (True, 8, 12, 24, 16.0))
def test_rejects_invalid_resolutions(resolution: int) -> None:
    with pytest.raises(ValueError, match="power of two at least 16"):
        periodic_grid_geometry(resolution)


@pytest.mark.parametrize("scale", (0.0, -1.0, float("inf"), float("nan"), True, 1.0j))
def test_rejects_invalid_training_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite positive real scalar"):
        _model(train_scale=scale)


@pytest.mark.parametrize("scale", (1e-300, 1e300))
def test_rejects_scale_underflow_or_overflow_in_model_dtype(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive in model dtype"):
        _model(train_scale=scale)


@pytest.mark.parametrize(
    "value",
    (
        torch.ones(16, 16),
        torch.ones(1, 16, 32),
        torch.ones(0, 16, 16),
        torch.ones(1, 16, 16, dtype=torch.int64),
        torch.ones(1, 16, 16, dtype=torch.complex64),
        torch.full((1, 16, 16), float("nan")),
    ),
)
def test_rejects_invalid_state_shape_type_or_finiteness(value: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        canonicalize_vorticity(value)


def test_model_rejects_mismatched_grid_dtype_and_outside_band_modes() -> None:
    model = _model()
    with pytest.raises(ValueError, match="resolution does not match"):
        model(torch.zeros(1, 32, 32))
    with pytest.raises(ValueError, match="device and dtype"):
        model(torch.zeros(1, 16, 16, dtype=torch.float64))
    with pytest.raises(ValueError, match="retained Fourier band"):
        PeriodicVorticityPCNO(16, train_scale=1.0, modes=6)
