from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.fft import dctn, idctn
from torch import nn

from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    build_structured_cell_grid,
    make_translated_front_case,
    pcno_case_input,
    physical_cosine_filter,
)
from utility.time_dependent_no.pcno_structured_representation import (
    REPRESENTATIONS,
    StructuredCosinePCNO,
    half_smooth_gain,
    orthonormal_dct_matrix,
)


class RecordingScalarBackbone(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.last_input: torch.Tensor | None = None
        self.last_aux: object | None = None
        self.last_fourier: object | None = None
        self.prepared: tuple[torch.Tensor, torch.Tensor] | None = None

    def prepare_fourier_tensors(
        self, nodes: torch.Tensor, node_weights: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        self.prepared = (nodes, node_weights)
        return (nodes, node_weights)

    def forward(
        self,
        values: torch.Tensor,
        aux: object,
        *,
        fourier_tensors: object | None = None,
    ) -> torch.Tensor:
        self.last_input = values
        self.last_aux = aux
        self.last_fourier = fourier_tensors
        return self.scale * values[..., -1:]


def _pcno_input(
    grid,
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260824)
    values = torch.randn(
        (2, grid.nx * grid.ny, 4),
        generator=generator,
        dtype=dtype,
    )
    return values.requires_grad_(requires_grad)


def test_torch_dct_matches_scipy_and_has_unit_gain_closure() -> None:
    generator = np.random.default_rng(20260824)
    field = generator.normal(size=(7, 11))
    matrix_y = orthonormal_dct_matrix(7).numpy()
    matrix_x = orthonormal_dct_matrix(11).numpy()

    torch_coefficients = matrix_y @ field @ matrix_x.T
    scipy_coefficients = dctn(field, type=2, axes=(0, 1), norm="ortho")
    np.testing.assert_allclose(
        torch_coefficients, scipy_coefficients, rtol=2.0e-14, atol=2.0e-14
    )

    torch_roundtrip = matrix_y.T @ torch_coefficients @ matrix_x
    scipy_roundtrip = idctn(
        scipy_coefficients, type=2, axes=(0, 1), norm="ortho"
    )
    np.testing.assert_allclose(torch_roundtrip, field, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(scipy_roundtrip, field, rtol=2.0e-14, atol=2.0e-14)


def test_half_smooth_gain_matches_existing_scipy_filter() -> None:
    grid = build_structured_cell_grid((16, 8))
    generator = np.random.default_rng(1701)
    field = generator.normal(size=grid.array_shape)
    matrix_y = orthonormal_dct_matrix(grid.ny).numpy()
    matrix_x = orthonormal_dct_matrix(grid.nx).numpy()
    gain = half_smooth_gain(grid).numpy()
    torch_filtered = matrix_y.T @ (
        gain * (matrix_y @ field @ matrix_x.T)
    ) @ matrix_x
    scipy_half_smooth = 0.5 * (
        field + physical_cosine_filter(field, grid, kind="smooth")
    )

    assert np.min(gain) >= 0.5
    assert np.max(gain) <= 1.0
    np.testing.assert_allclose(
        torch_filtered, scipy_half_smooth, rtol=2.0e-14, atol=2.0e-14
    )


def test_native_is_exact_tensor_bypass_and_delegates_pcno_api() -> None:
    grid = build_structured_cell_grid((8, 4))
    backbone = RecordingScalarBackbone()
    model = StructuredCosinePCNO(backbone, grid, "native")
    values = _pcno_input(grid)
    aux = (object(),)
    fourier = (object(),)

    output = model(values, aux, fourier_tensors=fourier)

    assert model.backbone is backbone
    assert backbone.last_input is values
    assert backbone.last_aux is aux
    assert backbone.last_fourier is fourier
    torch.testing.assert_close(output, values[..., -1:], rtol=0.0, atol=0.0)

    nodes = torch.randn(1, grid.nx * grid.ny, 2)
    weights = torch.randn(1, grid.nx * grid.ny, 1)
    prepared = model.prepare_fourier_tensors(nodes, weights)
    assert backbone.prepared is not None
    assert backbone.prepared[0] is nodes
    assert backbone.prepared[1] is weights
    assert prepared[0] is nodes
    assert prepared[1] is weights


@pytest.mark.parametrize(
    "representation",
    ("dct_pre_half_smooth", "dct_coupled_half_smooth"),
)
def test_only_scalar_state_is_encoded(
    representation: str,
) -> None:
    grid = build_structured_cell_grid((8, 4))
    backbone = RecordingScalarBackbone()
    model = StructuredCosinePCNO(backbone, grid, representation)
    values = _pcno_input(grid)

    aux = (object(),)
    model(values, aux, fourier_tensors=())

    assert backbone.last_input is not None
    assert backbone.last_aux is aux
    torch.testing.assert_close(
        backbone.last_input[..., :-1], values[..., :-1], rtol=0.0, atol=0.0
    )
    assert not torch.equal(backbone.last_input[..., -1:], values[..., -1:])


@pytest.mark.parametrize("family", ("step", "pulse"))
def test_float32_coupled_residual_and_physical_update_close(
    family: str,
) -> None:
    grid = build_structured_cell_grid((16, 8))
    case = make_translated_front_case(
        grid,
        family,
        position=ANCHORS[0] + 0.875 * grid.hx,
    )
    backbone = RecordingScalarBackbone(dtype=torch.float32)
    with torch.no_grad():
        backbone.scale.fill_(-0.25)
    model = StructuredCosinePCNO(
        backbone, grid, "dct_coupled_half_smooth"
    )
    values = pcno_case_input(case, grid, dtype=torch.float32)

    residual = model(values, (), fourier_tensors=())
    current = values[..., -1:]
    expected_residual = -0.25 * current
    torch.testing.assert_close(
        residual, expected_residual, rtol=2.0e-5, atol=2.0e-6
    )
    torch.testing.assert_close(
        current + residual,
        0.75 * current,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_coupled_decode_is_differentiable_and_closes_fixed_gain() -> None:
    grid = build_structured_cell_grid((8, 4))
    backbone = RecordingScalarBackbone()
    model = StructuredCosinePCNO(
        backbone, grid, "dct_coupled_half_smooth"
    )
    values = _pcno_input(grid, requires_grad=True)

    output = model(values, (), fourier_tensors=())
    torch.testing.assert_close(
        output, values[..., -1:], rtol=3.0e-14, atol=3.0e-14
    )

    weights = torch.linspace(
        0.5,
        1.5,
        output.numel(),
        dtype=output.dtype,
    ).reshape_as(output)
    loss = torch.sum(weights * output)
    loss.backward()
    assert values.grad is not None
    torch.testing.assert_close(
        values.grad[..., -1:], weights, rtol=5.0e-14, atol=5.0e-14
    )
    torch.testing.assert_close(
        values.grad[..., :-1],
        torch.zeros_like(values.grad[..., :-1]),
        rtol=0.0,
        atol=0.0,
    )
    assert backbone.scale.grad is not None
    torch.testing.assert_close(
        backbone.scale.grad,
        torch.sum(weights * values.detach()[..., -1:]),
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_coupled_decode_passes_autograd_gradcheck() -> None:
    grid = build_structured_cell_grid((4, 2))
    model = StructuredCosinePCNO(
        RecordingScalarBackbone(),
        grid,
        "dct_coupled_half_smooth",
    )
    static = _pcno_input(grid)[..., :-1]
    state = _pcno_input(grid)[..., -1:].detach().requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda current: model(
            torch.cat((static, current), dim=-1), (), fourier_tensors=()
        ),
        (state,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-3,
    )


def test_all_representations_have_the_backbone_parameter_count() -> None:
    grid = build_structured_cell_grid((8, 4))
    counts: dict[str, int] = {}
    for representation in REPRESENTATIONS:
        backbone = RecordingScalarBackbone()
        model = StructuredCosinePCNO(backbone, grid, representation)
        counts[representation] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        assert counts[representation] == sum(
            parameter.numel() for parameter in backbone.parameters()
        )
        assert all(
            name.startswith("backbone.") for name, _ in model.named_parameters()
        )
    assert len(set(counts.values())) == 1


def test_representation_validation_fails_closed() -> None:
    grid = build_structured_cell_grid((8, 4))
    with pytest.raises(ValueError, match="representation must be one of"):
        StructuredCosinePCNO(RecordingScalarBackbone(), grid, "unknown")
