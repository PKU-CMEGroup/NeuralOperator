"""Differentiable fixed cosine representations for the W26-L2 PCNO pilot.

Only the scalar state channel is encoded.  Coordinates, quadrature density,
PCNO geometry, and Fourier tensors retain the native contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from utility.time_dependent_no.pcno_shock_representation import (
    SMOOTH_FILTER_PASS_WAVENUMBER,
    SMOOTH_FILTER_STOP_WAVENUMBER,
    StructuredCellGrid,
)

REPRESENTATIONS = (
    "native",
    "dct_pre_half_smooth",
    "dct_coupled_half_smooth",
)


def orthonormal_dct_matrix(
    size: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the orthonormal DCT-II analysis matrix of shape ``[size,size]``."""

    if isinstance(size, bool) or int(size) != size or int(size) < 1:
        raise ValueError("size must be a positive integer")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    count = int(size)
    sample = torch.arange(count, dtype=dtype, device=device)
    mode = torch.arange(count, dtype=dtype, device=device).unsqueeze(1)
    matrix = torch.cos(math.pi * (sample + 0.5) * mode / count)
    matrix[0] *= math.sqrt(1.0 / count)
    if count > 1:
        matrix[1:] *= math.sqrt(2.0 / count)
    return matrix


def half_smooth_gain(
    grid: StructuredCellGrid,
    *,
    pass_wavenumber: float = SMOOTH_FILTER_PASS_WAVENUMBER,
    stop_wavenumber: float = SMOOTH_FILTER_STOP_WAVENUMBER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the frozen gain ``G(q)=0.5+0.5H(q)`` on ``[ny,nx]`` DCT modes."""

    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    pass_value = float(pass_wavenumber)
    stop_value = float(stop_wavenumber)
    if (
        not math.isfinite(pass_value)
        or not math.isfinite(stop_value)
        or pass_value < 0.0
        or stop_value <= pass_value
    ):
        raise ValueError(
            "wavenumbers must be finite with 0 <= pass_wavenumber < stop_wavenumber"
        )

    qx = torch.arange(grid.nx, dtype=dtype, device=device) / (
        2.0 * float(grid.lengths[0])
    )
    qy = torch.arange(grid.ny, dtype=dtype, device=device) / (
        2.0 * float(grid.lengths[1])
    )
    physical_q = torch.sqrt(qy[:, None].square() + qx[None, :].square())
    smooth = torch.ones_like(physical_q)
    smooth = torch.where(physical_q >= stop_value, 0.0, smooth)
    transition = (physical_q > pass_value) & (physical_q < stop_value)
    transition_gain = 0.5 * (
        1.0
        + torch.cos(
            math.pi
            * (physical_q - pass_value)
            / (stop_value - pass_value)
        )
    )
    smooth = torch.where(transition, transition_gain, smooth)
    return 0.5 + 0.5 * smooth


class StructuredCosinePCNO(nn.Module):
    """Wrap a scalar PCNO with one preregistered fixed DCT representation."""

    def __init__(
        self,
        backbone: nn.Module,
        grid: StructuredCellGrid,
        representation: str,
    ) -> None:
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(
                "representation must be one of "
                f"{REPRESENTATIONS}, got {representation!r}"
            )
        self.backbone = backbone
        self.grid = grid
        self.representation = representation

        reference = next(backbone.parameters(), None)
        dtype = reference.dtype if reference is not None else torch.float32
        device = reference.device if reference is not None else torch.device("cpu")
        self.register_buffer(
            "dct_x",
            orthonormal_dct_matrix(grid.nx, dtype=dtype, device=device),
            persistent=True,
        )
        self.register_buffer(
            "dct_y",
            orthonormal_dct_matrix(grid.ny, dtype=dtype, device=device),
            persistent=True,
        )
        gain = half_smooth_gain(grid, dtype=dtype, device=device)
        self.register_buffer("gain", gain, persistent=True)
        self.register_buffer("inverse_gain", gain.reciprocal(), persistent=True)

    def prepare_fourier_tensors(
        self, nodes: torch.Tensor, node_weights: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Delegate immutable-geometry Fourier preparation to the backbone."""

        return self.backbone.prepare_fourier_tensors(nodes, node_weights)

    def _transform_scalar(
        self, values: torch.Tensor, gain: torch.Tensor
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (
            self.grid.nx * self.grid.ny,
            1,
        ):
            raise ValueError("scalar field must have shape [batch,nx*ny,1]")
        spatial = values[..., 0].reshape(-1, self.grid.ny, self.grid.nx)
        coefficients = torch.matmul(self.dct_y, spatial)
        coefficients = torch.matmul(coefficients, self.dct_x.transpose(0, 1))
        coefficients = coefficients * gain
        decoded = torch.matmul(self.dct_y.transpose(0, 1), coefficients)
        decoded = torch.matmul(decoded, self.dct_x)
        return decoded.reshape(values.shape)

    def forward(
        self,
        x: torch.Tensor,
        aux: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Run the native or fixed encoded PCNO map under the usual signature."""

        if self.representation == "native":
            return self.backbone(x, aux, fourier_tensors=fourier_tensors)
        if x.ndim != 3 or x.shape[1] != self.grid.nx * self.grid.ny:
            raise ValueError("PCNO input must have shape [batch,nx*ny,channels]")
        encoded_state = self._transform_scalar(x[..., -1:], self.gain)
        encoded_input = torch.cat((x[..., :-1], encoded_state), dim=-1)
        output = self.backbone(
            encoded_input,
            aux,
            fourier_tensors=fourier_tensors,
        )
        if self.representation == "dct_coupled_half_smooth":
            return self._transform_scalar(output, self.inverse_gain)
        return output


__all__ = [
    "REPRESENTATIONS",
    "StructuredCosinePCNO",
    "half_smooth_gain",
    "orthonormal_dct_matrix",
]
