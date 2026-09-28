"""Periodic scalar-vorticity PCNO for the fixed Kolmogorov case study.

Arrays use (x, y) order on [0, 2*pi)^2. Vorticity represents the full state
only with the reference solver's fixed zero-mean velocity convention. The
known forcing is -4*cos(4*y), not a learned or hidden conditioning variable.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pcno.pcno import PCNO, compute_Fourier_modes


def _resolution(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 16
        or value & (value - 1)
    ):
        raise ValueError("resolution must be a power of two at least 16")
    return value


def _state(value: torch.Tensor) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError("vorticity must be a torch Tensor")
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] != value.shape[2]:
        raise ValueError("vorticity must have shape [batch, n, n] in (x, y) order")
    n = _resolution(value.shape[-1])
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError("vorticity must be real float32 or float64")
    if not torch.isfinite(value).all():
        raise ValueError("vorticity must contain only finite values")
    return n


def periodic_grid_geometry(
    resolution: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Build uniform quadrature and four-neighbor periodic least-squares edges.

    Flat index i*n+j means (x_i, y_j). Quadrature weights sum to one, matching
    PCNO's normalized integral convention; they are not unnormalized cell areas.
    The core Euclidean geometry helper uses raw coordinate differences, so it
    cannot be used at periodic seams. Here minimum-image offsets give the
    cardinal-stencil pseudoinverse dx/(2*h^2), including wraparound edges.
    """
    n = _resolution(resolution)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("geometry dtype must be float32 or float64")
    spacing = 2.0 * math.pi / n
    coordinate = torch.arange(n, dtype=torch.float64, device=device) * spacing
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    nodes = torch.stack((x, y), dim=-1).reshape(n * n, 2)
    indices = torch.arange(n * n, device=device).reshape(n, n)
    sources = torch.stack(
        (
            torch.roll(indices, -1, 0),
            torch.roll(indices, 1, 0),
            torch.roll(indices, -1, 1),
            torch.roll(indices, 1, 1),
        ),
        dim=-1,
    ).reshape(-1)
    targets = indices.reshape(-1).repeat_interleave(4)
    edges = torch.stack((targets, sources), dim=-1)
    raw_offsets = nodes[sources] - nodes[targets]
    offsets = torch.remainder(raw_offsets + math.pi, 2.0 * math.pi) - math.pi
    return {
        "nodes": nodes.to(dtype=dtype),
        "node_mask": torch.ones(n * n, 1, dtype=dtype, device=device),
        "node_weights": torch.full(
            (n * n, 1), 1.0 / (n * n), dtype=dtype, device=device
        ),
        "directed_edges": edges,
        "edge_offsets": offsets.to(dtype=dtype),
        "edge_gradient_weights": (offsets / (2.0 * spacing**2)).to(dtype=dtype),
    }


def canonicalize_vorticity(vorticity: torch.Tensor) -> torch.Tensor:
    """Differentiable mean-zero rectangular 2/3 Fourier restriction.

    Retains |kx|, |ky| <= floor(n/3), except the zero mode, exactly as the
    finite-grid reference contract on these power-of-two grids. No input is
    modified. This is a shared physical-state restriction, not a learned
    training-manifold projection; callers must keep its effect attributable.
    """
    n = _state(vorticity)
    modes = torch.fft.fftfreq(n, device=vorticity.device, dtype=vorticity.dtype) * n
    retained = modes.abs() <= n // 3
    mask = retained[:, None] & retained[None, :]
    mask[0, 0] = False
    coefficients = torch.fft.fft2(vorticity)
    return torch.fft.ifft2(coefficients * mask).real


class PeriodicVorticityPCNO(nn.Module):
    """Residual PCNO with explicit forcing input and shared output restriction.

    ``in_dim=2`` uses [omega/train_scale, cos(4*y)]. The second feature is the
    dimensionless known forcing shape for fixed A=1, k=4 and L=2*pi; actual
    vorticity forcing is -4 times that feature. There are no raw coordinate
    channels. ``train_scale`` is one positive scalar supplied by the caller
    from training data only; no statistics are fitted by this adapter.

    ``forward`` accepts physical-unit [batch,n,n] arrays and returns physical
    ``raw_next = omega + train_scale*PCNO(features)``, restricted ``next_state``,
    and ``projection_residual = raw_next - next_state``. It does not silently
    restrict inputs. Recurrent callers feed back ``next_state``. The output
    restriction must be shared by every compared arm and reported separately.
    """

    def __init__(
        self,
        resolution: int,
        *,
        train_scale: float,
        modes: int = 12,
        width: int = 64,
        depth: int = 4,
        fc_dim: int = 128,
    ) -> None:
        super().__init__()
        self.resolution = _resolution(resolution)
        if (
            isinstance(train_scale, bool)
            or not isinstance(train_scale, (int, float))
            or not math.isfinite(train_scale)
            or train_scale <= 0.0
        ):
            raise ValueError("train_scale must be one finite positive real scalar")
        scale_tensor = torch.tensor(float(train_scale), dtype=torch.float32)
        if not torch.isfinite(scale_tensor) or scale_tensor <= 0.0:
            raise ValueError(
                "train_scale must remain finite and positive in model dtype"
            )
        for name, value, lower in (
            ("modes", modes, 1),
            ("width", width, 1),
            ("depth", depth, 1),
            ("fc_dim", fc_dim, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer at least {lower}")
        if modes > self.resolution // 3:
            raise ValueError("PCNO modes must lie inside the retained Fourier band")
        geometry = periodic_grid_geometry(self.resolution)
        for name, value in geometry.items():
            self.register_buffer(name, value)
        self.register_buffer("train_scale", scale_tensor)
        self.register_buffer("forcing_shape", torch.cos(4.0 * self.nodes[:, 1:2]))
        fourier_modes = torch.as_tensor(
            compute_Fourier_modes(2, [modes, modes], [2 * math.pi, 2 * math.pi]),
            dtype=torch.float32,
        )
        self.pcno = PCNO(
            2,
            fourier_modes,
            nmeasures=1,
            layers=[width] * (depth + 1),
            fc_dim=fc_dim,
            in_dim=2,
            out_dim=1,
        )
        self._fourier_cache: tuple[torch.Tensor, ...] | None = None

    def _apply(self, fn, recurse: bool = True):
        # Cached tensors are not persistent state and never survive .to/.double.
        self._fourier_cache = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, vorticity: torch.Tensor) -> dict[str, torch.Tensor]:
        n = _state(vorticity)
        if n != self.resolution:
            raise ValueError("vorticity resolution does not match the model")
        if vorticity.device != self.nodes.device or vorticity.dtype != self.nodes.dtype:
            raise ValueError("vorticity device and dtype must match the model")
        batch = vorticity.shape[0]
        state = vorticity.reshape(batch, n * n, 1)
        features = torch.cat(
            (
                state / self.train_scale,
                self.forcing_shape.unsqueeze(0).expand(batch, -1, -1),
            ),
            dim=-1,
        )
        aux = tuple(
            value.unsqueeze(0).expand(batch, *value.shape)
            for value in (
                self.node_mask,
                self.nodes,
                self.node_weights,
                self.directed_edges,
                self.edge_gradient_weights,
            )
        )
        if self._fourier_cache is None:
            with torch.no_grad():
                self._fourier_cache = self.pcno.prepare_fourier_tensors(
                    self.nodes.unsqueeze(0), self.node_weights.unsqueeze(0)
                )
        fourier = tuple(
            value.expand(batch, *value.shape[1:]) for value in self._fourier_cache
        )
        update = self.pcno(features, aux, fourier_tensors=fourier).reshape(batch, n, n)
        raw_next = vorticity + self.train_scale * update
        next_state = canonicalize_vorticity(raw_next)
        return {
            "raw_next": raw_next,
            "next_state": next_state,
            "projection_residual": raw_next - next_state,
        }
