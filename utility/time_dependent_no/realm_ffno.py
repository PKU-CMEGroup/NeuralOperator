"""Independent factorized Fourier operator used by the D088 P1c smoke.

The implementation follows the frozen FFNO-M tensor contract without importing
or copying the official REALM package. Inputs and outputs are normalized states
with shape ``[batch, channel, y, x]``; coordinates are static released-grid
channels with the same spatial shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_GROUPS,
    ChannelGroups,
)


@dataclass(frozen=True)
class RealmFFNOConfig:
    state_channels: int = 12
    coordinate_channels: int = 2
    output_channels: int = 12
    width: int = 128
    layers: int = 4
    modes_y: int = 32
    modes_x: int = 32
    feedforward_factor: int = 4
    feedforward_layers: int = 2
    layer_norm: bool = True
    head_width: int = 128

    def __post_init__(self) -> None:
        integer_fields = (
            self.state_channels,
            self.coordinate_channels,
            self.output_channels,
            self.width,
            self.layers,
            self.modes_y,
            self.modes_x,
            self.feedforward_factor,
            self.feedforward_layers,
            self.head_width,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields
        ):
            raise TypeError("FFNO dimensions must be integers")
        if any(value <= 0 for value in integer_fields):
            raise ValueError("FFNO dimensions must be positive")

    @property
    def input_channels(self) -> int:
        return self.state_channels + self.coordinate_channels


class _PositionwiseMLP(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        factor: int,
        layers: int,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index in range(layers):
            input_width = width if index == 0 else width * factor
            output_width = width if index == layers - 1 else width * factor
            modules.append(nn.Linear(input_width, output_width))
            modules.append(
                nn.ReLU(inplace=True) if index < layers - 1 else nn.Identity()
            )
            if index == layers - 1:
                modules.append(
                    nn.LayerNorm(output_width) if layer_norm else nn.Identity()
                )
        self.layers = nn.Sequential(*modules)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _FactorizedSpectralBlock2d(nn.Module):
    """One-dimensional Fourier mixing along each Cartesian axis, then an MLP."""

    def __init__(self, config: RealmFFNOConfig) -> None:
        super().__init__()
        shape_x = (config.width, config.width, config.modes_x, 2)
        shape_y = (config.width, config.width, config.modes_y, 2)
        self.weight_x = nn.Parameter(torch.empty(shape_x))
        self.weight_y = nn.Parameter(torch.empty(shape_y))
        nn.init.xavier_normal_(self.weight_x)
        nn.init.xavier_normal_(self.weight_y)
        self.modes_x = config.modes_x
        self.modes_y = config.modes_y
        self.width = config.width
        self.backcast = _PositionwiseMLP(
            config.width,
            factor=config.feedforward_factor,
            layers=config.feedforward_layers,
            layer_norm=config.layer_norm,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-1] != self.width:
            raise ValueError("spectral block requires [batch, y, x, width]")
        channels_first = values.permute(0, 3, 1, 2)
        batch, _, height, width = channels_first.shape
        if self.modes_x > width // 2 + 1 or self.modes_y > height // 2 + 1:
            raise ValueError("registered Fourier modes exceed the input-grid support")

        spectrum_x = torch.fft.rfft(channels_first, dim=-1, norm="ortho")
        mixed_x = spectrum_x.new_zeros(batch, self.width, height, width // 2 + 1)
        mixed_x[..., : self.modes_x] = torch.einsum(
            "bcyk,cok->boyk",
            spectrum_x[..., : self.modes_x],
            torch.view_as_complex(self.weight_x),
        )
        physical_x = torch.fft.irfft(mixed_x, n=width, dim=-1, norm="ortho")

        spectrum_y = torch.fft.rfft(channels_first, dim=-2, norm="ortho")
        mixed_y = spectrum_y.new_zeros(batch, self.width, height // 2 + 1, width)
        mixed_y[:, :, : self.modes_y] = torch.einsum(
            "bckx,cok->bokx",
            spectrum_y[:, :, : self.modes_y],
            torch.view_as_complex(self.weight_y),
        )
        physical_y = torch.fft.irfft(mixed_y, n=height, dim=-2, norm="ortho")

        factorized = (physical_x + physical_y).permute(0, 2, 3, 1)
        return self.backcast(factorized)


class RealmFFNO2d(nn.Module):
    """FFNO-M direct-state map with static Cartesian coordinate channels."""

    def __init__(self, config: RealmFFNOConfig | None = None) -> None:
        super().__init__()
        config = RealmFFNOConfig() if config is None else config
        self.config = config
        self.input_projection = nn.Linear(config.input_channels, config.width)
        self.blocks = nn.ModuleList(
            _FactorizedSpectralBlock2d(config) for _ in range(config.layers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(config.width, config.head_width),
            nn.GELU(),
            nn.Linear(config.head_width, config.output_channels),
        )

    def forward(
        self,
        state: torch.Tensor,
        static_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1] != self.config.state_channels:
            raise ValueError("state must have shape [batch, state_channels, y, x]")
        if static_coordinates.ndim != 4 or (
            static_coordinates.shape[1] != self.config.coordinate_channels
        ):
            raise ValueError(
                "static_coordinates must have shape [case, coordinate_channels, y, x]"
            )
        if static_coordinates.shape[0] not in (1, state.shape[0]):
            raise ValueError(
                "coordinate case count must be one or match the state batch"
            )
        if static_coordinates.shape[2:] != state.shape[2:]:
            raise ValueError("state and coordinate spatial shapes must match")
        if (
            state.dtype != static_coordinates.dtype
            or state.device != static_coordinates.device
        ):
            raise ValueError("state and coordinates must share dtype and device")
        if not state.is_floating_point() or not static_coordinates.is_floating_point():
            raise TypeError("state and coordinates must be floating tensors")

        coordinates = static_coordinates.expand(state.shape[0], -1, -1, -1)
        hidden = torch.cat((state, coordinates), dim=1).permute(0, 2, 3, 1)
        hidden = F.gelu(self.input_projection(hidden))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_projection(hidden).permute(0, 3, 1, 2)


def normalize_realm_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Apply the pinned released-source coordinate normalization exactly.

    Each coordinate channel subtracts its own minimum. Both channels divide by
    the full range of channel zero, preserving their relative physical scale.
    """

    if coordinates.ndim != 4 or coordinates.shape[0] != 1:
        raise ValueError("coordinates must have shape [1, dim, y, x]")
    if not coordinates.is_floating_point() or not bool(
        torch.isfinite(coordinates).all()
    ):
        raise ValueError("coordinates must be finite floating values")
    spatial_axes = tuple(range(2, coordinates.ndim))
    minima = coordinates.amin(dim=spatial_axes, keepdim=True)
    axis_range = coordinates[:, :1].amax() - coordinates[:, :1].amin()
    if not bool(torch.isfinite(axis_range)) or float(axis_range.item()) <= 0.0:
        raise ValueError("coordinate channel zero must have positive finite range")
    return (coordinates - minima) / axis_range


def grouped_next_state_mse(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    groups: ChannelGroups = IGNITHIT_GROUPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum the registered group MSEs for one normalized next-state batch."""

    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("prediction and truth must share [batch, channel, y, x]")
    if not prediction.is_floating_point() or not truth.is_floating_point():
        raise TypeError("prediction and truth must be floating tensors")
    slices = groups.slices(expected_channels=prediction.shape[1])
    by_group: dict[str, torch.Tensor] = {}
    total = prediction.new_zeros(())
    for name, channel_slice in slices.items():
        if channel_slice.start == channel_slice.stop:
            continue
        value = (prediction[:, channel_slice] - truth[:, channel_slice]).square().mean()
        by_group[name] = value
        total = total + value
    return total, by_group


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def parameter_count_within_reported_tolerance(
    count: int,
    *,
    reported_count: int = 8_936_500,
    relative_tolerance: float = 0.005,
) -> bool:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("parameter count must be a positive integer")
    if (
        reported_count <= 0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
    ):
        raise ValueError("reported count and tolerance must be valid")
    return abs(count - reported_count) <= relative_tolerance * reported_count


__all__ = [
    "RealmFFNO2d",
    "RealmFFNOConfig",
    "grouped_next_state_mse",
    "normalize_realm_coordinates",
    "parameter_count_within_reported_tolerance",
    "trainable_parameter_count",
]
