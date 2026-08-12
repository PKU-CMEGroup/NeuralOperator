"""Domain-compatible output link for the D089 REALM direct baseline."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BoxCoxDomainLink(nn.Module):
    """Map raw normalized outputs into selected inverse Box--Cox domains.

    For a transformed channel, define the normalized lower bound whose inverse
    base is ``base_floor``. The returned normalized value is that lower bound
    plus a shifted softplus. The shift maps raw zero to normalized zero.
    Unselected channels pass through exactly.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        *,
        transformed_channels: tuple[int, ...],
        box_cox_lambda: float,
        base_floor: float,
        channel_axis: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise TypeError("mean and scale must be torch tensors")
        if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
            raise ValueError("mean and scale must be matching channel vectors")
        if not mean.is_floating_point() or not scale.is_floating_point():
            raise TypeError("mean and scale must use floating dtypes")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise ValueError("mean and scale must be finite")
        if not bool((scale > 0.0).all()):
            raise ValueError("scale must be positive")
        if (
            not math.isfinite(box_cox_lambda)
            or box_cox_lambda <= 0.0
            or not math.isfinite(base_floor)
            or base_floor <= 0.0
        ):
            raise ValueError("lambda and base floor must be finite and positive")
        if isinstance(channel_axis, bool) or not isinstance(channel_axis, int):
            raise TypeError("channel axis must be an integer")
        channels = tuple(transformed_channels)
        if not channels:
            raise ValueError("at least one transformed channel is required")
        if len(set(channels)) != len(channels):
            raise ValueError("transformed channel indices must be unique")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= mean.numel()
            for index in channels
        ):
            raise ValueError("transformed channel index is out of range")

        lower = ((base_floor - 1.0) / box_cox_lambda - mean) / scale
        interior_distance = -lower[list(channels)]
        if not bool((interior_distance > 0.0).all()):
            raise ValueError("normalized zero must lie strictly inside every domain")
        shift = interior_distance + torch.log(-torch.expm1(-interior_distance))

        self.register_buffer("mean", mean.detach().clone(), persistent=True)
        self.register_buffer("scale", scale.detach().clone(), persistent=True)
        self.register_buffer(
            "normalized_lower_bound", lower.detach().clone(), persistent=True
        )
        self.register_buffer(
            "zero_center_shift", shift.detach().clone(), persistent=True
        )
        self.transformed_channels = channels
        self.box_cox_lambda = float(box_cox_lambda)
        self.base_floor = float(base_floor)
        self.channel_axis = channel_axis

    def _resolved_axis(self, values: torch.Tensor) -> int:
        axis = self.channel_axis
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError("channel axis is invalid for the output rank")
        if values.shape[axis] != self.mean.numel():
            raise ValueError("output channel count does not match link statistics")
        if values.dtype != self.mean.dtype or values.device != self.mean.device:
            raise ValueError("output and link statistics must share dtype and device")
        return axis

    def forward(self, raw_normalized: torch.Tensor) -> torch.Tensor:
        if not isinstance(raw_normalized, torch.Tensor):
            raise TypeError("raw normalized output must be a torch tensor")
        if not raw_normalized.is_floating_point():
            raise TypeError("raw normalized output must use a floating dtype")
        if not bool(torch.isfinite(raw_normalized).all()):
            raise RuntimeError("raw normalized output must be finite")
        axis = self._resolved_axis(raw_normalized)

        result = raw_normalized.clone()
        for local_index, index in enumerate(self.transformed_channels):
            selection = [slice(None)] * raw_normalized.ndim
            selection[axis] = index
            key = tuple(selection)
            result[key] = self.normalized_lower_bound[index] + F.softplus(
                raw_normalized[key] + self.zero_center_shift[local_index]
            )
        return result

    def inverse_box_cox_base(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return bases for transformed channels in their registered order."""

        axis = self._resolved_axis(normalized)
        indices = torch.as_tensor(
            self.transformed_channels,
            device=normalized.device,
            dtype=torch.long,
        )
        selected = torch.index_select(normalized, axis, indices)
        shape = [1] * normalized.ndim
        shape[axis] = len(self.transformed_channels)
        mean = self.mean[list(self.transformed_channels)].reshape(shape)
        scale = self.scale[list(self.transformed_channels)].reshape(shape)
        return self.box_cox_lambda * (selected * scale + mean) + 1.0


class DomainLinkedMap(nn.Module):
    """Apply a parameter-free domain link to any matching direct map."""

    def __init__(self, backbone: nn.Module, output_link: BoxCoxDomainLink) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_link = output_link
        self.config = getattr(backbone, "config", None)

    def forward_raw(
        self,
        state: torch.Tensor,
        static_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unlinked proposal for explicit margin diagnostics."""

        return self.backbone(state, static_coordinates)

    def forward(
        self,
        state: torch.Tensor,
        static_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        return self.output_link(self.forward_raw(state, static_coordinates))


__all__ = ["BoxCoxDomainLink", "DomainLinkedMap"]
