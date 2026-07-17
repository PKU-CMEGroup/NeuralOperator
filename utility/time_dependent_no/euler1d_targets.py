"""Solver-facing target adapters for 1D Euler models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from utility.time_dependent_no.euler1d import (
    Euler1DBatch,
    conservative_to_primitive,
    decode_primitive,
    normal_flux_from_primitive,
    primitive_to_conservative,
    rusanov_flux_from_primitive,
    update_from_face_flux,
)
from utility.time_dependent_no.fv import gather_cells


@dataclass(frozen=True)
class TargetPrediction:
    """Decoded next-state prediction plus optional solver-facing quantities."""

    primitive: torch.Tensor
    conservative: torch.Tensor
    aux: dict[str, Any]


def canonicalize_owner_oriented_face_flux(
    face_flux: torch.Tensor,
    face_normal: torch.Tensor,
) -> torch.Tensor:
    """Remove the constant-physical-flux null mode from a 1D face field."""

    if face_flux.ndim != 3:
        raise ValueError("face_flux must have shape [batch, faces, channels]")
    if face_normal.shape != (*face_flux.shape[:-1], 1):
        raise ValueError(
            "face_normal must have shape "
            f"{(*face_flux.shape[:-1], 1)}, got {tuple(face_normal.shape)}"
        )
    normal = face_normal.to(dtype=face_flux.dtype, device=face_flux.device)
    null_norm_sq = normal.square().sum(dim=1, keepdim=True)
    if torch.any(null_norm_sq <= 0.0):
        raise ValueError("face_normal must define a nonzero null vector")
    coefficient = (face_flux * normal).sum(dim=1, keepdim=True) / null_norm_sq
    return face_flux - normal * coefficient


class StateTargetAdapter(nn.Module):
    """Direct next-state / flow-map target."""

    def __init__(self, *, positive_transform: str = "none") -> None:
        super().__init__()
        self.positive_transform = positive_transform

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        if raw.shape != batch.current_primitive.shape:
            raise ValueError(
                "state target raw output must have shape "
                f"{tuple(batch.current_primitive.shape)}, got {tuple(raw.shape)}"
            )
        primitive = decode_primitive(raw, positive_transform=self.positive_transform)
        conservative = primitive_to_conservative(primitive, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={"target_kind": "state"},
        )


class ConservativeStateTargetAdapter(nn.Module):
    """Direct next conservative state without an admissibility transform."""

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        if raw.shape != batch.current_conservative.shape:
            raise ValueError(
                "conservative state target raw output must have shape "
                f"{tuple(batch.current_conservative.shape)}, got {tuple(raw.shape)}"
            )
        primitive = conservative_to_primitive(raw, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=raw,
            aux={
                "target_kind": "conservative_state",
                "proposed_conservative": raw,
                "raw_recurrence": True,
            },
        )


class ConservativeResidualTargetAdapter(nn.Module):
    """Cell-wise conservative residual followed by primitive decoding.

    The residual represents the full learned coarse-step increment
    ``U^{n+1} - U^n``. The model is not conditioned on ``dt``; different
    coarse steps should be trained as separate fixed-step operators.
    """

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        if raw.shape != batch.current_primitive.shape:
            raise ValueError(
                "residual target raw output must have shape "
                f"{tuple(batch.current_primitive.shape)}, got {tuple(raw.shape)}"
            )
        conservative = batch.current_conservative + raw
        primitive = conservative_to_primitive(conservative, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "residual",
                "conservative_delta": raw,
                "proposed_conservative": conservative,
                "raw_recurrence": True,
            },
        )


class ProjectedConservativeResidualTargetAdapter(nn.Module):
    """Conservative residual projected to a learned boundary budget.

    ``raw[:, :-1]`` is a cell-wise conservative increment proposal and the
    final token ``raw[:, -1]`` is the net conservative boundary exchange over
    the fixed coarse step. The proposal is shifted by the minimum volume-L2
    correction whose cell integral matches that exchange. A zero-gauge shared
    face impulse is then reconstructed by cumulative divergence; its finite-
    volume update is exactly the projected cell increment.

    The learned boundary exchange leaves the decoder expressive for open and
    wall boundaries. Conservation is structural in the interior, while global
    change is isolated in one identifiable three-component quantity.
    """

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        batch_size, num_cells, num_variables = batch.current_conservative.shape
        expected = (batch_size, num_cells + 1, num_variables)
        if raw.shape != expected:
            raise ValueError(
                "projected residual raw output must contain one boundary "
                f"exchange token with shape {expected}, got {tuple(raw.shape)}"
            )

        volume = batch.geometry.cell_volume.to(
            dtype=raw.dtype,
            device=raw.device,
        ).unsqueeze(-1)
        domain_volume = volume.sum(dim=1)

        raw_delta = raw[:, :num_cells]
        boundary_exchange = raw[:, num_cells]
        raw_cell_budget = (volume * raw_delta).sum(dim=1)
        projection_correction = (boundary_exchange - raw_cell_budget) / domain_volume
        projected_delta = raw_delta + projection_correction.unsqueeze(1)

        conservative = batch.current_conservative + projected_delta
        primitive = conservative_to_primitive(conservative, gamma=batch.gamma)

        # Work first in a common left-to-right physical orientation q, where
        # V_i * delta_i = q_i - q_{i+1}. Convert q to the owner-oriented face
        # convention used by FiniteVolumeGeometry only after fixing its gauge.
        # The face field is diagnostic, not a supervised output, so detach it
        # from the state-loss graph to avoid retaining a needless cumsum graph.
        diagnostic_delta = projected_delta.detach()
        physical_impulse = torch.cat(
            (
                torch.zeros_like(diagnostic_delta[:, :1]),
                -torch.cumsum(volume * diagnostic_delta, dim=1),
            ),
            dim=1,
        )
        physical_impulse = physical_impulse - physical_impulse.mean(
            dim=1,
            keepdim=True,
        )
        face_normal = batch.geometry.face_normal.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        face_impulse = physical_impulse * face_normal

        dt = batch.dt.to(dtype=raw.dtype, device=raw.device).reshape(
            batch_size,
            1,
            1,
        )
        face_area = batch.geometry.face_area.to(
            dtype=raw.dtype,
            device=raw.device,
        ).unsqueeze(-1)
        face_flux = face_impulse / (dt * face_area)

        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "projected_residual",
                "raw_conservative_delta": raw_delta,
                "conservative_delta": projected_delta,
                "raw_cell_budget": raw_cell_budget,
                "boundary_exchange": boundary_exchange,
                "projection_correction": projection_correction,
                "canonical_face_impulse": face_impulse,
                "face_flux": face_flux,
                "flux_gauge_mode": "canonical",
                "proposed_conservative": conservative,
                "raw_recurrence": True,
            },
        )


class PrimitiveResidualTargetAdapter(nn.Module):
    """Positive primitive residual over one fixed coarse step.

    The model predicts ``[delta_log_rho, delta_u, delta_log_p]``. Zero output
    is therefore the identity map, while decoded density and pressure remain
    positive for arbitrary raw outputs.
    """

    def __init__(
        self,
        *,
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        max_log_change: float = 10.0,
    ) -> None:
        super().__init__()
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)
        self.max_log_change = float(max_log_change)

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        if raw.shape != batch.current_primitive.shape:
            raise ValueError(
                "primitive residual target raw output must have shape "
                f"{tuple(batch.current_primitive.shape)}, got {tuple(raw.shape)}"
            )
        delta_log_rho = raw[..., 0].clamp(
            min=-self.max_log_change,
            max=self.max_log_change,
        )
        delta_velocity = raw[..., 1]
        delta_log_pressure = raw[..., 2].clamp(
            min=-self.max_log_change,
            max=self.max_log_change,
        )
        current_rho = batch.current_primitive[..., 0].clamp_min(self.rho_floor)
        current_velocity = batch.current_primitive[..., 1]
        current_pressure = batch.current_primitive[..., 2].clamp_min(
            self.pressure_floor
        )

        rho = self.rho_floor + (current_rho - self.rho_floor) * torch.exp(delta_log_rho)
        velocity = current_velocity + delta_velocity
        pressure = self.pressure_floor + (
            current_pressure - self.pressure_floor
        ) * torch.exp(delta_log_pressure)
        primitive = torch.stack((rho, velocity, pressure), dim=-1)
        conservative = primitive_to_conservative(primitive, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={"target_kind": "primitive_residual", "primitive_delta": raw},
        )


class ConservativeUpdateLimiter(nn.Module):
    """Scale a conservative update until the decoded state is admissible.

    ``scope='cell'`` gives each cell its own limiter coefficient. This matches
    the historical residual target behavior. ``scope='sample'`` applies one
    coefficient to the whole sample, which preserves finite-volume flux-pair
    accounting for flux/interface updates.
    """

    def __init__(
        self,
        *,
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        safety: float = 0.999,
        bisection_steps: int = 24,
        scope: str = "cell",
    ) -> None:
        super().__init__()
        if not (0.0 < safety <= 1.0):
            raise ValueError("safety must be in (0, 1]")
        if bisection_steps < 1:
            raise ValueError("bisection_steps must be >= 1")
        if scope not in ("cell", "sample"):
            raise ValueError("scope must be 'cell' or 'sample'")
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)
        self.safety = float(safety)
        self.bisection_steps = int(bisection_steps)
        self.scope = scope

    def forward(
        self,
        current: torch.Tensor,
        delta: torch.Tensor,
        *,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.admissible_fraction(current, delta, gamma=gamma)
        return current + self.scale_delta(delta, theta), theta

    def admissible_fraction(
        self,
        current: torch.Tensor,
        delta: torch.Tensor,
        *,
        gamma: float,
    ) -> torch.Tensor:
        if self.scope == "sample":
            return self._sample_fraction(current, delta, gamma=gamma)
        return self._cell_fraction(current, delta, gamma=gamma)

    @staticmethod
    def scale_delta(delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        scale = theta
        while scale.ndim < delta.ndim:
            scale = scale.unsqueeze(-1)
        return scale * delta

    def _cell_fraction(
        self,
        current: torch.Tensor,
        delta: torch.Tensor,
        *,
        gamma: float,
    ) -> torch.Tensor:
        hi = torch.ones_like(current[..., 0])
        trial_ok = _is_admissible_conservative(
            current + delta,
            gamma=gamma,
            rho_floor=self.rho_floor,
            pressure_floor=self.pressure_floor,
        )
        if bool(trial_ok.all()):
            return hi

        lo = torch.zeros_like(hi)
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            candidate = current + self.scale_delta(delta, mid)
            ok = _is_admissible_conservative(
                candidate,
                gamma=gamma,
                rho_floor=self.rho_floor,
                pressure_floor=self.pressure_floor,
            )
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        return lo * self.safety

    def _sample_fraction(
        self,
        current: torch.Tensor,
        delta: torch.Tensor,
        *,
        gamma: float,
    ) -> torch.Tensor:
        batch_size = current.shape[0]
        hi = torch.ones(batch_size, dtype=current.dtype, device=current.device)
        trial_ok = (
            _is_admissible_conservative(
                current + delta,
                gamma=gamma,
                rho_floor=self.rho_floor,
                pressure_floor=self.pressure_floor,
            )
            .reshape(batch_size, -1)
            .all(dim=1)
        )
        if bool(trial_ok.all()):
            return hi

        lo = torch.zeros_like(hi)
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            candidate = current + self.scale_delta(delta, mid)
            ok = (
                _is_admissible_conservative(
                    candidate,
                    gamma=gamma,
                    rho_floor=self.rho_floor,
                    pressure_floor=self.pressure_floor,
                )
                .reshape(batch_size, -1)
                .all(dim=1)
            )
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        return lo * self.safety


def _is_admissible_conservative(
    conservative: torch.Tensor,
    *,
    gamma: float,
    rho_floor: float,
    pressure_floor: float,
) -> torch.Tensor:
    rho, momentum, energy = conservative.unbind(dim=-1)
    velocity = momentum / rho
    pressure = (gamma - 1.0) * (energy - 0.5 * momentum * velocity)
    return (
        torch.isfinite(rho)
        & torch.isfinite(pressure)
        & rho.gt(rho_floor)
        & pressure.gt(pressure_floor)
    )


class LimitedConservativeResidualTargetAdapter(nn.Module):
    """Conservative residual with a shared cellwise admissibility limiter."""

    def __init__(
        self,
        *,
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        safety: float = 0.999,
        bisection_steps: int = 24,
    ) -> None:
        super().__init__()
        self.limiter = ConservativeUpdateLimiter(
            rho_floor=rho_floor,
            pressure_floor=pressure_floor,
            safety=safety,
            bisection_steps=bisection_steps,
            scope="cell",
        )
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        if raw.shape != batch.current_primitive.shape:
            raise ValueError(
                "limited residual target raw output must have shape "
                f"{tuple(batch.current_primitive.shape)}, got {tuple(raw.shape)}"
            )
        proposed = batch.current_conservative + raw
        conservative, theta = self.limiter(
            batch.current_conservative,
            raw,
            gamma=batch.gamma,
        )
        primitive = conservative_to_primitive(
            conservative,
            gamma=batch.gamma,
            rho_floor=self.rho_floor,
            pressure_floor=self.pressure_floor,
        )
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "limited_residual",
                "conservative_delta": raw,
                "proposed_conservative": proposed,
                "limiter_theta": theta,
                "limiter_scope": self.limiter.scope,
            },
        )


class FluxTargetAdapter(nn.Module):
    """Face-flux target followed by a fixed conservative FV update."""

    def __init__(self, *, flux_gauge_mode: str = "raw") -> None:
        super().__init__()
        if flux_gauge_mode not in ("raw", "canonical"):
            raise ValueError(f"unsupported flux gauge mode: {flux_gauge_mode}")
        self.flux_gauge_mode = flux_gauge_mode

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 3)
        if raw.shape != expected:
            raise ValueError(
                f"flux target raw output must have shape {expected}, got {tuple(raw.shape)}"
            )
        face_flux = raw
        if self.flux_gauge_mode == "canonical":
            face_flux = canonicalize_owner_oriented_face_flux(
                raw,
                batch.geometry.face_normal,
            )
        conservative = update_from_face_flux(batch, face_flux)
        primitive = conservative_to_primitive(conservative, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "flux",
                "face_flux": face_flux,
                "flux_gauge_mode": self.flux_gauge_mode,
                "proposed_conservative": conservative,
                "raw_recurrence": True,
            },
        )


class LimitedFluxTargetAdapter(nn.Module):
    """Face-flux target with a samplewise conservative admissibility limiter."""

    def __init__(
        self,
        *,
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        safety: float = 0.999,
        bisection_steps: int = 24,
    ) -> None:
        super().__init__()
        self.limiter = ConservativeUpdateLimiter(
            rho_floor=rho_floor,
            pressure_floor=pressure_floor,
            safety=safety,
            bisection_steps=bisection_steps,
            scope="sample",
        )
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 3)
        if raw.shape != expected:
            raise ValueError(
                "limited flux target raw output must have shape "
                f"{expected}, got {tuple(raw.shape)}"
            )
        proposed = update_from_face_flux(batch, raw)
        delta = proposed - batch.current_conservative
        conservative, theta = self.limiter(
            batch.current_conservative,
            delta,
            gamma=batch.gamma,
        )
        primitive = conservative_to_primitive(
            conservative,
            gamma=batch.gamma,
            rho_floor=self.rho_floor,
            pressure_floor=self.pressure_floor,
        )
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "limited_flux",
                "face_flux": raw,
                "conservative_delta": delta,
                "proposed_conservative": proposed,
                "limiter_theta": theta,
                "limiter_scope": self.limiter.scope,
            },
        )


class PhysicalFluxCorrectionTargetAdapter(nn.Module):
    """Base Rusanov flux plus a bounded learned face-flux correction."""

    def __init__(
        self,
        *,
        correction_scale: float = 1.0,
        scale_floor: float = 1.0e-6,
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        safety: float = 0.999,
        bisection_steps: int = 24,
    ) -> None:
        super().__init__()
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be nonnegative")
        if scale_floor <= 0.0:
            raise ValueError("scale_floor must be positive")
        self.correction_scale = float(correction_scale)
        self.scale_floor = float(scale_floor)
        self.limiter = ConservativeUpdateLimiter(
            rho_floor=rho_floor,
            pressure_floor=pressure_floor,
            safety=safety,
            bisection_steps=bisection_steps,
            scope="sample",
        )
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 3)
        if raw.shape != expected:
            raise ValueError(
                "physical flux correction target raw output must have shape "
                f"{expected}, got {tuple(raw.shape)}"
            )
        owner_primitive, neighbor_primitive = owner_neighbor_primitives(batch)
        base_flux = rusanov_flux_from_primitive(
            owner_primitive,
            neighbor_primitive,
            batch.geometry.face_normal,
            gamma=batch.gamma,
        )
        local_scale = _rusanov_correction_scale(
            owner_primitive,
            neighbor_primitive,
            batch.geometry.face_normal,
            gamma=batch.gamma,
            scale_floor=self.scale_floor,
        )
        correction_bound = self.correction_scale * local_scale
        correction = correction_bound * torch.tanh(raw)
        face_flux = base_flux + correction
        proposed = update_from_face_flux(batch, face_flux)
        delta = proposed - batch.current_conservative
        conservative, theta = self.limiter(
            batch.current_conservative,
            delta,
            gamma=batch.gamma,
        )
        primitive = conservative_to_primitive(
            conservative,
            gamma=batch.gamma,
            rho_floor=self.rho_floor,
            pressure_floor=self.pressure_floor,
        )
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "physical_flux_correction",
                "base_face_flux": base_flux,
                "face_flux": face_flux,
                "flux_correction": correction,
                "flux_correction_scale": local_scale,
                "flux_correction_bound": correction_bound,
                "conservative_delta": delta,
                "proposed_conservative": proposed,
                "limiter_theta": theta,
                "limiter_scope": self.limiter.scope,
            },
        )


class InterfaceStateTargetAdapter(nn.Module):
    """Interface-state target followed by Rusanov flux and FV update."""

    def __init__(self, *, positive_transform: str = "none") -> None:
        super().__init__()
        self.positive_transform = positive_transform

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 2, 3)
        if raw.shape != expected:
            raise ValueError(
                "interface target raw output must have shape "
                f"{expected}, got {tuple(raw.shape)}"
            )
        interface_primitive = decode_primitive(
            raw,
            positive_transform=self.positive_transform,
        )
        owner_primitive = interface_primitive[..., 0, :]
        neighbor_primitive = interface_primitive[..., 1, :]
        face_flux = rusanov_flux_from_primitive(
            owner_primitive,
            neighbor_primitive,
            batch.geometry.face_normal,
            gamma=batch.gamma,
        )
        conservative = update_from_face_flux(batch, face_flux)
        primitive = conservative_to_primitive(
            conservative,
            gamma=batch.gamma,
            rho_floor=1.0e-8,
            pressure_floor=1.0e-8,
        )
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "interface",
                "interface_primitive": interface_primitive,
                "face_flux": face_flux,
            },
        )


def decode_relative_interface_traces(
    raw: torch.Tensor,
    batch: Euler1DBatch,
) -> torch.Tensor:
    """Decode directed multiplicative primitive corrections around local traces."""

    expected = (*batch.geometry.face_owner.shape, 2, 3)
    if raw.shape != expected:
        raise ValueError(
            "relative interface output must have shape "
            f"{expected}, got {tuple(raw.shape)}"
        )
    owner_base, neighbor_base = owner_neighbor_primitives(batch)
    base = torch.stack((owner_base, neighbor_base), dim=-2)
    rho, velocity, pressure = base.unbind(dim=-1)
    delta_log_rho, delta_velocity, delta_log_pressure = raw.unbind(dim=-1)
    sound_speed = torch.sqrt(
        (batch.gamma * pressure / rho).clamp_min(0.0)
    )
    traces = torch.stack(
        (
            rho * torch.exp(delta_log_rho),
            velocity + sound_speed * delta_velocity,
            pressure * torch.exp(delta_log_pressure),
        ),
        dim=-1,
    )

    normal = batch.geometry.face_normal.squeeze(-1)
    boundary = batch.geometry.face_neighbor.lt(0)
    left_boundary = boundary & normal.lt(0)
    right_boundary = boundary & normal.gt(0)
    owner_trace = traces[..., 0, :]
    neighbor_trace = traces[..., 1, :]
    neighbor_trace = torch.where(
        left_boundary.unsqueeze(-1),
        neighbor_base,
        neighbor_trace,
    )
    reflected_owner = owner_trace.clone()
    reflected_owner[..., 1] = -reflected_owner[..., 1]
    if batch.right_boundary_primitive is not None:
        right_state = batch.right_boundary_primitive.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        reflected_owner = torch.where(
            right_boundary.unsqueeze(-1),
            right_state.unsqueeze(1),
            reflected_owner,
        )
    neighbor_trace = torch.where(
        right_boundary.unsqueeze(-1),
        reflected_owner,
        neighbor_trace,
    )
    return torch.stack((owner_trace, neighbor_trace), dim=-2)


class RelativeInterfaceStateTargetAdapter(nn.Module):
    """Decode global relative traces, one shared flux, and an exact FV update."""

    def __init__(self, *, flux_mode: str = "rusanov") -> None:
        super().__init__()
        if flux_mode not in ("rusanov", "central"):
            raise ValueError(f"unsupported interface flux mode: {flux_mode}")
        self.flux_mode = flux_mode

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        interface_primitive = decode_relative_interface_traces(raw, batch)
        owner_primitive = interface_primitive[..., 0, :]
        neighbor_primitive = interface_primitive[..., 1, :]
        if self.flux_mode == "rusanov":
            face_flux = rusanov_flux_from_primitive(
                owner_primitive,
                neighbor_primitive,
                batch.geometry.face_normal,
                gamma=batch.gamma,
            )
        else:
            owner_flux = normal_flux_from_primitive(
                owner_primitive,
                batch.geometry.face_normal,
                gamma=batch.gamma,
            )
            neighbor_flux = normal_flux_from_primitive(
                neighbor_primitive,
                batch.geometry.face_normal,
                gamma=batch.gamma,
            )
            face_flux = 0.5 * (owner_flux + neighbor_flux)
        conservative = update_from_face_flux(batch, face_flux)
        primitive = conservative_to_primitive(conservative, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "relative_interface",
                "interface_corrections": raw,
                "interface_primitive": interface_primitive,
                "interface_flux_mode": self.flux_mode,
                "face_flux": face_flux,
                "proposed_conservative": conservative,
                "raw_recurrence": True,
            },
        )


class CPGNetInterfaceTargetAdapter(nn.Module):
    """Decode positive interface states with no cell-state limiter or floor."""

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 2, 3)
        if raw.shape != expected:
            raise ValueError(
                "CPGNet interface output has the wrong shape: "
                f"{tuple(raw.shape)} != {expected}"
            )
        owner_primitive = raw[..., 0, :]
        neighbor_primitive = raw[..., 1, :]
        face_flux = rusanov_flux_from_primitive(
            owner_primitive,
            neighbor_primitive,
            batch.geometry.face_normal,
            gamma=batch.gamma,
        )
        conservative = update_from_face_flux(batch, face_flux)
        primitive = conservative_to_primitive(conservative, gamma=batch.gamma)
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "cpg_interface",
                "interface_primitive": raw,
                "face_flux": face_flux,
                "proposed_conservative": conservative,
                "raw_recurrence": True,
            },
        )


class PositiveLimitedInterfaceStateTargetAdapter(nn.Module):
    """Positive interface states with Rusanov flux and update limiting."""

    def __init__(
        self,
        *,
        positive_transform: str = "softplus",
        rho_floor: float = 1.0e-8,
        pressure_floor: float = 1.0e-8,
        safety: float = 0.999,
        bisection_steps: int = 24,
    ) -> None:
        super().__init__()
        self.positive_transform = (
            "softplus" if positive_transform == "none" else positive_transform
        )
        self.limiter = ConservativeUpdateLimiter(
            rho_floor=rho_floor,
            pressure_floor=pressure_floor,
            safety=safety,
            bisection_steps=bisection_steps,
            scope="sample",
        )
        self.rho_floor = float(rho_floor)
        self.pressure_floor = float(pressure_floor)

    def forward(self, raw: torch.Tensor, batch: Euler1DBatch) -> TargetPrediction:
        expected = (*batch.geometry.face_owner.shape, 2, 3)
        if raw.shape != expected:
            raise ValueError(
                "positive limited interface target raw output must have shape "
                f"{expected}, got {tuple(raw.shape)}"
            )
        interface_primitive = decode_primitive(
            raw,
            positive_transform=self.positive_transform,
        )
        owner_primitive = interface_primitive[..., 0, :]
        neighbor_primitive = interface_primitive[..., 1, :]
        face_flux = rusanov_flux_from_primitive(
            owner_primitive,
            neighbor_primitive,
            batch.geometry.face_normal,
            gamma=batch.gamma,
        )
        proposed = update_from_face_flux(batch, face_flux)
        delta = proposed - batch.current_conservative
        conservative, theta = self.limiter(
            batch.current_conservative,
            delta,
            gamma=batch.gamma,
        )
        primitive = conservative_to_primitive(
            conservative,
            gamma=batch.gamma,
            rho_floor=self.rho_floor,
            pressure_floor=self.pressure_floor,
        )
        return TargetPrediction(
            primitive=primitive,
            conservative=conservative,
            aux={
                "target_kind": "positive_limited_interface",
                "interface_primitive": interface_primitive,
                "face_flux": face_flux,
                "conservative_delta": delta,
                "proposed_conservative": proposed,
                "limiter_theta": theta,
                "limiter_scope": self.limiter.scope,
            },
        )


def make_target_adapter(
    target: str,
    *,
    positive_transform: str = "none",
    flux_correction_scale: float = 1.0,
    flux_correction_scale_floor: float = 1.0e-6,
    flux_gauge_mode: str = "raw",
    interface_flux_mode: str = "rusanov",
) -> nn.Module:
    """Factory for target adapters used by FNO and CPG-style heads."""

    if target == "state":
        return StateTargetAdapter(positive_transform=positive_transform)
    if target == "conservative_state":
        return ConservativeStateTargetAdapter()
    if target == "residual":
        return ConservativeResidualTargetAdapter()
    if target == "projected_residual":
        return ProjectedConservativeResidualTargetAdapter()
    if target == "primitive_residual":
        return PrimitiveResidualTargetAdapter()
    if target == "limited_residual":
        return LimitedConservativeResidualTargetAdapter()
    if target == "flux":
        return FluxTargetAdapter(flux_gauge_mode=flux_gauge_mode)
    if target == "limited_flux":
        return LimitedFluxTargetAdapter()
    if target == "physical_flux_correction":
        return PhysicalFluxCorrectionTargetAdapter(
            correction_scale=flux_correction_scale,
            scale_floor=flux_correction_scale_floor,
        )
    if target == "interface":
        return InterfaceStateTargetAdapter(positive_transform=positive_transform)
    if target == "relative_interface":
        return RelativeInterfaceStateTargetAdapter(flux_mode=interface_flux_mode)
    if target == "positive_limited_interface":
        return PositiveLimitedInterfaceStateTargetAdapter(
            positive_transform=positive_transform
        )
    if target == "cpg_interface":
        return CPGNetInterfaceTargetAdapter()
    raise ValueError(f"unsupported target: {target}")


def owner_neighbor_primitives(batch: Euler1DBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current owner and neighbor/exterior primitive states per face."""

    owner = gather_cells(batch.current_primitive, batch.geometry.face_owner)
    neighbor = gather_cells(
        batch.current_primitive,
        batch.geometry.face_neighbor,
        fill_value=0.0,
    )
    exterior = _boundary_exterior_primitives(batch, owner)
    boundary = batch.geometry.face_neighbor.lt(0).unsqueeze(-1)
    return owner, torch.where(boundary, exterior, neighbor)


def _rusanov_correction_scale(
    owner_primitive: torch.Tensor,
    neighbor_primitive: torch.Tensor,
    normal: torch.Tensor,
    *,
    gamma: float,
    scale_floor: float,
) -> torch.Tensor:
    """Componentwise physical flux scale for bounded learned corrections."""

    owner_cons = primitive_to_conservative(owner_primitive, gamma=gamma)
    neighbor_cons = primitive_to_conservative(neighbor_primitive, gamma=gamma)
    owner_flux = normal_flux_from_primitive(owner_primitive, normal, gamma=gamma)
    neighbor_flux = normal_flux_from_primitive(neighbor_primitive, normal, gamma=gamma)

    n = normal.squeeze(-1)
    rho_l, u_l, p_l = owner_primitive.unbind(dim=-1)
    rho_r, u_r, p_r = neighbor_primitive.unbind(dim=-1)
    c_l = torch.sqrt((gamma * p_l / rho_l).clamp_min(0.0))
    c_r = torch.sqrt((gamma * p_r / rho_r).clamp_min(0.0))
    speed = torch.maximum(
        (u_l * n).abs() + c_l * n.abs(),
        (u_r * n).abs() + c_r * n.abs(),
    )

    advective_scale = 0.5 * (owner_flux.abs() + neighbor_flux.abs())
    jump_scale = 0.5 * speed.unsqueeze(-1) * (neighbor_cons - owner_cons).abs()
    return (advective_scale + jump_scale).clamp_min(scale_floor)


def _boundary_exterior_primitives(
    batch: Euler1DBatch,
    owner_primitive: torch.Tensor,
) -> torch.Tensor:
    """Build exterior states for 1D left inflow and right reflective faces."""

    exterior = owner_primitive.clone()
    left_boundary = batch.geometry.face_neighbor.lt(
        0
    ) & batch.geometry.face_normal.squeeze(-1).lt(0)
    right_boundary = batch.geometry.face_neighbor.lt(
        0
    ) & batch.geometry.face_normal.squeeze(-1).gt(0)

    if batch.left_boundary_primitive is not None:
        left_state = batch.left_boundary_primitive.to(
            dtype=owner_primitive.dtype,
            device=owner_primitive.device,
        )
        exterior = torch.where(
            left_boundary.unsqueeze(-1), left_state.unsqueeze(1), exterior
        )

    reflected = owner_primitive.clone()
    reflected[..., 1] = -reflected[..., 1]
    if batch.right_boundary_primitive is not None:
        right_state = batch.right_boundary_primitive.to(
            dtype=owner_primitive.dtype,
            device=owner_primitive.device,
        )
        reflected = torch.where(
            right_boundary.unsqueeze(-1), right_state.unsqueeze(1), reflected
        )
    exterior = torch.where(right_boundary.unsqueeze(-1), reflected, exterior)
    return exterior
