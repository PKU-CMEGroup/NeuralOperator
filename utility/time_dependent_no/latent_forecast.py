"""Bounded latent-state diagnostics for the 1D Euler flow-map project.

This module deliberately separates a persistent latent recurrence from the
existing physical-state recurrence used by the Line-1 models.  In particular,
``Euler1DLatentContext`` contains geometry and boundary metadata but no current
or target state, so a transition cannot silently decode and re-encode.

The first implementation stage is analytic: identity, volume-weighted POD, a
truth-derived multi-front registration oracle, conditional-future diagnostics,
and an intervention ledger.  Learned representations are added only if the
oracle gate shows that phase coordinates improve the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from torch import nn

from utility.time_dependent_no.euler1d import (
    Euler1DBatch,
    conservative_to_primitive,
)
from utility.time_dependent_no.euler1d_data import (
    conservative_to_primitive_np,
    primitive_to_conservative_np,
)


DecoderConstraint = Literal["none", "coarse_conservative"]


@dataclass(frozen=True)
class Euler1DLatentContext:
    """Geometry and case metadata available to a latent transition.

    The omission of the current and target states is intentional.  ``dt`` is
    supplied separately on every transition call because composed-timestep
    tests may use a different sequence with the same geometry.
    """

    cell_centers: torch.Tensor
    cell_volume: torch.Tensor
    gamma: float
    left_boundary_primitive: torch.Tensor
    right_initial_primitive: torch.Tensor

    def __post_init__(self) -> None:
        if self.cell_centers.ndim != 3 or self.cell_centers.shape[-1] != 1:
            raise ValueError("cell_centers must have shape [batch, cells, 1]")
        if self.cell_volume.shape != self.cell_centers.shape[:2]:
            raise ValueError("cell_volume must have shape [batch, cells]")
        expected_boundary = (self.cell_centers.shape[0], 3)
        if self.left_boundary_primitive.shape != expected_boundary:
            raise ValueError("left boundary must have shape [batch, 3]")
        if self.right_initial_primitive.shape != expected_boundary:
            raise ValueError("right initial state must have shape [batch, 3]")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than one")

    @classmethod
    def from_batch(cls, batch: Euler1DBatch) -> "Euler1DLatentContext":
        if batch.left_boundary_primitive is None:
            raise ValueError("latent context requires the left boundary state")
        if batch.right_initial_primitive is None:
            raise ValueError("latent context requires the right initial state")
        return cls(
            cell_centers=batch.geometry.cell_centers,
            cell_volume=batch.geometry.cell_volume,
            gamma=batch.gamma,
            left_boundary_primitive=batch.left_boundary_primitive,
            right_initial_primitive=batch.right_initial_primitive,
        )

    def to(self, device: torch.device | str) -> "Euler1DLatentContext":
        return Euler1DLatentContext(
            cell_centers=self.cell_centers.to(device),
            cell_volume=self.cell_volume.to(device),
            gamma=self.gamma,
            left_boundary_primitive=self.left_boundary_primitive.to(device),
            right_initial_primitive=self.right_initial_primitive.to(device),
        )


@dataclass(frozen=True)
class LatentLayout:
    """Declared spatial/global layout of one latent code."""

    spatial_tokens: int
    spatial_channels: int
    global_tokens: int = 0
    global_channels: int = 0

    def __post_init__(self) -> None:
        values = (
            self.spatial_tokens,
            self.spatial_channels,
            self.global_tokens,
            self.global_channels,
        )
        if any(value < 0 for value in values):
            raise ValueError("latent layout dimensions must be nonnegative")
        if self.scalar_size < 1:
            raise ValueError("latent layout must contain at least one scalar")

    @property
    def spatial_size(self) -> int:
        return self.spatial_tokens * self.spatial_channels

    @property
    def global_size(self) -> int:
        return self.global_tokens * self.global_channels

    @property
    def scalar_size(self) -> int:
        return self.spatial_size + self.global_size


@dataclass(frozen=True)
class LatentCode:
    """Flat storage plus an explicit spatial/global interpretation."""

    values: torch.Tensor
    layout: LatentLayout

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("latent values must have shape [batch, scalars]")
        if self.values.shape[1] != self.layout.scalar_size:
            raise ValueError("latent values do not match the declared layout")

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def spatial(self) -> torch.Tensor:
        end = self.layout.spatial_size
        return self.values[:, :end].reshape(
            self.batch_size,
            self.layout.spatial_tokens,
            self.layout.spatial_channels,
        )

    def global_features(self) -> torch.Tensor:
        start = self.layout.spatial_size
        return self.values[:, start:].reshape(
            self.batch_size,
            self.layout.global_tokens,
            self.layout.global_channels,
        )


@dataclass(frozen=True)
class LatentDecode:
    conservative: torch.Tensor
    primitive: torch.Tensor
    decoder_constraint: DecoderConstraint
    diagnostics: dict[str, torch.Tensor | float | bool]


class Euler1DRepresentation(Protocol):
    layout: LatentLayout

    def encode(
        self,
        conservative: torch.Tensor,
        context: Euler1DLatentContext,
    ) -> LatentCode: ...

    def decode(
        self,
        code: LatentCode,
        context: Euler1DLatentContext,
        *,
        decoder_constraint: DecoderConstraint = "none",
    ) -> LatentDecode: ...


class Euler1DLatentTransition(Protocol):
    def __call__(
        self,
        code: LatentCode,
        context: Euler1DLatentContext,
        dt: torch.Tensor,
    ) -> LatentCode: ...


class IdentityRepresentation(nn.Module):
    """No-compression reference with exact conservative recurrence state."""

    def __init__(self, num_cells: int) -> None:
        super().__init__()
        if num_cells < 1:
            raise ValueError("num_cells must be positive")
        self.num_cells = int(num_cells)
        self.layout = LatentLayout(
            spatial_tokens=self.num_cells,
            spatial_channels=3,
        )

    def encode(
        self,
        conservative: torch.Tensor,
        context: Euler1DLatentContext,
    ) -> LatentCode:
        expected = (context.cell_centers.shape[0], self.num_cells, 3)
        if conservative.shape != expected:
            raise ValueError(f"conservative must have shape {expected}")
        return LatentCode(conservative.reshape(conservative.shape[0], -1), self.layout)

    def decode(
        self,
        code: LatentCode,
        context: Euler1DLatentContext,
        *,
        decoder_constraint: DecoderConstraint = "none",
    ) -> LatentDecode:
        if decoder_constraint != "none":
            raise ValueError("identity representation has no constrained decoder")
        if code.layout != self.layout:
            raise ValueError("latent layout does not match identity representation")
        conservative = code.values.reshape(code.batch_size, self.num_cells, 3)
        primitive = conservative_to_primitive(conservative, gamma=context.gamma)
        return LatentDecode(
            conservative=conservative,
            primitive=primitive,
            decoder_constraint="none",
            diagnostics={},
        )


@dataclass(frozen=True)
class LatentStepRecord:
    step: int
    physical_time: float
    dt: float
    encode_calls: int
    transition_calls: int
    decode_calls: int
    decoder_constraint: DecoderConstraint
    decode_reencode: bool
    clipping: bool
    density_floor: bool
    pressure_floor: bool
    limiter: bool
    projection: bool
    reset: bool
    truth_replacement: bool
    analysis: bool
    per_cycle_fit: bool
    finite: bool
    admissible: bool
    failure_cause: str | None


@dataclass(frozen=True)
class LatentRollout:
    codes: tuple[LatentCode, ...]
    decoded: tuple[LatentDecode, ...]
    records: tuple[LatentStepRecord, ...]
    valid_length: int
    failure_cause: str | None


def rollout_latent_raw(
    representation: Euler1DRepresentation,
    transition: Euler1DLatentTransition,
    initial_conservative: torch.Tensor,
    context: Euler1DLatentContext,
    dt_sequence: torch.Tensor,
    *,
    decoder_constraint: DecoderConstraint = "none",
) -> LatentRollout:
    """Roll out a persistent code and terminate on the first raw invalid decode."""

    if dt_sequence.ndim == 1:
        dt_sequence = dt_sequence.unsqueeze(0).expand(initial_conservative.shape[0], -1)
    if dt_sequence.ndim != 2 or dt_sequence.shape[0] != initial_conservative.shape[0]:
        raise ValueError("dt_sequence must have shape [steps] or [batch, steps]")
    if not bool(torch.isfinite(dt_sequence).all()):
        raise ValueError("all timesteps must be finite")
    if torch.any(dt_sequence <= 0):
        raise ValueError("all timesteps must be positive")

    if dt_sequence.shape[0] > 1 and not torch.equal(
        dt_sequence, dt_sequence[:1].expand_as(dt_sequence)
    ):
        raise ValueError(
            "batched rollout requires one shared timestep sequence for scalar ledger time"
        )

    code = representation.encode(initial_conservative, context)
    if not bool(torch.isfinite(code.values).all()):
        failure_cause = "nonfinite_initial_latent_code"
        record = LatentStepRecord(
            step=0,
            physical_time=0.0,
            dt=0.0,
            encode_calls=1,
            transition_calls=0,
            decode_calls=0,
            decoder_constraint=decoder_constraint,
            decode_reencode=False,
            clipping=False,
            density_floor=False,
            pressure_floor=False,
            limiter=False,
            projection=False,
            reset=False,
            truth_replacement=False,
            analysis=False,
            per_cycle_fit=False,
            finite=False,
            admissible=False,
            failure_cause=failure_cause,
        )
        return LatentRollout(
            codes=(code,),
            decoded=(),
            records=(record,),
            valid_length=0,
            failure_cause=failure_cause,
        )
    codes = [code]
    decoded_states: list[LatentDecode] = []
    records: list[LatentStepRecord] = []
    physical_time = 0.0
    failure_cause: str | None = None

    for step in range(dt_sequence.shape[1]):
        dt = dt_sequence[:, step]
        code = transition(code, context, dt)
        if not isinstance(code, LatentCode):
            raise TypeError("latent transition must return LatentCode")
        representative_dt = float(dt[0].detach().cpu().item())
        physical_time += representative_dt
        if not bool(torch.isfinite(code.values).all()):
            codes.append(code)
            failure_cause = "nonfinite_latent_code"
            records.append(
                LatentStepRecord(
                    step=step + 1,
                    physical_time=physical_time,
                    dt=representative_dt,
                    encode_calls=1,
                    transition_calls=step + 1,
                    decode_calls=step,
                    decoder_constraint=decoder_constraint,
                    decode_reencode=False,
                    clipping=False,
                    density_floor=False,
                    pressure_floor=False,
                    limiter=False,
                    projection=False,
                    reset=False,
                    truth_replacement=False,
                    analysis=False,
                    per_cycle_fit=False,
                    finite=False,
                    admissible=False,
                    failure_cause=failure_cause,
                )
            )
            break
        decoded = representation.decode(
            code,
            context,
            decoder_constraint=decoder_constraint,
        )
        codes.append(code)
        decoded_states.append(decoded)

        finite = bool(
            torch.isfinite(decoded.conservative).all()
            and torch.isfinite(decoded.primitive).all()
        )
        density_positive = bool(torch.all(decoded.primitive[..., 0] > 0.0))
        pressure_positive = bool(torch.all(decoded.primitive[..., 2] > 0.0))
        admissible = finite and density_positive and pressure_positive
        if not finite:
            failure_cause = "nonfinite_raw_state"
        elif not density_positive:
            failure_cause = "nonpositive_raw_density"
        elif not pressure_positive:
            failure_cause = "nonpositive_raw_pressure"
        else:
            failure_cause = None

        records.append(
            LatentStepRecord(
                step=step + 1,
                physical_time=physical_time,
                dt=representative_dt,
                encode_calls=1,
                transition_calls=step + 1,
                decode_calls=step + 1,
                decoder_constraint=decoder_constraint,
                decode_reencode=False,
                clipping=False,
                density_floor=False,
                pressure_floor=False,
                limiter=False,
                projection=False,
                reset=False,
                truth_replacement=False,
                analysis=False,
                per_cycle_fit=False,
                finite=finite,
                admissible=admissible,
                failure_cause=failure_cause,
            )
        )
        if not admissible:
            break

    valid_length = sum(record.admissible for record in records)
    return LatentRollout(
        codes=tuple(codes),
        decoded=tuple(decoded_states),
        records=tuple(records),
        valid_length=valid_length,
        failure_cause=failure_cause,
    )


def decode_coarse_conservative(
    coarse_average: torch.Tensor,
    residual: torch.Tensor,
    cell_volume: torch.Tensor,
    *,
    constraint: DecoderConstraint,
) -> torch.Tensor:
    """Combine coarse averages and a fine residual with an optional exact mean.

    Tokens own equal contiguous numbers of fine cells.  The constrained path is
    a declared, parameter-free decoder layer; it is not a recurrent projection.
    """

    if coarse_average.ndim != 3 or coarse_average.shape[-1] != 3:
        raise ValueError("coarse_average must have shape [batch, tokens, 3]")
    if residual.ndim != 3 or residual.shape[-1] != 3:
        raise ValueError("residual must have shape [batch, cells, 3]")
    if cell_volume.shape != residual.shape[:2]:
        raise ValueError("cell_volume must have shape [batch, cells]")
    if coarse_average.shape[0] != residual.shape[0]:
        raise ValueError("coarse and residual batch sizes must match")
    tokens = coarse_average.shape[1]
    cells = residual.shape[1]
    if cells % tokens != 0:
        raise ValueError("fine cells must be divisible by coarse tokens")
    if constraint not in ("none", "coarse_conservative"):
        raise ValueError(f"unsupported decoder constraint: {constraint}")

    cells_per_token = cells // tokens
    residual_blocks = residual.reshape(residual.shape[0], tokens, cells_per_token, 3)
    if constraint == "coarse_conservative":
        volume_blocks = cell_volume.reshape(
            cell_volume.shape[0], tokens, cells_per_token, 1
        )
        residual_mean = (volume_blocks * residual_blocks).sum(dim=2) / (
            volume_blocks.sum(dim=2)
        )
        residual_blocks = residual_blocks - residual_mean.unsqueeze(2)
    decoded = coarse_average.unsqueeze(2) + residual_blocks
    return decoded.reshape(residual.shape)


def cell_edges_from_centers(x: np.ndarray) -> np.ndarray:
    """Return cell edges for a strictly increasing one-dimensional grid."""

    centers = np.asarray(x, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("x must be one-dimensional with at least two cells")
    differences = np.diff(centers)
    if not np.all(differences > 0.0):
        raise ValueError("x must be strictly increasing")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * differences[0]
    edges[-1] = centers[-1] + 0.5 * differences[-1]
    return edges


@dataclass(frozen=True)
class Euler1DFrontSet:
    """Two pressure-front slots and one pressure-balanced contact slot."""

    position_fraction: np.ndarray
    thickness_fraction: np.ndarray
    signed_strength: np.ndarray
    valid: np.ndarray
    score: np.ndarray

    def vector(self) -> np.ndarray:
        stacked = np.stack(
            (
                self.valid.astype(np.float64),
                self.position_fraction,
                self.thickness_fraction,
                self.signed_strength,
            ),
            axis=-1,
        )
        return stacked.reshape(*stacked.shape[:-2], 12)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "Euler1DFrontSet":
        value = np.asarray(vector, dtype=np.float64)
        if value.shape[-1] != 12:
            raise ValueError("front vector must have final dimension 12")
        if not np.isfinite(value).all():
            raise ValueError("front vector must be finite")
        slots = value.reshape(*value.shape[:-1], 3, 4)
        valid = slots[..., 0] >= 0.5
        return cls(
            position_fraction=np.where(valid, slots[..., 1], 0.0),
            thickness_fraction=np.where(valid, slots[..., 2], 0.0),
            signed_strength=np.where(valid, slots[..., 3], 0.0),
            valid=valid,
            score=np.zeros_like(slots[..., 0]),
        )


def _select_separated_peaks(
    score: np.ndarray,
    *,
    count: int,
    minimum_separation: int,
    threshold: float,
    allowed: np.ndarray | None = None,
) -> list[int]:
    order = np.argsort(score)[::-1]
    selected: list[int] = []
    for index in order.tolist():
        if score[index] < threshold:
            break
        if allowed is not None and not bool(allowed[index]):
            continue
        if any(abs(index - previous) < minimum_separation for previous in selected):
            continue
        selected.append(index)
        if len(selected) == count:
            break
    return selected


def _front_quantiles(
    score: np.ndarray,
    peak: int,
    face_x: np.ndarray,
) -> tuple[float, float]:
    cutoff = 0.1 * score[peak]
    left = peak
    right = peak
    while left > 0 and score[left - 1] >= cutoff:
        left -= 1
    while right + 1 < score.size and score[right + 1] >= cutoff:
        right += 1
    weights = score[left : right + 1]
    coordinates = face_x[left : right + 1]
    if float(weights.sum()) <= 0.0:
        return float(face_x[peak]), 0.0
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    quantiles = np.interp((0.1, 0.5, 0.9), cumulative, coordinates)
    return float(quantiles[1]), float(quantiles[2] - quantiles[0])


def _plateau_means(value: np.ndarray, face: int) -> tuple[float, float]:
    cells = value.size
    left_start = max(0, face - 5)
    left_stop = max(left_start + 1, face - 1)
    right_start = min(cells - 1, face + 2)
    right_stop = min(cells, face + 6)
    left = value[left_start:left_stop]
    right = value[right_start:right_stop]
    if left.size == 0:
        left = value[max(0, face) : max(0, face) + 1]
    if right.size == 0:
        right = value[min(cells - 1, face + 1) : min(cells - 1, face + 1) + 1]
    return float(left.mean()), float(right.mean())


def extract_euler1d_fronts(
    primitive: np.ndarray,
    x: np.ndarray,
    *,
    gamma: float = 1.4,
    score_threshold: float = 0.02,
) -> Euler1DFrontSet:
    """Extract privileged pressure fronts and a pressure-balanced contact.

    This hard, truth-derived extractor is an oracle control.  It is not a
    deployable learned encoder and it does not use trajectory history.
    """

    states = np.asarray(primitive, dtype=np.float64)
    squeeze = states.ndim == 2
    if squeeze:
        states = states[None, ...]
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError("primitive must have shape [samples, cells, 3]")
    coordinates = np.asarray(x, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = np.broadcast_to(coordinates, states.shape[:2])
    if coordinates.shape != states.shape[:2]:
        raise ValueError("x must have shape [cells] or [samples, cells]")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")

    output_shape = (states.shape[0], 3)
    position = np.zeros(output_shape, dtype=np.float64)
    thickness = np.zeros(output_shape, dtype=np.float64)
    strength = np.zeros(output_shape, dtype=np.float64)
    valid = np.zeros(output_shape, dtype=bool)
    selected_score = np.zeros(output_shape, dtype=np.float64)

    for sample in range(states.shape[0]):
        rho = states[sample, :, 0]
        pressure = states[sample, :, 2]
        x_sample = coordinates[sample]
        edges = cell_edges_from_centers(x_sample)
        domain_length = edges[-1] - edges[0]
        face_x = 0.5 * (x_sample[:-1] + x_sample[1:])
        rho_face = 0.5 * (rho[:-1] + rho[1:])
        pressure_face = 0.5 * (pressure[:-1] + pressure[1:])
        eps_rho = max(1.0e-12, 1.0e-10 * float(np.max(np.abs(rho))))
        eps_pressure = max(1.0e-12, 1.0e-10 * float(np.max(np.abs(pressure))))
        pressure_score = np.abs(np.diff(pressure)) / (
            np.abs(pressure_face) + eps_pressure
        )
        sound_speed_sq = gamma * pressure_face / np.maximum(rho_face, eps_rho)
        contact_characteristic = np.diff(rho) - np.diff(pressure) / np.maximum(
            sound_speed_sq, eps_pressure
        )
        contact_score = np.abs(contact_characteristic) / (np.abs(rho_face) + eps_rho)
        minimum_separation = max(4, int(np.ceil(0.03125 * states.shape[1])))
        pressure_peaks = _select_separated_peaks(
            pressure_score,
            count=2,
            minimum_separation=minimum_separation,
            threshold=score_threshold,
        )
        pressure_peaks.sort()

        contact_allowed = pressure_score <= 0.25 * contact_score
        for peak in pressure_peaks:
            lower = max(0, peak - minimum_separation + 1)
            upper = min(contact_allowed.size, peak + minimum_separation)
            contact_allowed[lower:upper] = False
        contact_peaks = _select_separated_peaks(
            contact_score,
            count=1,
            minimum_separation=minimum_separation,
            threshold=score_threshold,
            allowed=contact_allowed,
        )

        for slot, peak in enumerate(pressure_peaks):
            center, width = _front_quantiles(pressure_score, peak, face_x)
            left_pressure, right_pressure = _plateau_means(pressure, peak)
            scale = 0.5 * (abs(left_pressure) + abs(right_pressure)) + eps_pressure
            position[sample, slot] = (center - edges[0]) / domain_length
            thickness[sample, slot] = (
                max(width, float(np.min(np.diff(edges)))) / domain_length
            )
            strength[sample, slot] = (right_pressure - left_pressure) / scale
            valid[sample, slot] = True
            selected_score[sample, slot] = pressure_score[peak]

        if contact_peaks:
            peak = contact_peaks[0]
            center, width = _front_quantiles(contact_score, peak, face_x)
            left_rho, right_rho = _plateau_means(rho, peak)
            left_pressure, right_pressure = _plateau_means(pressure, peak)
            local_sound_sq = max(float(sound_speed_sq[peak]), eps_pressure)
            jump = (right_rho - left_rho) - (
                right_pressure - left_pressure
            ) / local_sound_sq
            scale = 0.5 * (abs(left_rho) + abs(right_rho)) + eps_rho
            position[sample, 2] = (center - edges[0]) / domain_length
            thickness[sample, 2] = (
                max(width, float(np.min(np.diff(edges)))) / domain_length
            )
            strength[sample, 2] = jump / scale
            valid[sample, 2] = True
            selected_score[sample, 2] = contact_score[peak]

    result = Euler1DFrontSet(
        position_fraction=position,
        thickness_fraction=thickness,
        signed_strength=strength,
        valid=valid,
        score=selected_score,
    )
    if not squeeze:
        return result
    return Euler1DFrontSet(
        position_fraction=result.position_fraction[0],
        thickness_fraction=result.thickness_fraction[0],
        signed_strength=result.signed_strength[0],
        valid=result.valid[0],
        score=result.score[0],
    )


def _piecewise_constant_integral(
    values: np.ndarray,
    source_edges: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    edges = np.asarray(source_edges, dtype=np.float64)
    points = np.asarray(query, dtype=np.float64)
    if state.ndim != 2 or edges.shape != (state.shape[0] + 1,):
        raise ValueError("values/source_edges shapes are inconsistent")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("source edges must be strictly increasing")
    tolerance = 1.0e-12 * max(1.0, abs(edges[-1] - edges[0]))
    if np.any(points < edges[0] - tolerance) or np.any(points > edges[-1] + tolerance):
        raise ValueError("query points lie outside the source interval")
    points = np.clip(points, edges[0], edges[-1])
    widths = np.diff(edges)
    cumulative = np.concatenate(
        (
            np.zeros((1, state.shape[1]), dtype=np.float64),
            np.cumsum(widths[:, None] * state, axis=0),
        ),
        axis=0,
    )
    indices = np.searchsorted(edges, points, side="right") - 1
    indices = np.clip(indices, 0, state.shape[0] - 1)
    result = cumulative[indices] + (points - edges[indices])[:, None] * state[indices]
    result[points == edges[-1]] = cumulative[-1]
    return result


def register_conservative_state(
    conservative: np.ndarray,
    x: np.ndarray,
    fronts: Euler1DFrontSet,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Conservatively map a physical state into one oracle phase chart."""

    state = np.asarray(conservative, dtype=np.float64)
    if state.ndim != 2 or state.shape[-1] != 3:
        raise ValueError("conservative must have shape [cells, 3]")
    edges = cell_edges_from_centers(x)
    normalized_edges = (edges - edges[0]) / (edges[-1] - edges[0])
    canonical_edges = np.linspace(0.0, 1.0, state.shape[0] + 1)
    positions = np.sort(
        np.asarray(fronts.position_fraction)[np.asarray(fronts.valid, dtype=bool)]
    )
    if positions.size:
        if np.any(np.diff(positions) <= 1.0e-8):
            raise ValueError("oracle front anchors must be strictly separated")
        reference = np.arange(1, positions.size + 1, dtype=np.float64) / (
            positions.size + 1
        )
        reference_knots = np.concatenate(([0.0], reference, [1.0]))
        physical_knots = np.concatenate(([0.0], positions, [1.0]))
    else:
        reference_knots = np.array([0.0, 1.0])
        physical_knots = np.array([0.0, 1.0])
    slopes = np.diff(physical_knots) / np.diff(reference_knots)
    mapped_edges = np.interp(canonical_edges, reference_knots, physical_knots)
    integrals = _piecewise_constant_integral(state, normalized_edges, mapped_edges)
    registered = np.diff(integrals, axis=0) / np.diff(canonical_edges)[:, None]
    return registered, {
        "valid_fronts": int(positions.size),
        "minimum_chart_slope": float(np.min(slopes)),
        "maximum_chart_slope": float(np.max(slopes)),
    }


def decode_registered_state(
    registered: np.ndarray,
    x: np.ndarray,
    fronts: Euler1DFrontSet,
) -> np.ndarray:
    """Invert ``register_conservative_state`` by conservative remapping."""

    state = np.asarray(registered, dtype=np.float64)
    if state.ndim != 2 or state.shape[-1] != 3:
        raise ValueError("registered must have shape [cells, 3]")
    edges = cell_edges_from_centers(x)
    physical_edges = (edges - edges[0]) / (edges[-1] - edges[0])
    canonical_edges = np.linspace(0.0, 1.0, state.shape[0] + 1)
    positions = np.sort(
        np.asarray(fronts.position_fraction)[np.asarray(fronts.valid, dtype=bool)]
    )
    if positions.size:
        reference = np.arange(1, positions.size + 1, dtype=np.float64) / (
            positions.size + 1
        )
        physical_knots = np.concatenate(([0.0], positions, [1.0]))
        reference_knots = np.concatenate(([0.0], reference, [1.0]))
    else:
        physical_knots = np.array([0.0, 1.0])
        reference_knots = np.array([0.0, 1.0])
    inverse_edges = np.interp(physical_edges, physical_knots, reference_knots)
    integrals = _piecewise_constant_integral(state, canonical_edges, inverse_edges)
    return np.diff(integrals, axis=0) / np.diff(physical_edges)[:, None]


@dataclass(frozen=True)
class VolumeWeightedPOD:
    """Training-only POD in the finite-volume, fixed-scale state metric."""

    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    cell_weight_sqrt: np.ndarray
    component_scale: np.ndarray
    total_variance: float

    @property
    def rank(self) -> int:
        return int(self.modes.shape[0])

    @property
    def retained_energy(self) -> float:
        if self.total_variance <= 0.0:
            return 1.0
        return float(self.eigenvalues[: self.rank].sum() / self.total_variance)

    @classmethod
    def fit(
        cls,
        conservative: np.ndarray,
        cell_volume: np.ndarray,
        component_scale: np.ndarray,
        *,
        rank: int | None = None,
        energy_fraction: float = 0.995,
    ) -> "VolumeWeightedPOD":
        states = np.asarray(conservative, dtype=np.float64)
        volume = np.asarray(cell_volume, dtype=np.float64)
        scale = np.asarray(component_scale, dtype=np.float64)
        if states.ndim != 3 or states.shape[-1] != 3:
            raise ValueError("conservative must have shape [samples, cells, 3]")
        if volume.shape != (states.shape[1],) or np.any(volume <= 0.0):
            raise ValueError("cell_volume must be positive with shape [cells]")
        if scale.shape != (3,) or np.any(scale <= 0.0):
            raise ValueError("component_scale must be positive with shape [3]")
        if states.shape[0] < 2:
            raise ValueError("at least two snapshots are required for POD")
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError("energy_fraction must lie in (0, 1]")

        mean = states.mean(axis=0)
        weight_sqrt = np.sqrt(volume / volume.sum())[:, None]
        weighted = ((states - mean[None]) / scale[None, None]) * weight_sqrt[None]
        matrix = weighted.reshape(states.shape[0], -1)
        covariance = matrix.T @ matrix / max(states.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = float(eigenvalues.sum())
        if rank is None:
            if total <= 0.0:
                selected_rank = 1
            else:
                selected_rank = int(
                    np.searchsorted(
                        np.cumsum(eigenvalues) / total,
                        energy_fraction,
                        side="left",
                    )
                    + 1
                )
        else:
            selected_rank = int(rank)
        if not 1 <= selected_rank <= matrix.shape[1]:
            raise ValueError("rank must lie in [1, cells * 3]")
        modes = eigenvectors[:, :selected_rank].T
        return cls(
            mean=mean,
            modes=modes,
            eigenvalues=eigenvalues,
            cell_weight_sqrt=weight_sqrt,
            component_scale=scale,
            total_variance=total,
        )

    def encode(self, conservative: np.ndarray) -> np.ndarray:
        states = np.asarray(conservative, dtype=np.float64)
        squeeze = states.ndim == 2
        if squeeze:
            states = states[None]
        if states.shape[1:] != self.mean.shape:
            raise ValueError("state shape does not match fitted POD")
        weighted = (
            (states - self.mean[None]) / self.component_scale[None, None]
        ) * self.cell_weight_sqrt[None]
        code = weighted.reshape(states.shape[0], -1) @ self.modes.T
        return code[0] if squeeze else code

    def decode(self, code: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(code, dtype=np.float64)
        squeeze = coefficients.ndim == 1
        if squeeze:
            coefficients = coefficients[None]
        if coefficients.ndim != 2 or coefficients.shape[1] != self.rank:
            raise ValueError("code shape does not match fitted POD rank")
        weighted = (coefficients @ self.modes).reshape(
            coefficients.shape[0], *self.mean.shape
        )
        states = (weighted / self.cell_weight_sqrt[None]) * self.component_scale[
            None, None
        ] + self.mean[None]
        return states[0] if squeeze else states


@dataclass(frozen=True)
class OracleFrontPOD:
    """Matched code containing a 12-scalar oracle front chart plus POD modes."""

    registered_pod: VolumeWeightedPOD
    total_size: int
    gamma: float

    @property
    def residual_rank(self) -> int:
        return self.registered_pod.rank

    @classmethod
    def fit(
        cls,
        primitive: np.ndarray,
        x: np.ndarray,
        cell_volume: np.ndarray,
        component_scale: np.ndarray,
        *,
        total_size: int,
        gamma: float = 1.4,
    ) -> "OracleFrontPOD":
        states = np.asarray(primitive, dtype=np.float64)
        coordinates = np.asarray(x, dtype=np.float64)
        if states.ndim != 3 or states.shape[-1] != 3:
            raise ValueError("primitive must have shape [samples, cells, 3]")
        if coordinates.ndim == 1:
            coordinates = np.broadcast_to(coordinates, states.shape[:2])
        if coordinates.shape != states.shape[:2]:
            raise ValueError("x must align with primitive snapshots")
        if total_size < 16:
            raise ValueError("oracle code needs at least 16 total scalars")
        fronts = extract_euler1d_fronts(states, coordinates, gamma=gamma)
        conservative = primitive_to_conservative_np(states, gamma)
        registered = np.empty_like(conservative)
        for index in range(states.shape[0]):
            sample_fronts = _front_sample(fronts, index)
            registered[index], _ = register_conservative_state(
                conservative[index], coordinates[index], sample_fronts
            )
        canonical_volume = np.full(
            states.shape[1],
            float(np.sum(cell_volume)) / states.shape[1],
            dtype=np.float64,
        )
        pod = VolumeWeightedPOD.fit(
            registered,
            canonical_volume,
            component_scale,
            rank=total_size - 12,
        )
        return cls(registered_pod=pod, total_size=total_size, gamma=gamma)

    def encode(self, primitive: np.ndarray, x: np.ndarray) -> np.ndarray:
        states = np.asarray(primitive, dtype=np.float64)
        squeeze = states.ndim == 2
        if squeeze:
            states = states[None]
        coordinates = np.asarray(x, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = np.broadcast_to(coordinates, states.shape[:2])
        fronts = extract_euler1d_fronts(states, coordinates, gamma=self.gamma)
        conservative = primitive_to_conservative_np(states, self.gamma)
        registered = np.empty_like(conservative)
        for index in range(states.shape[0]):
            registered[index], _ = register_conservative_state(
                conservative[index], coordinates[index], _front_sample(fronts, index)
            )
        coefficients = self.registered_pod.encode(registered)
        code = np.concatenate((fronts.vector(), coefficients), axis=-1)
        return code[0] if squeeze else code

    def decode_conservative(self, code: np.ndarray, x: np.ndarray) -> np.ndarray:
        values = np.asarray(code, dtype=np.float64)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.total_size:
            raise ValueError("oracle code has the wrong scalar size")
        coordinates = np.asarray(x, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = np.broadcast_to(
                coordinates, (values.shape[0], coordinates.size)
            )
        fronts = Euler1DFrontSet.from_vector(values[:, :12])
        registered = self.registered_pod.decode(values[:, 12:])
        decoded = np.empty_like(registered)
        for index in range(values.shape[0]):
            decoded[index] = decode_registered_state(
                registered[index], coordinates[index], _front_sample(fronts, index)
            )
        return decoded[0] if squeeze else decoded

    def decode(self, code: np.ndarray, x: np.ndarray) -> np.ndarray:
        conservative = self.decode_conservative(code, x)
        return conservative_to_primitive_np(conservative, self.gamma)


def _front_sample(fronts: Euler1DFrontSet, index: int) -> Euler1DFrontSet:
    return Euler1DFrontSet(
        position_fraction=np.asarray(fronts.position_fraction[index]),
        thickness_fraction=np.asarray(fronts.thickness_fraction[index]),
        signed_strength=np.asarray(fronts.signed_strength[index]),
        valid=np.asarray(fronts.valid[index]),
        score=np.asarray(fronts.score[index]),
    )


@dataclass(frozen=True)
class CodeWhitener:
    mean: np.ndarray
    transform: np.ndarray
    regularization: float

    @classmethod
    def fit(cls, code: np.ndarray) -> "CodeWhitener":
        values = np.asarray(code, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("code must have shape [samples, dimensions]")
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        trace = float(np.trace(covariance))
        regularization = 1.0e-6 * trace / max(values.shape[1], 1)
        regularization = max(regularization, 1.0e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        transform = eigenvectors @ np.diag(
            1.0 / np.sqrt(np.maximum(eigenvalues, 0.0) + regularization)
        )
        return cls(mean=mean, transform=transform, regularization=regularization)

    def apply(self, code: np.ndarray) -> np.ndarray:
        values = np.asarray(code, dtype=np.float64)
        if values.shape[-1] != self.mean.size:
            raise ValueError("code dimension does not match whitener")
        return (values - self.mean) @ self.transform


@dataclass(frozen=True)
class ClosureDiagnostics:
    neighbor_radius: np.ndarray
    current_mismatch: np.ndarray
    future_ambiguity: np.ndarray
    encoded_future_ambiguity: np.ndarray | None
    physical_knn_forecast_error: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_weights: np.ndarray
    step_scale: float
    latent_step_scale: float | None

    def summary(self) -> dict[str, float | int | None]:
        summary: dict[str, float | int | None] = {
            "queries": int(self.future_ambiguity.size),
            "step_scale": self.step_scale,
            "neighbor_radius_median": float(np.median(self.neighbor_radius)),
            "neighbor_radius_p90": float(np.quantile(self.neighbor_radius, 0.9)),
            "current_mismatch_median": float(np.median(self.current_mismatch)),
            "current_mismatch_p90": float(np.quantile(self.current_mismatch, 0.9)),
            "future_ambiguity_median": float(np.median(self.future_ambiguity)),
            "future_ambiguity_p90": float(np.quantile(self.future_ambiguity, 0.9)),
            "physical_knn_forecast_error_median": float(
                np.median(self.physical_knn_forecast_error)
            ),
            "physical_knn_forecast_error_p90": float(
                np.quantile(self.physical_knn_forecast_error, 0.9)
            ),
        }
        if self.encoded_future_ambiguity is None or self.latent_step_scale is None:
            summary.update(
                {
                    "latent_step_scale": None,
                    "neighbor_radius_over_latent_step_median": None,
                    "neighbor_radius_over_latent_step_p90": None,
                    "encoded_future_ambiguity_median": None,
                    "encoded_future_ambiguity_p90": None,
                }
            )
        else:
            normalized_radius = self.neighbor_radius / self.latent_step_scale
            summary.update(
                {
                    "latent_step_scale": self.latent_step_scale,
                    "neighbor_radius_over_latent_step_median": float(
                        np.median(normalized_radius)
                    ),
                    "neighbor_radius_over_latent_step_p90": float(
                        np.quantile(normalized_radius, 0.9)
                    ),
                    "encoded_future_ambiguity_median": float(
                        np.median(self.encoded_future_ambiguity)
                    ),
                    "encoded_future_ambiguity_p90": float(
                        np.quantile(self.encoded_future_ambiguity, 0.9)
                    ),
                }
            )
        return summary


def _state_distance(
    first: np.ndarray,
    second: np.ndarray,
    cell_volume: np.ndarray,
    component_scale: np.ndarray,
) -> np.ndarray:
    difference = (np.asarray(first) - np.asarray(second)) / component_scale
    squared = np.sum(difference**2, axis=-1)
    return np.sqrt(np.sum(squared * cell_volume, axis=-1) / np.sum(cell_volume))


def conditional_future_diagnostics(
    train_code: np.ndarray,
    train_current: np.ndarray,
    train_future: np.ndarray,
    query_code: np.ndarray,
    query_current: np.ndarray,
    query_future: np.ndarray,
    cell_volume: np.ndarray,
    component_scale: np.ndarray,
    *,
    neighbors: int = 8,
    train_future_code: np.ndarray | None = None,
    query_future_code: np.ndarray | None = None,
    train_group: np.ndarray | None = None,
    query_group: np.ndarray | None = None,
) -> ClosureDiagnostics:
    """Estimate cross-trajectory conditional future ambiguity with kNN."""

    train_values = np.asarray(train_code, dtype=np.float64)
    query_values = np.asarray(query_code, dtype=np.float64)
    train_current = np.asarray(train_current, dtype=np.float64)
    train_future = np.asarray(train_future, dtype=np.float64)
    query_current = np.asarray(query_current, dtype=np.float64)
    query_future = np.asarray(query_future, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    scale = np.asarray(component_scale, dtype=np.float64)
    if train_values.ndim != 2 or query_values.ndim != 2:
        raise ValueError("codes must be two-dimensional")
    if train_values.shape[1] != query_values.shape[1]:
        raise ValueError("train and query codes must share dimension")
    if train_current.shape != train_future.shape:
        raise ValueError("training current/future states must share shape")
    if query_current.shape != query_future.shape:
        raise ValueError("query current/future states must share shape")
    if train_current.shape[0] != train_values.shape[0]:
        raise ValueError("training codes and states must align")
    if query_current.shape[0] != query_values.shape[0]:
        raise ValueError("query codes and states must align")
    if not 1 <= neighbors < train_values.shape[0]:
        raise ValueError("neighbors must lie in [1, train samples - 1]")
    if volume.shape != (train_current.shape[1],) or scale.shape != (3,):
        raise ValueError("cell_volume/component_scale shapes are invalid")

    if not np.isfinite(train_values).all() or not np.isfinite(query_values).all():
        raise ValueError("current codes must be finite")
    if (train_future_code is None) != (query_future_code is None):
        raise ValueError("train_future_code and query_future_code must be paired")

    whitener = CodeWhitener.fit(train_values)
    train_white = whitener.apply(train_values)
    query_white = whitener.apply(query_values)
    if train_future_code is None:
        train_future_white = None
        query_future_white = None
        latent_step_scale = None
        encoded_future_ambiguity = None
    else:
        train_future_values = np.asarray(train_future_code, dtype=np.float64)
        query_future_values = np.asarray(query_future_code, dtype=np.float64)
        if train_future_values.shape != train_values.shape:
            raise ValueError("training current/future codes must share shape")
        if query_future_values.shape != query_values.shape:
            raise ValueError("query current/future codes must share shape")
        if (
            not np.isfinite(train_future_values).all()
            or not np.isfinite(query_future_values).all()
        ):
            raise ValueError("future codes must be finite")
        train_future_white = whitener.apply(train_future_values)
        query_future_white = whitener.apply(query_future_values)
        latent_step_scale = max(
            float(np.median(np.linalg.norm(train_future_white - train_white, axis=1))),
            1.0e-12,
        )
        encoded_future_ambiguity = np.empty(query_values.shape[0], dtype=np.float64)
    step_scale = float(
        np.median(_state_distance(train_future, train_current, volume, scale))
    )
    step_scale = max(step_scale, 1.0e-12)
    indices = np.empty((query_values.shape[0], neighbors), dtype=np.int64)
    weights = np.empty((query_values.shape[0], neighbors), dtype=np.float64)
    radius = np.empty(query_values.shape[0], dtype=np.float64)
    current_mismatch = np.empty_like(radius)
    future_ambiguity = np.empty_like(radius)
    forecast_error = np.empty_like(radius)
    train_group_array = None if train_group is None else np.asarray(train_group)
    query_group_array = None if query_group is None else np.asarray(query_group)
    if (train_group_array is None) != (query_group_array is None):
        raise ValueError("train_group and query_group must be supplied together")

    train_norm = np.sum(train_white**2, axis=1)
    for query_index, query in enumerate(query_white):
        distances_sq = np.maximum(
            train_norm + float(np.dot(query, query)) - 2.0 * (train_white @ query),
            0.0,
        )
        if train_group_array is not None:
            distances_sq[train_group_array == query_group_array[query_index]] = np.inf
        finite_candidates = int(np.count_nonzero(np.isfinite(distances_sq)))
        if finite_candidates < neighbors:
            raise ValueError("too few cross-group neighbors")
        selected = np.argpartition(distances_sq, neighbors - 1)[:neighbors]
        selected = selected[np.argsort(distances_sq[selected])]
        selected_distance = np.sqrt(distances_sq[selected])
        bandwidth = max(float(selected_distance[-1]), 1.0e-12)
        selected_weight = np.exp(-0.5 * (selected_distance / bandwidth) ** 2)
        selected_weight /= selected_weight.sum()
        indices[query_index] = selected
        weights[query_index] = selected_weight
        radius[query_index] = selected_distance[-1]

        current_distances = _state_distance(
            train_current[selected], query_current[query_index], volume, scale
        )
        future_distances = _state_distance(
            train_future[selected], query_future[query_index], volume, scale
        )
        current_mismatch[query_index] = float(
            np.sqrt(np.sum(selected_weight * current_distances**2)) / step_scale
        )
        future_ambiguity[query_index] = float(
            np.sqrt(np.sum(selected_weight * future_distances**2)) / step_scale
        )
        if encoded_future_ambiguity is not None:
            assert train_future_white is not None
            assert query_future_white is not None
            assert latent_step_scale is not None
            latent_future_distances = np.linalg.norm(
                train_future_white[selected] - query_future_white[query_index],
                axis=1,
            )
            encoded_future_ambiguity[query_index] = float(
                np.sqrt(np.sum(selected_weight * latent_future_distances**2))
                / latent_step_scale
            )
        forecast = np.sum(
            selected_weight[:, None, None] * train_future[selected], axis=0
        )
        forecast_error[query_index] = float(
            _state_distance(forecast, query_future[query_index], volume, scale)
            / step_scale
        )

    return ClosureDiagnostics(
        neighbor_radius=radius,
        current_mismatch=current_mismatch,
        future_ambiguity=future_ambiguity,
        encoded_future_ambiguity=encoded_future_ambiguity,
        physical_knn_forecast_error=forecast_error,
        neighbor_indices=indices,
        neighbor_weights=weights,
        step_scale=step_scale,
        latent_step_scale=latent_step_scale,
    )


@dataclass(frozen=True)
class LinearCodePCA:
    mean: np.ndarray
    modes: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, output_size: int) -> "LinearCodePCA":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("values must have shape [samples, dimensions]")
        if not 1 <= output_size <= array.shape[1]:
            raise ValueError("output_size is outside the input dimension")
        mean = array.mean(axis=0)
        centered = array - mean
        covariance = centered.T @ centered / max(array.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:output_size]
        return cls(mean=mean, modes=eigenvectors[:, order].T)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean) @ self.modes.T


def grouped_bootstrap_median_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int = 1000,
    seed: int = 20260720,
) -> dict[str, float | int]:
    """Paired trajectory-cluster bootstrap for a ratio of medians."""

    first = np.asarray(numerator, dtype=np.float64)
    second = np.asarray(denominator, dtype=np.float64)
    group_values = np.asarray(groups)
    if first.shape != second.shape or first.shape != group_values.shape:
        raise ValueError("numerator, denominator, and groups must share shape")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    unique = np.unique(group_values)
    if unique.size < 2:
        raise ValueError("at least two groups are required")
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate(
            [np.flatnonzero(group_values == group) for group in sampled]
        )
        ratios[repetition] = np.median(first[indices]) / max(
            np.median(second[indices]), 1.0e-12
        )
    estimate = float(np.median(first) / max(np.median(second), 1.0e-12))
    lower, upper = np.quantile(ratios, (0.025, 0.975))
    return {
        "groups": int(unique.size),
        "repetitions": repetitions,
        "estimate": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def front_reconstruction_metrics(
    truth: Euler1DFrontSet,
    prediction: Euler1DFrontSet,
    *,
    num_cells: int,
) -> dict[str, float | int]:
    """Report slot-matched front position, strength, and thickness errors."""

    truth_valid = np.asarray(truth.valid, dtype=bool)
    prediction_valid = np.asarray(prediction.valid, dtype=bool)
    common = truth_valid & prediction_valid
    if not np.any(common):
        return {
            "truth_fronts": int(np.count_nonzero(truth_valid)),
            "predicted_fronts": int(np.count_nonzero(prediction_valid)),
            "common_fronts": 0,
            "position_error_cells_median": float("nan"),
            "position_error_cells_p95": float("nan"),
            "strength_relative_error_median": float("nan"),
            "thickness_ratio_median": float("nan"),
            "thickness_ratio_p95": float("nan"),
        }
    position_error = (
        np.abs(truth.position_fraction[common] - prediction.position_fraction[common])
        * num_cells
    )
    strength_error = np.abs(
        prediction.signed_strength[common] - truth.signed_strength[common]
    ) / np.maximum(np.abs(truth.signed_strength[common]), 1.0e-8)
    thickness_ratio = prediction.thickness_fraction[common] / np.maximum(
        truth.thickness_fraction[common], 1.0 / num_cells
    )
    return {
        "truth_fronts": int(np.count_nonzero(truth_valid)),
        "predicted_fronts": int(np.count_nonzero(prediction_valid)),
        "common_fronts": int(np.count_nonzero(common)),
        "position_error_cells_median": float(np.median(position_error)),
        "position_error_cells_p95": float(np.quantile(position_error, 0.95)),
        "strength_relative_error_median": float(np.median(strength_error)),
        "thickness_ratio_median": float(np.median(thickness_ratio)),
        "thickness_ratio_p95": float(np.quantile(thickness_ratio, 0.95)),
    }


def fixed_scale_relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
    cell_volume: np.ndarray,
    component_scale: np.ndarray,
) -> np.ndarray:
    """Volume-weighted conservative relative L2 for each leading sample."""

    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("prediction and truth must share shape [..., cells, 3]")
    numerator = _state_distance(predicted, target, cell_volume, component_scale)
    zeros = np.zeros_like(target)
    denominator = _state_distance(target, zeros, cell_volume, component_scale)
    return numerator / np.maximum(denominator, 1.0e-12)
