"""Two-step response-filtered SP19 blocks for W26-L5.

The first state receives the frozen synchronized SP19 correction.  Two native
lookaheads then expose its finite propagated response, of which only the fixed
SP19 physical-modal component is retained in the second state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FixedCosineProjector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    FineDiscrepancyAudit,
)
from utility.time_dependent_no.pcno_propagated_sensitivity import (
    propagated_lookahead,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    pressure_profile_shock_metrics,
    restrict_nested_state,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    ModalCell,
    masked_modal_field,
    sparse_modal_correction,
    synchronized_sparse_modal_step,
)


@dataclass(frozen=True)
class ResponseProjectionAudit:
    status: str
    maximum_first_correction_integral_abs: float
    maximum_filtered_response_integral_abs: float
    maximum_first_boundary_abs: float
    maximum_filtered_boundary_abs: float
    maximum_projection_idempotence_abs: float
    full_response_rms: float
    filtered_response_rms: float
    retained_response_fraction: float | None


@dataclass(frozen=True)
class ResponseFilteredBlock:
    raw_first_state: np.ndarray
    corrected_first_state: np.ndarray
    raw_second_state: np.ndarray
    fully_corrected_second_state: np.ndarray
    full_response: np.ndarray
    filtered_response: np.ndarray
    filtered_second_state: np.ndarray
    first_correction: np.ndarray
    first_audit: FineDiscrepancyAudit
    projection_audit: ResponseProjectionAudit


ResponseMode = Literal["full", "filtered"]


@dataclass(frozen=True)
class ResponseFilteredTrajectory:
    states: tuple[np.ndarray, ...]
    blocks: tuple[ResponseFilteredBlock, ...]
    response_mode: ResponseMode


@dataclass(frozen=True)
class IntegralAnchorAudit:
    status: str
    maximum_integral_mismatch_before_abs: float
    maximum_integral_mismatch_after_abs: float
    maximum_boundary_correction_abs: float
    maximum_idempotence_abs: float
    correction_rms: float


@dataclass(frozen=True)
class ProjectedShadowTetherAudit:
    status: str
    full_difference_rms: float
    retained_difference_rms: float
    retained_difference_fraction: float | None
    maximum_integral_difference_abs: float
    maximum_boundary_difference_abs: float
    maximum_projection_idempotence_abs: float


@dataclass(frozen=True)
class SlewLimitedProjectedTetherAudit:
    status: str
    target_difference_rms: float
    previous_difference_rms: float
    requested_change_rms: float
    applied_change_rms: float
    shadow_increment_rms: float
    change_limit_rms: float
    applied_to_shadow_increment_ratio: float | None
    applied_scale: float
    cap_active: bool
    previous_projection_residual_rms: float
    retained_projection_residual_rms: float
    maximum_integral_difference_abs: float
    maximum_previous_boundary_abs: float
    maximum_boundary_difference_abs: float
    maximum_boundary_contraction_violation_abs: float
    maximum_projection_idempotence_abs: float
    maximum_update_identity_abs: float


@dataclass(frozen=True)
class ShadowAnchoredBlock:
    shadow_first_state: np.ndarray
    shadow_second_state: np.ndarray
    candidate_raw_first_state: np.ndarray
    candidate_first_before_anchor: np.ndarray
    anchored_first_state: np.ndarray
    candidate_raw_second_state: np.ndarray
    full_response: np.ndarray
    filtered_response: np.ndarray
    candidate_second_before_anchor: np.ndarray
    anchored_second_state: np.ndarray
    first_correction: np.ndarray
    first_anchor_correction: np.ndarray
    second_anchor_correction: np.ndarray
    first_audit: FineDiscrepancyAudit
    response_audit: ResponseProjectionAudit
    first_anchor_audit: IntegralAnchorAudit
    second_anchor_audit: IntegralAnchorAudit


FROZEN_POSITION_TRUST_THRESHOLD = 0.7375
FROZEN_POSITION_BUFFER_THRESHOLD = 0.8125


@dataclass(frozen=True)
class TransverseVelocityPositionDescriptor:
    status: str
    vertical_centroid: float | None
    normalized_wall_distance: float | None
    transverse_velocity_l1: float


@dataclass(frozen=True)
class TargetFreeFrontBranchAudit:
    status: str
    branch_changed: bool
    first_shadow_thickness_cells: int
    first_candidate_thickness_cells: int
    second_shadow_thickness_cells: int
    second_candidate_thickness_cells: int
    maximum_position_shift: float | None


@dataclass(frozen=True)
class PersistenceGainAudit:
    status: str
    baseline_alignment: float
    current_alignment: float
    alignment_ratio: float | None
    alignment_cosine: float | None
    alignment_denominator: float
    probe_norm: float
    offset_norm: float
    clipped_gain: float
    applied_gain: float


@dataclass(frozen=True)
class PersistenceProbe:
    correction: np.ndarray
    native_model_input: np.ndarray
    fine_model_input: np.ndarray
    fine_prediction: np.ndarray
    audit: FineDiscrepancyAudit


@dataclass(frozen=True)
class TerminalRampAudit:
    status: str
    trigger_block: int | None
    terminal_block: int
    applied_gain: float


def synchronized_persistence_probe(
    native_state: np.ndarray,
    native_prediction: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    fine_predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> PersistenceProbe:
    """Probe the sparse fine discrepancy while reusing one native prediction."""

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    native_next = np.asarray(native_prediction, dtype=np.float64)
    expected_native_shape = (contract.native[0] * contract.native[1], 4)
    if native_next.shape != expected_native_shape or not np.isfinite(
        native_next
    ).all():
        raise ValueError("native prediction must be finite and native-grid aligned")
    fine_input = np.asarray(
        prepared.model_inputs[contract.fine], dtype=np.float64
    ).copy()
    fine_next = np.asarray(
        fine_predictor(contract.fine, np.array(fine_input, copy=True)),
        dtype=np.float64,
    )
    expected_fine_shape = (contract.fine[0] * contract.fine[1], 4)
    if fine_next.shape != expected_fine_shape or not np.isfinite(fine_next).all():
        raise ValueError("fine prediction must be finite and fine-grid aligned")

    native_current = np.asarray(
        prepared.model_inputs[contract.native], dtype=np.float64
    )
    fine_current = np.asarray(prepared.model_inputs[contract.fine], dtype=np.float64)
    native_increment = native_next - native_current
    fine_on_native = restrict_nested_state(
        fine_next - fine_current,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=np.asarray(fine_on_native, dtype=np.float64),
        native_minus_coarse=np.zeros_like(native_increment),
        fine_minus_native=np.asarray(fine_on_native - native_increment),
    )
    correction, audit = sparse_modal_correction(
        basis,
        projector,
        policy="sp19_fine_away_half",
        volumes=volumes,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    return PersistenceProbe(
        correction=np.asarray(correction, dtype=np.float64).copy(),
        native_model_input=np.array(native_current, copy=True),
        fine_model_input=np.array(fine_input, copy=True),
        fine_prediction=np.array(fine_next, copy=True),
        audit=audit,
    )


def monotone_persistence_gain(
    frozen_offset: np.ndarray,
    probe_correction: np.ndarray,
    *,
    baseline_alignment: float | None,
    previous_gain: float,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    denominator_floor: float = 1.0e-12,
) -> PersistenceGainAudit:
    """Update a target-free output-offset gain from one synchronized probe.

    The raw signed alignment is normalized by its first coast value, clipped to
    ``[0, 1]``, and made nonincreasing.  An unresolved denominator fails closed
    to zero.  No state target or reference value is accepted by this helper.
    """

    offset = np.asarray(frozen_offset, dtype=np.float64)
    probe = np.asarray(probe_correction, dtype=np.float64)
    if offset.ndim != 2 or probe.shape != offset.shape:
        raise ValueError("persistence fields must be shape-aligned matrices")
    mass = _volumes(volumes, offset.shape[0])
    scale = _scale(component_scale, offset.shape[1])
    numeric = np.asarray(
        [
            previous_gain,
            denominator_floor,
            0.0 if baseline_alignment is None else baseline_alignment,
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(offset).all()
        or not np.isfinite(probe).all()
        or not np.isfinite(numeric).all()
        or not 0.0 <= previous_gain <= 1.0
        or denominator_floor <= 0.0
    ):
        raise ValueError("persistence gain inputs must be finite and bounded")

    scaled_offset = offset / scale[None, :]
    scaled_probe = probe / scale[None, :]
    current_alignment = float(
        np.sum(mass[:, None] * scaled_offset * scaled_probe)
    )
    offset_norm = float(
        np.sqrt(np.sum(mass[:, None] * scaled_offset * scaled_offset))
    )
    probe_norm = float(
        np.sqrt(np.sum(mass[:, None] * scaled_probe * scaled_probe))
    )
    cosine_denominator = offset_norm * probe_norm
    cosine = (
        current_alignment / cosine_denominator
        if cosine_denominator > denominator_floor
        else None
    )
    baseline = (
        current_alignment
        if baseline_alignment is None
        else float(baseline_alignment)
    )
    if baseline <= denominator_floor:
        return PersistenceGainAudit(
            status="unresolved_nonpositive_baseline_alignment",
            baseline_alignment=baseline,
            current_alignment=current_alignment,
            alignment_ratio=None,
            alignment_cosine=cosine,
            alignment_denominator=cosine_denominator,
            probe_norm=probe_norm,
            offset_norm=offset_norm,
            clipped_gain=0.0,
            applied_gain=0.0,
        )
    ratio = current_alignment / baseline
    if not np.isfinite(ratio):
        return PersistenceGainAudit(
            status="unresolved_nonfinite_alignment_ratio",
            baseline_alignment=baseline,
            current_alignment=current_alignment,
            alignment_ratio=None,
            alignment_cosine=cosine,
            alignment_denominator=cosine_denominator,
            probe_norm=probe_norm,
            offset_norm=offset_norm,
            clipped_gain=0.0,
            applied_gain=0.0,
        )
    clipped = float(np.clip(ratio, 0.0, 1.0))
    return PersistenceGainAudit(
        status="ok",
        baseline_alignment=baseline,
        current_alignment=current_alignment,
        alignment_ratio=float(ratio),
        alignment_cosine=cosine,
        alignment_denominator=cosine_denominator,
        probe_norm=probe_norm,
        offset_norm=offset_norm,
        clipped_gain=clipped,
        applied_gain=min(float(previous_gain), clipped),
    )


def phase_triggered_terminal_ramp_gain(
    *,
    block_index: int,
    terminal_block: int,
    alignment_cosine: float | None,
    trigger_block: int | None,
) -> TerminalRampAudit:
    """Keep unit gain until phase loss, then linearly retire by the endpoint."""

    if (
        isinstance(block_index, bool)
        or isinstance(terminal_block, bool)
        or not isinstance(block_index, (int, np.integer))
        or not isinstance(terminal_block, (int, np.integer))
        or block_index < 0
        or terminal_block < block_index
        or (
            trigger_block is not None
            and (
                isinstance(trigger_block, bool)
                or not isinstance(trigger_block, (int, np.integer))
                or trigger_block < 0
                or trigger_block > block_index
            )
        )
        or (alignment_cosine is not None and not np.isfinite(alignment_cosine))
    ):
        raise ValueError("terminal-ramp inputs must be finite and ordered")
    trigger = trigger_block
    if trigger is None and (alignment_cosine is None or alignment_cosine <= 0.0):
        trigger = int(block_index)
    if trigger is None:
        if block_index == terminal_block:
            return TerminalRampAudit(
                status="forced_terminal_raw",
                trigger_block=int(terminal_block),
                terminal_block=int(terminal_block),
                applied_gain=0.0,
            )
        return TerminalRampAudit(
            status="pretrigger_unit_gain",
            trigger_block=None,
            terminal_block=int(terminal_block),
            applied_gain=1.0,
        )
    span = terminal_block - trigger
    gain = 0.0 if span == 0 else (terminal_block - block_index) / span
    return TerminalRampAudit(
        status="phase_triggered_terminal_ramp",
        trigger_block=int(trigger),
        terminal_block=int(terminal_block),
        applied_gain=float(np.clip(gain, 0.0, 1.0)),
    )


def fixed_late_terminal_ramp_gain(
    *,
    block_index: int,
    ramp_start_block: int,
    terminal_block: int,
) -> float:
    """Return a unit plateau followed by a fixed linear retirement to zero."""

    values = (block_index, ramp_start_block, terminal_block)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ) or not 0 <= ramp_start_block < terminal_block:
        raise ValueError("late-ramp blocks must be ordered nonnegative integers")
    if not 0 <= block_index <= terminal_block:
        raise ValueError("block_index must lie within the registered horizon")
    if block_index <= ramp_start_block:
        return 1.0
    return float(
        (terminal_block - block_index) / (terminal_block - ramp_start_block)
    )


def transverse_velocity_position_descriptor(
    conservative_state: np.ndarray,
    *,
    nodes: np.ndarray,
    volumes: np.ndarray,
    y_min: float,
    y_max: float,
    density_floor: float = 1.0e-12,
    signal_floor: float = 1.0e-12,
) -> TransverseVelocityPositionDescriptor:
    """Estimate vortex height from one state without truth or spatial gradients."""

    state = np.asarray(conservative_state, dtype=np.float64)
    positions = np.asarray(nodes, dtype=np.float64)
    if (
        state.ndim != 2
        or state.shape[1] < 3
        or not np.isfinite(state).all()
        or positions.shape != (state.shape[0], 2)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("state and nodes must be finite, node-aligned matrices")
    mass = _volumes(volumes, state.shape[0])
    if (
        not np.isfinite([y_min, y_max, density_floor, signal_floor]).all()
        or y_min >= y_max
        or density_floor <= 0.0
        or signal_floor <= 0.0
    ):
        raise ValueError("position-descriptor bounds and floors are invalid")
    if np.any(positions[:, 1] < y_min) or np.any(positions[:, 1] > y_max):
        raise ValueError("node y coordinates lie outside the physical domain")
    density = state[:, 0]
    if np.any(density <= density_floor):
        raise ValueError("position descriptor requires positive density")

    transverse_velocity = state[:, 2] / density
    weights = mass * np.abs(transverse_velocity)
    signal = float(np.sum(weights))
    if signal <= signal_floor * float(np.sum(mass)):
        return TransverseVelocityPositionDescriptor(
            status="unresolved_transverse_velocity",
            vertical_centroid=None,
            normalized_wall_distance=None,
            transverse_velocity_l1=signal,
        )
    centroid = float(np.dot(weights, positions[:, 1]) / signal)
    wall_distance = min(centroid - y_min, y_max - centroid)
    normalized = float(2.0 * wall_distance / (y_max - y_min))
    if not -1.0e-12 <= normalized <= 1.0 + 1.0e-12:
        raise ValueError(
            "transverse-velocity centroid lies outside the physical domain"
        )
    return TransverseVelocityPositionDescriptor(
        status="ok",
        vertical_centroid=centroid,
        normalized_wall_distance=float(np.clip(normalized, 0.0, 1.0)),
        transverse_velocity_l1=signal,
    )


def position_trusts_cross_resolution(
    descriptor: TransverseVelocityPositionDescriptor,
    *,
    threshold: float = FROZEN_POSITION_TRUST_THRESHOLD,
) -> bool:
    """Return the frozen family-specific initial-position trust decision."""

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("position trust threshold must lie in [0, 1]")
    return bool(
        descriptor.status == "ok"
        and descriptor.normalized_wall_distance is not None
        and descriptor.normalized_wall_distance <= threshold
    )


def buffered_frozen_offset_position_route(
    descriptor: TransverseVelocityPositionDescriptor,
    *,
    trust_threshold: float = FROZEN_POSITION_TRUST_THRESHOLD,
    buffer_threshold: float = FROZEN_POSITION_BUFFER_THRESHOLD,
) -> str:
    """Route one initial state without truth or a case identifier."""

    if (
        not np.isfinite([trust_threshold, buffer_threshold]).all()
        or not 0.0 <= trust_threshold <= buffer_threshold <= 1.0
    ):
        raise ValueError("position route thresholds must be ordered in [0, 1]")
    if descriptor.status != "ok" or descriptor.normalized_wall_distance is None:
        return "raw_unresolved"
    distance = float(descriptor.normalized_wall_distance)
    if not np.isfinite(distance) or not 0.0 <= distance <= 1.0:
        raise ValueError("normalized wall distance must lie in [0, 1]")
    if distance <= trust_threshold:
        return "edge_candidate"
    if distance <= buffer_threshold:
        return "raw_uncertainty_buffer"
    return "interior_frozen_offset"


def target_free_front_branch_audit(
    candidate_first_state: np.ndarray,
    candidate_second_state: np.ndarray,
    shadow_first_state: np.ndarray,
    shadow_second_state: np.ndarray,
    *,
    resolution: Resolution,
    x_min: float,
    x_max: float,
    gamma: float,
    shock_center_x: float,
) -> TargetFreeFrontBranchAudit:
    """Compare candidate and raw pressure-front thickness branches without truth."""

    rows = []
    for candidate, shadow in (
        (candidate_first_state, shadow_first_state),
        (candidate_second_state, shadow_second_state),
    ):
        rows.append(
            pressure_profile_shock_metrics(
                np.asarray(candidate, dtype=np.float64),
                np.asarray(shadow, dtype=np.float64),
                resolution=resolution,
                x_min=x_min,
                x_max=x_max,
                gamma=gamma,
                shock_center_x=shock_center_x,
            )
        )
    candidate_thickness = tuple(
        int(row["prediction_shock_thickness_cells"]) for row in rows
    )
    shadow_thickness = tuple(
        int(row["reference_shock_thickness_cells"]) for row in rows
    )
    shifts = [
        float(row["shock_position_absolute_error"])
        for row in rows
        if row["shock_position_absolute_error"] is not None
    ]
    changed = candidate_thickness != shadow_thickness
    return TargetFreeFrontBranchAudit(
        status="branch_changed" if changed else "matched",
        branch_changed=changed,
        first_shadow_thickness_cells=shadow_thickness[0],
        first_candidate_thickness_cells=candidate_thickness[0],
        second_shadow_thickness_cells=shadow_thickness[1],
        second_candidate_thickness_cells=candidate_thickness[1],
        maximum_position_shift=max(shifts) if shifts else None,
    )


def _volumes(value: np.ndarray, nodes: int) -> np.ndarray:
    mass = np.asarray(value, dtype=np.float64)
    if mass.shape == (nodes, 1):
        mass = mass.reshape(nodes)
    if mass.shape != (nodes,) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("volumes must be finite, positive, and node aligned")
    return mass


def _scale(value: Sequence[float] | np.ndarray, components: int) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float64)
    if (
        scale.shape != (components,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("component scale must be finite and positive")
    return scale


def _integral(field: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    return np.sum(volumes[:, None] * field, axis=0)


def _weighted_rms(
    field: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.sum(volumes[:, None] * np.square(field / component_scale[None, :]))
            / (np.sum(volumes) * field.shape[1])
        )
    )


def physical_integral_anchor(
    candidate_state: np.ndarray,
    shadow_state: np.ndarray,
    *,
    volumes: np.ndarray,
    interior_mask: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, IntegralAnchorAudit]:
    """Match shadow integrals with the minimum boundary-zero weighted-L2 shift."""

    candidate = np.asarray(candidate_state, dtype=np.float64)
    shadow = np.asarray(shadow_state, dtype=np.float64)
    if (
        candidate.ndim != 2
        or shadow.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(shadow).all()
    ):
        raise ValueError("candidate and shadow states must be finite and shape aligned")
    mass = _volumes(volumes, candidate.shape[0])
    scale = _scale(component_scale, candidate.shape[1])
    interior = np.asarray(interior_mask, dtype=bool)
    if interior.shape != (candidate.shape[0],) or not np.any(interior):
        raise ValueError("interior_mask must be node aligned and nonempty")
    target = _integral(shadow, mass)
    before = target - _integral(candidate, mass)
    correction = np.zeros_like(candidate)
    correction[interior] = before[None, :] / float(np.sum(mass[interior]))
    anchored = candidate + correction
    after = target - _integral(anchored, mass)
    boundary = correction[~interior]
    audit = IntegralAnchorAudit(
        status="ok",
        maximum_integral_mismatch_before_abs=float(np.max(np.abs(before))),
        maximum_integral_mismatch_after_abs=float(np.max(np.abs(after))),
        maximum_boundary_correction_abs=(
            float(np.max(np.abs(boundary))) if boundary.size else 0.0
        ),
        maximum_idempotence_abs=float(
            np.max(np.abs(after)) / float(np.sum(mass[interior]))
        ),
        correction_rms=_weighted_rms(correction, volumes=mass, component_scale=scale),
    )
    return np.array(anchored, copy=True), correction, audit


def projected_shadow_tether(
    candidate_state: np.ndarray,
    shadow_state: np.ndarray,
    *,
    projector: FixedCosineProjector,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> tuple[np.ndarray, np.ndarray, ProjectedShadowTetherAudit]:
    """Retain only the frozen nonconstant SP19 candidate-shadow displacement."""

    candidate = np.asarray(candidate_state, dtype=np.float64)
    shadow = np.asarray(shadow_state, dtype=np.float64)
    if (
        candidate.ndim != 2
        or shadow.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(shadow).all()
    ):
        raise ValueError("candidate and shadow states must be finite and shape aligned")
    mass = _volumes(volumes, candidate.shape[0])
    scale = _scale(component_scale, candidate.shape[1])
    difference = candidate - shadow
    retained = masked_modal_field(
        difference,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    projected_twice = masked_modal_field(
        retained,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    tethered = shadow + retained
    full_rms = _weighted_rms(difference, volumes=mass, component_scale=scale)
    retained_rms = _weighted_rms(retained, volumes=mass, component_scale=scale)
    boundary = retained[~np.asarray(projector.interior_mask, dtype=bool)]
    audit = ProjectedShadowTetherAudit(
        status="ok",
        full_difference_rms=full_rms,
        retained_difference_rms=retained_rms,
        retained_difference_fraction=(
            retained_rms / full_rms if full_rms > 0.0 else None
        ),
        maximum_integral_difference_abs=float(
            np.max(np.abs(_integral(retained, mass)))
        ),
        maximum_boundary_difference_abs=(
            float(np.max(np.abs(boundary))) if boundary.size else 0.0
        ),
        maximum_projection_idempotence_abs=float(
            np.max(np.abs(projected_twice - retained))
        ),
    )
    return np.array(tethered, copy=True), np.array(retained, copy=True), audit


def slew_limited_projected_shadow_tether(
    candidate_state: np.ndarray,
    shadow_state: np.ndarray,
    previous_accepted_state: np.ndarray,
    previous_shadow_state: np.ndarray,
    *,
    projector: FixedCosineProjector,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    relative_change_limit: float,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> tuple[np.ndarray, np.ndarray, SlewLimitedProjectedTetherAudit]:
    """Slew-limit a projected accepted-shadow displacement.

    The returned displacement remains in the frozen nonconstant modal subspace.
    Its change from the previous accepted-shadow displacement is bounded in the
    physical weighted state norm by ``relative_change_limit`` times the raw
    shadow increment norm.
    """

    candidate = np.asarray(candidate_state, dtype=np.float64)
    shadow = np.asarray(shadow_state, dtype=np.float64)
    previous_accepted = np.asarray(previous_accepted_state, dtype=np.float64)
    previous_shadow = np.asarray(previous_shadow_state, dtype=np.float64)
    if (
        candidate.ndim != 2
        or shadow.shape != candidate.shape
        or previous_accepted.shape != candidate.shape
        or previous_shadow.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(shadow).all()
        or not np.isfinite(previous_accepted).all()
        or not np.isfinite(previous_shadow).all()
    ):
        raise ValueError("current and previous states must be finite and shape aligned")
    if not np.isfinite(relative_change_limit) or not 0.0 <= relative_change_limit <= 1.0:
        raise ValueError("relative_change_limit must lie in [0, 1]")

    mass = _volumes(volumes, candidate.shape[0])
    scale = _scale(component_scale, candidate.shape[1])
    _, target_difference, target_audit = projected_shadow_tether(
        candidate,
        shadow,
        projector=projector,
        volumes=mass,
        component_scale=scale,
        active_cells=active_cells,
    )
    previous_difference = previous_accepted - previous_shadow
    previous_projected = masked_modal_field(
        previous_difference,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    requested_change = target_difference - previous_difference
    shadow_increment = shadow - previous_shadow
    requested_rms = _weighted_rms(
        requested_change, volumes=mass, component_scale=scale
    )
    shadow_increment_rms = _weighted_rms(
        shadow_increment, volumes=mass, component_scale=scale
    )
    change_limit_rms = relative_change_limit * shadow_increment_rms
    applied_scale = (
        min(1.0, change_limit_rms / requested_rms)
        if requested_rms > 0.0
        else 1.0
    )
    retained = previous_difference + applied_scale * requested_change
    retained_twice = masked_modal_field(
        retained,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    tethered = shadow + retained
    applied_change = retained - previous_difference
    applied_change_rms = _weighted_rms(
        applied_change, volumes=mass, component_scale=scale
    )
    boundary = retained[~np.asarray(projector.interior_mask, dtype=bool)]
    previous_boundary = previous_difference[
        ~np.asarray(projector.interior_mask, dtype=bool)
    ]
    expected = previous_difference + applied_scale * requested_change
    audit = SlewLimitedProjectedTetherAudit(
        status="ok" if shadow_increment_rms > 1.0e-12 else "zero_shadow_increment",
        target_difference_rms=target_audit.retained_difference_rms,
        previous_difference_rms=_weighted_rms(
            previous_difference, volumes=mass, component_scale=scale
        ),
        requested_change_rms=requested_rms,
        applied_change_rms=applied_change_rms,
        shadow_increment_rms=shadow_increment_rms,
        change_limit_rms=change_limit_rms,
        applied_to_shadow_increment_ratio=(
            applied_change_rms / shadow_increment_rms
            if shadow_increment_rms > 1.0e-12
            else None
        ),
        applied_scale=applied_scale,
        cap_active=applied_scale < 1.0,
        previous_projection_residual_rms=_weighted_rms(
            previous_projected - previous_difference,
            volumes=mass,
            component_scale=scale,
        ),
        retained_projection_residual_rms=_weighted_rms(
            retained_twice - retained,
            volumes=mass,
            component_scale=scale,
        ),
        maximum_integral_difference_abs=float(
            np.max(np.abs(_integral(retained, mass)))
        ),
        maximum_previous_boundary_abs=(
            float(np.max(np.abs(previous_boundary)))
            if previous_boundary.size
            else 0.0
        ),
        maximum_boundary_difference_abs=(
            float(np.max(np.abs(boundary))) if boundary.size else 0.0
        ),
        maximum_boundary_contraction_violation_abs=(
            max(
                0.0,
                float(np.max(np.abs(boundary)))
                - float(np.max(np.abs(previous_boundary))),
            )
            if boundary.size
            else 0.0
        ),
        maximum_projection_idempotence_abs=(
            target_audit.maximum_projection_idempotence_abs
        ),
        maximum_update_identity_abs=float(np.max(np.abs(retained - expected))),
    )
    return np.array(tethered, copy=True), np.array(retained, copy=True), audit


def relaxed_projected_shadow_tether(
    candidate_state: np.ndarray,
    shadow_state: np.ndarray,
    previous_accepted_state: np.ndarray,
    previous_shadow_state: np.ndarray,
    *,
    projector: FixedCosineProjector,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    relaxation: float,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> tuple[np.ndarray, np.ndarray, SlewLimitedProjectedTetherAudit]:
    """Relax the accepted-shadow displacement toward its projected target."""

    if not np.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
        raise ValueError("relaxation must lie in [0, 1]")
    candidate = np.asarray(candidate_state, dtype=np.float64)
    shadow = np.asarray(shadow_state, dtype=np.float64)
    previous_accepted = np.asarray(previous_accepted_state, dtype=np.float64)
    previous_shadow = np.asarray(previous_shadow_state, dtype=np.float64)
    if (
        candidate.ndim != 2
        or shadow.shape != candidate.shape
        or previous_accepted.shape != candidate.shape
        or previous_shadow.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(shadow).all()
        or not np.isfinite(previous_accepted).all()
        or not np.isfinite(previous_shadow).all()
    ):
        raise ValueError("current and previous states must be finite and shape aligned")

    mass = _volumes(volumes, candidate.shape[0])
    scale = _scale(component_scale, candidate.shape[1])
    _, target_difference, target_audit = projected_shadow_tether(
        candidate,
        shadow,
        projector=projector,
        volumes=mass,
        component_scale=scale,
        active_cells=active_cells,
    )
    previous_difference = previous_accepted - previous_shadow
    previous_projected = masked_modal_field(
        previous_difference,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    requested_change = target_difference - previous_difference
    applied_change = relaxation * requested_change
    retained = previous_difference + applied_change
    retained_twice = masked_modal_field(
        retained,
        projector,
        component_scale=scale,
        active_cells=active_cells,
    )
    shadow_increment = shadow - previous_shadow
    requested_rms = _weighted_rms(
        requested_change, volumes=mass, component_scale=scale
    )
    applied_rms = _weighted_rms(
        applied_change, volumes=mass, component_scale=scale
    )
    shadow_increment_rms = _weighted_rms(
        shadow_increment, volumes=mass, component_scale=scale
    )
    boundary_mask = ~np.asarray(projector.interior_mask, dtype=bool)
    previous_boundary = previous_difference[boundary_mask]
    boundary = retained[boundary_mask]
    tethered = shadow + retained
    audit = SlewLimitedProjectedTetherAudit(
        status="ok" if requested_rms > 1.0e-12 else "zero_target_gap",
        target_difference_rms=target_audit.retained_difference_rms,
        previous_difference_rms=_weighted_rms(
            previous_difference, volumes=mass, component_scale=scale
        ),
        requested_change_rms=requested_rms,
        applied_change_rms=applied_rms,
        shadow_increment_rms=shadow_increment_rms,
        change_limit_rms=relaxation * requested_rms,
        applied_to_shadow_increment_ratio=(
            applied_rms / shadow_increment_rms
            if shadow_increment_rms > 1.0e-12
            else None
        ),
        applied_scale=relaxation,
        cap_active=relaxation < 1.0 and requested_rms > 0.0,
        previous_projection_residual_rms=_weighted_rms(
            previous_projected - previous_difference,
            volumes=mass,
            component_scale=scale,
        ),
        retained_projection_residual_rms=_weighted_rms(
            retained_twice - retained,
            volumes=mass,
            component_scale=scale,
        ),
        maximum_integral_difference_abs=float(
            np.max(np.abs(_integral(retained, mass)))
        ),
        maximum_previous_boundary_abs=(
            float(np.max(np.abs(previous_boundary)))
            if previous_boundary.size
            else 0.0
        ),
        maximum_boundary_difference_abs=(
            float(np.max(np.abs(boundary))) if boundary.size else 0.0
        ),
        maximum_boundary_contraction_violation_abs=(
            max(
                0.0,
                float(np.max(np.abs(boundary)))
                - float(np.max(np.abs(previous_boundary))),
            )
            if boundary.size
            else 0.0
        ),
        maximum_projection_idempotence_abs=(
            target_audit.maximum_projection_idempotence_abs
        ),
        maximum_update_identity_abs=float(
            np.max(
                np.abs(
                    retained
                    - (previous_difference + relaxation * requested_change)
                )
            )
        ),
    )
    return np.array(tethered, copy=True), np.array(retained, copy=True), audit


def synchronized_response_filtered_block(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    volumes: np.ndarray,
    residual_scale: Sequence[float] | np.ndarray,
    state_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> ResponseFilteredBlock:
    """Construct one two-state block using exactly four model predictions."""

    first = synchronized_sparse_modal_step(
        native_state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        policy="sp19_fine_away_half",
        volumes=volumes,
        component_scale=residual_scale,
        active_cells=active_cells,
    )
    raw_first = np.asarray(first.predictions[contract.native], dtype=np.float64)
    corrected_first = np.asarray(first.next_native_state, dtype=np.float64)

    def native_predictor(state: np.ndarray) -> np.ndarray:
        return predictor(contract.native, state)

    lookahead = propagated_lookahead(
        raw_first,
        np.asarray(first.correction, dtype=np.float64),
        predictor=native_predictor,
    )
    filtered = masked_modal_field(
        lookahead.response,
        projector,
        component_scale=state_scale,
        active_cells=active_cells,
    )
    projected_twice = masked_modal_field(
        filtered,
        projector,
        component_scale=state_scale,
        active_cells=active_cells,
    )
    mass = _volumes(volumes, raw_first.shape[0])
    scale = _scale(state_scale, raw_first.shape[1])
    boundary = ~np.asarray(projector.interior_mask, dtype=bool)
    first_boundary = np.asarray(first.correction, dtype=np.float64)[boundary]
    filtered_boundary = filtered[boundary]
    full_rms = _weighted_rms(
        lookahead.response,
        volumes=mass,
        component_scale=scale,
    )
    filtered_rms = _weighted_rms(
        filtered,
        volumes=mass,
        component_scale=scale,
    )
    audit = ResponseProjectionAudit(
        status="ok",
        maximum_first_correction_integral_abs=float(
            np.max(np.abs(_integral(first.correction, mass)))
        ),
        maximum_filtered_response_integral_abs=float(
            np.max(np.abs(_integral(filtered, mass)))
        ),
        maximum_first_boundary_abs=(
            float(np.max(np.abs(first_boundary))) if first_boundary.size else 0.0
        ),
        maximum_filtered_boundary_abs=(
            float(np.max(np.abs(filtered_boundary))) if filtered_boundary.size else 0.0
        ),
        maximum_projection_idempotence_abs=float(
            np.max(np.abs(projected_twice - filtered))
        ),
        full_response_rms=full_rms,
        filtered_response_rms=filtered_rms,
        retained_response_fraction=(
            filtered_rms / full_rms if full_rms > 1.0e-12 else None
        ),
    )
    return ResponseFilteredBlock(
        raw_first_state=np.array(raw_first, copy=True),
        corrected_first_state=np.array(corrected_first, copy=True),
        raw_second_state=np.array(lookahead.raw_prediction, copy=True),
        fully_corrected_second_state=np.array(
            lookahead.corrected_prediction, copy=True
        ),
        full_response=np.array(lookahead.response, copy=True),
        filtered_response=np.array(filtered, copy=True),
        filtered_second_state=np.array(lookahead.raw_prediction + filtered, copy=True),
        first_correction=np.asarray(first.correction, dtype=np.float64).copy(),
        first_audit=first.audit,
        projection_audit=audit,
    )


def synchronized_shadow_anchored_block(
    accepted_native_state: np.ndarray,
    shadow_native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    volumes: np.ndarray,
    residual_scale: Sequence[float] | np.ndarray,
    state_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> ShadowAnchoredBlock:
    """Emit two anchored states from one accepted state and one raw shadow."""

    accepted = np.asarray(accepted_native_state, dtype=np.float64)
    shadow = np.asarray(shadow_native_state, dtype=np.float64)
    if accepted.ndim != 2 or shadow.shape != accepted.shape:
        raise ValueError("accepted and shadow states must be shape-aligned matrices")
    mass = _volumes(volumes, accepted.shape[0])
    state_component_scale = _scale(state_scale, accepted.shape[1])

    shadow_first = np.asarray(predictor(contract.native, shadow), dtype=np.float64)
    shadow_second = np.asarray(
        predictor(contract.native, shadow_first), dtype=np.float64
    )
    first = synchronized_sparse_modal_step(
        accepted,
        contract=contract,
        projector=projector,
        predictor=predictor,
        policy="sp19_fine_away_half",
        volumes=mass,
        component_scale=residual_scale,
        active_cells=active_cells,
    )
    candidate_raw_first = np.asarray(
        first.predictions[contract.native], dtype=np.float64
    )
    candidate_first = np.asarray(first.next_native_state, dtype=np.float64)
    anchored_first, first_anchor, first_anchor_audit = physical_integral_anchor(
        candidate_first,
        shadow_first,
        volumes=mass,
        interior_mask=projector.interior_mask,
        component_scale=state_component_scale,
    )

    def native_predictor(state: np.ndarray) -> np.ndarray:
        return predictor(contract.native, state)

    lookahead = propagated_lookahead(
        candidate_raw_first,
        anchored_first - candidate_raw_first,
        predictor=native_predictor,
    )
    filtered = masked_modal_field(
        lookahead.response,
        projector,
        component_scale=state_component_scale,
        active_cells=active_cells,
    )
    projected_twice = masked_modal_field(
        filtered,
        projector,
        component_scale=state_component_scale,
        active_cells=active_cells,
    )
    candidate_second = np.asarray(lookahead.raw_prediction + filtered, dtype=np.float64)
    anchored_second, second_anchor, second_anchor_audit = physical_integral_anchor(
        candidate_second,
        shadow_second,
        volumes=mass,
        interior_mask=projector.interior_mask,
        component_scale=state_component_scale,
    )
    boundary = ~np.asarray(projector.interior_mask, dtype=bool)
    first_boundary = np.asarray(first.correction, dtype=np.float64)[boundary]
    filtered_boundary = filtered[boundary]
    full_rms = _weighted_rms(
        lookahead.response,
        volumes=mass,
        component_scale=state_component_scale,
    )
    filtered_rms = _weighted_rms(
        filtered,
        volumes=mass,
        component_scale=state_component_scale,
    )
    response_audit = ResponseProjectionAudit(
        status="ok",
        maximum_first_correction_integral_abs=float(
            np.max(np.abs(_integral(first.correction, mass)))
        ),
        maximum_filtered_response_integral_abs=float(
            np.max(np.abs(_integral(filtered, mass)))
        ),
        maximum_first_boundary_abs=(
            float(np.max(np.abs(first_boundary))) if first_boundary.size else 0.0
        ),
        maximum_filtered_boundary_abs=(
            float(np.max(np.abs(filtered_boundary))) if filtered_boundary.size else 0.0
        ),
        maximum_projection_idempotence_abs=float(
            np.max(np.abs(projected_twice - filtered))
        ),
        full_response_rms=full_rms,
        filtered_response_rms=filtered_rms,
        retained_response_fraction=(
            filtered_rms / full_rms if full_rms > 1.0e-12 else None
        ),
    )
    return ShadowAnchoredBlock(
        shadow_first_state=np.array(shadow_first, copy=True),
        shadow_second_state=np.array(shadow_second, copy=True),
        candidate_raw_first_state=np.array(candidate_raw_first, copy=True),
        candidate_first_before_anchor=np.array(candidate_first, copy=True),
        anchored_first_state=np.array(anchored_first, copy=True),
        candidate_raw_second_state=np.array(lookahead.raw_prediction, copy=True),
        full_response=np.array(lookahead.response, copy=True),
        filtered_response=np.array(filtered, copy=True),
        candidate_second_before_anchor=np.array(candidate_second, copy=True),
        anchored_second_state=np.array(anchored_second, copy=True),
        first_correction=np.asarray(first.correction, dtype=np.float64).copy(),
        first_anchor_correction=np.array(first_anchor, copy=True),
        second_anchor_correction=np.array(second_anchor, copy=True),
        first_audit=first.audit,
        response_audit=response_audit,
        first_anchor_audit=first_anchor_audit,
        second_anchor_audit=second_anchor_audit,
    )


def recurrent_response_filtered_blocks(
    initial_state: np.ndarray,
    *,
    block_count: int,
    block_builder: Callable[[np.ndarray], ResponseFilteredBlock],
    response_mode: ResponseMode,
) -> ResponseFilteredTrajectory:
    """Advance one accepted trajectory by fixed two-state blocks."""

    initial = np.asarray(initial_state, dtype=np.float64)
    if initial.ndim != 2 or not np.isfinite(initial).all():
        raise ValueError("initial_state must be a finite [nodes, components] array")
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if response_mode not in {"full", "filtered"}:
        raise ValueError("response_mode must be 'full' or 'filtered'")

    states = [np.array(initial, copy=True)]
    blocks = []
    current = np.array(initial, copy=True)
    for _ in range(block_count):
        block = block_builder(np.array(current, copy=True))
        first = np.asarray(block.corrected_first_state, dtype=np.float64)
        second = np.asarray(
            block.fully_corrected_second_state
            if response_mode == "full"
            else block.filtered_second_state,
            dtype=np.float64,
        )
        if first.shape != initial.shape or second.shape != initial.shape:
            raise ValueError("block outputs must match the native state shape")
        states.extend((np.array(first, copy=True), np.array(second, copy=True)))
        blocks.append(block)
        current = np.array(second, copy=True)
    return ResponseFilteredTrajectory(
        states=tuple(states),
        blocks=tuple(blocks),
        response_mode=response_mode,
    )
