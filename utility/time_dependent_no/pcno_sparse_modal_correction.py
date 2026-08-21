"""Sparse physical-modal fine-discrepancy correction for W26-L5.

The active mode/component cells are selected once from calibration evidence.
At inference the correction uses only synchronized native and fine predictions
of one native state; no reference value is accepted by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FixedCosineProjector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    DENOMINATOR_FLOOR,
    MAX_CORRECTION_TO_NATIVE_INCREMENT,
    FineDiscrepancyAudit,
    FineDiscrepancyStep,
    fine_discrepancy_correction,
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    as_model_state,
    restrict_nested_state,
    weighted_scaled_rms,
)

ModalCell = tuple[int, int]
SparsePolicy = Literal["zero", "sp19_fine_away_half"]

SPARSE_POLICY: SparsePolicy = "sp19_fine_away_half"
POLICIES: tuple[SparsePolicy, ...] = ("zero", SPARSE_POLICY)
AFFINE_MODAL_POLICY = "sp19_fine_full_affine"
AFFINE_MODAL_RIDGE = 1.0e-6
AFFINE_MODAL_LAST_INPUT_CALL = 29
BINARY_POSITION_PHASE_POLICY = "binary_position_phase_sp19_affine"
BINARY_POSITION_PHASE_CALLS = (*range(8), *range(22, 30))
FROZEN_ACTIVE_CELLS: tuple[ModalCell, ...] = (
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (7, 0),
    (7, 2),
    (7, 3),
)


@dataclass(frozen=True)
class AffineModalCorrectionAudit:
    """Target-free closure record for one frozen affine modal correction."""

    policy: str
    status: str
    input_call: int
    normalized_phase: float
    native_increment_rms: float
    raw_correction_rms: float
    correction_rms: float
    correction_to_native_increment: float | None
    applied_scale: float
    cap_active: bool
    maximum_scaled_component_mean_abs: float
    maximum_excluded_abs: float
    maximum_inactive_coordinate_abs: float
    maximum_modal_reconstruction_abs: float


@dataclass(frozen=True)
class AffineModalStep:
    """One two-call affine modal update with one retained native state."""

    prepared_inputs: Any
    predictions: Mapping[Resolution, np.ndarray]
    basis: NativeIncrementBasis
    correction: np.ndarray
    audit: AffineModalCorrectionAudit
    next_native_state: np.ndarray


@dataclass(frozen=True)
class BinaryAffineModalStep:
    """One conditional affine update that never queries fine when inactive."""

    prepared_inputs: Any | None
    predictions: Mapping[Resolution, np.ndarray]
    basis: NativeIncrementBasis
    correction: np.ndarray
    audit: AffineModalCorrectionAudit
    next_native_state: np.ndarray
    position_selected: bool
    correction_active: bool
    logical_call_count: int


@dataclass(frozen=True)
class ShadowTetheredBinaryAffineStep:
    """One raw-shadow call, one affine proposal, and one retained native state."""

    shadow_prediction: np.ndarray
    proposal: BinaryAffineModalStep
    retained_displacement: np.ndarray
    tether_audit: Any
    next_native_state: np.ndarray
    logical_call_count: int
    maximum_update_identity_abs: float


def _strict_index(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def binary_position_phase_correction_active(
    *, position_selected: bool, input_call: int
) -> bool:
    """Apply the frozen correction only to selected positions and phase calls."""

    if not isinstance(position_selected, (bool, np.bool_)):
        raise TypeError("position_selected must be Boolean")
    call = _strict_index(input_call, name="input_call")
    if call < 0 or call > AFFINE_MODAL_LAST_INPUT_CALL:
        raise ValueError("binary position-phase input_call is outside 0..29")
    return bool(position_selected and call in BINARY_POSITION_PHASE_CALLS)


def validate_active_cells(
    active_cells: Sequence[ModalCell],
    *,
    rank: int,
    components: int,
) -> tuple[ModalCell, ...]:
    """Return a canonical nonconstant modal mask or fail closed."""

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 2:
        raise ValueError("rank must be an integer at least two")
    if (
        isinstance(components, bool)
        or not isinstance(components, (int, np.integer))
        or components < 1
    ):
        raise ValueError("components must be a positive integer")
    canonical: list[ModalCell] = []
    for raw_cell in active_cells:
        if not isinstance(raw_cell, (tuple, list)) or len(raw_cell) != 2:
            raise ValueError("each active cell must be a (mode, component) pair")
        mode = _strict_index(raw_cell[0], name="mode index")
        component = _strict_index(raw_cell[1], name="component index")
        if not 1 <= mode < int(rank):
            raise ValueError("active modes must be nonconstant and below rank")
        if not 0 <= component < int(components):
            raise ValueError("active component is outside the field")
        canonical.append((mode, component))
    if len(set(canonical)) != len(canonical):
        raise ValueError("active modal cells must be unique")
    if not canonical:
        raise ValueError("a nonzero sparse policy needs at least one active cell")
    return tuple(sorted(canonical))


def positive_half_skill_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int = 8,
    components: int = 4,
) -> tuple[ModalCell, ...]:
    """Derive the strictly-positive half-gain mask from a complete table."""

    expected = {
        (mode, component)
        for mode in range(int(rank))
        for component in range(int(components))
    }
    observed: dict[ModalCell, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("modal summary rows must be mappings")
        mode = _strict_index(row.get("mode_index"), name="mode_index")
        component = _strict_index(row.get("component"), name="component")
        cell = (mode, component)
        if cell not in expected:
            raise ValueError(f"unexpected modal summary cell: {cell}")
        if cell in observed:
            raise ValueError(f"duplicate modal summary cell: {cell}")
        if row.get("fixed_half_skill_status") != "ok":
            raise ValueError(f"unresolved fixed-half skill for cell {cell}")
        raw_skill = row.get("fixed_half_skill")
        if isinstance(raw_skill, (bool, np.bool_)) or not isinstance(
            raw_skill, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"fixed-half skill must be numeric for cell {cell}")
        skill = float(raw_skill)
        if not np.isfinite(skill):
            raise ValueError(f"fixed-half skill must be finite for cell {cell}")
        observed[cell] = skill
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise ValueError(f"modal summary inventory differs; missing={missing}")
    return validate_active_cells(
        [cell for cell, skill in observed.items() if cell[0] > 0 and skill > 0.0],
        rank=rank,
        components=components,
    )


def masked_modal_field(
    field: np.ndarray,
    projector: FixedCosineProjector,
    *,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> np.ndarray:
    """Project a field onto exactly the declared modal/component cells."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("field must be a finite node-by-component array")
    coordinates = modal_coordinates(
        values,
        projector,
        component_scale=component_scale,
    )
    cells = validate_active_cells(
        active_cells,
        rank=coordinates.shape[0],
        components=coordinates.shape[1],
    )
    retained = np.zeros_like(coordinates)
    for mode, component in cells:
        retained[mode, component] = coordinates[mode, component]
    return reconstruct_modal_field(
        retained,
        projector,
        component_scale=component_scale,
    )


def masked_increment_basis(
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> NativeIncrementBasis:
    """Replace only the deployable fine feature by its sparse projection."""

    native = np.asarray(basis.native_increment, dtype=np.float64)
    feature = np.asarray(basis.fine_minus_native, dtype=np.float64)
    if native.ndim != 2 or feature.shape != native.shape:
        raise ValueError("native increment and fine discrepancy must align")
    masked = masked_modal_field(
        feature,
        projector,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    return NativeIncrementBasis(
        native_increment=np.array(native, copy=True),
        coarse_on_native=np.asarray(basis.coarse_on_native, dtype=np.float64).copy(),
        fine_on_native=native + masked,
        native_minus_coarse=np.asarray(
            basis.native_minus_coarse, dtype=np.float64
        ).copy(),
        fine_minus_native=masked,
    )


def masked_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
    *,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> DiagnosticSnapshot:
    """Create a scoring snapshot with the same target and a masked feature."""

    return DiagnosticSnapshot(
        case_id=snapshot.case_id,
        group_id=snapshot.group_id,
        input_call=int(snapshot.input_call),
        basis=masked_increment_basis(
            snapshot.basis,
            projector,
            component_scale=snapshot.component_scale,
            active_cells=active_cells,
        ),
        target_correction=np.asarray(
            snapshot.target_correction, dtype=np.float64
        ).copy(),
        volumes=np.asarray(snapshot.volumes, dtype=np.float64).copy(),
        component_scale=np.asarray(snapshot.component_scale, dtype=np.float64).copy(),
        masks={key: np.asarray(value).copy() for key, value in snapshot.masks.items()},
    )


def sparse_modal_correction(
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    policy: SparsePolicy,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
    maximum_relative_norm: float = MAX_CORRECTION_TO_NATIVE_INCREMENT,
) -> tuple[np.ndarray, FineDiscrepancyAudit]:
    """Apply zero or the fixed sparse half-gain correction."""

    if policy not in POLICIES:
        raise ValueError(f"unknown sparse-modal policy: {policy!r}")
    if policy == "zero":
        correction, audit = fine_discrepancy_correction(
            basis,
            projector,
            policy="zero",
            volumes=volumes,
            component_scale=component_scale,
            maximum_relative_norm=maximum_relative_norm,
        )
        return correction, replace(audit, policy=policy)
    masked = masked_increment_basis(
        basis,
        projector,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    correction, audit = fine_discrepancy_correction(
        masked,
        projector,
        policy="rank7_fine_away_half",
        volumes=volumes,
        component_scale=component_scale,
        maximum_relative_norm=maximum_relative_norm,
    )
    return correction, replace(audit, policy=policy)


def affine_modal_correction(
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    input_call: int,
    start_coefficients: Sequence[Sequence[float]] | np.ndarray,
    end_coefficients: Sequence[Sequence[float]] | np.ndarray,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
    maximum_relative_norm: float = MAX_CORRECTION_TO_NATIVE_INCREMENT,
) -> tuple[np.ndarray, AffineModalCorrectionAudit]:
    """Decode the frozen two-call affine phase map without accepting truth."""

    call = _strict_index(input_call, name="input_call")
    if call < 0 or call > AFFINE_MODAL_LAST_INPUT_CALL:
        raise ValueError("affine modal input_call is outside the fixed horizon")
    if not np.isfinite(maximum_relative_norm) or maximum_relative_norm <= 0.0:
        raise ValueError("maximum_relative_norm must be finite and positive")
    native = np.asarray(basis.native_increment, dtype=np.float64)
    fine_feature = np.asarray(basis.fine_minus_native, dtype=np.float64)
    if (
        native.ndim != 2
        or fine_feature.shape != native.shape
        or not np.isfinite(native).all()
        or not np.isfinite(fine_feature).all()
    ):
        raise ValueError("native increment and fine discrepancy must be finite and align")
    cells = validate_active_cells(
        active_cells,
        rank=projector.q_matrix.shape[1],
        components=native.shape[1],
    )
    start = np.asarray(start_coefficients, dtype=np.float64)
    end = np.asarray(end_coefficients, dtype=np.float64)
    expected_shape = (len(cells), len(cells))
    if (
        start.shape != expected_shape
        or end.shape != expected_shape
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
    ):
        raise ValueError(f"affine modal coefficients must have shape {expected_shape}")
    scale = np.asarray(component_scale, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if (
        scale.shape != (native.shape[1],)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or mass.shape != (native.shape[0],)
        or not np.isfinite(mass).all()
        or np.any(mass <= 0.0)
    ):
        raise ValueError("affine modal metric contract is invalid")

    coordinates = modal_coordinates(
        fine_feature,
        projector,
        component_scale=scale,
    )
    feature = np.asarray([coordinates[cell] for cell in cells], dtype=np.float64)
    phase = call / AFFINE_MODAL_LAST_INPUT_CALL
    effective = (1.0 - phase) * start + phase * end
    predicted = feature @ effective
    correction_coordinates = np.zeros_like(coordinates)
    for cell, value in zip(cells, predicted, strict=True):
        correction_coordinates[cell] = value
    raw_correction = reconstruct_modal_field(
        correction_coordinates,
        projector,
        component_scale=scale,
    )
    reconstructed_coordinates = modal_coordinates(
        raw_correction,
        projector,
        component_scale=scale,
    )
    reconstructed = reconstruct_modal_field(
        reconstructed_coordinates,
        projector,
        component_scale=scale,
    )
    raw_rms = weighted_scaled_rms(
        raw_correction,
        volumes=mass,
        component_scale=scale,
    )
    native_rms = weighted_scaled_rms(
        native,
        volumes=mass,
        component_scale=scale,
    )
    status = "ok"
    applied_scale = 1.0
    if raw_rms <= DENOMINATOR_FLOOR:
        status = "unresolved_zero_correction"
        applied_scale = 0.0
    elif native_rms <= DENOMINATOR_FLOOR:
        status = "unresolved_zero_native_increment"
        applied_scale = 0.0
    else:
        applied_scale = min(1.0, maximum_relative_norm * native_rms / raw_rms)
    correction = applied_scale * raw_correction
    correction_rms = weighted_scaled_rms(
        correction,
        volumes=mass,
        component_scale=scale,
    )
    ratio = correction_rms / native_rms if native_rms > DENOMINATOR_FLOOR else None
    scaled_mean = (
        np.einsum(
            "n,nc->c",
            mass,
            correction / scale[None, :],
            optimize=True,
        )
        / float(np.sum(mass))
    )
    inactive = {
        (mode, component)
        for mode in range(coordinates.shape[0])
        for component in range(coordinates.shape[1])
    } - set(cells)
    excluded = correction[~projector.interior_mask]
    return correction, AffineModalCorrectionAudit(
        policy=AFFINE_MODAL_POLICY,
        status=status,
        input_call=call,
        normalized_phase=float(phase),
        native_increment_rms=native_rms,
        raw_correction_rms=raw_rms,
        correction_rms=correction_rms,
        correction_to_native_increment=ratio,
        applied_scale=float(applied_scale),
        cap_active=bool(applied_scale < 1.0),
        maximum_scaled_component_mean_abs=float(np.max(np.abs(scaled_mean))),
        maximum_excluded_abs=(
            0.0 if excluded.size == 0 else float(np.max(np.abs(excluded)))
        ),
        maximum_inactive_coordinate_abs=(
            0.0
            if not inactive
            else float(max(abs(correction_coordinates[cell]) for cell in inactive))
        ),
        maximum_modal_reconstruction_abs=float(
            np.max(np.abs(reconstructed - raw_correction))
        ),
    )


def synchronized_affine_modal_step(
    native_state: np.ndarray,
    *,
    input_call: int,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    start_coefficients: Sequence[Sequence[float]] | np.ndarray,
    end_coefficients: Sequence[Sequence[float]] | np.ndarray,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> AffineModalStep:
    """Make native/fine same-state calls and retain one corrected native state."""

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    predictions: dict[Resolution, np.ndarray] = {}
    for resolution in (contract.native, contract.fine):
        prediction = np.asarray(
            predictor(
                resolution,
                np.array(prepared.model_inputs[resolution], copy=True),
            ),
            dtype=np.float64,
        )
        expected_shape = (resolution[0] * resolution[1], 4)
        if prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise ValueError(
                f"prediction at {resolution} must be finite with shape {expected_shape}"
            )
        predictions[resolution] = prediction
    native_current = np.asarray(prepared.model_inputs[contract.native], dtype=np.float64)
    fine_current = np.asarray(prepared.model_inputs[contract.fine], dtype=np.float64)
    native_increment = predictions[contract.native] - native_current
    fine_increment = predictions[contract.fine] - fine_current
    fine_on_native = restrict_nested_state(
        fine_increment,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=fine_on_native,
        native_minus_coarse=np.zeros_like(native_increment),
        fine_minus_native=fine_on_native - native_increment,
    )
    correction, audit = affine_modal_correction(
        basis,
        projector,
        input_call=input_call,
        start_coefficients=start_coefficients,
        end_coefficients=end_coefficients,
        volumes=volumes,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    return AffineModalStep(
        prepared_inputs=prepared,
        predictions=predictions,
        basis=basis,
        correction=correction,
        audit=audit,
        next_native_state=np.asarray(
            predictions[contract.native], dtype=np.float64
        )
        + correction,
    )


def synchronized_binary_affine_modal_step(
    native_state: np.ndarray,
    *,
    position_selected: bool,
    input_call: int,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    start_coefficients: Sequence[Sequence[float]] | np.ndarray,
    end_coefficients: Sequence[Sequence[float]] | np.ndarray,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> BinaryAffineModalStep:
    """Update one native state and skip the fine query on every inactive call."""

    state = np.asarray(native_state, dtype=np.float64)
    expected_native_shape = (contract.native[0] * contract.native[1], 4)
    if state.shape != expected_native_shape or not np.isfinite(state).all():
        raise ValueError(
            f"native state must be finite with shape {expected_native_shape}"
        )
    active = binary_position_phase_correction_active(
        position_selected=position_selected,
        input_call=input_call,
    )
    if active:
        step = synchronized_affine_modal_step(
            state,
            input_call=input_call,
            contract=contract,
            projector=projector,
            predictor=predictor,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
            volumes=volumes,
            component_scale=component_scale,
            active_cells=active_cells,
        )
        return BinaryAffineModalStep(
            prepared_inputs=step.prepared_inputs,
            predictions=step.predictions,
            basis=step.basis,
            correction=step.correction,
            audit=step.audit,
            next_native_state=step.next_native_state,
            position_selected=bool(position_selected),
            correction_active=True,
            logical_call_count=2,
        )

    native_input = as_model_state(state)
    native_prediction = np.asarray(
        predictor(contract.native, np.array(native_input, copy=True)),
        dtype=np.float64,
    )
    if (
        native_prediction.shape != expected_native_shape
        or not np.isfinite(native_prediction).all()
    ):
        raise ValueError(
            f"prediction at {contract.native} must be finite with shape "
            f"{expected_native_shape}"
        )
    zero = np.zeros_like(native_prediction)
    native_increment = native_prediction - native_input
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=np.array(native_increment, copy=True),
        native_minus_coarse=zero,
        fine_minus_native=zero,
    )
    native_rms = weighted_scaled_rms(
        native_increment,
        volumes=volumes,
        component_scale=component_scale,
    )
    audit = AffineModalCorrectionAudit(
        policy=BINARY_POSITION_PHASE_POLICY,
        status="inactive_exact_raw_native",
        input_call=_strict_index(input_call, name="input_call"),
        normalized_phase=(
            _strict_index(input_call, name="input_call")
            / AFFINE_MODAL_LAST_INPUT_CALL
        ),
        native_increment_rms=native_rms,
        raw_correction_rms=0.0,
        correction_rms=0.0,
        correction_to_native_increment=(0.0 if native_rms > 0.0 else None),
        applied_scale=0.0,
        cap_active=False,
        maximum_scaled_component_mean_abs=0.0,
        maximum_excluded_abs=0.0,
        maximum_inactive_coordinate_abs=0.0,
        maximum_modal_reconstruction_abs=0.0,
    )
    return BinaryAffineModalStep(
        prepared_inputs=None,
        predictions={contract.native: native_prediction},
        basis=basis,
        correction=zero,
        audit=audit,
        next_native_state=native_prediction,
        position_selected=bool(position_selected),
        correction_active=False,
        logical_call_count=1,
    )


def synchronized_shadow_tethered_binary_affine_step(
    accepted_native_state: np.ndarray,
    shadow_native_state: np.ndarray,
    *,
    position_selected: bool,
    input_call: int,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    start_coefficients: Sequence[Sequence[float]] | np.ndarray,
    end_coefficients: Sequence[Sequence[float]] | np.ndarray,
    volumes: np.ndarray,
    residual_scale: Sequence[float] | np.ndarray,
    state_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> ShadowTetheredBinaryAffineStep:
    """Advance an independent raw shadow and project one affine proposal to it."""

    from utility.time_dependent_no.pcno_response_filtered_block import (
        projected_shadow_tether,
    )

    shadow_input = as_model_state(shadow_native_state)
    expected_shape = (contract.native[0] * contract.native[1], 4)
    if shadow_input.shape != expected_shape or not np.isfinite(shadow_input).all():
        raise ValueError(
            f"shadow state must be finite with shape {expected_shape}"
        )
    shadow_prediction = np.asarray(
        predictor(contract.native, np.array(shadow_input, copy=True)),
        dtype=np.float64,
    )
    if (
        shadow_prediction.shape != expected_shape
        or not np.isfinite(shadow_prediction).all()
    ):
        raise ValueError(
            f"shadow prediction must be finite with shape {expected_shape}"
        )
    proposal = synchronized_binary_affine_modal_step(
        accepted_native_state,
        position_selected=position_selected,
        input_call=input_call,
        contract=contract,
        projector=projector,
        predictor=predictor,
        start_coefficients=start_coefficients,
        end_coefficients=end_coefficients,
        volumes=volumes,
        component_scale=residual_scale,
        active_cells=active_cells,
    )
    next_state, retained, tether_audit = projected_shadow_tether(
        proposal.next_native_state,
        shadow_prediction,
        projector=projector,
        volumes=volumes,
        component_scale=state_scale,
        active_cells=active_cells,
    )
    return ShadowTetheredBinaryAffineStep(
        shadow_prediction=np.array(shadow_prediction, copy=True),
        proposal=proposal,
        retained_displacement=np.array(retained, copy=True),
        tether_audit=tether_audit,
        next_native_state=np.array(next_state, copy=True),
        logical_call_count=1 + proposal.logical_call_count,
        maximum_update_identity_abs=float(
            np.max(np.abs(next_state - shadow_prediction - retained))
        ),
    )


def synchronized_sparse_modal_step(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    policy: SparsePolicy,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
) -> FineDiscrepancyStep:
    """Make two same-state predictions and retain one corrected native state."""

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    predictions: dict[Resolution, np.ndarray] = {}
    for resolution in (contract.native, contract.fine):
        prediction = np.asarray(
            predictor(
                resolution,
                np.array(prepared.model_inputs[resolution], copy=True),
            ),
            dtype=np.float64,
        )
        expected_shape = (resolution[0] * resolution[1], 4)
        if prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise ValueError(
                f"prediction at {resolution} must be finite with shape {expected_shape}"
            )
        predictions[resolution] = prediction

    native_current = np.asarray(
        prepared.model_inputs[contract.native], dtype=np.float64
    )
    fine_current = np.asarray(prepared.model_inputs[contract.fine], dtype=np.float64)
    native_increment = predictions[contract.native] - native_current
    fine_increment = predictions[contract.fine] - fine_current
    fine_on_native = restrict_nested_state(
        fine_increment,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    basis = NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=fine_on_native,
        native_minus_coarse=np.zeros_like(native_increment),
        fine_minus_native=fine_on_native - native_increment,
    )
    correction, audit = sparse_modal_correction(
        basis,
        projector,
        policy=policy,
        volumes=volumes,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    return FineDiscrepancyStep(
        prepared_inputs=prepared,
        predictions=predictions,
        basis=masked_increment_basis(
            basis,
            projector,
            component_scale=component_scale,
            active_cells=active_cells,
        )
        if policy == SPARSE_POLICY
        else basis,
        correction=correction,
        audit=audit,
        next_native_state=predictions[contract.native] + correction,
    )
