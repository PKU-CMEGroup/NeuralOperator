"""Fixed native/fine cross-resolution correction primitives for W26-L5.

The deployable feature is built only from synchronized native and fine model
predictions of one native state.  Reference error is used by experiment
scorers, never by the correction or its trust region.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    PreparedCommonNativeInputs,
    ResolutionContract,
    case_first_statistics,
    prepare_common_native_inputs,
    score_scalar_correction,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    VIEW_SPECS,
    FixedCosineProjector,
    _statistics_for_view,
    _validate_snapshot_inventory,
    score_cell,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    restrict_nested_state,
    weighted_scaled_rms,
)

CorrectionPolicy = Literal[
    "zero",
    "rank8_fine_away_half",
    "rank7_fine_away_half",
]

POLICIES: tuple[CorrectionPolicy, ...] = (
    "zero",
    "rank8_fine_away_half",
    "rank7_fine_away_half",
)
NONZERO_POLICIES: tuple[CorrectionPolicy, ...] = POLICIES[1:]
FINE_AWAY_GAIN = 0.5
MAX_CORRECTION_TO_NATIVE_INCREMENT = 0.10
DENOMINATOR_FLOOR = 1.0e-8


@dataclass(frozen=True)
class FineDiscrepancyAudit:
    """Target-free audit for one applied correction."""

    policy: CorrectionPolicy
    status: str
    raw_gain: float
    applied_scale: float
    native_increment_rms: float
    raw_correction_rms: float
    correction_rms: float
    correction_to_native_increment: float | None
    cap_active: bool
    constant_mode_energy_fraction: float | None
    maximum_scaled_component_mean_abs: float
    maximum_excluded_abs: float
    maximum_modal_reconstruction_abs: float


@dataclass(frozen=True)
class FineDiscrepancyStep:
    """One two-call synchronized recurrent update."""

    prepared_inputs: PreparedCommonNativeInputs
    predictions: Mapping[Resolution, np.ndarray]
    basis: NativeIncrementBasis
    correction: np.ndarray
    audit: FineDiscrepancyAudit
    next_native_state: np.ndarray


def _component_scale(
    value: Sequence[float] | np.ndarray,
    components: int,
) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float64)
    if scale.shape != (components,) or not np.isfinite(scale).all():
        raise ValueError("component_scale must be finite and match components")
    if np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive")
    return scale


def _volumes(value: np.ndarray, nodes: int) -> np.ndarray:
    volumes = np.asarray(value, dtype=np.float64).reshape(-1)
    if volumes.shape != (nodes,) or not np.isfinite(volumes).all():
        raise ValueError("volumes must be finite and match nodes")
    if np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive")
    return volumes


def _field(value: np.ndarray, *, nodes: int, components: int, name: str) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    if field.shape != (nodes, components) or not np.isfinite(field).all():
        raise ValueError(f"{name} must be finite with shape {(nodes, components)}")
    return field


def modal_coordinates(
    field: np.ndarray,
    projector: FixedCosineProjector,
    *,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return weighted-QR coordinates of a component-scaled native field."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != projector.basis.shape[0]:
        raise ValueError("field must align with the projector node axis")
    if not np.isfinite(values).all():
        raise ValueError("field must be finite")
    scale = _component_scale(component_scale, values.shape[1])
    weighted = projector.square_root_mass[:, None] * (
        values[projector.interior_mask] / scale[None, :]
    )
    return np.asarray(projector.q_matrix.T @ weighted, dtype=np.float64)


def reconstruct_modal_field(
    coordinates: np.ndarray,
    projector: FixedCosineProjector,
    *,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Reconstruct one native field from weighted-QR modal coordinates."""

    modal = np.asarray(coordinates, dtype=np.float64)
    rank = projector.q_matrix.shape[1]
    if modal.ndim != 2 or modal.shape[0] != rank or not np.isfinite(modal).all():
        raise ValueError("modal coordinates must be finite with the projector rank")
    scale = _component_scale(component_scale, modal.shape[1])
    weighted = projector.q_matrix @ modal
    interior = weighted / projector.square_root_mass[:, None]
    field = np.zeros((projector.basis.shape[0], modal.shape[1]), dtype=np.float64)
    field[projector.interior_mask] = interior * scale[None, :]
    return field


def _scaled_component_means(
    field: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> np.ndarray:
    mass = float(volumes.sum())
    if mass <= 0.0:
        raise ValueError("physical-volume mass must be positive")
    return (
        np.einsum(
            "n,nc->c",
            volumes,
            field / component_scale[None, :],
            optimize=True,
        )
        / mass
    )


def fine_discrepancy_correction(
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    policy: CorrectionPolicy,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    maximum_relative_norm: float = MAX_CORRECTION_TO_NATIVE_INCREMENT,
) -> tuple[np.ndarray, FineDiscrepancyAudit]:
    """Apply a fixed fine-away correction with a target-free norm cap."""

    if policy not in POLICIES:
        raise ValueError(f"unknown fine-discrepancy policy: {policy!r}")
    native = np.asarray(basis.native_increment, dtype=np.float64)
    fine_feature = np.asarray(basis.fine_minus_native, dtype=np.float64)
    if native.ndim != 2 or native.shape != fine_feature.shape:
        raise ValueError("native increment and fine discrepancy must align")
    if not np.isfinite(native).all() or not np.isfinite(fine_feature).all():
        raise ValueError("native increment and fine discrepancy must be finite")
    nodes, components = native.shape
    mass = _volumes(volumes, nodes)
    scale = _component_scale(component_scale, components)
    if not np.isfinite(maximum_relative_norm) or maximum_relative_norm <= 0.0:
        raise ValueError("maximum_relative_norm must be finite and positive")

    zero = np.zeros_like(native)
    native_rms = weighted_scaled_rms(
        native,
        volumes=mass,
        component_scale=scale,
    )
    if policy == "zero":
        return zero, FineDiscrepancyAudit(
            policy=policy,
            status="zero",
            raw_gain=0.0,
            applied_scale=0.0,
            native_increment_rms=native_rms,
            raw_correction_rms=0.0,
            correction_rms=0.0,
            correction_to_native_increment=0.0
            if native_rms > DENOMINATOR_FLOOR
            else None,
            cap_active=False,
            constant_mode_energy_fraction=0.0,
            maximum_scaled_component_mean_abs=0.0,
            maximum_excluded_abs=0.0,
            maximum_modal_reconstruction_abs=0.0,
        )

    coordinates = modal_coordinates(
        fine_feature,
        projector,
        component_scale=scale,
    )
    total_energy = float(np.sum(np.square(coordinates)))
    constant_energy = float(np.sum(np.square(coordinates[0])))
    constant_fraction = (
        constant_energy / total_energy if total_energy > DENOMINATOR_FLOOR**2 else None
    )
    active_coordinates = np.array(coordinates, copy=True)
    if policy == "rank7_fine_away_half":
        active_coordinates[0] = 0.0
    raw_correction = reconstruct_modal_field(
        -FINE_AWAY_GAIN * active_coordinates,
        projector,
        component_scale=scale,
    )
    reconstructed = reconstruct_modal_field(
        modal_coordinates(
            raw_correction,
            projector,
            component_scale=scale,
        ),
        projector,
        component_scale=scale,
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - raw_correction)))
    raw_rms = weighted_scaled_rms(
        raw_correction,
        volumes=mass,
        component_scale=scale,
    )

    status = "ok"
    applied_scale = 1.0
    if raw_rms <= DENOMINATOR_FLOOR:
        status = "unresolved_zero_discrepancy"
        applied_scale = 0.0
    elif native_rms <= DENOMINATOR_FLOOR:
        status = "unresolved_zero_native_increment"
        applied_scale = 0.0
    else:
        applied_scale = min(
            1.0,
            float(maximum_relative_norm) * native_rms / raw_rms,
        )
    correction = applied_scale * raw_correction
    correction_rms = weighted_scaled_rms(
        correction,
        volumes=mass,
        component_scale=scale,
    )
    ratio = correction_rms / native_rms if native_rms > DENOMINATOR_FLOOR else None
    means = _scaled_component_means(
        correction,
        volumes=mass,
        component_scale=scale,
    )
    excluded = correction[~projector.interior_mask]
    excluded_max = 0.0 if excluded.size == 0 else float(np.max(np.abs(excluded)))
    return correction, FineDiscrepancyAudit(
        policy=policy,
        status=status,
        raw_gain=FINE_AWAY_GAIN,
        applied_scale=float(applied_scale),
        native_increment_rms=native_rms,
        raw_correction_rms=raw_rms,
        correction_rms=correction_rms,
        correction_to_native_increment=ratio,
        cap_active=bool(applied_scale < 1.0),
        constant_mode_energy_fraction=constant_fraction,
        maximum_scaled_component_mean_abs=float(np.max(np.abs(means))),
        maximum_excluded_abs=excluded_max,
        maximum_modal_reconstruction_abs=reconstruction_error,
    )


def candidate_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
    *,
    policy: CorrectionPolicy,
) -> tuple[DiagnosticSnapshot, FineDiscrepancyAudit]:
    """Encode the already-applied correction as one scoring feature."""

    correction, audit = fine_discrepancy_correction(
        snapshot.basis,
        projector,
        policy=policy,
        volumes=snapshot.volumes,
        component_scale=snapshot.component_scale,
    )
    zero = np.zeros_like(correction)
    transformed = DiagnosticSnapshot(
        case_id=snapshot.case_id,
        group_id=snapshot.group_id,
        input_call=snapshot.input_call,
        basis=NativeIncrementBasis(
            native_increment=zero,
            coarse_on_native=zero,
            fine_on_native=zero,
            native_minus_coarse=zero,
            fine_minus_native=correction,
        ),
        target_correction=np.asarray(snapshot.target_correction, dtype=np.float64),
        volumes=np.asarray(snapshot.volumes, dtype=np.float64),
        component_scale=np.asarray(snapshot.component_scale, dtype=np.float64),
        masks=snapshot.masks,
    )
    return transformed, audit


def score_candidate_cell(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    cell: str,
    policy: CorrectionPolicy,
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> dict[str, Any]:
    """Score one fixed policy with the established case-first view semantics."""

    transformed: list[DiagnosticSnapshot] = []
    audit_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        candidate, audit = candidate_snapshot(snapshot, projector, policy=policy)
        transformed.append(candidate)
        audit_rows.append(
            {
                "case_id": snapshot.case_id,
                "input_call": int(snapshot.input_call),
                **asdict(audit),
            }
        )
    payload = score_cell(
        transformed,
        cell=cell,
        coefficients=(0.0, 1.0),
        resolution=resolution,
        projector=projector,
    )
    payload["policy"] = policy
    payload["audit_rows"] = audit_rows
    return payload


def score_candidate_inventory(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    label: str,
    expected_case_ids: Sequence[str],
    expected_input_calls: Sequence[int],
    policy: CorrectionPolicy,
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> dict[str, Any]:
    """Score a fixed policy on one explicitly declared case/call inventory.

    This is the generic counterpart of the frozen registered-cell scorer.  It
    deliberately reuses that module's view transformations and case-first
    sufficient-statistic semantics rather than defining a second metric.
    """

    transformed: list[DiagnosticSnapshot] = []
    audit_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        candidate, audit = candidate_snapshot(snapshot, projector, policy=policy)
        transformed.append(candidate)
        audit_rows.append(
            {
                "case_id": snapshot.case_id,
                "input_call": int(snapshot.input_call),
                **asdict(audit),
            }
        )
    ordered = _validate_snapshot_inventory(
        transformed,
        expected_case_ids=expected_case_ids,
        expected_input_calls=expected_input_calls,
    )
    rows: list[dict[str, Any]] = []
    maximum_closure = {
        "band": 0.0,
        "subspace_reconstruction": 0.0,
        "subspace_orthogonality": 0.0,
    }
    for view in VIEW_SPECS:
        statistics_rows = []
        for snapshot in ordered:
            statistics, closure = _statistics_for_view(
                snapshot,
                view,
                resolution=resolution,
                projector=projector,
            )
            statistics_rows.append((snapshot.case_id, statistics))
            maximum_closure["band"] = max(
                maximum_closure["band"],
                float(closure["maximum_reconstruction_abs_residual_scaled"]),
                float(closure["maximum_instantaneous_energy_relative_closure"]),
            )
            maximum_closure["subspace_reconstruction"] = max(
                maximum_closure["subspace_reconstruction"],
                float(closure["maximum_subspace_reconstruction_abs"]),
            )
            maximum_closure["subspace_orthogonality"] = max(
                maximum_closure["subspace_orthogonality"],
                abs(float(closure["maximum_parallel_orthogonal_weighted_inner"])),
            )
        population_score = score_scalar_correction(
            case_first_statistics(statistics_rows),
            (0.0, 1.0),
        )
        case_rows = []
        for case_id in sorted(str(value) for value in expected_case_ids):
            case_statistics = case_first_statistics(
                [row for row in statistics_rows if row[0] == case_id]
            )
            case_score = score_scalar_correction(case_statistics, (0.0, 1.0))
            case_rows.append(
                {
                    "cell": label,
                    "view": view.key,
                    "scope": "case",
                    "case_id": case_id,
                    **asdict(case_score),
                }
            )
        resolved_cosines = [
            float(row["cosine"])
            for row in case_rows
            if row["cosine_status"] == "ok" and row["cosine"] is not None
        ]
        resolved_correlations = [
            float(row["correlation"])
            for row in case_rows
            if row["correlation_status"] == "ok" and row["correlation"] is not None
        ]
        all_resolved = len(resolved_cosines) == len(expected_case_ids) and len(
            resolved_correlations
        ) == len(expected_case_ids)
        rows.append(
            {
                "cell": label,
                "view": view.key,
                "scope": "population",
                "case_id": None,
                **asdict(population_score),
                "median_case_cosine": (
                    float(np.median(resolved_cosines)) if all_resolved else None
                ),
                "median_case_correlation": (
                    float(np.median(resolved_correlations)) if all_resolved else None
                ),
                "all_case_scores_resolved": all_resolved,
            }
        )
        rows.extend(case_rows)
    return {
        "cell": label,
        "case_ids": sorted(str(value) for value in expected_case_ids),
        "input_calls": sorted(int(value) for value in expected_input_calls),
        "views": [view.key for view in VIEW_SPECS],
        "rows": rows,
        "maximum_closure": maximum_closure,
        "policy": policy,
        "audit_rows": sorted(
            audit_rows,
            key=lambda row: (str(row["case_id"]), int(row["input_call"])),
        ),
    }


def score_precomputed_correction_inventory(
    snapshots: Sequence[DiagnosticSnapshot],
    corrections: Sequence[np.ndarray],
    *,
    label: str,
    expected_case_ids: Sequence[str],
    expected_input_calls: Sequence[int],
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> dict[str, Any]:
    """Score aligned, already-computed corrections with maintained view metrics."""

    rows = tuple(snapshots)
    values = tuple(corrections)
    if len(rows) != len(values):
        raise ValueError("precomputed corrections must align with snapshots")
    transformed = []
    for snapshot, correction in zip(rows, values, strict=True):
        field = np.asarray(correction, dtype=np.float64)
        target = np.asarray(snapshot.target_correction, dtype=np.float64)
        if field.shape != target.shape or not np.isfinite(field).all():
            raise ValueError("precomputed correction must be finite and target-shaped")
        zero = np.zeros_like(field)
        transformed.append(
            DiagnosticSnapshot(
                case_id=snapshot.case_id,
                group_id=snapshot.group_id,
                input_call=int(snapshot.input_call),
                basis=NativeIncrementBasis(
                    native_increment=zero,
                    coarse_on_native=zero,
                    fine_on_native=zero,
                    native_minus_coarse=zero,
                    fine_minus_native=field,
                ),
                target_correction=target,
                volumes=np.asarray(snapshot.volumes, dtype=np.float64),
                component_scale=np.asarray(
                    snapshot.component_scale, dtype=np.float64
                ),
                masks=snapshot.masks,
            )
        )
    ordered = _validate_snapshot_inventory(
        transformed,
        expected_case_ids=expected_case_ids,
        expected_input_calls=expected_input_calls,
    )
    score_rows: list[dict[str, Any]] = []
    maximum_closure = {
        "band": 0.0,
        "subspace_reconstruction": 0.0,
        "subspace_orthogonality": 0.0,
    }
    for view in VIEW_SPECS:
        statistics_rows = []
        for snapshot in ordered:
            statistics, closure = _statistics_for_view(
                snapshot,
                view,
                resolution=resolution,
                projector=projector,
            )
            statistics_rows.append((snapshot.case_id, statistics))
            maximum_closure["band"] = max(
                maximum_closure["band"],
                float(closure["maximum_reconstruction_abs_residual_scaled"]),
                float(closure["maximum_instantaneous_energy_relative_closure"]),
            )
            maximum_closure["subspace_reconstruction"] = max(
                maximum_closure["subspace_reconstruction"],
                float(closure["maximum_subspace_reconstruction_abs"]),
            )
            maximum_closure["subspace_orthogonality"] = max(
                maximum_closure["subspace_orthogonality"],
                abs(float(closure["maximum_parallel_orthogonal_weighted_inner"])),
            )
        population_score = score_scalar_correction(
            case_first_statistics(statistics_rows), (0.0, 1.0)
        )
        case_rows = []
        for case_id in sorted(str(value) for value in expected_case_ids):
            case_score = score_scalar_correction(
                case_first_statistics(
                    [row for row in statistics_rows if row[0] == case_id]
                ),
                (0.0, 1.0),
            )
            case_rows.append(
                {
                    "cell": label,
                    "view": view.key,
                    "scope": "case",
                    "case_id": case_id,
                    **asdict(case_score),
                }
            )
        resolved_cosines = [
            float(row["cosine"])
            for row in case_rows
            if row["cosine_status"] == "ok" and row["cosine"] is not None
        ]
        resolved_correlations = [
            float(row["correlation"])
            for row in case_rows
            if row["correlation_status"] == "ok"
            and row["correlation"] is not None
        ]
        all_resolved = (
            len(resolved_cosines) == len(expected_case_ids)
            and len(resolved_correlations) == len(expected_case_ids)
        )
        score_rows.append(
            {
                "cell": label,
                "view": view.key,
                "scope": "population",
                "case_id": None,
                **asdict(population_score),
                "median_case_cosine": (
                    float(np.median(resolved_cosines)) if all_resolved else None
                ),
                "median_case_correlation": (
                    float(np.median(resolved_correlations))
                    if all_resolved
                    else None
                ),
                "all_case_scores_resolved": all_resolved,
            }
        )
        score_rows.extend(case_rows)
    return {
        "cell": label,
        "case_ids": sorted(str(value) for value in expected_case_ids),
        "input_calls": sorted(int(value) for value in expected_input_calls),
        "views": [view.key for view in VIEW_SPECS],
        "rows": score_rows,
        "maximum_closure": maximum_closure,
        "policy": "precomputed_target_free_correction",
    }


def _fine_only_basis(
    prepared: PreparedCommonNativeInputs,
    predictions: Mapping[Resolution, np.ndarray],
    *,
    contract: ResolutionContract,
) -> NativeIncrementBasis:
    expected = {contract.native, contract.fine}
    if set(predictions) != expected:
        raise ValueError(
            "fine-discrepancy inference requires native and fine predictions"
        )
    native_current = prepared.model_inputs[contract.native]
    fine_current = prepared.model_inputs[contract.fine]
    native_prediction = _field(
        predictions[contract.native],
        nodes=contract.native[0] * contract.native[1],
        components=4,
        name="native prediction",
    )
    fine_prediction = _field(
        predictions[contract.fine],
        nodes=contract.fine[0] * contract.fine[1],
        components=4,
        name="fine prediction",
    )
    native_increment = native_prediction - native_current
    fine_increment = fine_prediction - fine_current
    fine_on_native = restrict_nested_state(
        fine_increment,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    return NativeIncrementBasis(
        native_increment=native_increment,
        coarse_on_native=np.array(native_increment, copy=True),
        fine_on_native=fine_on_native,
        native_minus_coarse=np.zeros_like(native_increment),
        fine_minus_native=fine_on_native - native_increment,
    )


def synchronized_fine_discrepancy_step(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    policy: CorrectionPolicy,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> FineDiscrepancyStep:
    """Make exactly two synchronized model calls and retain one native state."""

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
    basis = _fine_only_basis(prepared, predictions, contract=contract)
    correction, audit = fine_discrepancy_correction(
        basis,
        projector,
        policy=policy,
        volumes=volumes,
        component_scale=component_scale,
    )
    next_state = np.asarray(predictions[contract.native], dtype=np.float64) + correction
    return FineDiscrepancyStep(
        prepared_inputs=prepared,
        predictions=predictions,
        basis=basis,
        correction=correction,
        audit=audit,
        next_native_state=next_state,
    )


def select_calibration_policy(
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the lower-rank-RMS eligible fixed policy, failing closed."""

    if set(evidence) != set(NONZERO_POLICIES):
        raise ValueError("calibration evidence must contain both nonzero policies")
    eligible: list[tuple[float, int, CorrectionPolicy]] = []
    checks: dict[str, dict[str, bool]] = {}
    for policy in NONZERO_POLICIES:
        row = evidence[policy]
        expected = {
            "inventory_exact",
            "full_skill_nonnegative",
            "rank8_skill_at_least_0p05",
            "minimum_eight_group_wins",
            "all_controls_no_harm",
            "all_proposals_finite_admissible",
            "all_audits_resolved",
            "closure_passed",
            "rank8_rms_ratio",
        }
        if set(row) != expected:
            raise ValueError(f"calibration evidence fields differ for {policy}")
        row_checks = {
            key: bool(row[key]) for key in expected if key != "rank8_rms_ratio"
        }
        ratio = row["rank8_rms_ratio"]
        ratio_ok = (
            ratio is not None and np.isfinite(float(ratio)) and float(ratio) < 1.0
        )
        row_checks["rank8_rms_ratio_resolved"] = bool(ratio_ok)
        checks[policy] = row_checks
        if all(row_checks.values()):
            rank = 7 if policy == "rank7_fine_away_half" else 8
            eligible.append((float(ratio), rank, policy))
    selected: CorrectionPolicy = "zero"
    if eligible:
        selected = min(eligible, key=lambda value: (value[0], value[1], value[2]))[2]
    return {
        "status": "qualified" if selected != "zero" else "not_qualified",
        "selected_policy": selected,
        "checks": checks,
    }


def teacher_forced_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    """Freeze the exact all-case teacher gate before recurrence."""

    expected = {
        "calibration_qualified",
        "inventory_exact",
        "full_skill_nonnegative",
        "rank8_skill_at_least_0p05",
        "minimum_four_case_wins",
        "both_half_horizons_positive",
        "all_controls_no_harm",
        "all_proposals_finite_admissible",
        "all_audits_resolved",
        "closure_passed",
        "source_and_artifacts_exact",
    }
    if set(checks) != expected or any(
        type(value) is not bool for value in checks.values()
    ):
        raise ValueError("teacher-forced gate evidence inventory differs")
    passed = all(checks.values())
    return {
        "status": "recurrent_pilot_authorized" if passed else "stopped",
        "recurrent_pilot_authorized": passed,
        "fresh_confirmation": False,
        "sealed_population_authorized": False,
        "checks": dict(checks),
    }


def recurrent_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    """Apply the registered recurrent efficacy and no-harm conjunction."""

    expected = {
        "teacher_gate_passed",
        "inventory_exact",
        "all_rollouts_complete_finite_admissible",
        "median_endpoint_ratio_at_most_0p98",
        "minimum_four_endpoint_wins",
        "maximum_endpoint_ratio_at_most_1p02",
        "aggregate_state_rms_ratio_at_most_0p99",
        "increment_and_cumulative_ratios_at_most_one",
        "all_controls_no_harm",
        "two_call_common_source_closure",
        "correction_audits_pass",
        "deterministic_prefix_exact",
        "source_and_artifacts_exact",
    }
    if set(checks) != expected or any(
        type(value) is not bool for value in checks.values()
    ):
        raise ValueError("recurrent gate evidence inventory differs")
    passed = all(checks.values())
    return {
        "status": "adaptive_recurrent_pass" if passed else "adaptive_recurrent_failed",
        "adaptive_recurrent_pass": passed,
        "fresh_confirmation": False,
        "sealed_population_authorized": False,
        "checks": dict(checks),
    }
