"""Typed A2 scoring and gates for W26-L5 teacher-forced correction.

This module is model independent.  The experiment runner owns checkpoint and
reference loading; this file owns the immutable open-population inventory,
case-first scoring, fixed physical-cosine diagnostics, and fail-closed gates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DENOMINATOR_FLOOR,
    MODEL_NAMES,
    DiagnosticSnapshot,
    NativeIncrementBasis,
    case_first_statistics,
    fit_scalar_correction,
    score_scalar_correction,
    snapshot_statistics,
)
from utility.time_dependent_no.pcno_defect_corrections import physical_cosine_basis

CALIBRATION_STRENGTHS = (1, 2, 3, 4, 5, 7, 8, 9, 10)
EVALUATION_STRENGTHS = (0, 6, 11)
POSITION_SUFFIXES = ("y00", "y08")
CALIBRATION_CASE_IDS = tuple(
    f"sv_e{strength:02d}_{suffix}"
    for strength in CALIBRATION_STRENGTHS
    for suffix in POSITION_SUFFIXES
)
EVALUATION_CASE_IDS = tuple(
    f"sv_e{strength:02d}_{suffix}"
    for strength in EVALUATION_STRENGTHS
    for suffix in POSITION_SUFFIXES
)
CALIBRATION_GROUPS = {
    f"e{strength:02d}": tuple(
        f"sv_e{strength:02d}_{suffix}" for suffix in POSITION_SUFFIXES
    )
    for strength in CALIBRATION_STRENGTHS
}
FIT_INPUT_CALLS = tuple(range(20))
HELD_OUT_INPUT_CALLS = tuple(range(20, 30))
ALL_INPUT_CALLS = tuple(range(30))

CELL_CASES = {
    "calibration_fit": CALIBRATION_CASE_IDS,
    "time_only": CALIBRATION_CASE_IDS,
    "case_only": EVALUATION_CASE_IDS,
    "joint_held_out": EVALUATION_CASE_IDS,
    "joint_late_1": EVALUATION_CASE_IDS,
    "joint_late_2": EVALUATION_CASE_IDS,
}
CELL_CALLS = {
    "calibration_fit": FIT_INPUT_CALLS,
    "time_only": HELD_OUT_INPUT_CALLS,
    "case_only": FIT_INPUT_CALLS,
    "joint_held_out": HELD_OUT_INPUT_CALLS,
    "joint_late_1": tuple(range(20, 25)),
    "joint_late_2": tuple(range(25, 30)),
}

ComponentTuple = tuple[int, ...]
SubspaceName = Literal["parallel", "orthogonal", "excluded"]


@dataclass(frozen=True)
class ViewSpec:
    key: str
    band: str | None = None
    region: str | None = None
    components: ComponentTuple = (0, 1, 2, 3)
    subspace: SubspaceName | None = None


VIEW_SPECS = (
    ViewSpec("full"),
    ViewSpec("band_large", band="large"),
    ViewSpec("band_transition", band="transition"),
    ViewSpec("band_local", band="local"),
    ViewSpec("region_boundary", region="boundary_le_0.05"),
    ViewSpec("region_shock", region="partition_shock"),
    ViewSpec("region_vortex", region="partition_vortex"),
    ViewSpec("region_smooth", region="partition_smooth"),
    ViewSpec("smooth_local", band="local", region="partition_smooth"),
    ViewSpec("component_density", components=(0,)),
    ViewSpec("component_x_momentum", components=(1,)),
    ViewSpec("component_y_momentum", components=(2,)),
    ViewSpec("component_energy", components=(3,)),
    ViewSpec("rank8_parallel", subspace="parallel"),
    ViewSpec("rank8_orthogonal", subspace="orthogonal"),
    ViewSpec("rank8_excluded", subspace="excluded"),
)
VIEW_BY_KEY = {view.key: view for view in VIEW_SPECS}
REQUIRED_FIELD_CONTROL_VIEWS = (
    "full",
    "band_transition",
    "band_local",
    "region_boundary",
    "region_shock",
    "region_vortex",
    "smooth_local",
    "component_density",
    "component_x_momentum",
    "component_y_momentum",
    "component_energy",
)
RELATION_PAIRS = (
    ("coarse_on_native", "native"),
    ("fine_on_native", "native"),
    ("coarse_on_native", "fine_on_native"),
)
FRONT_CONTROL_KEYS = (
    "front_position",
    "front_strength_log_ratio",
    "front_thickness_log_ratio",
)
INTEGRAL_COMPONENT_NAMES = ("density", "x_momentum", "y_momentum", "energy")


@dataclass(frozen=True)
class CalibrationClosureEvidence:
    checkpoint_contract: bool
    reference_contract: bool
    common_source_inventory: bool
    prediction_inventory: bool
    exact_source_identity: bool
    maximum_pre_model_nesting_floor: float
    maximum_post_fp32_nesting_floor: float
    maximum_sign_closure: float
    maximum_increment_integral_closure: float
    maximum_transfer_floor_closure: float
    maximum_band_closure: float
    maximum_region_partition_error: int
    maximum_repeat_abs_difference: float


@dataclass(frozen=True)
class CellScoreEvidence:
    cell: str
    view: str
    case_ids: tuple[str, ...]
    input_calls: tuple[int, ...]
    skill_vs_zero: float | None
    rms_ratio_vs_zero: float | None
    centered_r2: float | None
    median_case_cosine: float | None
    median_case_correlation: float | None
    score_status: str
    all_case_scores_resolved: bool


@dataclass(frozen=True)
class ControlRatio:
    key: str
    scope: str
    zero_rms: float
    corrected_rms: float
    ratio: float | None
    status: str


@dataclass(frozen=True)
class ProposalEvidence:
    case_id: str
    input_call: int
    finite: bool
    admissible: bool


@dataclass(frozen=True)
class FixedCosineProjector:
    """Cached inherited weighted-QR projection on dynamic type-0 cells."""

    basis: np.ndarray
    modes: tuple[tuple[int, int], ...]
    interior_mask: np.ndarray
    square_root_mass: np.ndarray
    q_matrix: np.ndarray

    def split(self, field: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        values = np.asarray(field, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.basis.shape[0]:
            raise ValueError("projected field must have shape [native_nodes, components]")
        if not np.isfinite(values).all():
            raise ValueError("projected field must be finite")
        interior_values = values[self.interior_mask]
        weighted_values = self.square_root_mass[:, None] * interior_values
        weighted_projection = self.q_matrix @ (self.q_matrix.T @ weighted_values)
        projection = weighted_projection / self.square_root_mass[:, None]
        parallel = np.zeros_like(values)
        orthogonal = np.zeros_like(values)
        excluded = np.zeros_like(values)
        parallel[self.interior_mask] = projection
        orthogonal[self.interior_mask] = interior_values - projection
        excluded[~self.interior_mask] = values[~self.interior_mask]
        reconstruction = parallel + orthogonal + excluded
        inner = float(
            np.sum(
                weighted_projection
                * (weighted_values - weighted_projection)
            )
        )
        return (
            {
                "parallel": parallel,
                "orthogonal": orthogonal,
                "excluded": excluded,
            },
            {
                "maximum_reconstruction_abs": float(
                    np.max(np.abs(reconstruction - values))
                ),
                "parallel_orthogonal_weighted_inner": inner,
            },
        )


def build_fixed_cosine_projector(
    nodes: np.ndarray,
    volumes: np.ndarray,
    node_type: np.ndarray,
    *,
    rank: int = 8,
    domain_bounds: Sequence[float] = (0.0, 2.0, 0.0, 1.0),
) -> FixedCosineProjector:
    positions = np.asarray(nodes, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    codes = np.asarray(node_type)
    if mass.shape != (positions.shape[0],) or not np.isfinite(mass).all():
        raise ValueError("volumes must be finite and align with nodes")
    if np.any(mass <= 0.0):
        raise ValueError("volumes must be positive")
    if codes.shape != mass.shape:
        raise ValueError("node_type must align with nodes")
    interior = codes == 0
    basis, modes = physical_cosine_basis(
        positions,
        rank=rank,
        domain_bounds=domain_bounds,
    )
    if int(interior.sum()) < rank:
        raise ValueError("type-0 support is smaller than the fixed cosine rank")
    root_mass = np.sqrt(mass[interior])
    weighted_design = root_mass[:, None] * basis[interior]
    q_matrix, r_matrix = np.linalg.qr(weighted_design, mode="reduced")
    singular_values = np.linalg.svd(r_matrix, compute_uv=False)
    tolerance = (
        max(weighted_design.shape)
        * np.finfo(np.float64).eps
        * max(float(singular_values[0]), 1.0)
    )
    if int(np.count_nonzero(singular_values > tolerance)) != rank:
        raise ValueError("fixed physical-cosine basis is rank deficient")
    return FixedCosineProjector(
        basis=np.asarray(basis, dtype=np.float64),
        modes=modes,
        interior_mask=np.asarray(interior, dtype=np.bool_),
        square_root_mass=root_mass,
        q_matrix=np.asarray(q_matrix, dtype=np.float64),
    )


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    value = {key: item for key, item in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def with_payload_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256_scope"] = (
        "canonical sorted compact JSON excluding only payload_sha256"
    )
    result["payload_sha256"] = canonical_payload_sha256(result)
    return result


def verify_payload_sha256(payload: Mapping[str, Any]) -> None:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("payload lacks a valid SHA-256 field")
    actual = canonical_payload_sha256(payload)
    if actual != expected:
        raise ValueError("payload SHA-256 mismatch")


def _active_indices(model: str) -> tuple[int, ...]:
    return {
        "zero": (),
        "coarse_only": (0,),
        "fine_only": (1,),
        "two_term": (0, 1),
    }[model]


def qualify_calibration(
    crossfit: Mapping[str, Any],
    closure: CalibrationClosureEvidence,
) -> dict[str, Any]:
    """Apply the pre-evaluation calibration qualification exactly and fail closed."""

    selected = crossfit.get("selected_model")
    if selected not in MODEL_NAMES:
        selected = None
    expected_cases = sorted(CALIBRATION_CASE_IDS)
    expected_groups = sorted(CALIBRATION_GROUPS)
    inventory_checks = {
        "case_inventory_exact": crossfit.get("case_ids") == expected_cases,
        "group_inventory_exact": crossfit.get("groups") == expected_groups,
        "group_membership_exact": crossfit.get("expected_groups")
        == {key: sorted(CALIBRATION_GROUPS[key]) for key in expected_groups},
        "call_inventory_exact": crossfit.get("expected_input_calls")
        == list(FIT_INPUT_CALLS),
        "model_inventory_exact": set(crossfit.get("models", {})) == set(MODEL_NAMES),
        "selection_band_exact": crossfit.get("selection_band") == "large",
    }
    coefficients_value = crossfit.get("selected_coefficients")
    coefficients = None
    if (
        isinstance(coefficients_value, (list, tuple))
        and len(coefficients_value) == 2
        and all(np.isfinite(float(value)) for value in coefficients_value)
    ):
        coefficients = tuple(float(value) for value in coefficients_value)
    active = () if selected is None else _active_indices(selected)
    selected_rows = (
        crossfit.get("models", {}).get(selected, {}) if selected is not None else {}
    )
    folds = selected_rows.get("folds", [])
    fold_groups = [fold.get("held_out_group") for fold in folds]
    fold_inventory_exact = (
        len(folds) == len(expected_groups)
        and sorted(fold_groups) == expected_groups
        and len(set(fold_groups)) == len(expected_groups)
    )

    stability_rows: list[dict[str, Any]] = []
    stability_pass = bool(active) and coefficients is not None and fold_inventory_exact
    for index in active:
        fold_values: list[float] = []
        if fold_inventory_exact:
            for fold in folds:
                value = fold.get("coefficients")
                if (
                    fold.get("fit_status") != "ok"
                    or not isinstance(value, (list, tuple))
                    or len(value) != 2
                    or not np.isfinite(float(value[index]))
                ):
                    fold_values = []
                    break
                fold_values.append(float(value[index]))
        full_value = None if coefficients is None else coefficients[index]
        same_sign = 0
        median = None
        iqr = None
        relative_iqr = None
        status = "unresolved"
        if fold_values and full_value is not None and full_value != 0.0:
            array = np.asarray(fold_values, dtype=np.float64)
            same_sign = int(np.count_nonzero(np.sign(array) == np.sign(full_value)))
            median = float(np.median(array))
            q25, q75 = np.percentile(array, (25.0, 75.0))
            iqr = float(q75 - q25)
            if median != 0.0 and np.isfinite(median):
                relative_iqr = float(iqr / abs(median))
                if same_sign >= 8 and relative_iqr <= 0.5:
                    status = "ok"
        stability_pass = stability_pass and status == "ok"
        stability_rows.append(
            {
                "coefficient_index": index,
                "full_fit_value": full_value,
                "fold_values": fold_values,
                "same_sign_fold_count": same_sign,
                "fold_median": median,
                "fold_iqr": iqr,
                "fold_iqr_over_abs_median": relative_iqr,
                "status": status,
            }
        )

    all_oof_resolved = False
    selected_oof_improves = False
    if selected_rows:
        case_scores = selected_rows.get("out_of_fold_case_scores", [])
        all_oof_resolved = (
            len(case_scores) == len(CALIBRATION_CASE_IDS)
            and {row.get("case_id") for row in case_scores}
            == set(CALIBRATION_CASE_IDS)
            and all(
                row.get(name) == "ok"
                for row in case_scores
                for name in (
                    "skill_status",
                    "centered_r2_status",
                    "cosine_status",
                    "correlation_status",
                )
            )
        )
        rms_ratio = selected_rows.get("rms_ratio_vs_zero")
        selected_oof_improves = (
            rms_ratio is not None
            and np.isfinite(float(rms_ratio))
            and float(rms_ratio) < 1.0
        )

    full_fit = crossfit.get("selected_full_fit", {})
    condition = crossfit.get("selected_condition_number")
    full_fit_ok = (
        crossfit.get("selected_fit_status") == "ok"
        and full_fit.get("rank") == len(active)
        and condition is not None
        and np.isfinite(float(condition))
        and float(condition) <= 1.0e6
    )
    closure_checks = {
        "checkpoint_contract": closure.checkpoint_contract,
        "reference_contract": closure.reference_contract,
        "common_source_inventory": closure.common_source_inventory,
        "prediction_inventory": closure.prediction_inventory,
        "exact_source_identity": closure.exact_source_identity,
        "pre_model_nesting_exact": closure.maximum_pre_model_nesting_floor == 0.0,
        "post_fp32_nesting_finite": np.isfinite(
            closure.maximum_post_fp32_nesting_floor
        ),
        "mapped_sign_closure": closure.maximum_sign_closure <= 1.0e-12,
        "increment_integral_closure": (
            closure.maximum_increment_integral_closure <= 1.0e-12
        ),
        "transfer_floor_closure": closure.maximum_transfer_floor_closure <= 1.0e-12,
        "band_closure": closure.maximum_band_closure <= 1.0e-10,
        "region_partition_exact": closure.maximum_region_partition_error == 0,
        "repeatability_resolved": np.isfinite(closure.maximum_repeat_abs_difference),
    }
    checks = {
        **inventory_checks,
        "selected_nonzero_model": selected in MODEL_NAMES[1:],
        "selected_coefficients_finite": coefficients is not None,
        "fold_inventory_exact": fold_inventory_exact,
        "all_oof_metrics_resolved": all_oof_resolved,
        "selected_oof_improves_zero": selected_oof_improves,
        "active_gram_full_rank_well_conditioned": full_fit_ok,
        "coefficient_sign_scale_stable": stability_pass,
        **closure_checks,
    }
    return {
        "status": "qualified" if all(checks.values()) else "not_qualified",
        "checks": {key: bool(value) for key, value in checks.items()},
        "selected_model": selected,
        "selected_coefficients": coefficients,
        "coefficient_stability": stability_rows,
        "calibration_closure": asdict(closure),
    }


def require_qualified_calibration(
    path: Path,
    *,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload_sha256(payload)
    if payload.get("schema") != "pcno_cross_resolution_teacher_forced_calibration_v1":
        raise ValueError("unsupported calibration artifact schema")
    if payload.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("calibration source-manifest identity mismatch")
    qualification = payload.get("qualification")
    if not isinstance(qualification, Mapping):
        raise TypeError("calibration artifact lacks typed qualification evidence")
    checks = qualification.get("checks")
    if (
        qualification.get("status") != "qualified"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("calibration did not qualify; evaluation targets stay closed")
    coefficients = qualification.get("selected_coefficients")
    if (
        qualification.get("selected_model") not in MODEL_NAMES[1:]
        or not isinstance(coefficients, list)
        or len(coefficients) != 2
        or not all(np.isfinite(float(value)) for value in coefficients)
    ):
        raise ValueError("qualified calibration has invalid frozen coefficients")
    return payload


def _validate_snapshot_inventory(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    expected_case_ids: Sequence[str],
    expected_input_calls: Sequence[int],
) -> tuple[DiagnosticSnapshot, ...]:
    expected_cases = tuple(sorted(str(value) for value in expected_case_ids))
    expected_calls = tuple(sorted(int(value) for value in expected_input_calls))
    if len(set(expected_cases)) != len(expected_cases) or not expected_cases:
        raise ValueError("expected case inventory must be nonempty and unique")
    if len(set(expected_calls)) != len(expected_calls) or not expected_calls:
        raise ValueError("expected call inventory must be nonempty and unique")
    expected = {(case_id, call) for case_id in expected_cases for call in expected_calls}
    actual = [(snapshot.case_id, int(snapshot.input_call)) for snapshot in snapshots]
    if len(actual) != len(set(actual)):
        raise ValueError("snapshot inventory contains duplicate case/call pairs")
    if set(actual) != expected:
        raise ValueError("snapshot inventory does not match the declared cell")
    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    reference_volumes = np.asarray(ordered[0].volumes)
    reference_scales = np.asarray(ordered[0].component_scale)
    if any(
        not np.array_equal(np.asarray(row.volumes), reference_volumes)
        or not np.array_equal(np.asarray(row.component_scale), reference_scales)
        for row in ordered[1:]
    ):
        raise ValueError("all snapshots must share one frozen metric contract")
    return ordered


def _subspace_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
    subspace: SubspaceName,
) -> tuple[DiagnosticSnapshot, dict[str, float]]:
    fields = (
        snapshot.basis.native_minus_coarse,
        snapshot.basis.fine_minus_native,
        snapshot.target_correction,
    )
    selected: list[np.ndarray] = []
    maximum_reconstruction = 0.0
    maximum_orthogonality = 0.0
    for field in fields:
        split, closure = projector.split(field)
        selected.append(split[subspace])
        maximum_reconstruction = max(
            maximum_reconstruction,
            closure["maximum_reconstruction_abs"],
        )
        maximum_orthogonality = max(
            maximum_orthogonality,
            abs(closure["parallel_orthogonal_weighted_inner"]),
        )
    zero = np.zeros_like(selected[0])
    return (
        DiagnosticSnapshot(
            case_id=snapshot.case_id,
            group_id=snapshot.group_id,
            input_call=snapshot.input_call,
            basis=NativeIncrementBasis(
                native_increment=zero,
                coarse_on_native=zero,
                fine_on_native=zero,
                native_minus_coarse=selected[0],
                fine_minus_native=selected[1],
            ),
            target_correction=selected[2],
            volumes=snapshot.volumes,
            component_scale=snapshot.component_scale,
            masks=snapshot.masks,
        ),
        {
            "maximum_subspace_reconstruction_abs": maximum_reconstruction,
            "maximum_parallel_orthogonal_weighted_inner": maximum_orthogonality,
        },
    )


def _statistics_for_view(
    snapshot: DiagnosticSnapshot,
    view: ViewSpec,
    *,
    resolution: tuple[int, int],
    projector: FixedCosineProjector,
):
    active = snapshot
    subspace_closure = {
        "maximum_subspace_reconstruction_abs": 0.0,
        "maximum_parallel_orthogonal_weighted_inner": 0.0,
    }
    if view.subspace is not None:
        active, subspace_closure = _subspace_snapshot(snapshot, projector, view.subspace)
    statistics, band_closure = snapshot_statistics(
        active,
        resolution=resolution,
        band=view.band,
        region=view.region,
        components=view.components,
    )
    return statistics, {**band_closure, **subspace_closure}


def score_cell(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    cell: str,
    coefficients: tuple[float, float],
    resolution: tuple[int, int],
    projector: FixedCosineProjector,
) -> dict[str, Any]:
    if cell not in CELL_CASES or cell not in CELL_CALLS:
        raise ValueError(f"unknown registered cell: {cell}")
    ordered = _validate_snapshot_inventory(
        snapshots,
        expected_case_ids=CELL_CASES[cell],
        expected_input_calls=CELL_CALLS[cell],
    )
    rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
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
                float(closure["maximum_parallel_orthogonal_weighted_inner"]),
            )
        population_statistics = case_first_statistics(statistics_rows)
        population_score = score_scalar_correction(population_statistics, coefficients)
        case_score_rows = []
        for case_id in sorted(CELL_CASES[cell]):
            case_statistics = case_first_statistics(
                [(owner, value) for owner, value in statistics_rows if owner == case_id]
            )
            case_score = score_scalar_correction(case_statistics, coefficients)
            case_score_rows.append(
                {
                    "cell": cell,
                    "view": view.key,
                    "scope": "case",
                    "case_id": case_id,
                    **asdict(case_score),
                }
            )
            if view.key in {"full", "band_large"}:
                oracle_fit = fit_scalar_correction(case_statistics, model="two_term")
                oracle_score = (
                    score_scalar_correction(case_statistics, oracle_fit.coefficients)
                    if oracle_fit.status == "ok" and oracle_fit.coefficients is not None
                    else None
                )
                oracle_rows.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "case_id": case_id,
                        "deployable": False,
                        "fit": asdict(oracle_fit),
                        "score": None if oracle_score is None else asdict(oracle_score),
                    }
                )
        resolved_cosines = [
            float(row["cosine"])
            for row in case_score_rows
            if row["cosine_status"] == "ok" and row["cosine"] is not None
        ]
        resolved_correlations = [
            float(row["correlation"])
            for row in case_score_rows
            if row["correlation_status"] == "ok" and row["correlation"] is not None
        ]
        all_resolved = (
            len(resolved_cosines) == len(CELL_CASES[cell])
            and len(resolved_correlations) == len(CELL_CASES[cell])
        )
        population_row = {
            "cell": cell,
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
        rows.append(population_row)
        rows.extend(case_score_rows)
    return {
        "cell": cell,
        "case_ids": sorted(CELL_CASES[cell]),
        "input_calls": list(CELL_CALLS[cell]),
        "views": [view.key for view in VIEW_SPECS],
        "rows": rows,
        "oracle_rows": oracle_rows,
        "maximum_closure": maximum_closure,
    }


def _relation_snapshot(
    snapshot: DiagnosticSnapshot,
    left: str,
    right: str,
) -> DiagnosticSnapshot:
    target = np.asarray(snapshot.target_correction, dtype=np.float64)
    epsilon = {
        "native": -target,
        "coarse_on_native": -target - snapshot.basis.native_minus_coarse,
        "fine_on_native": snapshot.basis.fine_minus_native - target,
    }
    zero = np.zeros_like(target)
    return DiagnosticSnapshot(
        case_id=snapshot.case_id,
        group_id=snapshot.group_id,
        input_call=snapshot.input_call,
        basis=NativeIncrementBasis(
            native_increment=zero,
            coarse_on_native=zero,
            fine_on_native=zero,
            native_minus_coarse=epsilon[left],
            fine_minus_native=zero,
        ),
        target_correction=epsilon[right],
        volumes=snapshot.volumes,
        component_scale=snapshot.component_scale,
        masks=snapshot.masks,
    )


def error_relation_rows(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    cell: str,
    resolution: tuple[int, int],
    projector: FixedCosineProjector,
) -> list[dict[str, Any]]:
    ordered = _validate_snapshot_inventory(
        snapshots,
        expected_case_ids=CELL_CASES[cell],
        expected_input_calls=CELL_CALLS[cell],
    )
    output: list[dict[str, Any]] = []
    for view in VIEW_SPECS:
        for left, right in RELATION_PAIRS:
            statistics_rows = []
            for snapshot in ordered:
                relation = _relation_snapshot(snapshot, left, right)
                statistics, _ = _statistics_for_view(
                    relation,
                    view,
                    resolution=resolution,
                    projector=projector,
                )
                statistics_rows.append((snapshot.case_id, statistics))
            scopes = [("population", None, statistics_rows)]
            scopes.extend(
                (
                    "case",
                    case_id,
                    [(owner, value) for owner, value in statistics_rows if owner == case_id],
                )
                for case_id in sorted(CELL_CASES[cell])
            )
            for scope, case_id, selected_rows in scopes:
                statistics = case_first_statistics(selected_rows)
                fit = fit_scalar_correction(statistics, model="coarse_only")
                unit_score = score_scalar_correction(statistics, (1.0, 0.0))
                fitted_score = (
                    score_scalar_correction(statistics, fit.coefficients)
                    if fit.status == "ok" and fit.coefficients is not None
                    else None
                )
                output.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "feature_error": left,
                        "target_error": right,
                        "scope": scope,
                        "case_id": case_id,
                        "signed_cosine": unit_score.cosine,
                        "signed_cosine_denominator": unit_score.cosine_denominator,
                        "signed_cosine_status": unit_score.cosine_status,
                        "pearson": unit_score.correlation,
                        "pearson_denominator": unit_score.correlation_denominator,
                        "pearson_status": unit_score.correlation_status,
                        "fitted_slope": (
                            None if fit.coefficients is None else fit.coefficients[0]
                        ),
                        "fit_status": fit.status,
                        "fitted_skill_vs_zero": (
                            None if fitted_score is None else fitted_score.skill_vs_zero
                        ),
                        "fitted_skill_status": (
                            None if fitted_score is None else fitted_score.skill_status
                        ),
                        "offline_only": True,
                    }
                )
    return output


def score_evidence_from_cell(payload: Mapping[str, Any], *, view: str) -> CellScoreEvidence:
    matches = [
        row
        for row in payload.get("rows", [])
        if row.get("scope") == "population" and row.get("view") == view
    ]
    if len(matches) != 1:
        raise ValueError("cell payload lacks one exact population/view row")
    row = matches[0]
    return CellScoreEvidence(
        cell=str(payload.get("cell")),
        view=view,
        case_ids=tuple(str(value) for value in payload.get("case_ids", [])),
        input_calls=tuple(int(value) for value in payload.get("input_calls", [])),
        skill_vs_zero=row.get("skill_vs_zero"),
        rms_ratio_vs_zero=row.get("rms_ratio_vs_zero"),
        centered_r2=row.get("centered_r2"),
        median_case_cosine=row.get("median_case_cosine"),
        median_case_correlation=row.get("median_case_correlation"),
        score_status=str(row.get("status")),
        all_case_scores_resolved=bool(row.get("all_case_scores_resolved")),
    )


def prospective_gate(
    *,
    cell_payloads: Mapping[str, Mapping[str, Any]],
    controls: Sequence[ControlRatio],
    proposals: Sequence[ProposalEvidence],
) -> dict[str, Any]:
    """Apply the immutable P1 gate with exact inventories and no missing controls."""

    expected_cells = {
        "time_only",
        "case_only",
        "joint_held_out",
        "joint_late_1",
        "joint_late_2",
    }
    cell_inventory_exact = set(cell_payloads) == expected_cells
    evidence: dict[str, CellScoreEvidence] = {}
    if cell_inventory_exact:
        evidence = {
            cell: score_evidence_from_cell(cell_payloads[cell], view="band_large")
            for cell in sorted(expected_cells)
        }
    cell_contract_exact = cell_inventory_exact and all(
        item.case_ids == tuple(sorted(CELL_CASES[cell]))
        and item.input_calls == CELL_CALLS[cell]
        for cell, item in evidence.items()
    )

    primary_rows = cell_payloads.get("joint_held_out", {}).get("rows", [])
    primary_case_large = [
        row
        for row in primary_rows
        if row.get("scope") == "case" and row.get("view") == "band_large"
    ]
    primary_case_inventory_exact = (
        len(primary_case_large) == len(EVALUATION_CASE_IDS)
        and {row.get("case_id") for row in primary_case_large}
        == set(EVALUATION_CASE_IDS)
    )
    passing_cases = sum(
        row.get("cosine_status") == "ok"
        and row.get("skill_status") == "ok"
        and row.get("cosine") is not None
        and float(row["cosine"]) > 0.0
        and row.get("rms_ratio_vs_zero") is not None
        and float(row["rms_ratio_vs_zero"]) <= 1.0
        for row in primary_case_large
    )

    expected_control_keys = {
        *(f"field::{view}" for view in REQUIRED_FIELD_CONTROL_VIEWS),
        *FRONT_CONTROL_KEYS,
        *(f"integral_rms::{name}" for name in INTEGRAL_COMPONENT_NAMES),
        *(f"integral_endpoint::{name}" for name in INTEGRAL_COMPONENT_NAMES),
    }
    expected_scopes = {"population", *EVALUATION_CASE_IDS}
    expected_control_inventory = {
        (key, scope) for key in expected_control_keys for scope in expected_scopes
    }
    actual_control_pairs = [(row.key, row.scope) for row in controls]
    controls_exact = (
        len(actual_control_pairs) == len(set(actual_control_pairs))
        and set(actual_control_pairs) == expected_control_inventory
    )
    controls_pass = controls_exact and all(
        row.status == "ok"
        and np.isfinite(row.zero_rms)
        and row.zero_rms > DENOMINATOR_FLOOR
        and np.isfinite(row.corrected_rms)
        and row.ratio is not None
        and np.isfinite(row.ratio)
        and row.ratio <= 1.05
        for row in controls
    )

    expected_proposals = {
        (case_id, call)
        for case_id in EVALUATION_CASE_IDS
        for call in HELD_OUT_INPUT_CALLS
    }
    actual_proposals = [(row.case_id, row.input_call) for row in proposals]
    proposal_inventory_exact = (
        len(actual_proposals) == len(set(actual_proposals))
        and set(actual_proposals) == expected_proposals
    )
    proposals_pass = proposal_inventory_exact and all(
        row.finite and row.admissible for row in proposals
    )

    primary = evidence.get("joint_held_out")
    late_one = evidence.get("joint_late_1")
    late_two = evidence.get("joint_late_2")
    case_only = evidence.get("case_only")
    time_only = evidence.get("time_only")

    def resolved(item: CellScoreEvidence | None) -> bool:
        return bool(
            item is not None
            and item.score_status == "ok"
            and item.all_case_scores_resolved
            and item.skill_vs_zero is not None
            and item.rms_ratio_vs_zero is not None
            and item.median_case_cosine is not None
        )

    checks = {
        "cell_inventory_exact": cell_inventory_exact,
        "cell_contract_exact": cell_contract_exact,
        "primary_metrics_resolved": resolved(primary),
        "primary_r2_zero_at_least_0p10": bool(
            resolved(primary) and float(primary.skill_vs_zero) >= 0.10
        ),
        "primary_rms_ratio_at_most_0p95": bool(
            resolved(primary) and float(primary.rms_ratio_vs_zero) <= 0.95
        ),
        "primary_median_cosine_at_least_0p20": bool(
            resolved(primary) and float(primary.median_case_cosine) >= 0.20
        ),
        "primary_case_inventory_exact": primary_case_inventory_exact,
        "at_least_four_of_six_primary_cases_pass": (
            primary_case_inventory_exact and passing_cases >= 4
        ),
        "late_block_20_24_nonnegative_r2_positive_cosine": bool(
            resolved(late_one)
            and float(late_one.skill_vs_zero) >= 0.0
            and float(late_one.median_case_cosine) > 0.0
        ),
        "late_block_25_29_nonnegative_r2_positive_cosine": bool(
            resolved(late_two)
            and float(late_two.skill_vs_zero) >= 0.0
            and float(late_two.median_case_cosine) > 0.0
        ),
        "case_only_nonnegative_r2_positive_cosine": bool(
            resolved(case_only)
            and float(case_only.skill_vs_zero) >= 0.0
            and float(case_only.median_case_cosine) > 0.0
        ),
        "time_only_nonnegative_r2_positive_cosine": bool(
            resolved(time_only)
            and float(time_only.skill_vs_zero) >= 0.0
            and float(time_only.median_case_cosine) > 0.0
        ),
        "control_inventory_exact": controls_exact,
        "all_controls_no_harm": controls_pass,
        "proposal_inventory_exact": proposal_inventory_exact,
        "all_primary_proposals_finite_admissible": proposals_pass,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "p2_authorized": False,
        "checks": {key: bool(value) for key, value in checks.items()},
        "primary_passing_case_count": int(passing_cases),
        "cell_evidence": {key: asdict(value) for key, value in evidence.items()},
        "failed_controls": [
            asdict(row)
            for row in controls
            if row.status != "ok"
            or row.zero_rms <= DENOMINATOR_FLOOR
            or row.ratio is None
            or not np.isfinite(row.ratio)
            or row.ratio > 1.05
        ],
    }
