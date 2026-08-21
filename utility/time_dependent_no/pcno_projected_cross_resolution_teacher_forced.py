"""Fail-closed A2 contracts for projected cross-resolution diagnostics.

This module keeps the rank-8 A1 implementation immutable.  It adds compact
per-snapshot sufficient-statistic records so calibration and later open
evaluation cells can be combined without retaining fields or repeating model
calls.  Reference error remains an offline label and is never an inference
input.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DENOMINATOR_FLOOR,
    MODEL_NAMES,
    DiagnosticSnapshot,
    ScalarCorrectionStatistics,
    case_first_statistics,
    fit_scalar_correction,
    score_scalar_correction,
    snapshot_statistics,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    CELL_CALLS,
    CELL_CASES,
    EVALUATION_CASE_IDS,
    FIT_INPUT_CALLS,
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    REQUIRED_FIELD_CONTROL_VIEWS,
    ControlRatio,
    FixedCosineProjector,
    ProposalEvidence,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_correction import (
    PROJECTED_SELECTION_VIEW,
    ProjectionClosure,
    projected_coefficient_stability,
    projected_fit_snapshot,
)
from utility.time_dependent_no.pcno_resolution_transfer import Resolution


@dataclass(frozen=True)
class ProjectedStatisticRecord:
    """One case/call rank-8 fit record with its metric identity."""

    case_id: str
    group_id: str
    input_call: int
    metric_contract_sha256: str
    statistics: ScalarCorrectionStatistics
    projection_closure: ProjectionClosure


@dataclass(frozen=True)
class ProjectedCalibrationClosureEvidence:
    """Numerical and inventory evidence required before evaluation opens."""

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


def all_open_strength_groups() -> dict[str, tuple[str, ...]]:
    """Return the fixed 12-pair open-validation grouping."""

    cases = (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
    groups: dict[str, list[str]] = {}
    for case_id in cases:
        pieces = case_id.split("_")
        if len(pieces) != 3 or not pieces[1].startswith("e"):
            raise ValueError(f"unsupported shock-vortex case ID: {case_id!r}")
        groups.setdefault(pieces[1], []).append(case_id)
    return {
        group_id: tuple(sorted(members)) for group_id, members in sorted(groups.items())
    }


def _metric_contract_sha256(snapshot: DiagnosticSnapshot) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("physical_volumes", snapshot.volumes),
        ("component_scale", snapshot.component_scale),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_statistics(value: ScalarCorrectionStatistics) -> None:
    integer_fields = (value.snapshot_count, value.value_count)
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or int(item) <= 0
        for item in integer_fields
    ):
        raise ValueError("statistics counts must be positive integers")
    if int(value.snapshot_count) != 1:
        raise ValueError("each persisted record must represent exactly one snapshot")
    arrays = (
        np.asarray(value.feature_sum, dtype=np.float64),
        np.asarray(value.gram, dtype=np.float64),
        np.asarray(value.cross, dtype=np.float64),
    )
    if arrays[0].shape != (2,) or arrays[1].shape != (2, 2) or arrays[2].shape != (2,):
        raise ValueError("projected scalar statistics have invalid array shapes")
    scalars = (value.weight_sum, value.target_sum, value.target_square)
    if (
        not all(np.isfinite(float(item)) for item in scalars)
        or float(value.weight_sum) <= 0.0
        or float(value.target_square) < 0.0
        or any(not np.isfinite(array).all() for array in arrays)
    ):
        raise ValueError("projected scalar statistics must be finite")


def _validate_closure(value: ProjectionClosure) -> None:
    fields = asdict(value)
    if any(
        not np.isfinite(float(item)) or float(item) < 0.0 for item in fields.values()
    ):
        raise ValueError("projection closure fields must be finite and nonnegative")


def projected_statistic_record(
    snapshot: DiagnosticSnapshot,
    *,
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> ProjectedStatisticRecord:
    """Reduce one projected fit snapshot to a compact additive record."""

    projected, closure = projected_fit_snapshot(snapshot, projector)
    statistics, band_closure = snapshot_statistics(projected, resolution=resolution)
    if any(float(value) != 0.0 for value in band_closure.values()):
        raise AssertionError("unbanded projected statistics must have zero closure")
    record = ProjectedStatisticRecord(
        case_id=snapshot.case_id,
        group_id=snapshot.group_id,
        input_call=snapshot.input_call,
        metric_contract_sha256=_metric_contract_sha256(snapshot),
        statistics=statistics,
        projection_closure=closure,
    )
    _validate_record(record)
    return record


def projected_statistic_records(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> tuple[ProjectedStatisticRecord, ...]:
    """Build deterministic compact records from field snapshots."""

    return tuple(
        projected_statistic_record(
            snapshot,
            resolution=resolution,
            projector=projector,
        )
        for snapshot in sorted(
            snapshots,
            key=lambda row: (row.case_id, int(row.input_call)),
        )
    )


def projected_statistic_record_to_dict(
    record: ProjectedStatisticRecord,
) -> dict[str, Any]:
    """Encode one record losslessly for canonical JSON storage."""

    _validate_record(record)
    statistics = record.statistics
    return {
        "case_id": record.case_id,
        "group_id": record.group_id,
        "input_call": int(record.input_call),
        "metric_contract_sha256": record.metric_contract_sha256,
        "statistics": {
            "snapshot_count": int(statistics.snapshot_count),
            "value_count": int(statistics.value_count),
            "weight_sum": float(statistics.weight_sum),
            "feature_sum": np.asarray(
                statistics.feature_sum, dtype=np.float64
            ).tolist(),
            "target_sum": float(statistics.target_sum),
            "gram": np.asarray(statistics.gram, dtype=np.float64).tolist(),
            "cross": np.asarray(statistics.cross, dtype=np.float64).tolist(),
            "target_square": float(statistics.target_square),
        },
        "projection_closure": asdict(record.projection_closure),
    }


def projected_statistic_record_from_dict(
    payload: Mapping[str, Any],
) -> ProjectedStatisticRecord:
    """Decode one record with an exact fail-closed schema."""

    expected = {
        "case_id",
        "group_id",
        "input_call",
        "metric_contract_sha256",
        "statistics",
        "projection_closure",
    }
    if set(payload) != expected:
        raise ValueError("projected statistic record fields differ from the schema")
    statistics_payload = payload.get("statistics")
    if not isinstance(statistics_payload, Mapping) or set(statistics_payload) != {
        "snapshot_count",
        "value_count",
        "weight_sum",
        "feature_sum",
        "target_sum",
        "gram",
        "cross",
        "target_square",
    }:
        raise ValueError("projected statistic payload fields differ from the schema")
    closure_payload = payload.get("projection_closure")
    closure_fields = {
        "maximum_reconstruction_abs",
        "maximum_idempotence_abs",
        "relative_parallel_orthogonal_inner",
        "maximum_excluded_parallel_abs",
    }
    if (
        not isinstance(closure_payload, Mapping)
        or set(closure_payload) != closure_fields
    ):
        raise ValueError("projection closure fields differ from the schema")
    record = ProjectedStatisticRecord(
        case_id=payload["case_id"],
        group_id=payload["group_id"],
        input_call=payload["input_call"],
        metric_contract_sha256=payload["metric_contract_sha256"],
        statistics=ScalarCorrectionStatistics(
            snapshot_count=statistics_payload["snapshot_count"],
            value_count=statistics_payload["value_count"],
            weight_sum=float(statistics_payload["weight_sum"]),
            feature_sum=np.asarray(statistics_payload["feature_sum"], dtype=np.float64),
            target_sum=float(statistics_payload["target_sum"]),
            gram=np.asarray(statistics_payload["gram"], dtype=np.float64),
            cross=np.asarray(statistics_payload["cross"], dtype=np.float64),
            target_square=float(statistics_payload["target_square"]),
        ),
        projection_closure=ProjectionClosure(
            **{key: float(closure_payload[key]) for key in sorted(closure_fields)}
        ),
    )
    _validate_record(record)
    return record


def _validate_record(record: ProjectedStatisticRecord) -> None:
    if not isinstance(record.case_id, str) or not record.case_id:
        raise ValueError("record case IDs must be nonempty strings")
    if not isinstance(record.group_id, str) or not record.group_id:
        raise ValueError("record group IDs must be nonempty strings")
    if (
        isinstance(record.input_call, (bool, np.bool_))
        or not isinstance(record.input_call, (int, np.integer))
        or int(record.input_call) < 0
    ):
        raise ValueError("record input calls must be nonnegative integers")
    digest = record.metric_contract_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("record metric-contract digest must be lowercase SHA-256")
    _validate_statistics(record.statistics)
    _validate_closure(record.projection_closure)


def _validate_grouped_records(
    records: Sequence[ProjectedStatisticRecord],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> tuple[
    tuple[ProjectedStatisticRecord, ...],
    dict[str, str],
    tuple[str, ...],
    tuple[int, ...],
]:
    if not records:
        raise ValueError("cross-fitting requires statistic records")
    if not expected_groups or len(expected_groups) < 2:
        raise ValueError("expected_groups must declare at least two groups")
    case_to_group: dict[str, str] = {}
    for group_id, members_value in expected_groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be nonempty strings")
        if isinstance(members_value, (str, bytes)):
            raise TypeError("each expected group must contain a case sequence")
        members = tuple(members_value)
        if not members or any(
            not isinstance(case_id, str) or not case_id for case_id in members
        ):
            raise ValueError("expected case IDs must be nonempty strings")
        if len(set(members)) != len(members):
            raise ValueError(f"expected group {group_id!r} contains duplicate cases")
        for case_id in members:
            if case_id in case_to_group:
                raise ValueError(f"case {case_id!r} appears in multiple groups")
            case_to_group[case_id] = group_id

    calls_value = tuple(expected_input_calls)
    if not calls_value or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in calls_value
    ):
        raise ValueError("expected input calls must be nonnegative integers")
    calls = tuple(sorted(int(value) for value in calls_value))
    if len(set(calls)) != len(calls):
        raise ValueError("expected input calls must be unique")

    expected_pairs = {
        (case_id, input_call) for case_id in case_to_group for input_call in calls
    }
    actual_pairs: set[tuple[str, int]] = set()
    metric_digests: set[str] = set()
    for record in records:
        _validate_record(record)
        expected_group = case_to_group.get(record.case_id)
        if expected_group is None:
            raise ValueError(f"unexpected case ID: {record.case_id!r}")
        if record.group_id != expected_group:
            raise ValueError(f"case {record.case_id!r} is assigned to the wrong group")
        pair = (record.case_id, int(record.input_call))
        if pair in actual_pairs:
            raise ValueError(f"duplicate case/call statistic record: {pair!r}")
        actual_pairs.add(pair)
        metric_digests.add(record.metric_contract_sha256)
    if actual_pairs != expected_pairs:
        raise ValueError(
            "case/call statistic inventory does not match the frozen design: "
            f"missing={sorted(expected_pairs - actual_pairs)}, "
            f"extra={sorted(actual_pairs - expected_pairs)}"
        )
    if len(metric_digests) != 1:
        raise ValueError("all statistic records must share one metric contract")
    ordered = tuple(sorted(records, key=lambda row: (row.case_id, int(row.input_call))))
    return ordered, case_to_group, tuple(sorted(expected_groups)), calls


def validate_projected_statistic_inventory(
    records: Sequence[ProjectedStatisticRecord],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> tuple[ProjectedStatisticRecord, ...]:
    """Validate and canonically order a persisted statistic inventory."""

    ordered, _, _, _ = _validate_grouped_records(
        records,
        expected_groups=expected_groups,
        expected_input_calls=expected_input_calls,
    )
    return ordered


def _merge_projection_closures(
    values: Sequence[ProjectionClosure],
) -> ProjectionClosure:
    if not values:
        raise ValueError("at least one projection closure is required")
    return ProjectionClosure(
        maximum_reconstruction_abs=max(
            value.maximum_reconstruction_abs for value in values
        ),
        maximum_idempotence_abs=max(value.maximum_idempotence_abs for value in values),
        relative_parallel_orthogonal_inner=max(
            value.relative_parallel_orthogonal_inner for value in values
        ),
        maximum_excluded_parallel_abs=max(
            value.maximum_excluded_parallel_abs for value in values
        ),
    )


def _case_score_rows(
    statistics_rows: Sequence[tuple[str, ScalarCorrectionStatistics]],
    coefficients: tuple[float, float],
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for case_id in sorted(case_ids):
        statistics = case_first_statistics(
            [row for row in statistics_rows if row[0] == case_id]
        )
        rows.append(
            {
                "case_id": case_id,
                **asdict(score_scalar_correction(statistics, coefficients)),
            }
        )
    return rows


def projected_grouped_crossfit_from_records(
    records: Sequence[ProjectedStatisticRecord],
    *,
    projector: FixedCosineProjector,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Cross-fit registered scalar models using persisted rank-8 statistics."""

    if projector.q_matrix.ndim != 2 or projector.q_matrix.shape[1] != 8:
        raise ValueError("the registered correction requires a rank-8 projector")
    if len(projector.modes) != 8:
        raise ValueError("the registered correction requires eight fixed modes")
    ordered, case_to_group, groups, calls = _validate_grouped_records(
        records,
        expected_groups=expected_groups,
        expected_input_calls=expected_input_calls,
    )
    case_ids = tuple(sorted(case_to_group))
    model_rows: dict[str, Any] = {}
    for model in MODEL_NAMES:
        folds = []
        weighted_zero = 0.0
        weighted_corrected = 0.0
        scored_cases = 0
        all_ok = True
        for held_out_group in groups:
            train_rows = [
                (record.case_id, record.statistics)
                for record in ordered
                if record.group_id != held_out_group
            ]
            held_out_rows = [
                (record.case_id, record.statistics)
                for record in ordered
                if record.group_id == held_out_group
            ]
            fit = fit_scalar_correction(case_first_statistics(train_rows), model=model)
            held_out_cases = tuple(
                sorted(
                    case_id
                    for case_id in case_ids
                    if case_to_group[case_id] == held_out_group
                )
            )
            score = None
            case_scores: list[dict[str, Any]] = []
            if fit.status == "ok" and fit.coefficients is not None:
                score = score_scalar_correction(
                    case_first_statistics(held_out_rows),
                    fit.coefficients,
                )
                case_scores = _case_score_rows(
                    held_out_rows,
                    fit.coefficients,
                    held_out_cases,
                )
                if score.skill_status == "ok":
                    weighted_zero += len(held_out_cases) * score.zero_sse
                    weighted_corrected += len(held_out_cases) * score.corrected_sse
                    scored_cases += len(held_out_cases)
                else:
                    all_ok = False
            else:
                all_ok = False
            folds.append(
                {
                    "held_out_group": held_out_group,
                    "held_out_case_ids": list(held_out_cases),
                    "fit_case_ids": sorted(
                        case_id
                        for case_id in case_ids
                        if case_to_group[case_id] != held_out_group
                    ),
                    "fit_status": fit.status,
                    "coefficients": fit.coefficients,
                    "condition_number": fit.condition_number,
                    "population_skill_status": None
                    if score is None
                    else score.skill_status,
                    "skill_vs_zero": None if score is None else score.skill_vs_zero,
                    "rms_ratio_vs_zero": None
                    if score is None
                    else score.rms_ratio_vs_zero,
                    "case_scores": case_scores,
                }
            )

        skill = None
        rms_ratio = None
        if all_ok and scored_cases == len(case_ids) and weighted_zero > 0.0:
            ratio = weighted_corrected / weighted_zero
            skill = float(1.0 - ratio)
            rms_ratio = float(np.sqrt(ratio))
        out_of_fold_case_scores = sorted(
            [row for fold in folds for row in fold["case_scores"]],
            key=lambda row: str(row["case_id"]),
        )
        resolved_cosines = [
            float(row["cosine"])
            for row in out_of_fold_case_scores
            if row["cosine_status"] == "ok" and row["cosine"] is not None
        ]
        resolved_correlations = [
            float(row["correlation"])
            for row in out_of_fold_case_scores
            if row["correlation_status"] == "ok" and row["correlation"] is not None
        ]
        model_rows[model] = {
            "status": "ok" if skill is not None else "failed",
            "skill_vs_zero": skill,
            "rms_ratio_vs_zero": rms_ratio,
            "zero_sse_case_sum": weighted_zero if scored_cases else None,
            "corrected_sse_case_sum": weighted_corrected if scored_cases else None,
            "scored_case_count": scored_cases,
            "out_of_fold_case_scores": out_of_fold_case_scores,
            "median_oof_case_cosine": (
                float(np.median(resolved_cosines))
                if len(resolved_cosines) == len(case_ids)
                else None
            ),
            "median_oof_case_correlation": (
                float(np.median(resolved_correlations))
                if len(resolved_correlations) == len(case_ids)
                else None
            ),
            "folds": folds,
        }

    eligible = [
        model
        for model in MODEL_NAMES
        if model_rows[model]["status"] == "ok"
        and model_rows[model]["rms_ratio_vs_zero"] is not None
    ]
    if not eligible:
        raise ValueError("no projected correction model completed every group fold")
    complexity = {"zero": 0, "coarse_only": 1, "fine_only": 1, "two_term": 2}
    selected = min(
        eligible,
        key=lambda model: (
            float(model_rows[model]["rms_ratio_vs_zero"]),
            complexity[model],
            MODEL_NAMES.index(model),
        ),
    )
    full_statistics = case_first_statistics(
        [(record.case_id, record.statistics) for record in ordered]
    )
    full_fit = fit_scalar_correction(full_statistics, model=selected)
    closure = _merge_projection_closures(
        [record.projection_closure for record in ordered]
    )
    return {
        "selected_model": selected,
        "selected_coefficients": full_fit.coefficients,
        "selected_fit_status": full_fit.status,
        "selected_condition_number": full_fit.condition_number,
        "selected_full_fit": {
            **asdict(full_fit),
            "gram": full_statistics.gram.tolist(),
            "cross": full_statistics.cross.tolist(),
            "target_square": full_statistics.target_square,
            "target_rms": float(
                np.sqrt(full_statistics.target_square / full_statistics.weight_sum)
            ),
            "weight_sum": full_statistics.weight_sum,
        },
        "models": model_rows,
        "case_ids": list(case_ids),
        "groups": list(groups),
        "expected_groups": {
            group_id: sorted(expected_groups[group_id]) for group_id in groups
        },
        "expected_input_calls": list(calls),
        "selection_view": PROJECTED_SELECTION_VIEW,
        "projection_rank": int(projector.q_matrix.shape[1]),
        "projection_modes": [list(mode) for mode in projector.modes],
        "projection_closure": asdict(closure),
        "metric_contract_sha256": ordered[0].metric_contract_sha256,
    }


def qualify_projected_calibration(
    crossfit: Mapping[str, Any],
    closure: ProjectedCalibrationClosureEvidence,
) -> dict[str, Any]:
    """Apply the fixed projected calibration gate before target access expands."""

    selected = crossfit.get("selected_model")
    if selected not in MODEL_NAMES:
        selected = None
    expected_groups = sorted(CALIBRATION_GROUPS)
    inventory_checks = {
        "case_inventory_exact": crossfit.get("case_ids")
        == sorted(CALIBRATION_CASE_IDS),
        "group_inventory_exact": crossfit.get("groups") == expected_groups,
        "group_membership_exact": crossfit.get("expected_groups")
        == {key: sorted(CALIBRATION_GROUPS[key]) for key in expected_groups},
        "call_inventory_exact": crossfit.get("expected_input_calls")
        == list(FIT_INPUT_CALLS),
        "model_inventory_exact": set(crossfit.get("models", {})) == set(MODEL_NAMES),
        "selection_view_exact": crossfit.get("selection_view")
        == PROJECTED_SELECTION_VIEW,
        "projection_rank_exact": crossfit.get("projection_rank") == 8,
    }
    coefficients_value = crossfit.get("selected_coefficients")
    coefficients = None
    if (
        isinstance(coefficients_value, (list, tuple))
        and len(coefficients_value) == 2
        and all(np.isfinite(float(value)) for value in coefficients_value)
    ):
        coefficients = tuple(float(value) for value in coefficients_value)
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
    case_scores = selected_rows.get("out_of_fold_case_scores", [])
    all_oof_resolved = (
        len(case_scores) == len(CALIBRATION_CASE_IDS)
        and {row.get("case_id") for row in case_scores} == set(CALIBRATION_CASE_IDS)
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
    selected_oof_skill = selected_rows.get("skill_vs_zero")
    selected_oof_rms = selected_rows.get("rms_ratio_vs_zero")
    selected_oof_improves = bool(
        selected_oof_skill is not None
        and selected_oof_rms is not None
        and np.isfinite(float(selected_oof_skill))
        and np.isfinite(float(selected_oof_rms))
        and float(selected_oof_skill) > 0.0
        and float(selected_oof_rms) < 1.0
    )
    active_dimension = {"zero": 0, "coarse_only": 1, "fine_only": 1, "two_term": 2}.get(
        selected, 0
    )
    full_fit = crossfit.get("selected_full_fit", {})
    condition = crossfit.get("selected_condition_number")
    full_fit_ok = bool(
        active_dimension > 0
        and crossfit.get("selected_fit_status") == "ok"
        and full_fit.get("rank") == active_dimension
        and condition is not None
        and np.isfinite(float(condition))
        and float(condition) <= 1.0e6
    )
    stability = projected_coefficient_stability(
        crossfit,
        minimum_same_sign_folds=8,
        maximum_relative_iqr=0.5,
    )
    projection = crossfit.get("projection_closure", {})
    projection_closure_ok = bool(
        isinstance(projection, Mapping)
        and set(projection)
        == {
            "maximum_reconstruction_abs",
            "maximum_idempotence_abs",
            "relative_parallel_orthogonal_inner",
            "maximum_excluded_parallel_abs",
        }
        and float(projection["maximum_reconstruction_abs"]) <= 1.0e-10
        and float(projection["maximum_idempotence_abs"]) <= 1.0e-10
        and float(projection["relative_parallel_orthogonal_inner"]) <= 1.0e-10
        and float(projection["maximum_excluded_parallel_abs"]) == 0.0
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
        "projection_closure": projection_closure_ok,
    }
    checks = {
        **inventory_checks,
        "selected_nonzero_model": selected in MODEL_NAMES[1:],
        "selected_coefficients_finite": coefficients is not None,
        "fold_inventory_exact": fold_inventory_exact,
        "all_oof_metrics_resolved": all_oof_resolved,
        "selected_oof_improves_zero": selected_oof_improves,
        "active_gram_full_rank_well_conditioned": full_fit_ok,
        "coefficient_sign_scale_stable": stability["status"] == "passed",
        **closure_checks,
    }
    return {
        "status": "qualified" if all(checks.values()) else "not_qualified",
        "checks": {key: bool(value) for key, value in checks.items()},
        "selected_model": selected,
        "selected_coefficients": coefficients,
        "coefficient_stability": stability,
        "calibration_closure": asdict(closure),
    }


def _population_row(payload: Mapping[str, Any], view: str) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in payload.get("rows", [])
        if row.get("scope") == "population" and row.get("view") == view
    ]
    return matches[0] if len(matches) == 1 else None


def adaptive_projected_gate(
    *,
    cell_payloads: Mapping[str, Mapping[str, Any]],
    controls: Sequence[ControlRatio],
    proposals: Sequence[ProposalEvidence],
    all_open_crossfit: Mapping[str, Any],
    calibration_qualification: Mapping[str, Any],
    control_reconstruction_closure: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide only whether adaptive evidence supports requesting fresh work."""

    expected_cells = {
        "time_only",
        "case_only",
        "joint_held_out",
        "joint_late_1",
        "joint_late_2",
    }
    cell_inventory_exact = set(cell_payloads) == expected_cells
    cell_contract_exact = cell_inventory_exact and all(
        payload.get("case_ids") == sorted(CELL_CASES[cell])
        and payload.get("input_calls") == list(CELL_CALLS[cell])
        for cell, payload in cell_payloads.items()
    )
    full_rows = {
        cell: _population_row(payload, "full")
        for cell, payload in cell_payloads.items()
    }
    rank8_rows = {
        cell: _population_row(payload, "rank8_parallel")
        for cell, payload in cell_payloads.items()
    }
    full_nonnegative = cell_inventory_exact and all(
        row is not None
        and row.get("skill_status") == "ok"
        and row.get("skill_vs_zero") is not None
        and np.isfinite(float(row["skill_vs_zero"]))
        and float(row["skill_vs_zero"]) >= 0.0
        for row in full_rows.values()
    )
    primary_rank8 = rank8_rows.get("joint_held_out")
    primary_rank8_positive = bool(
        primary_rank8 is not None
        and primary_rank8.get("skill_status") == "ok"
        and primary_rank8.get("skill_vs_zero") is not None
        and primary_rank8.get("rms_ratio_vs_zero") is not None
        and float(primary_rank8["skill_vs_zero"]) > 0.0
        and float(primary_rank8["rms_ratio_vs_zero"]) < 1.0
    )

    unchanged_rows = [
        row
        for payload in cell_payloads.values()
        for row in payload.get("rows", [])
        if row.get("view") in {"rank8_orthogonal", "rank8_excluded"}
    ]
    expected_unchanged_count = sum(
        2 * (1 + len(CELL_CASES[cell])) for cell in expected_cells
    )
    orthogonal_excluded_unchanged = len(
        unchanged_rows
    ) == expected_unchanged_count and all(
        row.get("skill_status") == "ok"
        and row.get("zero_sse") is not None
        and row.get("corrected_sse") is not None
        and abs(float(row["corrected_sse"]) - float(row["zero_sse"]))
        <= 1.0e-10 * max(abs(float(row["zero_sse"])), 1.0)
        for row in unchanged_rows
    )
    projection_closure_exact = cell_inventory_exact and all(
        payload.get("maximum_closure", {}).get("band", float("inf")) <= 1.0e-10
        and payload.get("maximum_closure", {}).get(
            "subspace_reconstruction", float("inf")
        )
        <= 1.0e-10
        and payload.get("projection_closure", {}).get(
            "maximum_reconstruction_abs", float("inf")
        )
        <= 1.0e-10
        and payload.get("projection_closure", {}).get(
            "maximum_idempotence_abs", float("inf")
        )
        <= 1.0e-10
        and payload.get("projection_closure", {}).get(
            "relative_parallel_orthogonal_inner", float("inf")
        )
        <= 1.0e-10
        and payload.get("projection_closure", {}).get(
            "maximum_excluded_parallel_abs", float("inf")
        )
        == 0.0
        for payload in cell_payloads.values()
    )

    selected = all_open_crossfit.get("selected_model")
    selected_rows = all_open_crossfit.get("models", {}).get(selected, {})
    open_case_scores = selected_rows.get("out_of_fold_case_scores", [])
    open_groups = all_open_strength_groups()
    open_inventory_exact = bool(
        all_open_crossfit.get("case_ids")
        == sorted((*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS))
        and all_open_crossfit.get("groups") == sorted(open_groups)
        and all_open_crossfit.get("expected_groups")
        == {key: sorted(open_groups[key]) for key in sorted(open_groups)}
        and all_open_crossfit.get("expected_input_calls") == list(ALL_INPUT_CALLS)
    )
    open_oof_resolved = bool(
        len(open_case_scores) == len((*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS))
        and all(
            row.get(name) == "ok"
            for row in open_case_scores
            for name in (
                "skill_status",
                "centered_r2_status",
                "cosine_status",
                "correlation_status",
            )
        )
    )
    open_skill = selected_rows.get("skill_vs_zero")
    open_crossfit_positive = bool(
        selected in MODEL_NAMES[1:]
        and selected_rows.get("status") == "ok"
        and open_skill is not None
        and np.isfinite(float(open_skill))
        and float(open_skill) > 0.0
    )
    active_dimension = {"zero": 0, "coarse_only": 1, "fine_only": 1, "two_term": 2}.get(
        selected, 0
    )
    open_condition = all_open_crossfit.get("selected_condition_number")
    open_fit_resolved = bool(
        active_dimension > 0
        and all_open_crossfit.get("selected_fit_status") == "ok"
        and all_open_crossfit.get("selected_full_fit", {}).get("rank")
        == active_dimension
        and open_condition is not None
        and np.isfinite(float(open_condition))
        and float(open_condition) <= 1.0e6
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
    actual_controls = [(row.key, row.scope) for row in controls]
    controls_exact = (
        len(actual_controls) == len(set(actual_controls))
        and set(actual_controls) == expected_control_inventory
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
        (case_id, input_call)
        for case_id in EVALUATION_CASE_IDS
        for input_call in CELL_CALLS["joint_held_out"]
    }
    actual_proposals = [(row.case_id, row.input_call) for row in proposals]
    proposals_exact = (
        len(actual_proposals) == len(set(actual_proposals))
        and set(actual_proposals) == expected_proposals
    )
    proposals_pass = proposals_exact and all(
        row.finite and row.admissible for row in proposals
    )
    control_reconstruction_exact = bool(
        set(control_reconstruction_closure)
        == {"target_reconstruction", "excluded_correction", "model_calls"}
        and np.isfinite(float(control_reconstruction_closure["target_reconstruction"]))
        and float(control_reconstruction_closure["target_reconstruction"]) <= 1.0e-12
        and np.isfinite(float(control_reconstruction_closure["excluded_correction"]))
        and float(control_reconstruction_closure["excluded_correction"]) == 0.0
        and control_reconstruction_closure["model_calls"] == 0
    )
    calibration_checks = calibration_qualification.get("checks")
    calibration_qualified = bool(
        calibration_qualification.get("status") == "qualified"
        and isinstance(calibration_checks, Mapping)
        and calibration_checks
        and all(value is True for value in calibration_checks.values())
    )
    checks = {
        "calibration_remains_qualified": calibration_qualified,
        "cell_inventory_exact": cell_inventory_exact,
        "cell_contract_exact": cell_contract_exact,
        "all_aggregate_full_field_skills_nonnegative": full_nonnegative,
        "joint_held_out_rank8_skill_positive": primary_rank8_positive,
        "orthogonal_excluded_no_change": orthogonal_excluded_unchanged,
        "projection_closure": projection_closure_exact,
        "all_open_lopo_inventory_exact": open_inventory_exact,
        "all_open_lopo_metrics_resolved": open_oof_resolved,
        "all_open_lopo_skill_positive": open_crossfit_positive,
        "all_open_lopo_fit_full_rank_well_conditioned": open_fit_resolved,
        "control_inventory_exact": controls_exact,
        "all_controls_no_harm": controls_pass,
        "proposal_inventory_exact": proposals_exact,
        "all_primary_proposals_finite_admissible": proposals_pass,
        "control_reconstruction_closure": control_reconstruction_exact,
    }
    return {
        "status": "continuation_request_supported"
        if all(checks.values())
        else "stopped",
        "continuation_request_supported": bool(all(checks.values())),
        "fresh_confirmation": False,
        "recurrence_authorized": False,
        "sealed_population_authorized": False,
        "checks": {key: bool(value) for key, value in checks.items()},
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
