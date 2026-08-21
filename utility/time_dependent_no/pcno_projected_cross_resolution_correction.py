"""Rank-8 projected cross-resolution correction primitives for W26-L5.

The frozen W26-L5 A1/A2 modules remain unchanged.  This module reuses their
typed snapshots, scalar sufficient statistics, and physical-cosine projector
while restricting a correction to the already registered weighted rank-8
subspace.  Reference error is an offline fit/score label only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_correction import (
    MODEL_NAMES,
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    ScalarCorrectionStatistics,
    SynchronizedFusionStep,
    case_first_statistics,
    common_native_increment_basis,
    fit_scalar_correction,
    prepare_common_native_inputs,
    score_scalar_correction,
    snapshot_statistics,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FixedCosineProjector,
    score_cell,
)
from utility.time_dependent_no.pcno_resolution_transfer import Resolution

PROJECTED_SELECTION_VIEW = "rank8_parallel"


@dataclass(frozen=True)
class ProjectionClosure:
    """Numerical closure of one weighted orthogonal projection."""

    maximum_reconstruction_abs: float
    maximum_idempotence_abs: float
    relative_parallel_orthogonal_inner: float
    maximum_excluded_parallel_abs: float


def _maximum_absolute(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _require_registered_projector(projector: FixedCosineProjector) -> None:
    if projector.q_matrix.ndim != 2 or projector.q_matrix.shape[1] != 8:
        raise ValueError("the registered correction requires a rank-8 projector")
    if len(projector.modes) != 8:
        raise ValueError("the registered correction requires eight fixed modes")


def _merge_closures(values: Sequence[ProjectionClosure]) -> ProjectionClosure:
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


def project_parallel_field(
    field: np.ndarray,
    projector: FixedCosineProjector,
) -> tuple[np.ndarray, ProjectionClosure]:
    """Project one native field and report reconstruction/orthogonality closure."""

    _require_registered_projector(projector)
    split, base_closure = projector.split(field)
    parallel = split["parallel"]
    orthogonal = split["orthogonal"]
    repeated, _ = projector.split(parallel)

    interior = projector.interior_mask
    root_mass = projector.square_root_mass[:, None]
    weighted_parallel = root_mass * parallel[interior]
    weighted_orthogonal = root_mass * orthogonal[interior]
    denominator = float(
        np.linalg.norm(weighted_parallel) * np.linalg.norm(weighted_orthogonal)
    )
    relative_inner = 0.0
    if denominator > np.finfo(np.float64).tiny:
        relative_inner = (
            abs(float(np.sum(weighted_parallel * weighted_orthogonal))) / denominator
        )

    closure = ProjectionClosure(
        maximum_reconstruction_abs=float(base_closure["maximum_reconstruction_abs"]),
        maximum_idempotence_abs=_maximum_absolute(repeated["parallel"] - parallel),
        relative_parallel_orthogonal_inner=float(relative_inner),
        maximum_excluded_parallel_abs=_maximum_absolute(parallel[~interior]),
    )
    return parallel, closure


def _project_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
    *,
    project_target: bool,
) -> tuple[DiagnosticSnapshot, ProjectionClosure]:
    coarse_feature, coarse_closure = project_parallel_field(
        snapshot.basis.native_minus_coarse,
        projector,
    )
    fine_feature, fine_closure = project_parallel_field(
        snapshot.basis.fine_minus_native,
        projector,
    )
    closures = [coarse_closure, fine_closure]
    if project_target:
        target, target_closure = project_parallel_field(
            snapshot.target_correction,
            projector,
        )
        closures.append(target_closure)
    else:
        target = np.array(snapshot.target_correction, dtype=np.float64, copy=True)

    zero = np.zeros_like(coarse_feature)
    return (
        DiagnosticSnapshot(
            case_id=snapshot.case_id,
            group_id=snapshot.group_id,
            input_call=snapshot.input_call,
            basis=NativeIncrementBasis(
                native_increment=zero,
                coarse_on_native=zero,
                fine_on_native=zero,
                native_minus_coarse=coarse_feature,
                fine_minus_native=fine_feature,
            ),
            target_correction=target,
            volumes=np.asarray(snapshot.volumes, dtype=np.float64),
            component_scale=np.asarray(snapshot.component_scale, dtype=np.float64),
            masks=snapshot.masks,
        ),
        _merge_closures(closures),
    )


def projected_fit_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
) -> tuple[DiagnosticSnapshot, ProjectionClosure]:
    """Project both deployable features and the offline fit target."""

    return _project_snapshot(snapshot, projector, project_target=True)


def projected_candidate_snapshot(
    snapshot: DiagnosticSnapshot,
    projector: FixedCosineProjector,
) -> tuple[DiagnosticSnapshot, ProjectionClosure]:
    """Project deployable features while retaining the full scoring target."""

    return _project_snapshot(snapshot, projector, project_target=False)


def projected_correction_field(
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Return ``Pi(alpha*x_c + beta*x_f)`` on the native grid."""

    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("correction coefficients must be finite")
    coarse = np.asarray(basis.native_minus_coarse)
    fine = np.asarray(basis.fine_minus_native)
    if coarse.shape != fine.shape:
        raise ValueError("cross-resolution features must have identical shapes")
    if alpha == 0.0 and beta == 0.0:
        return np.zeros_like(coarse, dtype=np.float64)
    combined = float(alpha) * np.asarray(coarse, dtype=np.float64) + float(
        beta
    ) * np.asarray(fine, dtype=np.float64)
    parallel, _ = project_parallel_field(combined, projector)
    return parallel


def projected_native_prediction(
    native_prediction: np.ndarray,
    basis: NativeIncrementBasis,
    projector: FixedCosineProjector,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Apply only the projected correction; zero is an exact raw copy."""

    raw = np.asarray(native_prediction)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("correction coefficients must be finite")
    if alpha == 0.0 and beta == 0.0:
        return np.array(raw, copy=True)
    correction = projected_correction_field(
        basis,
        projector,
        alpha=alpha,
        beta=beta,
    )
    if raw.shape != correction.shape:
        raise ValueError("native prediction and correction must have identical shapes")
    return np.asarray(raw, dtype=np.float64) + correction


def projected_synchronized_fusion_step(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    projector: FixedCosineProjector,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    alpha: float,
    beta: float,
) -> SynchronizedFusionStep:
    """Evaluate three synchronized grids and retain one corrected native state."""

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    predictions: dict[Resolution, np.ndarray] = {}
    for resolution in (contract.coarse, contract.native, contract.fine):
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
    basis = common_native_increment_basis(
        prepared,
        predictions,
        contract=contract,
    )
    next_native = projected_native_prediction(
        predictions[contract.native],
        basis,
        projector,
        alpha=alpha,
        beta=beta,
    )
    return SynchronizedFusionStep(
        prepared_inputs=prepared,
        predictions=predictions,
        basis=basis,
        next_native_state=next_native,
    )


def _validate_grouped_inventory(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> tuple[
    tuple[DiagnosticSnapshot, ...],
    dict[str, str],
    tuple[str, ...],
    tuple[int, ...],
]:
    if not snapshots:
        raise ValueError("cross-fitting requires snapshots")
    if not expected_groups or len(expected_groups) < 2:
        raise ValueError("expected_groups must declare at least two groups")

    case_to_group: dict[str, str] = {}
    for group_id, members in expected_groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be nonempty strings")
        if isinstance(members, (str, bytes)):
            raise TypeError("each expected group must contain a case sequence")
        cases = tuple(members)
        if not cases or any(
            not isinstance(case_id, str) or not case_id for case_id in cases
        ):
            raise ValueError("expected case IDs must be nonempty strings")
        if len(set(cases)) != len(cases):
            raise ValueError(f"expected group {group_id!r} contains duplicate cases")
        for case_id in cases:
            if case_id in case_to_group:
                raise ValueError(f"case {case_id!r} appears in multiple groups")
            case_to_group[case_id] = group_id

    calls = tuple(expected_input_calls)
    if not calls or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in calls
    ):
        raise ValueError("expected input calls must be nonnegative integers")
    calls = tuple(sorted(int(value) for value in calls))
    if len(set(calls)) != len(calls):
        raise ValueError("expected input calls must be unique")

    expected_pairs = {
        (case_id, input_call) for case_id in case_to_group for input_call in calls
    }
    actual_pairs: set[tuple[str, int]] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot.case_id, str) or not snapshot.case_id:
            raise ValueError("snapshot case IDs must be nonempty strings")
        if not isinstance(snapshot.group_id, str) or not snapshot.group_id:
            raise ValueError("snapshot group IDs must be nonempty strings")
        if (
            isinstance(snapshot.input_call, (bool, np.bool_))
            or not isinstance(snapshot.input_call, (int, np.integer))
            or int(snapshot.input_call) < 0
        ):
            raise ValueError("snapshot input calls must be nonnegative integers")
        expected_group = case_to_group.get(snapshot.case_id)
        if expected_group is None:
            raise ValueError(f"unexpected case ID: {snapshot.case_id!r}")
        if snapshot.group_id != expected_group:
            raise ValueError(
                f"case {snapshot.case_id!r} is assigned to the wrong group"
            )
        pair = (snapshot.case_id, int(snapshot.input_call))
        if pair in actual_pairs:
            raise ValueError(f"duplicate case/call snapshot: {pair!r}")
        actual_pairs.add(pair)
    if actual_pairs != expected_pairs:
        raise ValueError(
            "case/call inventory does not match the frozen design: "
            f"missing={sorted(expected_pairs - actual_pairs)}, "
            f"extra={sorted(actual_pairs - expected_pairs)}"
        )

    ordered = tuple(
        sorted(snapshots, key=lambda row: (row.case_id, int(row.input_call)))
    )
    reference_volumes = np.asarray(ordered[0].volumes)
    reference_scales = np.asarray(ordered[0].component_scale)
    if any(
        not np.array_equal(np.asarray(row.volumes), reference_volumes)
        or not np.array_equal(np.asarray(row.component_scale), reference_scales)
        for row in ordered[1:]
    ):
        raise ValueError("all snapshots must share one frozen metric contract")
    return ordered, case_to_group, tuple(sorted(expected_groups)), calls


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


def projected_grouped_crossfit(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    resolution: Resolution,
    projector: FixedCosineProjector,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Select fixed scalars by group cross-fit in the rank-8 parallel space."""

    ordered, case_to_group, groups, calls = _validate_grouped_inventory(
        snapshots,
        expected_groups=expected_groups,
        expected_input_calls=expected_input_calls,
    )
    statistics_by_snapshot: list[
        tuple[DiagnosticSnapshot, ScalarCorrectionStatistics]
    ] = []
    projection_closures = []
    for snapshot in ordered:
        projected, closure = projected_fit_snapshot(snapshot, projector)
        statistics, band_closure = snapshot_statistics(
            projected,
            resolution=resolution,
        )
        if any(float(value) != 0.0 for value in band_closure.values()):
            raise AssertionError("unbanded projected statistics must have zero closure")
        statistics_by_snapshot.append((snapshot, statistics))
        projection_closures.append(closure)

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
                (snapshot.case_id, statistics)
                for snapshot, statistics in statistics_by_snapshot
                if snapshot.group_id != held_out_group
            ]
            held_out_rows = [
                (snapshot.case_id, statistics)
                for snapshot, statistics in statistics_by_snapshot
                if snapshot.group_id == held_out_group
            ]
            train_statistics = case_first_statistics(train_rows)
            held_out_statistics = case_first_statistics(held_out_rows)
            fit = fit_scalar_correction(train_statistics, model=model)
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
                    held_out_statistics,
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
                    "population_skill_status": (
                        None if score is None else score.skill_status
                    ),
                    "skill_vs_zero": None if score is None else score.skill_vs_zero,
                    "rms_ratio_vs_zero": (
                        None if score is None else score.rms_ratio_vs_zero
                    ),
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
        [
            (snapshot.case_id, statistics)
            for snapshot, statistics in statistics_by_snapshot
        ]
    )
    full_fit = fit_scalar_correction(full_statistics, model=selected)
    closure = _merge_closures(projection_closures)
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
    }


def projected_coefficient_stability(
    crossfit: Mapping[str, Any],
    *,
    minimum_same_sign_folds: int,
    maximum_relative_iqr: float = 0.5,
) -> dict[str, Any]:
    """Fail-closed sign/scale stability for the selected projected model."""

    if (
        isinstance(minimum_same_sign_folds, bool)
        or int(minimum_same_sign_folds) != minimum_same_sign_folds
        or minimum_same_sign_folds < 1
    ):
        raise ValueError("minimum_same_sign_folds must be a positive integer")
    if not np.isfinite(maximum_relative_iqr) or maximum_relative_iqr < 0.0:
        raise ValueError("maximum_relative_iqr must be finite and nonnegative")

    selected = crossfit.get("selected_model")
    active = {
        "zero": (),
        "coarse_only": (0,),
        "fine_only": (1,),
        "two_term": (0, 1),
    }.get(selected, ())
    models = crossfit.get("models")
    rows = models.get(selected, {}) if isinstance(models, Mapping) else {}
    folds = rows.get("folds", []) if isinstance(rows, Mapping) else []
    coefficients = crossfit.get("selected_coefficients")
    results = []
    passed = bool(active) and isinstance(coefficients, (tuple, list))
    for index in active:
        values = []
        for fold in folds:
            fold_coefficients = fold.get("coefficients")
            if (
                fold.get("fit_status") != "ok"
                or not isinstance(fold_coefficients, (tuple, list))
                or len(fold_coefficients) != 2
                or not np.isfinite(float(fold_coefficients[index]))
            ):
                values = []
                break
            values.append(float(fold_coefficients[index]))
        full_value = (
            float(coefficients[index])
            if isinstance(coefficients, (tuple, list))
            and len(coefficients) == 2
            and np.isfinite(float(coefficients[index]))
            else None
        )
        same_sign = 0
        median = None
        iqr = None
        relative_iqr = None
        status = "unresolved"
        if values and full_value is not None and full_value != 0.0:
            array = np.asarray(values, dtype=np.float64)
            same_sign = int(np.count_nonzero(np.sign(array) == np.sign(full_value)))
            median = float(np.median(array))
            q25, q75 = np.percentile(array, (25.0, 75.0))
            iqr = float(q75 - q25)
            if median != 0.0 and np.isfinite(median):
                relative_iqr = float(iqr / abs(median))
                if same_sign >= int(minimum_same_sign_folds) and relative_iqr <= float(
                    maximum_relative_iqr
                ):
                    status = "ok"
        passed = passed and status == "ok"
        results.append(
            {
                "coefficient_index": index,
                "full_fit_value": full_value,
                "fold_values": values,
                "same_sign_fold_count": same_sign,
                "fold_median": median,
                "fold_iqr": iqr,
                "fold_iqr_over_abs_median": relative_iqr,
                "status": status,
            }
        )
    return {"status": "passed" if passed else "failed", "rows": results}


def score_projected_cell(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    cell: str,
    coefficients: tuple[float, float],
    resolution: Resolution,
    projector: FixedCosineProjector,
) -> dict[str, Any]:
    """Score the projected policy against full A2 views and controls."""

    transformed = []
    closures = []
    for snapshot in snapshots:
        projected, closure = projected_candidate_snapshot(snapshot, projector)
        transformed.append(projected)
        closures.append(closure)
    payload = score_cell(
        transformed,
        cell=cell,
        coefficients=coefficients,
        resolution=resolution,
        projector=projector,
    )
    payload["correction_policy"] = PROJECTED_SELECTION_VIEW
    payload["projection_closure"] = asdict(_merge_closures(closures))
    return payload
