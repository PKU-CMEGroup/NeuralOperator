"""Truth-free branch-consistency scores for W26-L5 A21.

The score consumes only synchronized predicted increments from one physical
state. Reference fields are accepted only by the retrospective population
scorer, never by :func:`branch_consistency_score` or :func:`select_branch`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    FixedCosineProjector,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    modal_coordinates,
)
from utility.time_dependent_no.pcno_resolution_transfer import weighted_scaled_rms
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    ModalCell,
    validate_active_cells,
)

DENOMINATOR_FLOOR = 1.0e-8
TIE_ABSOLUTE_TOLERANCE = 1.0e-12
TIE_RELATIVE_TOLERANCE = 1.0e-12

BranchName = Literal["raw", "corrected"]


@dataclass(frozen=True)
class BranchConsistencyScore:
    """Dimensionless SP19 commutator score for one candidate state."""

    status: str
    score: float | None
    commutator_rms: float
    native_increment_rms: float
    commutator_modal_energy: float
    native_increment_modal_energy: float


@dataclass(frozen=True)
class BranchDecision:
    """Truth-free choice between raw and corrected candidate states."""

    selected_branch: BranchName
    status: str
    resolved: bool
    score_difference_corrected_minus_raw: float | None


def _modal_masked_field(
    field: np.ndarray,
    projector: FixedCosineProjector,
    *,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell],
) -> tuple[np.ndarray, float]:
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
    weighted = projector.q_matrix @ retained
    interior = weighted / projector.square_root_mass[:, None]
    scale = np.asarray(component_scale, dtype=np.float64)
    masked = np.zeros_like(values)
    masked[projector.interior_mask] = interior * scale[None, :]
    return masked, float(np.sum(np.square(retained)))


def branch_consistency_score(
    native_increment: np.ndarray,
    fine_on_native_increment: np.ndarray,
    projector: FixedCosineProjector,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    active_cells: Sequence[ModalCell] = FROZEN_ACTIVE_CELLS,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> BranchConsistencyScore:
    """Score a state using its synchronized fine/native SP19 commutator."""

    native = np.asarray(native_increment, dtype=np.float64)
    fine = np.asarray(fine_on_native_increment, dtype=np.float64)
    if native.shape != fine.shape:
        raise ValueError("native and mapped-fine increments must have equal shape")
    if native.ndim != 2 or not np.isfinite(native).all() or not np.isfinite(fine).all():
        raise ValueError("increments must be finite node-by-component arrays")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive and finite")

    native_sp19, native_energy = _modal_masked_field(
        native,
        projector,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    commutator_sp19, commutator_energy = _modal_masked_field(
        fine - native,
        projector,
        component_scale=component_scale,
        active_cells=active_cells,
    )
    numerator = weighted_scaled_rms(
        commutator_sp19,
        volumes=volumes,
        component_scale=component_scale,
    )
    denominator = weighted_scaled_rms(
        native_sp19,
        volumes=volumes,
        component_scale=component_scale,
    )
    if denominator <= denominator_floor:
        return BranchConsistencyScore(
            status="small_native_sp19_increment",
            score=None,
            commutator_rms=numerator,
            native_increment_rms=denominator,
            commutator_modal_energy=commutator_energy,
            native_increment_modal_energy=native_energy,
        )
    score = numerator / denominator
    if not np.isfinite(score):
        return BranchConsistencyScore(
            status="nonfinite_ratio",
            score=None,
            commutator_rms=numerator,
            native_increment_rms=denominator,
            commutator_modal_energy=commutator_energy,
            native_increment_modal_energy=native_energy,
        )
    return BranchConsistencyScore(
        status="ok",
        score=float(score),
        commutator_rms=numerator,
        native_increment_rms=denominator,
        commutator_modal_energy=commutator_energy,
        native_increment_modal_energy=native_energy,
    )


def select_branch(
    raw: BranchConsistencyScore,
    corrected: BranchConsistencyScore,
    *,
    absolute_tolerance: float = TIE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = TIE_RELATIVE_TOLERANCE,
) -> BranchDecision:
    """Select the lower resolved score; ties and failures abstain to raw."""

    if (
        not np.isfinite([absolute_tolerance, relative_tolerance]).all()
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("selection tolerances must be finite and nonnegative")
    if raw.status != "ok" or corrected.status != "ok":
        return BranchDecision(
            selected_branch="raw",
            status="unresolved_abstain_raw",
            resolved=False,
            score_difference_corrected_minus_raw=None,
        )
    if raw.score is None or corrected.score is None:  # defensive typed boundary
        raise ValueError("ok score status requires a numeric score")
    difference = float(corrected.score - raw.score)
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * max(abs(raw.score), abs(corrected.score)),
    )
    if abs(difference) <= tolerance:
        return BranchDecision(
            selected_branch="raw",
            status="tie_abstain_raw",
            resolved=False,
            score_difference_corrected_minus_raw=difference,
        )
    return BranchDecision(
        selected_branch="corrected" if difference < 0.0 else "raw",
        status="resolved",
        resolved=True,
        score_difference_corrected_minus_raw=difference,
    )


def _strict_row_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{key} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return result


def _signed_relation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    inner = float(np.dot(x, y))
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    cosine_denominator = x_norm * y_norm
    cosine = None if cosine_denominator <= DENOMINATOR_FLOOR else inner / cosine_denominator
    x_centered = x - float(x.mean())
    y_centered = y - float(y.mean())
    pearson_denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    pearson = (
        None
        if pearson_denominator <= DENOMINATOR_FLOOR
        else float(np.dot(x_centered, y_centered) / pearson_denominator)
    )
    nonzero = (np.abs(x) > TIE_ABSOLUTE_TOLERANCE) & (
        np.abs(y) > TIE_ABSOLUTE_TOLERANCE
    )
    return {
        "count": int(x.size),
        "cosine": None if cosine is None else float(cosine),
        "cosine_status": "small_denominator" if cosine is None else "ok",
        "cosine_denominator": cosine_denominator,
        "pearson": pearson,
        "pearson_status": "small_denominator" if pearson is None else "ok",
        "pearson_denominator": pearson_denominator,
        "sign_agreement": (
            None
            if not np.any(nonzero)
            else float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero])))
        ),
        "sign_comparison_count": int(np.count_nonzero(nonzero)),
    }


def score_branch_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    error_prefix: str,
) -> dict[str, Any]:
    """Case-first retrospective scoring for one error horizon."""

    if not rows:
        raise ValueError("branch rows must be nonempty")
    case_ids = sorted({str(row.get("case_id", "")) for row in rows})
    if any(not case_id for case_id in case_ids):
        raise ValueError("each row needs a nonempty case_id")
    pairs = [(str(row["case_id"]), int(row["input_call"])) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("case/call rows must be unique")

    case_scores: list[dict[str, Any]] = []
    total_sse = {name: 0.0 for name in ("raw", "corrected", "selected", "oracle")}
    for case_id in case_ids:
        subset = [row for row in rows if row["case_id"] == case_id]
        if not subset:
            raise AssertionError("case inventory construction failed")
        case_record: dict[str, Any] = {"case_id": case_id, "count": len(subset)}
        for name in total_sse:
            values = np.asarray(
                [_strict_row_value(row, f"{error_prefix}_{name}_rms") for row in subset],
                dtype=np.float64,
            )
            mean_square = float(np.mean(np.square(values)))
            total_sse[name] += mean_square
            case_record[f"{name}_rms"] = float(np.sqrt(mean_square))
        raw_case = case_record["raw_rms"]
        for name in ("corrected", "selected", "oracle"):
            case_record[f"{name}_to_raw_ratio"] = (
                None if raw_case <= DENOMINATOR_FLOOR else case_record[f"{name}_rms"] / raw_case
            )
        case_scores.append(case_record)

    population_rms = {
        name: float(np.sqrt(total_sse[name] / len(case_ids))) for name in total_sse
    }
    ratios: dict[str, float | None] = {}
    for numerator, denominator in (
        ("selected", "raw"),
        ("selected", "corrected"),
        ("oracle", "raw"),
    ):
        ratios[f"{numerator}_to_{denominator}_rms_ratio"] = (
            None
            if population_rms[denominator] <= DENOMINATOR_FLOOR
            else population_rms[numerator] / population_rms[denominator]
        )

    resolved_rows = [row for row in rows if bool(row.get("selector_resolved"))]
    accuracy_flags = []
    for row in resolved_rows:
        raw_error = _strict_row_value(row, f"{error_prefix}_raw_rms")
        corrected_error = _strict_row_value(row, f"{error_prefix}_corrected_rms")
        if abs(raw_error - corrected_error) <= TIE_ABSOLUTE_TOLERANCE:
            continue
        truth_winner = "raw" if raw_error < corrected_error else "corrected"
        accuracy_flags.append(str(row.get("selected_branch")) == truth_winner)

    score_difference = np.asarray(
        [float(row["score_difference_corrected_minus_raw"]) for row in resolved_rows],
        dtype=np.float64,
    )
    error_difference = np.asarray(
        [
            _strict_row_value(row, f"{error_prefix}_corrected_rms") ** 2
            - _strict_row_value(row, f"{error_prefix}_raw_rms") ** 2
            for row in resolved_rows
        ],
        dtype=np.float64,
    )
    relation = (
        _signed_relation(score_difference, error_difference)
        if resolved_rows
        else {
            "count": 0,
            "cosine": None,
            "cosine_status": "empty",
            "cosine_denominator": 0.0,
            "pearson": None,
            "pearson_status": "empty",
            "pearson_denominator": 0.0,
            "sign_agreement": None,
            "sign_comparison_count": 0,
        }
    )
    maximum_case_ratio = max(
        float(case["selected_to_raw_ratio"])
        for case in case_scores
        if case["selected_to_raw_ratio"] is not None
    )
    return {
        "error_prefix": error_prefix,
        "case_count": len(case_ids),
        "row_count": len(rows),
        "resolved_count": len(resolved_rows),
        "abstention_count": len(rows) - len(resolved_rows),
        "selector_accuracy": (
            None if not accuracy_flags else float(np.mean(accuracy_flags))
        ),
        "selector_accuracy_count": len(accuracy_flags),
        "population_rms": population_rms,
        **ratios,
        "strict_selected_case_win_count": sum(
            case["selected_rms"] < case["raw_rms"] for case in case_scores
        ),
        "maximum_selected_to_raw_case_rms_ratio": maximum_case_ratio,
        "score_error_relation": relation,
        "case_scores": case_scores,
    }


def synthetic_summary() -> dict[str, Any]:
    """Small serialization-friendly closure used by readiness tests."""

    raw = BranchConsistencyScore("ok", 0.4, 0.2, 0.5, 1.0, 2.0)
    corrected = BranchConsistencyScore("ok", 0.3, 0.15, 0.5, 0.5, 2.0)
    decision = select_branch(raw, corrected)
    unresolved = select_branch(
        BranchConsistencyScore("small_native_sp19_increment", None, 0.0, 0.0, 0.0, 0.0),
        corrected,
    )
    return {
        "decision": asdict(decision),
        "unresolved": asdict(unresolved),
        "passed": decision.selected_branch == "corrected"
        and decision.resolved
        and unresolved.selected_branch == "raw"
        and not unresolved.resolved,
    }
