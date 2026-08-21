"""Offline residual-geometry diagnostics for synchronized grid predictions.

The inputs are physical-volume/component-scaled modal coordinates already
written by the W26-L5 teacher-forced evaluator.  This module has no model,
dataset, reference-array, or recurrence dependency.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DENOMINATOR_FLOOR = 1.0e-8
FIXED_BETA = -0.5
ViewName = Literal["all_nonconstant", "sp19", "complement"]
VIEW_NAMES: tuple[ViewName, ...] = (
    "all_nonconstant",
    "sp19",
    "complement",
)
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class ModalSnapshot:
    """One synchronized coarse/native/fine diagnostic in modal coordinates."""

    case_id: str
    group_id: str
    input_call: int
    cells: tuple[tuple[int, int], ...]
    coarse: np.ndarray
    fine: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class GeometryRecord:
    """One fixed modal view for a case and input call."""

    case_id: str
    group_id: str
    input_call: int
    view: ViewName
    coarse: np.ndarray
    fine: np.ndarray
    target: np.ndarray
    persistence_cosine: float | None
    persistence_denominator: float
    persistence_status: str


PHASE_AUDIT_FEATURES = (
    "target_previous_cosine",
    "log_target_previous_ratio",
    "log_requested_previous_ratio",
    "log_applied_shadow_increment_ratio",
    "log_correction_native_increment_ratio",
)
PHASE_MODEL_FEATURES = {
    "time_only": ("normalized_time",),
    "audit_only": PHASE_AUDIT_FEATURES,
    "combined": ("normalized_time", *PHASE_AUDIT_FEATURES),
}
PHASE_RIDGE = 1.0e-6


@dataclass(frozen=True)
class RecurrentPhaseRecord:
    """One saved raw/corrected call paired with its truth-free tether audit."""

    population: str
    case_id: str
    output_call: int
    raw_state_error: float
    corrected_state_error: float
    benefit: float
    features: Mapping[str, float]


def recurrent_phase_records(
    call_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    population: str,
    expected_cases: Sequence[str],
    corrected_policy: str,
    expected_calls: Sequence[int] = tuple(range(9, 31)),
) -> tuple[RecurrentPhaseRecord, ...]:
    """Parse exact recurrent inventories and derive truth-free scalar features."""

    cases = tuple(expected_cases)
    calls = tuple(expected_calls)
    if (
        not population
        or not cases
        or len(set(cases)) != len(cases)
        or not calls
        or len(set(calls)) != len(calls)
    ):
        raise ValueError(
            "phase population, cases, and calls must be nonempty and unique"
        )
    if any(not isinstance(case, str) or not case for case in cases):
        raise ValueError("phase case IDs must be nonempty strings")
    if any(isinstance(call, bool) or not isinstance(call, int) for call in calls):
        raise ValueError("phase calls must be canonical integers")

    expected = {(case, call) for case in cases for call in calls}
    metric_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in call_rows:
        case_id = row.get("case_id")
        policy = row.get("policy")
        if case_id not in cases or policy not in {"zero", corrected_policy}:
            continue
        output_call = _parse_integer(row.get("output_call"), name="output_call")
        if output_call not in calls:
            continue
        key = (str(case_id), str(policy), output_call)
        if key in metric_lookup:
            raise ValueError("duplicate recurrent call metric")
        metric_lookup[key] = row
    expected_metrics = {
        (case, policy, call)
        for case, call in expected
        for policy in ("zero", corrected_policy)
    }
    if set(metric_lookup) != expected_metrics:
        raise ValueError("recurrent call metric inventory mismatch")

    audit_lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in audit_rows:
        case_id = row.get("case_id")
        if case_id not in cases:
            continue
        output_call = _parse_integer(row.get("output_call"), name="output_call")
        if output_call not in calls:
            continue
        key = (str(case_id), output_call)
        if key in audit_lookup:
            raise ValueError("duplicate recurrent tether audit")
        audit_lookup[key] = row
    if set(audit_lookup) != expected:
        raise ValueError("recurrent tether audit inventory mismatch")

    records = []
    for case_id in cases:
        for output_call in calls:
            raw = metric_lookup[(case_id, "zero", output_call)]
            corrected = metric_lookup[(case_id, corrected_policy, output_call)]
            audit = audit_lookup[(case_id, output_call)]
            raw_error = _parse_finite(raw.get("state_error"), name="raw state error")
            corrected_error = _parse_finite(
                corrected.get("state_error"), name="corrected state error"
            )
            if raw_error <= DENOMINATOR_FLOOR or corrected_error < 0.0:
                raise ValueError("state-error benefit denominator is unresolved")

            target = _parse_finite(
                audit.get("target_difference_rms"), name="target difference RMS"
            )
            previous = _parse_finite(
                audit.get("previous_difference_rms"),
                name="previous difference RMS",
            )
            requested = _parse_finite(
                audit.get("requested_change_rms"), name="requested change RMS"
            )
            phase_denominator = 2.0 * target * previous
            if phase_denominator <= DENOMINATOR_FLOOR:
                raise ValueError("target/previous phase denominator is unresolved")
            phase_cosine = (
                target * target + previous * previous - requested * requested
            ) / phase_denominator
            if phase_cosine < -1.0 - 1.0e-10 or phase_cosine > 1.0 + 1.0e-10:
                raise ValueError("target/previous phase cosine violates norm closure")
            phase_cosine = min(1.0, max(-1.0, phase_cosine))

            applied_ratio = _parse_finite(
                audit.get("applied_to_shadow_increment_ratio"),
                name="applied/shadow ratio",
            )
            correction_ratio = _parse_finite(
                corrected.get("correction_to_native_increment"),
                name="correction/native ratio",
            )
            positive_values = (
                target,
                previous,
                requested,
                applied_ratio,
                correction_ratio,
            )
            if any(value <= DENOMINATOR_FLOOR for value in positive_values):
                raise ValueError("phase log feature has an unresolved denominator")
            features = {
                "normalized_time": output_call / 30.0,
                "target_previous_cosine": phase_cosine,
                "log_target_previous_ratio": math.log(target / previous),
                "log_requested_previous_ratio": math.log(requested / previous),
                "log_applied_shadow_increment_ratio": math.log(applied_ratio),
                "log_correction_native_increment_ratio": math.log(correction_ratio),
            }
            records.append(
                RecurrentPhaseRecord(
                    population=population,
                    case_id=case_id,
                    output_call=output_call,
                    raw_state_error=raw_error,
                    corrected_state_error=corrected_error,
                    benefit=1.0 - (corrected_error / raw_error) ** 2,
                    features=features,
                )
            )
    return tuple(records)


def _phase_weights(records: Sequence[RecurrentPhaseRecord]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.case_id] += 1
    if not counts:
        raise ValueError("phase records must be nonempty")
    return np.asarray(
        [1.0 / (len(counts) * counts[record.case_id]) for record in records],
        dtype=np.float64,
    )


def fit_phase_ridge(
    records: Sequence[RecurrentPhaseRecord],
    *,
    feature_names: Sequence[str],
    ridge: float = PHASE_RIDGE,
) -> dict[str, Any]:
    """Fit one case-balanced standardized ridge model."""

    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or ridge < 0.0:
        raise ValueError("phase feature names must be unique and ridge nonnegative")
    rows = tuple(records)
    if len({row.case_id for row in rows}) < 2:
        raise ValueError("phase fitting requires at least two cases")
    x = np.asarray([[row.features[name] for name in names] for row in rows])
    y = np.asarray([row.benefit for row in rows], dtype=np.float64)
    weights = _phase_weights(rows)
    mean = np.sum(weights[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0))
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(scale <= DENOMINATOR_FLOOR)
    ):
        raise ValueError("phase feature standardization is unresolved")
    z = (x - mean) / scale
    y_mean = float(np.sum(weights * y))
    centered = y - y_mean
    gram = z.T @ (weights[:, None] * z) + ridge * np.eye(len(names))
    rhs = z.T @ (weights * centered)
    coefficients = np.linalg.solve(gram, rhs)
    return {
        "feature_names": names,
        "ridge": ridge,
        "feature_mean": mean,
        "feature_scale": scale,
        "intercept": y_mean,
        "standardized_coefficients": coefficients,
        "original_coefficients": coefficients / scale,
        "training_cases": tuple(sorted({row.case_id for row in rows})),
    }


def predict_phase_ridge(
    fit: Mapping[str, Any], records: Sequence[RecurrentPhaseRecord]
) -> np.ndarray:
    names = tuple(fit["feature_names"])
    x = np.asarray([[row.features[name] for name in names] for row in records])
    z = (x - np.asarray(fit["feature_mean"])) / np.asarray(fit["feature_scale"])
    return float(fit["intercept"]) + z @ np.asarray(fit["standardized_coefficients"])


def score_phase_predictions(
    records: Sequence[RecurrentPhaseRecord], predictions: Sequence[float]
) -> dict[str, Any]:
    rows = tuple(records)
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([row.benefit for row in rows], dtype=np.float64)
    if predicted.shape != target.shape or not np.isfinite(predicted).all():
        raise ValueError("phase predictions must be finite and inventory aligned")
    weights = _phase_weights(rows)
    baseline = float(np.sum(weights * target * target))
    error = float(np.sum(weights * (predicted - target) ** 2))
    cosine_denominator = float(
        np.sqrt(np.sum(weights * predicted**2) * np.sum(weights * target**2))
    )
    predicted_centered = predicted - np.sum(weights * predicted)
    target_centered = target - np.sum(weights * target)
    pearson_denominator = float(
        np.sqrt(
            np.sum(weights * predicted_centered**2)
            * np.sum(weights * target_centered**2)
        )
    )
    sign_mask = np.abs(target) > 1.0e-12
    actual_help = target[sign_mask] > 0.0
    predicted_help = predicted[sign_mask] > 0.0
    actual_help_count = int(np.sum(actual_help))
    actual_harm_count = int(np.sum(~actual_help))
    true_help_count = int(np.sum(actual_help & predicted_help))
    true_harm_count = int(np.sum(~actual_help & ~predicted_help))
    false_safe_count = int(np.sum(~actual_help & predicted_help))
    false_harm_count = int(np.sum(actual_help & ~predicted_help))
    help_recall = true_help_count / actual_help_count if actual_help_count else None
    harm_recall = true_harm_count / actual_harm_count if actual_harm_count else None
    return {
        "r2_vs_zero": 1.0 - error / baseline
        if baseline > DENOMINATOR_FLOOR**2
        else None,
        "r2_status": "ok" if baseline > DENOMINATOR_FLOOR**2 else "unresolved",
        "signed_cosine": (
            float(np.sum(weights * predicted * target) / cosine_denominator)
            if cosine_denominator > DENOMINATOR_FLOOR
            else None
        ),
        "signed_cosine_status": (
            "ok" if cosine_denominator > DENOMINATOR_FLOOR else "unresolved"
        ),
        "centered_pearson": (
            float(
                np.sum(weights * predicted_centered * target_centered)
                / pearson_denominator
            )
            if pearson_denominator > DENOMINATOR_FLOOR
            else None
        ),
        "centered_pearson_status": (
            "ok" if pearson_denominator > DENOMINATOR_FLOOR else "unresolved"
        ),
        "sign_accuracy": (
            float(np.mean(np.sign(predicted[sign_mask]) == np.sign(target[sign_mask])))
            if np.any(sign_mask)
            else None
        ),
        "resolved_sign_count": int(np.sum(sign_mask)),
        "actual_help_count": actual_help_count,
        "actual_harm_count": actual_harm_count,
        "predicted_help_count": int(np.sum(predicted_help)),
        "predicted_harm_count": int(np.sum(~predicted_help)),
        "true_help_count": true_help_count,
        "true_harm_count": true_harm_count,
        "false_safe_count": false_safe_count,
        "false_harm_count": false_harm_count,
        "help_recall": help_recall,
        "harm_recall": harm_recall,
        "balanced_sign_accuracy": (
            0.5 * (help_recall + harm_recall)
            if help_recall is not None and harm_recall is not None
            else None
        ),
        "case_count": len({row.case_id for row in rows}),
        "row_count": len(rows),
    }


def leave_one_case_out_phase(
    records: Sequence[RecurrentPhaseRecord], *, feature_names: Sequence[str]
) -> dict[str, Any]:
    rows = tuple(records)
    cases = tuple(sorted({row.case_id for row in rows}))
    predictions = np.empty(len(rows), dtype=np.float64)
    folds = []
    for case_id in cases:
        train = [row for row in rows if row.case_id != case_id]
        held_indices = [
            index for index, row in enumerate(rows) if row.case_id == case_id
        ]
        held = [rows[index] for index in held_indices]
        fit = fit_phase_ridge(train, feature_names=feature_names)
        held_predictions = predict_phase_ridge(fit, held)
        predictions[held_indices] = held_predictions
        folds.append(
            {
                "held_out_case": case_id,
                "standardized_coefficients": dict(
                    zip(
                        fit["feature_names"],
                        fit["standardized_coefficients"],
                        strict=True,
                    )
                ),
                "score": score_phase_predictions(held, held_predictions),
            }
        )
    signs = {}
    for name in feature_names:
        values = np.asarray([fold["standardized_coefficients"][name] for fold in folds])
        signs[name] = {
            "same_nonzero_sign": bool(
                np.all(np.abs(values) > 1.0e-12)
                and (np.all(values > 0.0) or np.all(values < 0.0))
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "predictions": predictions,
        "score": score_phase_predictions(rows, predictions),
        "folds": folds,
        "coefficient_stability": signs,
    }


ENERGY_COMPONENTS = ("density", "x_momentum", "y_momentum", "energy")
ENERGY_LOCAL_REGIONS = ("boundary", "shock", "vortex", "smooth")


def recurrent_error_energy_rows(
    call_rows: Sequence[Mapping[str, Any]],
    *,
    population: str,
    expected_cases: Sequence[str],
    corrected_policy: str,
    expected_calls: Sequence[int] = tuple(range(9, 31)),
) -> tuple[dict[str, Any], ...]:
    """Derive additive rank/component benefits from paired scalar error rows."""

    cases = tuple(expected_cases)
    calls = tuple(expected_calls)
    expected = {(case, call) for case in cases for call in calls}
    if (
        not population
        or not cases
        or len(set(cases)) != len(cases)
        or not calls
        or len(set(calls)) != len(calls)
    ):
        raise ValueError("energy population, cases, and calls must be unique")
    lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in call_rows:
        case_id = row.get("case_id")
        policy = row.get("policy")
        if case_id not in cases or policy not in {"zero", corrected_policy}:
            continue
        output_call = _parse_integer(row.get("output_call"), name="output_call")
        if output_call not in calls:
            continue
        key = (str(case_id), str(policy), output_call)
        if key in lookup:
            raise ValueError("duplicate recurrent energy metric")
        lookup[key] = row
    expected_keys = {
        (case, policy, call)
        for case, call in expected
        for policy in ("zero", corrected_policy)
    }
    if set(lookup) != expected_keys:
        raise ValueError("recurrent energy metric inventory mismatch")

    result = []
    for case_id in cases:
        for output_call in calls:
            arms = {
                "raw": lookup[(case_id, "zero", output_call)],
                "corrected": lookup[(case_id, corrected_policy, output_call)],
            }
            energy: dict[str, dict[str, float]] = {}
            maximum_component_closure = 0.0
            for arm, row in arms.items():
                full = _parse_finite(row.get("state_error"), name="state error") ** 2
                rank8 = (
                    _parse_finite(row.get("rank8_state_error"), name="rank8 error") ** 2
                )
                remainder = full - rank8
                if remainder < -1.0e-12:
                    raise ValueError("rank8 energy exceeds full energy")
                remainder = max(0.0, remainder)
                components = {
                    name: _parse_finite(
                        row.get(f"component_{name}_state_error"),
                        name=f"{name} component error",
                    )
                    ** 2
                    for name in ENERGY_COMPONENTS
                }
                component_closure = abs(full - sum(components.values()))
                if component_closure > 1.0e-12:
                    raise ValueError("component energy does not close to full energy")
                maximum_component_closure = max(
                    maximum_component_closure, component_closure
                )
                energy[arm] = {
                    "full": full,
                    "rank8": rank8,
                    "remainder": remainder,
                    **{
                        f"component_{name}": value for name, value in components.items()
                    },
                    **{
                        f"local_{name}": _parse_finite(
                            row.get(f"{name}_state_error"),
                            name=f"{name} local error",
                        )
                        ** 2
                        for name in ENERGY_LOCAL_REGIONS
                    },
                }
            benefit = {
                name: energy["raw"][name] - energy["corrected"][name]
                for name in energy["raw"]
            }
            benefit_closure = abs(
                benefit["full"] - benefit["rank8"] - benefit["remainder"]
            )
            if benefit_closure > 1.0e-12:
                raise ValueError("rank8/remainder benefit does not close")
            denominator_status = (
                "ok"
                if abs(benefit["full"]) > 1.0e-12
                else "unresolved_small_denominator"
            )
            row = {
                "population": population,
                "case_id": case_id,
                "output_call": output_call,
                **{
                    f"{arm}_{name}_energy": value
                    for arm, arm_energy in energy.items()
                    for name, value in arm_energy.items()
                },
                **{f"{name}_benefit": value for name, value in benefit.items()},
                "full_benefit_denominator": benefit["full"],
                "benefit_share_status": denominator_status,
                "rank8_benefit_share": (
                    benefit["rank8"] / benefit["full"]
                    if denominator_status == "ok"
                    else None
                ),
                "remainder_benefit_share": (
                    benefit["remainder"] / benefit["full"]
                    if denominator_status == "ok"
                    else None
                ),
                "maximum_component_energy_closure_abs": maximum_component_closure,
                "rank8_remainder_benefit_closure_abs": benefit_closure,
            }
            result.append(row)
    return tuple(result)


def _parse_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a canonical nonnegative integer")
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, str) and _INTEGER.fullmatch(value)
    ):
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be a canonical nonnegative integer")
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _parse_finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def snapshots_from_modal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cells: Sequence[tuple[int, int]],
    expected_calls: Sequence[int],
) -> tuple[ModalSnapshot, ...]:
    """Parse modal CSV rows and fail closed on an incomplete inventory."""

    canonical_cells = tuple(expected_cells)
    if not canonical_cells or len(set(canonical_cells)) != len(canonical_cells):
        raise ValueError("expected_cells must be nonempty and unique")
    canonical_calls = tuple(expected_calls)
    if not canonical_calls or len(set(canonical_calls)) != len(canonical_calls):
        raise ValueError("expected_calls must be nonempty and unique")
    if any(
        isinstance(call, bool)
        or not isinstance(call, (int, np.integer))
        or int(call) < 0
        for call in canonical_calls
    ):
        raise ValueError("expected_calls must contain nonnegative integers")
    canonical_calls = tuple(int(call) for call in canonical_calls)

    grouped: dict[
        tuple[str, str, int],
        dict[tuple[int, int], tuple[float, float, float]],
    ] = {}
    case_groups: dict[str, str] = {}
    for row in rows:
        case_id = row.get("case_id")
        group_id = row.get("group_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a nonempty string")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a nonempty string")
        if case_id in case_groups and case_groups[case_id] != group_id:
            raise ValueError("one case_id cannot belong to multiple groups")
        case_groups[case_id] = group_id
        input_call = _parse_integer(row.get("input_call"), name="input_call")
        mode = _parse_integer(row.get("mode_index"), name="mode_index")
        component = _parse_integer(row.get("component"), name="component")
        cell = (mode, component)
        key = (case_id, group_id, input_call)
        values = grouped.setdefault(key, {})
        if cell in values:
            raise ValueError(f"duplicate modal cell for {key}: {cell}")
        values[cell] = (
            _parse_finite(row.get("coarse_coordinate"), name="coarse_coordinate"),
            _parse_finite(row.get("fine_coordinate"), name="fine_coordinate"),
            _parse_finite(row.get("target_coordinate"), name="target_coordinate"),
        )

    if not grouped:
        raise ValueError("modal rows must be nonempty")
    expected_cell_set = set(canonical_cells)
    expected_call_set = set(canonical_calls)
    by_case: dict[str, set[int]] = defaultdict(set)
    snapshots: list[ModalSnapshot] = []
    for (case_id, group_id, input_call), values in sorted(grouped.items()):
        if set(values) != expected_cell_set:
            raise ValueError(
                f"modal cell inventory mismatch for {(case_id, input_call)}"
            )
        by_case[case_id].add(input_call)
        ordered = [values[cell] for cell in canonical_cells]
        snapshots.append(
            ModalSnapshot(
                case_id=case_id,
                group_id=group_id,
                input_call=input_call,
                cells=canonical_cells,
                coarse=np.asarray([value[0] for value in ordered], dtype=np.float64),
                fine=np.asarray([value[1] for value in ordered], dtype=np.float64),
                target=np.asarray([value[2] for value in ordered], dtype=np.float64),
            )
        )
    for case_id, calls in by_case.items():
        if calls != expected_call_set:
            raise ValueError(f"input-call inventory mismatch for {case_id}")
    return tuple(snapshots)


def _pair_relation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_square = float(left @ left)
    right_square = float(right @ right)
    cross = float(left @ right)
    denominator = math.sqrt(max(left_square * right_square, 0.0))
    resolved = denominator > DENOMINATOR_FLOOR**2
    return {
        "cross": cross,
        "left_square": left_square,
        "right_square": right_square,
        "denominator": denominator,
        "cosine": cross / denominator if resolved else None,
        "status": "ok" if resolved else "small_denominator",
    }


def build_geometry_records(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
) -> tuple[GeometryRecord, ...]:
    """Create all-nonconstant, SP19, and orthogonal-complement records."""

    if not snapshots:
        raise ValueError("snapshots must be nonempty")
    cells = snapshots[0].cells
    if any(snapshot.cells != cells for snapshot in snapshots):
        raise ValueError("all snapshots must use the same modal cells")
    active = set(active_cells)
    if not active or not active.issubset(set(cells)):
        raise ValueError("active_cells must be a nonempty subset of modal cells")
    masks: dict[ViewName, np.ndarray] = {
        "all_nonconstant": np.asarray([mode != 0 for mode, _ in cells]),
        "sp19": np.asarray([cell in active for cell in cells]),
        "complement": np.asarray(
            [mode != 0 and (mode, component) not in active for mode, component in cells]
        ),
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("every registered modal view must be nonempty")

    ordered = sorted(snapshots, key=lambda row: (row.case_id, row.input_call))
    previous: dict[tuple[str, ViewName], np.ndarray] = {}
    records: list[GeometryRecord] = []
    for snapshot in ordered:
        for view in VIEW_NAMES:
            mask = masks[view]
            coarse = np.asarray(snapshot.coarse[mask], dtype=np.float64)
            fine = np.asarray(snapshot.fine[mask], dtype=np.float64)
            target = np.asarray(snapshot.target[mask], dtype=np.float64)
            prior = previous.get((snapshot.case_id, view))
            if prior is None:
                persistence = {
                    "cosine": None,
                    "denominator": 0.0,
                    "status": "no_previous_call",
                }
            else:
                persistence = _pair_relation(prior, fine)
            records.append(
                GeometryRecord(
                    case_id=snapshot.case_id,
                    group_id=snapshot.group_id,
                    input_call=snapshot.input_call,
                    view=view,
                    coarse=coarse,
                    fine=fine,
                    target=target,
                    persistence_cosine=persistence["cosine"],
                    persistence_denominator=float(persistence["denominator"]),
                    persistence_status=str(persistence["status"]),
                )
            )
            previous[(snapshot.case_id, view)] = fine
    return tuple(records)


def relation_rows(records: Sequence[GeometryRecord]) -> list[dict[str, Any]]:
    """Return per-snapshot signed relations and fixed-coefficient diagnostics."""

    rows: list[dict[str, Any]] = []
    for record in records:
        coarse_fine = _pair_relation(record.coarse, record.fine)
        fine_target = _pair_relation(record.fine, record.target)
        fixed_error = record.target - FIXED_BETA * record.fine
        target_square = float(record.target @ record.target)
        fixed_square = float(fixed_error @ fixed_error)
        fine_square = float(record.fine @ record.fine)
        rows.append(
            {
                "case_id": record.case_id,
                "group_id": record.group_id,
                "input_call": record.input_call,
                "view": record.view,
                "coarse_fine_cosine": coarse_fine["cosine"],
                "coarse_fine_denominator": coarse_fine["denominator"],
                "coarse_fine_status": coarse_fine["status"],
                "fine_target_cosine": fine_target["cosine"],
                "fine_target_denominator": fine_target["denominator"],
                "fine_target_status": fine_target["status"],
                "persistence_cosine": record.persistence_cosine,
                "persistence_denominator": record.persistence_denominator,
                "persistence_status": record.persistence_status,
                "oracle_coefficient": (
                    fine_target["cross"] / fine_square
                    if fine_square > DENOMINATOR_FLOOR**2
                    else None
                ),
                "oracle_skill": (
                    fine_target["cross"] ** 2 / (fine_square * target_square)
                    if fine_square > DENOMINATOR_FLOOR**2
                    and target_square > DENOMINATOR_FLOOR**2
                    else None
                ),
                "target_square": target_square,
                "fine_square": fine_square,
                "coarse_square": float(record.coarse @ record.coarse),
                "fixed_beta_error_square": fixed_square,
                "fixed_beta_rms_ratio": (
                    math.sqrt(fixed_square / target_square)
                    if target_square > DENOMINATOR_FLOOR**2
                    else None
                ),
            }
        )
    return rows


def _case_first_mean(
    records: Sequence[GeometryRecord], getter: Any
) -> tuple[float, dict[str, float]]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(float(getter(record)))
    case_values = {
        case_id: float(np.mean(values)) for case_id, values in sorted(by_case.items())
    }
    return float(np.mean(list(case_values.values()))), case_values


def score_policy(
    records: Sequence[GeometryRecord],
    *,
    beta: float,
    persistence_threshold: float | None = None,
    policy: str,
) -> dict[str, Any]:
    """Score one fixed scalar policy with equal case and call weights."""

    if not records or not math.isfinite(beta):
        raise ValueError("records must be nonempty and beta must be finite")
    views = {record.view for record in records}
    if len(views) != 1:
        raise ValueError("score_policy requires exactly one modal view")
    if persistence_threshold is not None and not math.isfinite(persistence_threshold):
        raise ValueError("persistence_threshold must be finite")

    harmful_applied = 0
    applied_count = 0

    def corrected_square(record: GeometryRecord) -> float:
        nonlocal applied_count, harmful_applied
        use = beta != 0.0 and (
            persistence_threshold is None
            or (
                record.persistence_cosine is not None
                and record.persistence_cosine >= persistence_threshold
            )
        )
        applied_count += int(use)
        error = record.target - (beta * record.fine if use else 0.0)
        value = float(error @ error)
        if use and value > float(record.target @ record.target):
            harmful_applied += 1
        return value

    zero_sse, zero_by_case = _case_first_mean(
        records, lambda record: record.target @ record.target
    )
    corrected_sse, corrected_by_case = _case_first_mean(records, corrected_square)
    resolved = zero_sse > DENOMINATOR_FLOOR**2
    case_rows = []
    for case_id in zero_by_case:
        case_resolved = zero_by_case[case_id] > DENOMINATOR_FLOOR**2
        case_rows.append(
            {
                "case_id": case_id,
                "zero_sse": zero_by_case[case_id],
                "corrected_sse": corrected_by_case[case_id],
                "rms_ratio_vs_zero": (
                    math.sqrt(corrected_by_case[case_id] / zero_by_case[case_id])
                    if case_resolved
                    else None
                ),
                "status": "ok" if case_resolved else "small_denominator",
            }
        )
    ratios = [
        float(row["rms_ratio_vs_zero"])
        for row in case_rows
        if row["rms_ratio_vs_zero"] is not None
    ]
    return {
        "policy": policy,
        "beta": beta,
        "persistence_threshold": persistence_threshold,
        "snapshot_count": len(records),
        "case_count": len(case_rows),
        "zero_sse_case_mean": zero_sse,
        "corrected_sse_case_mean": corrected_sse,
        "skill_vs_zero": 1.0 - corrected_sse / zero_sse if resolved else None,
        "rms_ratio_vs_zero": math.sqrt(corrected_sse / zero_sse) if resolved else None,
        "status": "ok" if resolved else "small_denominator",
        "applied_count": applied_count,
        "applied_fraction": applied_count / len(records),
        "harmful_applied_count": harmful_applied,
        "harmful_applied_fraction": (
            harmful_applied / applied_count if applied_count else 0.0
        ),
        "case_win_count": sum(ratio < 1.0 for ratio in ratios),
        "maximum_case_rms_ratio": max(ratios) if ratios else None,
        "case_scores": case_rows,
    }


def fit_case_first_coefficient(records: Sequence[GeometryRecord]) -> dict[str, Any]:
    """Fit one through-origin scalar using equal case and call weights."""

    cross, _ = _case_first_mean(records, lambda record: record.fine @ record.target)
    denominator, _ = _case_first_mean(records, lambda record: record.fine @ record.fine)
    resolved = denominator > DENOMINATOR_FLOOR**2
    coefficient = cross / denominator if resolved else None
    return {
        "coefficient": coefficient,
        "cross": cross,
        "denominator": denominator,
        "feature_rms": math.sqrt(max(denominator, 0.0)),
        "status": "ok" if resolved else "small_denominator",
    }


def individual_alignment_from_statistics(
    *,
    denominator: float,
    cross: float,
    target_square: float,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """Return the scale-free alignment of one feature with a target."""

    values = (denominator, cross, target_square, denominator_floor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("individual alignment statistics must be finite")
    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive")
    if denominator < 0.0 or target_square < 0.0:
        raise ValueError("individual alignment energies must be nonnegative")
    if denominator <= denominator_floor**2:
        return {
            "status": "small_feature_denominator",
            "denominator": denominator,
            "cross": cross,
            "target_square": target_square,
            "coefficient": None,
            "alignment": None,
            "explained_fraction": None,
        }
    if target_square <= denominator_floor**2:
        return {
            "status": "small_target_denominator",
            "denominator": denominator,
            "cross": cross,
            "target_square": target_square,
            "coefficient": None,
            "alignment": None,
            "explained_fraction": None,
        }
    alignment = cross / math.sqrt(denominator * target_square)
    if abs(alignment) > 1.0 + 1.0e-10:
        raise ValueError("individual alignment violates Cauchy-Schwarz")
    alignment = min(1.0, max(-1.0, alignment))
    return {
        "status": "ok",
        "denominator": denominator,
        "cross": cross,
        "target_square": target_square,
        "coefficient": cross / denominator,
        "alignment": alignment,
        "explained_fraction": alignment * alignment,
    }


def two_feature_geometry_from_statistics(
    gram: Sequence[Sequence[float]] | np.ndarray,
    cross: Sequence[float] | np.ndarray,
    target_square: float,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """Analyze a two-feature least-squares problem without mixing raw scales."""

    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(cross, dtype=np.float64)
    target = float(target_square)
    floor = float(denominator_floor)
    if matrix.shape != (2, 2) or vector.shape != (2,):
        raise ValueError("two-feature statistics require a 2x2 Gram and length-2 cross")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(vector).all()
        or not math.isfinite(target)
    ):
        raise ValueError("two-feature statistics must be finite")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("denominator_floor must be positive")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1.0e-12:
        raise ValueError("two-feature Gram matrix is not symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) < -1.0e-12:
        raise ValueError("two-feature Gram matrix is not positive semidefinite")
    if target < 0.0:
        raise ValueError("two-feature target energy must be nonnegative")

    result: dict[str, Any] = {
        "gram": matrix.tolist(),
        "cross": vector.tolist(),
        "target_square": target,
        "gram_eigenvalues": eigenvalues.tolist(),
        "gram_symmetry_error": symmetry_error,
        "feature_c": individual_alignment_from_statistics(
            denominator=float(matrix[0, 0]),
            cross=float(vector[0]),
            target_square=target,
            denominator_floor=floor,
        ),
        "feature_f": individual_alignment_from_statistics(
            denominator=float(matrix[1, 1]),
            cross=float(vector[1]),
            target_square=target,
            denominator_floor=floor,
        ),
        "correlation_matrix": None,
        "discrepancy_correlation": None,
        "normalized_cross": None,
        "normalized_eigenvalues": None,
        "condition_number": None,
        "raw_coefficients": None,
        "standardized_coefficients": None,
        "standardized_direction": None,
        "direction_status": "unresolved",
        "joint_explained_fraction": None,
        "incremental_f_after_c": None,
        "incremental_c_after_f": None,
        "normal_equation_closure_max_abs": None,
        "normalized_equation_closure_max_abs": None,
        "explained_fraction_closure_abs": None,
    }
    if target <= floor**2:
        result["status"] = "small_target_denominator"
        return result
    diagonal = np.diag(matrix)
    if bool(np.any(diagonal <= floor**2)):
        result["status"] = "small_feature_denominator"
        return result

    scale = np.sqrt(diagonal)
    correlation = matrix / np.outer(scale, scale)
    normalized_cross = vector / (scale * math.sqrt(target))
    normalized_eigenvalues = np.linalg.eigvalsh(correlation)
    result.update(
        {
            "correlation_matrix": correlation.tolist(),
            "discrepancy_correlation": float(correlation[0, 1]),
            "normalized_cross": normalized_cross.tolist(),
            "normalized_eigenvalues": normalized_eigenvalues.tolist(),
        }
    )
    if float(normalized_eigenvalues[0]) <= floor:
        result["status"] = "rank_deficient"
        return result

    raw_coefficients = np.linalg.solve(matrix, vector)
    standardized = np.linalg.solve(correlation, normalized_cross)
    joint = float(vector @ raw_coefficients / target)
    normalized_joint = float(normalized_cross @ standardized)
    if joint < -1.0e-10 or joint > 1.0 + 1.0e-10:
        raise ValueError("joint explained fraction is outside [0, 1]")
    joint = min(1.0, max(0.0, joint))
    direction_norm = float(np.linalg.norm(standardized))
    direction = standardized / direction_norm if direction_norm > floor else None
    feature_c_fraction = result["feature_c"]["explained_fraction"]
    feature_f_fraction = result["feature_f"]["explained_fraction"]
    assert feature_c_fraction is not None and feature_f_fraction is not None
    if joint < feature_c_fraction - 1.0e-10 or joint < feature_f_fraction - 1.0e-10:
        raise ValueError("joint fit explains less than an individual feature")
    result.update(
        {
            "status": "ok",
            "condition_number": float(
                normalized_eigenvalues[-1] / normalized_eigenvalues[0]
            ),
            "raw_coefficients": raw_coefficients.tolist(),
            "standardized_coefficients": standardized.tolist(),
            "standardized_direction": (
                direction.tolist() if direction is not None else None
            ),
            "direction_status": "ok" if direction is not None else "small_direction",
            "joint_explained_fraction": joint,
            "incremental_f_after_c": max(0.0, joint - feature_c_fraction),
            "incremental_c_after_f": max(0.0, joint - feature_f_fraction),
            "normal_equation_closure_max_abs": float(
                np.max(np.abs(matrix @ raw_coefficients - vector))
            ),
            "normalized_equation_closure_max_abs": float(
                np.max(np.abs(correlation @ standardized - normalized_cross))
            ),
            "explained_fraction_closure_abs": abs(joint - normalized_joint),
        }
    )
    return result


def two_feature_geometry(records: Sequence[GeometryRecord]) -> dict[str, Any]:
    """Compute case-first two-discrepancy sufficient statistics and geometry."""

    if not records:
        raise ValueError("two-feature geometry requires nonempty records")
    views = {record.view for record in records}
    if len(views) != 1:
        raise ValueError("two-feature geometry requires exactly one modal view")
    by_case: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for record in records:
        coarse = np.asarray(record.coarse, dtype=np.float64)
        fine = np.asarray(record.fine, dtype=np.float64)
        target = np.asarray(record.target, dtype=np.float64)
        if (
            coarse.shape != fine.shape
            or coarse.shape != target.shape
            or coarse.ndim != 1
        ):
            raise ValueError(
                "two-feature modal coordinates must share one vector shape"
            )
        if (
            not np.isfinite(coarse).all()
            or not np.isfinite(fine).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError("two-feature modal coordinates must be finite")
        feature = np.stack((coarse, fine))
        by_case[record.case_id].append(
            (feature @ feature.T, feature @ target, float(target @ target))
        )
    gram = np.mean(
        [np.mean([row[0] for row in values], axis=0) for values in by_case.values()],
        axis=0,
    )
    cross = np.mean(
        [np.mean([row[1] for row in values], axis=0) for values in by_case.values()],
        axis=0,
    )
    target_square = float(
        np.mean([np.mean([row[2] for row in values]) for values in by_case.values()])
    )
    result = two_feature_geometry_from_statistics(gram, cross, target_square)
    result.update(
        {
            "case_count": len(by_case),
            "snapshot_count": len(records),
            "call_counts": {
                case_id: len(values) for case_id, values in sorted(by_case.items())
            },
            "view": next(iter(views)),
        }
    )
    return result


def standardized_direction_relation(
    first: Sequence[float] | np.ndarray | None,
    second: Sequence[float] | np.ndarray | None,
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """Compare two scale-free two-feature directions with fail-closed status."""

    if first is None or second is None:
        return {"cosine": None, "denominator": None, "status": "unresolved_direction"}
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("standardized directions must have length two")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("standardized directions must be finite")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= denominator_floor:
        return {
            "cosine": None,
            "denominator": denominator,
            "status": "small_denominator",
        }
    cosine = float(left @ right / denominator)
    if abs(cosine) > 1.0 + 1.0e-10:
        raise ValueError("standardized direction cosine violates Cauchy-Schwarz")
    return {
        "cosine": min(1.0, max(-1.0, cosine)),
        "denominator": denominator,
        "status": "ok",
    }


def fit_persistence_threshold(
    records: Sequence[GeometryRecord], *, beta: float = FIXED_BETA
) -> dict[str, Any]:
    """Fit a threshold retrospectively; ties select fewer corrections."""

    if not records or not math.isfinite(beta):
        raise ValueError("records must be nonempty and beta must be finite")
    by_case: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    case_count = len(by_case)
    weighted_delta_by_persistence: dict[float, float] = defaultdict(float)
    zero_sse = 0.0
    for case_records in by_case.values():
        weight = 1.0 / (case_count * len(case_records))
        for record in case_records:
            zero_square = float(record.target @ record.target)
            zero_sse += weight * zero_square
            if record.persistence_cosine is None:
                continue
            error = record.target - beta * record.fine
            delta = float(error @ error) - zero_square
            weighted_delta_by_persistence[float(record.persistence_cosine)] += (
                weight * delta
            )
    if not weighted_delta_by_persistence:
        return {
            "status": "no_resolved_persistence",
            "threshold": None,
            "candidate_count": 0,
            "score": None,
        }
    never = 2.0
    if zero_sse <= DENOMINATOR_FLOOR**2:
        return {
            "status": "small_denominator",
            "threshold": None,
            "candidate_count": len(weighted_delta_by_persistence) + 1,
            "score": None,
        }
    candidates = [(never, zero_sse)]
    cumulative_delta = 0.0
    for threshold in sorted(weighted_delta_by_persistence, reverse=True):
        cumulative_delta += weighted_delta_by_persistence[threshold]
        candidates.append((threshold, zero_sse + cumulative_delta))
    threshold, _ = min(candidates, key=lambda item: (item[1], -item[0]))
    score = score_policy(
        records,
        beta=beta,
        persistence_threshold=threshold,
        policy="fitted_persistence",
    )
    return {
        "status": "ok",
        "threshold": threshold,
        "candidate_count": len(weighted_delta_by_persistence) + 1,
        "score": score,
    }


def grouped_persistence_crossfit(
    records: Sequence[GeometryRecord], *, beta: float = FIXED_BETA
) -> dict[str, Any]:
    """Leave one complete group out when fitting the persistence threshold."""

    groups = sorted({record.group_id for record in records})
    if len(groups) < 2:
        raise ValueError("cross-fitting requires at least two groups")
    decisions: dict[tuple[str, int], float] = {}
    folds = []
    for group_id in groups:
        train = [record for record in records if record.group_id != group_id]
        held_out = [record for record in records if record.group_id == group_id]
        fit = fit_persistence_threshold(train, beta=beta)
        if fit["status"] != "ok" or fit["threshold"] is None:
            raise ValueError("persistence threshold was unresolved in a fold")
        threshold = float(fit["threshold"])
        score = score_policy(
            held_out,
            beta=beta,
            persistence_threshold=threshold,
            policy="crossfit_persistence",
        )
        folds.append(
            {
                "held_out_group": group_id,
                "fit_groups": [value for value in groups if value != group_id],
                "threshold": threshold,
                "fit_score": fit["score"],
                "held_out_score": score,
            }
        )
        for record in held_out:
            decisions[(record.case_id, record.input_call)] = threshold

    by_case_zero: dict[str, list[float]] = defaultdict(list)
    by_case_corrected: dict[str, list[float]] = defaultdict(list)
    for record in records:
        threshold = decisions[(record.case_id, record.input_call)]
        use = (
            record.persistence_cosine is not None
            and record.persistence_cosine >= threshold
        )
        error = record.target - (beta * record.fine if use else 0.0)
        by_case_zero[record.case_id].append(float(record.target @ record.target))
        by_case_corrected[record.case_id].append(float(error @ error))
    zero = float(np.mean([np.mean(value) for value in by_case_zero.values()]))
    corrected = float(np.mean([np.mean(value) for value in by_case_corrected.values()]))
    resolved = zero > DENOMINATOR_FLOOR**2
    return {
        "status": "ok" if resolved else "small_denominator",
        "group_count": len(groups),
        "folds": folds,
        "fold_thresholds": [float(fold["threshold"]) for fold in folds],
        "zero_sse_case_mean": zero,
        "corrected_sse_case_mean": corrected,
        "skill_vs_zero": 1.0 - corrected / zero if resolved else None,
        "rms_ratio_vs_zero": math.sqrt(corrected / zero) if resolved else None,
    }


def grouped_coefficient_crossfit(
    records: Sequence[GeometryRecord],
) -> dict[str, Any]:
    """Leave one complete group out when fitting the scalar coefficient."""

    groups = sorted({record.group_id for record in records})
    if len(groups) < 2:
        raise ValueError("cross-fitting requires at least two groups")
    fold_coefficients: dict[str, float] = {}
    folds = []
    for group_id in groups:
        train = [record for record in records if record.group_id != group_id]
        held_out = [record for record in records if record.group_id == group_id]
        fit = fit_case_first_coefficient(train)
        if fit["status"] != "ok" or fit["coefficient"] is None:
            raise ValueError("scalar coefficient was unresolved in a fold")
        coefficient = float(fit["coefficient"])
        fold_coefficients[group_id] = coefficient
        folds.append(
            {
                "held_out_group": group_id,
                "fit_groups": [value for value in groups if value != group_id],
                "coefficient": coefficient,
                "fit": fit,
                "held_out_score": score_policy(
                    held_out,
                    beta=coefficient,
                    policy="crossfit_coefficient",
                ),
            }
        )

    by_case_zero: dict[str, list[float]] = defaultdict(list)
    by_case_corrected: dict[str, list[float]] = defaultdict(list)
    for record in records:
        coefficient = fold_coefficients[record.group_id]
        error = record.target - coefficient * record.fine
        by_case_zero[record.case_id].append(float(record.target @ record.target))
        by_case_corrected[record.case_id].append(float(error @ error))
    zero = float(np.mean([np.mean(value) for value in by_case_zero.values()]))
    corrected = float(np.mean([np.mean(value) for value in by_case_corrected.values()]))
    resolved = zero > DENOMINATOR_FLOOR**2
    coefficients = np.asarray(list(fold_coefficients.values()), dtype=np.float64)
    q25, q75 = np.percentile(coefficients, (25.0, 75.0))
    median = float(np.median(coefficients))
    relative_iqr = (
        float((q75 - q25) / abs(median)) if abs(median) > DENOMINATOR_FLOOR else None
    )
    signs = np.sign(coefficients)
    return {
        "status": "ok" if resolved else "small_denominator",
        "group_count": len(groups),
        "folds": folds,
        "fold_coefficients": coefficients.tolist(),
        "median_fold_coefficient": median,
        "relative_iqr": relative_iqr,
        "same_nonzero_sign": bool(np.all(signs == signs[0]) and signs[0] != 0.0),
        "zero_sse_case_mean": zero,
        "corrected_sse_case_mean": corrected,
        "skill_vs_zero": 1.0 - corrected / zero if resolved else None,
        "rms_ratio_vs_zero": math.sqrt(corrected / zero) if resolved else None,
    }


def summarize_relations(records: Sequence[GeometryRecord]) -> dict[str, Any]:
    """Summarize signed geometry without treating nodes as samples."""

    rows = relation_rows(records)
    coarse_fine = [
        float(row["coarse_fine_cosine"])
        for row in rows
        if row["coarse_fine_cosine"] is not None
    ]
    fine_target = [
        float(row["fine_target_cosine"])
        for row in rows
        if row["fine_target_cosine"] is not None
    ]
    paired = [
        (
            float(row["coarse_fine_cosine"]),
            float(row["fine_target_cosine"]),
        )
        for row in rows
        if row["coarse_fine_cosine"] is not None
        and row["fine_target_cosine"] is not None
    ]
    if len(paired) > 1:
        left = np.asarray([value[0] for value in paired])
        right = np.asarray([value[1] for value in paired])
        left -= left.mean()
        right -= right.mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    else:
        left = right = np.asarray([], dtype=np.float64)
        denominator = 0.0
    pearson_resolved = denominator > DENOMINATOR_FLOOR**2
    persistence = [
        float(record.persistence_cosine)
        for record in records
        if record.persistence_cosine is not None
    ]
    oracle_coefficients = [
        float(row["oracle_coefficient"])
        for row in rows
        if row["oracle_coefficient"] is not None
    ]
    oracle_skills = [
        float(row["oracle_skill"]) for row in rows if row["oracle_skill"] is not None
    ]
    fit = fit_case_first_coefficient(records)
    coarse_square, _ = _case_first_mean(
        records, lambda record: record.coarse @ record.coarse
    )
    fine_square, _ = _case_first_mean(records, lambda record: record.fine @ record.fine)
    target_square, _ = _case_first_mean(
        records, lambda record: record.target @ record.target
    )
    return {
        "snapshot_count": len(records),
        "case_count": len({record.case_id for record in records}),
        "coarse_fine_median_cosine": (
            float(np.median(coarse_fine)) if coarse_fine else None
        ),
        "coarse_fine_unresolved_count": len(rows) - len(coarse_fine),
        "fine_target_median_cosine": (
            float(np.median(fine_target)) if fine_target else None
        ),
        "fine_target_unresolved_count": len(rows) - len(fine_target),
        "coarse_fine_to_fine_target_pearson": (
            float(left @ right / denominator) if pearson_resolved else None
        ),
        "coarse_fine_to_fine_target_pearson_denominator": denominator,
        "coarse_fine_to_fine_target_pearson_status": (
            "ok" if pearson_resolved else "small_denominator"
        ),
        "median_consecutive_fine_cosine": (
            float(np.median(persistence)) if persistence else None
        ),
        "persistence_resolved_count": len(persistence),
        "persistence_unresolved_count": len(records) - len(persistence),
        "coarse_square_case_mean": coarse_square,
        "fine_square_case_mean": fine_square,
        "target_square_case_mean": target_square,
        "case_first_fit": fit,
        "median_snapshot_oracle_coefficient": (
            float(np.median(oracle_coefficients)) if oracle_coefficients else None
        ),
        "median_snapshot_oracle_skill": (
            float(np.median(oracle_skills)) if oracle_skills else None
        ),
        "oracle_deployable": False,
    }


def orthogonal_partition_closure(
    records: Sequence[GeometryRecord],
) -> dict[str, Any]:
    """Verify SP19 plus complement squared energy equals all nonconstant."""

    grouped: dict[tuple[str, int], dict[str, GeometryRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.case_id, record.input_call)][record.view] = record
    maxima = {name: 0.0 for name in ("coarse", "fine", "target")}
    for key, views in grouped.items():
        if set(views) != set(VIEW_NAMES):
            raise ValueError(f"view inventory mismatch for {key}")
        for name, maximum in maxima.items():
            all_values = getattr(views["all_nonconstant"], name)
            all_energy = float(all_values @ all_values)
            partition = sum(
                float(getattr(views[view], name) @ getattr(views[view], name))
                for view in ("sp19", "complement")
            )
            maxima[name] = max(maximum, abs(all_energy - partition))
    return {
        "snapshot_count": len(grouped),
        "maximum_coarse_energy_closure_abs": maxima["coarse"],
        "maximum_fine_energy_closure_abs": maxima["fine"],
        "maximum_target_energy_closure_abs": maxima["target"],
        "status": "ok" if max(maxima.values()) <= 1.0e-12 else "failed",
    }


ModalLinearFamily = Literal[
    "fine_diagonal",
    "fine_full",
    "coarse_fine_full",
]
MODAL_LINEAR_FAMILIES: tuple[ModalLinearFamily, ...] = (
    "fine_diagonal",
    "fine_full",
    "coarse_fine_full",
)
MODAL_LINEAR_RIDGES = (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
MODAL_LINEAR_TIE_TOLERANCE = 1.0e-12
ModalAffineFamily = Literal[
    "fine_full_affine",
    "coarse_fine_full_affine",
]
MODAL_AFFINE_FAMILIES: tuple[ModalAffineFamily, ...] = (
    "fine_full_affine",
    "coarse_fine_full_affine",
)
MODAL_AFFINE_RIDGE = 1.0e-6
MODAL_AFFINE_LAST_INPUT_CALL = 29
MODAL_CAUSAL_OBSERVER_DECAYS = (0.0, 0.5, 0.8, 0.95)
MODAL_CAUSAL_OBSERVER_RIDGE = 1.0e-6
MODAL_CAUSAL_OBSERVER_FAMILY = "fine_full_causal_ema"
MODAL_PHASE_PORTRAIT_RIDGE = 1.0e-6
MODAL_PHASE_PORTRAIT_FAMILY = "fine_full_discrepancy_phase_portrait"
MODAL_EVENT_CLOCK_RIDGE = 1.0e-6
MODAL_EVENT_CLOCK_FAMILY = "fine_full_causal_discrepancy_event_clock"


def modal_snapshot_arrays(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Return one strict, ordered modal view without changing snapshot order."""

    rows = tuple(snapshots)
    cells = tuple(active_cells)
    if not rows or not cells or len(set(cells)) != len(cells):
        raise ValueError("modal snapshots and active cells must be nonempty and unique")
    source_cells = rows[0].cells
    if any(row.cells != source_cells for row in rows):
        raise ValueError("all modal snapshots must use identical cells")
    if any(cell not in source_cells for cell in cells):
        raise ValueError("active modal cells must be present in every snapshot")
    indices = np.asarray([source_cells.index(cell) for cell in cells], dtype=np.int64)
    inventory: set[tuple[str, int]] = set()
    case_groups: dict[str, str] = {}
    for row in rows:
        key = (row.case_id, row.input_call)
        if key in inventory:
            raise ValueError("duplicate modal snapshot")
        inventory.add(key)
        if row.case_id in case_groups and case_groups[row.case_id] != row.group_id:
            raise ValueError("one modal case cannot belong to multiple groups")
        case_groups[row.case_id] = row.group_id
    coarse = np.stack([row.coarse[indices] for row in rows]).astype(
        np.float64, copy=False
    )
    fine = np.stack([row.fine[indices] for row in rows]).astype(np.float64, copy=False)
    target = np.stack([row.target[indices] for row in rows]).astype(
        np.float64, copy=False
    )
    if (
        not np.isfinite(coarse).all()
        or not np.isfinite(fine).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("modal snapshot arrays must be finite")
    return {
        "snapshots": rows,
        "active_cells": cells,
        "coarse": coarse,
        "fine": fine,
        "target": target,
    }


def _modal_snapshot_weights(snapshots: Sequence[ModalSnapshot]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for row in snapshots:
        counts[row.case_id] += 1
    if not counts:
        raise ValueError("modal snapshots must be nonempty")
    return np.asarray(
        [1.0 / (len(counts) * counts[row.case_id]) for row in snapshots],
        dtype=np.float64,
    )


def _modal_feature_matrix(
    arrays: Mapping[str, Any], *, family: ModalLinearFamily
) -> np.ndarray:
    if family not in MODAL_LINEAR_FAMILIES:
        raise ValueError(f"unknown modal linear family: {family!r}")
    fine = np.asarray(arrays["fine"], dtype=np.float64)
    if family in {"fine_diagonal", "fine_full"}:
        return fine
    return np.concatenate(
        (np.asarray(arrays["coarse"], dtype=np.float64), fine), axis=1
    )


def fit_modal_linear_map(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    family: ModalLinearFamily,
    ridge: float,
) -> dict[str, Any]:
    """Fit one uncentered, case-balanced linear modal correction map."""

    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("modal linear ridge must be finite and positive")
    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    arrays = modal_snapshot_arrays(ordered, active_cells=active_cells)
    features = _modal_feature_matrix(arrays, family=family)
    target = np.asarray(arrays["target"], dtype=np.float64)
    weights = _modal_snapshot_weights(ordered)
    scale = np.sqrt(np.sum(weights[:, None] * features * features, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal linear feature RMS is unresolved")
    standardized = features / scale
    output_count = target.shape[1]
    if family == "fine_diagonal":
        if standardized.shape[1] != output_count:
            raise ValueError("diagonal modal map requires matching input/output cells")
        numerator = np.sum(weights[:, None] * standardized * target, axis=0)
        denominator = np.sum(weights[:, None] * standardized**2, axis=0) + ridge
        coefficients = np.zeros((output_count, output_count), dtype=np.float64)
        np.fill_diagonal(coefficients, numerator / denominator)
    else:
        gram = standardized.T @ (weights[:, None] * standardized)
        gram += ridge * np.eye(standardized.shape[1], dtype=np.float64)
        rhs = standardized.T @ (weights[:, None] * target)
        coefficients = np.linalg.solve(gram, rhs)
    original = coefficients / scale[:, None]
    if not np.isfinite(original).all():
        raise ValueError("modal linear coefficients must be finite")
    singular_values = np.linalg.svd(original, compute_uv=False)
    rank_tolerance = (
        max(original.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "family": family,
        "ridge": float(ridge),
        "active_cells": tuple(active_cells),
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "original_coefficients": original,
        "intercept": np.zeros(output_count, dtype=np.float64),
        "training_cases": tuple(sorted({row.case_id for row in ordered})),
        "training_groups": tuple(sorted({row.group_id for row in ordered})),
        "coefficient_frobenius_norm": float(np.linalg.norm(original)),
        "singular_values": singular_values,
        "effective_rank": int(np.sum(singular_values > rank_tolerance)),
        "effective_rank_tolerance": rank_tolerance,
    }


def predict_modal_linear_map(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> np.ndarray:
    """Apply a fitted map in original coordinate units."""

    family = str(fit.get("family"))
    if family not in MODAL_LINEAR_FAMILIES:
        raise ValueError("modal linear fit has an unknown family")
    active_cells = tuple(tuple(cell) for cell in fit["active_cells"])
    arrays = modal_snapshot_arrays(snapshots, active_cells=active_cells)
    features = _modal_feature_matrix(arrays, family=family)
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(active_cells)):
        raise ValueError("modal linear coefficient shape is inconsistent")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if intercept.shape != (len(active_cells),) or np.any(intercept != 0.0):
        raise ValueError("modal linear map must have an exact zero intercept")
    predicted = features @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("modal linear predictions must be finite")
    return predicted


def modal_causal_observer_features(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    decay: float,
) -> np.ndarray:
    """Build normalized causal EMA features from fine-grid discrepancies.

    Each case owns an independent observer state.  Normalization removes the
    cold-start amplitude bias, so a constant discrepancy remains unchanged and
    ``decay=0`` is exactly the current fine discrepancy.
    """

    if (
        not math.isfinite(decay)
        or decay < 0.0
        or decay >= 1.0
        or decay not in MODAL_CAUSAL_OBSERVER_DECAYS
    ):
        raise ValueError("modal causal-observer decay differs from the fixed inventory")
    rows = tuple(snapshots)
    arrays = modal_snapshot_arrays(rows, active_cells=active_cells)
    fine = np.asarray(arrays["fine"], dtype=np.float64)
    features = np.empty_like(fine)
    by_case: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_case[row.case_id].append(index)
    for indices in by_case.values():
        ordered = sorted(indices, key=lambda index: rows[index].input_call)
        calls = [rows[index].input_call for index in ordered]
        if calls != list(range(len(calls))):
            raise ValueError(
                "modal causal-observer calls must be contiguous and start at zero"
            )
        numerator = np.zeros(fine.shape[1], dtype=np.float64)
        normalizer = 0.0
        for index in ordered:
            numerator = decay * numerator + (1.0 - decay) * fine[index]
            normalizer = decay * normalizer + (1.0 - decay)
            if normalizer <= DENOMINATOR_FLOOR:
                raise ValueError("modal causal-observer normalization is unresolved")
            features[index] = numerator / normalizer
    if not np.isfinite(features).all():
        raise ValueError("modal causal-observer features must be finite")
    return features


def fit_modal_causal_observer(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    decay: float,
    label_filter: Any | None = None,
) -> dict[str, Any]:
    """Fit one case-balanced causal EMA-to-residual map.

    ``label_filter`` masks target labels only.  Observer features are still
    computed from each complete inference-available discrepancy prefix.
    """

    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    arrays = modal_snapshot_arrays(ordered, active_cells=active_cells)
    features = modal_causal_observer_features(
        ordered, active_cells=active_cells, decay=decay
    )
    selected = np.asarray(
        [True if label_filter is None else bool(label_filter(row)) for row in ordered],
        dtype=bool,
    )
    if not np.any(selected):
        raise ValueError("modal causal-observer fit requires at least one target label")
    selected_rows = tuple(
        row for row, keep in zip(ordered, selected, strict=True) if keep
    )
    target = np.asarray(arrays["target"], dtype=np.float64)[selected]
    design = features[selected]
    weights = _modal_snapshot_weights(selected_rows)
    scale = np.sqrt(np.sum(weights[:, None] * design * design, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal causal-observer feature RMS is unresolved")
    standardized = design / scale
    gram = standardized.T @ (weights[:, None] * standardized)
    gram += MODAL_CAUSAL_OBSERVER_RIDGE * np.eye(
        standardized.shape[1], dtype=np.float64
    )
    rhs = standardized.T @ (weights[:, None] * target)
    coefficients = np.linalg.solve(gram, rhs)
    original = coefficients / scale[:, None]
    if not np.isfinite(original).all():
        raise ValueError("modal causal-observer coefficients must be finite")
    singular_values = np.linalg.svd(original, compute_uv=False)
    rank_tolerance = (
        max(original.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "family": MODAL_CAUSAL_OBSERVER_FAMILY,
        "decay": float(decay),
        "ridge": MODAL_CAUSAL_OBSERVER_RIDGE,
        "active_cells": tuple(active_cells),
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "original_coefficients": original,
        "intercept": np.zeros(target.shape[1], dtype=np.float64),
        "training_cases": tuple(sorted({row.case_id for row in selected_rows})),
        "training_groups": tuple(sorted({row.group_id for row in selected_rows})),
        "training_label_count": len(selected_rows),
        "coefficient_frobenius_norm": float(np.linalg.norm(original)),
        "singular_values": singular_values,
        "effective_rank": int(np.sum(singular_values > rank_tolerance)),
        "effective_rank_tolerance": rank_tolerance,
    }


def predict_modal_causal_observer(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> np.ndarray:
    """Apply a fitted causal observer without reading target coordinates."""

    if (
        fit.get("family") != MODAL_CAUSAL_OBSERVER_FAMILY
        or float(fit.get("ridge", math.nan)) != MODAL_CAUSAL_OBSERVER_RIDGE
    ):
        raise ValueError("modal causal-observer fit differs from the fixed contract")
    active_cells = tuple(tuple(cell) for cell in fit["active_cells"])
    features = modal_causal_observer_features(
        snapshots,
        active_cells=active_cells,
        decay=float(fit["decay"]),
    )
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(active_cells)):
        raise ValueError("modal causal-observer coefficient shape is inconsistent")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if intercept.shape != (len(active_cells),) or np.any(intercept != 0.0):
        raise ValueError("modal causal observer must have an exact zero intercept")
    predicted = features @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("modal causal-observer predictions must be finite")
    return predicted


def modal_discrepancy_phase_portrait_features(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return current fine discrepancy and its causal backward difference."""

    rows = tuple(snapshots)
    arrays = modal_snapshot_arrays(rows, active_cells=active_cells)
    fine = np.asarray(arrays["fine"], dtype=np.float64)
    velocity = np.empty_like(fine)
    by_case: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_case[row.case_id].append(index)
    for indices in by_case.values():
        ordered = sorted(indices, key=lambda index: rows[index].input_call)
        calls = [rows[index].input_call for index in ordered]
        if calls != list(range(len(calls))):
            raise ValueError(
                "modal phase-portrait calls must be contiguous and start at zero"
            )
        previous: np.ndarray | None = None
        for index in ordered:
            current = fine[index]
            velocity[index] = 0.0 if previous is None else current - previous
            previous = current
    features = np.concatenate((fine, velocity), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("modal phase-portrait features must be finite")
    return features


def fit_modal_discrepancy_phase_portrait(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    label_filter: Any | None = None,
) -> dict[str, Any]:
    """Fit the fixed current-plus-backward-difference residual map."""

    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    arrays = modal_snapshot_arrays(ordered, active_cells=active_cells)
    features = modal_discrepancy_phase_portrait_features(
        ordered, active_cells=active_cells
    )
    selected = np.asarray(
        [True if label_filter is None else bool(label_filter(row)) for row in ordered],
        dtype=bool,
    )
    if not np.any(selected):
        raise ValueError("modal phase-portrait fit requires at least one target label")
    selected_rows = tuple(
        row for row, keep in zip(ordered, selected, strict=True) if keep
    )
    target = np.asarray(arrays["target"], dtype=np.float64)[selected]
    design = features[selected]
    weights = _modal_snapshot_weights(selected_rows)
    scale = np.sqrt(np.sum(weights[:, None] * design * design, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal phase-portrait feature RMS is unresolved")
    standardized = design / scale
    gram = standardized.T @ (weights[:, None] * standardized)
    gram += MODAL_PHASE_PORTRAIT_RIDGE * np.eye(standardized.shape[1], dtype=np.float64)
    rhs = standardized.T @ (weights[:, None] * target)
    coefficients = np.linalg.solve(gram, rhs)
    original = coefficients / scale[:, None]
    if not np.isfinite(original).all():
        raise ValueError("modal phase-portrait coefficients must be finite")
    cell_count = len(tuple(active_cells))
    singular_values = np.linalg.svd(original, compute_uv=False)
    rank_tolerance = (
        max(original.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "family": MODAL_PHASE_PORTRAIT_FAMILY,
        "ridge": MODAL_PHASE_PORTRAIT_RIDGE,
        "active_cells": tuple(active_cells),
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "original_coefficients": original,
        "current_coefficients": original[:cell_count],
        "velocity_coefficients": original[cell_count:],
        "intercept": np.zeros(target.shape[1], dtype=np.float64),
        "training_cases": tuple(sorted({row.case_id for row in selected_rows})),
        "training_groups": tuple(sorted({row.group_id for row in selected_rows})),
        "training_label_count": len(selected_rows),
        "coefficient_frobenius_norm": float(np.linalg.norm(original)),
        "singular_values": singular_values,
        "effective_rank": int(np.sum(singular_values > rank_tolerance)),
        "effective_rank_tolerance": rank_tolerance,
    }


def predict_modal_discrepancy_phase_portrait(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> np.ndarray:
    """Apply the fixed phase-portrait map without target or future inputs."""

    if (
        fit.get("family") != MODAL_PHASE_PORTRAIT_FAMILY
        or float(fit.get("ridge", math.nan)) != MODAL_PHASE_PORTRAIT_RIDGE
    ):
        raise ValueError("modal phase-portrait fit differs from the fixed contract")
    active_cells = tuple(tuple(cell) for cell in fit["active_cells"])
    features = modal_discrepancy_phase_portrait_features(
        snapshots, active_cells=active_cells
    )
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(active_cells)):
        raise ValueError("modal phase-portrait coefficient shape is inconsistent")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if intercept.shape != (len(active_cells),) or np.any(intercept != 0.0):
        raise ValueError("modal phase-portrait map must have an exact zero intercept")
    predicted = features @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("modal phase-portrait predictions must be finite")
    return predicted


def modal_causal_discrepancy_event_clock_features(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Return the fixed bounded path-length clock, current map, and velocity."""

    rows = tuple(snapshots)
    arrays = modal_snapshot_arrays(rows, active_cells=active_cells)
    fine = np.asarray(arrays["fine"], dtype=np.float64)
    velocity = np.empty_like(fine)
    event_clock = np.empty(len(rows), dtype=np.float64)
    resolved = np.empty(len(rows), dtype=bool)
    denominator = np.empty(len(rows), dtype=np.float64)
    by_case: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_case[row.case_id].append(index)
    for indices in by_case.values():
        ordered = sorted(indices, key=lambda index: rows[index].input_call)
        calls = [rows[index].input_call for index in ordered]
        if calls != list(range(len(calls))):
            raise ValueError(
                "modal event-clock calls must be contiguous and start at zero"
            )
        initial_norm = float(np.linalg.norm(fine[ordered[0]]))
        path_length = 0.0
        previous: np.ndarray | None = None
        for index in ordered:
            current = fine[index]
            difference = (
                np.zeros_like(current) if previous is None else current - previous
            )
            velocity[index] = difference
            path_length += float(np.linalg.norm(difference))
            denominator[index] = path_length + initial_norm
            resolved[index] = denominator[index] > DENOMINATOR_FLOOR
            event_clock[index] = (
                path_length / denominator[index] if resolved[index] else 0.0
            )
            previous = current
    if (
        not np.isfinite(velocity).all()
        or not np.isfinite(event_clock).all()
        or not np.isfinite(denominator).all()
    ):
        raise ValueError("modal event-clock features must be finite")
    features = np.concatenate(
        (
            fine * (1.0 - event_clock[:, None]),
            fine * event_clock[:, None],
            velocity,
        ),
        axis=1,
    )
    return {
        "features": features,
        "event_clock": event_clock,
        "velocity": velocity,
        "denominator": denominator,
        "resolved": resolved,
    }


def fit_modal_causal_discrepancy_event_clock(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    label_filter: Any | None = None,
) -> dict[str, Any]:
    """Fit the fixed bounded event-clock residual map."""

    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    arrays = modal_snapshot_arrays(ordered, active_cells=active_cells)
    feature_record = modal_causal_discrepancy_event_clock_features(
        ordered, active_cells=active_cells
    )
    if not np.all(feature_record["resolved"]):
        raise ValueError("modal event-clock denominator is unresolved")
    selected = np.asarray(
        [True if label_filter is None else bool(label_filter(row)) for row in ordered],
        dtype=bool,
    )
    if not np.any(selected):
        raise ValueError("modal event-clock fit requires at least one target label")
    selected_rows = tuple(
        row for row, keep in zip(ordered, selected, strict=True) if keep
    )
    target = np.asarray(arrays["target"], dtype=np.float64)[selected]
    design = np.asarray(feature_record["features"], dtype=np.float64)[selected]
    weights = _modal_snapshot_weights(selected_rows)
    scale = np.sqrt(np.sum(weights[:, None] * design * design, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal event-clock feature RMS is unresolved")
    standardized = design / scale
    gram = standardized.T @ (weights[:, None] * standardized)
    gram += MODAL_EVENT_CLOCK_RIDGE * np.eye(standardized.shape[1], dtype=np.float64)
    rhs = standardized.T @ (weights[:, None] * target)
    coefficients = np.linalg.solve(gram, rhs)
    original = coefficients / scale[:, None]
    if not np.isfinite(original).all():
        raise ValueError("modal event-clock coefficients must be finite")
    cell_count = len(tuple(active_cells))
    singular_values = np.linalg.svd(original, compute_uv=False)
    rank_tolerance = (
        max(original.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "family": MODAL_EVENT_CLOCK_FAMILY,
        "ridge": MODAL_EVENT_CLOCK_RIDGE,
        "active_cells": tuple(active_cells),
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "original_coefficients": original,
        "start_coefficients": original[:cell_count],
        "end_coefficients": original[cell_count : 2 * cell_count],
        "velocity_coefficients": original[2 * cell_count :],
        "intercept": np.zeros(target.shape[1], dtype=np.float64),
        "training_cases": tuple(sorted({row.case_id for row in selected_rows})),
        "training_groups": tuple(sorted({row.group_id for row in selected_rows})),
        "training_label_count": len(selected_rows),
        "coefficient_frobenius_norm": float(np.linalg.norm(original)),
        "singular_values": singular_values,
        "effective_rank": int(np.sum(singular_values > rank_tolerance)),
        "effective_rank_tolerance": rank_tolerance,
    }


def predict_modal_causal_discrepancy_event_clock(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> np.ndarray:
    """Apply the fixed event-clock map without target or future inputs."""

    if (
        fit.get("family") != MODAL_EVENT_CLOCK_FAMILY
        or float(fit.get("ridge", math.nan)) != MODAL_EVENT_CLOCK_RIDGE
    ):
        raise ValueError("modal event-clock fit differs from the fixed contract")
    active_cells = tuple(tuple(cell) for cell in fit["active_cells"])
    feature_record = modal_causal_discrepancy_event_clock_features(
        snapshots, active_cells=active_cells
    )
    if not np.all(feature_record["resolved"]):
        raise ValueError("modal event-clock denominator is unresolved")
    features = np.asarray(feature_record["features"], dtype=np.float64)
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(active_cells)):
        raise ValueError("modal event-clock coefficient shape is inconsistent")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if intercept.shape != (len(active_cells),) or np.any(intercept != 0.0):
        raise ValueError("modal event-clock map must have an exact zero intercept")
    predicted = features @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("modal event-clock predictions must be finite")
    return predicted


def _modal_affine_design(
    arrays: Mapping[str, Any],
    *,
    family: ModalAffineFamily,
) -> np.ndarray:
    if family not in MODAL_AFFINE_FAMILIES:
        raise ValueError(f"unknown modal affine family: {family!r}")
    rows = tuple(arrays["snapshots"])
    calls = []
    for row in rows:
        value = row.input_call
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("modal affine input_call must be an integer")
        if value < 0 or value > MODAL_AFFINE_LAST_INPUT_CALL:
            raise ValueError("modal affine input_call is outside the fixed horizon")
        calls.append(int(value))
    fine = np.asarray(arrays["fine"], dtype=np.float64)
    base = (
        fine
        if family == "fine_full_affine"
        else np.concatenate(
            (np.asarray(arrays["coarse"], dtype=np.float64), fine), axis=1
        )
    )
    phase = np.asarray(calls, dtype=np.float64) / MODAL_AFFINE_LAST_INPUT_CALL
    return np.concatenate(
        (base * (1.0 - phase[:, None]), base * phase[:, None]), axis=1
    )


def fit_modal_affine_phase_map(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    family: ModalAffineFamily,
) -> dict[str, Any]:
    """Fit the fixed zero-intercept affine-in-call modal correction map."""

    ordered = tuple(sorted(snapshots, key=lambda row: (row.case_id, row.input_call)))
    arrays = modal_snapshot_arrays(ordered, active_cells=active_cells)
    features = _modal_affine_design(arrays, family=family)
    target = np.asarray(arrays["target"], dtype=np.float64)
    weights = _modal_snapshot_weights(ordered)
    scale = np.sqrt(np.sum(weights[:, None] * features * features, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal affine feature RMS is unresolved")
    standardized = features / scale
    gram = standardized.T @ (weights[:, None] * standardized)
    gram += MODAL_AFFINE_RIDGE * np.eye(standardized.shape[1], dtype=np.float64)
    rhs = standardized.T @ (weights[:, None] * target)
    coefficients = np.linalg.solve(gram, rhs)
    original = coefficients / scale[:, None]
    if not np.isfinite(original).all():
        raise ValueError("modal affine coefficients must be finite")
    base_feature_count = original.shape[0] // 2
    if 2 * base_feature_count != original.shape[0]:
        raise AssertionError("modal affine endpoint blocks do not close")
    singular_values = np.linalg.svd(original, compute_uv=False)
    rank_tolerance = (
        max(original.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "family": family,
        "ridge": MODAL_AFFINE_RIDGE,
        "last_input_call": MODAL_AFFINE_LAST_INPUT_CALL,
        "active_cells": tuple(active_cells),
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "original_coefficients": original,
        "start_coefficients": original[:base_feature_count],
        "end_coefficients": original[base_feature_count:],
        "base_feature_count": base_feature_count,
        "intercept": np.zeros(target.shape[1], dtype=np.float64),
        "training_cases": tuple(sorted({row.case_id for row in ordered})),
        "training_groups": tuple(sorted({row.group_id for row in ordered})),
        "coefficient_frobenius_norm": float(np.linalg.norm(original)),
        "singular_values": singular_values,
        "effective_rank": int(np.sum(singular_values > rank_tolerance)),
        "effective_rank_tolerance": rank_tolerance,
    }


def predict_modal_affine_phase_map(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> np.ndarray:
    """Apply the fixed affine-in-call map without using a target coordinate."""

    family = str(fit.get("family"))
    if family not in MODAL_AFFINE_FAMILIES:
        raise ValueError("modal affine fit has an unknown family")
    if (
        float(fit.get("ridge", math.nan)) != MODAL_AFFINE_RIDGE
        or int(fit.get("last_input_call", -1)) != MODAL_AFFINE_LAST_INPUT_CALL
    ):
        raise ValueError("modal affine fit differs from the fixed phase contract")
    active_cells = tuple(tuple(cell) for cell in fit["active_cells"])
    arrays = modal_snapshot_arrays(snapshots, active_cells=active_cells)
    features = _modal_affine_design(arrays, family=family)
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(active_cells)):
        raise ValueError("modal affine coefficient shape is inconsistent")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if intercept.shape != (len(active_cells),) or np.any(intercept != 0.0):
        raise ValueError("modal affine map must have an exact zero intercept")
    predicted = features @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("modal affine predictions must be finite")
    return predicted


def score_modal_linear_predictions(
    snapshots: Sequence[ModalSnapshot],
    predictions: Sequence[Sequence[float]] | np.ndarray,
    *,
    active_cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Score aligned modal predictions with equal calls then equal cases."""

    rows = tuple(snapshots)
    arrays = modal_snapshot_arrays(rows, active_cells=active_cells)
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(arrays["target"], dtype=np.float64)
    if predicted.shape != target.shape or not np.isfinite(predicted).all():
        raise ValueError("modal linear predictions must be finite and aligned")
    order = np.asarray(
        sorted(
            range(len(rows)),
            key=lambda index: (rows[index].case_id, rows[index].input_call),
        )
    )
    ordered_rows = tuple(rows[index] for index in order)
    target = target[order]
    predicted = predicted[order]
    weights = _modal_snapshot_weights(ordered_rows)
    zero_energy = np.sum(target * target, axis=1)
    prediction_energy = np.sum(predicted * predicted, axis=1)
    cross = np.sum(target * predicted, axis=1)
    corrected_energy = np.sum((target - predicted) ** 2, axis=1)
    benefit = zero_energy - corrected_energy
    identity = 2.0 * cross - prediction_energy
    maximum_identity_error = float(np.max(np.abs(benefit - identity)))
    if maximum_identity_error > MODAL_LINEAR_TIE_TOLERANCE:
        raise ValueError("modal linear benefit identity does not close")

    zero = float(np.sum(weights * zero_energy))
    corrected = float(np.sum(weights * corrected_energy))
    resolved = zero > DENOMINATOR_FLOOR**2
    cosine_denominator = np.sqrt(np.maximum(zero_energy * prediction_energy, 0.0))
    cosine_resolved = cosine_denominator > DENOMINATOR_FLOOR**2
    cosines = np.divide(
        cross,
        cosine_denominator,
        out=np.full_like(cross, np.nan),
        where=cosine_resolved,
    )

    case_scores = []
    for case_id in sorted({row.case_id for row in ordered_rows}):
        selected = np.asarray([row.case_id == case_id for row in ordered_rows])
        case_zero = float(np.mean(zero_energy[selected]))
        case_corrected = float(np.mean(corrected_energy[selected]))
        case_resolved = case_zero > DENOMINATOR_FLOOR**2
        case_scores.append(
            {
                "case_id": case_id,
                "group_id": next(
                    row.group_id for row in ordered_rows if row.case_id == case_id
                ),
                "zero_sse": case_zero,
                "corrected_sse": case_corrected,
                "skill_vs_zero": (
                    1.0 - case_corrected / case_zero if case_resolved else None
                ),
                "rms_ratio_vs_zero": (
                    math.sqrt(case_corrected / case_zero) if case_resolved else None
                ),
                "status": "ok" if case_resolved else "small_denominator",
            }
        )
    group_scores = []
    for group_id in sorted({row.group_id for row in ordered_rows}):
        selected = [row for row in case_scores if row["group_id"] == group_id]
        group_zero = float(np.mean([row["zero_sse"] for row in selected]))
        group_corrected = float(np.mean([row["corrected_sse"] for row in selected]))
        group_resolved = group_zero > DENOMINATOR_FLOOR**2
        group_scores.append(
            {
                "group_id": group_id,
                "case_count": len(selected),
                "zero_sse": group_zero,
                "corrected_sse": group_corrected,
                "skill_vs_zero": (
                    1.0 - group_corrected / group_zero if group_resolved else None
                ),
                "rms_ratio_vs_zero": (
                    math.sqrt(group_corrected / group_zero) if group_resolved else None
                ),
                "status": "ok" if group_resolved else "small_denominator",
            }
        )
    ratios = [
        float(row["rms_ratio_vs_zero"])
        for row in case_scores
        if row["rms_ratio_vs_zero"] is not None
    ]
    return {
        "status": "ok" if resolved else "small_denominator",
        "snapshot_count": len(rows),
        "case_count": len(case_scores),
        "group_count": len(group_scores),
        "zero_sse_case_mean": zero,
        "corrected_sse_case_mean": corrected,
        "skill_vs_zero": 1.0 - corrected / zero if resolved else None,
        "rms_ratio_vs_zero": math.sqrt(corrected / zero) if resolved else None,
        "case_win_count": sum(ratio < 1.0 for ratio in ratios),
        "group_win_count": sum(
            row["rms_ratio_vs_zero"] is not None
            and float(row["rms_ratio_vs_zero"]) < 1.0
            for row in group_scores
        ),
        "maximum_case_rms_ratio": max(ratios) if ratios else None,
        "helpful_row_count": int(np.sum(benefit > 0.0)),
        "harmful_row_count": int(np.sum(benefit < 0.0)),
        "zero_benefit_row_count": int(np.sum(benefit == 0.0)),
        "harmful_row_fraction": float(np.mean(benefit < 0.0)),
        "median_signed_cosine": (
            float(np.nanmedian(cosines)) if np.any(cosine_resolved) else None
        ),
        "signed_cosine_resolved_count": int(np.sum(cosine_resolved)),
        "signed_cosine_unresolved_count": int(np.sum(~cosine_resolved)),
        "maximum_benefit_identity_abs": maximum_identity_error,
        "case_scores": case_scores,
        "group_scores": group_scores,
    }


def grouped_modal_linear_crossfit(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    family: ModalLinearFamily,
    ridge: float,
) -> dict[str, Any]:
    """Leave one complete strength group out for one fixed map family/ridge."""

    rows = tuple(snapshots)
    groups = tuple(sorted({row.group_id for row in rows}))
    if len(groups) < 2:
        raise ValueError("modal linear cross-fitting requires at least two groups")
    predictions = np.empty((len(rows), len(tuple(active_cells))), dtype=np.float64)
    folds = []
    fits = []
    for held_group in groups:
        train = [row for row in rows if row.group_id != held_group]
        indices = [
            index for index, row in enumerate(rows) if row.group_id == held_group
        ]
        held = [rows[index] for index in indices]
        fit = fit_modal_linear_map(
            train,
            active_cells=active_cells,
            family=family,
            ridge=ridge,
        )
        held_predictions = predict_modal_linear_map(fit, held)
        predictions[indices] = held_predictions
        fits.append(fit)
        folds.append(
            {
                "held_out_group": held_group,
                "fit_groups": tuple(group for group in groups if group != held_group),
                "held_out_score": score_modal_linear_predictions(
                    held,
                    held_predictions,
                    active_cells=active_cells,
                ),
                "coefficient_frobenius_norm": fit["coefficient_frobenius_norm"],
                "effective_rank": fit["effective_rank"],
            }
        )
    return {
        "family": family,
        "ridge": float(ridge),
        "predictions": predictions,
        "score": score_modal_linear_predictions(
            rows, predictions, active_cells=active_cells
        ),
        "folds": folds,
        "fits": tuple(fits),
    }


def _modal_candidate_is_preferred(
    candidate: Mapping[str, Any], best: Mapping[str, Any] | None
) -> bool:
    if best is None:
        return True
    candidate_skill = float(candidate["skill_vs_zero"])
    best_skill = float(best["skill_vs_zero"])
    if candidate_skill > best_skill + MODAL_LINEAR_TIE_TOLERANCE:
        return True
    if abs(candidate_skill - best_skill) > MODAL_LINEAR_TIE_TOLERANCE:
        return False
    candidate_rank = MODAL_LINEAR_FAMILIES.index(candidate["family"])
    best_rank = MODAL_LINEAR_FAMILIES.index(best["family"])
    if candidate_rank != best_rank:
        return candidate_rank < best_rank
    return float(candidate["ridge"]) > float(best["ridge"])


def select_grouped_modal_linear_map(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    families: Sequence[ModalLinearFamily] = MODAL_LINEAR_FAMILIES,
    ridges: Sequence[float] = MODAL_LINEAR_RIDGES,
) -> dict[str, Any]:
    """Select a family/ridge from group-held-out predictions only."""

    family_values = tuple(families)
    ridge_values = tuple(float(value) for value in ridges)
    if (
        not family_values
        or len(set(family_values)) != len(family_values)
        or any(value not in MODAL_LINEAR_FAMILIES for value in family_values)
        or not ridge_values
        or len(set(ridge_values)) != len(ridge_values)
        or any(not math.isfinite(value) or value <= 0.0 for value in ridge_values)
    ):
        raise ValueError("modal linear candidate inventory is invalid")
    candidates = []
    selected: dict[str, Any] | None = None
    selected_crossfit: dict[str, Any] | None = None
    for family in family_values:
        for ridge in ridge_values:
            crossfit = grouped_modal_linear_crossfit(
                snapshots,
                active_cells=active_cells,
                family=family,
                ridge=ridge,
            )
            score = crossfit["score"]
            if score["status"] != "ok" or score["skill_vs_zero"] is None:
                raise ValueError("modal linear candidate score is unresolved")
            candidate = {
                "family": family,
                "ridge": ridge,
                "skill_vs_zero": float(score["skill_vs_zero"]),
                "rms_ratio_vs_zero": float(score["rms_ratio_vs_zero"]),
                "case_win_count": int(score["case_win_count"]),
                "group_win_count": int(score["group_win_count"]),
                "harmful_row_count": int(score["harmful_row_count"]),
            }
            candidates.append(candidate)
            if _modal_candidate_is_preferred(candidate, selected):
                selected = candidate
                selected_crossfit = crossfit
    if selected is None or selected_crossfit is None:  # pragma: no cover
        raise AssertionError("modal linear selection produced no candidate")
    return {
        "selected": selected,
        "candidates": candidates,
        "selected_crossfit": selected_crossfit,
    }


def nested_grouped_modal_linear_map(
    snapshots: Sequence[ModalSnapshot],
    *,
    active_cells: Sequence[tuple[int, int]],
    families: Sequence[ModalLinearFamily] = MODAL_LINEAR_FAMILIES,
    ridges: Sequence[float] = MODAL_LINEAR_RIDGES,
) -> dict[str, Any]:
    """Nest candidate selection inside each outer strength-held-out fold."""

    rows = tuple(snapshots)
    groups = tuple(sorted({row.group_id for row in rows}))
    if len(groups) < 3:
        raise ValueError("nested modal linear selection requires at least three groups")
    predictions = np.empty((len(rows), len(tuple(active_cells))), dtype=np.float64)
    folds = []
    for held_group in groups:
        train = [row for row in rows if row.group_id != held_group]
        indices = [
            index for index, row in enumerate(rows) if row.group_id == held_group
        ]
        held = [rows[index] for index in indices]
        inner = select_grouped_modal_linear_map(
            train,
            active_cells=active_cells,
            families=families,
            ridges=ridges,
        )
        selected = inner["selected"]
        fit = fit_modal_linear_map(
            train,
            active_cells=active_cells,
            family=selected["family"],
            ridge=float(selected["ridge"]),
        )
        held_predictions = predict_modal_linear_map(fit, held)
        predictions[indices] = held_predictions
        folds.append(
            {
                "held_out_group": held_group,
                "fit_groups": tuple(group for group in groups if group != held_group),
                "selected": selected,
                "inner_candidates": inner["candidates"],
                "held_out_score": score_modal_linear_predictions(
                    held,
                    held_predictions,
                    active_cells=active_cells,
                ),
            }
        )
    return {
        "predictions": predictions,
        "score": score_modal_linear_predictions(
            rows, predictions, active_cells=active_cells
        ),
        "folds": folds,
    }


def modal_linear_map_stability(
    fits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure orientation and scale stability in original coordinate units."""

    rows = tuple(fits)
    if len(rows) < 2:
        raise ValueError("modal map stability requires at least two fits")
    family = rows[0]["family"]
    ridge = float(rows[0]["ridge"])
    decay = rows[0].get("decay")
    matrices = []
    for fit in rows:
        if (
            fit["family"] != family
            or float(fit["ridge"]) != ridge
            or fit.get("decay") != decay
        ):
            raise ValueError("modal map stability requires one family and ridge")
        matrix = np.asarray(fit["original_coefficients"], dtype=np.float64)
        if matrices and matrix.shape != matrices[0].shape:
            raise ValueError("modal map stability coefficient shapes differ")
        matrices.append(matrix)
    norms = np.asarray([np.linalg.norm(matrix) for matrix in matrices])
    if np.any(norms <= DENOMINATOR_FLOOR) or not np.isfinite(norms).all():
        raise ValueError("modal map stability norm is unresolved")
    cosines = [
        float(np.sum(matrices[left] * matrices[right]) / (norms[left] * norms[right]))
        for left in range(len(matrices))
        for right in range(left + 1, len(matrices))
    ]
    result = {
        "family": family,
        "ridge": ridge,
        "fit_count": len(rows),
        "minimum_pairwise_frobenius_cosine": min(cosines),
        "median_pairwise_frobenius_cosine": float(np.median(cosines)),
        "minimum_frobenius_norm": float(np.min(norms)),
        "maximum_frobenius_norm": float(np.max(norms)),
        "maximum_to_minimum_norm_ratio": float(np.max(norms) / np.min(norms)),
    }
    if decay is not None:
        result["decay"] = float(decay)
    return result


def modal_affine_map_stability(
    fits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whole-map and endpoint-map stability for affine phase fits."""

    rows = tuple(fits)
    if len(rows) < 2:
        raise ValueError("modal affine stability requires at least two fits")
    if any(str(fit.get("family")) not in MODAL_AFFINE_FAMILIES for fit in rows):
        raise ValueError("modal affine stability received an unknown family")
    whole = modal_linear_map_stability(rows)

    def endpoint(name: str) -> dict[str, Any]:
        derived = []
        for fit in rows:
            derived.append(
                {
                    "family": f"{fit['family']}::{name}",
                    "ridge": float(fit["ridge"]),
                    "original_coefficients": np.asarray(
                        fit[f"{name}_coefficients"], dtype=np.float64
                    ),
                }
            )
        return modal_linear_map_stability(derived)

    return {
        "fit_count": len(rows),
        "whole": whole,
        "start": endpoint("start"),
        "end": endpoint("end"),
    }


def cap_modal_predictions(
    predictions: Sequence[Sequence[float]] | np.ndarray,
    native_increment_rms: Sequence[float] | np.ndarray,
    *,
    total_volume: float = 2.0,
    maximum_relative_norm: float = 0.05,
) -> dict[str, Any]:
    """Apply the maintained target-free relative norm cap in modal units."""

    values = np.asarray(predictions, dtype=np.float64)
    native = np.asarray(native_increment_rms, dtype=np.float64)
    if values.ndim != 2 or native.shape != (values.shape[0],):
        raise ValueError("modal cap predictions and native RMS must align")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(native).all()
        or not math.isfinite(total_volume)
        or total_volume <= 0.0
        or not math.isfinite(maximum_relative_norm)
        or maximum_relative_norm <= 0.0
    ):
        raise ValueError("modal cap inputs must be finite and positive")
    modal_norm = np.linalg.norm(values, axis=1)
    raw_rms = modal_norm / math.sqrt(total_volume)
    resolved = native > DENOMINATOR_FLOOR
    raw_ratio = np.divide(
        raw_rms,
        native,
        out=np.full_like(raw_rms, np.nan),
        where=resolved,
    )
    scale = np.zeros_like(raw_rms)
    scale[resolved] = np.minimum(
        1.0,
        maximum_relative_norm / np.maximum(raw_ratio[resolved], DENOMINATOR_FLOOR),
    )
    # A zero prediction is already safe and should remain an exact identity.
    scale[resolved & (modal_norm == 0.0)] = 1.0
    capped = values * scale[:, None]
    capped_rms = np.linalg.norm(capped, axis=1) / math.sqrt(total_volume)
    capped_ratio = np.divide(
        capped_rms,
        native,
        out=np.full_like(capped_rms, np.nan),
        where=resolved,
    )
    maximum_violation = (
        float(np.max(np.maximum(capped_ratio[resolved] - maximum_relative_norm, 0.0)))
        if np.any(resolved)
        else 0.0
    )
    return {
        "predictions": capped,
        "scale": scale,
        "raw_correction_rms": raw_rms,
        "correction_rms": capped_rms,
        "raw_correction_to_native_increment": raw_ratio,
        "correction_to_native_increment": capped_ratio,
        "resolved": resolved,
        "status": np.where(resolved, "ok", "unresolved_native_increment"),
        "cap_active": resolved & (scale < 1.0),
        "maximum_relative_norm": maximum_relative_norm,
        "total_volume": total_volume,
        "maximum_cap_violation": maximum_violation,
    }


def modal_linear_fit_diagnostics(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> dict[str, Any]:
    """Report conditioning, cancellation, and float32 feature sensitivity."""

    family = str(fit.get("family"))
    if family not in MODAL_LINEAR_FAMILIES:
        raise ValueError("modal linear diagnostic fit has an unknown family")
    arrays = modal_snapshot_arrays(
        snapshots,
        active_cells=tuple(tuple(cell) for cell in fit["active_cells"]),
    )
    features = _modal_feature_matrix(arrays, family=family)
    scale = np.asarray(fit["feature_scale"], dtype=np.float64)
    if scale.shape != (features.shape[1],) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal linear diagnostic feature scale is unresolved")
    standardized = features / scale
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > DENOMINATOR_FLOOR
        else None
    )
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    prediction = features @ coefficients
    rounded = features.astype(np.float32).astype(np.float64) @ coefficients
    perturbation = np.linalg.norm(rounded - prediction, axis=1)
    denominator = np.linalg.norm(prediction, axis=1)
    relative = np.divide(
        perturbation,
        denominator,
        out=np.full_like(perturbation, np.nan),
        where=denominator > DENOMINATOR_FLOOR,
    )
    resolved = np.isfinite(relative)
    result: dict[str, Any] = {
        "family": family,
        "ridge": float(fit["ridge"]),
        "standardized_design_singular_values": singular_values,
        "standardized_design_condition_number": condition,
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
        "coefficient_spectral_norm": float(np.linalg.norm(coefficients, ord=2)),
        "float32_prediction_relative_change": relative,
        "float32_prediction_resolved_count": int(np.sum(resolved)),
        "float32_prediction_unresolved_count": int(np.sum(~resolved)),
        "maximum_float32_prediction_relative_change": (
            float(np.max(relative[resolved])) if np.any(resolved) else None
        ),
    }
    if family == "coarse_fine_full":
        cells = len(tuple(fit["active_cells"]))
        coarse = np.asarray(arrays["coarse"], dtype=np.float64) @ coefficients[:cells]
        fine = np.asarray(arrays["fine"], dtype=np.float64) @ coefficients[cells:]
        coarse_norm = np.linalg.norm(coarse, axis=1)
        fine_norm = np.linalg.norm(fine, axis=1)
        combined_norm = np.linalg.norm(coarse + fine, axis=1)
        cosine_denominator = coarse_norm * fine_norm
        cosine = np.divide(
            np.sum(coarse * fine, axis=1),
            cosine_denominator,
            out=np.full_like(coarse_norm, np.nan),
            where=cosine_denominator > DENOMINATOR_FLOOR**2,
        )
        cancellation = np.divide(
            coarse_norm + fine_norm,
            combined_norm,
            out=np.full_like(combined_norm, np.nan),
            where=combined_norm > DENOMINATOR_FLOOR,
        )
        result.update(
            {
                "coarse_block_norm": coarse_norm,
                "fine_block_norm": fine_norm,
                "combined_norm": combined_norm,
                "block_cosine": cosine,
                "cancellation_amplification": cancellation,
                "cancellation_resolved_count": int(np.sum(np.isfinite(cancellation))),
                "cancellation_unresolved_count": int(
                    np.sum(~np.isfinite(cancellation))
                ),
                "median_cancellation_amplification": (
                    float(np.nanmedian(cancellation))
                    if np.any(np.isfinite(cancellation))
                    else None
                ),
                "maximum_cancellation_amplification": (
                    float(np.nanmax(cancellation))
                    if np.any(np.isfinite(cancellation))
                    else None
                ),
            }
        )
    return result


def modal_affine_fit_diagnostics(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> dict[str, Any]:
    """Report conditioning, endpoint geometry, and float32 feature sensitivity."""

    family = str(fit.get("family"))
    if family not in MODAL_AFFINE_FAMILIES:
        raise ValueError("modal affine diagnostic fit has an unknown family")
    if (
        float(fit.get("ridge", math.nan)) != MODAL_AFFINE_RIDGE
        or int(fit.get("last_input_call", -1)) != MODAL_AFFINE_LAST_INPUT_CALL
    ):
        raise ValueError("modal affine diagnostic fit differs from the fixed contract")
    arrays = modal_snapshot_arrays(
        snapshots,
        active_cells=tuple(tuple(cell) for cell in fit["active_cells"]),
    )
    features = _modal_affine_design(arrays, family=family)
    scale = np.asarray(fit["feature_scale"], dtype=np.float64)
    if scale.shape != (features.shape[1],) or np.any(scale <= DENOMINATOR_FLOOR):
        raise ValueError("modal affine diagnostic feature scale is unresolved")
    standardized = features / scale
    design_singular_values = np.linalg.svd(standardized, compute_uv=False)
    design_condition = (
        float(design_singular_values[0] / design_singular_values[-1])
        if design_singular_values[-1] > DENOMINATOR_FLOOR
        else None
    )
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    if coefficients.shape != (features.shape[1], len(tuple(fit["active_cells"]))):
        raise ValueError("modal affine diagnostic coefficient shape differs")
    prediction = features @ coefficients
    rounded_prediction = features.astype(np.float32).astype(np.float64) @ coefficients
    perturbation = np.linalg.norm(rounded_prediction - prediction, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    relative = np.divide(
        perturbation,
        prediction_norm,
        out=np.full_like(perturbation, np.nan),
        where=prediction_norm > DENOMINATOR_FLOOR,
    )
    resolved = np.isfinite(relative)
    start = np.asarray(fit["start_coefficients"], dtype=np.float64)
    end = np.asarray(fit["end_coefficients"], dtype=np.float64)
    if start.shape != end.shape or start.shape[0] * 2 != coefficients.shape[0]:
        raise ValueError("modal affine endpoint coefficient shapes differ")
    start_norm = float(np.linalg.norm(start))
    end_norm = float(np.linalg.norm(end))
    endpoint_denominator = start_norm * end_norm
    result: dict[str, Any] = {
        "family": family,
        "ridge": float(fit["ridge"]),
        "standardized_design_singular_values": design_singular_values,
        "standardized_design_condition_number": design_condition,
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
        "coefficient_spectral_norm": float(np.linalg.norm(coefficients, ord=2)),
        "start_coefficient_frobenius_norm": start_norm,
        "end_coefficient_frobenius_norm": end_norm,
        "end_to_start_norm_ratio": (
            end_norm / start_norm if start_norm > DENOMINATOR_FLOOR else None
        ),
        "start_end_coefficient_cosine": (
            float(np.sum(start * end) / endpoint_denominator)
            if endpoint_denominator > DENOMINATOR_FLOOR**2
            else None
        ),
        "float32_prediction_relative_change": relative,
        "float32_prediction_resolved_count": int(np.sum(resolved)),
        "float32_prediction_unresolved_count": int(np.sum(~resolved)),
        "maximum_float32_prediction_relative_change": (
            float(np.max(relative[resolved])) if np.any(resolved) else None
        ),
    }
    if family == "coarse_fine_full_affine":
        cells = len(tuple(fit["active_cells"]))
        calls = np.asarray(
            [row.input_call for row in arrays["snapshots"]], dtype=np.float64
        )
        phase = calls / MODAL_AFFINE_LAST_INPUT_CALL
        effective = (1.0 - phase[:, None, None]) * start[None, :, :] + phase[
            :, None, None
        ] * end[None, :, :]
        coarse = np.einsum(
            "ni,nio->no",
            np.asarray(arrays["coarse"], dtype=np.float64),
            effective[:, :cells],
            optimize=True,
        )
        fine = np.einsum(
            "ni,nio->no",
            np.asarray(arrays["fine"], dtype=np.float64),
            effective[:, cells:],
            optimize=True,
        )
        coarse_norm = np.linalg.norm(coarse, axis=1)
        fine_norm = np.linalg.norm(fine, axis=1)
        combined_norm = np.linalg.norm(coarse + fine, axis=1)
        denominator = coarse_norm * fine_norm
        cosine = np.divide(
            np.sum(coarse * fine, axis=1),
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > DENOMINATOR_FLOOR**2,
        )
        amplification_denominator = combined_norm
        amplification = np.divide(
            coarse_norm + fine_norm,
            amplification_denominator,
            out=np.full_like(combined_norm, np.nan),
            where=amplification_denominator > DENOMINATOR_FLOOR,
        )
        cancellation_resolved = np.isfinite(amplification)
        result.update(
            {
                "coarse_block_norm": coarse_norm,
                "fine_block_norm": fine_norm,
                "combined_norm": combined_norm,
                "block_cosine": cosine,
                "cancellation_amplification": amplification,
                "cancellation_resolved_count": int(np.sum(cancellation_resolved)),
                "cancellation_unresolved_count": int(np.sum(~cancellation_resolved)),
                "median_cancellation_amplification": (
                    float(np.median(amplification[cancellation_resolved]))
                    if np.any(cancellation_resolved)
                    else None
                ),
                "maximum_cancellation_amplification": (
                    float(np.max(amplification[cancellation_resolved]))
                    if np.any(cancellation_resolved)
                    else None
                ),
            }
        )
    return result
