#!/usr/bin/env python3
"""Nested cross-fit an Euler1D state-conditioned discrepancy coefficient map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    json_safe_with_paths,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)

WORKING_ID = "W26-L5-P6-RFB19-A36-EULER1D-STATE-CONDITIONED-NESTED-CROSSFIT"
SCHEMA = "euler1d_state_conditioned_nested_crossfit_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_euler1d_state_conditioned_crossfit.py",
    "tests/time_dependent_no/test_euler1d_cross_resolution_correction.py",
)
EXPECTED_HASHES = {
    "a35_result": "47b5aa363c32c43379cc76f9fb8df9858b065f19616babd90674089c6876d86b",
    "family_contract": (
        "bb17c9e953e44f3a484f406e2858a1f40e89e242745370e2529a6136720ccddb"
    ),
    "metadata_carrier": (
        "a47f8c728a012bc84cee92274226588317fa30037154a39fc9652c2335b188e4"
    ),
    "sampler_source": (
        "6da99e27a29706cbe822e2d61688790dd6ccc48b606d55b23b6c45bd0a8705ce"
    ),
}
EXPECTED_PAYLOAD_HASHES = {
    "a35_result": "bb05d4469f3ce31a35a64c61866b63eb1be6767c23881978ea10d74c1a8ed7ef"
}
EXPECTED_CASE_NUMBERS = (
    17,
    32,
    45,
    104,
    108,
    113,
    123,
    141,
    174,
    259,
    321,
    370,
    409,
    440,
    479,
    507,
)
EXPECTED_CASE_IDS = tuple(f"case_{case_id:03d}" for case_id in EXPECTED_CASE_NUMBERS)
PHASE_SCOPES = ("calls_0_49", "calls_50_99")
DESCRIPTOR_NAMES = ("log_density_ratio", "log_pressure_ratio", "velocity_jump")
RIDGES = (1.0e-4, 1.0e-2, 1.0, 100.0)
DENOMINATOR_FLOOR = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-18
RIDGE_TIE_TOLERANCE = 1.0e-12
MINIMUM_COMBINED_SKILL = 0.01
MINIMUM_SKILL_INCREMENT = 0.02
MINIMUM_SUPPORTED_CASES = 12
MINIMUM_SUPPORTED_WINS = 10
MINIMUM_TOTAL_WINS = 12
MAXIMUM_CASE_RMS_RATIO = 1.05
MAXIMUM_COEFFICIENT_MAGNITUDE = 3.0
MINIMUM_DOMINANT_RIDGE_COUNT = 8
MINIMUM_SLOPE_COSINE = 0.8
MAXIMUM_SLOPE_NORM_RATIO = 3.0
EXPECTED_METADATA_SHA256 = (
    "ea4e528a559557cf602435810a5d5ca0c19fac873bba4d8969b8a137f988bae0"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _with_json_payload_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the complete audit tree before canonical payload hashing."""

    safe = json_safe_with_paths(payload)
    if not isinstance(safe, dict):
        raise TypeError("JSON-safe payload must remain an object")
    return with_payload_sha256(safe)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_close(actual: float, expected: float, *, name: str) -> float:
    difference = abs(actual - expected)
    if not math.isclose(
        actual,
        expected,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{name} does not reproduce A35")
    return difference


def riemann_descriptors(
    left_states: np.ndarray,
    right_states: np.ndarray,
    *,
    gamma: float,
    case_numbers: Sequence[int],
) -> dict[str, np.ndarray]:
    """Build the three frozen dimensionless descriptors from primitive metadata."""

    left = np.asarray(left_states, dtype=np.float64)
    right = np.asarray(right_states, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 3 or right.shape != left.shape:
        raise ValueError("left/right primitive metadata must align as [cases, 3]")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("primitive metadata must be finite")
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    indices = tuple(case_numbers)
    if (
        not indices
        or any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or int(index) < 0
            or int(index) >= left.shape[0]
            for index in indices
        )
        or len({int(index) for index in indices}) != len(indices)
    ):
        raise ValueError("case numbers must be unique valid integer indices")
    selected_left = left[np.asarray(indices, dtype=np.int64)]
    selected_right = right[np.asarray(indices, dtype=np.int64)]
    if np.any(selected_left[:, (0, 2)] <= 0.0) or np.any(
        selected_right[:, (0, 2)] <= 0.0
    ):
        raise ValueError("selected density and pressure metadata must be positive")
    sound_left = np.sqrt(gamma * selected_left[:, 2] / selected_left[:, 0])
    sound_right = np.sqrt(gamma * selected_right[:, 2] / selected_right[:, 0])
    denominator = sound_left + sound_right
    if np.any(denominator <= DENOMINATOR_FLOOR):
        raise ValueError("velocity-jump normalization is unresolved")
    values = np.column_stack(
        (
            np.log(selected_left[:, 0] / selected_right[:, 0]),
            np.log(selected_left[:, 2] / selected_right[:, 2]),
            (selected_left[:, 1] - selected_right[:, 1]) / denominator,
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Riemann descriptors must be finite")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise ValueError("Riemann descriptor vectors must be unique")
    return {
        f"case_{int(case_number):03d}": values[index]
        for index, case_number in enumerate(indices)
    }


def _metadata_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.asarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _load_descriptors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"left_states", "right_states", "gamma"}
        if not required.issubset(arrays.files):
            raise KeyError("native NPZ lacks primitive boundary metadata")
        stored = {key: np.asarray(arrays[key]) for key in sorted(required)}
    if stored["left_states"].shape != (512, 3) or stored["right_states"].shape != (
        512,
        3,
    ):
        raise ValueError("metadata carrier state shapes differ from the contract")
    if stored["gamma"].shape != ():
        raise ValueError("metadata carrier gamma must be scalar")
    if any(value.dtype != np.dtype(np.float32) for value in stored.values()):
        raise ValueError("metadata carrier values must be float32")
    metadata_sha256 = _metadata_sha256(stored)
    if metadata_sha256 != EXPECTED_METADATA_SHA256:
        raise ValueError("metadata carrier digest differs from preregistration")
    descriptors = riemann_descriptors(
        stored["left_states"],
        stored["right_states"],
        gamma=float(stored["gamma"].item()),
        case_numbers=EXPECTED_CASE_NUMBERS,
    )
    return descriptors, {
        "metadata_sha256": metadata_sha256,
        "left_states_shape": list(stored["left_states"].shape),
        "right_states_shape": list(stored["right_states"].shape),
        "gamma_shape": list(stored["gamma"].shape),
        "left_states_dtype": stored["left_states"].dtype.str,
        "right_states_dtype": stored["right_states"].dtype.str,
        "gamma_dtype": stored["gamma"].dtype.str,
    }


def _load_statistics(a35: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    reconstructed = a35.get("reconstructed_statistics")
    if not isinstance(reconstructed, Mapping):
        raise TypeError("A35 reconstructed statistics are missing")
    output: dict[str, dict[str, dict[str, float]]] = {}
    for scope in PHASE_SCOPES:
        value = reconstructed.get(scope)
        if not isinstance(value, Mapping):
            raise TypeError(f"A35 {scope} statistics are missing")
        rows = value.get("case_statistics")
        if not isinstance(rows, list):
            raise TypeError(f"A35 {scope} case statistics are missing")
        by_case: dict[str, dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("A35 case statistics must be objects")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in by_case:
                raise ValueError("A35 case statistic identity is invalid")
            statistic = {
                "target_square": _finite_float(
                    row.get("target_square"), name="target square"
                ),
                "feature_square": _finite_float(
                    row.get("feature_square"), name="feature square"
                ),
                "cross": _finite_float(row.get("cross"), name="cross"),
            }
            if statistic["target_square"] < 0.0:
                raise ValueError("A35 target energy must be nonnegative")
            if math.sqrt(statistic["feature_square"]) <= DENOMINATOR_FLOOR:
                raise ValueError("A35 feature energy is unresolved")
            by_case[case_id] = statistic
        if tuple(sorted(by_case)) != tuple(sorted(EXPECTED_CASE_IDS)):
            raise ValueError("A35 case inventory differs")
        output[scope] = by_case
    return output


def _validate_descriptors(
    descriptors: Mapping[str, np.ndarray],
    *,
    case_ids: Sequence[str],
) -> None:
    if set(descriptors) != set(case_ids):
        raise ValueError("descriptor and statistic case inventories differ")
    matrix = np.asarray(
        [descriptors[case_id] for case_id in case_ids], dtype=np.float64
    )
    if matrix.shape != (len(case_ids), len(DESCRIPTOR_NAMES)):
        raise ValueError("descriptor vectors differ from the frozen dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("descriptor vectors must be finite")
    if np.unique(matrix, axis=0).shape[0] != matrix.shape[0]:
        raise ValueError("descriptor vectors must be unique")
    if np.any(np.std(matrix, axis=0) <= DENOMINATOR_FLOOR):
        raise ValueError("a descriptor is constant on the population")


def _fit_state_map(
    descriptors: Mapping[str, np.ndarray],
    statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    train_case_ids: Sequence[str],
    ridge: float,
) -> dict[str, Any]:
    cases = tuple(sorted(train_case_ids))
    if len(cases) < 2 or len(set(cases)) != len(cases):
        raise ValueError("state-map fitting requires unique training cases")
    if ridge not in RIDGES:
        raise ValueError("ridge differs from the frozen inventory")
    matrix = np.asarray([descriptors[case_id] for case_id in cases], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    if np.any(scale <= DENOMINATOR_FLOOR) or not np.isfinite(scale).all():
        raise ValueError("training-fold descriptor standardization is unresolved")
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(cases), dtype=np.float64), standardized))
    penalty = np.diag((0.0, ridge, ridge, ridge))
    phase_fits = {}
    for scope in PHASE_SCOPES:
        feature_square = np.asarray(
            [statistics[scope][case_id]["feature_square"] for case_id in cases],
            dtype=np.float64,
        )
        cross = np.asarray(
            [statistics[scope][case_id]["cross"] for case_id in cases],
            dtype=np.float64,
        )
        mean_feature = float(np.mean(feature_square))
        if math.sqrt(mean_feature) <= DENOMINATOR_FLOOR:
            raise ValueError("training-fold feature energy is unresolved")
        weights = feature_square / mean_feature
        pseudo_target = cross / feature_square
        normal = design.T @ (weights[:, None] * design) + penalty
        right = design.T @ (weights * pseudo_target)
        try:
            coefficient = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError as error:
            raise ValueError("state-map normal equation is singular") from error
        if not np.isfinite(coefficient).all():
            raise ValueError("state-map coefficient is nonfinite")
        phase_fits[scope] = {
            "coefficient": coefficient,
            "standardized_slope": coefficient[1:],
            "normal_condition_number": float(np.linalg.cond(normal)),
            "mean_feature_square": mean_feature,
        }
    return {
        "train_case_ids": list(cases),
        "ridge": ridge,
        "descriptor_mean": mean,
        "descriptor_scale": scale,
        "standardized_training_descriptors": standardized,
        "phase_fits": phase_fits,
    }


def _support_audit(
    fit: Mapping[str, Any],
    descriptors: Mapping[str, np.ndarray],
    *,
    query_case_id: str,
) -> dict[str, Any]:
    training = np.asarray(fit["standardized_training_descriptors"], dtype=np.float64)
    if training.ndim != 2 or training.shape[0] < 2:
        raise ValueError("support audit requires at least two training descriptors")
    pairwise = np.linalg.norm(training[:, None, :] - training[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    nearest_other = np.min(pairwise, axis=1)
    radius = float(np.max(nearest_other))
    mean = np.asarray(fit["descriptor_mean"], dtype=np.float64)
    scale = np.asarray(fit["descriptor_scale"], dtype=np.float64)
    query = (np.asarray(descriptors[query_case_id], dtype=np.float64) - mean) / scale
    nearest = float(np.min(np.linalg.norm(training - query[None, :], axis=1)))
    return {
        "supported": nearest <= radius,
        "nearest_training_distance": nearest,
        "training_support_radius": radius,
        "support_margin": radius - nearest,
        "standardized_query": query,
    }


def _predict_coefficients(
    fit: Mapping[str, Any],
    descriptors: Mapping[str, np.ndarray],
    *,
    query_case_id: str,
) -> dict[str, float]:
    mean = np.asarray(fit["descriptor_mean"], dtype=np.float64)
    scale = np.asarray(fit["descriptor_scale"], dtype=np.float64)
    standardized = (
        np.asarray(descriptors[query_case_id], dtype=np.float64) - mean
    ) / scale
    design = np.concatenate(([1.0], standardized))
    prediction = {}
    for scope in PHASE_SCOPES:
        coefficient = np.asarray(
            fit["phase_fits"][scope]["coefficient"], dtype=np.float64
        )
        prediction[scope] = float(design @ coefficient)
    if not all(math.isfinite(value) for value in prediction.values()):
        raise ValueError("predicted coefficients must be finite")
    return prediction


def _square_error(statistic: Mapping[str, float], coefficient: float) -> float:
    value = (
        statistic["target_square"]
        - 2.0 * coefficient * statistic["cross"]
        + coefficient * coefficient * statistic["feature_square"]
    )
    tolerance = max(
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE * abs(statistic["target_square"]),
    )
    if value < -tolerance:
        raise ValueError("quadratic score produced negative energy")
    return max(float(value), 0.0)


def _select_ridge(
    descriptors: Mapping[str, np.ndarray],
    statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    candidate_case_ids: Sequence[str],
) -> dict[str, Any]:
    cases = tuple(sorted(candidate_case_ids))
    if len(cases) < 3:
        raise ValueError("inner selection requires at least three cases")
    candidates = []
    for ridge in RIDGES:
        zero = 0.0
        corrected = 0.0
        supported_count = 0
        rows = []
        for held_out_case in cases:
            train = [case_id for case_id in cases if case_id != held_out_case]
            fit = _fit_state_map(
                descriptors,
                statistics,
                train_case_ids=train,
                ridge=ridge,
            )
            support = _support_audit(fit, descriptors, query_case_id=held_out_case)
            unguarded = _predict_coefficients(
                fit, descriptors, query_case_id=held_out_case
            )
            guarded = {
                scope: unguarded[scope] if support["supported"] else 0.0
                for scope in PHASE_SCOPES
            }
            case_zero = 0.0
            case_corrected = 0.0
            for scope in PHASE_SCOPES:
                statistic = statistics[scope][held_out_case]
                case_zero += 0.5 * statistic["target_square"]
                case_corrected += 0.5 * _square_error(statistic, guarded[scope])
            zero += case_zero / len(cases)
            corrected += case_corrected / len(cases)
            supported_count += int(support["supported"])
            rows.append(
                {
                    "held_out_case_id": held_out_case,
                    "supported": support["supported"],
                    "zero_square": case_zero,
                    "corrected_square": case_corrected,
                }
            )
        candidates.append(
            {
                "ridge": ridge,
                "zero_square_case_mean": zero,
                "corrected_square_case_mean": corrected,
                "skill_vs_zero": 1.0 - corrected / zero,
                "supported_count": supported_count,
                "rows": rows,
            }
        )
    selected = candidates[0]
    for candidate in candidates[1:]:
        difference = (
            candidate["corrected_square_case_mean"]
            - selected["corrected_square_case_mean"]
        )
        if difference < -RIDGE_TIE_TOLERANCE or (
            abs(difference) <= RIDGE_TIE_TOLERANCE
            and candidate["ridge"] > selected["ridge"]
        ):
            selected = candidate
    return {
        "candidate_case_ids": list(cases),
        "candidates": candidates,
        "selected_ridge": selected["ridge"],
        "selected_corrected_square_case_mean": selected["corrected_square_case_mean"],
    }


def _score_predictions(
    statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
    predictions: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    case_ids = tuple(sorted(predictions))
    phase_values = {scope: {"zero": [], "corrected": []} for scope in PHASE_SCOPES}
    case_rows = []
    for case_id in case_ids:
        zero = 0.0
        corrected = 0.0
        phase_rows = {}
        for scope in PHASE_SCOPES:
            statistic = statistics[scope][case_id]
            phase_zero = statistic["target_square"]
            phase_corrected = _square_error(statistic, predictions[case_id][scope])
            phase_values[scope]["zero"].append(phase_zero)
            phase_values[scope]["corrected"].append(phase_corrected)
            zero += 0.5 * phase_zero
            corrected += 0.5 * phase_corrected
            phase_rows[f"{scope}_coefficient"] = predictions[case_id][scope]
            phase_rows[f"{scope}_rms_ratio"] = math.sqrt(phase_corrected / phase_zero)
        case_rows.append(
            {
                "case_id": case_id,
                "zero_square": zero,
                "corrected_square": corrected,
                "rms_ratio_vs_zero": math.sqrt(corrected / zero),
                "skill_vs_zero": 1.0 - corrected / zero,
                **phase_rows,
            }
        )

    def score(zero: Sequence[float], corrected: Sequence[float]) -> dict[str, float]:
        zero_mean = float(np.mean(np.asarray(zero, dtype=np.float64)))
        corrected_mean = float(np.mean(np.asarray(corrected, dtype=np.float64)))
        if math.sqrt(zero_mean) <= DENOMINATOR_FLOOR:
            raise ValueError("score target denominator is unresolved")
        return {
            "zero_square_case_mean": zero_mean,
            "corrected_square_case_mean": corrected_mean,
            "skill_vs_zero": 1.0 - corrected_mean / zero_mean,
            "rms_ratio_vs_zero": math.sqrt(corrected_mean / zero_mean),
        }

    return {
        "combined": score(
            [row["zero_square"] for row in case_rows],
            [row["corrected_square"] for row in case_rows],
        ),
        **{
            scope: score(
                phase_values[scope]["zero"],
                phase_values[scope]["corrected"],
            )
            for scope in PHASE_SCOPES
        },
        "case_scores": case_rows,
        "case_win_count": sum(row["rms_ratio_vs_zero"] < 1.0 for row in case_rows),
        "maximum_case_rms_ratio": max(row["rms_ratio_vs_zero"] for row in case_rows),
    }


def _a35_baseline_predictions(
    a35: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    crossfit = a35.get("crossfit")
    if not isinstance(crossfit, Mapping):
        raise TypeError("A35 cross-fit result is missing")
    folds = crossfit.get("folds")
    if not isinstance(folds, list):
        raise TypeError("A35 folds are missing")
    global_predictions = {}
    phase_predictions = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError("A35 folds must be objects")
        case_id = fold.get("held_out_case_id")
        if not isinstance(case_id, str) or not case_id or case_id in global_predictions:
            raise ValueError("A35 held-out case identity is invalid")
        global_coefficient = _finite_float(
            fold.get("global_coefficient"), name="A35 global coefficient"
        )
        global_predictions[case_id] = {
            scope: global_coefficient for scope in PHASE_SCOPES
        }
        phase_predictions[case_id] = {
            "calls_0_49": _finite_float(
                fold.get("early_coefficient"), name="A35 early coefficient"
            ),
            "calls_50_99": _finite_float(
                fold.get("late_coefficient"), name="A35 late coefficient"
            ),
        }
    if tuple(sorted(global_predictions)) != tuple(sorted(EXPECTED_CASE_IDS)):
        raise ValueError("A35 fold inventory differs")
    return global_predictions, phase_predictions


def _slope_stability(folds: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    vectors = [
        np.asarray(fold["fit"]["phase_fits"][scope]["standardized_slope"])
        for fold in folds
    ]
    norms = np.asarray([np.linalg.norm(vector) for vector in vectors], dtype=np.float64)
    resolved = norms > DENOMINATOR_FLOOR
    if not np.all(resolved):
        return {
            "status": "unresolved_zero_slope",
            "resolved_count": int(np.count_nonzero(resolved)),
            "fold_count": len(vectors),
            "median_pairwise_cosine": None,
            "maximum_to_minimum_norm_ratio": None,
            "norms": norms,
        }
    cosines = [
        float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
        for left, right in combinations(vectors, 2)
    ]
    return {
        "status": "ok",
        "resolved_count": len(vectors),
        "fold_count": len(vectors),
        "median_pairwise_cosine": float(np.median(cosines)),
        "minimum_pairwise_cosine": min(cosines),
        "maximum_to_minimum_norm_ratio": float(np.max(norms) / np.min(norms)),
        "norms": norms,
    }


def nested_state_crossfit(
    descriptors: Mapping[str, np.ndarray],
    statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Select ridge in inner folds, then score one guarded map per outer case."""

    case_ids = tuple(sorted(statistics[PHASE_SCOPES[0]]))
    _validate_descriptors(descriptors, case_ids=case_ids)
    if any(set(statistics[scope]) != set(case_ids) for scope in PHASE_SCOPES):
        raise ValueError("phase statistic inventories differ")
    folds = []
    guarded_predictions = {}
    unguarded_predictions = {}
    for held_out_case in case_ids:
        train = tuple(case_id for case_id in case_ids if case_id != held_out_case)
        selection = _select_ridge(
            descriptors,
            statistics,
            candidate_case_ids=train,
        )
        fit = _fit_state_map(
            descriptors,
            statistics,
            train_case_ids=train,
            ridge=selection["selected_ridge"],
        )
        support = _support_audit(fit, descriptors, query_case_id=held_out_case)
        unguarded = _predict_coefficients(fit, descriptors, query_case_id=held_out_case)
        guarded = {
            scope: unguarded[scope] if support["supported"] else 0.0
            for scope in PHASE_SCOPES
        }
        guarded_predictions[held_out_case] = guarded
        unguarded_predictions[held_out_case] = unguarded
        folds.append(
            {
                "held_out_case_id": held_out_case,
                "fit_case_ids": list(train),
                "selected_ridge": selection["selected_ridge"],
                "inner_selection": selection,
                "support": support,
                "guarded_coefficients": guarded,
                "unguarded_coefficients": unguarded,
                "fit": fit,
            }
        )
    full_selection = _select_ridge(
        descriptors,
        statistics,
        candidate_case_ids=case_ids,
    )
    full_fit = _fit_state_map(
        descriptors,
        statistics,
        train_case_ids=case_ids,
        ridge=full_selection["selected_ridge"],
    )
    guarded_score = _score_predictions(statistics, guarded_predictions)
    unguarded_score = _score_predictions(statistics, unguarded_predictions)
    supported_cases = [
        fold["held_out_case_id"] for fold in folds if fold["support"]["supported"]
    ]
    guarded_case_rows = {row["case_id"]: row for row in guarded_score["case_scores"]}
    ridge_counts = Counter(float(fold["selected_ridge"]) for fold in folds)
    slope_stability = {scope: _slope_stability(folds, scope) for scope in PHASE_SCOPES}
    return {
        "case_ids": list(case_ids),
        "folds": folds,
        "guarded_score": guarded_score,
        "unguarded_score": unguarded_score,
        "supported_case_ids": supported_cases,
        "abstained_case_ids": sorted(set(case_ids) - set(supported_cases)),
        "supported_case_win_count": sum(
            guarded_case_rows[case_id]["rms_ratio_vs_zero"] < 1.0
            for case_id in supported_cases
        ),
        "ridge_selection_counts": {
            str(ridge): ridge_counts.get(ridge, 0) for ridge in RIDGES
        },
        "dominant_ridge_count": max(ridge_counts.values()),
        "slope_stability": slope_stability,
        "full_population_selection": full_selection,
        "full_population_fit": full_fit,
    }


def _git_status_short(paths: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    input_hashes = {
        "a35_result": sha256_file(args.a35_result),
        "family_contract": sha256_file(args.family_contract),
        "metadata_carrier": sha256_file(args.metadata_carrier),
        "sampler_source": sha256_file(args.sampler_source),
    }
    if input_hashes != EXPECTED_HASHES:
        raise ValueError(f"input hashes differ from preregistration: {input_hashes}")
    a35 = _read_json(args.a35_result)
    verify_payload_sha256(a35)
    parent_payload_hashes = {"a35_result": a35["payload_sha256"]}
    if parent_payload_hashes != EXPECTED_PAYLOAD_HASHES:
        raise ValueError("A35 payload hash differs from preregistration")
    if a35.get("working_id") != (
        "W26-L5-P6-RFB19-A35-EULER1D-PHASE-CONDITIONED-CROSSFIT"
    ):
        raise ValueError("A35 working identity differs")
    family_contract = _read_json(args.family_contract)
    if (
        family_contract.get("status") != "ok"
        or family_contract.get("contract", {}).get("physical_case_identity_exact")
        is not True
    ):
        raise ValueError("D030 physical-case identity contract is unresolved")

    statistics = _load_statistics(a35)
    descriptors, metadata_contract = _load_descriptors(args.metadata_carrier)
    _validate_descriptors(descriptors, case_ids=EXPECTED_CASE_IDS)
    nested = nested_state_crossfit(descriptors, statistics)
    global_predictions, phase_predictions = _a35_baseline_predictions(a35)
    global_score = _score_predictions(statistics, global_predictions)
    phase_score = _score_predictions(statistics, phase_predictions)
    a35_scores = a35["crossfit"]["scores"]
    reproduction = {
        "global_combined_skill_abs": _require_close(
            global_score["combined"]["skill_vs_zero"],
            _finite_float(
                a35_scores["global"]["combined"]["skill_vs_zero"],
                name="A35 global skill",
            ),
            name="A35 global skill",
        ),
        "phase_combined_skill_abs": _require_close(
            phase_score["combined"]["skill_vs_zero"],
            _finite_float(
                a35_scores["two_phase"]["combined"]["skill_vs_zero"],
                name="A35 phase skill",
            ),
            name="A35 phase skill",
        ),
    }
    for scope in PHASE_SCOPES:
        reproduction[f"global_{scope}_skill_abs"] = _require_close(
            global_score[scope]["skill_vs_zero"],
            _finite_float(
                a35_scores["global"][scope]["skill_vs_zero"],
                name=f"A35 global {scope} skill",
            ),
            name=f"A35 global {scope} skill",
        )
        reproduction[f"phase_{scope}_skill_abs"] = _require_close(
            phase_score[scope]["skill_vs_zero"],
            _finite_float(
                a35_scores["two_phase"][scope]["skill_vs_zero"],
                name=f"A35 phase {scope} skill",
            ),
            name=f"A35 phase {scope} skill",
        )

    guarded = nested["guarded_score"]
    supported_cases = set(nested["supported_case_ids"])
    maximum_coefficient = (
        max(
            abs(value)
            for fold in nested["folds"]
            if fold["held_out_case_id"] in supported_cases
            for value in fold["guarded_coefficients"].values()
        )
        if supported_cases
        else 0.0
    )
    abstentions_exact = all(
        all(value == 0.0 for value in fold["guarded_coefficients"].values())
        for fold in nested["folds"]
        if fold["held_out_case_id"] not in supported_cases
    )
    readiness_checks = {
        "combined_skill_at_least_0p01": guarded["combined"]["skill_vs_zero"]
        >= MINIMUM_COMBINED_SKILL,
        "each_phase_skill_positive": all(
            guarded[scope]["skill_vs_zero"] > 0.0 for scope in PHASE_SCOPES
        ),
        "skill_increment_over_phase_at_least_0p02": (
            guarded["combined"]["skill_vs_zero"]
            - phase_score["combined"]["skill_vs_zero"]
        )
        >= MINIMUM_SKILL_INCREMENT,
        "at_least_12_supported_cases": len(supported_cases) >= MINIMUM_SUPPORTED_CASES,
        "at_least_10_supported_case_wins": nested["supported_case_win_count"]
        >= MINIMUM_SUPPORTED_WINS,
        "abstentions_are_exact_zero": abstentions_exact,
        "at_least_12_total_case_wins": guarded["case_win_count"] >= MINIMUM_TOTAL_WINS,
        "maximum_case_rms_ratio_at_most_1p05": guarded["maximum_case_rms_ratio"]
        <= MAXIMUM_CASE_RMS_RATIO,
        "maximum_nonzero_coefficient_at_most_3": maximum_coefficient
        <= MAXIMUM_COEFFICIENT_MAGNITUDE,
        "one_ridge_selected_in_at_least_8_folds": nested["dominant_ridge_count"]
        >= MINIMUM_DOMINANT_RIDGE_COUNT,
        "phase_slope_stability": all(
            nested["slope_stability"][scope]["status"] == "ok"
            and nested["slope_stability"][scope]["median_pairwise_cosine"]
            >= MINIMUM_SLOPE_COSINE
            and nested["slope_stability"][scope]["maximum_to_minimum_norm_ratio"]
            <= MAXIMUM_SLOPE_NORM_RATIO
            for scope in PHASE_SCOPES
        ),
    }
    validity_checks = {
        "input_hashes_exact": input_hashes == EXPECTED_HASHES,
        "a35_payload_exact": parent_payload_hashes == EXPECTED_PAYLOAD_HASHES,
        "d030_physical_case_identity_exact": True,
        "metadata_digest_exact": metadata_contract["metadata_sha256"]
        == EXPECTED_METADATA_SHA256,
        "case_inventory_exact": set(descriptors) == set(EXPECTED_CASE_IDS),
        "descriptors_finite_unique_nonconstant": True,
        "a35_statistics_resolved": True,
        "a35_scores_reproduced": all(
            difference <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
            for difference in reproduction.values()
        ),
        "nested_fold_inventory_exact": len(nested["folds"]) == len(EXPECTED_CASE_IDS)
        and all(
            fold["held_out_case_id"] not in fold["fit_case_ids"]
            and len(fold["fit_case_ids"]) == len(EXPECTED_CASE_IDS) - 1
            for fold in nested["folds"]
        ),
        "all_scores_and_coefficients_finite": all(
            math.isfinite(float(value))
            for fold in nested["folds"]
            for value in fold["unguarded_coefficients"].values()
        ),
    }
    readiness_passed = all(validity_checks.values()) and all(readiness_checks.values())

    descriptor_rows = [
        {
            "case_id": case_id,
            **{
                name: float(descriptors[case_id][index])
                for index, name in enumerate(DESCRIPTOR_NAMES)
            },
        }
        for case_id in EXPECTED_CASE_IDS
    ]
    guarded_by_case = {row["case_id"]: row for row in guarded["case_scores"]}
    case_rows = []
    fold_rows = []
    for fold in nested["folds"]:
        case_id = fold["held_out_case_id"]
        support = fold["support"]
        case_rows.append(
            {
                **guarded_by_case[case_id],
                "supported": support["supported"],
                "nearest_training_distance": support["nearest_training_distance"],
                "training_support_radius": support["training_support_radius"],
                "selected_ridge": fold["selected_ridge"],
                "unguarded_early_coefficient": fold["unguarded_coefficients"][
                    "calls_0_49"
                ],
                "unguarded_late_coefficient": fold["unguarded_coefficients"][
                    "calls_50_99"
                ],
            }
        )
        fold_rows.append(
            {
                "held_out_case_id": case_id,
                "selected_ridge": fold["selected_ridge"],
                "supported": support["supported"],
                "nearest_training_distance": support["nearest_training_distance"],
                "training_support_radius": support["training_support_radius"],
                "guarded_early_coefficient": fold["guarded_coefficients"]["calls_0_49"],
                "guarded_late_coefficient": fold["guarded_coefficients"]["calls_50_99"],
                "early_slope": fold["fit"]["phase_fits"]["calls_0_49"][
                    "standardized_slope"
                ],
                "late_slope": fold["fit"]["phase_fits"]["calls_50_99"][
                    "standardized_slope"
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "descriptors.csv", descriptor_rows)
    write_csv(args.output_dir / "folds.csv", fold_rows)
    write_csv(args.output_dir / "case_scores.csv", case_rows)
    artifact_hashes = {
        name: sha256_file(args.output_dir / name)
        for name in ("descriptors.csv", "folds.csv", "case_scores.csv")
    }
    payload = _with_json_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": (
                "qualified_for_capped_validation_replay"
                if readiness_passed
                else "completed_readiness_failed"
            ),
            "contract": {
                "family": "Euler1D D030",
                "population": "16 open validation cases",
                "descriptors": list(DESCRIPTOR_NAMES),
                "descriptor_source": "fixed left/right primitive problem metadata",
                "feature": "uncapped projected fine-to-native discrepancy, k=1--7",
                "target": "signed native one-step correction target, k=1--7",
                "ridge_inventory": list(RIDGES),
                "selection": "nested leave-one-case-out, guarded combined SSE",
                "support": "maximum nearest-other training descriptor distance",
                "cap_included": False,
                "recurrence_included": False,
            },
            "input_hashes": input_hashes,
            "parent_payload_hashes": parent_payload_hashes,
            "metadata_contract": metadata_contract,
            "source_hashes": sha256_files(SOURCE_PATHS),
            "source_status": _git_status_short(SOURCE_PATHS),
            "validity_checks": validity_checks,
            "a35_reproduction": reproduction,
            "baselines": {
                "global_scalar": global_score,
                "phase_only": phase_score,
            },
            "nested_crossfit": nested,
            "maximum_supported_coefficient_magnitude": maximum_coefficient,
            "readiness_checks": readiness_checks,
            "readiness_passed": readiness_passed,
            "artifact_hashes": artifact_hashes,
            "execution": {
                "model_calls": 0,
                "checkpoint_loaded": False,
                "state_or_reference_trajectory_loaded": False,
                "initial_boundary_metadata_loaded": True,
                "metadata_carrier_trajectory_arrays_loaded": False,
                "prediction_or_recurrence_executed": False,
                "test_population_opened": False,
            },
            "claim_boundary": (
                "An adaptive retrospective uncapped validation diagnostic only. "
                "Passing can authorize one separately preregistered capped replay on "
                "the same open validation cases; it cannot establish cap safety, "
                "autoregressive benefit, independent confirmation, family transfer, "
                "conservation, resolution invariance, or Richardson extrapolation."
            ),
        }
    )
    atomic_write_json(args.output_dir / "state_crossfit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a35-result", type=Path, required=True)
    parser.add_argument("--family-contract", type=Path, required=True)
    parser.add_argument("--metadata-carrier", type=Path, required=True)
    parser.add_argument("--sampler-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
