#!/usr/bin/env python3
"""Evaluate the frozen A37 causal discrepancy observer on retained modal rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file, sha256_files
from utility.time_dependent_no.pcno_artifacts import write_csv_with_paths as write_csv
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_residual_geometry import (
    MODAL_CAUSAL_OBSERVER_DECAYS,
    MODAL_CAUSAL_OBSERVER_FAMILY,
    MODAL_CAUSAL_OBSERVER_RIDGE,
    MODAL_LINEAR_TIE_TOLERANCE,
    ModalSnapshot,
    cap_modal_predictions,
    fit_modal_causal_observer,
    modal_causal_observer_features,
    modal_linear_map_stability,
    predict_modal_causal_observer,
    score_modal_linear_predictions,
    snapshots_from_modal_rows,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A37-SP19-CAUSAL-DISCREPANCY-OBSERVER"
SCHEMA = "pcno_sp19_causal_discrepancy_observer_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_causal_discrepancy_observer.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "a27_result": "78bc1a9ff78304c5724e817fc1813f98ba335513dc72fe63addf5a1ca1103331",
    "a27_scores": "0077ca19246c7cbdb6cbd7f12acb9f77e07a7249f020bf3da3a55411820c188a",
    "calibration_result": "d9533285d89e70e61c16833ceeb27cd542cd65f44925a746c94786df26faf8cd",
    "calibration_modal_records": "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb",
    "calibration_audits": "08cd55bc130b42ac9f48b252b86b71710bcdbef466bf77618365a83a0e70f9b0",
    "teacher_result": "bbbc3cdbd9afb64b8b8e63169b8acb9a9a4614957d6f4461412b4427405f63be",
    "teacher_modal_records": "0ac7915de0c47d519a8e9e3d61d2b582a3266e63b191f6951cf9033f4198c4e5",
    "teacher_audits": "8260ba50c1d29cceff1e1e4d13ba9d75408890a0124fbd6f4e1a57e17b288009",
}
EXPECTED_PAYLOADS = {
    "a27_result": "fb7decd0fe14af9ed5387acc06985370946af2267d5b63510399fff3170e96d5",
    "calibration_result": "0f4351cf92ae16beec11cba57ac5aa4973df5579e73e11cc8d0e616a25e93c59",
    "teacher_result": "1ba8b8e30d0e6b5bd7a7a7a2d4925b23fbb56b558180502243f08e7c393c6046",
}
MODE_CELLS = tuple((mode, component) for mode in range(8) for component in range(4))
CALLS = tuple(range(30))
BANDS = ((0, 7), (8, 14), (15, 21), (22, 29))
GROUPS = {
    "calibration": ("e01", "e02", "e03", "e04", "e05", "e07", "e08", "e09", "e10"),
    "teacher": ("e00", "e06", "e11"),
}
CASES = {
    population: tuple(
        f"sv_{group}_y{position}" for group in groups for position in ("00", "08")
    )
    for population, groups in GROUPS.items()
}
TOTAL_VOLUME = 2.0
MAXIMUM_RELATIVE_NORM = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "a27_result": args.a27_result,
        "a27_scores": args.a27_scores,
        "calibration_result": args.calibration_result,
        "calibration_modal_records": args.calibration_modal_records,
        "calibration_audits": args.calibration_audits,
        "teacher_result": args.teacher_result,
        "teacher_modal_records": args.teacher_modal_records,
        "teacher_audits": args.teacher_audits,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("A37 inputs differ from the frozen artifacts")
    payloads = {
        "a27_result": _read_json(args.a27_result),
        "calibration_result": _read_json(args.calibration_result),
        "teacher_result": _read_json(args.teacher_result),
    }
    for name, payload in payloads.items():
        verify_payload_sha256(payload)
        if payload.get("payload_sha256") != EXPECTED_PAYLOADS[name]:
            raise ValueError(f"{name} payload differs from the frozen artifact")
    a27 = payloads["a27_result"]
    calibration = payloads["calibration_result"]
    teacher = payloads["teacher_result"]
    if (
        a27.get("status") != "stopped_no_candidate"
        or a27.get("decision", {}).get("selected") != "none"
        or a27.get("artifact_hashes", {}).get("robustness_scores.csv")
        != EXPECTED_HASHES["a27_scores"]
    ):
        raise ValueError("A27 stop and owned-score identity differ")
    if (
        calibration.get("modal_records_sha256")
        != EXPECTED_HASHES["calibration_modal_records"]
        or calibration.get("artifact_hashes", {}).get("calibration_audits.csv")
        != EXPECTED_HASHES["calibration_audits"]
        or teacher.get("modal_records_sha256")
        != EXPECTED_HASHES["teacher_modal_records"]
        or teacher.get("artifact_hashes", {}).get("teacher_audits.csv")
        != EXPECTED_HASHES["teacher_audits"]
        or teacher.get("calibration_sha256") != EXPECTED_HASHES["calibration_result"]
        or teacher.get("calibration_payload_sha256")
        != EXPECTED_PAYLOADS["calibration_result"]
    ):
        raise ValueError("A2 modal and audit ownership differs")
    return {"hashes": hashes, **payloads}


def _validate_population(rows: Sequence[ModalSnapshot], *, population: str) -> None:
    expected = {(case, call) for case in CASES[population] for call in CALLS}
    actual = {(row.case_id, row.input_call) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"{population} case/call inventory differs")
    if {row.group_id for row in rows} != set(GROUPS[population]):
        raise ValueError(f"{population} group inventory differs")
    if any(row.case_id.split("_")[1] != row.group_id for row in rows):
        raise ValueError(f"{population} case/group membership differs")


def _native_rms(
    rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[ModalSnapshot],
    *,
    population: str,
) -> np.ndarray:
    lookup: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("policy") != "rank7_fine_away_half":
            continue
        key = (str(row.get("case_id")), int(str(row.get("input_call"))))
        if key in lookup:
            raise ValueError(f"duplicate {population} native-increment audit")
        value = float(row.get("native_increment_rms", "nan"))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{population} native-increment RMS is unresolved")
        lookup[key] = value
    expected = {(row.case_id, row.input_call) for row in snapshots}
    if set(lookup) != expected:
        raise ValueError(f"{population} native-increment audit inventory differs")
    return np.asarray([lookup[(row.case_id, row.input_call)] for row in snapshots])


def _subset(
    snapshots: Sequence[ModalSnapshot],
    native_rms: np.ndarray,
    predicate: Any,
) -> tuple[list[ModalSnapshot], np.ndarray, np.ndarray]:
    indices = np.asarray(
        [index for index, row in enumerate(snapshots) if predicate(row)],
        dtype=np.int64,
    )
    return [snapshots[index] for index in indices], native_rms[indices], indices


def _score_prediction(
    snapshots: Sequence[ModalSnapshot],
    predictions: np.ndarray,
    native_rms: np.ndarray,
) -> dict[str, Any]:
    uncapped = score_modal_linear_predictions(
        snapshots, predictions, active_cells=FROZEN_ACTIVE_CELLS
    )
    cap = cap_modal_predictions(
        predictions,
        native_rms,
        total_volume=TOTAL_VOLUME,
        maximum_relative_norm=MAXIMUM_RELATIVE_NORM,
    )
    capped = score_modal_linear_predictions(
        snapshots, cap["predictions"], active_cells=FROZEN_ACTIVE_CELLS
    )
    resolved = np.asarray(cap["resolved"], dtype=bool)
    return {
        "uncapped": uncapped,
        "capped": capped,
        "cap": {
            "row_count": len(snapshots),
            "resolved_count": int(np.sum(resolved)),
            "unresolved_count": int(np.sum(~resolved)),
            "active_count": int(np.sum(cap["cap_active"])),
            "minimum_scale": float(np.min(cap["scale"])),
            "maximum_raw_ratio": float(
                np.nanmax(cap["raw_correction_to_native_increment"])
            ),
            "maximum_capped_ratio": float(
                np.nanmax(cap["correction_to_native_increment"])
            ),
            "maximum_cap_violation": float(cap["maximum_cap_violation"]),
        },
    }


def _fit_and_score(
    train: Sequence[ModalSnapshot],
    test_context: Sequence[ModalSnapshot],
    native_rms: np.ndarray,
    *,
    decay: float,
    label_filter: Any | None = None,
    score_filter: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    fit = fit_modal_causal_observer(
        train,
        active_cells=FROZEN_ACTIVE_CELLS,
        decay=decay,
        label_filter=label_filter,
    )
    prediction = predict_modal_causal_observer(fit, test_context)
    if score_filter is None:
        scored_rows = list(test_context)
        scored_prediction = prediction
        scored_native = native_rms
    else:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(test_context)
                if bool(score_filter(row))
            ],
            dtype=np.int64,
        )
        if indices.size == 0:
            raise ValueError("modal causal-observer score filter is empty")
        scored_rows = [test_context[index] for index in indices]
        scored_prediction = prediction[indices]
        scored_native = native_rms[indices]
    return (
        fit,
        _score_prediction(scored_rows, scored_prediction, scored_native),
        prediction,
    )


def _select_decay(snapshots: Sequence[ModalSnapshot]) -> dict[str, Any]:
    rows = tuple(snapshots)
    groups = tuple(sorted({row.group_id for row in rows}))
    candidates = []
    for decay in MODAL_CAUSAL_OBSERVER_DECAYS:
        tests: list[ModalSnapshot] = []
        predictions = []
        for group in groups:
            train = [row for row in rows if row.group_id != group]
            held = [row for row in rows if row.group_id == group]
            fit = fit_modal_causal_observer(
                train, active_cells=FROZEN_ACTIVE_CELLS, decay=decay
            )
            tests.extend(held)
            predictions.extend(predict_modal_causal_observer(fit, held))
        score = score_modal_linear_predictions(
            tests, np.asarray(predictions), active_cells=FROZEN_ACTIVE_CELLS
        )
        candidates.append(
            {
                "decay": decay,
                "skill_vs_zero": score["skill_vs_zero"],
                "rms_ratio_vs_zero": score["rms_ratio_vs_zero"],
                "case_win_count": score["case_win_count"],
                "group_win_count": score["group_win_count"],
                "harmful_row_count": score["harmful_row_count"],
            }
        )
    selected = candidates[0]
    for candidate in candidates[1:]:
        candidate_skill = float(candidate["skill_vs_zero"])
        selected_skill = float(selected["skill_vs_zero"])
        if candidate_skill > selected_skill + MODAL_LINEAR_TIE_TOLERANCE or (
            abs(candidate_skill - selected_skill) <= MODAL_LINEAR_TIE_TOLERANCE
            and float(candidate["decay"]) < float(selected["decay"])
        ):
            selected = candidate
    return {"selected": selected, "candidates": candidates}


def _nested_strength_crossfit(
    calibration: Sequence[ModalSnapshot], calibration_native: np.ndarray
) -> dict[str, Any]:
    groups = tuple(sorted(GROUPS["calibration"]))
    tests: list[ModalSnapshot] = []
    predictions = []
    native = []
    folds = []
    fixed_zero_predictions = []
    fixed_zero_tests: list[ModalSnapshot] = []
    fixed_zero_native = []
    fits = []
    for group in groups:
        train = [row for row in calibration if row.group_id != group]
        held, held_native, _ = _subset(
            calibration,
            calibration_native,
            lambda row, value=group: row.group_id == value,
        )
        selection = _select_decay(train)
        decay = float(selection["selected"]["decay"])
        fit, score, prediction = _fit_and_score(train, held, held_native, decay=decay)
        _, zero_score, zero_prediction = _fit_and_score(
            train, held, held_native, decay=0.0
        )
        fits.append(fit)
        tests.extend(held)
        predictions.extend(prediction)
        native.extend(held_native)
        fixed_zero_tests.extend(held)
        fixed_zero_predictions.extend(zero_prediction)
        fixed_zero_native.extend(held_native)
        folds.append(
            {
                "held_out_group": group,
                "fit_groups": tuple(value for value in groups if value != group),
                "selected_decay": decay,
                "inner_candidates": selection["candidates"],
                "held_out_score": score,
                "fixed_zero_score": zero_score,
            }
        )
    return {
        "score": _score_prediction(tests, np.asarray(predictions), np.asarray(native)),
        "fixed_zero_score": _score_prediction(
            fixed_zero_tests,
            np.asarray(fixed_zero_predictions),
            np.asarray(fixed_zero_native),
        ),
        "folds": folds,
        "fits": fits,
    }


def _fit_diagnostics(
    fit: Mapping[str, Any], snapshots: Sequence[ModalSnapshot]
) -> dict[str, Any]:
    features = modal_causal_observer_features(
        snapshots,
        active_cells=FROZEN_ACTIVE_CELLS,
        decay=float(fit["decay"]),
    )
    scale = np.asarray(fit["feature_scale"], dtype=np.float64)
    standardized = features / scale
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 1.0e-8
        else None
    )
    coefficients = np.asarray(fit["original_coefficients"], dtype=np.float64)
    prediction = features @ coefficients
    rounded = features.astype(np.float32).astype(np.float64) @ coefficients
    denominator = np.linalg.norm(prediction, axis=1)
    relative = np.divide(
        np.linalg.norm(rounded - prediction, axis=1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1.0e-8,
    )
    resolved = np.isfinite(relative)
    return {
        "standardized_design_singular_values": singular_values,
        "standardized_design_condition_number": condition,
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
        "coefficient_spectral_norm": float(np.linalg.norm(coefficients, ord=2)),
        "float32_prediction_resolved_count": int(np.sum(resolved)),
        "float32_prediction_unresolved_count": int(np.sum(~resolved)),
        "maximum_float32_prediction_relative_change": (
            float(np.max(relative[resolved])) if np.any(resolved) else None
        ),
    }


def _score_row(
    *,
    evaluation: str,
    decay: float,
    score: Mapping[str, Any],
    held_group: str | None = None,
    band: tuple[int, int] | None = None,
) -> dict[str, Any]:
    capped = score["capped"]
    return {
        "evaluation": evaluation,
        "decay": decay,
        "held_group": held_group,
        "first_input_call": None if band is None else band[0],
        "last_input_call": None if band is None else band[1],
        "snapshot_count": capped["snapshot_count"],
        "case_count": capped["case_count"],
        "group_count": capped["group_count"],
        "uncapped_skill": score["uncapped"]["skill_vs_zero"],
        "uncapped_rms_ratio": score["uncapped"]["rms_ratio_vs_zero"],
        "capped_skill": capped["skill_vs_zero"],
        "capped_rms_ratio": capped["rms_ratio_vs_zero"],
        "capped_case_wins": capped["case_win_count"],
        "capped_group_wins": capped["group_win_count"],
        "capped_harmful_rows": capped["harmful_row_count"],
        "median_signed_cosine": capped["median_signed_cosine"],
        "maximum_case_rms_ratio": capped["maximum_case_rms_ratio"],
        "cap_active_count": score["cap"]["active_count"],
        "minimum_cap_scale": score["cap"]["minimum_scale"],
        "maximum_raw_correction_ratio": score["cap"]["maximum_raw_ratio"],
        "maximum_capped_correction_ratio": score["cap"]["maximum_capped_ratio"],
        "maximum_cap_violation": score["cap"]["maximum_cap_violation"],
        "status": capped["status"],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    parents = _verify_inputs(args)
    calibration = snapshots_from_modal_rows(
        _read_csv(args.calibration_modal_records),
        expected_cells=MODE_CELLS,
        expected_calls=CALLS,
    )
    teacher = snapshots_from_modal_rows(
        _read_csv(args.teacher_modal_records),
        expected_cells=MODE_CELLS,
        expected_calls=CALLS,
    )
    _validate_population(calibration, population="calibration")
    _validate_population(teacher, population="teacher")
    calibration_native = _native_rms(
        _read_csv(args.calibration_audits), calibration, population="calibration"
    )
    teacher_native = _native_rms(
        _read_csv(args.teacher_audits), teacher, population="teacher"
    )

    selection = _select_decay(calibration)
    selected_decay = float(selection["selected"]["decay"])
    nested = _nested_strength_crossfit(calibration, calibration_native)
    score_rows = []

    dual_scores = []
    for held_group in GROUPS["calibration"]:
        for band in BANDS:
            start, stop = band
            train = [row for row in calibration if row.group_id != held_group]
            test_context, native, _ = _subset(
                calibration,
                calibration_native,
                lambda row, group=held_group: row.group_id == group,
            )
            _, score, _ = _fit_and_score(
                train,
                test_context,
                native,
                decay=selected_decay,
                label_filter=lambda row, lo=start, hi=stop: (
                    not (lo <= row.input_call <= hi)
                ),
                score_filter=lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
            )
            dual_scores.append(score)
            score_rows.append(
                _score_row(
                    evaluation="dual_held_calibration",
                    decay=selected_decay,
                    score=score,
                    held_group=held_group,
                    band=band,
                )
            )

    temporal_fits = []
    teacher_band_scores = []
    for band in BANDS:
        start, stop = band
        fit, score, _ = _fit_and_score(
            calibration,
            teacher,
            teacher_native,
            decay=selected_decay,
            label_filter=lambda row, lo=start, hi=stop: (
                not (lo <= row.input_call <= hi)
            ),
            score_filter=lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
        )
        temporal_fits.append(fit)
        teacher_band_scores.append(score)
        score_rows.append(
            _score_row(
                evaluation="leave_band_out_teacher",
                decay=selected_decay,
                score=score,
                band=band,
            )
        )

    full_fit, teacher_score, teacher_prediction = _fit_and_score(
        calibration, teacher, teacher_native, decay=selected_decay
    )
    score_rows.append(
        _score_row(
            evaluation="frozen_full_teacher",
            decay=selected_decay,
            score=teacher_score,
        )
    )
    teacher_full_band_scores = []
    for band in BANDS:
        start, stop = band
        test, native, indices = _subset(
            teacher,
            teacher_native,
            lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
        )
        score = _score_prediction(test, teacher_prediction[indices], native)
        teacher_full_band_scores.append(score)
        score_rows.append(
            _score_row(
                evaluation="frozen_full_teacher_band",
                decay=selected_decay,
                score=score,
                band=band,
            )
        )

    fixed_teacher = {}
    for decay in MODAL_CAUSAL_OBSERVER_DECAYS:
        _, score, _ = _fit_and_score(calibration, teacher, teacher_native, decay=decay)
        fixed_teacher[str(decay)] = score
        score_rows.append(
            _score_row(
                evaluation="fixed_decay_teacher",
                decay=decay,
                score=score,
            )
        )

    selected_fit_groups: dict[float, list[Mapping[str, Any]]] = {}
    for fit in nested["fits"]:
        selected_fit_groups.setdefault(float(fit["decay"]), []).append(fit)
    positive_fit_groups = {
        decay: fits for decay, fits in selected_fit_groups.items() if decay > 0.0
    }
    if positive_fit_groups:
        dominant_decay, dominant_fits = max(
            positive_fit_groups.items(), key=lambda item: (len(item[1]), -item[0])
        )
    else:
        dominant_decay, dominant_fits = 0.0, []
    strength_stability = (
        modal_linear_map_stability(dominant_fits) if len(dominant_fits) >= 2 else None
    )
    temporal_stability = modal_linear_map_stability(temporal_fits)
    diagnostics = _fit_diagnostics(full_fit, teacher)
    selected_counts = Counter(float(fold["selected_decay"]) for fold in nested["folds"])
    positive_dominant = max(
        (count for decay, count in selected_counts.items() if decay > 0.0),
        default=0,
    )
    nested_score = nested["score"]["capped"]
    nested_zero = nested["fixed_zero_score"]["capped"]
    all_gate_scores = [
        nested["score"],
        nested["fixed_zero_score"],
        *dual_scores,
        *teacher_band_scores,
        teacher_score,
        *teacher_full_band_scores,
    ]
    readiness = {
        "nested_skill_strictly_positive": nested_score["skill_vs_zero"] > 0.0,
        "nested_all_cases_groups_win_without_harm": (
            nested_score["case_win_count"] == 18
            and nested_score["group_win_count"] == 9
            and nested_score["harmful_row_count"] == 0
        ),
        "nested_skill_exceeds_decay_zero_by_0p01": (
            nested_score["skill_vs_zero"] >= nested_zero["skill_vs_zero"] + 0.01
        ),
        "positive_decay_selected_at_least_six_folds": positive_dominant >= 6,
        "all_36_dual_held_cells_positive_two_case_wins": all(
            score["capped"]["skill_vs_zero"] > 0.0
            and score["capped"]["case_win_count"] == 2
            for score in dual_scores
        ),
        "all_four_leave_band_out_teacher_scores_pass": all(
            score["capped"]["skill_vs_zero"] > 0.0
            and score["capped"]["case_win_count"] == 6
            and score["capped"]["group_win_count"] == 3
            and score["capped"]["harmful_row_count"] == 0
            for score in teacher_band_scores
        ),
        "teacher_full_skill_at_least_0p95": (
            teacher_score["capped"]["skill_vs_zero"] >= 0.95
        ),
        "teacher_all_cases_groups_bands_win_without_harm": (
            teacher_score["capped"]["case_win_count"] == 6
            and teacher_score["capped"]["group_win_count"] == 3
            and teacher_score["capped"]["harmful_row_count"] == 0
            and all(
                score["capped"]["case_win_count"] == 6
                and score["capped"]["group_win_count"] == 3
                and score["capped"]["harmful_row_count"] == 0
                for score in teacher_full_band_scores
            )
        ),
        "strength_and_temporal_minimum_cosines_exceed_0p90": (
            strength_stability is not None
            and strength_stability["minimum_pairwise_frobenius_cosine"] > 0.90
            and temporal_stability["minimum_pairwise_frobenius_cosine"] > 0.90
        ),
        "strength_and_temporal_norm_ratios_at_most_1p50": (
            strength_stability is not None
            and strength_stability["maximum_to_minimum_norm_ratio"] <= 1.50
            and temporal_stability["maximum_to_minimum_norm_ratio"] <= 1.50
        ),
        "all_cap_rows_resolved": all(
            score["cap"]["unresolved_count"] == 0 for score in all_gate_scores
        ),
        "all_capped_ratios_at_most_0p05": max(
            score["cap"]["maximum_capped_ratio"] for score in all_gate_scores
        )
        <= MAXIMUM_RELATIVE_NORM + MODAL_LINEAR_TIE_TOLERANCE,
        "all_cap_bookkeeping_closes": max(
            score["cap"]["maximum_cap_violation"] for score in all_gate_scores
        )
        <= MODAL_LINEAR_TIE_TOLERANCE,
        "float32_prediction_change_at_most_1e_5": (
            diagnostics["maximum_float32_prediction_relative_change"] is not None
            and diagnostics["maximum_float32_prediction_relative_change"] <= 1.0e-5
        ),
    }
    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "a27_stop_exact": True,
        "calibration_inventory_exact_18x30x32": len(calibration) == 540,
        "teacher_inventory_exact_6x30x32": len(teacher) == 180,
        "native_increment_audits_exact": (
            len(calibration_native) == 540 and len(teacher_native) == 180
        ),
        "observer_contract_exact": (
            MODAL_CAUSAL_OBSERVER_DECAYS == (0.0, 0.5, 0.8, 0.95)
            and MODAL_CAUSAL_OBSERVER_RIDGE == 1.0e-6
            and MODAL_CAUSAL_OBSERVER_FAMILY == "fine_full_causal_ema"
        ),
        "causal_prefix_inventory_exact": all(
            np.isfinite(
                modal_causal_observer_features(
                    calibration,
                    active_cells=FROZEN_ACTIVE_CELLS,
                    decay=decay,
                )
            ).all()
            for decay in MODAL_CAUSAL_OBSERVER_DECAYS
        ),
        "nested_fold_count_exact_9": len(nested["folds"]) == 9,
        "dual_holdout_count_exact_36": len(dual_scores) == 36,
        "zero_intercept_exact": np.array_equal(
            full_fit["intercept"], np.zeros(len(FROZEN_ACTIVE_CELLS))
        ),
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "gradient_and_transfer_paths_unchanged": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A37 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "observer_scores.csv", score_rows)
    write_csv(
        args.output_dir / "observer_folds.csv",
        [
            {
                "held_out_group": fold["held_out_group"],
                "fit_groups": "|".join(fold["fit_groups"]),
                "selected_decay": fold["selected_decay"],
                "held_out_capped_skill": fold["held_out_score"]["capped"][
                    "skill_vs_zero"
                ],
                "held_out_capped_rms_ratio": fold["held_out_score"]["capped"][
                    "rms_ratio_vs_zero"
                ],
                "held_out_case_wins": fold["held_out_score"]["capped"][
                    "case_win_count"
                ],
                "fixed_zero_capped_skill": fold["fixed_zero_score"]["capped"][
                    "skill_vs_zero"
                ],
            }
            for fold in nested["folds"]
        ],
    )
    payload = with_payload_sha256(
        _json_safe(
            {
                "schema": SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "completed_readiness_passed"
                    if all(readiness.values())
                    else "completed_readiness_failed"
                ),
                "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
                "source_status": _git_status_short(SOURCE_PATHS),
                "input_hashes": parents["hashes"],
                "parent_payload_hashes": {
                    "a27": parents["a27_result"]["payload_sha256"],
                    "calibration": parents["calibration_result"]["payload_sha256"],
                    "teacher": parents["teacher_result"]["payload_sha256"],
                },
                "contract": {
                    "decays": MODAL_CAUSAL_OBSERVER_DECAYS,
                    "ridge": MODAL_CAUSAL_OBSERVER_RIDGE,
                    "groups": GROUPS,
                    "cases": CASES,
                    "calls": CALLS,
                    "bands": BANDS,
                    "active_cells": FROZEN_ACTIVE_CELLS,
                    "total_volume": TOTAL_VOLUME,
                    "maximum_relative_norm": MAXIMUM_RELATIVE_NORM,
                    "weighting": "equal calls within case then equal cases",
                    "selection": "nested leave-one-strength-out; ties choose smaller decay",
                    "temporal_label_holdout": (
                        "complete causal discrepancy prefix; held-band target labels excluded"
                    ),
                },
                "checks": checks,
                "selection": selection,
                "selected_decay": selected_decay,
                "nested_strength_crossfit": {
                    "score": nested["score"],
                    "fixed_zero_score": nested["fixed_zero_score"],
                    "folds": nested["folds"],
                    "selected_decay_counts": dict(sorted(selected_counts.items())),
                },
                "dual_held_summary": {
                    "cell_count": len(dual_scores),
                    "minimum_capped_skill": min(
                        score["capped"]["skill_vs_zero"] for score in dual_scores
                    ),
                    "failed_cell_count": sum(
                        score["capped"]["skill_vs_zero"] <= 0.0
                        or score["capped"]["case_win_count"] < 2
                        for score in dual_scores
                    ),
                    "total_harmful_rows": sum(
                        score["capped"]["harmful_row_count"] for score in dual_scores
                    ),
                },
                "leave_band_out_teacher": {
                    "minimum_capped_skill": min(
                        score["capped"]["skill_vs_zero"]
                        for score in teacher_band_scores
                    ),
                    "failed_band_count": sum(
                        score["capped"]["skill_vs_zero"] <= 0.0
                        or score["capped"]["case_win_count"] < 6
                        or score["capped"]["group_win_count"] < 3
                        or score["capped"]["harmful_row_count"] > 0
                        for score in teacher_band_scores
                    ),
                    "total_harmful_rows": sum(
                        score["capped"]["harmful_row_count"]
                        for score in teacher_band_scores
                    ),
                },
                "fixed_decay_teacher_scores": fixed_teacher,
                "frozen_full_teacher": teacher_score,
                "teacher_band_scores": teacher_full_band_scores,
                "strength_stability": strength_stability,
                "strength_stability_decay": dominant_decay,
                "strength_stability_fit_count": len(dominant_fits),
                "temporal_stability": temporal_stability,
                "diagnostics": diagnostics,
                "full_fit": {
                    key: full_fit[key]
                    for key in (
                        "family",
                        "decay",
                        "ridge",
                        "active_cells",
                        "feature_scale",
                        "original_coefficients",
                        "intercept",
                        "coefficient_frobenius_norm",
                        "singular_values",
                        "effective_rank",
                    )
                },
                "readiness": {"checks": readiness, "passed": all(readiness.values())},
                "decision": {
                    "authorizes_separate_interior_preregistration": all(
                        readiness.values()
                    ),
                    "authorizes_recurrence": False,
                },
                "artifact_hashes": {
                    "observer_scores.csv": sha256_file(
                        args.output_dir / "observer_scores.csv"
                    ),
                    "observer_folds.csv": sha256_file(
                        args.output_dir / "observer_folds.csv"
                    ),
                },
                "claim_boundary": (
                    "Adaptive-open zero-model shock-vortex diagnostic of one fixed "
                    "four-decay causal fine-discrepancy observer. Target labels are "
                    "offline-only. No recurrent, cross-family, Euler1D, conservation, "
                    "convergence, bump, off-grid, asymptotic-order, or Richardson claim."
                ),
            }
        )
    )
    atomic_write_json(args.output_dir / "causal_discrepancy_observer.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a27-result", type=Path, required=True)
    parser.add_argument("--a27-scores", type=Path, required=True)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--calibration-modal-records", type=Path, required=True)
    parser.add_argument("--calibration-audits", type=Path, required=True)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--teacher-modal-records", type=Path, required=True)
    parser.add_argument("--teacher-audits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
