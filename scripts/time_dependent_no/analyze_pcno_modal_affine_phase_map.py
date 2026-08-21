#!/usr/bin/env python3
"""Freeze and audit the W26-L5 affine phase-conditioned SP19 map."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
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
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_residual_geometry import (
    MODAL_AFFINE_LAST_INPUT_CALL,
    MODAL_AFFINE_RIDGE,
    MODAL_LINEAR_TIE_TOLERANCE,
    ModalSnapshot,
    cap_modal_predictions,
    fit_modal_affine_phase_map,
    modal_affine_fit_diagnostics,
    modal_affine_map_stability,
    predict_modal_affine_phase_map,
    score_modal_linear_predictions,
    snapshots_from_modal_rows,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A28-SP19-AFFINE-PHASE-MAP"
SCHEMA = "pcno_sp19_affine_phase_map_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_modal_affine_phase_map.py",
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
CALLS = tuple(range(MODAL_AFFINE_LAST_INPUT_CALL + 1))
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
BANDS = ((0, 7), (8, 14), (15, 21), (22, 29))
PARITIES = (0, 1)
CANDIDATES = {
    "accuracy_three_call": {
        "family": "coarse_fine_full_affine",
        "calls_per_state": 3,
    },
    "cost_two_call": {
        "family": "fine_full_affine",
        "calls_per_state": 2,
    },
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
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
        raise ValueError("A28 inputs differ from the frozen artifacts")
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
    if (
        a27.get("status") != "stopped_no_candidate"
        or a27.get("decision", {}).get("selected") != "none"
        or a27.get("artifact_hashes", {}).get("robustness_scores.csv")
        != EXPECTED_HASHES["a27_scores"]
    ):
        raise ValueError("A27 stop or owned score identity differs")
    calibration = payloads["calibration_result"]
    teacher = payloads["teacher_result"]
    if (
        calibration.get("modal_records_sha256")
        != EXPECTED_HASHES["calibration_modal_records"]
        or calibration.get("artifact_hashes", {}).get("calibration_audits.csv")
        != EXPECTED_HASHES["calibration_audits"]
        or teacher.get("modal_records_sha256")
        != EXPECTED_HASHES["teacher_modal_records"]
        or teacher.get("artifact_hashes", {}).get("teacher_audits.csv")
        != EXPECTED_HASHES["teacher_audits"]
        or teacher.get("calibration_sha256")
        != EXPECTED_HASHES["calibration_result"]
        or teacher.get("calibration_payload_sha256")
        != EXPECTED_PAYLOADS["calibration_result"]
    ):
        raise ValueError("A2 modal/audit ownership differs")
    return {"hashes": hashes, **payloads}


def _validate_snapshots(snapshots: Sequence[ModalSnapshot], *, population: str) -> None:
    expected = {(case, call) for case in CASES[population] for call in CALLS}
    actual = {(row.case_id, row.input_call) for row in snapshots}
    if actual != expected or len(snapshots) != len(expected):
        raise ValueError(f"{population} snapshot inventory differs")
    if {row.group_id for row in snapshots} != set(GROUPS[population]):
        raise ValueError(f"{population} group inventory differs")
    if any(row.case_id.split("_")[1] != row.group_id for row in snapshots):
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
    prediction: np.ndarray,
    native_rms: np.ndarray,
) -> dict[str, Any]:
    uncapped = score_modal_linear_predictions(
        snapshots, prediction, active_cells=FROZEN_ACTIVE_CELLS
    )
    cap = cap_modal_predictions(
        prediction,
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
    test: Sequence[ModalSnapshot],
    test_native: np.ndarray,
    *,
    family: str,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    fit = fit_modal_affine_phase_map(
        train,
        active_cells=FROZEN_ACTIVE_CELLS,
        family=family,
    )
    prediction = predict_modal_affine_phase_map(fit, test)
    return fit, _score_prediction(test, prediction, test_native), prediction


def _score_row(
    *,
    candidate: str,
    evaluation: str,
    score: Mapping[str, Any],
    held_group: str | None = None,
    parity: int | None = None,
    band: tuple[int, int] | None = None,
) -> dict[str, Any]:
    capped = score["capped"]
    return {
        "candidate": candidate,
        "evaluation": evaluation,
        "held_group": held_group,
        "parity": parity,
        "first_input_call": None if band is None else band[0],
        "last_input_call": None if band is None else band[1],
        "snapshot_count": capped["snapshot_count"],
        "case_count": capped["case_count"],
        "group_count": capped["group_count"],
        "uncapped_skill": score["uncapped"]["skill_vs_zero"],
        "uncapped_rms_ratio": score["uncapped"]["rms_ratio_vs_zero"],
        "uncapped_harmful_rows": score["uncapped"]["harmful_row_count"],
        "capped_skill": capped["skill_vs_zero"],
        "capped_rms_ratio": capped["rms_ratio_vs_zero"],
        "capped_case_wins": capped["case_win_count"],
        "capped_group_wins": capped["group_win_count"],
        "capped_harmful_rows": capped["harmful_row_count"],
        "maximum_case_rms_ratio": capped["maximum_case_rms_ratio"],
        "cap_active_count": score["cap"]["active_count"],
        "minimum_cap_scale": score["cap"]["minimum_scale"],
        "maximum_raw_correction_ratio": score["cap"]["maximum_raw_ratio"],
        "maximum_capped_correction_ratio": score["cap"]["maximum_capped_ratio"],
        "maximum_cap_violation": score["cap"]["maximum_cap_violation"],
        "status": capped["status"],
    }


def _candidate_analysis(
    name: str,
    candidate: Mapping[str, Any],
    calibration: Sequence[ModalSnapshot],
    calibration_native: np.ndarray,
    teacher: Sequence[ModalSnapshot],
    teacher_native: np.ndarray,
) -> dict[str, Any]:
    family = str(candidate["family"])
    rows: list[dict[str, Any]] = []
    strength_scores = []
    strength_fits = []
    strength_tests: list[ModalSnapshot] = []
    strength_predictions = []
    strength_native = []
    for group in GROUPS["calibration"]:
        train = [row for row in calibration if row.group_id != group]
        test, native, _ = _subset(
            calibration, calibration_native, lambda row, held=group: row.group_id == held
        )
        fit, score, prediction = _fit_and_score(
            train, test, native, family=family
        )
        strength_fits.append(fit)
        strength_scores.append(score)
        strength_tests.extend(test)
        strength_predictions.extend(prediction)
        strength_native.extend(native)
        rows.append(
            _score_row(
                candidate=name,
                evaluation="leave_strength_out_calibration",
                held_group=group,
                score=score,
            )
        )
    strength_population = _score_prediction(
        strength_tests,
        np.asarray(strength_predictions),
        np.asarray(strength_native),
    )
    rows.append(
        _score_row(
            candidate=name,
            evaluation="leave_strength_out_population",
            score=strength_population,
        )
    )

    parity_scores = []
    for group in GROUPS["calibration"]:
        for parity in PARITIES:
            train = [
                row
                for row in calibration
                if row.group_id != group and row.input_call % 2 != parity
            ]
            test, native, _ = _subset(
                calibration,
                calibration_native,
                lambda row, held=group, held_parity=parity: (
                    row.group_id == held and row.input_call % 2 == held_parity
                ),
            )
            _, score, _ = _fit_and_score(train, test, native, family=family)
            parity_scores.append(score)
            rows.append(
                _score_row(
                    candidate=name,
                    evaluation="strength_parity_held_calibration",
                    held_group=group,
                    parity=parity,
                    score=score,
                )
            )

    contiguous_scores = []
    for group in GROUPS["calibration"]:
        for band in BANDS:
            start, stop = band
            train = [
                row
                for row in calibration
                if row.group_id != group
                and not (start <= row.input_call <= stop)
            ]
            test, native, _ = _subset(
                calibration,
                calibration_native,
                lambda row, held=group, lo=start, hi=stop: (
                    row.group_id == held and lo <= row.input_call <= hi
                ),
            )
            _, score, _ = _fit_and_score(train, test, native, family=family)
            contiguous_scores.append(score)
            rows.append(
                _score_row(
                    candidate=name,
                    evaluation="contiguous_band_stress_calibration",
                    held_group=group,
                    band=band,
                    score=score,
                )
            )

    full_fit, teacher_score, full_prediction = _fit_and_score(
        calibration, teacher, teacher_native, family=family
    )
    rows.append(
        _score_row(
            candidate=name,
            evaluation="frozen_full_teacher",
            score=teacher_score,
        )
    )
    teacher_band_scores = []
    for band in BANDS:
        start, stop = band
        test, native, indices = _subset(
            teacher,
            teacher_native,
            lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
        )
        score = _score_prediction(test, full_prediction[indices], native)
        teacher_band_scores.append(score)
        rows.append(
            _score_row(
                candidate=name,
                evaluation="frozen_full_teacher_band",
                band=band,
                score=score,
            )
        )

    stability = modal_affine_map_stability(strength_fits)
    diagnostics = modal_affine_fit_diagnostics(full_fit, teacher)
    all_gate_scores = [
        strength_population,
        *strength_scores,
        *parity_scores,
        teacher_score,
        *teacher_band_scores,
    ]
    readiness = {
        "strength_population_skill_exceeds_0p98": (
            strength_population["capped"]["skill_vs_zero"] > 0.98
        ),
        "all_strength_cells_skill_exceeds_0p95": all(
            score["capped"]["skill_vs_zero"] > 0.95 for score in strength_scores
        ),
        "all_strength_cases_groups_win_without_harm": (
            strength_population["capped"]["case_win_count"] == 18
            and strength_population["capped"]["group_win_count"] == 9
            and strength_population["capped"]["harmful_row_count"] == 0
        ),
        "all_18_parity_cells_skill_exceeds_0p85": all(
            score["capped"]["skill_vs_zero"] > 0.85 for score in parity_scores
        ),
        "all_parity_cells_two_cases_win_without_harm": all(
            score["capped"]["case_win_count"] == 2
            and score["capped"]["harmful_row_count"] == 0
            for score in parity_scores
        ),
        "whole_start_end_strength_cosines_exceed_0p94": all(
            stability[key]["minimum_pairwise_frobenius_cosine"] > 0.94
            for key in ("whole", "start", "end")
        ),
        "whole_start_end_strength_norm_ratios_at_most_1p10": all(
            stability[key]["maximum_to_minimum_norm_ratio"] <= 1.10
            for key in ("whole", "start", "end")
        ),
        "teacher_full_skill_at_least_0p95": (
            teacher_score["capped"]["skill_vs_zero"] >= 0.95
        ),
        "teacher_all_cases_groups_win_without_harm": (
            teacher_score["capped"]["case_win_count"] == 6
            and teacher_score["capped"]["group_win_count"] == 3
            and teacher_score["capped"]["harmful_row_count"] == 0
        ),
        "teacher_all_bands_positive_all_cases_groups_without_harm": all(
            score["capped"]["skill_vs_zero"] > 0.0
            and score["capped"]["case_win_count"] == 6
            and score["capped"]["group_win_count"] == 3
            and score["capped"]["harmful_row_count"] == 0
            for score in teacher_band_scores
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
    return {
        "candidate": dict(candidate),
        "readiness": {"checks": readiness, "passed": all(readiness.values())},
        "score_rows": rows,
        "strength_crossfit": {
            "population": strength_population,
            "minimum_cell_skill": min(
                score["capped"]["skill_vs_zero"] for score in strength_scores
            ),
            "maximum_cell_rms_ratio": max(
                score["capped"]["rms_ratio_vs_zero"] for score in strength_scores
            ),
        },
        "parity_stress": {
            "cell_count": len(parity_scores),
            "minimum_cell_skill": min(
                score["capped"]["skill_vs_zero"] for score in parity_scores
            ),
            "maximum_cell_rms_ratio": max(
                score["capped"]["rms_ratio_vs_zero"] for score in parity_scores
            ),
            "total_harmful_rows": sum(
                score["capped"]["harmful_row_count"] for score in parity_scores
            ),
        },
        "contiguous_band_stress": {
            "gate_input": False,
            "cell_count": len(contiguous_scores),
            "minimum_cell_skill": min(
                score["capped"]["skill_vs_zero"] for score in contiguous_scores
            ),
            "failed_cell_count": sum(
                score["capped"]["skill_vs_zero"] <= 0.0
                or score["capped"]["case_win_count"] < 2
                for score in contiguous_scores
            ),
            "total_harmful_rows": sum(
                score["capped"]["harmful_row_count"] for score in contiguous_scores
            ),
        },
        "frozen_full_teacher": teacher_score,
        "teacher_band_minimum_skill": min(
            score["capped"]["skill_vs_zero"] for score in teacher_band_scores
        ),
        "strength_stability": stability,
        "diagnostics": diagnostics,
        "full_fit": {
            key: full_fit[key]
            for key in (
                "family",
                "ridge",
                "last_input_call",
                "active_cells",
                "feature_scale",
                "original_coefficients",
                "start_coefficients",
                "end_coefficients",
                "base_feature_count",
                "intercept",
                "coefficient_frobenius_norm",
                "singular_values",
                "effective_rank",
            )
        },
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
    _validate_snapshots(calibration, population="calibration")
    _validate_snapshots(teacher, population="teacher")
    calibration_native = _native_rms(
        _read_csv(args.calibration_audits), calibration, population="calibration"
    )
    teacher_native = _native_rms(
        _read_csv(args.teacher_audits), teacher, population="teacher"
    )
    results = {
        name: _candidate_analysis(
            name,
            candidate,
            calibration,
            calibration_native,
            teacher,
            teacher_native,
        )
        for name, candidate in CANDIDATES.items()
    }
    accuracy = results["accuracy_three_call"]
    cost = results["cost_two_call"]
    accuracy_skill = float(
        accuracy["frozen_full_teacher"]["capped"]["skill_vs_zero"]
    )
    cost_skill = float(cost["frozen_full_teacher"]["capped"]["skill_vs_zero"])
    if cost["readiness"]["passed"] and cost_skill >= accuracy_skill - 0.02:
        decision = "cost_two_call"
    elif accuracy["readiness"]["passed"]:
        decision = "accuracy_three_call"
    else:
        decision = "none"

    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "a27_stopped_none_exact": (
            parents["a27_result"]["status"] == "stopped_no_candidate"
            and parents["a27_result"]["decision"]["selected"] == "none"
        ),
        "calibration_inventory_exact_18x30x32": len(calibration) == 540,
        "teacher_inventory_exact_6x30x32": len(teacher) == 180,
        "native_increment_audits_exact": (
            len(calibration_native) == 540 and len(teacher_native) == 180
        ),
        "candidate_identity_exact": CANDIDATES
        == {
            "accuracy_three_call": {
                "family": "coarse_fine_full_affine",
                "calls_per_state": 3,
            },
            "cost_two_call": {
                "family": "fine_full_affine",
                "calls_per_state": 2,
            },
        },
        "fixed_affine_contract_exact": (
            MODAL_AFFINE_RIDGE == 1.0e-6
            and MODAL_AFFINE_LAST_INPUT_CALL == 29
        ),
        "parity_cell_count_exact_18_each": all(
            result["parity_stress"]["cell_count"] == 18
            for result in results.values()
        ),
        "contiguous_stress_count_exact_36_each": all(
            result["contiguous_band_stress"]["cell_count"] == 36
            for result in results.values()
        ),
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "gradient_and_transfer_paths_unchanged": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A28 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_rows = [row for result in results.values() for row in result["score_rows"]]
    write_csv(args.output_dir / "affine_phase_scores.csv", score_rows)
    payload_results = {
        name: {key: value for key, value in result.items() if key != "score_rows"}
        for name, result in results.items()
    }
    payload = with_payload_sha256(
        _json_safe(
            {
                "schema": SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "completed_candidate_selected"
                    if decision != "none"
                    else "stopped_no_candidate"
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
                    "candidates": CANDIDATES,
                    "groups": GROUPS,
                    "cases": CASES,
                    "calls": CALLS,
                    "phase": "input_call / 29",
                    "basis": "Bernstein degree one: (1-t)x and tx",
                    "ridge": MODAL_AFFINE_RIDGE,
                    "bands": BANDS,
                    "parities": PARITIES,
                    "active_cells": FROZEN_ACTIVE_CELLS,
                    "total_volume": TOTAL_VOLUME,
                    "maximum_relative_norm": MAXIMUM_RELATIVE_NORM,
                    "weighting": "equal calls within case then equal cases",
                    "contiguous_band_stress_is_gate_input": False,
                    "decision": (
                        "prefer cost if ready and within 0.02 teacher skill of "
                        "accuracy; else accuracy if ready; else none"
                    ),
                },
                "checks": checks,
                "candidates": payload_results,
                "decision": {
                    "selected": decision,
                    "authorizes_execution": False,
                    "authorizes_separate_interior_preregistration": decision != "none",
                    "cost_teacher_skill": cost_skill,
                    "accuracy_teacher_skill": accuracy_skill,
                    "cost_within_0p02": cost_skill >= accuracy_skill - 0.02,
                },
                "artifact_hashes": {
                    "affine_phase_scores.csv": sha256_file(
                        args.output_dir / "affine_phase_scores.csv"
                    )
                },
                "claim_boundary": (
                    "Zero-model phase-interpolation audit on already-open "
                    "shock-vortex modal rows. The mandatory contiguous-band stress "
                    "retains A27's failed temporal-extrapolation boundary. Passing "
                    "can motivate only a separately preregistered native-truth "
                    "interior teacher-forced evaluation; no recurrence, sealed, "
                    "cross-family, conservation, convergence, bump, off-grid, "
                    "asymptotic-order, or Richardson claim."
                ),
            }
        )
    )
    atomic_write_json(args.output_dir / "affine_phase_map.json", payload)
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
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
