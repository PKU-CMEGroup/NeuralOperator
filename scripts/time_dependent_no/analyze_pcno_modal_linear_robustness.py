#!/usr/bin/env python3
"""Audit cost, temporal extrapolation, and cap robustness of A26 maps."""

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
    MODAL_LINEAR_TIE_TOLERANCE,
    ModalSnapshot,
    cap_modal_predictions,
    fit_modal_linear_map,
    grouped_modal_linear_crossfit,
    modal_linear_fit_diagnostics,
    modal_linear_map_stability,
    predict_modal_linear_map,
    score_modal_linear_predictions,
    snapshots_from_modal_rows,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A27-SP19-LINEAR-MAP-ROBUSTNESS"
SCHEMA = "pcno_sp19_linear_map_robustness_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_modal_linear_robustness.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "a26_result": "2ddbd1791d0d35279887f754a1fe71ac026d0b62d9ef47683e99904f62f334ac",
    "a26_population_scores": "46f91973a5d08f10a2b2d891feddc96ddfd5cb7314ff7c46a49ab608c263e5d7",
    "a26_snapshot_scores": "0c4dde1b07f7d1496307edbb73712bd2bd557024fe67b0059a7b1cc256fe55b0",
    "a26_structure_benefits": "f41448c0e83c15419b116118cf4b89ac63c93de69227d5c8564369c75ebea78b",
    "a26_selection_decisions": "6d4657c96fff359b5e68a1bddb2c846702232dd585869eb62a1e11b121c4a2f1",
    "calibration_result": "d9533285d89e70e61c16833ceeb27cd542cd65f44925a746c94786df26faf8cd",
    "calibration_modal_records": "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb",
    "calibration_audits": "08cd55bc130b42ac9f48b252b86b71710bcdbef466bf77618365a83a0e70f9b0",
    "teacher_result": "bbbc3cdbd9afb64b8b8e63169b8acb9a9a4614957d6f4461412b4427405f63be",
    "teacher_modal_records": "0ac7915de0c47d519a8e9e3d61d2b582a3266e63b191f6951cf9033f4198c4e5",
    "teacher_audits": "8260ba50c1d29cceff1e1e4d13ba9d75408890a0124fbd6f4e1a57e17b288009",
}
EXPECTED_PAYLOADS = {
    "a26_result": "fc5c4cf8bfdd8dc3460fa2cd86ce95e59973a75df964ba0249490dc6e28d340b",
    "calibration_result": "0f4351cf92ae16beec11cba57ac5aa4973df5579e73e11cc8d0e616a25e93c59",
    "teacher_result": "1ba8b8e30d0e6b5bd7a7a7a2d4925b23fbb56b558180502243f08e7c393c6046",
}
MODE_CELLS = tuple(
    (mode, component) for mode in range(8) for component in range(4)
)
CALLS = tuple(range(30))
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
CANDIDATES = {
    "accuracy_three_call": {
        "family": "coarse_fine_full",
        "ridge": 1.0e-6,
        "calls_per_state": 3,
    },
    "cost_two_call": {
        "family": "fine_full",
        "ridge": 1.0e-6,
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
        "a26_result": args.a26_result,
        "a26_population_scores": args.a26_population_scores,
        "a26_snapshot_scores": args.a26_snapshot_scores,
        "a26_structure_benefits": args.a26_structure_benefits,
        "a26_selection_decisions": args.a26_selection_decisions,
        "calibration_result": args.calibration_result,
        "calibration_modal_records": args.calibration_modal_records,
        "calibration_audits": args.calibration_audits,
        "teacher_result": args.teacher_result,
        "teacher_modal_records": args.teacher_modal_records,
        "teacher_audits": args.teacher_audits,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("A27 inputs differ from the frozen artifacts")
    payloads = {
        "a26_result": _read_json(args.a26_result),
        "calibration_result": _read_json(args.calibration_result),
        "teacher_result": _read_json(args.teacher_result),
    }
    for name, payload in payloads.items():
        verify_payload_sha256(payload)
        if payload.get("payload_sha256") != EXPECTED_PAYLOADS[name]:
            raise ValueError(f"{name} payload differs from the frozen artifact")
    a26 = payloads["a26_result"]
    owned = {
        "a26_population_scores": "population_scores.csv",
        "a26_snapshot_scores": "snapshot_scores.csv",
        "a26_structure_benefits": "structure_benefits.csv",
        "a26_selection_decisions": "selection_decisions.csv",
    }
    for name, filename in owned.items():
        if a26.get("artifact_hashes", {}).get(filename) != EXPECTED_HASHES[name]:
            raise ValueError(f"A26 does not own {filename}")
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
    ):
        raise ValueError("A2 parents do not own their modal/audit inputs")
    if (
        teacher.get("calibration_sha256") != EXPECTED_HASHES["calibration_result"]
        or teacher.get("calibration_payload_sha256")
        != EXPECTED_PAYLOADS["calibration_result"]
    ):
        raise ValueError("teacher result is not bound to calibration")
    if (
        a26.get("final_selection", {}).get("selected", {}).get("family")
        != "coarse_fine_full"
        or float(
            a26.get("final_selection", {}).get("selected", {}).get("ridge", math.nan)
        )
        != 1.0e-6
    ):
        raise ValueError("A26 selected identity differs from A27")
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
    audit_rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[ModalSnapshot],
    *,
    population: str,
) -> np.ndarray:
    lookup = {}
    for row in audit_rows:
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
) -> tuple[list[ModalSnapshot], np.ndarray]:
    indices = [index for index, row in enumerate(snapshots) if predicate(row)]
    return [snapshots[index] for index in indices], native_rms[indices]


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
    ratios = cap["correction_to_native_increment"]
    resolved = cap["resolved"]
    return {
        "uncapped": uncapped,
        "capped": capped,
        "cap": {
            "row_count": len(snapshots),
            "resolved_count": int(np.sum(resolved)),
            "unresolved_count": int(np.sum(~resolved)),
            "active_count": int(np.sum(cap["cap_active"])),
            "minimum_scale": float(np.min(cap["scale"])),
            "maximum_raw_ratio": float(np.nanmax(cap["raw_correction_to_native_increment"])),
            "maximum_capped_ratio": float(np.nanmax(ratios)),
            "maximum_cap_violation": float(cap["maximum_cap_violation"]),
        },
    }


def _fit_and_score(
    train: Sequence[ModalSnapshot],
    test: Sequence[ModalSnapshot],
    test_native_rms: np.ndarray,
    *,
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    fit = fit_modal_linear_map(
        train,
        active_cells=FROZEN_ACTIVE_CELLS,
        family=candidate["family"],
        ridge=float(candidate["ridge"]),
    )
    prediction = predict_modal_linear_map(fit, test)
    return fit, _score_prediction(test, prediction, test_native_rms), prediction


def _score_row(
    *,
    candidate_name: str,
    evaluation: str,
    held_group: str | None,
    band: tuple[int, int] | None,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    capped = score["capped"]
    uncapped = score["uncapped"]
    cap = score["cap"]
    return {
        "candidate": candidate_name,
        "evaluation": evaluation,
        "held_group": held_group,
        "first_input_call": band[0] if band is not None else None,
        "last_input_call": band[1] if band is not None else None,
        "snapshot_count": capped["snapshot_count"],
        "case_count": capped["case_count"],
        "group_count": capped["group_count"],
        "uncapped_skill": uncapped["skill_vs_zero"],
        "uncapped_rms_ratio": uncapped["rms_ratio_vs_zero"],
        "uncapped_harmful_rows": uncapped["harmful_row_count"],
        "capped_skill": capped["skill_vs_zero"],
        "capped_rms_ratio": capped["rms_ratio_vs_zero"],
        "capped_case_wins": capped["case_win_count"],
        "capped_group_wins": capped["group_win_count"],
        "capped_harmful_rows": capped["harmful_row_count"],
        "maximum_case_rms_ratio": capped["maximum_case_rms_ratio"],
        "cap_active_count": cap["active_count"],
        "minimum_cap_scale": cap["minimum_scale"],
        "maximum_raw_correction_ratio": cap["maximum_raw_ratio"],
        "maximum_capped_correction_ratio": cap["maximum_capped_ratio"],
        "maximum_cap_violation": cap["maximum_cap_violation"],
        "status": capped["status"],
    }


def _candidate_analysis(
    candidate_name: str,
    candidate: Mapping[str, Any],
    calibration: Sequence[ModalSnapshot],
    calibration_native: np.ndarray,
    teacher: Sequence[ModalSnapshot],
    teacher_native: np.ndarray,
) -> dict[str, Any]:
    rows = []
    dual_scores = []
    for held_group in GROUPS["calibration"]:
        for band in BANDS:
            start, stop = band
            train = [
                row
                for row in calibration
                if row.group_id != held_group
                and not (start <= row.input_call <= stop)
            ]
            test, test_native = _subset(
                calibration,
                calibration_native,
                lambda row, group=held_group, lo=start, hi=stop: (
                    row.group_id == group and lo <= row.input_call <= hi
                ),
            )
            fit, score, _ = _fit_and_score(
                train,
                test,
                test_native,
                candidate=candidate,
            )
            dual_scores.append(score)
            rows.append(
                _score_row(
                    candidate_name=candidate_name,
                    evaluation="dual_held_calibration",
                    held_group=held_group,
                    band=band,
                    score=score,
                )
            )

    temporal_fits = []
    teacher_band_scores = []
    for band in BANDS:
        start, stop = band
        train = [
            row for row in calibration if not (start <= row.input_call <= stop)
        ]
        test, test_native = _subset(
            teacher,
            teacher_native,
            lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
        )
        fit, score, _ = _fit_and_score(
            train,
            test,
            test_native,
            candidate=candidate,
        )
        temporal_fits.append(fit)
        teacher_band_scores.append(score)
        rows.append(
            _score_row(
                candidate_name=candidate_name,
                evaluation="leave_band_out_teacher",
                held_group=None,
                band=band,
                score=score,
            )
        )

    full_fit, full_score, full_prediction = _fit_and_score(
        calibration,
        teacher,
        teacher_native,
        candidate=candidate,
    )
    rows.append(
        _score_row(
            candidate_name=candidate_name,
            evaluation="frozen_full_teacher",
            held_group=None,
            band=None,
            score=full_score,
        )
    )
    for band in BANDS:
        start, stop = band
        test, test_native = _subset(
            teacher,
            teacher_native,
            lambda row, lo=start, hi=stop: lo <= row.input_call <= hi,
        )
        indices = [
            index for index, row in enumerate(teacher) if start <= row.input_call <= stop
        ]
        score = _score_prediction(test, full_prediction[indices], test_native)
        rows.append(
            _score_row(
                candidate_name=candidate_name,
                evaluation="frozen_full_teacher_band",
                held_group=None,
                band=band,
                score=score,
            )
        )

    strength_crossfit = grouped_modal_linear_crossfit(
        calibration,
        active_cells=FROZEN_ACTIVE_CELLS,
        family=candidate["family"],
        ridge=float(candidate["ridge"]),
    )
    strength_stability = modal_linear_map_stability(strength_crossfit["fits"])
    temporal_stability = modal_linear_map_stability(temporal_fits)
    diagnostics = modal_linear_fit_diagnostics(full_fit, teacher)
    dual_pass = all(
        score["capped"]["skill_vs_zero"] is not None
        and score["capped"]["skill_vs_zero"] > 0.0
        and score["capped"]["case_win_count"] == 2
        for score in dual_scores
    )
    teacher_band_pass = all(
        score["capped"]["skill_vs_zero"] is not None
        and score["capped"]["skill_vs_zero"] > 0.0
        and score["capped"]["case_win_count"] == 6
        and score["capped"]["group_win_count"] == 3
        and score["capped"]["harmful_row_count"] == 0
        for score in teacher_band_scores
    )
    frozen_bands = [
        row for row in rows if row["evaluation"] == "frozen_full_teacher_band"
    ]
    all_scores = [*dual_scores, *teacher_band_scores, full_score]
    readiness = {
        "all_36_dual_held_cells_positive_and_two_case_wins": dual_pass,
        "all_four_leave_band_out_teacher_scores_pass": teacher_band_pass,
        "frozen_full_teacher_all_cases_and_groups_win": (
            full_score["capped"]["case_win_count"] == 6
            and full_score["capped"]["group_win_count"] == 3
        ),
        "frozen_full_teacher_all_bands_positive": all(
            row["capped_skill"] is not None and row["capped_skill"] > 0.0
            for row in frozen_bands
        ),
        "frozen_full_teacher_no_harmful_rows": full_score["capped"][
            "harmful_row_count"
        ]
        == 0,
        "all_cap_rows_resolved": all(
            score["cap"]["unresolved_count"] == 0 for score in all_scores
        ),
        "all_capped_ratios_at_most_0p05": max(
            score["cap"]["maximum_capped_ratio"] for score in all_scores
        )
        <= MAXIMUM_RELATIVE_NORM + MODAL_LINEAR_TIE_TOLERANCE,
        "all_cap_bookkeeping_closes": max(
            score["cap"]["maximum_cap_violation"] for score in all_scores
        )
        <= MODAL_LINEAR_TIE_TOLERANCE,
        "strength_minimum_map_cosine_exceeds_0p90": strength_stability[
            "minimum_pairwise_frobenius_cosine"
        ]
        > 0.90,
        "strength_map_norm_ratio_at_most_1p50": strength_stability[
            "maximum_to_minimum_norm_ratio"
        ]
        <= 1.50,
        "temporal_minimum_map_cosine_exceeds_0p90": temporal_stability[
            "minimum_pairwise_frobenius_cosine"
        ]
        > 0.90,
        "temporal_map_norm_ratio_at_most_1p50": temporal_stability[
            "maximum_to_minimum_norm_ratio"
        ]
        <= 1.50,
        "float32_prediction_change_at_most_1e_5": diagnostics[
            "maximum_float32_prediction_relative_change"
        ]
        is not None
        and diagnostics["maximum_float32_prediction_relative_change"] <= 1.0e-5,
    }
    return {
        "candidate": dict(candidate),
        "readiness": {"checks": readiness, "passed": all(readiness.values())},
        "score_rows": rows,
        "dual_held_summary": {
            "cell_count": len(dual_scores),
            "minimum_capped_skill": min(
                score["capped"]["skill_vs_zero"] for score in dual_scores
            ),
            "maximum_capped_rms_ratio": max(
                score["capped"]["rms_ratio_vs_zero"] for score in dual_scores
            ),
            "total_harmful_rows": sum(
                score["capped"]["harmful_row_count"] for score in dual_scores
            ),
        },
        "leave_band_out_teacher": {
            "minimum_capped_skill": min(
                score["capped"]["skill_vs_zero"] for score in teacher_band_scores
            ),
            "maximum_capped_rms_ratio": max(
                score["capped"]["rms_ratio_vs_zero"]
                for score in teacher_band_scores
            ),
            "total_harmful_rows": sum(
                score["capped"]["harmful_row_count"]
                for score in teacher_band_scores
            ),
        },
        "frozen_full_teacher": full_score,
        "strength_stability": strength_stability,
        "temporal_stability": temporal_stability,
        "diagnostics": diagnostics,
        "full_fit": {
            key: full_fit[key]
            for key in (
                "family",
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
        _read_csv(args.calibration_audits),
        calibration,
        population="calibration",
    )
    teacher_native = _native_rms(
        _read_csv(args.teacher_audits),
        teacher,
        population="teacher",
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
    cost_score = float(cost["frozen_full_teacher"]["capped"]["skill_vs_zero"])
    accuracy_score = float(
        accuracy["frozen_full_teacher"]["capped"]["skill_vs_zero"]
    )
    if (
        cost["readiness"]["passed"]
        and cost_score >= accuracy_score - 0.05
        and cost["frozen_full_teacher"]["capped"]["harmful_row_count"]
        <= accuracy["frozen_full_teacher"]["capped"]["harmful_row_count"]
    ):
        decision = "cost_two_call"
    elif accuracy["readiness"]["passed"]:
        decision = "accuracy_three_call"
    else:
        decision = "none"

    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "calibration_inventory_exact_18x30x32": len(calibration) == 540,
        "teacher_inventory_exact_6x30x32": len(teacher) == 180,
        "native_increment_audits_exact": len(calibration_native) == 540
        and len(teacher_native) == 180,
        "candidate_identity_exact": CANDIDATES
        == {
            "accuracy_three_call": {
                "family": "coarse_fine_full",
                "ridge": 1.0e-6,
                "calls_per_state": 3,
            },
            "cost_two_call": {
                "family": "fine_full",
                "ridge": 1.0e-6,
                "calls_per_state": 2,
            },
        },
        "dual_holdout_count_exact_36_each": all(
            result["dual_held_summary"]["cell_count"] == 36
            for result in results.values()
        ),
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "gradient_and_transfer_paths_unchanged": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A27 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_rows = [
        row for result in results.values() for row in result["score_rows"]
    ]
    write_csv(args.output_dir / "robustness_scores.csv", score_rows)
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
                    "a26": parents["a26_result"]["payload_sha256"],
                    "calibration": parents["calibration_result"]["payload_sha256"],
                    "teacher": parents["teacher_result"]["payload_sha256"],
                },
                "contract": {
                    "candidates": CANDIDATES,
                    "groups": GROUPS,
                    "cases": CASES,
                    "calls": CALLS,
                    "bands": BANDS,
                    "active_cells": FROZEN_ACTIVE_CELLS,
                    "total_volume": TOTAL_VOLUME,
                    "maximum_relative_norm": MAXIMUM_RELATIVE_NORM,
                    "weighting": "equal calls within case then equal cases",
                    "decision": (
                        "prefer cost if ready and within 0.05 teacher skill of accuracy; "
                        "else accuracy if ready; else none"
                    ),
                },
                "checks": checks,
                "candidates": results,
                "decision": {
                    "selected": decision,
                    "authorizes_execution": False,
                    "cost_teacher_skill": cost_score,
                    "accuracy_teacher_skill": accuracy_score,
                    "cost_within_0p05": cost_score >= accuracy_score - 0.05,
                },
                "artifact_hashes": {
                    "robustness_scores.csv": sha256_file(
                        args.output_dir / "robustness_scores.csv"
                    )
                },
                "claim_boundary": (
                    "Zero-model cost/cap/temporal robustness audit on already-open "
                    "teacher-forced shock-vortex rows. Selection can motivate only a "
                    "separately preregistered interior teacher-forced evaluation. No "
                    "recurrence, terminal safety, cross-family, conservation, "
                    "convergence, bump, off-grid, or Richardson claim."
                ),
            }
        )
    )
    atomic_write_json(args.output_dir / "modal_linear_robustness.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a26-result", type=Path, required=True)
    parser.add_argument("--a26-population-scores", type=Path, required=True)
    parser.add_argument("--a26-snapshot-scores", type=Path, required=True)
    parser.add_argument("--a26-structure-benefits", type=Path, required=True)
    parser.add_argument("--a26-selection-decisions", type=Path, required=True)
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
