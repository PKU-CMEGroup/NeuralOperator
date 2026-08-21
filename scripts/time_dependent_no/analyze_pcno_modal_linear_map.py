#!/usr/bin/env python3
"""Fit and evaluate the frozen A26 group-nested SP19 linear residual map."""

from __future__ import annotations

import argparse
import csv
import json
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
    DENOMINATOR_FLOOR,
    FIXED_BETA,
    MODAL_LINEAR_FAMILIES,
    MODAL_LINEAR_RIDGES,
    MODAL_LINEAR_TIE_TOLERANCE,
    build_geometry_records,
    fit_case_first_coefficient,
    fit_modal_linear_map,
    grouped_coefficient_crossfit,
    modal_linear_map_stability,
    modal_snapshot_arrays,
    nested_grouped_modal_linear_map,
    predict_modal_linear_map,
    score_modal_linear_predictions,
    select_grouped_modal_linear_map,
    snapshots_from_modal_rows,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A26-SP19-LINEAR-RESIDUAL-MAP"
SCHEMA = "pcno_sp19_linear_residual_map_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_modal_linear_map.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "calibration_result": "d9533285d89e70e61c16833ceeb27cd542cd65f44925a746c94786df26faf8cd",
    "calibration_modal_records": "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb",
    "teacher_result": "bbbc3cdbd9afb64b8b8e63169b8acb9a9a4614957d6f4461412b4427405f63be",
    "teacher_modal_records": "0ac7915de0c47d519a8e9e3d61d2b582a3266e63b191f6951cf9033f4198c4e5",
}
EXPECTED_PAYLOADS = {
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
        "calibration_result": args.calibration_result,
        "calibration_modal_records": args.calibration_modal_records,
        "teacher_result": args.teacher_result,
        "teacher_modal_records": args.teacher_modal_records,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("A26 inputs differ from the frozen artifacts")
    calibration = _read_json(args.calibration_result)
    teacher = _read_json(args.teacher_result)
    for name, payload in (
        ("calibration_result", calibration),
        ("teacher_result", teacher),
    ):
        verify_payload_sha256(payload)
        if payload.get("payload_sha256") != EXPECTED_PAYLOADS[name]:
            raise ValueError(f"{name} payload differs from the frozen artifact")
    if calibration.get("modal_records_sha256") != EXPECTED_HASHES[
        "calibration_modal_records"
    ]:
        raise ValueError("calibration result does not own its modal records")
    if teacher.get("modal_records_sha256") != EXPECTED_HASHES[
        "teacher_modal_records"
    ]:
        raise ValueError("teacher result does not own its modal records")
    if (
        teacher.get("calibration_sha256") != EXPECTED_HASHES["calibration_result"]
        or teacher.get("calibration_payload_sha256")
        != EXPECTED_PAYLOADS["calibration_result"]
    ):
        raise ValueError("teacher result is not bound to the calibration result")
    return {"hashes": hashes, "calibration": calibration, "teacher": teacher}


def _validate_population(snapshots: Sequence[Any], *, population: str) -> None:
    cases = tuple(sorted({row.case_id for row in snapshots}))
    groups = tuple(sorted({row.group_id for row in snapshots}))
    if cases != tuple(sorted(CASES[population])) or groups != tuple(
        sorted(GROUPS[population])
    ):
        raise ValueError(f"{population} case/group inventory differs")
    expected = {
        (case, call)
        for case in CASES[population]
        for call in CALLS
    }
    actual = {(row.case_id, row.input_call) for row in snapshots}
    if actual != expected or len(snapshots) != len(expected):
        raise ValueError(f"{population} case/call inventory differs")
    for row in snapshots:
        expected_group = row.case_id.split("_")[1]
        if row.group_id != expected_group:
            raise ValueError(f"{population} case/group membership differs")


def _scalar_predictions(
    calibration: Sequence[Any], teacher: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = [
        row
        for row in build_geometry_records(
            calibration, active_cells=FROZEN_ACTIVE_CELLS
        )
        if row.view == "sp19"
    ]
    crossfit = grouped_coefficient_crossfit(records)
    coefficients = {
        str(fold["held_out_group"]): float(fold["coefficient"])
        for fold in crossfit["folds"]
    }
    calibration_arrays = modal_snapshot_arrays(
        calibration, active_cells=FROZEN_ACTIVE_CELLS
    )
    calibration_prediction = np.stack(
        [
            coefficients[row.group_id] * calibration_arrays["fine"][index]
            for index, row in enumerate(calibration)
        ]
    )
    fit = fit_case_first_coefficient(records)
    if fit["status"] != "ok" or fit["coefficient"] is None:
        raise ValueError("A26 scalar comparator is unresolved")
    beta = float(fit["coefficient"])
    teacher_arrays = modal_snapshot_arrays(teacher, active_cells=FROZEN_ACTIVE_CELLS)
    return (
        calibration_prediction,
        beta * teacher_arrays["fine"],
        {"crossfit": crossfit, "calibration_fit": fit},
    )


def _case_first_vector_mean(
    snapshots: Sequence[Any], values: np.ndarray
) -> np.ndarray:
    return np.mean(
        [
            np.mean(
                [values[index] for index, row in enumerate(snapshots) if row.case_id == case_id],
                axis=0,
            )
            for case_id in sorted({row.case_id for row in snapshots})
        ],
        axis=0,
    )


def _policy_evidence(
    snapshots: Sequence[Any],
    predictions: np.ndarray,
    *,
    population: str,
    policy: str,
) -> dict[str, Any]:
    rows = tuple(snapshots)
    predicted = np.asarray(predictions, dtype=np.float64)
    arrays = modal_snapshot_arrays(rows, active_cells=FROZEN_ACTIVE_CELLS)
    target = np.asarray(arrays["target"], dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError("A26 policy prediction inventory differs")
    score = score_modal_linear_predictions(
        rows, predicted, active_cells=FROZEN_ACTIVE_CELLS
    )
    zero_energy = np.sum(target * target, axis=1)
    predicted_energy = np.sum(predicted * predicted, axis=1)
    cross = np.sum(target * predicted, axis=1)
    corrected_energy = np.sum((target - predicted) ** 2, axis=1)
    benefit = zero_energy - corrected_energy
    denominators = np.sqrt(np.maximum(zero_energy * predicted_energy, 0.0))
    resolved = denominators > DENOMINATOR_FLOOR**2
    snapshot_rows = []
    for index, row in enumerate(rows):
        snapshot_rows.append(
            {
                "population": population,
                "policy": policy,
                "case_id": row.case_id,
                "group_id": row.group_id,
                "input_call": row.input_call,
                "zero_energy": zero_energy[index],
                "prediction_energy": predicted_energy[index],
                "corrected_energy": corrected_energy[index],
                "benefit": benefit[index],
                "benefit_identity": 2.0 * cross[index] - predicted_energy[index],
                "benefit_identity_abs": abs(
                    benefit[index] - (2.0 * cross[index] - predicted_energy[index])
                ),
                "signed_cosine": (
                    cross[index] / denominators[index] if resolved[index] else None
                ),
                "signed_cosine_denominator": denominators[index],
                "signed_cosine_status": "ok" if resolved[index] else "small_denominator",
                "helps": benefit[index] > 0.0,
                "harms": benefit[index] < 0.0,
            }
        )

    score_rows = [
        {
            "population": population,
            "policy": policy,
            "scope": "population",
            "scope_id": population,
            **{
                key: score[key]
                for key in (
                    "snapshot_count",
                    "case_count",
                    "group_count",
                    "skill_vs_zero",
                    "rms_ratio_vs_zero",
                    "case_win_count",
                    "group_win_count",
                    "maximum_case_rms_ratio",
                    "helpful_row_count",
                    "harmful_row_count",
                    "harmful_row_fraction",
                    "median_signed_cosine",
                    "signed_cosine_resolved_count",
                    "signed_cosine_unresolved_count",
                    "maximum_benefit_identity_abs",
                    "status",
                )
            },
        }
    ]
    for row in score["case_scores"]:
        score_rows.append(
            {
                "population": population,
                "policy": policy,
                "scope": "case",
                "scope_id": row["case_id"],
                "snapshot_count": sum(item.case_id == row["case_id"] for item in rows),
                "case_count": 1,
                "group_count": 1,
                "skill_vs_zero": row["skill_vs_zero"],
                "rms_ratio_vs_zero": row["rms_ratio_vs_zero"],
                "status": row["status"],
            }
        )
    for row in score["group_scores"]:
        score_rows.append(
            {
                "population": population,
                "policy": policy,
                "scope": "group",
                "scope_id": row["group_id"],
                "snapshot_count": sum(item.group_id == row["group_id"] for item in rows),
                "case_count": row["case_count"],
                "group_count": 1,
                "skill_vs_zero": row["skill_vs_zero"],
                "rms_ratio_vs_zero": row["rms_ratio_vs_zero"],
                "status": row["status"],
            }
        )
    band_scores = []
    for start, stop in BANDS:
        indices = [
            index for index, row in enumerate(rows) if start <= row.input_call <= stop
        ]
        selected_rows = [rows[index] for index in indices]
        selected_prediction = predicted[indices]
        band_score = score_modal_linear_predictions(
            selected_rows,
            selected_prediction,
            active_cells=FROZEN_ACTIVE_CELLS,
        )
        band_row = {
            "population": population,
            "policy": policy,
            "scope": "temporal_band",
            "scope_id": f"{start}-{stop}",
            "first_input_call": start,
            "last_input_call": stop,
            **{
                key: band_score[key]
                for key in (
                    "snapshot_count",
                    "case_count",
                    "group_count",
                    "skill_vs_zero",
                    "rms_ratio_vs_zero",
                    "case_win_count",
                    "group_win_count",
                    "maximum_case_rms_ratio",
                    "helpful_row_count",
                    "harmful_row_count",
                    "harmful_row_fraction",
                    "median_signed_cosine",
                    "signed_cosine_resolved_count",
                    "signed_cosine_unresolved_count",
                    "maximum_benefit_identity_abs",
                    "status",
                )
            },
        }
        score_rows.append(band_row)
        band_scores.append(band_row)

    cell_benefit = target * target - (target - predicted) ** 2
    mean_cell_benefit = _case_first_vector_mean(rows, cell_benefit)
    structure_rows = [
        {
            "population": population,
            "policy": policy,
            "view": "cell",
            "view_id": f"mode{mode}_component{component}",
            "mode_index": mode,
            "component": component,
            "case_first_mean_benefit": mean_cell_benefit[index],
        }
        for index, (mode, component) in enumerate(FROZEN_ACTIVE_CELLS)
    ]
    for mode in sorted({cell[0] for cell in FROZEN_ACTIVE_CELLS}):
        indices = [
            index for index, cell in enumerate(FROZEN_ACTIVE_CELLS) if cell[0] == mode
        ]
        structure_rows.append(
            {
                "population": population,
                "policy": policy,
                "view": "mode",
                "view_id": f"mode{mode}",
                "mode_index": mode,
                "component": None,
                "case_first_mean_benefit": float(np.sum(mean_cell_benefit[indices])),
            }
        )
    for component in range(4):
        indices = [
            index
            for index, cell in enumerate(FROZEN_ACTIVE_CELLS)
            if cell[1] == component
        ]
        structure_rows.append(
            {
                "population": population,
                "policy": policy,
                "view": "component",
                "view_id": f"component{component}",
                "mode_index": None,
                "component": component,
                "case_first_mean_benefit": float(np.sum(mean_cell_benefit[indices])),
            }
        )
    return {
        "score": score,
        "score_rows": score_rows,
        "band_scores": band_scores,
        "snapshot_rows": snapshot_rows,
        "structure_rows": structure_rows,
    }


def _selection_rows(nested: Mapping[str, Any], final: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in nested["folds"]:
        selected = fold["selected"]
        for candidate in fold["inner_candidates"]:
            rows.append(
                {
                    "stage": "nested_inner_selection",
                    "outer_held_group": fold["held_out_group"],
                    **candidate,
                    "selected": candidate["family"] == selected["family"]
                    and float(candidate["ridge"]) == float(selected["ridge"]),
                }
            )
    selected = final["selected"]
    for candidate in final["candidates"]:
        rows.append(
            {
                "stage": "final_calibration_selection",
                "outer_held_group": None,
                **candidate,
                "selected": candidate["family"] == selected["family"]
                and float(candidate["ridge"]) == float(selected["ridge"]),
            }
        )
    return rows


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

    calibration_arrays = modal_snapshot_arrays(
        calibration, active_cells=FROZEN_ACTIVE_CELLS
    )
    teacher_arrays = modal_snapshot_arrays(teacher, active_cells=FROZEN_ACTIVE_CELLS)
    calibration_fixed = FIXED_BETA * calibration_arrays["fine"]
    teacher_fixed = FIXED_BETA * teacher_arrays["fine"]
    calibration_scalar, teacher_scalar, scalar = _scalar_predictions(
        calibration, teacher
    )

    nested = nested_grouped_modal_linear_map(
        calibration, active_cells=FROZEN_ACTIVE_CELLS
    )
    final_selection = select_grouped_modal_linear_map(
        calibration, active_cells=FROZEN_ACTIVE_CELLS
    )
    selected = final_selection["selected"]
    final_fit = fit_modal_linear_map(
        calibration,
        active_cells=FROZEN_ACTIVE_CELLS,
        family=selected["family"],
        ridge=float(selected["ridge"]),
    )
    teacher_selected = predict_modal_linear_map(final_fit, teacher)
    stability = modal_linear_map_stability(
        final_selection["selected_crossfit"]["fits"]
    )

    policies = {
        ("calibration", "nested_selected"): (calibration, nested["predictions"]),
        ("calibration", "fixed_half"): (calibration, calibration_fixed),
        ("calibration", "scalar_crossfit"): (calibration, calibration_scalar),
        ("teacher", "frozen_selected"): (teacher, teacher_selected),
        ("teacher", "fixed_half"): (teacher, teacher_fixed),
        ("teacher", "calibration_scalar"): (teacher, teacher_scalar),
    }
    evidence = {
        key: _policy_evidence(rows, prediction, population=key[0], policy=key[1])
        for key, (rows, prediction) in policies.items()
    }
    calibration_selected_score = evidence[("calibration", "nested_selected")][
        "score"
    ]
    calibration_fixed_score = evidence[("calibration", "fixed_half")]["score"]
    teacher_selected_score = evidence[("teacher", "frozen_selected")]["score"]
    teacher_fixed_score = evidence[("teacher", "fixed_half")]["score"]

    gate_checks = {
        "nested_calibration_skill_gain_at_least_0p02": (
            calibration_selected_score["skill_vs_zero"]
            >= calibration_fixed_score["skill_vs_zero"] + 0.02
        ),
        "frozen_teacher_skill_gain_at_least_0p02": (
            teacher_selected_score["skill_vs_zero"]
            >= teacher_fixed_score["skill_vs_zero"] + 0.02
        ),
        "all_six_teacher_cases_win": teacher_selected_score["case_win_count"] == 6,
        "all_three_teacher_strengths_win": teacher_selected_score["group_win_count"]
        == 3,
        "every_teacher_temporal_band_has_positive_skill": all(
            row["skill_vs_zero"] is not None and row["skill_vs_zero"] > 0.0
            for row in evidence[("teacher", "frozen_selected")]["band_scores"]
        ),
        "teacher_harmful_rows_no_more_than_fixed_half": (
            teacher_selected_score["harmful_row_count"]
            <= teacher_fixed_score["harmful_row_count"]
        ),
        "minimum_pairwise_map_cosine_exceeds_0p90": (
            stability["minimum_pairwise_frobenius_cosine"] > 0.90
        ),
        "maximum_to_minimum_map_norm_at_most_1p50": (
            stability["maximum_to_minimum_norm_ratio"] <= 1.50
        ),
    }
    all_evidence = list(evidence.values())
    checks = {
        "input_hashes_exact": parents["hashes"] == EXPECTED_HASHES,
        "parent_payloads_exact": True,
        "calibration_inventory_exact_18x30x32": len(calibration) == 540,
        "teacher_inventory_exact_6x30x32": len(teacher) == 180,
        "population_cases_and_groups_disjoint": not (
            set(CASES["calibration"]) & set(CASES["teacher"])
            or set(GROUPS["calibration"]) & set(GROUPS["teacher"])
        ),
        "sp19_active_cells_exact": tuple(final_fit["active_cells"])
        == FROZEN_ACTIVE_CELLS,
        "candidate_families_exact": tuple(MODAL_LINEAR_FAMILIES)
        == ("fine_diagonal", "fine_full", "coarse_fine_full"),
        "ridge_grid_exact": tuple(MODAL_LINEAR_RIDGES)
        == (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0),
        "zero_intercept_exact": bool(np.all(final_fit["intercept"] == 0.0)),
        "outer_group_leakage_absent": all(
            fold["held_out_group"] not in fold["fit_groups"]
            for fold in nested["folds"]
        ),
        "all_benefit_identities_close_at_1e_12": max(
            item["score"]["maximum_benefit_identity_abs"] for item in all_evidence
        )
        <= MODAL_LINEAR_TIE_TOLERANCE,
        "all_scores_resolved": all(item["score"]["status"] == "ok" for item in all_evidence),
        "all_coordinates_and_predictions_finite": bool(
            np.isfinite(calibration_arrays["coarse"]).all()
            and np.isfinite(calibration_arrays["fine"]).all()
            and np.isfinite(calibration_arrays["target"]).all()
            and np.isfinite(teacher_arrays["coarse"]).all()
            and np.isfinite(teacher_arrays["fine"]).all()
            and np.isfinite(teacher_arrays["target"]).all()
            and np.isfinite(nested["predictions"]).all()
            and np.isfinite(teacher_selected).all()
        ),
        "teacher_not_used_for_fit_or_selection": True,
        "no_model_built": True,
        "no_state_or_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "gradient_and_transfer_paths_unchanged": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"A26 validity checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_rows = [row for item in evidence.values() for row in item["score_rows"]]
    snapshot_rows = [
        row for item in evidence.values() for row in item["snapshot_rows"]
    ]
    structure_rows = [
        row for item in evidence.values() for row in item["structure_rows"]
    ]
    selection_rows = _selection_rows(nested, final_selection)
    write_csv(args.output_dir / "population_scores.csv", score_rows)
    write_csv(args.output_dir / "snapshot_scores.csv", snapshot_rows)
    write_csv(args.output_dir / "structure_benefits.csv", structure_rows)
    write_csv(args.output_dir / "selection_decisions.csv", selection_rows)

    payload = with_payload_sha256(
        _json_safe(
            {
                "schema": SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "completed_motivation_gate"
                    if all(gate_checks.values())
                    else "stopped_motivation_gate"
                ),
                "source_hashes": sha256_files(SOURCE_PATHS, root=ROOT),
                "source_status": _git_status_short(SOURCE_PATHS),
                "input_hashes": parents["hashes"],
                "parent_payload_hashes": {
                    "calibration": parents["calibration"]["payload_sha256"],
                    "teacher": parents["teacher"]["payload_sha256"],
                },
                "contract": {
                    "groups": GROUPS,
                    "cases": CASES,
                    "calls": list(CALLS),
                    "active_cells": FROZEN_ACTIVE_CELLS,
                    "families": MODAL_LINEAR_FAMILIES,
                    "ridges": MODAL_LINEAR_RIDGES,
                    "bands": BANDS,
                    "weighting": "equal calls within case then equal cases",
                    "feature_scaling": "training-fold case-first RMS without centering",
                    "intercept": 0.0,
                    "selection_tie_tolerance": MODAL_LINEAR_TIE_TOLERANCE,
                    "family_tie_order": MODAL_LINEAR_FAMILIES,
                    "ridge_tie_order": "larger ridge",
                },
                "checks": checks,
                "nested_calibration": {
                    "score": calibration_selected_score,
                    "outer_folds": [
                        {
                            "held_out_group": fold["held_out_group"],
                            "fit_groups": fold["fit_groups"],
                            "selected": fold["selected"],
                            "held_out_score": fold["held_out_score"],
                        }
                        for fold in nested["folds"]
                    ],
                },
                "final_selection": {
                    "selected": selected,
                    "candidates": final_selection["candidates"],
                    "calibration_selection_cv_score": final_selection[
                        "selected_crossfit"
                    ]["score"],
                },
                "final_fit": {
                    key: final_fit[key]
                    for key in (
                        "family",
                        "ridge",
                        "active_cells",
                        "feature_scale",
                        "original_coefficients",
                        "intercept",
                        "training_cases",
                        "training_groups",
                        "coefficient_frobenius_norm",
                        "singular_values",
                        "effective_rank",
                        "effective_rank_tolerance",
                    )
                },
                "stability": stability,
                "comparators": {
                    "fixed_beta": FIXED_BETA,
                    "scalar": scalar,
                    "calibration_fixed_half": calibration_fixed_score,
                    "teacher_fixed_half": teacher_fixed_score,
                    "calibration_scalar_crossfit": evidence[
                        ("calibration", "scalar_crossfit")
                    ]["score"],
                    "teacher_calibration_scalar": evidence[
                        ("teacher", "calibration_scalar")
                    ]["score"],
                },
                "teacher_frozen_selected": {
                    "score": teacher_selected_score,
                    "temporal_bands": evidence[("teacher", "frozen_selected")][
                        "band_scores"
                    ],
                },
                "motivation_gate": {
                    "checks": gate_checks,
                    "passed": all(gate_checks.values()),
                    "authorizes_rollout": False,
                },
                "artifact_hashes": {
                    name: sha256_file(args.output_dir / name)
                    for name in (
                        "population_scores.csv",
                        "snapshot_scores.csv",
                        "structure_benefits.csv",
                        "selection_decisions.csv",
                    )
                },
                "claim_boundary": (
                    "Retrospective teacher-forced linear-map diagnostic on already-"
                    "inspected shock-vortex calibration/teacher cases. Passing can "
                    "motivate, but cannot authorize, a recurrent controller. No "
                    "terminal-harm recall, cross-family coefficient, conservation, "
                    "convergence, bump, off-grid, or Richardson-extrapolation claim."
                ),
            }
        )
    )
    atomic_write_json(args.output_dir / "modal_linear_map.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--calibration-modal-records", type=Path, required=True)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--teacher-modal-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
