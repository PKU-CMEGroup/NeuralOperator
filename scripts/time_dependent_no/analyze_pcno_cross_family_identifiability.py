#!/usr/bin/env python3
"""Audit scale-free two-discrepancy identifiability across completed PDE families."""

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
    VIEW_NAMES,
    build_geometry_records,
    individual_alignment_from_statistics,
    snapshots_from_modal_rows,
    standardized_direction_relation,
    two_feature_geometry,
    two_feature_geometry_from_statistics,
)

WORKING_ID = "W26-L5-P6-RFB19-A34-CROSS-FAMILY-DISCREPANCY-IDENTIFIABILITY"
SCHEMA = "pcno_cross_family_discrepancy_identifiability_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_cross_family_identifiability.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_HASHES = {
    "a20_result": "439666d2b1016115f2e4750fcd876178153757ade94e31cd7543a5d84772cb29",
    "shock_vortex_calibration_modal_records": (
        "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb"
    ),
    "shock_vortex_teacher_modal_records": (
        "0ac7915de0c47d519a8e9e3d61d2b582a3266e63b191f6951cf9033f4198c4e5"
    ),
    "euler1d_calibration": (
        "9840df436915412a8d83aeae428e8ec01a7f4a3041f3ed55395932d90768f04c"
    ),
}
EXPECTED_PAYLOAD_HASHES = {
    "a20_result": "d4df3d9a1dc025c890b383a78acff9cd1f840995011eede713ae98c517105c42",
    "euler1d_calibration": (
        "4d3eadbec43568fbcbdb3886faa909dc321d5527d5c2df13c6118f9a04ebc74f"
    ),
}
EXPECTED_CALLS = tuple(range(30))
EXPECTED_CASE_COUNTS = {"calibration": 18, "teacher": 6}
EXPECTED_GROUPS = {
    "calibration": ("e01", "e02", "e03", "e04", "e05", "e07", "e08", "e09", "e10"),
    "teacher": ("e00", "e06", "e11"),
}
SCOPES = {
    "all": EXPECTED_CALLS,
    "calls_0_14": tuple(range(15)),
    "calls_15_29": tuple(range(15, 30)),
}
JOINT_FRACTION_FLOOR = 0.05
DIRECTION_COSINE_FLOOR = 0.8
CONDITION_NUMBER_CEILING = 100.0
STRICT_TOLERANCE = 1.0e-12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _verify_payload(
    payload: Mapping[str, Any], *, name: str, expected_payload_sha256: str
) -> None:
    verify_payload_sha256(payload)
    if payload.get("payload_sha256") != expected_payload_sha256:
        raise ValueError(f"{name} payload differs from the frozen artifact")


def _sign(value: float | None) -> int | None:
    if value is None or not math.isfinite(value) or value == 0.0:
        return None
    return 1 if value > 0.0 else -1


def _geometry_row(
    *, family: str, population: str, scope: str, view: str, geometry: Mapping[str, Any]
) -> dict[str, Any]:
    feature_c = geometry["feature_c"]
    feature_f = geometry["feature_f"]
    raw = geometry.get("raw_coefficients") or (None, None)
    standardized = geometry.get("standardized_coefficients") or (None, None)
    direction = geometry.get("standardized_direction") or (None, None)
    return {
        "family": family,
        "population": population,
        "scope": scope,
        "view": view,
        "status": geometry["status"],
        "case_count": geometry.get("case_count"),
        "snapshot_count": geometry.get("snapshot_count"),
        "discrepancy_correlation": geometry.get("discrepancy_correlation"),
        "condition_number": geometry.get("condition_number"),
        "coarse_alignment": feature_c["alignment"],
        "fine_alignment": feature_f["alignment"],
        "coarse_explained_fraction": feature_c["explained_fraction"],
        "fine_explained_fraction": feature_f["explained_fraction"],
        "joint_explained_fraction": geometry.get("joint_explained_fraction"),
        "incremental_f_after_c": geometry.get("incremental_f_after_c"),
        "incremental_c_after_f": geometry.get("incremental_c_after_f"),
        "raw_coarse_coefficient": raw[0],
        "raw_fine_coefficient": raw[1],
        "standardized_coarse_coefficient": standardized[0],
        "standardized_fine_coefficient": standardized[1],
        "direction_coarse": direction[0],
        "direction_fine": direction[1],
        "normal_equation_closure_max_abs": geometry.get(
            "normal_equation_closure_max_abs"
        ),
        "normalized_equation_closure_max_abs": geometry.get(
            "normalized_equation_closure_max_abs"
        ),
        "explained_fraction_closure_abs": geometry.get(
            "explained_fraction_closure_abs"
        ),
    }


def _shock_vortex_geometry(
    rows: Sequence[Mapping[str, str]],
    *,
    population: str,
    active_cells: Sequence[tuple[int, int]],
    expected_cells: Sequence[tuple[int, int]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    snapshots = snapshots_from_modal_rows(
        rows,
        expected_cells=expected_cells,
        expected_calls=EXPECTED_CALLS,
    )
    case_ids = {snapshot.case_id for snapshot in snapshots}
    groups = tuple(sorted({snapshot.group_id for snapshot in snapshots}))
    if len(case_ids) != EXPECTED_CASE_COUNTS[population]:
        raise ValueError(f"{population} case inventory differs")
    if groups != EXPECTED_GROUPS[population]:
        raise ValueError(f"{population} group inventory differs")
    records = build_geometry_records(snapshots, active_cells=active_cells)
    details: dict[str, dict[str, dict[str, Any]]] = {}
    output_rows = []
    for scope_name, calls in SCOPES.items():
        details[scope_name] = {}
        selected_calls = set(calls)
        for view in VIEW_NAMES:
            selected = [
                record
                for record in records
                if record.view == view and record.input_call in selected_calls
            ]
            geometry = two_feature_geometry(selected)
            expected_count = EXPECTED_CASE_COUNTS[population] * len(calls)
            if geometry["snapshot_count"] != expected_count:
                raise ValueError(f"{population}/{scope_name}/{view} inventory differs")
            details[scope_name][view] = geometry
            output_rows.append(
                _geometry_row(
                    family="shock_vortex",
                    population=population,
                    scope=scope_name,
                    view=view,
                    geometry=geometry,
                )
            )
    return details, output_rows


def _euler1d_geometry(
    payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    structure = payload.get("structure_diagnostics")
    if not isinstance(structure, Mapping):
        raise TypeError("Euler1D structure diagnostics are missing")
    stored = structure.get("two_feature_low_mode_fit")
    if not isinstance(stored, Mapping) or stored.get("status") != "ok":
        raise ValueError("Euler1D two-feature low-mode fit is unresolved")
    overall = two_feature_geometry_from_statistics(
        stored["gram"], stored["cross"], stored["target_square"]
    )
    if overall["status"] != "ok":
        raise ValueError("Euler1D two-feature geometry is unresolved")
    raw = overall["raw_coefficients"]
    if raw is None or max(
        abs(float(left) - float(right))
        for left, right in zip(raw, stored["coefficients_coarse_fine"], strict=True)
    ) > STRICT_TOLERANCE:
        raise ValueError("Euler1D two-feature coefficient replay differs")
    if abs(
        float(overall["joint_explained_fraction"])
        - float(stored["skill_vs_zero"])
    ) > STRICT_TOLERANCE:
        raise ValueError("Euler1D two-feature skill replay differs")

    temporal: dict[str, Any] = {}
    rows = [
        _geometry_row(
            family="euler1d",
            population="validation",
            scope="calls_0_99",
            view="low_k1_k7",
            geometry=overall,
        )
    ]
    relation_scopes = structure.get("relations")
    if not isinstance(relation_scopes, Mapping):
        raise TypeError("Euler1D time-scope relations are missing")
    for source_scope, output_scope in (
        ("calls_0_49", "calls_0_49"),
        ("calls_50_99", "calls_50_99"),
    ):
        relations = relation_scopes[source_scope]["correction_target_relations"]
        coarse = relations["coarse_low"]["fit"]
        fine = relations["fine_low"]["fit"]
        coarse_target = float(coarse["target_rms"]) ** 2
        fine_target = float(fine["target_rms"]) ** 2
        if abs(coarse_target - fine_target) > STRICT_TOLERANCE:
            raise ValueError("Euler1D half-horizon target energies differ")
        temporal[output_scope] = {
            "status": "individual_only_unrecorded_joint_cross_term",
            "coarse": individual_alignment_from_statistics(
                denominator=float(coarse["denominator"]),
                cross=float(coarse["cross"]),
                target_square=coarse_target,
            ),
            "fine": individual_alignment_from_statistics(
                denominator=float(fine["denominator"]),
                cross=float(fine["cross"]),
                target_square=fine_target,
            ),
            "joint_geometry_available": False,
            "joint_cross_term_imputed": False,
        }
        rows.append(
            {
                "family": "euler1d",
                "population": "validation",
                "scope": output_scope,
                "view": "low_k1_k7_individual_only",
                "status": temporal[output_scope]["status"],
                "case_count": 16,
                "snapshot_count": 800,
                "discrepancy_correlation": None,
                "condition_number": None,
                "coarse_alignment": temporal[output_scope]["coarse"]["alignment"],
                "fine_alignment": temporal[output_scope]["fine"]["alignment"],
                "coarse_explained_fraction": temporal[output_scope]["coarse"][
                    "explained_fraction"
                ],
                "fine_explained_fraction": temporal[output_scope]["fine"][
                    "explained_fraction"
                ],
                "joint_explained_fraction": None,
                "incremental_f_after_c": None,
                "incremental_c_after_f": None,
                "raw_coarse_coefficient": temporal[output_scope]["coarse"][
                    "coefficient"
                ],
                "raw_fine_coefficient": temporal[output_scope]["fine"][
                    "coefficient"
                ],
                "standardized_coarse_coefficient": None,
                "standardized_fine_coefficient": None,
                "direction_coarse": None,
                "direction_fine": None,
                "normal_equation_closure_max_abs": None,
                "normalized_equation_closure_max_abs": None,
                "explained_fraction_closure_abs": None,
            }
        )
    return overall, temporal, rows


def _comparison(
    *,
    name: str,
    first_label: str,
    first: Mapping[str, Any],
    second_label: str,
    second: Mapping[str, Any],
) -> dict[str, Any]:
    relation = standardized_direction_relation(
        first.get("standardized_direction"), second.get("standardized_direction")
    )
    return {
        "comparison": name,
        "first": first_label,
        "second": second_label,
        "cosine": relation["cosine"],
        "denominator": relation["denominator"],
        "status": relation["status"],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "a20_result": args.a20_result,
        "shock_vortex_calibration_modal_records": args.shock_vortex_calibration_modal_records,
        "shock_vortex_teacher_modal_records": args.shock_vortex_teacher_modal_records,
        "euler1d_calibration": args.euler1d_calibration,
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if input_hashes != EXPECTED_HASHES:
        raise ValueError("A34 input hashes differ from the frozen contract")
    a20 = _read_json(args.a20_result)
    euler1d = _read_json(args.euler1d_calibration)
    _verify_payload(
        a20,
        name="A20 result",
        expected_payload_sha256=EXPECTED_PAYLOAD_HASHES["a20_result"],
    )
    _verify_payload(
        euler1d,
        name="Euler1D calibration",
        expected_payload_sha256=EXPECTED_PAYLOAD_HASHES["euler1d_calibration"],
    )
    if a20.get("working_id") != "W26-L5-P6-RFB19-A20-RESIDUAL-GEOMETRY":
        raise ValueError("A20 working identity differs")
    if a20.get("input_hashes", {}).get("euler1d_calibration") != EXPECTED_HASHES[
        "euler1d_calibration"
    ]:
        raise ValueError("A20 does not own the frozen Euler1D parent")
    for name in (
        "shock_vortex_calibration_modal_records",
        "shock_vortex_teacher_modal_records",
    ):
        if a20.get("input_hashes", {}).get(name) != EXPECTED_HASHES[name]:
            raise ValueError(f"A20 does not own {name}")

    contract = a20.get("contract")
    if not isinstance(contract, Mapping):
        raise TypeError("A20 contract is missing")
    active_cells = tuple(tuple(int(value) for value in cell) for cell in contract["active_cells"])
    expected_cells = tuple(tuple(int(value) for value in cell) for cell in contract["all_cells"])
    if len(active_cells) != 19 or len(expected_cells) != 32:
        raise ValueError("A20 modal-cell contract differs")
    if tuple(contract.get("input_calls", ())) != EXPECTED_CALLS:
        raise ValueError("A20 input-call contract differs")

    shock_vortex: dict[str, Any] = {}
    geometry_rows = []
    for population, path in (
        ("calibration", args.shock_vortex_calibration_modal_records),
        ("teacher", args.shock_vortex_teacher_modal_records),
    ):
        details, rows = _shock_vortex_geometry(
            _read_csv(path),
            population=population,
            active_cells=active_cells,
            expected_cells=expected_cells,
        )
        shock_vortex[population] = details
        geometry_rows.extend(rows)
    euler_overall, euler_temporal, euler_rows = _euler1d_geometry(euler1d)
    geometry_rows.extend(euler_rows)

    primary = {
        "shock_vortex_calibration": shock_vortex["calibration"]["all"][
            "all_nonconstant"
        ],
        "shock_vortex_teacher": shock_vortex["teacher"]["all"][
            "all_nonconstant"
        ],
        "euler1d_validation": euler_overall,
    }
    comparisons = [
        _comparison(
            name="shock_vortex_calibration_vs_teacher",
            first_label="shock_vortex_calibration_all_nonconstant",
            first=primary["shock_vortex_calibration"],
            second_label="shock_vortex_teacher_all_nonconstant",
            second=primary["shock_vortex_teacher"],
        ),
        _comparison(
            name="shock_vortex_calibration_vs_euler1d",
            first_label="shock_vortex_calibration_all_nonconstant",
            first=primary["shock_vortex_calibration"],
            second_label="euler1d_validation_low_k1_k7",
            second=primary["euler1d_validation"],
        ),
        _comparison(
            name="shock_vortex_teacher_vs_euler1d",
            first_label="shock_vortex_teacher_all_nonconstant",
            first=primary["shock_vortex_teacher"],
            second_label="euler1d_validation_low_k1_k7",
            second=primary["euler1d_validation"],
        ),
    ]
    for population in ("calibration", "teacher"):
        comparisons.append(
            _comparison(
                name=f"shock_vortex_{population}_early_vs_late",
                first_label=f"shock_vortex_{population}_calls_0_14_all_nonconstant",
                first=shock_vortex[population]["calls_0_14"]["all_nonconstant"],
                second_label=f"shock_vortex_{population}_calls_15_29_all_nonconstant",
                second=shock_vortex[population]["calls_15_29"]["all_nonconstant"],
            )
        )
    for view in VIEW_NAMES:
        comparisons.append(
            _comparison(
                name=f"shock_vortex_calibration_vs_teacher_{view}",
                first_label=f"shock_vortex_calibration_{view}",
                first=shock_vortex["calibration"]["all"][view],
                second_label=f"shock_vortex_teacher_{view}",
                second=shock_vortex["teacher"]["all"][view],
            )
        )

    comparison_lookup = {row["comparison"]: row for row in comparisons}
    euler_signs = {
        feature: {
            scope: _sign(euler_temporal[scope][feature]["alignment"])
            for scope in ("calls_0_49", "calls_50_99")
        }
        for feature in ("coarse", "fine")
    }
    validity_checks = {
        "input_hashes_exact": input_hashes == EXPECTED_HASHES,
        "payload_hashes_exact": True,
        "a20_parent_ownership_exact": True,
        "shock_vortex_inventory_exact": True,
        "all_joint_geometries_resolved": all(
            row["status"] == "ok"
            for row in geometry_rows
            if row["view"] != "low_k1_k7_individual_only"
        ),
        "all_direction_comparisons_resolved": all(
            row["status"] == "ok" for row in comparisons
        ),
        "normal_equations_close": all(
            row["normal_equation_closure_max_abs"] is None
            or float(row["normal_equation_closure_max_abs"]) <= STRICT_TOLERANCE
            for row in geometry_rows
        ),
        "normalized_equations_close": all(
            row["normalized_equation_closure_max_abs"] is None
            or float(row["normalized_equation_closure_max_abs"]) <= STRICT_TOLERANCE
            for row in geometry_rows
        ),
        "explained_fraction_identity_closes": all(
            row["explained_fraction_closure_abs"] is None
            or float(row["explained_fraction_closure_abs"]) <= STRICT_TOLERANCE
            for row in geometry_rows
        ),
        "euler_half_joint_cross_term_not_imputed": all(
            not euler_temporal[scope]["joint_cross_term_imputed"]
            and not euler_temporal[scope]["joint_geometry_available"]
            for scope in ("calls_0_49", "calls_50_99")
        ),
        "no_model_built": True,
        "no_checkpoint_or_dataset_loaded": True,
        "no_state_or_reference_array_loaded": True,
        "no_prediction_or_recurrence": True,
        "families_not_pooled": True,
    }
    readiness_checks = {
        "joint_fraction_at_least_0p05_all_primary_cells": all(
            float(value["joint_explained_fraction"]) >= JOINT_FRACTION_FLOOR
            for value in primary.values()
        ),
        "primary_direction_cosines_at_least_0p8": all(
            float(comparison_lookup[name]["cosine"]) >= DIRECTION_COSINE_FLOOR
            for name in (
                "shock_vortex_calibration_vs_teacher",
                "shock_vortex_calibration_vs_euler1d",
                "shock_vortex_teacher_vs_euler1d",
            )
        ),
        "shock_vortex_time_direction_cosines_at_least_0p8": all(
            float(comparison_lookup[f"shock_vortex_{population}_early_vs_late"]["cosine"])
            >= DIRECTION_COSINE_FLOOR
            for population in ("calibration", "teacher")
        ),
        "euler1d_individual_alignment_signs_stable": all(
            values["calls_0_49"] is not None
            and values["calls_0_49"] == values["calls_50_99"]
            for values in euler_signs.values()
        ),
        "condition_numbers_at_most_100": all(
            row["condition_number"] is None
            or float(row["condition_number"]) <= CONDITION_NUMBER_CEILING
            for row in geometry_rows
        ),
    }
    if not all(validity_checks.values()):
        raise AssertionError(f"A34 validity checks failed: {validity_checks}")
    readiness_passed = all(readiness_checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "geometry.csv", geometry_rows)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    source_hashes = sha256_files(SOURCE_PATHS, root=ROOT)
    payload = with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": (
                "completed_readiness_passed"
                if readiness_passed
                else "completed_readiness_failed"
            ),
            "source_hashes": source_hashes,
            "source_status": _git_status_short(SOURCE_PATHS),
            "input_hashes": input_hashes,
            "parent_payload_hashes": EXPECTED_PAYLOAD_HASHES,
            "contract": {
                "feature_signs": {
                    "coarse": "native_increment_minus_mapped_coarse_increment",
                    "fine": "mapped_fine_increment_minus_native_increment",
                    "target": "reference_increment_minus_native_increment",
                },
                "weighting": "equal_calls_within_case_then_equal_cases",
                "primary_cell_comparison": (
                    "shock_vortex_all_nonconstant_vs_euler1d_low_k1_k7"
                ),
                "joint_fraction_floor": JOINT_FRACTION_FLOOR,
                "direction_cosine_floor": DIRECTION_COSINE_FLOOR,
                "condition_number_ceiling": CONDITION_NUMBER_CEILING,
                "euler_half_joint_cross_term_available": False,
                "euler_half_joint_cross_term_imputed": False,
            },
            "validity_checks": validity_checks,
            "readiness_checks": readiness_checks,
            "readiness_passed": readiness_passed,
            "shock_vortex": shock_vortex,
            "euler1d": {"overall": euler_overall, "temporal": euler_temporal},
            "primary_geometries": primary,
            "comparisons": comparisons,
            "euler1d_temporal_alignment_signs": euler_signs,
            "artifact_hashes": {
                "geometry.csv": sha256_file(args.output_dir / "geometry.csv"),
                "comparisons.csv": sha256_file(args.output_dir / "comparisons.csv"),
            },
            "execution": {
                "model_calls": 0,
                "checkpoint_or_dataset_loaded": False,
                "state_or_reference_array_loaded": False,
                "prediction_or_recurrence_executed": False,
                "sealed_population_opened": False,
            },
            "identifiability_limit": (
                "For fixed discrepancies c and f, changing the unavailable target "
                "y changes the optimal relation without changing prediction-derived "
                "features. A34 tests empirical second-order stability, not "
                "distribution-free identifiability."
            ),
            "claim_boundary": (
                "Retrospective zero-model cross-family geometry only. No universal "
                "coefficient, selector, rollout, bump, conservation, convergence, "
                "resolution-invariance, operator-learning, direct-off-grid, sealed, "
                "or Richardson-extrapolation claim is authorized."
            ),
        }
    )
    atomic_write_json(args.output_dir / "identifiability.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a20-result", type=Path, required=True)
    parser.add_argument(
        "--shock-vortex-calibration-modal-records", type=Path, required=True
    )
    parser.add_argument(
        "--shock-vortex-teacher-modal-records", type=Path, required=True
    )
    parser.add_argument("--euler1d-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
