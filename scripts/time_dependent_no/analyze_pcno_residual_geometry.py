#!/usr/bin/env python3
"""Replay the W26-L5 cross-family residual-geometry diagnostic offline."""

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
    FIXED_BETA,
    VIEW_NAMES,
    build_geometry_records,
    fit_case_first_coefficient,
    fit_persistence_threshold,
    grouped_coefficient_crossfit,
    grouped_persistence_crossfit,
    orthogonal_partition_closure,
    relation_rows,
    score_policy,
    snapshots_from_modal_rows,
    summarize_relations,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
)

WORKING_ID = "W26-L5-P6-RFB19-A20-RESIDUAL-GEOMETRY"
SCHEMA = "pcno_cross_family_residual_geometry_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_residual_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_residual_geometry.py",
    "tests/time_dependent_no/test_pcno_residual_geometry.py",
)
EXPECTED_INPUT_HASHES = {
    "shock_vortex_calibration_result": (
        "d9533285d89e70e61c16833ceeb27cd542cd65f44925a746c94786df26faf8cd"
    ),
    "shock_vortex_calibration_modal_records": (
        "c16df1dca5b77fc3b760a0e92a2e35dbd4adaa1910fc1d0998ded12b8d1103bb"
    ),
    "shock_vortex_teacher_result": (
        "bbbc3cdbd9afb64b8b8e63169b8acb9a9a4614957d6f4461412b4427405f63be"
    ),
    "shock_vortex_teacher_modal_records": (
        "0ac7915de0c47d519a8e9e3d61d2b582a3266e63b191f6951cf9033f4198c4e5"
    ),
    "euler1d_calibration": (
        "9840df436915412a8d83aeae428e8ec01a7f4a3041f3ed55395932d90768f04c"
    ),
}
EXPECTED_PAYLOAD_HASHES = {
    "shock_vortex_calibration_result": (
        "0f4351cf92ae16beec11cba57ac5aa4973df5579e73e11cc8d0e616a25e93c59"
    ),
    "shock_vortex_teacher_result": (
        "1ba8b8e30d0e6b5bd7a7a7a2d4925b23fbb56b558180502243f08e7c393c6046"
    ),
    "euler1d_calibration": (
        "4d3eadbec43568fbcbdb3886faa909dc321d5527d5c2df13c6118f9a04ebc74f"
    ),
}
EXPECTED_MODE_CELLS = tuple((mode, component) for mode in range(8) for component in range(4))
EXPECTED_CALLS = tuple(range(30))
EXPECTED_CASE_COUNTS = {"calibration": 18, "teacher": 6}
EXPECTED_GROUPS = {
    "calibration": ("e01", "e02", "e03", "e04", "e05", "e07", "e08", "e09", "e10"),
    "teacher": ("e00", "e06", "e11"),
}


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


def _verify_json_payload(
    payload: Mapping[str, Any], *, name: str, expected_payload_sha256: str
) -> None:
    verify_payload_sha256(payload)
    if payload.get("payload_sha256") != expected_payload_sha256:
        raise ValueError(f"{name} payload differs from the frozen artifact")


def _verify_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = {
        "shock_vortex_calibration_result": args.shock_vortex_calibration_result,
        "shock_vortex_calibration_modal_records": args.shock_vortex_calibration_modal_records,
        "shock_vortex_teacher_result": args.shock_vortex_teacher_result,
        "shock_vortex_teacher_modal_records": args.shock_vortex_teacher_modal_records,
        "euler1d_calibration": args.euler1d_calibration,
    }
    for name, path in paths.items():
        if sha256_file(path) != EXPECTED_INPUT_HASHES[name]:
            raise ValueError(f"{name} differs from the frozen artifact")
    calibration = _read_json(paths["shock_vortex_calibration_result"])
    teacher = _read_json(paths["shock_vortex_teacher_result"])
    euler1d = _read_json(paths["euler1d_calibration"])
    for name, payload in (
        ("shock_vortex_calibration_result", calibration),
        ("shock_vortex_teacher_result", teacher),
        ("euler1d_calibration", euler1d),
    ):
        _verify_json_payload(
            payload,
            name=name,
            expected_payload_sha256=EXPECTED_PAYLOAD_HASHES[name],
        )
    if calibration.get("modal_records_sha256") != EXPECTED_INPUT_HASHES[
        "shock_vortex_calibration_modal_records"
    ]:
        raise ValueError("calibration result does not own the frozen modal records")
    if teacher.get("modal_records_sha256") != EXPECTED_INPUT_HASHES[
        "shock_vortex_teacher_modal_records"
    ]:
        raise ValueError("teacher result does not own the frozen modal records")
    if teacher.get("calibration_sha256") != EXPECTED_INPUT_HASHES[
        "shock_vortex_calibration_result"
    ]:
        raise ValueError("teacher result is not bound to the calibration result")
    if teacher.get("calibration_payload_sha256") != calibration.get("payload_sha256"):
        raise ValueError("teacher/calibration payload binding differs")
    return {"calibration": calibration, "teacher": teacher, "euler1d": euler1d}


def _view_records(records: Sequence[Any], view: str) -> list[Any]:
    return [record for record in records if record.view == view]


def _score_population(records: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    details = {}
    for view in VIEW_NAMES:
        selected = _view_records(records, view)
        relations = summarize_relations(selected)
        fit = fit_case_first_coefficient(selected)
        if fit["status"] != "ok" or fit["coefficient"] is None:
            raise ValueError(f"coefficient unresolved for {view}")
        policies = {
            "zero": score_policy(selected, beta=0.0, policy="zero"),
            "fixed_beta": score_policy(
                selected, beta=FIXED_BETA, policy="fixed_beta"
            ),
            "previous_available": score_policy(
                selected,
                beta=FIXED_BETA,
                persistence_threshold=-1.0,
                policy="previous_available",
            ),
            "in_sample_fit_diagnostic": score_policy(
                selected,
                beta=float(fit["coefficient"]),
                policy="in_sample_fit_diagnostic",
            ),
        }
        details[view] = {
            "relations": relations,
            "fit": {**fit, "deployable": False},
            "policies": policies,
        }
        for policy, score in policies.items():
            summaries.append(
                {
                    "view": view,
                    "policy": policy,
                    "coarse_fine_median_cosine": relations[
                        "coarse_fine_median_cosine"
                    ],
                    "fine_target_median_cosine": relations[
                        "fine_target_median_cosine"
                    ],
                    "coarse_fine_to_fine_target_pearson": relations[
                        "coarse_fine_to_fine_target_pearson"
                    ],
                    "median_consecutive_fine_cosine": relations[
                        "median_consecutive_fine_cosine"
                    ],
                    "fitted_coefficient": fit["coefficient"],
                    "policy_beta": score["beta"],
                    "skill_vs_zero": score["skill_vs_zero"],
                    "rms_ratio_vs_zero": score["rms_ratio_vs_zero"],
                    "case_win_count": score["case_win_count"],
                    "case_count": score["case_count"],
                    "maximum_case_rms_ratio": score["maximum_case_rms_ratio"],
                    "applied_fraction": score["applied_fraction"],
                    "harmful_applied_fraction": score[
                        "harmful_applied_fraction"
                    ],
                    "status": score["status"],
                }
            )
    return summaries, details


def _euler1d_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    structure = payload["structure_diagnostics"]
    relations = structure["relations"]["all"]
    fine_native = relations["mapped_prediction_error_relations"][
        "fine_error_to_native_error"
    ]["unit_feature_score"]
    fine_low = relations["correction_target_relations"]["fine_low"]
    stability = payload["coefficient_stability"]
    vortex = payload["candidates"]["vortex_fixed"]
    family = payload["candidates"]["family_scalar"]
    return {
        "family": "historical_euler1d_d030",
        "coefficient_sign_convention": "target_minus_beta_times_fine_discrepancy",
        "mapped_fine_native_error_median_cosine": fine_native[
            "median_case_cosine"
        ],
        "fine_discrepancy_target_low_median_cosine": fine_low[
            "unit_feature_score"
        ]["median_case_cosine"],
        "fine_discrepancy_target_low_fit_coefficient": fine_low["fit"][
            "coefficient"
        ],
        "vortex_fixed_beta": vortex["coefficient"],
        "vortex_fixed_full_skill": vortex["view_scores"]["full"][
            "skill_vs_zero"
        ],
        "vortex_fixed_low_skill": vortex["view_scores"]["low_k1_k7"][
            "skill_vs_zero"
        ],
        "vortex_fixed_early_low_skill": vortex["early_low_score"][
            "skill_vs_zero"
        ],
        "vortex_fixed_late_low_skill": vortex["late_low_score"][
            "skill_vs_zero"
        ],
        "family_scalar_beta": family["coefficient"],
        "family_scalar_low_skill": family["view_scores"]["low_k1_k7"][
            "skill_vs_zero"
        ],
        "family_scalar_crossfit_low_skill": stability["crossfit"]["oof_score"][
            "skill_vs_zero"
        ],
        "early_coefficient": stability["early_fit"]["coefficient"],
        "late_coefficient": stability["late_fit"]["coefficient"],
        "fold_coefficients": stability["crossfit"]["fold_coefficients"],
        "fold_relative_iqr": stability["crossfit"]["relative_iqr"],
        "coefficient_stability_passed": stability["passed"],
        "fine_temporal": structure["temporal"]["fine_discrepancy"],
        "target_temporal": structure["temporal"]["target"],
        "mapped_error_triplet_rank1_energy": structure[
            "mapped_error_triplet_rank1_energy"
        ],
        "target_singular_energy": structure["target_singular_energy"],
        "pooled_with_shock_vortex": False,
        "coefficient_transfer_supported": False,
    }


def _mechanism_comparison(
    calibration_details: Mapping[str, Any],
    teacher_details: Mapping[str, Any],
    euler1d: Mapping[str, Any],
    threshold_fits: Mapping[str, Any],
    coefficient_crossfits: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_cal = {
        view: calibration_details[view]["policies"]["fixed_beta"]
        for view in VIEW_NAMES
    }
    fixed_teacher = {
        view: teacher_details[view]["policies"]["fixed_beta"]
        for view in VIEW_NAMES
    }
    sp19_concentrated = all(
        (
            fixed_cal["sp19"]["skill_vs_zero"] > 0.0,
            fixed_teacher["sp19"]["skill_vs_zero"] > 0.0,
            fixed_cal["complement"]["skill_vs_zero"] < 0.0,
            fixed_teacher["complement"]["skill_vs_zero"] < 0.0,
        )
    )
    calibration_all = calibration_details["all_nonconstant"]["relations"]
    teacher_all = teacher_details["all_nonconstant"]["relations"]
    strong_shared_grid_subspace = all(
        (
            calibration_all["coarse_fine_median_cosine"] > 0.8,
            teacher_all["coarse_fine_median_cosine"] > 0.8,
            euler1d["mapped_fine_native_error_median_cosine"] > 0.8,
        )
    )
    weak_truth_direction_proxy = all(
        (
            abs(calibration_all["coarse_fine_to_fine_target_pearson"]) < 0.2,
            abs(teacher_all["coarse_fine_to_fine_target_pearson"]) < 0.2,
            abs(euler1d["fine_discrepancy_target_low_median_cosine"]) < 0.2,
        )
    )
    calibration_sp19_energy_fraction = (
        calibration_details["sp19"]["relations"]["target_square_case_mean"]
        / calibration_all["target_square_case_mean"]
    )
    teacher_sp19_energy_fraction = (
        teacher_details["sp19"]["relations"]["target_square_case_mean"]
        / teacher_all["target_square_case_mean"]
    )
    persistence_incremental_skill = {}
    for view in VIEW_NAMES:
        fit_skill = threshold_fits[view]["calibration_fit"]["score"][
            "skill_vs_zero"
        ]
        calibration_baseline = calibration_details[view]["policies"][
            "previous_available"
        ]["skill_vs_zero"]
        teacher_skill = threshold_fits[view]["teacher_description"][
            "skill_vs_zero"
        ]
        teacher_baseline = teacher_details[view]["policies"][
            "previous_available"
        ]["skill_vs_zero"]
        persistence_incremental_skill[view] = {
            "calibration": fit_skill - calibration_baseline,
            "teacher": teacher_skill - teacher_baseline,
            "strictly_positive_on_both": (
                fit_skill > calibration_baseline + 1.0e-12
                and teacher_skill > teacher_baseline + 1.0e-12
            ),
        }
    shock_vortex_sign = math.copysign(
        1.0, calibration_details["all_nonconstant"]["fit"]["coefficient"]
    )
    euler1d_sign = math.copysign(1.0, euler1d["family_scalar_beta"])
    shared_scalar_supported = bool(
        shock_vortex_sign == euler1d_sign
        and euler1d["coefficient_stability_passed"]
        and coefficient_crossfits["all_nonconstant"]["same_nonzero_sign"]
    )
    return {
        "shock_vortex_fixed_beta_benefit_concentrated_in_sp19": sp19_concentrated,
        "shock_vortex_sp19_all_cases_improve": (
            fixed_cal["sp19"]["case_win_count"] == fixed_cal["sp19"]["case_count"]
            and fixed_teacher["sp19"]["case_win_count"]
            == fixed_teacher["sp19"]["case_count"]
        ),
        "shock_vortex_complement_all_cases_harmed": (
            fixed_cal["complement"]["case_win_count"] == 0
            and fixed_teacher["complement"]["case_win_count"] == 0
        ),
        "shock_vortex_target_energy_fraction_in_sp19": {
            "calibration": calibration_sp19_energy_fraction,
            "teacher": teacher_sp19_energy_fraction,
        },
        "strong_shared_grid_error_subspace": strong_shared_grid_subspace,
        "weak_observable_agreement_to_truth_direction_relation": (
            weak_truth_direction_proxy
        ),
        "observable_grid_agreement_as_truth_direction_proxy_supported": bool(
            strong_shared_grid_subspace and not weak_truth_direction_proxy
        ),
        "evidence_for_false_identification": {
            "shock_vortex_calibration_all": calibration_all,
            "shock_vortex_teacher_all": teacher_all,
            "euler1d_mapped_fine_native_error_median_cosine": euler1d[
                "mapped_fine_native_error_median_cosine"
            ],
            "euler1d_fine_discrepancy_target_low_median_cosine": euler1d[
                "fine_discrepancy_target_low_median_cosine"
            ],
        },
        "shared_scalar_coefficient_supported": shared_scalar_supported,
        "persistence_threshold_incremental_skill": persistence_incremental_skill,
        "persistence_threshold_adds_skill_in_all_or_sp19_on_both_populations": any(
            persistence_incremental_skill[view]["strictly_positive_on_both"]
            for view in ("all_nonconstant", "sp19")
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    parents = _verify_inputs(args)
    population_records = {}
    relation_output = []
    summary_output = []
    details = {}
    closures = {}
    for name, modal_path in (
        ("calibration", args.shock_vortex_calibration_modal_records),
        ("teacher", args.shock_vortex_teacher_modal_records),
    ):
        snapshots = snapshots_from_modal_rows(
            _read_csv(modal_path),
            expected_cells=EXPECTED_MODE_CELLS,
            expected_calls=EXPECTED_CALLS,
        )
        case_ids = {snapshot.case_id for snapshot in snapshots}
        groups = tuple(sorted({snapshot.group_id for snapshot in snapshots}))
        if len(case_ids) != EXPECTED_CASE_COUNTS[name] or groups != EXPECTED_GROUPS[name]:
            raise ValueError(f"{name} case/group inventory differs")
        records = build_geometry_records(
            snapshots, active_cells=FROZEN_ACTIVE_CELLS
        )
        population_records[name] = records
        rows = relation_rows(records)
        relation_output.extend({"population": name, **row} for row in rows)
        summaries, population_details = _score_population(records)
        summary_output.extend({"population": name, **row} for row in summaries)
        details[name] = population_details
        closures[name] = orthogonal_partition_closure(records)

    threshold_fits = {}
    crossfit_rows = []
    coefficient_crossfits = {}
    for view in VIEW_NAMES:
        calibration_view = _view_records(population_records["calibration"], view)
        teacher_view = _view_records(population_records["teacher"], view)
        fitted = fit_persistence_threshold(calibration_view)
        if fitted["status"] != "ok" or fitted["threshold"] is None:
            raise ValueError(f"persistence threshold unresolved for {view}")
        threshold = float(fitted["threshold"])
        teacher_score = score_policy(
            teacher_view,
            beta=FIXED_BETA,
            persistence_threshold=threshold,
            policy="frozen_calibration_persistence",
        )
        crossfit = grouped_persistence_crossfit(calibration_view)
        threshold_fits[view] = {
            "calibration_fit": fitted,
            "calibration_crossfit": crossfit,
            "teacher_description": teacher_score,
            "prospective": False,
        }
        coefficient_crossfits[view] = grouped_coefficient_crossfit(calibration_view)
        for fold in crossfit["folds"]:
            crossfit_rows.append(
                {
                    "model": "persistence_threshold",
                    "view": view,
                    "held_out_group": fold["held_out_group"],
                    "threshold": fold["threshold"],
                    "coefficient": FIXED_BETA,
                    "held_out_skill": fold["held_out_score"]["skill_vs_zero"],
                    "held_out_rms_ratio": fold["held_out_score"][
                        "rms_ratio_vs_zero"
                    ],
                }
            )
        for fold in coefficient_crossfits[view]["folds"]:
            crossfit_rows.append(
                {
                    "model": "scalar_coefficient",
                    "view": view,
                    "held_out_group": fold["held_out_group"],
                    "threshold": None,
                    "coefficient": fold["coefficient"],
                    "held_out_skill": fold["held_out_score"]["skill_vs_zero"],
                    "held_out_rms_ratio": fold["held_out_score"][
                        "rms_ratio_vs_zero"
                    ],
                }
            )

    euler1d = _euler1d_summary(parents["euler1d"])
    input_hashes = {
        "shock_vortex_calibration_result": sha256_file(
            args.shock_vortex_calibration_result
        ),
        "shock_vortex_calibration_modal_records": sha256_file(
            args.shock_vortex_calibration_modal_records
        ),
        "shock_vortex_teacher_result": sha256_file(args.shock_vortex_teacher_result),
        "shock_vortex_teacher_modal_records": sha256_file(
            args.shock_vortex_teacher_modal_records
        ),
        "euler1d_calibration": sha256_file(args.euler1d_calibration),
    }
    checks = {
        "input_hashes_exact": input_hashes == EXPECTED_INPUT_HASHES,
        "payload_hashes_exact": True,
        "calibration_inventory_exact": len(
            {record.case_id for record in population_records["calibration"]}
        )
        == EXPECTED_CASE_COUNTS["calibration"],
        "teacher_inventory_exact": len(
            {record.case_id for record in population_records["teacher"]}
        )
        == EXPECTED_CASE_COUNTS["teacher"],
        "orthogonal_partition_closure": all(
            closure["status"] == "ok" for closure in closures.values()
        ),
        "all_relation_denominators_resolved": all(
            details[population][view]["relations"][key] == 0
            for population in ("calibration", "teacher")
            for view in VIEW_NAMES
            for key in (
                "coarse_fine_unresolved_count",
                "fine_target_unresolved_count",
            )
        ),
        "call_zero_persistence_fails_closed": all(
            record.persistence_status == "no_previous_call"
            for records in population_records.values()
            for record in records
            if record.input_call == 0
        ),
        "euler1d_not_pooled": euler1d["pooled_with_shock_vortex"] is False,
        "no_model_built": True,
        "no_reference_array_loaded": True,
        "recurrence_not_executed": True,
        "gradient_path_unchanged": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"residual-geometry checks failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "view_summary.csv", summary_output)
    write_csv(args.output_dir / "snapshot_relations.csv", relation_output)
    write_csv(args.output_dir / "crossfit.csv", crossfit_rows)
    source_hashes = sha256_files(SOURCE_PATHS, root=ROOT)
    payload = with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": "completed_retrospective_diagnostic",
            "source_hashes": source_hashes,
            "source_status": _git_status_short(SOURCE_PATHS),
            "input_hashes": input_hashes,
            "parent_payload_hashes": {
                name: value["payload_sha256"] for name, value in parents.items()
            },
            "contract": {
                "feature_signs": {
                    "coarse": "native_increment_minus_mapped_coarse_increment",
                    "fine": "mapped_fine_increment_minus_native_increment",
                    "target": "reference_increment_minus_native_increment",
                    "score_error": "target_minus_beta_times_fine",
                },
                "views": list(VIEW_NAMES),
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "all_cells": [list(cell) for cell in EXPECTED_MODE_CELLS],
                "input_calls": list(EXPECTED_CALLS),
                "weighting": "equal_calls_within_case_then_equal_cases",
                "fixed_beta": FIXED_BETA,
                "denominator_floor": 1.0e-8,
                "persistence_fit": (
                    "maximize calibration case-first skill; exact ties choose "
                    "higher threshold and fewer corrections"
                ),
            },
            "checks": checks,
            "closures": closures,
            "shock_vortex": details,
            "persistence_threshold_diagnostics": threshold_fits,
            "coefficient_crossfits": coefficient_crossfits,
            "euler1d": euler1d,
            "mechanism_comparison": _mechanism_comparison(
                details["calibration"],
                details["teacher"],
                euler1d,
                threshold_fits,
                coefficient_crossfits,
            ),
            "artifact_hashes": {
                "view_summary.csv": sha256_file(args.output_dir / "view_summary.csv"),
                "snapshot_relations.csv": sha256_file(
                    args.output_dir / "snapshot_relations.csv"
                ),
                "crossfit.csv": sha256_file(args.output_dir / "crossfit.csv"),
            },
            "claim_boundary": (
                "Retrospective zero-model residual-geometry evidence only. The "
                "teacher split was already inspected; no selector, rollout, "
                "cross-family scalar, conservation, convergence, bump, direct "
                "off-grid, or Richardson-extrapolation claim is authorized."
            ),
        }
    )
    atomic_write_json(args.output_dir / "residual_geometry.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shock-vortex-calibration-result", type=Path, required=True)
    parser.add_argument(
        "--shock-vortex-calibration-modal-records", type=Path, required=True
    )
    parser.add_argument("--shock-vortex-teacher-result", type=Path, required=True)
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
