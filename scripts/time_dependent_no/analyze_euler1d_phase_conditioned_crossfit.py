#!/usr/bin/env python3
"""Cross-fit a two-phase Euler1D fine-discrepancy schedule from stored statistics."""

from __future__ import annotations

import argparse
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

WORKING_ID = "W26-L5-P6-RFB19-A35-EULER1D-PHASE-CONDITIONED-CROSSFIT"
SCHEMA = "euler1d_phase_conditioned_crossfit_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_euler1d_phase_conditioned_crossfit.py",
    "tests/time_dependent_no/test_euler1d_cross_resolution_correction.py",
)
EXPECTED_HASHES = {
    "a34_result": "588c7bc5701347a1b887edd78060e9fafd5de0f599220899cd3032becf63a062",
    "euler1d_calibration": (
        "9840df436915412a8d83aeae428e8ec01a7f4a3041f3ed55395932d90768f04c"
    ),
}
EXPECTED_PAYLOAD_HASHES = {
    "a34_result": "c9e0c59b6abbeef42e6633c69eb65451e78e8203fa9dc7c30b013ede79e5a196",
    "euler1d_calibration": (
        "4d3eadbec43568fbcbdb3886faa909dc321d5527d5c2df13c6118f9a04ebc74f"
    ),
}
EXPECTED_CASE_IDS = tuple(
    f"case_{case_id:03d}"
    for case_id in (
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
)
SCOPE_CALLS = {
    "all": tuple(range(100)),
    "calls_0_49": tuple(range(50)),
    "calls_50_99": tuple(range(50, 100)),
}
PHASE_SCOPES = ("calls_0_49", "calls_50_99")
DENOMINATOR_FLOOR = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-18
MINIMUM_COMBINED_SKILL = 0.01
MINIMUM_SKILL_INCREMENT = 0.01
MINIMUM_CASE_WINS = 12
MAXIMUM_CASE_RMS_RATIO = 1.05
MAXIMUM_RELATIVE_IQR = 0.5


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


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
        raise ValueError(
            f"{name} does not reproduce the stored statistic: "
            f"actual={actual}, expected={expected}"
        )
    return difference


def _canonical_calls(
    value: Any, *, expected: Sequence[int], name: str
) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(call, (bool, np.bool_)) or not isinstance(call, (int, np.integer))
        for call in value
    ):
        raise TypeError(f"{name} must be an integer list")
    calls = tuple(int(call) for call in value)
    if calls != tuple(expected):
        raise ValueError(f"{name} differs from the registered call inventory")
    return calls


def reconstruct_scope_statistics(
    relation: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
    expected_calls: Sequence[int],
) -> dict[str, Any]:
    """Recover case-first scalar sufficient statistics from a unit-feature score."""

    fit = relation.get("fit")
    unit = relation.get("unit_feature_score")
    if not isinstance(fit, Mapping) or not isinstance(unit, Mapping):
        raise TypeError("relation lacks fit or unit-feature score")
    rows = unit.get("case_scores")
    if not isinstance(rows, list):
        raise TypeError("unit-feature case scores must be a list")
    expected_cases = tuple(expected_case_ids)
    if (
        not expected_cases
        or any(
            not isinstance(case_id, str) or not case_id for case_id in expected_cases
        )
        or len(set(expected_cases)) != len(expected_cases)
    ):
        raise ValueError("expected case IDs must be unique nonempty strings")

    statistics: list[dict[str, Any]] = []
    seen: set[str] = set()
    maximum_cauchy_excess = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("case score rows must be objects")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case score has an invalid case ID")
        if case_id in seen:
            raise ValueError(f"duplicate case score: {case_id}")
        seen.add(case_id)
        calls = _canonical_calls(
            row.get("input_calls"),
            expected=expected_calls,
            name=f"{case_id} input_calls",
        )
        target_square = _finite_float(row.get("zero_sse"), name="zero_sse")
        unit_corrected_square = _finite_float(
            row.get("corrected_sse"), name="corrected_sse"
        )
        feature_rms = _finite_float(row.get("correction_rms"), name="correction_rms")
        feature_square = feature_rms * feature_rms
        if target_square < 0.0 or unit_corrected_square < 0.0:
            raise ValueError("stored squared errors must be nonnegative")
        if feature_rms <= DENOMINATOR_FLOOR:
            raise ValueError("stored feature RMS is unresolved")
        cross = 0.5 * (target_square + feature_square - unit_corrected_square)
        cauchy_excess = cross * cross - target_square * feature_square
        cauchy_tolerance = max(
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE
            * max(abs(cross * cross), abs(target_square * feature_square)),
        )
        if cauchy_excess > cauchy_tolerance:
            raise ValueError("reconstructed cross term violates Cauchy-Schwarz")
        maximum_cauchy_excess = max(maximum_cauchy_excess, cauchy_excess)
        statistics.append(
            {
                "case_id": case_id,
                "input_calls": list(calls),
                "target_square": target_square,
                "feature_square": feature_square,
                "cross": cross,
                "unit_corrected_square": unit_corrected_square,
            }
        )

    actual_cases = tuple(sorted(seen))
    if actual_cases != tuple(sorted(expected_cases)):
        raise ValueError("case inventory differs from the registered population")
    statistics.sort(key=lambda row: row["case_id"])
    target_square = float(np.mean([row["target_square"] for row in statistics]))
    feature_square = float(np.mean([row["feature_square"] for row in statistics]))
    cross = float(np.mean([row["cross"] for row in statistics]))
    unit_corrected_square = float(
        np.mean([row["unit_corrected_square"] for row in statistics])
    )
    coefficient = cross / feature_square

    stored_cross = _finite_float(fit.get("cross"), name="fit cross")
    stored_denominator = _finite_float(fit.get("denominator"), name="fit denominator")
    stored_coefficient = _finite_float(fit.get("coefficient"), name="fit coefficient")
    stored_target_rms = _finite_float(fit.get("target_rms"), name="fit target RMS")
    stored_feature_rms = _finite_float(fit.get("feature_rms"), name="fit feature RMS")
    closure = {
        "cross_abs": _require_close(cross, stored_cross, name="fit cross"),
        "denominator_abs": _require_close(
            feature_square, stored_denominator, name="fit denominator"
        ),
        "coefficient_abs": _require_close(
            coefficient, stored_coefficient, name="fit coefficient"
        ),
        "target_square_abs": _require_close(
            target_square, stored_target_rms * stored_target_rms, name="target square"
        ),
        "feature_square_abs": _require_close(
            feature_square,
            stored_feature_rms * stored_feature_rms,
            name="feature square",
        ),
        "unit_zero_abs": _require_close(
            target_square,
            _finite_float(unit.get("zero_sse_case_mean"), name="unit zero SSE"),
            name="unit zero SSE",
        ),
        "unit_corrected_abs": _require_close(
            unit_corrected_square,
            _finite_float(
                unit.get("corrected_sse_case_mean"), name="unit corrected SSE"
            ),
            name="unit corrected SSE",
        ),
        "unit_feature_abs": _require_close(
            feature_square,
            _finite_float(unit.get("correction_rms"), name="unit correction RMS") ** 2,
            name="unit feature square",
        ),
    }
    return {
        "case_statistics": statistics,
        "aggregate": {
            "case_count": len(statistics),
            "input_calls": list(expected_calls),
            "target_square": target_square,
            "feature_square": feature_square,
            "cross": cross,
            "coefficient": coefficient,
            "unit_corrected_square": unit_corrected_square,
        },
        "maximum_reconstruction_abs": max(closure.values()),
        "maximum_cauchy_excess": maximum_cauchy_excess,
        "closure": closure,
    }


def verify_equal_phase_partition(
    reconstructed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify that the two stored 50-call phases average to the 100-call scope."""

    full_rows = {row["case_id"]: row for row in reconstructed["all"]["case_statistics"]}
    early_rows = {
        row["case_id"]: row for row in reconstructed["calls_0_49"]["case_statistics"]
    }
    late_rows = {
        row["case_id"]: row for row in reconstructed["calls_50_99"]["case_statistics"]
    }
    if (
        not full_rows
        or set(full_rows) != set(early_rows)
        or set(full_rows) != set(late_rows)
    ):
        raise ValueError("phase case inventories do not match")
    closure: dict[str, float] = {}
    for case_id in sorted(full_rows):
        for key in (
            "target_square",
            "feature_square",
            "cross",
            "unit_corrected_square",
        ):
            expected = 0.5 * (early_rows[case_id][key] + late_rows[case_id][key])
            closure[f"{case_id}:{key}"] = _require_close(
                full_rows[case_id][key],
                expected,
                name=f"{case_id} full/phase {key}",
            )
    return {
        "maximum_case_phase_closure_abs": max(closure.values()),
        "case_phase_closure": closure,
    }


def _safe_square_error(statistic: Mapping[str, Any], coefficient: float) -> float:
    value = (
        statistic["target_square"]
        - 2.0 * coefficient * statistic["cross"]
        + coefficient * coefficient * statistic["feature_square"]
    )
    tolerance = max(
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE * abs(float(statistic["target_square"])),
    )
    if value < -tolerance:
        raise ValueError("quadratic score produced a negative squared error")
    return max(float(value), 0.0)


def _fit_coefficient(rows: Sequence[Mapping[str, Any]]) -> float:
    denominator = float(sum(row["feature_square"] for row in rows))
    cross = float(sum(row["cross"] for row in rows))
    if math.sqrt(max(denominator / len(rows), 0.0)) <= DENOMINATOR_FLOOR:
        raise ValueError("leave-one-case-out feature denominator is unresolved")
    return cross / denominator


def _score(zero: Sequence[float], corrected: Sequence[float]) -> dict[str, Any]:
    zero_mean = float(np.mean(np.asarray(zero, dtype=np.float64)))
    corrected_mean = float(np.mean(np.asarray(corrected, dtype=np.float64)))
    if math.sqrt(max(zero_mean, 0.0)) <= DENOMINATOR_FLOOR:
        return {
            "zero_square_case_mean": zero_mean,
            "corrected_square_case_mean": corrected_mean,
            "skill_vs_zero": None,
            "rms_ratio_vs_zero": None,
            "status": "small_target",
        }
    return {
        "zero_square_case_mean": zero_mean,
        "corrected_square_case_mean": corrected_mean,
        "skill_vs_zero": 1.0 - corrected_mean / zero_mean,
        "rms_ratio_vs_zero": math.sqrt(corrected_mean / zero_mean),
        "status": "ok",
    }


def _coefficient_stability(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("coefficient inventory is unresolved")
    median = float(np.median(array))
    q25, q75 = np.percentile(array, (25.0, 75.0))
    iqr = float(q75 - q25)
    relative_iqr = iqr / abs(median) if abs(median) > DENOMINATOR_FLOOR else None
    return {
        "values": array.tolist(),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": median,
        "iqr": iqr,
        "iqr_over_abs_median": relative_iqr,
        "negative_count": int(np.count_nonzero(array < 0.0)),
        "positive_count": int(np.count_nonzero(array > 0.0)),
        "zero_count": int(np.count_nonzero(array == 0.0)),
    }


def crossfit_phase_schedule(
    reconstructed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one global scalar with an early/late schedule, held out by case."""

    by_scope = {
        scope: {row["case_id"]: row for row in reconstructed[scope]["case_statistics"]}
        for scope in ("all", *PHASE_SCOPES)
    }
    case_ids = tuple(sorted(by_scope["all"]))
    if len(case_ids) < 2 or any(
        set(rows) != set(case_ids) for rows in by_scope.values()
    ):
        raise ValueError("cross-fit phase inventories do not match")

    folds = []
    case_scores = []
    phase_values: dict[str, dict[str, list[float]]] = {
        scope: {"zero": [], "global": [], "two_phase": [], "oracle": []}
        for scope in PHASE_SCOPES
    }
    for held_out_case in case_ids:
        train_cases = [case_id for case_id in case_ids if case_id != held_out_case]
        global_coefficient = _fit_coefficient(
            [by_scope["all"][case_id] for case_id in train_cases]
        )
        phase_coefficients = {
            scope: _fit_coefficient(
                [by_scope[scope][case_id] for case_id in train_cases]
            )
            for scope in PHASE_SCOPES
        }
        fold = {
            "held_out_case_id": held_out_case,
            "fit_case_ids": train_cases,
            "global_coefficient": global_coefficient,
            "early_coefficient": phase_coefficients["calls_0_49"],
            "late_coefficient": phase_coefficients["calls_50_99"],
        }
        folds.append(fold)
        phase_rows = []
        for scope in PHASE_SCOPES:
            statistic = by_scope[scope][held_out_case]
            zero = float(statistic["target_square"])
            global_square = _safe_square_error(statistic, global_coefficient)
            phase_square = _safe_square_error(statistic, phase_coefficients[scope])
            oracle_coefficient = statistic["cross"] / statistic["feature_square"]
            oracle_square = _safe_square_error(statistic, oracle_coefficient)
            for model, value in (
                ("zero", zero),
                ("global", global_square),
                ("two_phase", phase_square),
                ("oracle", oracle_square),
            ):
                phase_values[scope][model].append(value)
            phase_rows.append(
                {
                    "scope": scope,
                    "zero_square": zero,
                    "global_square": global_square,
                    "two_phase_square": phase_square,
                    "oracle_square": oracle_square,
                    "oracle_coefficient": oracle_coefficient,
                }
            )
        zero_combined = float(np.mean([row["zero_square"] for row in phase_rows]))
        global_combined = float(np.mean([row["global_square"] for row in phase_rows]))
        phase_combined = float(np.mean([row["two_phase_square"] for row in phase_rows]))
        oracle_combined = float(np.mean([row["oracle_square"] for row in phase_rows]))
        case_scores.append(
            {
                "case_id": held_out_case,
                "zero_square": zero_combined,
                "global_square": global_combined,
                "two_phase_square": phase_combined,
                "oracle_square": oracle_combined,
                "global_rms_ratio": math.sqrt(global_combined / zero_combined),
                "two_phase_rms_ratio": math.sqrt(phase_combined / zero_combined),
                "oracle_rms_ratio": math.sqrt(oracle_combined / zero_combined),
                "global_skill": 1.0 - global_combined / zero_combined,
                "two_phase_skill": 1.0 - phase_combined / zero_combined,
                "oracle_skill": 1.0 - oracle_combined / zero_combined,
                "early_oracle_coefficient": phase_rows[0]["oracle_coefficient"],
                "late_oracle_coefficient": phase_rows[1]["oracle_coefficient"],
            }
        )

    model_scores: dict[str, Any] = {}
    for model in ("global", "two_phase", "oracle"):
        phase_scores = {
            scope: _score(phase_values[scope]["zero"], phase_values[scope][model])
            for scope in PHASE_SCOPES
        }
        zero_combined = [row["zero_square"] for row in case_scores]
        corrected_key = f"{model}_square"
        corrected_combined = [row[corrected_key] for row in case_scores]
        model_scores[model] = {
            "combined": _score(zero_combined, corrected_combined),
            **phase_scores,
        }

    stability = {
        "global": _coefficient_stability(
            [fold["global_coefficient"] for fold in folds]
        ),
        "calls_0_49": _coefficient_stability(
            [fold["early_coefficient"] for fold in folds]
        ),
        "calls_50_99": _coefficient_stability(
            [fold["late_coefficient"] for fold in folds]
        ),
    }
    full_fit = {
        scope: _fit_coefficient(list(by_scope[scope].values()))
        for scope in ("all", *PHASE_SCOPES)
    }
    return {
        "case_ids": list(case_ids),
        "full_fit_coefficients": full_fit,
        "folds": folds,
        "case_scores": case_scores,
        "scores": model_scores,
        "coefficient_stability": stability,
        "two_phase_case_win_count": sum(
            row["two_phase_rms_ratio"] < 1.0 for row in case_scores
        ),
        "maximum_two_phase_case_rms_ratio": max(
            row["two_phase_rms_ratio"] for row in case_scores
        ),
        "two_phase_skill_increment_over_global": (
            model_scores["two_phase"]["combined"]["skill_vs_zero"]
            - model_scores["global"]["combined"]["skill_vs_zero"]
        ),
    }


def readiness_checks(crossfit: Mapping[str, Any]) -> dict[str, bool]:
    early = crossfit["coefficient_stability"]["calls_0_49"]
    late = crossfit["coefficient_stability"]["calls_50_99"]
    two_phase = crossfit["scores"]["two_phase"]
    return {
        "every_early_fold_coefficient_negative": early["negative_count"]
        == len(early["values"]),
        "every_late_fold_coefficient_positive": late["positive_count"]
        == len(late["values"]),
        "phase_relative_iqr_at_most_0p5": all(
            value["iqr_over_abs_median"] is not None
            and value["iqr_over_abs_median"] <= MAXIMUM_RELATIVE_IQR
            for value in (early, late)
        ),
        "combined_two_phase_skill_at_least_0p01": two_phase["combined"]["status"]
        == "ok"
        and two_phase["combined"]["skill_vs_zero"] >= MINIMUM_COMBINED_SKILL,
        "each_phase_two_phase_skill_positive": all(
            two_phase[scope]["status"] == "ok"
            and two_phase[scope]["skill_vs_zero"] > 0.0
            for scope in PHASE_SCOPES
        ),
        "two_phase_skill_increment_over_global_at_least_0p01": crossfit[
            "two_phase_skill_increment_over_global"
        ]
        >= MINIMUM_SKILL_INCREMENT,
        "at_least_12_of_16_case_wins": crossfit["two_phase_case_win_count"]
        >= MINIMUM_CASE_WINS,
        "maximum_case_rms_ratio_at_most_1p05": crossfit[
            "maximum_two_phase_case_rms_ratio"
        ]
        <= MAXIMUM_CASE_RMS_RATIO,
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
    input_paths = {
        "a34_result": args.a34_result,
        "euler1d_calibration": args.euler1d_calibration,
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if input_hashes != EXPECTED_HASHES:
        raise ValueError(f"input hashes differ from preregistration: {input_hashes}")
    a34 = _read_json(args.a34_result)
    euler = _read_json(args.euler1d_calibration)
    verify_payload_sha256(a34)
    verify_payload_sha256(euler)
    parent_payload_hashes = {
        "a34_result": a34["payload_sha256"],
        "euler1d_calibration": euler["payload_sha256"],
    }
    if parent_payload_hashes != EXPECTED_PAYLOAD_HASHES:
        raise ValueError("parent payload hashes differ from preregistration")
    if (
        a34.get("input_hashes", {}).get("euler1d_calibration")
        != input_hashes["euler1d_calibration"]
    ):
        raise ValueError("A34 does not bind the registered Euler1D parent")
    if a34.get("parent_payload_hashes", {}).get("euler1d_calibration") != euler.get(
        "payload_sha256"
    ):
        raise ValueError("A34 Euler1D payload binding differs")

    relations = euler.get("structure_diagnostics", {}).get("relations")
    if not isinstance(relations, Mapping):
        raise TypeError("Euler1D relation diagnostics are missing")
    reconstructed = {}
    for scope, calls in SCOPE_CALLS.items():
        relation = (
            relations.get(scope, {})
            .get("correction_target_relations", {})
            .get("fine_low")
        )
        if not isinstance(relation, Mapping):
            raise TypeError(f"Euler1D fine-low relation is missing for {scope}")
        reconstructed[scope] = reconstruct_scope_statistics(
            relation,
            expected_case_ids=EXPECTED_CASE_IDS,
            expected_calls=calls,
        )
    phase_partition = verify_equal_phase_partition(reconstructed)
    crossfit = crossfit_phase_schedule(reconstructed)
    registered_checks = readiness_checks(crossfit)
    validity_checks = {
        "input_hashes_exact": input_hashes == EXPECTED_HASHES,
        "payload_hashes_exact": parent_payload_hashes == EXPECTED_PAYLOAD_HASHES,
        "a34_euler_parent_exact": True,
        "case_and_call_inventory_exact": True,
        "stored_statistics_reproduced": all(
            scope["maximum_reconstruction_abs"]
            <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
            for scope in reconstructed.values()
        ),
        "equal_phase_partition_closed": True,
        "all_denominators_resolved": True,
        "all_scores_finite": all(
            math.isfinite(float(row[key]))
            for row in crossfit["case_scores"]
            for key in (
                "zero_square",
                "global_square",
                "two_phase_square",
                "oracle_square",
            )
        ),
    }
    readiness_passed = all(validity_checks.values()) and all(registered_checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fold_rows = [
        {
            **fold,
            "fit_case_count": len(fold["fit_case_ids"]),
        }
        for fold in crossfit["folds"]
    ]
    write_csv(args.output_dir / "folds.csv", fold_rows)
    write_csv(args.output_dir / "case_scores.csv", crossfit["case_scores"])
    artifact_hashes = {
        "folds.csv": sha256_file(args.output_dir / "folds.csv"),
        "case_scores.csv": sha256_file(args.output_dir / "case_scores.csv"),
    }
    payload = with_payload_sha256(
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
                "feature": "uncapped projected fine-to-native discrepancy, k=1--7",
                "target": "signed native one-step correction target, k=1--7",
                "case_weighting": "equal case then equal call; phases weighted equally",
                "phase_schedule": {
                    "calls_0_49": "one fixed early coefficient",
                    "calls_50_99": "one fixed late coefficient",
                },
                "coefficient_sign": "correction = beta * fine discrepancy",
                "cap_included": False,
                "recurrence_included": False,
            },
            "input_hashes": input_hashes,
            "parent_payload_hashes": parent_payload_hashes,
            "source_hashes": sha256_files(SOURCE_PATHS),
            "source_status": _git_status_short(SOURCE_PATHS),
            "validity_checks": validity_checks,
            "phase_partition": phase_partition,
            "reconstructed_statistics": reconstructed,
            "crossfit": crossfit,
            "readiness_checks": registered_checks,
            "readiness_passed": readiness_passed,
            "artifact_hashes": artifact_hashes,
            "execution": {
                "model_calls": 0,
                "checkpoint_or_dataset_loaded": False,
                "state_or_reference_array_loaded": False,
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
    atomic_write_json(args.output_dir / "phase_crossfit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a34-result", type=Path, required=True)
    parser.add_argument("--euler1d-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
