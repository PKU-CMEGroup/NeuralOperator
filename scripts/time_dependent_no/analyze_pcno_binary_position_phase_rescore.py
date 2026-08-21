#!/usr/bin/env python3
"""Run the zero-forward A30 binary position/phase rescore of A29 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

WORKING_ID = "W26-L5-P6-RFB19-A30-SP19-BINARY-POSITION-PHASE-RESCORE"
SCHEMA = "pcno_sp19_binary_position_phase_rescore_v1"
SOURCE_MANIFEST_SCHEMA = "pcno_sp19_binary_position_phase_source_manifest_v1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_pcno_binary_position_phase_rescore.py",
    "tests/time_dependent_no/test_pcno_sparse_modal_correction.py",
)
A29_RESULT_SHA256 = "72674d726e7223b25a7402f7556b5c989de827dbb41d9ec6e68a20baa5a85360"
A29_PAYLOAD_SHA256 = "5f80ad90cfd7cc884465825574c51a7f3b7b1ed889b56c46b2f290aeda2ad6b1"
POSITION_SHA256 = {
    "e12": "81abcb85e0dd5c7e85f1fdcda6910179bc8e7d74a498dc9ee6a7c2d42b9b7965",
    "e14": "5f3fe47098579f5fac1044ec46bc5cc4cd9bd12610e66edb3bf43e26ac81a4b4",
}
A29_ARTIFACTS = {
    "control_rows.csv": "77dd9c81081d4b3bad26db9eff139d4f90caf9485886046297618cdb63797620",
    "correction_audits.csv": "61e3d83fc6460e73bcab9630df221db55b7ecb54fdd427ff7a25c1e17388e080",
    "front_controls.csv": "763a7233f5895265923efd0bdc1613f23fe5524c0b63afa1a9bdfb480e40b172",
    "integral_controls.csv": "6ec336887cebc036895afef0c6a534c987798ed1dda1fefbe847d24d70635245",
    "nesting_floors.csv": "717c8c22b569aff80cee593162a27d6556fbce2667f7e106c08f622b9180aa49",
    "proposal_rows.csv": "0c34eada794501528a80b15de256c98c6854bf4d2393daad901a378ff8c0e06a",
    "reference_checks.csv": "8fc26a6be7ba39866257c71c4930a9068cfb5a1438f3a634a90026c75060be73",
    "score_rows.csv": "9ebf12d1724c46651b2b36c2f8239b059e025824611fbd54251562fed4aea107",
    "source_manifest.json": "8d46a9300a24b78bef036a69dca4bb762b9e29b8350aee4e574b32689a31419e",
}

GROUP_CASES = {
    "e12": tuple(f"sv_e12_y{index:02d}" for index in range(1, 8)),
    "e14": tuple(f"sv_e14_y{index:02d}" for index in range(1, 8)),
}
CASE_IDS = (*GROUP_CASES["e12"], *GROUP_CASES["e14"])
CASE_TO_GROUP = {
    case_id: group for group, cases in GROUP_CASES.items() for case_id in cases
}
BANDS = {
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
}
ACTIVE_BANDS = {"band_0_7", "band_22_29"}
POSITION_BUFFER = 0.8125
INPUT_CALLS = tuple(range(30))
VIEWS = (
    "full",
    "band_large",
    "band_transition",
    "band_local",
    "region_boundary",
    "region_shock",
    "region_vortex",
    "region_smooth",
    "smooth_local",
    "component_density",
    "component_x_momentum",
    "component_y_momentum",
    "component_energy",
    "rank8_parallel",
    "rank8_orthogonal",
    "rank8_excluded",
)
FIELD_CONTROL_VIEWS = VIEWS[:13]
FRONT_KEYS = ("front_position", "front_strength_log_ratio", "front_thickness_log_ratio")
INTEGRAL_COMPONENTS = ("density", "x_momentum", "y_momentum", "energy")
DENOMINATOR_FLOOR = 1.0e-8
CONTROL_LIMIT = 1.05
EXACT_TOLERANCE = 1.0e-14


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selector_active(normalized_wall_distance: float, input_call: int) -> bool:
    """Return the frozen target-free A30 binary selector decision."""

    if isinstance(input_call, bool) or not isinstance(input_call, (int, np.integer)):
        raise TypeError("input_call must be an integer")
    if int(input_call) not in INPUT_CALLS:
        raise ValueError("input_call must be in 0..29")
    distance = float(normalized_wall_distance)
    if not np.isfinite(distance):
        raise ValueError("normalized wall distance must be finite")
    return distance <= POSITION_BUFFER and (
        int(input_call) <= 7 or int(input_call) >= 22
    )


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _serialized_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"invalid serialized Boolean {value!r}")


def _case_row(
    payload: Mapping[str, Any], view: str, case_id: str
) -> Mapping[str, Any]:
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("scope") == "case"
        and row.get("view") == view
        and row.get("case_id") == case_id
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {view!r} row for {case_id!r}")
    return rows[0]


def sufficient_statistics(row: Mapping[str, Any]) -> dict[str, float]:
    """Recover exact uncentered target/correction statistics from an A29 row."""

    target_square = _finite_float(row["zero_sse"], name="zero_sse")
    corrected_square = _finite_float(row["corrected_sse"], name="corrected_sse")
    correction_rms = _finite_float(row["correction_rms"], name="correction_rms")
    correction_square = correction_rms * correction_rms
    cross = 0.5 * (target_square + correction_square - corrected_square)
    if min(target_square, corrected_square, correction_square) < -EXACT_TOLERANCE:
        raise ValueError("negative sufficient statistic")
    target_square = max(0.0, target_square)
    corrected_square = max(0.0, corrected_square)
    correction_square = max(0.0, correction_square)
    if row.get("cosine_status") == "ok":
        denominator = math.sqrt(target_square * correction_square)
        if denominator <= DENOMINATOR_FLOOR or row.get("cosine") is None:
            raise ValueError("resolved cosine lacks a valid denominator")
        if abs(cross / denominator - float(row["cosine"])) > 5.0e-13:
            raise ValueError("A29 cosine does not close with saved SSE statistics")
    return {
        "target_square": target_square,
        "correction_square": correction_square,
        "cross": cross,
        "corrected_square": corrected_square,
    }


def aggregate_statistics(
    statistics: Sequence[Mapping[str, float]], weights: Sequence[float]
) -> dict[str, float]:
    if not statistics or len(statistics) != len(weights):
        raise ValueError("statistics and weights must have the same nonzero length")
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.isclose(weight_array.sum(), 1.0, atol=1.0e-15, rtol=0.0):
        raise ValueError("weights must sum to one")
    keys = ("target_square", "correction_square", "cross")
    return {
        key: float(
            sum(
                float(weight) * float(row[key])
                for row, weight in zip(statistics, weights, strict=True)
            )
        )
        for key in keys
    }


def score_statistics(
    statistics: Mapping[str, float],
    *,
    correlation: float | None = None,
    correlation_status: str = "unavailable_missing_centered_sufficient_statistics",
) -> dict[str, Any]:
    target_square = _finite_float(statistics["target_square"], name="target_square")
    correction_square = _finite_float(
        statistics["correction_square"], name="correction_square"
    )
    cross = _finite_float(statistics["cross"], name="cross")
    corrected_square = target_square - 2.0 * cross + correction_square
    if min(target_square, correction_square, corrected_square) < -EXACT_TOLERANCE:
        raise ValueError("aggregated statistic is negative")
    target_square = max(0.0, target_square)
    correction_square = max(0.0, correction_square)
    corrected_square = max(0.0, corrected_square)
    target_rms = math.sqrt(target_square)
    correction_rms = math.sqrt(correction_square)
    corrected_rms = math.sqrt(corrected_square)
    if target_rms <= DENOMINATOR_FLOOR:
        if corrected_rms <= DENOMINATOR_FLOOR:
            skill = 0.0
            ratio = 1.0
            skill_status = "exact_zero_no_change"
        else:
            skill = None
            ratio = None
            skill_status = "small_denominator_harm"
    else:
        skill = 1.0 - corrected_square / target_square
        ratio = corrected_rms / target_rms
        skill_status = "ok"
    cosine_denominator = target_rms * correction_rms
    if cosine_denominator > DENOMINATOR_FLOOR:
        cosine = cross / cosine_denominator
        cosine_status = "ok"
    else:
        cosine = None
        cosine_status = (
            "zero_correction" if correction_rms == 0.0 else "small_denominator"
        )
    return {
        "zero_sse": target_square,
        "corrected_sse": corrected_square,
        "target_rms": target_rms,
        "correction_rms": correction_rms,
        "skill_vs_zero": skill,
        "rms_ratio_vs_zero": ratio,
        "skill_status": skill_status,
        "cosine": cosine,
        "cosine_denominator": cosine_denominator,
        "cosine_status": cosine_status,
        "correlation": correlation,
        "correlation_status": correlation_status,
    }


def _selected_error(row: Mapping[str, Any], *, active: bool) -> dict[str, float]:
    zero = _finite_float(row["zero_error"], name="zero_error")
    corrected = _finite_float(row["corrected_error"], name="corrected_error")
    return {"zero_error": zero, "corrected_error": corrected if active else zero}


def rms_control(
    *, key: str, scope: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows:
        return {
            "key": key,
            "scope": scope,
            "zero_rms": 0.0,
            "corrected_rms": 0.0,
            "ratio": None,
            "status": "incomplete",
        }
    try:
        zero = np.asarray(
            [_finite_float(row["zero_error"], name="zero_error") for row in rows]
        )
        corrected = np.asarray(
            [
                _finite_float(row["corrected_error"], name="corrected_error")
                for row in rows
            ]
        )
    except (KeyError, TypeError, ValueError):
        return {
            "key": key,
            "scope": scope,
            "zero_rms": 0.0,
            "corrected_rms": 0.0,
            "ratio": None,
            "status": "incomplete",
        }
    zero_rms = float(np.sqrt(np.mean(zero * zero)))
    corrected_rms = float(np.sqrt(np.mean(corrected * corrected)))
    if zero_rms <= DENOMINATOR_FLOOR:
        if corrected_rms <= DENOMINATOR_FLOOR:
            ratio = 1.0
            status = "exact_zero_no_change"
        else:
            ratio = None
            status = "small_denominator_harm"
    else:
        ratio = corrected_rms / zero_rms
        status = "ok" if np.isfinite(ratio) else "nonfinite"
    return {
        "key": key,
        "scope": scope,
        "zero_rms": zero_rms,
        "corrected_rms": corrected_rms,
        "ratio": ratio,
        "status": status,
    }


def _control_passed(row: Mapping[str, Any]) -> bool:
    ratio = row.get("ratio")
    return bool(
        row.get("status") in {"ok", "exact_zero_no_change"}
        and ratio is not None
        and np.isfinite(float(ratio))
        and float(ratio) <= CONTROL_LIMIT
    )


def _load_positions(
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    paths = {
        "e12": args.e12_position_decisions,
        "e14": args.e14_position_decisions,
    }
    distances: dict[str, float] = {}
    output: list[dict[str, Any]] = []
    for group, path in paths.items():
        if sha256_file(path) != POSITION_SHA256[group]:
            raise ValueError(f"{group} position-decision SHA-256 mismatch")
        rows = _read_csv(path)
        expected = {f"sv_{group}_y{index:02d}" for index in range(9)}
        if {row.get("case_id") for row in rows} != expected or len(rows) != 9:
            raise ValueError(f"{group} position inventory is not exact")
        for row in rows:
            if (
                row.get("status") != "ok"
                or float(row["position_buffer_threshold"]) != POSITION_BUFFER
            ):
                raise ValueError(
                    f"invalid position contract for {row.get('case_id')!r}"
                )
            case_id = str(row["case_id"])
            if case_id not in CASE_IDS:
                continue
            distance = _finite_float(
                row["normalized_wall_distance"], name="normalized_wall_distance"
            )
            distances[case_id] = distance
            active_calls = [
                call for call in INPUT_CALLS if selector_active(distance, call)
            ]
            output.append(
                {
                    "case_id": case_id,
                    "group_id": group,
                    "normalized_wall_distance": distance,
                    "position_buffer": POSITION_BUFFER,
                    "position_selected": distance <= POSITION_BUFFER,
                    "active_input_calls": ",".join(
                        str(value) for value in active_calls
                    ),
                    "active_call_count": len(active_calls),
                }
            )
    if set(distances) != set(CASE_IDS):
        raise ValueError("interior position inventory is incomplete")
    return distances, sorted(output, key=lambda row: row["case_id"])


def _verify_a29(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    if sha256_file(args.a29_result) != A29_RESULT_SHA256:
        raise ValueError("A29 result file SHA-256 mismatch")
    result = _read_json(args.a29_result)
    verify_payload_sha256(result)
    if (
        result.get("payload_sha256") != A29_PAYLOAD_SHA256
        or result.get("schema") != "pcno_sp19_affine_transfer_evaluation_v1"
        or result.get("working_id")
        != "W26-L5-P6-RFB19-A29-SP19-AFFINE-TRANSFER-E12-E14"
        or result.get("status") != "stopped_teacher_forced_transfer"
    ):
        raise ValueError("unexpected A29 result identity")
    root = args.a29_result.parent
    saved = result.get("artifact_inventory_before_summary")
    if not isinstance(saved, dict) or set(saved) != set(A29_ARTIFACTS):
        raise ValueError("A29 artifact inventory is not exact")
    tables: dict[str, list[dict[str, str]]] = {}
    for name, expected_hash in A29_ARTIFACTS.items():
        path = root / name
        if (
            sha256_file(path) != expected_hash
            or saved[name].get("sha256") != expected_hash
        ):
            raise ValueError(f"A29 artifact mismatch: {name}")
        if name.endswith(".csv"):
            tables[name] = _read_csv(path)
    return result, tables


def _build_scores(
    result: Mapping[str, Any], distances: Mapping[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float, float]:
    payloads = result.get("score_payloads")
    if not isinstance(payloads, dict):
        raise TypeError("A29 score payloads are missing")
    expected_payloads = {"overall", *BANDS}
    for group in GROUP_CASES:
        expected_payloads.add(f"group_{group}")
        expected_payloads.update(f"group_{group}_{band}" for band in BANDS)
    if set(payloads) != expected_payloads:
        raise ValueError("A29 score-payload inventory is not exact")

    source: dict[
        tuple[str, str, str], tuple[dict[str, float], Mapping[str, Any]]
    ] = {}
    maximum_cosine_closure = 0.0
    for band in BANDS:
        payload = payloads[band]
        if (
            tuple(payload.get("case_ids", ())) != CASE_IDS
            or tuple(payload.get("views", ())) != VIEWS
        ):
            raise ValueError(f"unexpected A29 payload contract for {band}")
        for case_id in CASE_IDS:
            for view in VIEWS:
                row = _case_row(payload, view, case_id)
                statistics = sufficient_statistics(row)
                if row.get("cosine_status") == "ok":
                    denominator = math.sqrt(
                        statistics["target_square"]
                        * statistics["correction_square"]
                    )
                    maximum_cosine_closure = max(
                        maximum_cosine_closure,
                        abs(
                            statistics["cross"] / denominator
                            - float(row["cosine"])
                        ),
                    )
                source[(band, case_id, view)] = (statistics, row)

    case_statistics: dict[tuple[str, str, str], dict[str, float]] = {}
    score_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    maximum_zero_reconstruction = 0.0
    maximum_corrected_reconstruction = 0.0
    band_weights = [len(calls) / len(INPUT_CALLS) for calls in BANDS.values()]
    for case_id in CASE_IDS:
        position_selected = distances[case_id] <= POSITION_BUFFER
        for view in VIEWS:
            selected_bands: list[dict[str, float]] = []
            original_bands: list[dict[str, float]] = []
            for band, band_calls in BANDS.items():
                original, original_row = source[(band, case_id, view)]
                active = position_selected and band in ACTIVE_BANDS
                selected = {
                    "target_square": original["target_square"],
                    "correction_square": (
                        original["correction_square"] if active else 0.0
                    ),
                    "cross": original["cross"] if active else 0.0,
                }
                case_statistics[(band, case_id, view)] = selected
                selected_bands.append(selected)
                original_bands.append(original)
                correlation = (
                    original_row.get("correlation")
                    if active and original_row.get("correlation_status") == "ok"
                    else None
                )
                score_rows.append(
                    {
                        "cell": band,
                        "view": view,
                        "scope": "case",
                        "group_id": CASE_TO_GROUP[case_id],
                        "case_id": case_id,
                        "position_selected": position_selected,
                        "correction_active": active,
                        "active_call_count": len(band_calls) if active else 0,
                        **score_statistics(
                            selected,
                            correlation=(
                                None if correlation is None else float(correlation)
                            ),
                            correlation_status=(
                                "ok"
                                if correlation is not None
                                else (
                                    "zero_correction"
                                    if not active
                                    else str(original_row.get("correlation_status"))
                                )
                            ),
                        ),
                    }
                )
            overall = aggregate_statistics(selected_bands, band_weights)
            case_statistics[("overall", case_id, view)] = overall
            score_rows.append(
                {
                    "cell": "overall",
                    "view": view,
                    "scope": "case",
                    "group_id": CASE_TO_GROUP[case_id],
                    "case_id": case_id,
                    "position_selected": position_selected,
                    "correction_active": position_selected,
                    "active_call_count": 16 if position_selected else 0,
                    **score_statistics(overall),
                }
            )
            original_overall = aggregate_statistics(original_bands, band_weights)
            saved_overall = _case_row(payloads["overall"], view, case_id)
            zero_difference = abs(
                original_overall["target_square"]
                - float(saved_overall["zero_sse"])
            )
            corrected_difference = abs(
                (
                    original_overall["target_square"]
                    - 2.0 * original_overall["cross"]
                    + original_overall["correction_square"]
                )
                - float(saved_overall["corrected_sse"])
            )
            maximum_zero_reconstruction = max(
                maximum_zero_reconstruction, zero_difference
            )
            maximum_corrected_reconstruction = max(
                maximum_corrected_reconstruction, corrected_difference
            )
            reconstruction_rows.append(
                {
                    "case_id": case_id,
                    "view": view,
                    "zero_sse_reconstruction_abs": zero_difference,
                    "original_corrected_sse_reconstruction_abs": corrected_difference,
                }
            )

    for cell in ("overall", *BANDS):
        for view in VIEWS:
            for group, cases in GROUP_CASES.items():
                stats = aggregate_statistics(
                    [case_statistics[(cell, case_id, view)] for case_id in cases],
                    [1.0 / len(cases)] * len(cases),
                )
                active_per_case = (
                    16
                    if cell == "overall"
                    else (len(BANDS[cell]) if cell in ACTIVE_BANDS else 0)
                )
                score_rows.append(
                    {
                        "cell": cell,
                        "view": view,
                        "scope": "group",
                        "group_id": group,
                        "case_id": None,
                        "position_selected": None,
                        "correction_active": (
                            cell == "overall" or cell in ACTIVE_BANDS
                        ),
                        "active_call_count": sum(
                            active_per_case
                            for case_id in cases
                            if distances[case_id] <= POSITION_BUFFER
                        ),
                        **score_statistics(stats),
                    }
                )
            population = aggregate_statistics(
                [case_statistics[(cell, case_id, view)] for case_id in CASE_IDS],
                [1.0 / len(CASE_IDS)] * len(CASE_IDS),
            )
            active_per_case = (
                16
                if cell == "overall"
                else (len(BANDS[cell]) if cell in ACTIVE_BANDS else 0)
            )
            score_rows.append(
                {
                    "cell": cell,
                    "view": view,
                    "scope": "population",
                    "group_id": None,
                    "case_id": None,
                    "position_selected": None,
                    "correction_active": (
                        cell == "overall" or cell in ACTIVE_BANDS
                    ),
                    "active_call_count": sum(
                        active_per_case
                        for case_id in CASE_IDS
                        if distances[case_id] <= POSITION_BUFFER
                    ),
                    **score_statistics(population),
                }
            )
    return (
        score_rows,
        reconstruction_rows,
        maximum_zero_reconstruction,
        maximum_corrected_reconstruction,
        maximum_cosine_closure,
    )


def _score_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell: str,
    view: str,
    scope: str,
    identifier: str | None = None,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["cell"] == cell
        and row["view"] == view
        and row["scope"] == scope
        and (
            identifier is None
            or row.get("case_id") == identifier
            or row.get("group_id") == identifier
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"missing rescore row {(cell, view, scope, identifier)!r}")
    return matches[0]


def _build_controls(
    score_rows: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    distances: Mapping[str, float],
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for view in FIELD_CONTROL_VIEWS:
        for scope in ("population", *CASE_IDS):
            row = _score_row(
                score_rows,
                cell="overall",
                view=view,
                scope="population" if scope == "population" else "case",
                identifier=None if scope == "population" else scope,
            )
            controls.append(
                {
                    "key": f"field::{view}",
                    "scope": scope,
                    "zero_rms": row["target_rms"],
                    "corrected_rms": math.sqrt(float(row["corrected_sse"])),
                    "ratio": row["rms_ratio_vs_zero"],
                    "status": (
                        "ok"
                        if row["skill_status"] == "ok"
                        else row["skill_status"]
                    ),
                }
            )

    front = tables["front_controls.csv"]
    integral = tables["integral_controls.csv"]
    expected_front = len(CASE_IDS) * len(INPUT_CALLS) * len(FRONT_KEYS)
    expected_integral = len(CASE_IDS) * len(INPUT_CALLS) * len(INTEGRAL_COMPONENTS)
    if len(front) != expected_front or len(integral) != expected_integral:
        raise ValueError("A29 per-row control inventory is not rectangular")
    for rows, discriminator, values in (
        (front, "key", FRONT_KEYS),
        (integral, "component", INTEGRAL_COMPONENTS),
    ):
        keys = [
            (row["case_id"], int(row["input_call"]), row[discriminator])
            for row in rows
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate A29 control row")
        expected = {
            (case_id, call, value)
            for case_id in CASE_IDS
            for call in INPUT_CALLS
            for value in values
        }
        if set(keys) != expected:
            raise ValueError("A29 per-row control keys are not exact")

    def selected(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
        return [
            _selected_error(
                row,
                active=selector_active(
                    distances[str(row["case_id"])], int(row["input_call"])
                ),
            )
            for row in rows
        ]

    for key in FRONT_KEYS:
        for scope in ("population", *CASE_IDS):
            rows = [
                row
                for row in front
                if row["key"] == key
                and (scope == "population" or row["case_id"] == scope)
            ]
            controls.append(
                rms_control(key=key, scope=scope, rows=selected(rows))
            )
    for component in INTEGRAL_COMPONENTS:
        for scope in ("population", *CASE_IDS):
            rows = [
                row
                for row in integral
                if row["component"] == component
                and (scope == "population" or row["case_id"] == scope)
            ]
            controls.append(
                rms_control(
                    key=f"integral_rms::{component}",
                    scope=scope,
                    rows=selected(rows),
                )
            )
            controls.append(
                rms_control(
                    key=f"integral_endpoint::{component}",
                    scope=scope,
                    rows=selected(
                        [row for row in rows if int(row["input_call"]) == 29]
                    ),
                )
            )
    for row in controls:
        row["policy"] = "binary_position_phase_sp19_affine"
        row["cell"] = "overall"
    if len(controls) != 360:
        raise ValueError("maintained control inventory must contain 360 rows")
    return controls


def _inherited_checks(
    result: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    distances: Mapping[str, float],
) -> dict[str, bool]:
    active = {
        (case_id, call)
        for case_id in CASE_IDS
        for call in INPUT_CALLS
        if selector_active(distances[case_id], call)
    }
    audits = tables["correction_audits.csv"]
    nesting = tables["nesting_floors.csv"]
    proposals = tables["proposal_rows.csv"]
    reference = tables["reference_checks.csv"]
    expected = {
        (case_id, call) for case_id in CASE_IDS for call in INPUT_CALLS
    }
    for name, rows in (
        ("audit", audits),
        ("nesting", nesting),
        ("proposal", proposals),
    ):
        keys = {(row["case_id"], int(row["input_call"])) for row in rows}
        if len(rows) != len(expected) or keys != expected:
            raise ValueError(f"A29 {name} inventory is not exact")
    active_audits = [
        row
        for row in audits
        if (row["case_id"], int(row["input_call"])) in active
    ]
    active_nesting = [
        row
        for row in nesting
        if (row["case_id"], int(row["input_call"])) in active
    ]
    active_proposals = [
        row
        for row in proposals
        if (row["case_id"], int(row["input_call"])) in active
    ]
    execution = result["execution"]
    return {
        "active_inventory_exact": len(active) == 64,
        "active_correction_closure": len(active_audits) == 64
        and all(
            row["status"] == "ok"
            and float(row["correction_to_native_increment"]) <= 0.05 + 1.0e-12
            and float(row["maximum_scaled_component_mean_abs"]) <= 1.0e-12
            and float(row["maximum_excluded_abs"]) <= 1.0e-12
            and float(row["maximum_inactive_coordinate_abs"]) <= 1.0e-12
            and float(row["maximum_modal_reconstruction_abs"]) <= 1.0e-12
            for row in active_audits
        ),
        "active_common_source_nesting": len(active_nesting) == 64
        and all(
            float(row["pre_model_coarse_from_native_max_abs"]) <= 1.0e-12
            and float(row["pre_model_fine_to_native_max_abs"]) <= 1.0e-12
            for row in active_nesting
        ),
        "active_proposals_finite_admissible": len(active_proposals) == 64
        and all(
            _serialized_bool(row["finite"])
            and _serialized_bool(row["admissible"])
            for row in active_proposals
        ),
        "reference_inventory_exact": len(reference) == len(CASE_IDS)
        and {row.get("case_id") for row in reference} == set(CASE_IDS)
        and all(
            row.get("retained_resolution") == "250x100"
            and row.get("state_shape") == "[61,25000,4]"
            and float(row["restriction_crosscheck_max_abs"]) <= 1.0e-12
            and (
                (
                    row["case_id"].startswith("sv_e12_")
                    and row.get("state_dtype") == "float64"
                    and row.get("frozen_training_reference_sha256")
                    == row.get("active_reference_artifact_sha256")
                    and len(str(row.get("active_reference_artifact_sha256"))) == 64
                )
                or (
                    row["case_id"].startswith("sv_e14_")
                    and row.get("state_dtype") == "float32"
                    and row.get("reference_schema")
                    == "pcno_shard_native_reference_v1"
                    and row.get("serialization_floor")
                    == "exact_original_reference_after_float32_cast"
                    and len(str(row.get("shard_state_digest"))) == 64
                )
            )
            for row in reference
        ),
        "inactive_rows_exact_raw_by_construction": len(expected - active) == 356,
        "deterministic_repeat": float(
            execution["maximum_repeat_abs_difference"]
        )
        <= 1.0e-6,
        "native_then_fine_call_order": int(execution["call_order_violations"])
        == 0,
        "fine_truth_not_loaded": result.get("fine_reference_loaded") is False,
        "a29_recurrence_not_run": result.get("recurrence_executed") is False,
    }


def _strict_positive(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("skill_status") == "ok" and float(row["skill_vs_zero"]) > 0.0
    )


def _exact_zero(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("skill_status") in {"ok", "exact_zero_no_change"}
        and abs(float(row["skill_vs_zero"])) <= EXACT_TOLERANCE
        and abs(float(row["rms_ratio_vs_zero"]) - 1.0) <= EXACT_TOLERANCE
    )


def _gate(
    score_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    distances: Mapping[str, float],
    *,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    required_views = ("full", "rank8_parallel")
    for scope, identifiers in (
        ("population", (None,)),
        ("group", tuple(GROUP_CASES)),
    ):
        for identifier in identifiers:
            label = "population" if identifier is None else f"group_{identifier}"
            checks[f"{label}_overall_positive"] = all(
                _strict_positive(
                    _score_row(
                        score_rows,
                        cell="overall",
                        view=view,
                        scope=scope,
                        identifier=identifier,
                    )
                )
                for view in required_views
            )
    for case_id in CASE_IDS:
        selected = distances[case_id] <= POSITION_BUFFER
        checks[f"case_{case_id}"] = all(
            (_strict_positive(row) if selected else _exact_zero(row))
            for view in required_views
            for row in (
                _score_row(
                    score_rows,
                    cell="overall",
                    view=view,
                    scope="case",
                    identifier=case_id,
                ),
            )
        )
        for band in BANDS:
            active = selected and band in ACTIVE_BANDS
            checks[f"case_{case_id}_{band}"] = all(
                (_strict_positive(row) if active else _exact_zero(row))
                for view in required_views
                for row in (
                    _score_row(
                        score_rows,
                        cell=band,
                        view=view,
                        scope="case",
                        identifier=case_id,
                    ),
                )
            )
    for band in BANDS:
        active = band in ACTIVE_BANDS
        for scope, identifiers in (
            ("population", (None,)),
            ("group", tuple(GROUP_CASES)),
        ):
            for identifier in identifiers:
                label = (
                    "population" if identifier is None else f"group_{identifier}"
                )
                checks[f"{label}_{band}"] = all(
                    (_strict_positive(row) if active else _exact_zero(row))
                    for view in required_views
                    for row in (
                        _score_row(
                            score_rows,
                            cell=band,
                            view=view,
                            scope=scope,
                            identifier=identifier,
                        ),
                    )
                )
    checks["all_360_controls_no_harm"] = len(controls) == 360 and all(
        _control_passed(row) for row in controls
    )
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "strict_positive_skill_equality_fails": True,
        "control_ratio_limit": CONTROL_LIMIT,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    result, tables = _verify_a29(args)
    distances, selector_rows = _load_positions(args)
    (
        score_rows,
        reconstruction_rows,
        maximum_zero,
        maximum_original_corrected,
        maximum_cosine,
    ) = _build_scores(result, distances)
    controls = _build_controls(score_rows, tables, distances)
    source_hashes = sha256_files(SOURCE_PATHS, root=ROOT)
    structural_checks = {
        "source_inventory_exact": set(source_hashes) == set(SOURCE_PATHS),
        "a29_result_and_artifacts_exact": True,
        "position_artifacts_exact": True,
        "zero_sse_reconstruction": maximum_zero <= EXACT_TOLERANCE,
        "original_corrected_sse_reconstruction": (
            maximum_original_corrected <= EXACT_TOLERANCE
        ),
        "cosine_sufficient_statistics_close": maximum_cosine <= 5.0e-13,
        "selector_population_exact": sum(
            row["position_selected"] for row in selector_rows
        )
        == 4,
        "prospective_logical_inventory_exact": sum(
            row["active_call_count"] for row in selector_rows
        )
        == 64,
        "model_calls_zero": True,
        "predictions_loaded_zero": True,
        "state_arrays_loaded_zero": True,
        "truth_arrays_loaded_zero": True,
        "recurrence_not_run": True,
        **_inherited_checks(result, tables, distances),
    }
    gate = _gate(
        score_rows, controls, distances, structural_checks=structural_checks
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "binary_position_phase_scores.csv", score_rows)
    write_csv(output_dir / "binary_position_phase_controls.csv", controls)
    write_csv(output_dir / "selector_rows.csv", selector_rows)
    write_csv(output_dir / "reconstruction_checks.csv", reconstruction_rows)
    write_csv(
        output_dir / "gate_checks.csv",
        [
            {"check": key, "passed": value}
            for key, value in sorted(gate["checks"].items())
        ],
    )
    source_manifest = with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": source_hashes,
            "input_sha256": {
                "a29_result": sha256_file(args.a29_result),
                "e12_position_decisions": sha256_file(
                    args.e12_position_decisions
                ),
                "e14_position_decisions": sha256_file(
                    args.e14_position_decisions
                ),
                **{
                    f"a29_{name}": value
                    for name, value in A29_ARTIFACTS.items()
                },
            },
            "policy": {
                "normalized_wall_distance_at_most": POSITION_BUFFER,
                "active_input_calls": [*range(8), *range(22, 30)],
                "gain_when_active": 1.0,
                "gain_otherwise": 0.0,
                "descriptor_source": "call_zero_native_physical_state",
                "case_id_used_for_decision": False,
            },
            "data_access": {
                "checkpoint_loaded": False,
                "predictions_loaded": False,
                "states_loaded": False,
                "truth_arrays_loaded": False,
                "fine_truth_loaded": False,
            },
        }
    )
    atomic_write_json(output_dir / "source_manifest.json", source_manifest)
    artifacts = {
        path.name: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    native_seconds = float(result["execution"]["native_forward_seconds"])
    fine_seconds = float(result["execution"]["fine_forward_seconds"])
    projected_fine_seconds = fine_seconds * 64.0 / 420.0
    payload = with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": (
                "retrospective_protocol_ready"
                if gate["status"] == "passed"
                else "stopped"
            ),
            "evidence_status": "retrospective_open_population_rescore",
            "gate": gate,
            "population_scores": {
                view: dict(
                    _score_row(
                        score_rows,
                        cell="overall",
                        view=view,
                        scope="population",
                    )
                )
                for view in ("full", "rank8_parallel")
            },
            "group_scores": {
                group: {
                    view: dict(
                        _score_row(
                            score_rows,
                            cell="overall",
                            view=view,
                            scope="group",
                            identifier=group,
                        )
                    )
                    for view in ("full", "rank8_parallel")
                }
                for group in GROUP_CASES
            },
            "maximum_zero_sse_reconstruction_abs": maximum_zero,
            "maximum_original_corrected_sse_reconstruction_abs": (
                maximum_original_corrected
            ),
            "maximum_cosine_sufficient_statistic_abs": maximum_cosine,
            "maximum_control_ratio": max(
                float(row["ratio"])
                for row in controls
                if row["ratio"] is not None
            ),
            "failed_controls": [
                dict(row) for row in controls if not _control_passed(row)
            ],
            "prospective_cost": {
                "native_calls": 420,
                "fine_calls": 64,
                "total_calls": 484,
                "logical_ratio_vs_raw_native": 484.0 / 420.0,
                "projected_native_forward_seconds_from_a29": native_seconds,
                "projected_fine_forward_seconds_from_a29_per_call_mean": (
                    projected_fine_seconds
                ),
                "projected_forward_ratio_vs_raw_native": (
                    native_seconds + projected_fine_seconds
                )
                / native_seconds,
                "projection_is_not_a_measured_a30_rollout_cost": True,
            },
            "execution": {
                "model_calls": 0,
                "checkpoint_loaded": False,
                "predictions_loaded": False,
                "states_loaded": False,
                "truth_arrays_loaded": False,
                "recurrence_executed": False,
                "animations_generated": False,
            },
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "artifact_inventory_before_summary": artifacts,
            "claim_boundary": (
                "Retrospective zero-forward E12/E14 mechanism/readiness evidence "
                "only; no prospective rollout, cross-family transfer, convergence "
                "order, or Richardson-extrapolation claim."
            ),
        }
    )
    atomic_write_json(output_dir / "binary_position_phase_rescore.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a29-result", type=Path, required=True)
    parser.add_argument("--e12-position-decisions", type=Path, required=True)
    parser.add_argument("--e14-position-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
