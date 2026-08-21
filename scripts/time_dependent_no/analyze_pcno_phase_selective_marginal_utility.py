"""Cross-fit inference-available histories against frozen A43/A44 utility labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
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

WORKING_ID = "W26-L5-P6-RFB19-A45-MARGINAL-UTILITY-AUDIT"
RESULT_SCHEMA = "pcno_phase_selective_marginal_utility_v1"
SOURCE_SCHEMA = "pcno_phase_selective_marginal_utility_source_v1"
SYNTHETIC_SCHEMA = "pcno_phase_selective_marginal_utility_synthetic_v1"

ZERO_POLICY = "zero"
CANDIDATE_POLICY = "phase_selective_one_native_coast_shadow_tether"
INPUT_CALLS = tuple(range(30))
EXACT_CALLS = (*range(8), *range(22, 30))
COAST_CALLS = tuple(range(8, 22))
BANDS = {
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
}
GROUP_CASES = {
    "e00": ("sv_e00_y01", "sv_e00_y07"),
    "e11": ("sv_e11_y01", "sv_e11_y07"),
    "e12": ("sv_e12_y01", "sv_e12_y07"),
    "e14": ("sv_e14_y01", "sv_e14_y07"),
}
ACTIVE_CASES = tuple(case for cases in GROUP_CASES.values() for case in cases)
CASE_GROUP = {
    case_id: group_id for group_id, cases in GROUP_CASES.items() for case_id in cases
}
PACKET_CASES = {
    "a43": (*GROUP_CASES["e12"], *GROUP_CASES["e14"]),
    "a44_r1": (*GROUP_CASES["e00"], *GROUP_CASES["e11"]),
}

PACKET_SPECS = {
    "a43": {
        "result_sha256": (
            "69eeb489609637beadad49ba65706c6392d61b29cfd55153093243f9d1e4560b"
        ),
        "payload_sha256": (
            "ebfe93a7dad4cba3f45b1bcbf01d54f58de6b041b25a5283c4f59803253a9730"
        ),
        "schema": "pcno_phase_selective_coast_replay_v1",
        "working_id": "W26-L5-P6-RFB19-A43-PHASE-SELECTIVE-COAST-REPLAY",
        "status": "qualified_phase_selective_replay",
        "allowed_checkout_drift": {
            "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md"
        },
    },
    "a44_r1": {
        "result_sha256": (
            "4eededf5489872e8771d1492fda720955f6a8624bedd00262e91f2f9cffaefa2"
        ),
        "payload_sha256": (
            "12a646705954ccc63182b31769f58c483f0be716cd09515fa75e3ca69d709963"
        ),
        "schema": "pcno_phase_selective_coast_confirmation_v2",
        "working_id": "W26-L5-P6-RFB19-A44-R1-NATIVE-SHARD-CONFIRMATION",
        "status": "qualified_disjoint_case_confirmation",
        "allowed_checkout_drift": {
            "docs/time_dependent_no/W26_L5_A43_INDEPENDENT_CONFIRMATION_PREREGISTRATION.md",
            "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_confirmation.py",
            "tests/time_dependent_no/test_pcno_phase_selective_coast_confirmation.py",
        },
    },
}

REQUIRED_INPUT_FILES = (
    "rollout_call_metrics.csv",
    "route_audits.csv",
    "position_decisions.csv",
    "view_scores.csv",
    "source_manifest.json",
)
OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_MARGINAL_UTILITY_AUDIT_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_pcno_phase_selective_marginal_utility.py",
    "tests/time_dependent_no/test_pcno_phase_selective_marginal_utility.py",
)
DEPENDENCY_PATHS = (
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
)

FEATURES = (
    "lag_displacement",
    "lag_intervention",
    "lag_displacement_change",
)
MODELS = ("zero", "constant", "phase", "history", "phase_history")
MODEL_FEATURES = {
    "zero": (),
    "constant": (),
    "phase": ("exact_phase",),
    "history": FEATURES,
    "phase_history": ("exact_phase", *FEATURES),
}
CALL_OUTCOMES = {
    "full_state": "state_error",
    "rank8_state": "rank8_state_error",
    "increment_defect": "increment_defect",
    "cumulative_defect": "cumulative_defect",
}
PRIMARY_OUTCOMES = ("full_state", "rank8_state")
STRUCTURAL_VIEWS = (
    "full",
    "rank8_parallel",
    "band_large",
    "band_transition",
    "band_local",
    "region_boundary",
    "region_shock",
    "region_vortex",
    "region_smooth",
)
REQUIRED_STRUCTURAL_VIEWS = ("rank8_parallel", "band_large")

RIDGE_PENALTY = 1.0
FEATURE_SCALE_FLOOR = 1.0e-12
DENOMINATOR_FLOOR_SQUARED = 1.0e-24
COEFFICIENT_FLOOR = 1.0e-12
COEFFICIENT_SCALE_LIMIT = 4.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _as_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric field {key!r}")
    return value


def _as_int(row: Mapping[str, Any], key: str) -> int:
    try:
        raw = row[key]
        value = int(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid integer field {key!r}") from error
    if isinstance(raw, bool) or str(value) != str(raw):
        raise ValueError(f"noncanonical integer field {key!r}")
    return value


def _as_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if value in (True, "True"):
        return True
    if value in (False, "False"):
        return False
    raise ValueError(f"invalid boolean field {key!r}")


def _safe_inventory_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"artifact inventory path escapes its root: {relative}")
    return candidate


def _verify_inventory(
    root: Path, inventory: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not inventory:
        raise ValueError("artifact inventory is empty")
    verified: dict[str, dict[str, Any]] = {}
    for relative in sorted(inventory):
        expected = inventory[relative]
        if not isinstance(relative, str) or not isinstance(expected, Mapping):
            raise TypeError("artifact inventory row is malformed")
        path = _safe_inventory_path(root, relative)
        if not path.is_file():
            raise ValueError(f"artifact is missing: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(expected.get("bytes", -1)) or digest != expected.get("sha256"):
            raise ValueError(f"artifact inventory mismatch: {relative}")
        verified[relative] = {"bytes": size, "sha256": digest}
    if not set(REQUIRED_INPUT_FILES).issubset(verified):
        raise ValueError("required A45 input files are absent from the inventory")
    return verified


def _compare_source_checkout(
    manifest: Mapping[str, Any], allowed_drift: set[str]
) -> dict[str, Any]:
    source = manifest.get("source_sha256")
    if not isinstance(source, Mapping):
        raise TypeError("input source manifest lacks source_sha256")
    changed: list[dict[str, Any]] = []
    checked = 0
    for bucket in ("owned", "dependencies"):
        rows = source.get(bucket)
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError(f"input source manifest lacks {bucket} hashes")
        for relative in sorted(rows):
            expected = rows[relative]
            path = ROOT / str(relative)
            checked += 1
            if not path.is_file():
                changed.append(
                    {
                        "bucket": bucket,
                        "path": str(relative),
                        "status": "missing",
                        "expected_sha256": expected,
                        "actual_sha256": None,
                    }
                )
                continue
            actual = sha256_file(path)
            if actual != expected:
                changed.append(
                    {
                        "bucket": bucket,
                        "path": str(relative),
                        "status": "changed",
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                    }
                )
    unexpected = [row for row in changed if row["path"] not in allowed_drift]
    missing_allowed = sorted(allowed_drift - {row["path"] for row in changed})
    return {
        "checked_files": checked,
        "changed_files": changed,
        "unexpected_changes": unexpected,
        "allowed_drift_currently_matching": missing_allowed,
        "status": "passed" if not unexpected else "failed",
    }


def verify_packet(result_path: Path, packet_id: str) -> dict[str, Any]:
    spec = PACKET_SPECS[packet_id]
    if sha256_file(result_path) != spec["result_sha256"]:
        raise ValueError(f"{packet_id} result file SHA-256 mismatch")
    payload = _read_json(result_path)
    verify_payload_sha256(payload)
    gate = payload.get("gate")
    if (
        payload.get("schema") != spec["schema"]
        or payload.get("working_id") != spec["working_id"]
        or payload.get("status") != spec["status"]
        or payload.get("payload_sha256") != spec["payload_sha256"]
        or not isinstance(gate, Mapping)
        or gate.get("status") != "qualified"
        or gate.get("failed_checks") != []
        or not all(gate.get("checks", {}).values())
        or payload.get("failed_controls") != []
        or payload.get("fine_reference_loaded") is not False
    ):
        raise ValueError(f"{packet_id} is not the exact qualified input")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping):
        raise TypeError(f"{packet_id} lacks a bound artifact inventory")
    root = result_path.resolve().parent
    verified_inventory = _verify_inventory(root, inventory)
    source_path = root / "source_manifest.json"
    if sha256_file(source_path) != payload.get(
        "source_manifest_sha256"
    ) or verified_inventory["source_manifest.json"]["sha256"] != payload.get(
        "source_manifest_sha256"
    ):
        raise ValueError(f"{packet_id} source manifest file differs")
    source_manifest = _read_json(source_path)
    verify_payload_sha256(source_manifest)
    if source_manifest.get("payload_sha256") != payload.get(
        "source_manifest_payload_sha256"
    ):
        raise ValueError(f"{packet_id} source manifest payload differs")
    checkout = _compare_source_checkout(
        source_manifest, set(spec["allowed_checkout_drift"])
    )
    if checkout["status"] != "passed":
        raise ValueError(f"{packet_id} has unexpected current source drift")
    return {
        "packet_id": packet_id,
        "root": root,
        "payload": payload,
        "source_manifest": source_manifest,
        "verified_inventory": verified_inventory,
        "checkout_comparison": checkout,
    }


def _expected_route(input_call: int) -> str:
    if input_call in EXACT_CALLS:
        return "exact_a32_window"
    if input_call in COAST_CALLS:
        return "surrogate_coast"
    raise ValueError("input call is outside the registered horizon")


def _band_for_call(input_call: int) -> str:
    matches = [name for name, calls in BANDS.items() if input_call in calls]
    if len(matches) != 1:
        raise ValueError("input call does not belong to exactly one band")
    return matches[0]


def _load_packet_tables(packet: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(packet["root"])
    return {
        "metrics": _read_csv(root / "rollout_call_metrics.csv"),
        "routes": _read_csv(root / "route_audits.csv"),
        "positions": _read_csv(root / "position_decisions.csv"),
        "views": _read_csv(root / "view_scores.csv"),
    }


def build_call_rows(packets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    route_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    packet_by_case: dict[str, str] = {}
    seen_active: set[str] = set()
    for packet in packets:
        packet_id = str(packet["packet_id"])
        tables = _load_packet_tables(packet)
        expected_cases = set(PACKET_CASES[packet_id])
        positions = {row.get("case_id"): row for row in tables["positions"]}
        active = {
            str(case_id)
            for case_id, row in positions.items()
            if _as_bool(row, "position_selected")
        }
        if active != expected_cases:
            raise ValueError(f"{packet_id} active-case inventory differs")
        for case_id in active:
            row = positions[case_id]
            if not _as_bool(row, "expected_selected"):
                raise ValueError(f"{packet_id} selected case was not expected")
            packet_by_case[case_id] = packet_id
        seen_active.update(active)
        for row in tables["metrics"]:
            case_id = str(row.get("case_id"))
            if case_id not in active:
                continue
            input_call = _as_int(row, "input_call")
            policy = str(row.get("policy"))
            key = (case_id, input_call)
            if policy not in (ZERO_POLICY, CANDIDATE_POLICY) or policy in paired[key]:
                raise ValueError("active metric policy inventory differs")
            paired[key][policy] = row
        for row in tables["routes"]:
            case_id = str(row.get("case_id"))
            if case_id not in active:
                continue
            key = (case_id, _as_int(row, "input_call"))
            if key in route_rows:
                raise ValueError("duplicate active route row")
            route_rows[key] = row
    if seen_active != set(ACTIVE_CASES):
        raise ValueError("combined active-case inventory differs")
    expected_keys = {(case, call) for case in ACTIVE_CASES for call in INPUT_CALLS}
    if set(paired) != expected_keys or set(route_rows) != expected_keys:
        raise ValueError("active case/call rectangle differs")

    records: list[dict[str, Any]] = []
    candidate_history: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for case_id in ACTIVE_CASES:
        for input_call in INPUT_CALLS:
            key = (case_id, input_call)
            policies = paired[key]
            if set(policies) != {ZERO_POLICY, CANDIDATE_POLICY}:
                raise ValueError("active metric pairing differs")
            zero = policies[ZERO_POLICY]
            candidate = policies[CANDIDATE_POLICY]
            route_audit = route_rows[key]
            route = _expected_route(input_call)
            if (
                candidate.get("route") != route
                or route_audit.get("route") != route
                or not _as_bool(candidate, "position_selected")
                or not _as_bool(route_audit, "position_selected")
                or _as_bool(candidate, "correction_active")
                != (route == "exact_a32_window")
                or _as_bool(route_audit, "correction_active")
                != (route == "exact_a32_window")
                or not _as_bool(candidate, "finite")
                or not _as_bool(candidate, "admissible")
                or not _as_bool(zero, "finite")
                or not _as_bool(zero, "admissible")
            ):
                raise ValueError("active route or rollout status differs")
            retained = _as_float(candidate, "retained_displacement_rms")
            intervention = _as_float(candidate, "correction_rms")
            if retained < 0.0 or intervention < 0.0:
                raise ValueError("negative displacement/intervention norm")
            previous = candidate_history[case_id].get(input_call - 1)
            previous_previous = candidate_history[case_id].get(input_call - 2)
            lag_displacement = 0.0 if previous is None else previous["retained"]
            lag_intervention = 0.0 if previous is None else previous["intervention"]
            lag_displacement_change = lag_displacement - (
                0.0 if previous_previous is None else previous_previous["retained"]
            )
            candidate_history[case_id][input_call] = {
                "retained": retained,
                "intervention": intervention,
            }
            record: dict[str, Any] = {
                "row_id": f"call|{case_id}|{input_call:02d}",
                "packet_id": packet_by_case[case_id],
                "group_id": CASE_GROUP[case_id],
                "case_id": case_id,
                "input_call": input_call,
                "band": _band_for_call(input_call),
                "route": route,
                "exact_phase": float(route == "exact_a32_window"),
                "lag_displacement": lag_displacement,
                "lag_intervention": lag_intervention,
                "lag_displacement_change": lag_displacement_change,
                "current_retained_displacement_rms_audit_only": retained,
                "current_correction_rms_audit_only": intervention,
            }
            for outcome, column in CALL_OUTCOMES.items():
                zero_error = _as_float(zero, column)
                corrected_error = _as_float(candidate, column)
                if zero_error < 0.0 or corrected_error < 0.0:
                    raise ValueError("error norm is negative")
                record[outcome] = zero_error**2 - corrected_error**2
                record[f"zero_{outcome}_error"] = zero_error
                record[f"corrected_{outcome}_error"] = corrected_error
            records.append(record)
    records.sort(key=_row_sort_key)
    return records


def build_structural_rows(
    packets: Sequence[Mapping[str, Any]], call_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    features_by_case_band: dict[tuple[str, str], dict[str, float]] = {}
    for case_id in ACTIVE_CASES:
        for band, calls in BANDS.items():
            rows = [
                row
                for row in call_rows
                if row["case_id"] == case_id and row["input_call"] in calls
            ]
            if len(rows) != len(calls):
                raise ValueError("call rows do not close a structural band")
            features_by_case_band[(case_id, band)] = {
                feature: float(np.mean([float(row[feature]) for row in rows]))
                for feature in FEATURES
            }

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for packet in packets:
        tables = _load_packet_tables(packet)
        active = set(PACKET_CASES[str(packet["packet_id"])])
        for row in tables["views"]:
            case_id = str(row.get("case_id"))
            cell = str(row.get("cell"))
            view = str(row.get("view"))
            if (
                row.get("scope") != "case"
                or case_id not in active
                or cell not in BANDS
                or view not in STRUCTURAL_VIEWS
            ):
                continue
            key = (case_id, cell, view)
            if key in indexed:
                raise ValueError("duplicate structural case/band/view row")
            indexed[key] = row
    expected = {
        (case_id, band, view)
        for case_id in ACTIVE_CASES
        for band in BANDS
        for view in STRUCTURAL_VIEWS
    }
    if set(indexed) != expected:
        raise ValueError("structural case/band/view rectangle differs")

    records: list[dict[str, Any]] = []
    for case_id, band, view in sorted(expected):
        row = indexed[(case_id, band, view)]
        zero_sse = _as_float(row, "zero_sse")
        corrected_sse = _as_float(row, "corrected_sse")
        route = (
            "exact_a32_window"
            if band in ("band_0_7", "band_22_29")
            else "surrogate_coast"
        )
        record = {
            "row_id": f"struct|{case_id}|{band}|{view}",
            "group_id": CASE_GROUP[case_id],
            "case_id": case_id,
            "band": band,
            "route": route,
            "view": view,
            "exact_phase": float(route == "exact_a32_window"),
            "utility": zero_sse - corrected_sse,
            "zero_sse": zero_sse,
            "corrected_sse": corrected_sse,
            **features_by_case_band[(case_id, band)],
        }
        records.append(record)
    records.sort(key=_row_sort_key)
    return records


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("group_id", "")),
        str(row.get("case_id", "")),
        int(row.get("input_call", -1)),
        str(row.get("band", "")),
        str(row.get("view", "")),
        str(row.get("row_id", "")),
    )


def _fit_model(
    rows: Sequence[Mapping[str, Any]], target: str, model: str
) -> dict[str, Any]:
    ordered = sorted(rows, key=_row_sort_key)
    y = np.asarray([float(row[target]) for row in ordered], dtype=np.float64)
    if model == "zero":
        return {"model": model, "features": (), "prediction": 0.0}
    if model == "constant":
        return {
            "model": model,
            "features": (),
            "intercept_standardized": float(np.mean(y)),
            "intercept_original": float(np.mean(y)),
            "means": {},
            "scales": {},
            "coefficients_standardized": {},
            "coefficients_original": {},
        }
    features = MODEL_FEATURES[model]
    raw = np.asarray(
        [[float(row[feature]) for feature in features] for row in ordered],
        dtype=np.float64,
    )
    means = np.mean(raw, axis=0)
    scales = np.std(raw, axis=0, ddof=0)
    if np.any(scales <= FEATURE_SCALE_FLOOR) or not np.all(np.isfinite(scales)):
        raise ValueError(f"{model} has an unresolved training feature scale")
    standardized = (raw - means) / scales
    design = np.column_stack((np.ones(len(ordered), dtype=np.float64), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    original_slopes = coefficients[1:] / scales
    original_intercept = coefficients[0] - float(np.dot(original_slopes, means))
    return {
        "model": model,
        "features": features,
        "intercept_standardized": float(coefficients[0]),
        "intercept_original": float(original_intercept),
        "means": dict(zip(features, means.tolist(), strict=True)),
        "scales": dict(zip(features, scales.tolist(), strict=True)),
        "coefficients_standardized": dict(
            zip(features, coefficients[1:].tolist(), strict=True)
        ),
        "coefficients_original": dict(
            zip(features, original_slopes.tolist(), strict=True)
        ),
    }


def _predict_model(fit: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    model = str(fit["model"])
    if model == "zero":
        return 0.0
    value = float(fit["intercept_standardized"])
    for feature in fit["features"]:
        value += float(fit["coefficients_standardized"][feature]) * (
            (float(row[feature]) - float(fit["means"][feature]))
            / float(fit["scales"][feature])
        )
    return float(value)


def crossfit_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[str],
    split_key: str,
    crossfit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=_row_sort_key)
    folds = sorted({str(row[split_key]) for row in ordered})
    if len(folds) < 2:
        raise ValueError("cross-fit requires at least two groups")
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for holdout in folds:
        train = [row for row in ordered if str(row[split_key]) != holdout]
        test = [row for row in ordered if str(row[split_key]) == holdout]
        if not train or not test or {str(row[split_key]) for row in train} & {holdout}:
            raise ValueError("grouped cross-fit leakage or empty fold")
        train_ids = tuple(str(row["row_id"]) for row in train)
        test_ids = tuple(str(row["row_id"]) for row in test)
        if set(train_ids) & set(test_ids):
            raise ValueError("cross-fit row leakage")
        for target in targets:
            for model in MODELS:
                fit = _fit_model(train, target, model)
                if model == "zero":
                    coefficients.append(
                        {
                            "crossfit": crossfit,
                            "holdout": holdout,
                            "target": target,
                            "model": model,
                            "feature": "intercept",
                            "training_mean": 0.0,
                            "training_scale": 1.0,
                            "coefficient_standardized": 0.0,
                            "coefficient_original": 0.0,
                            "train_rows": len(train),
                            "test_rows": len(test),
                        }
                    )
                else:
                    coefficients.append(
                        {
                            "crossfit": crossfit,
                            "holdout": holdout,
                            "target": target,
                            "model": model,
                            "feature": "intercept",
                            "training_mean": 0.0,
                            "training_scale": 1.0,
                            "coefficient_standardized": fit["intercept_standardized"],
                            "coefficient_original": fit["intercept_original"],
                            "train_rows": len(train),
                            "test_rows": len(test),
                        }
                    )
                    for feature in fit["features"]:
                        coefficients.append(
                            {
                                "crossfit": crossfit,
                                "holdout": holdout,
                                "target": target,
                                "model": model,
                                "feature": feature,
                                "training_mean": fit["means"][feature],
                                "training_scale": fit["scales"][feature],
                                "coefficient_standardized": fit[
                                    "coefficients_standardized"
                                ][feature],
                                "coefficient_original": fit["coefficients_original"][
                                    feature
                                ],
                                "train_rows": len(train),
                                "test_rows": len(test),
                            }
                        )
                for row in test:
                    predictions.append(
                        {
                            "crossfit": crossfit,
                            "holdout": holdout,
                            "target": target,
                            "model": model,
                            "row_id": row["row_id"],
                            "group_id": row["group_id"],
                            "case_id": row["case_id"],
                            "input_call": row.get("input_call"),
                            "band": row.get("band"),
                            "route": row["route"],
                            "view": row.get("view"),
                            "label": float(row[target]),
                            "prediction": _predict_model(fit, row),
                        }
                    )
    expected = len(ordered) * len(targets) * len(MODELS)
    if len(predictions) != expected:
        raise ValueError("cross-fit prediction rectangle differs")
    predictions.sort(
        key=lambda row: (
            row["crossfit"],
            row["target"],
            row["model"],
            row["group_id"],
            row["case_id"],
            -1 if row["input_call"] is None else int(row["input_call"]),
            "" if row["band"] is None else row["band"],
        )
    )
    coefficients.sort(
        key=lambda row: (
            row["crossfit"],
            row["target"],
            row["model"],
            row["holdout"],
            row["feature"],
        )
    )
    return predictions, coefficients


def _metric_summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    residual = labels - predictions
    sse = float(np.dot(residual, residual))
    zero_sse = float(np.dot(labels, labels))
    prediction_sse = float(np.dot(predictions, predictions))
    cosine_denominator = math.sqrt(max(0.0, zero_sse * prediction_sse))
    centered_labels = labels - float(np.mean(labels))
    centered_predictions = predictions - float(np.mean(predictions))
    centered_label_sse = float(np.dot(centered_labels, centered_labels))
    centered_prediction_sse = float(np.dot(centered_predictions, centered_predictions))
    correlation_denominator = math.sqrt(
        max(0.0, centered_label_sse * centered_prediction_sse)
    )
    skill_status = "ok" if zero_sse > DENOMINATOR_FLOOR_SQUARED else "small_denominator"
    cosine_status = (
        "ok"
        if cosine_denominator**2 > DENOMINATOR_FLOOR_SQUARED
        else "small_denominator"
    )
    correlation_status = (
        "ok"
        if correlation_denominator**2 > DENOMINATOR_FLOOR_SQUARED
        else "small_denominator"
    )
    positive = labels > 0.0
    negative = labels < 0.0
    nonzero = positive | negative
    predicted_positive = predictions > 0.0
    predicted_negative = predictions < 0.0
    sign_status = "ok" if np.any(nonzero) else "no_nonzero_labels"
    sign_accuracy = (
        float(np.mean(np.sign(predictions[nonzero]) == np.sign(labels[nonzero])))
        if np.any(nonzero)
        else None
    )
    if np.any(positive) and np.any(negative):
        balanced_status = "ok"
        balanced_accuracy = 0.5 * (
            float(np.mean(predicted_positive[positive]))
            + float(np.mean(predicted_negative[negative]))
        )
    else:
        balanced_status = "single_sign"
        balanced_accuracy = None
    return {
        "rows": int(labels.size),
        "label_positive_count": int(np.count_nonzero(positive)),
        "label_negative_count": int(np.count_nonzero(negative)),
        "label_zero_count": int(labels.size - np.count_nonzero(nonzero)),
        "sse": sse,
        "zero_sse": zero_sse,
        "label_rms": float(math.sqrt(zero_sse / labels.size)),
        "prediction_rms": float(math.sqrt(prediction_sse / labels.size)),
        "skill_vs_zero": None if skill_status != "ok" else 1.0 - sse / zero_sse,
        "skill_vs_zero_denominator": zero_sse,
        "skill_vs_zero_status": skill_status,
        "cosine": (
            None
            if cosine_status != "ok"
            else float(np.dot(labels, predictions) / cosine_denominator)
        ),
        "cosine_denominator": cosine_denominator,
        "cosine_status": cosine_status,
        "pearson": (
            None
            if correlation_status != "ok"
            else float(
                np.dot(centered_labels, centered_predictions) / correlation_denominator
            )
        ),
        "pearson_denominator": correlation_denominator,
        "pearson_status": correlation_status,
        "sign_accuracy": sign_accuracy,
        "sign_accuracy_status": sign_status,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_status": balanced_status,
    }


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    crossfits = sorted({str(row["crossfit"]) for row in predictions})
    for crossfit in crossfits:
        crossfit_rows = [row for row in predictions if row["crossfit"] == crossfit]
        targets = sorted({str(row["target"]) for row in crossfit_rows})
        for target in targets:
            target_rows = [row for row in crossfit_rows if row["target"] == target]
            scopes: list[tuple[str, list[Mapping[str, Any]]]] = [
                ("overall", target_rows)
            ]
            for group_id in sorted({str(row["group_id"]) for row in target_rows}):
                scopes.append(
                    (
                        f"group:{group_id}",
                        [row for row in target_rows if row["group_id"] == group_id],
                    )
                )
            for route in ("exact_a32_window", "surrogate_coast"):
                scopes.append(
                    (
                        f"route:{route}",
                        [row for row in target_rows if row["route"] == route],
                    )
                )
            for scope, scoped_all_models in scopes:
                phase_rows = [
                    row for row in scoped_all_models if row["model"] == "phase"
                ]
                phase_sse = None
                if phase_rows:
                    phase_sse = float(
                        np.sum(
                            [
                                (float(row["label"]) - float(row["prediction"])) ** 2
                                for row in phase_rows
                            ],
                            dtype=np.float64,
                        )
                    )
                for model in MODELS:
                    model_rows = [
                        row for row in scoped_all_models if row["model"] == model
                    ]
                    if not model_rows:
                        raise ValueError("metric scope has an empty model inventory")
                    labels = np.asarray(
                        [float(row["label"]) for row in model_rows], dtype=np.float64
                    )
                    predicted = np.asarray(
                        [float(row["prediction"]) for row in model_rows],
                        dtype=np.float64,
                    )
                    summary = _metric_summary(labels, predicted)
                    phase_status = (
                        "ok"
                        if model == "phase_history"
                        and phase_sse is not None
                        and phase_sse > DENOMINATOR_FLOOR_SQUARED
                        else (
                            "not_applicable"
                            if model != "phase_history"
                            else "small_denominator"
                        )
                    )
                    summary.update(
                        {
                            "crossfit": crossfit,
                            "target": target,
                            "model": model,
                            "scope": scope,
                            "skill_vs_phase": (
                                None
                                if phase_status != "ok"
                                else 1.0 - float(summary["sse"]) / float(phase_sse)
                            ),
                            "skill_vs_phase_denominator": phase_sse,
                            "skill_vs_phase_status": phase_status,
                        }
                    )
                    rows.append(summary)
    rows.sort(
        key=lambda row: (row["crossfit"], row["target"], row["model"], row["scope"])
    )
    return rows


def coefficient_stability(
    coefficients: Sequence[Mapping[str, Any]], target: str, feature: str
) -> dict[str, Any]:
    rows = [
        row
        for row in coefficients
        if row["crossfit"] == "leave_strength_group_out"
        and row["target"] == target
        and row["model"] == "phase_history"
        and row["feature"] == feature
    ]
    if len(rows) != len(GROUP_CASES):
        raise ValueError("coefficient fold inventory differs")
    values = np.asarray(
        [float(row["coefficient_standardized"]) for row in rows], dtype=np.float64
    )
    resolved = bool(np.all(np.abs(values) > COEFFICIENT_FLOOR))
    common_sign = bool(resolved and np.all(np.sign(values) == np.sign(values[0])))
    scale_ratio = (
        float(np.max(np.abs(values)) / np.min(np.abs(values))) if resolved else None
    )
    stable = bool(
        resolved and common_sign and scale_ratio is not None and scale_ratio <= 4.0
    )
    return {
        "target": target,
        "feature": feature,
        "fold_values": {
            str(row["holdout"]): float(row["coefficient_standardized"]) for row in rows
        },
        "resolved": resolved,
        "common_sign": common_sign,
        "scale_ratio": scale_ratio,
        "scale_limit": COEFFICIENT_SCALE_LIMIT,
        "stable": stable,
    }


def _metric_lookup(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    scope: str,
    model: str = "phase_history",
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["crossfit"] == "leave_strength_group_out"
        and row["target"] == target
        and row["scope"] == scope
        and row["model"] == model
    ]
    if len(matches) != 1:
        raise ValueError("metric lookup is not unique")
    return matches[0]


def prospective_gate(
    call_metrics: Sequence[Mapping[str, Any]],
    structural_metrics: Sequence[Mapping[str, Any]],
    stabilities: Sequence[Mapping[str, Any]],
    *,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    primary_overall = []
    primary_groups = []
    primary_routes = []
    for target in PRIMARY_OUTCOMES:
        overall = _metric_lookup(call_metrics, target=target, scope="overall")
        primary_overall.append(
            overall["skill_vs_zero_status"] == "ok"
            and float(overall["skill_vs_zero"]) > 0.0
            and overall["skill_vs_phase_status"] == "ok"
            and float(overall["skill_vs_phase"]) > 0.0
            and overall["cosine_status"] == "ok"
            and float(overall["cosine"]) > 0.0
            and overall["pearson_status"] == "ok"
            and float(overall["pearson"]) > 0.0
        )
        for group_id in GROUP_CASES:
            row = _metric_lookup(call_metrics, target=target, scope=f"group:{group_id}")
            primary_groups.append(
                row["skill_vs_phase_status"] == "ok"
                and float(row["skill_vs_phase"]) > 0.0
            )
        for route in ("exact_a32_window", "surrogate_coast"):
            row = _metric_lookup(call_metrics, target=target, scope=f"route:{route}")
            primary_routes.append(
                row["skill_vs_phase_status"] == "ok"
                and float(row["skill_vs_phase"]) > 0.0
            )
    checks["primary_overall_predictability"] = all(primary_overall)
    checks["primary_every_strength_group_adds_to_phase"] = all(primary_groups)
    checks["primary_exact_and_coast_add_to_phase"] = all(primary_routes)
    checks["primary_coefficient_sign_scale_stable"] = all(
        bool(row["stable"]) for row in stabilities
    )
    structural_required = []
    for view in REQUIRED_STRUCTURAL_VIEWS:
        row = _metric_lookup(structural_metrics, target=view, scope="overall")
        structural_required.append(
            row["skill_vs_phase_status"] == "ok" and float(row["skill_vs_phase"]) > 0.0
        )
    checks["rank8_and_large_structural_predictability"] = all(structural_required)
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "equality_fails": True,
    }


def _source_manifest(
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": {
                "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
                "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
            },
            "inputs": {
                str(packet["packet_id"]): {
                    "result_sha256": PACKET_SPECS[str(packet["packet_id"])][
                        "result_sha256"
                    ],
                    "payload_sha256": packet["payload"]["payload_sha256"],
                    "source_manifest_sha256": sha256_file(
                        Path(packet["root"]) / "source_manifest.json"
                    ),
                    "source_manifest_payload_sha256": packet["source_manifest"][
                        "payload_sha256"
                    ],
                    "verified_artifact_count": len(packet["verified_inventory"]),
                    "checkout_comparison": packet["checkout_comparison"],
                }
                for packet in packets
            },
            "contract": {
                "active_cases": list(ACTIVE_CASES),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "input_calls": list(INPUT_CALLS),
                "exact_calls": list(EXACT_CALLS),
                "coast_calls": list(COAST_CALLS),
                "bands": {key: list(value) for key, value in BANDS.items()},
                "features": list(FEATURES),
                "models": list(MODELS),
                "ridge_penalty": RIDGE_PENALTY,
                "feature_scale_floor": FEATURE_SCALE_FLOOR,
                "denominator_floor_squared": DENOMINATOR_FLOOR_SQUARED,
                "coefficient_floor": COEFFICIENT_FLOOR,
                "coefficient_scale_limit": COEFFICIENT_SCALE_LIMIT,
                "call_outcomes": CALL_OUTCOMES,
                "structural_views": list(STRUCTURAL_VIEWS),
                "counterfactual_route_available": False,
                "true_error_predictor": False,
            },
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "active_case_inventory_8": len(ACTIVE_CASES) == 8,
        "strength_group_inventory_4": len(GROUP_CASES) == 4,
        "call_inventory_30": INPUT_CALLS == tuple(range(30)),
        "exact_inventory_16": len(EXACT_CALLS) == 16,
        "coast_inventory_14": len(COAST_CALLS) == 14,
        "band_partition_exact": sorted(
            call for calls in BANDS.values() for call in calls
        )
        == list(INPUT_CALLS),
        "predictor_inventory_3": len(FEATURES) == 3,
        "model_inventory_5": len(MODELS) == 5,
        "owned_source_inventory_3": len(OWNED_SOURCE_PATHS) == 3,
        "counterfactual_not_available": True,
    }
    return {
        "schema": SYNTHETIC_SCHEMA,
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "model_calls": 0,
        "truth_arrays_loaded": False,
        "recurrent_policy_simulated": False,
    }


def _run_crossfits(
    call_rows: Sequence[Mapping[str, Any]],
    structural_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    call_predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for split_key, name in (
        ("group_id", "leave_strength_group_out"),
        ("case_id", "leave_case_out"),
    ):
        predictions, fitted = crossfit_predictions(
            call_rows,
            targets=tuple(CALL_OUTCOMES),
            split_key=split_key,
            crossfit=name,
        )
        call_predictions.extend(predictions)
        coefficients.extend(fitted)
    call_metrics = summarize_predictions(call_predictions)

    structural_predictions: list[dict[str, Any]] = []
    structural_coefficients: list[dict[str, Any]] = []
    for view in STRUCTURAL_VIEWS:
        rows = [
            dict(row, **{view: row["utility"]})
            for row in structural_rows
            if row["view"] == view
        ]
        for split_key, name in (
            ("group_id", "leave_strength_group_out"),
            ("case_id", "leave_case_out"),
        ):
            predictions, fitted = crossfit_predictions(
                rows, targets=(view,), split_key=split_key, crossfit=name
            )
            structural_predictions.extend(predictions)
            structural_coefficients.extend(fitted)
    structural_metrics = summarize_predictions(structural_predictions)
    coefficients.extend(structural_coefficients)
    return {
        "call_predictions": call_predictions,
        "call_metrics": call_metrics,
        "structural_predictions": structural_predictions,
        "structural_metrics": structural_metrics,
        "coefficients": coefficients,
    }


def run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    packets = (
        verify_packet(args.a43_result, "a43"),
        verify_packet(args.a44_result, "a44_r1"),
    )
    call_rows = build_call_rows(packets)
    structural_rows = build_structural_rows(packets, call_rows)
    results = _run_crossfits(call_rows, structural_rows)
    stabilities = [
        coefficient_stability(results["coefficients"], target, feature)
        for target in PRIMARY_OUTCOMES
        for feature in FEATURES
    ]
    structural_checks = {
        "input_packet_count_exact": len(packets) == 2,
        "input_artifact_inventories_verified": all(
            len(packet["verified_inventory"]) == 15 for packet in packets
        ),
        "input_source_drift_only_registered": all(
            packet["checkout_comparison"]["status"] == "passed" for packet in packets
        ),
        "active_case_inventory_exact": {row["case_id"] for row in call_rows}
        == set(ACTIVE_CASES),
        "active_call_rectangle_exact": len(call_rows)
        == len(ACTIVE_CASES) * len(INPUT_CALLS),
        "route_inventory_exact": sum(
            row["route"] == "exact_a32_window" for row in call_rows
        )
        == len(ACTIVE_CASES) * len(EXACT_CALLS)
        and sum(row["route"] == "surrogate_coast" for row in call_rows)
        == len(ACTIVE_CASES) * len(COAST_CALLS),
        "structural_rectangle_exact": len(structural_rows)
        == len(ACTIVE_CASES) * len(BANDS) * len(STRUCTURAL_VIEWS),
        "call_zero_history_exact_zero": all(
            row["lag_displacement"] == 0.0
            and row["lag_intervention"] == 0.0
            and row["lag_displacement_change"] == 0.0
            for row in call_rows
            if row["input_call"] == 0
        ),
        "no_current_call_feature": all(
            feature
            not in {
                "current_retained_displacement_rms_audit_only",
                "current_correction_rms_audit_only",
            }
            for feature in FEATURES
        ),
        "truth_error_predictor_absent": all(
            "error" not in feature and "truth" not in feature for feature in FEATURES
        ),
        "counterfactual_route_unavailable": True,
        "recurrent_policy_not_simulated": True,
    }
    gate = prospective_gate(
        results["call_metrics"],
        results["structural_metrics"],
        stabilities,
        structural_checks=structural_checks,
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("A45 output directory already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _source_manifest(packets)
    write_csv(output_dir / "call_predictions.csv", results["call_predictions"])
    write_csv(output_dir / "fold_metrics.csv", results["call_metrics"])
    write_csv(output_dir / "coefficients.csv", results["coefficients"])
    write_csv(
        output_dir / "structural_predictions.csv", results["structural_predictions"]
    )
    write_csv(output_dir / "structural_metrics.csv", results["structural_metrics"])
    atomic_write_json(output_dir / "source_manifest.json", source_manifest)
    artifacts = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    payload = with_payload_sha256(
        {
            "schema": RESULT_SCHEMA,
            "working_id": WORKING_ID,
            "status": (
                "qualified_feature_signal"
                if gate["status"] == "qualified"
                else "stopped_logged_utility_diagnostic"
            ),
            "gate": gate,
            "coefficient_stability": stabilities,
            "primary_metrics": [
                row
                for row in results["call_metrics"]
                if row["crossfit"] == "leave_strength_group_out"
                and row["target"] in PRIMARY_OUTCOMES
                and row["model"] == "phase_history"
            ],
            "structural_primary_metrics": [
                row
                for row in results["structural_metrics"]
                if row["crossfit"] == "leave_strength_group_out"
                and row["target"] in REQUIRED_STRUCTURAL_VIEWS
                and row["model"] == "phase_history"
            ],
            "input_results": {
                str(packet["packet_id"]): {
                    "result_sha256": PACKET_SPECS[str(packet["packet_id"])][
                        "result_sha256"
                    ],
                    "payload_sha256": packet["payload"]["payload_sha256"],
                    "verified_artifact_count": len(packet["verified_inventory"]),
                }
                for packet in packets
            },
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "artifact_inventory_before_summary": artifacts,
            "population": {
                "active_cases": list(ACTIVE_CASES),
                "call_rows": len(call_rows),
                "structural_rows": len(structural_rows),
                "groups": list(GROUP_CASES),
            },
            "execution": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "device": "cpu",
                "model_calls": 0,
                "truth_arrays_loaded": False,
                "state_arrays_loaded": False,
                "animation_bundles_loaded": False,
                "recurrence_executed": False,
                "recurrent_policy_simulated": False,
                "counterfactual_route_available": False,
                "new_population_opened": False,
            },
            "claim_boundary": (
                "Grouped prediction of logged corrected-versus-raw recurrent "
                "advantage from lagged inference-available scalar histories on "
                "eight already-open same-checkpoint dynamic-FV cases. No causal "
                "fine-refresh, controller, recurrence, cost, latency, independent "
                "checkpoint/test, cross-family, conservation, convergence, "
                "off-grid, Richardson, or deployment claim."
            ),
        }
    )
    atomic_write_json(output_dir / "marginal_utility_audit.json", payload)
    return payload, 0 if gate["status"] == "qualified" else 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--a43-result", type=Path, required=True)
    analyze.add_argument("--a44-result", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    payload, exit_code = run_analysis(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
