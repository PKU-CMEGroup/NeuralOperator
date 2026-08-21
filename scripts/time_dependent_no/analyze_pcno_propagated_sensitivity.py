#!/usr/bin/env python3
"""Calibrate a target-free propagated-sensitivity gate for W26-L5 SP19.

The scientific command is calibration-only.  It replays the native proposal,
reconstructs SP19 from a frozen calibration modal artifact, and evaluates raw
and corrected native lookaheads.  No evaluation, recurrent, bump, or sealed
population command exists in this entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as collection,
)
from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as parent,
)
from scripts.time_dependent_no import (
    evaluate_pcno_sparse_modal_correction as sparse_runner,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file, sha256_files
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    NativeIncrementBasis,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    FINE_AWAY_GAIN,
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_propagated_sensitivity import (
    PolicyRecord,
    grouped_directional_crossfit,
    grouped_gain_threshold_crossfit,
    modal_partition_fields,
    propagated_lookahead,
    weighted_quadratic_statistics,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    predict_resolution_sample,
    reference_at_resolution,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    SPARSE_POLICY,
    sparse_modal_correction,
)

WORKING_ID = "W26-L5-P5-PSJ-A2-CAL"
PREFLIGHT_SCHEMA = "pcno_propagated_sensitivity_preflight_v1"
RESULT_SCHEMA = "pcno_propagated_sensitivity_calibration_v1"
INPUT_CALLS = tuple(range(29))
NATIVE_RESOLUTION = parent.NATIVE_RESOLUTION
EXPECTED_MASK_CONTRACT_SHA256 = (
    "d89a1cf4c3c1f4e1fc9da4698ad16bcd5ddcd70c6c3ccfa2a2097cd735178391"
)
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_PROPAGATED_SENSITIVITY_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_propagated_sensitivity.py",
    "scripts/time_dependent_no/analyze_pcno_propagated_sensitivity.py",
    "tests/time_dependent_no/test_pcno_propagated_sensitivity.py",
)
MODAL_VIEW_NAMES = (
    "constant",
    "low_active_modes_1_3",
    "upper_active_modes_4_7",
    "rank8_inactive",
    "high_rank_remainder",
    "sp19_active",
)
REGION_NAMES = ("boundary", "shock", "vortex", "smooth")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
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


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def _verify_parent_artifacts(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mask = sparse_runner._require_mask_contract(args.mask_contract)
    if sha256_file(args.mask_contract) != EXPECTED_MASK_CONTRACT_SHA256:
        raise ValueError("SP19 mask contract differs from the preregistration")
    calibration, parent_source = sparse_runner._verify_parent_inputs(args)
    if sha256_file(args.modal_records) != sparse_runner.EXPECTED_MODAL_RECORDS_SHA256:
        raise ValueError("frozen calibration modal records differ")
    if calibration.get("artifact_hashes", {}).get(
        "calibration_modal_records.csv"
    ) != sha256_file(args.modal_records):
        raise ValueError("modal records are not owned by the parent calibration")
    return mask, calibration, parent_source


def _preflight_payload(
    args: argparse.Namespace,
    *,
    mask: Mapping[str, Any],
    calibration: Mapping[str, Any],
    parent_source: Mapping[str, Any],
    case_inventory_present: bool,
    all_cases_open_validation: bool,
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": PREFLIGHT_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed",
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "parent_calibration_sha256": sha256_file(args.parent_calibration),
            "parent_calibration_payload_sha256": calibration["payload_sha256"],
            "parent_source_manifest_sha256": sha256_file(args.parent_source_manifest),
            "parent_source_payload_sha256": parent_source["payload_sha256"],
            "modal_records_sha256": sha256_file(args.modal_records),
            "mask_contract_sha256": sha256_file(args.mask_contract),
            "mask_contract_payload_sha256": mask["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "case_ids": list(CALIBRATION_CASE_IDS),
                "groups": {
                    key: list(value) for key, value in CALIBRATION_GROUPS.items()
                },
                "input_calls": list(INPUT_CALLS),
                "evaluation_cases": "closed",
                "strength_ood_and_test": "sealed",
            },
            "case_inventory_present": case_inventory_present,
            "all_cases_open_validation": all_cases_open_validation,
            "checkpoint_model_built": False,
            "reference_arrays_loaded": False,
            "gradient_path_modified": False,
            "live_inference_logical_calls_per_decision": {
                "native_current": 1,
                "fine_current": 1,
                "native_raw_lookahead": 1,
                "native_corrected_lookahead": 1,
                "total": 4,
            },
            "collection_replay_logical_calls_per_sample": 3,
            "claim_boundary": (
                "Calibration-only dynamic-FV propagated-sensitivity diagnostic; "
                "no recurrent, evaluation, bump, sealed, conservation, or "
                "cross-family coefficient claim."
            ),
        }
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    mask, calibration, frozen_parent_source = _verify_parent_artifacts(args)
    _, manifest, store, current_parent_source = parent._open_contract(args)
    try:
        if current_parent_source != frozen_parent_source:
            raise ValueError("current parent source differs from the frozen source")
        splits = {
            case_id: collection.family_case_provenance(manifest, case_id)["split"]
            for case_id in CALIBRATION_CASE_IDS
        }
        payload = _preflight_payload(
            args,
            mask=mask,
            calibration=calibration,
            parent_source=frozen_parent_source,
            case_inventory_present=all(
                case_id in store.keys for case_id in CALIBRATION_CASE_IDS
            ),
            all_cases_open_validation=all(
                value == "validation" for value in splits.values()
            ),
        )
    finally:
        store.close()
    if (
        not payload["case_inventory_present"]
        or not payload["all_cases_open_validation"]
    ):
        raise ValueError("calibration population is incomplete or not open validation")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("status") != "passed":
        raise ValueError("propagated-sensitivity preflight did not pass")
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("source differs from the frozen preflight")
    expected = {
        "parent_calibration_sha256": sha256_file(args.parent_calibration),
        "parent_source_manifest_sha256": sha256_file(args.parent_source_manifest),
        "modal_records_sha256": sha256_file(args.modal_records),
        "mask_contract_sha256": sha256_file(args.mask_contract),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("parent artifact identity differs from the frozen preflight")
    return payload


def _read_modal_records(path: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    lookup: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    observed_cells: dict[tuple[str, int], set[tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row["case_id"]
            group_id = row["group_id"]
            input_call = int(row["input_call"])
            mode = int(row["mode_index"])
            component = int(row["component"])
            if case_id not in CALIBRATION_CASE_IDS or group_id != case_id.split("_")[1]:
                raise ValueError("modal record case/group inventory differs")
            if (
                input_call not in ALL_INPUT_CALLS
                or not 0 <= mode < 8
                or not 0 <= component < 4
            ):
                raise ValueError("modal record call/cell inventory differs")
            key = (case_id, input_call)
            if key not in lookup:
                lookup[key] = {
                    "fine": np.zeros((8, 4), dtype=np.float64),
                    "target": np.zeros((8, 4), dtype=np.float64),
                }
                observed_cells[key] = set()
            cell = (mode, component)
            if cell in observed_cells[key]:
                raise ValueError("duplicate modal record cell")
            observed_cells[key].add(cell)
            lookup[key]["fine"][cell] = float(row["fine_coordinate"])
            lookup[key]["target"][cell] = float(row["target_coordinate"])
    expected_pairs = {
        (case_id, input_call)
        for case_id in CALIBRATION_CASE_IDS
        for input_call in ALL_INPUT_CALLS
    }
    expected_cells = {(mode, component) for mode in range(8) for component in range(4)}
    if set(lookup) != expected_pairs or any(
        cells != expected_cells for cells in observed_cells.values()
    ):
        raise ValueError(
            "modal record inventory is not the complete frozen calibration"
        )
    return lookup


def _reconstructed_correction(
    coordinates: np.ndarray,
    *,
    native_increment: np.ndarray,
    runtime: Any,
) -> tuple[np.ndarray, Any, float]:
    feature = reconstruct_modal_field(
        coordinates,
        runtime.native_projector,
        component_scale=runtime.normalization.residual_scale,
    )
    basis = NativeIncrementBasis(
        native_increment=np.asarray(native_increment, dtype=np.float64),
        coarse_on_native=np.asarray(native_increment, dtype=np.float64).copy(),
        fine_on_native=np.asarray(native_increment, dtype=np.float64) + feature,
        native_minus_coarse=np.zeros_like(native_increment, dtype=np.float64),
        fine_minus_native=feature,
    )
    correction, audit = sparse_modal_correction(
        basis,
        runtime.native_projector,
        policy=SPARSE_POLICY,
        volumes=runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
        component_scale=runtime.normalization.residual_scale,
        active_cells=FROZEN_ACTIVE_CELLS,
    )
    expected = np.zeros_like(coordinates)
    for cell in FROZEN_ACTIVE_CELLS:
        expected[cell] = -FINE_AWAY_GAIN * coordinates[cell]
    actual = modal_coordinates(
        correction,
        runtime.native_projector,
        component_scale=runtime.normalization.residual_scale,
    )
    return correction, audit, float(np.max(np.abs(actual - expected)))


def _view_rows(
    *,
    case_id: str,
    group_id: str,
    input_call: int,
    baseline_error: np.ndarray,
    response: np.ndarray,
    target: np.ndarray,
    runtime: Any,
) -> tuple[list[dict[str, Any]], float, int]:
    volumes = runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures
    scale = runtime.normalization.state_scale
    error_parts = modal_partition_fields(
        baseline_error,
        runtime.native_projector,
        component_scale=scale,
    )
    response_parts = modal_partition_fields(
        response,
        runtime.native_projector,
        component_scale=scale,
    )
    modal_sum_error = sum(
        error_parts[name]
        for name in (
            "constant",
            "low_active_modes_1_3",
            "upper_active_modes_4_7",
            "rank8_inactive",
            "high_rank_remainder",
        )
    )
    modal_sum_response = sum(
        response_parts[name]
        for name in (
            "constant",
            "low_active_modes_1_3",
            "upper_active_modes_4_7",
            "rank8_inactive",
            "high_rank_remainder",
        )
    )
    closure = max(
        float(np.max(np.abs(modal_sum_error - baseline_error))),
        float(np.max(np.abs(modal_sum_response - response))),
    )
    rows: list[dict[str, Any]] = []

    def add(view: str, error: np.ndarray, change: np.ndarray, mask=None) -> None:
        rows.append(
            {
                "case_id": case_id,
                "group_id": group_id,
                "input_call": input_call,
                "view": view,
                **weighted_quadratic_statistics(
                    error,
                    change,
                    volumes=volumes,
                    component_scale=scale,
                    mask=mask,
                ),
            }
        )

    add("full", baseline_error, response)
    for name in MODAL_VIEW_NAMES:
        add(name, error_parts[name], response_parts[name])
    for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
        error = np.zeros_like(baseline_error)
        change = np.zeros_like(response)
        error[:, component] = baseline_error[:, component]
        change[:, component] = response[:, component]
        add(f"component_{name}", error, change)
    masks, _, _ = shock_vortex_regions(
        target,
        runtime.geometry_by_resolution[NATIVE_RESOLUTION].nodes,
        resolution=NATIVE_RESOLUTION,
        gamma=runtime.normalization.gamma,
    )
    partition_count = sum(
        masks[f"partition_{name}"].astype(np.int8) for name in REGION_NAMES
    )
    for name in REGION_NAMES:
        add(
            f"region_{name}",
            baseline_error,
            response,
            masks[f"partition_{name}"],
        )
    return rows, closure, int(np.max(np.abs(partition_count - 1)))


def _account(execution: dict[str, Any], timing: Mapping[str, Any]) -> None:
    execution["logical_model_calls"] += 1
    forwards = timing["forward_seconds"]
    execution["actual_forward_passes_including_repeats"] += len(forwards)
    execution["total_forward_seconds"] += float(sum(forwards))
    repeat = float(timing["repeat_max_abs"])
    if math.isfinite(repeat):
        execution["maximum_repeat_abs_difference"] = max(
            execution["maximum_repeat_abs_difference"], repeat
        )
    peak = timing.get("peak_gpu_memory_bytes")
    if peak is not None:
        execution["maximum_peak_gpu_memory_bytes"] = max(
            execution["maximum_peak_gpu_memory_bytes"], int(peak)
        )


def _predict_native(
    runtime: Any,
    state: np.ndarray,
    *,
    repeats: int,
    execution: dict[str, Any],
) -> np.ndarray:
    prediction, timing = predict_resolution_sample(
        runtime.model,
        runtime.sample_by_resolution[NATIVE_RESOLUTION],
        state,
        device=runtime.device,
        amp="none",
        repeats=repeats,
    )
    _account(execution, timing)
    return np.asarray(prediction, dtype=np.float64)


def _flatten_policy_record(record: PolicyRecord) -> dict[str, Any]:
    row = {
        "case_id": record.case_id,
        "group_id": record.group_id,
        "input_call": record.input_call,
        "response_gain": record.response_gain,
        "raw_state_energy": record.raw_state_energy,
        "corrected_state_energy": record.corrected_state_energy,
        "corrected_valid": record.corrected_valid,
    }
    for key, value in record.raw_control_energy.items():
        row[f"raw::{key}"] = value
        row[f"corrected::{key}"] = record.corrected_control_energy[key]
    return row


def _threshold_fold_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fold in payload["folds"]:
        rows.append(
            {
                "held_group": fold["held_group"],
                "selected_name": fold["selected_name"],
                "selected_threshold": fold["selected_threshold"],
                "training_coverage": fold["training"]["coverage"],
                "training_state_rms_ratio": fold["training"]["state_rms_ratio"],
                "held_coverage": fold["held_out"]["coverage"],
                "held_state_rms_ratio": fold["held_out"]["state_rms_ratio"],
                "held_all_selected_valid": fold["held_out"]["all_selected_valid"],
            }
        )
    return rows


def run_calibration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight, args)
    mask, calibration, frozen_parent_source = _verify_parent_artifacts(args)
    modal_lookup = _read_modal_records(args.modal_records)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, current_parent_source = parent._build_runtime(args)
    try:
        if current_parent_source != frozen_parent_source:
            raise ValueError("runtime parent source differs from the frozen source")
        started = perf_counter()
        native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
        sample_rows: list[dict[str, Any]] = []
        view_rows: list[dict[str, Any]] = []
        policy_records: list[PolicyRecord] = []
        reference_rows: list[dict[str, Any]] = []
        execution = {
            "logical_model_calls": 0,
            "actual_forward_passes_including_repeats": 0,
            "total_forward_seconds": 0.0,
            "maximum_repeat_abs_difference": 0.0,
            "maximum_peak_gpu_memory_bytes": 0,
            "live_logical_calls_per_decision": 4,
            "collection_logical_calls_per_sample": 3,
        }
        maxima = {
            "frozen_target_modal_abs": 0.0,
            "correction_modal_abs": 0.0,
            "immediate_correction_integral_abs": 0.0,
            "modal_partition_abs": 0.0,
            "quadratic_closure_abs": 0.0,
            "region_partition_error": 0,
            "correction_to_native_increment": 0.0,
            "propagated_integral_abs": 0.0,
        }
        for case_index, case_id in enumerate(CALIBRATION_CASE_IDS, start=1):
            provenance = collection.family_case_provenance(runtime.manifest, case_id)
            if provenance["split"] != "validation" or case_id not in runtime.store.keys:
                raise ValueError(
                    "calibration case is outside the open bound population"
                )
            reference, reference_check = load_resolution_reference(
                runtime.args.family_root,
                runtime.args.multires_reference_root,
                runtime.store,
                runtime.manifest,
                case_id,
                training_resolution=NATIVE_RESOLUTION,
            )
            reference_rows.append(reference_check)
            reference_resolution = tuple(
                int(value) for value in reference["retained_resolution"]
            )
            if reference_resolution != parent.RESOLUTION_CONTRACT.fine:
                raise ValueError(
                    "propagated sensitivity requires the retained fine source"
                )
            print(
                f"propagated sensitivity: case {case_index}/{len(CALIBRATION_CASE_IDS)} "
                f"{case_id} calls=0..28",
                flush=True,
            )
            for input_call in INPUT_CALLS:
                states = []
                for offset in (0, 1, 2):
                    state = reference_at_resolution(
                        reference["conservative_states"][(input_call + offset) * 2],
                        reference_resolution=reference_resolution,
                        target_resolution=NATIVE_RESOLUTION,
                    )
                    if state is None:
                        raise ValueError(
                            "required two-step native reference is unavailable"
                        )
                    states.append(np.asarray(state, dtype=np.float64))
                native_input, native_target, two_step_target = states
                repeats = args.repeat_forward if input_call == INPUT_CALLS[0] else 1
                raw_proposal = _predict_native(
                    runtime,
                    native_input,
                    repeats=repeats,
                    execution=execution,
                )
                frozen = modal_lookup[(case_id, input_call)]
                target_coordinates = modal_coordinates(
                    native_target - raw_proposal,
                    runtime.native_projector,
                    component_scale=runtime.normalization.residual_scale,
                )
                maxima["frozen_target_modal_abs"] = max(
                    maxima["frozen_target_modal_abs"],
                    float(np.max(np.abs(target_coordinates - frozen["target"]))),
                )
                correction, audit, modal_closure = _reconstructed_correction(
                    frozen["fine"],
                    native_increment=raw_proposal - native_input,
                    runtime=runtime,
                )
                if audit.status != "ok" or audit.cap_active:
                    raise ValueError(
                        "the frozen SP19 correction unexpectedly hit its cap"
                    )
                maxima["correction_modal_abs"] = max(
                    maxima["correction_modal_abs"], modal_closure
                )
                maxima["correction_to_native_increment"] = max(
                    maxima["correction_to_native_increment"],
                    float(audit.correction_to_native_increment),
                )
                correction_integral = collection._physical_integral(
                    correction, native_geometry.node_measures
                )
                maxima["immediate_correction_integral_abs"] = max(
                    maxima["immediate_correction_integral_abs"],
                    float(np.max(np.abs(correction_integral))),
                )

                def predictor(
                    state: np.ndarray, repeat_count: int = repeats
                ) -> np.ndarray:
                    return _predict_native(
                        runtime,
                        state,
                        repeats=repeat_count,
                        execution=execution,
                    )

                lookahead = propagated_lookahead(
                    raw_proposal,
                    correction,
                    predictor=predictor,
                )
                baseline_error = lookahead.raw_prediction - two_step_target
                response = lookahead.response
                corrected_error = baseline_error + response
                propagated_integral = collection._physical_integral(
                    response, native_geometry.node_measures
                )
                maxima["propagated_integral_abs"] = max(
                    maxima["propagated_integral_abs"],
                    float(np.max(np.abs(propagated_integral))),
                )
                rows, modal_partition_abs, region_partition_error = _view_rows(
                    case_id=case_id,
                    group_id=case_id.split("_")[1],
                    input_call=input_call,
                    baseline_error=baseline_error,
                    response=response,
                    target=two_step_target,
                    runtime=runtime,
                )
                if any(row["status"] != "ok" for row in rows):
                    raise ValueError(
                        "a registered propagated-sensitivity view is unresolved"
                    )
                view_rows.extend(rows)
                maxima["modal_partition_abs"] = max(
                    maxima["modal_partition_abs"], modal_partition_abs
                )
                maxima["region_partition_error"] = max(
                    maxima["region_partition_error"], region_partition_error
                )
                maxima["quadratic_closure_abs"] = max(
                    maxima["quadratic_closure_abs"],
                    max(float(row["quadratic_closure_abs"]) for row in rows),
                )
                full = next(row for row in rows if row["view"] == "full")
                correction_rms = weighted_scaled_rms(
                    correction,
                    volumes=native_geometry.node_measures,
                    component_scale=runtime.normalization.state_scale,
                )
                response_rms = weighted_scaled_rms(
                    response,
                    volumes=native_geometry.node_measures,
                    component_scale=runtime.normalization.state_scale,
                )
                if correction_rms <= 1.0e-12:
                    raise ValueError("SP19 correction has an unresolved norm")
                response_gain = response_rms / correction_rms
                raw_front = collection._front_errors(
                    lookahead.raw_prediction, two_step_target, runtime
                )
                corrected_front = collection._front_errors(
                    lookahead.corrected_prediction, two_step_target, runtime
                )
                raw_integral = collection._physical_integral(
                    baseline_error, native_geometry.node_measures
                )
                corrected_integral = collection._physical_integral(
                    corrected_error, native_geometry.node_measures
                )
                raw_controls = {
                    **{
                        f"front::{key}": float(raw_front[key]) ** 2
                        for key in FRONT_CONTROL_KEYS
                    },
                    **{
                        f"integral::{name}": float(raw_integral[component]) ** 2
                        for component, name in enumerate(INTEGRAL_COMPONENT_NAMES)
                    },
                }
                corrected_controls = {
                    **{
                        f"front::{key}": float(corrected_front[key]) ** 2
                        for key in FRONT_CONTROL_KEYS
                    },
                    **{
                        f"integral::{name}": float(corrected_integral[component]) ** 2
                        for component, name in enumerate(INTEGRAL_COMPONENT_NAMES)
                    },
                }
                immediate_admissibility = conservative_admissibility_summary(
                    raw_proposal + correction,
                    gamma=runtime.normalization.gamma,
                )
                lookahead_admissibility = conservative_admissibility_summary(
                    lookahead.corrected_prediction,
                    gamma=runtime.normalization.gamma,
                )
                valid = bool(
                    np.isfinite(raw_proposal + correction).all()
                    and np.isfinite(lookahead.corrected_prediction).all()
                    and immediate_admissibility["admissible"]
                    and lookahead_admissibility["admissible"]
                )
                record = PolicyRecord(
                    case_id=case_id,
                    group_id=case_id.split("_")[1],
                    input_call=input_call,
                    response_gain=response_gain,
                    raw_state_energy=float(full["error_energy"]),
                    corrected_state_energy=float(full["corrected_energy"]),
                    raw_control_energy=raw_controls,
                    corrected_control_energy=corrected_controls,
                    corrected_valid=valid,
                )
                policy_records.append(record)
                sample_rows.append(
                    {
                        **_flatten_policy_record(record),
                        "correction_rms": correction_rms,
                        "response_rms": response_rms,
                        "response_error_cross": full["cross"],
                        "response_error_cosine": full["cosine"],
                        "immediate_minimum_density": immediate_admissibility[
                            "minimum_density"
                        ],
                        "immediate_minimum_pressure": immediate_admissibility[
                            "minimum_pressure"
                        ],
                        "lookahead_minimum_density": lookahead_admissibility[
                            "minimum_density"
                        ],
                        "lookahead_minimum_pressure": lookahead_admissibility[
                            "minimum_pressure"
                        ],
                        **{
                            f"immediate_integral::{name}": float(
                                correction_integral[component]
                            )
                            for component, name in enumerate(INTEGRAL_COMPONENT_NAMES)
                        },
                        **{
                            f"propagated_integral::{name}": float(
                                propagated_integral[component]
                            )
                            for component, name in enumerate(INTEGRAL_COMPONENT_NAMES)
                        },
                    }
                )
            del reference
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        threshold = grouped_gain_threshold_crossfit(
            policy_records,
            expected_groups=CALIBRATION_GROUPS,
            expected_input_calls=INPUT_CALLS,
        )
        directional = grouped_directional_crossfit(
            view_rows,
            expected_groups=CALIBRATION_GROUPS,
            expected_input_calls=INPUT_CALLS,
        )
        execution["wall_seconds"] = perf_counter() - started
        execution["snapshot_count"] = len(policy_records)
        execution["device"] = str(runtime.device)
        execution["amp"] = "none"
        closure_checks = {
            "inventory_exact": len(policy_records)
            == len(CALIBRATION_CASE_IDS) * len(INPUT_CALLS),
            "frozen_target_modal_replay_at_most_1e_6": maxima["frozen_target_modal_abs"]
            <= 1.0e-6,
            "correction_modal_closure_at_most_1e_10": maxima["correction_modal_abs"]
            <= 1.0e-10,
            "immediate_integral_neutral_at_most_1e_10": maxima[
                "immediate_correction_integral_abs"
            ]
            <= 1.0e-10,
            "modal_partition_closure_at_most_1e_10": maxima["modal_partition_abs"]
            <= 1.0e-10,
            "quadratic_closure_at_most_1e_12": maxima["quadratic_closure_abs"]
            <= 1.0e-12,
            "region_partition_exact": maxima["region_partition_error"] == 0,
            "correction_cap_inactive": maxima["correction_to_native_increment"] < 0.10,
            "deterministic_repeat_at_most_1e_6": execution[
                "maximum_repeat_abs_difference"
            ]
            <= 1.0e-6,
        }
        scientific_gate = threshold["prospective_gate"]
        qualified = (
            all(closure_checks.values()) and scientific_gate["evaluation_authorized"]
        )
        write_csv(output_dir / "sample_records.csv", sample_rows)
        write_csv(output_dir / "directional_view_records.csv", view_rows)
        write_csv(
            output_dir / "directional_crossfit_folds.csv",
            directional["fold_rows"],
        )
        write_csv(
            output_dir / "threshold_crossfit_folds.csv",
            _threshold_fold_rows(threshold),
        )
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        atomic_write_json(output_dir / "preflight.json", preflight)
        files = (
            "sample_records.csv",
            "directional_view_records.csv",
            "directional_crossfit_folds.csv",
            "threshold_crossfit_folds.csv",
            "reference_checks.csv",
            "preflight.json",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        payload = with_payload_sha256(
            {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "qualified" if qualified else "stopped",
                "population_status": "adaptive_open_validation_calibration_only",
                "correction": {
                    "policy": SPARSE_POLICY,
                    "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                    "gain": -FINE_AWAY_GAIN,
                    "true_error_available_at_inference": False,
                    "propagated_response_target_free": True,
                },
                "threshold_crossfit": threshold,
                "directional_crossfit": directional,
                "closure_checks": closure_checks,
                "closure_maxima": maxima,
                "execution": execution,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "parent_calibration_sha256": sha256_file(args.parent_calibration),
                "parent_calibration_payload_sha256": calibration["payload_sha256"],
                "mask_contract_sha256": sha256_file(args.mask_contract),
                "mask_contract_payload_sha256": mask["payload_sha256"],
                "modal_records_sha256": sha256_file(args.modal_records),
                "artifact_hashes": artifact_hashes,
                "evaluation_executed": False,
                "recurrence_executed": False,
                "sealed_population_opened": False,
                "claim_boundary": (
                    "Cross-fitted calibration evidence for one dynamic-FV checkpoint "
                    "and frozen SP19 correction. The response is target-free; fitting "
                    "and scoring use calibration truth. No bump, cross-family, recurrent, "
                    "conservative-solver, or sealed claim."
                ),
            }
        )
        atomic_write_json(output_dir / "calibration.json", payload)
        return payload, 0 if qualified else 4
    finally:
        collection._close_runtime(runtime)


def synthetic_summary() -> dict[str, Any]:
    raw = np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0
    correction = np.full_like(raw, 0.05)
    calls: list[np.ndarray] = []

    def predictor(state: np.ndarray) -> np.ndarray:
        calls.append(np.array(state, copy=True))
        return 1.2 * state + 0.1

    lookahead = propagated_lookahead(raw, correction, predictor=predictor)
    error = lookahead.raw_prediction - (1.1 * raw)
    stats = weighted_quadratic_statistics(
        error,
        lookahead.response,
        volumes=np.ones(raw.shape[0], dtype=np.float64),
        component_scale=np.ones(raw.shape[1], dtype=np.float64),
    )
    checks = {
        "exactly_two_lookahead_calls": len(calls) == 2,
        "raw_then_corrected_inputs": np.array_equal(calls[0], raw)
        and np.array_equal(calls[1], raw + correction),
        "target_free_response_exact": np.allclose(lookahead.response, 1.2 * correction),
        "quadratic_identity": float(stats["quadratic_closure_abs"]) <= 1.0e-14,
        "evaluation_command_absent": True,
        "recurrent_command_absent": True,
    }
    return with_payload_sha256(
        {
            "schema": "pcno_propagated_sensitivity_synthetic_v1",
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
        }
    )


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parent._add_external_arguments(parser)
    parser.add_argument("--mask-contract", type=Path, required=True)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--parent-source-manifest", type=Path, required=True)
    parser.add_argument("--modal-records", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate")
    _add_external_arguments(calibration)
    parent._add_runtime_arguments(calibration)
    calibration.add_argument("--preflight", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        exit_code = 0 if payload["status"] == "passed" else 2
    elif args.command == "preflight":
        payload = run_preflight(args)
        exit_code = 0
    elif args.command == "calibrate":
        if args.repeat_forward < 1:
            raise ValueError("repeat-forward must be positive")
        payload, exit_code = run_calibration(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
