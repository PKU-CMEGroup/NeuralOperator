"""Test target-free accepted-shadow response surrogates for the A32 tether."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import benchmark_pcno_paired_native_batch as a40
from scripts.time_dependent_no import evaluate_pcno_affine_shadow_tether_rollout as a32
from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no import evaluate_pcno_modal_affine_transfer as a29
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    runtime_environment,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    NativeIncrementBasis,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_fine_discrepancy_correction import (
    modal_coordinates,
    reconstruct_modal_field,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    as_model_state,
    predict_resolution_sample,
    restrict_nested_state,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    projected_shadow_tether,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    affine_modal_correction,
    binary_position_phase_correction_active,
)

WORKING_ID = "W26-L5-P6-RFB19-A41-SHADOW-RESPONSE-GEOMETRY"
PREFLIGHT_SCHEMA = "pcno_shadow_response_geometry_preflight_v1"
RESULT_SCHEMA = "pcno_shadow_response_geometry_v1"
SOURCE_SCHEMA = "pcno_shadow_response_geometry_source_v1"
A40_RESULT_SHA256 = "c4f4250a51877d6ab86d31ac223a14cf1d2779a10bc23b018e0245de076fd489"
A40_PAYLOAD_SHA256 = "6cb0f2a481523a9988540ea345452c048f313c4443abf7d9578e219b58b3e743"
MAP_NAMES = ("zero", "identity", "scalar", "diagonal")
CALIBRATION_CASES = ("sv_e12_y01", "sv_e12_y07")
ZERO_CASES = ("sv_e12_y04",)
TRANSFER_CASES = ("sv_e14_y01", "sv_e14_y07")
ACTIVE_CASES = (*CALIBRATION_CASES, *TRANSFER_CASES)
FIXED_BANDS = {
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
}
DENOMINATOR_FLOOR = 1.0e-10
REPEAT_TOLERANCE = 1.0e-6
STRICT_TOLERANCE = 1.0e-12
REPEATS = 2
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_pcno_shadow_response_geometry.py",
    "tests/time_dependent_no/test_pcno_shadow_response_geometry.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/benchmark_pcno_paired_native_batch.py",
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "utility/time_dependent_no/pcno_cross_resolution_correction.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "utility/time_dependent_no/pcno_fine_discrepancy_correction.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _verify_a40(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A40_RESULT_SHA256:
        raise ValueError("A40 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != a40.RESULT_SCHEMA
        or payload.get("working_id") != a40.WORKING_ID
        or payload.get("payload_sha256") != A40_PAYLOAD_SHA256
        or payload.get("status") != "stopped_batching_feasibility"
        or payload.get("gate", {}).get("failed_checks")
        != ["batch_sequential_equivalence", "material_native_speedup"]
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
    ):
        raise ValueError("A40 is not the exact stopped feasibility result")
    return payload


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace,
    *,
    base_source: Mapping[str, Any],
    a32_source: Mapping[str, Any],
    bundle_payload_sha256: str,
) -> dict[str, Any]:
    a40_payload = _verify_a40(args.a40_result)
    start, end, _ = a32.a31._verify_a28(args.a28_map)
    if start.shape != end.shape or start.shape != (len(FROZEN_ACTIVE_CELLS),) * 2:
        raise ValueError("A28 affine coefficient shape differs")
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a32_result_sha256": a40.A32_RESULT_SHA256,
                "a32_payload_sha256": a40.A32_PAYLOAD_SHA256,
                "a32_bundle_manifest_sha256": a40.A32_BUNDLE_MANIFEST_SHA256,
                "a32_bundle_manifest_payload_sha256": bundle_payload_sha256,
                "a40_result_sha256": A40_RESULT_SHA256,
                "a40_payload_sha256": a40_payload["payload_sha256"],
                "a28_result_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": a32.a31.A28_PAYLOAD_SHA256,
            },
            "base_runtime_source_manifest": dict(base_source),
            "a32_source_manifest": dict(a32_source),
            "population": {
                "calibration_cases": list(CALIBRATION_CASES),
                "zero_closure_cases": list(ZERO_CASES),
                "frozen_transfer_cases": list(TRANSFER_CASES),
                "input_calls": list(a40.BENCHMARK_OUTPUT_CALLS),
                "pair_count": a40.EXPECTED_PAIR_COUNT,
                "truth_arrays_indexed": False,
                "new_population_opened": False,
            },
            "response_contract": {
                "maps": list(MAP_NAMES),
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "zero_intercept": True,
                "denominator_floor": DENOMINATOR_FLOOR,
                "native_repeats": REPEATS,
                "active_fine_repeats": REPEATS,
                "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
                "fixed_bands": {key: list(value) for key, value in FIXED_BANDS.items()},
            },
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "map_order_exact": MAP_NAMES == ("zero", "identity", "scalar", "diagonal"),
        "case_inventory_exact": len(CALIBRATION_CASES) == 2
        and len(ZERO_CASES) == 1
        and len(TRANSFER_CASES) == 2,
        "pair_inventory_75": a40.EXPECTED_PAIR_COUNT == 75,
        "active_fine_inventory_32": len(ACTIVE_CASES) * 8 == 32,
        "active_cell_inventory_19": len(FROZEN_ACTIVE_CELLS) == 19,
        "owned_source_inventory_3": len(OWNED_SOURCE_PATHS) == 3,
        "cublas_workspace_config_frozen": REQUIRED_CUBLAS_WORKSPACE_CONFIG == ":4096:8",
    }
    return {
        "schema": "pcno_shadow_response_geometry_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, bundle_manifest, a32_source = a40._verify_a32_contract(
        args.a32_result, args.bundle_manifest
    )
    _, _, _, store, base_source = base._open_contract(a29._base_args(args))
    try:
        source = _source_manifest(
            args,
            base_source=base_source,
            a32_source=a32_source,
            bundle_payload_sha256=bundle_manifest["payload_sha256"],
        )
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "a32_base_runtime_exact": a32_source["base_runtime_source_manifest"]
            == base_source,
            "lineage_exact": source["lineage"]["a40_result_sha256"]
            == A40_RESULT_SHA256,
            "pair_inventory_exact": a40.EXPECTED_PAIR_COUNT == 75,
            "active_fine_inventory_exact": len(ACTIVE_CASES) * 8 == 32,
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest": source,
                "checkpoint_model_built": False,
                "bundle_state_arrays_loaded": False,
                "truth_arrays_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A41 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or not all(payload.get("checks", {}).values())
        or payload.get("checkpoint_model_built") is not False
        or payload.get("bundle_state_arrays_loaded") is not False
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("model_calls") != 0
        or payload.get("recurrence_executed") is not False
        or payload.get("source_manifest", {}).get("source_sha256") != _source_hashes()
    ):
        raise ValueError("A41 preflight differs from the frozen contract")
    return payload


def _active_coordinates(field: np.ndarray, runtime: Any) -> np.ndarray:
    coordinates = modal_coordinates(
        field,
        runtime.native_projector,
        component_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float64),
    )
    return np.asarray([coordinates[cell] for cell in FROZEN_ACTIVE_CELLS])


def _field_from_active(coordinates: np.ndarray, runtime: Any) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (len(FROZEN_ACTIVE_CELLS),) or not np.isfinite(values).all():
        raise ValueError("active response coordinates must be finite with length 19")
    modal = np.zeros((runtime.native_projector.q_matrix.shape[1], 4), dtype=np.float64)
    for cell, value in zip(FROZEN_ACTIVE_CELLS, values, strict=True):
        modal[cell] = value
    return reconstruct_modal_field(
        modal,
        runtime.native_projector,
        component_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float64),
    )


def fit_response_maps(x: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray | float]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(z, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape != targets.shape
        or features.shape[1] != len(FROZEN_ACTIVE_CELLS)
        or not np.isfinite(features).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("response fit arrays must be finite and shape aligned")
    scalar_denominator = float(np.sum(np.square(features)))
    diagonal_denominator = np.sum(np.square(features), axis=0)
    if scalar_denominator <= DENOMINATOR_FLOOR**2 or np.any(
        diagonal_denominator <= DENOMINATOR_FLOOR**2
    ):
        raise ValueError("response fit has a small denominator")
    return {
        "scalar": float(np.sum(features * targets) / scalar_denominator),
        "diagonal": np.sum(features * targets, axis=0) / diagonal_denominator,
    }


def predict_response(
    x: np.ndarray,
    map_name: str,
    coefficients: Mapping[str, np.ndarray | float],
) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    if features.shape != (len(FROZEN_ACTIVE_CELLS),) or not np.isfinite(features).all():
        raise ValueError("response feature must be finite with length 19")
    if map_name == "zero":
        return np.zeros_like(features)
    if map_name == "identity":
        return np.array(features, copy=True)
    if map_name == "scalar":
        return float(coefficients["scalar"]) * features
    if map_name == "diagonal":
        diagonal = np.asarray(coefficients["diagonal"], dtype=np.float64)
        if diagonal.shape != features.shape or not np.isfinite(diagonal).all():
            raise ValueError("diagonal response coefficients differ")
        return diagonal * features
    raise ValueError(f"unknown response map: {map_name}")


def _fine_on_native(
    runtime: Any, accepted: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    prepared = prepare_common_native_inputs(accepted, contract=a32.CONTRACT)
    fine_input = np.asarray(
        prepared.model_inputs[a32.FINE_RESOLUTION], dtype=np.float64
    )
    prediction, timing = predict_resolution_sample(
        runtime.model,
        runtime.sample_by_resolution[a32.FINE_RESOLUTION],
        fine_input,
        device=runtime.device,
        amp="none",
        repeats=REPEATS,
    )
    fine_increment = prediction - fine_input
    restricted = restrict_nested_state(
        fine_increment,
        fine_resolution=a32.FINE_RESOLUTION,
        coarse_resolution=a32.NATIVE_RESOLUTION,
    )
    return restricted, {
        "forward_seconds": float(sum(timing["forward_seconds"])),
        "peak_gpu_memory_bytes": int(timing["peak_gpu_memory_bytes"]),
        "repeat_max_abs": float(timing["repeat_max_abs"]),
        "pre_model_fine_to_native_max_abs": float(
            prepared.nesting_floors["pre_model_fine_to_native_max_abs"]
        ),
        "post_fp32_fine_to_native_max_abs": float(
            prepared.nesting_floors["post_fp32_fine_to_native_max_abs"]
        ),
    }


def _tether_offset(
    runtime: Any,
    *,
    accepted: np.ndarray,
    shadow_prediction: np.ndarray,
    accepted_prediction: np.ndarray,
    input_call: int,
    position_selected: bool,
    fine_on_native: np.ndarray | None,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    volumes = np.asarray(
        runtime.geometry_by_resolution[a32.NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    native_input = as_model_state(accepted)
    native_increment = accepted_prediction - native_input
    active = binary_position_phase_correction_active(
        position_selected=position_selected, input_call=input_call
    )
    if active:
        if fine_on_native is None:
            raise ValueError("active response row lacks a fine prediction")
        basis = NativeIncrementBasis(
            native_increment=native_increment,
            coarse_on_native=np.array(native_increment, copy=True),
            fine_on_native=np.asarray(fine_on_native, dtype=np.float64),
            native_minus_coarse=np.zeros_like(native_increment),
            fine_minus_native=np.asarray(fine_on_native, dtype=np.float64)
            - native_increment,
        )
        correction, correction_audit = affine_modal_correction(
            basis,
            runtime.native_projector,
            input_call=input_call,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
            volumes=volumes,
            component_scale=residual_scale,
        )
    else:
        correction = np.zeros_like(native_increment)
        correction_audit = None
    proposal = accepted_prediction + correction
    next_state, offset, tether_audit = projected_shadow_tether(
        proposal,
        shadow_prediction,
        projector=runtime.native_projector,
        volumes=volumes,
        component_scale=state_scale,
    )
    return offset, {
        "correction_active": active,
        "correction_to_native_increment": (
            correction_audit.correction_to_native_increment
            if correction_audit is not None
            else 0.0
        ),
        "cap_active": correction_audit.cap_active
        if correction_audit is not None
        else False,
        "maximum_integral_difference_abs": tether_audit.maximum_integral_difference_abs,
        "maximum_boundary_difference_abs": tether_audit.maximum_boundary_difference_abs,
        "maximum_projection_idempotence_abs": (
            tether_audit.maximum_projection_idempotence_abs
        ),
        "maximum_update_identity_abs": float(
            np.max(np.abs(next_state - shadow_prediction - offset))
        ),
    }


def _collect_records(
    runtime: Any,
    pairs: Sequence[Mapping[str, Any]],
    *,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    timing = {
        "native_logical_predictions": 0,
        "native_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_logical_predictions": 0,
        "fine_actual_forward_passes": 0,
        "fine_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "maximum_repeat_abs": 0.0,
        "maximum_pre_model_fine_to_native_abs": 0.0,
        "maximum_post_fp32_fine_to_native_abs": 0.0,
    }
    for index, pair in enumerate(pairs, start=1):
        case_id = str(pair["case_id"])
        input_call = int(pair["output_call"])
        print(f"A41 pair {index}/{len(pairs)}: {case_id} call {input_call}", flush=True)
        shadow = np.asarray(pair["shadow"], dtype=np.float64)
        accepted = np.asarray(pair["accepted"], dtype=np.float64)
        predictions = []
        for state in (shadow, accepted):
            prediction, call_timing = predict_resolution_sample(
                runtime.model,
                runtime.sample_by_resolution[a32.NATIVE_RESOLUTION],
                state,
                device=runtime.device,
                amp="none",
                repeats=REPEATS,
            )
            predictions.append(prediction)
            timing["native_logical_predictions"] += 1
            timing["native_actual_forward_passes"] += REPEATS
            timing["native_forward_seconds"] += float(
                sum(call_timing["forward_seconds"])
            )
            timing["maximum_peak_gpu_memory_bytes"] = max(
                timing["maximum_peak_gpu_memory_bytes"],
                int(call_timing["peak_gpu_memory_bytes"]),
            )
            timing["maximum_repeat_abs"] = max(
                timing["maximum_repeat_abs"], float(call_timing["repeat_max_abs"])
            )
        shadow_prediction, accepted_prediction = predictions
        position_selected = case_id in ACTIVE_CASES
        active = binary_position_phase_correction_active(
            position_selected=position_selected, input_call=input_call
        )
        if active:
            fine_increment, fine_timing = _fine_on_native(runtime, accepted)
            timing["fine_logical_predictions"] += 1
            timing["fine_actual_forward_passes"] += REPEATS
            timing["fine_forward_seconds"] += float(fine_timing["forward_seconds"])
            timing["maximum_peak_gpu_memory_bytes"] = max(
                timing["maximum_peak_gpu_memory_bytes"],
                int(fine_timing["peak_gpu_memory_bytes"]),
            )
            timing["maximum_repeat_abs"] = max(
                timing["maximum_repeat_abs"], float(fine_timing["repeat_max_abs"])
            )
            timing["maximum_pre_model_fine_to_native_abs"] = max(
                timing["maximum_pre_model_fine_to_native_abs"],
                float(fine_timing["pre_model_fine_to_native_max_abs"]),
            )
            timing["maximum_post_fp32_fine_to_native_abs"] = max(
                timing["maximum_post_fp32_fine_to_native_abs"],
                float(fine_timing["post_fp32_fine_to_native_max_abs"]),
            )
        else:
            fine_increment = None
        exact_offset, exact_audit = _tether_offset(
            runtime,
            accepted=accepted,
            shadow_prediction=shadow_prediction,
            accepted_prediction=accepted_prediction,
            input_call=input_call,
            position_selected=position_selected,
            fine_on_native=fine_increment,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
        )
        displacement = accepted - shadow
        output_displacement = accepted_prediction - shadow_prediction
        split, closure = runtime.native_projector.split(
            output_displacement - displacement
        )
        records.append(
            {
                "case_id": case_id,
                "input_call": input_call,
                "band": next(
                    name for name, calls in FIXED_BANDS.items() if input_call in calls
                ),
                "group": "e12" if case_id.startswith("sv_e12_") else "e14",
                "active_case": position_selected,
                "correction_active": active,
                "shadow": shadow,
                "accepted": accepted,
                "shadow_prediction": shadow_prediction,
                "fine_on_native": fine_increment,
                "x": _active_coordinates(displacement, runtime),
                "z": _active_coordinates(output_displacement, runtime),
                "exact_offset": exact_offset,
                "exact_audit": exact_audit,
                "input_displacement_rms": weighted_scaled_rms(
                    displacement,
                    volumes=np.asarray(
                        runtime.geometry_by_resolution[
                            a32.NATIVE_RESOLUTION
                        ].node_measures,
                        dtype=np.float64,
                    ),
                    component_scale=np.asarray(
                        runtime.normalization.state_scale, dtype=np.float64
                    ),
                ),
                "residual_response_full_rms": weighted_scaled_rms(
                    output_displacement - displacement,
                    volumes=np.asarray(
                        runtime.geometry_by_resolution[
                            a32.NATIVE_RESOLUTION
                        ].node_measures,
                        dtype=np.float64,
                    ),
                    component_scale=np.asarray(
                        runtime.normalization.state_scale, dtype=np.float64
                    ),
                ),
                "residual_response_parallel_rms": weighted_scaled_rms(
                    split["parallel"],
                    volumes=np.asarray(
                        runtime.geometry_by_resolution[
                            a32.NATIVE_RESOLUTION
                        ].node_measures,
                        dtype=np.float64,
                    ),
                    component_scale=np.asarray(
                        runtime.normalization.state_scale, dtype=np.float64
                    ),
                ),
                "residual_response_orthogonal_rms": weighted_scaled_rms(
                    split["orthogonal"],
                    volumes=np.asarray(
                        runtime.geometry_by_resolution[
                            a32.NATIVE_RESOLUTION
                        ].node_measures,
                        dtype=np.float64,
                    ),
                    component_scale=np.asarray(
                        runtime.normalization.state_scale, dtype=np.float64
                    ),
                ),
                "residual_response_excluded_rms": weighted_scaled_rms(
                    split["excluded"],
                    volumes=np.asarray(
                        runtime.geometry_by_resolution[
                            a32.NATIVE_RESOLUTION
                        ].node_measures,
                        dtype=np.float64,
                    ),
                    component_scale=np.asarray(
                        runtime.normalization.state_scale, dtype=np.float64
                    ),
                ),
                "split_reconstruction_max_abs": closure["maximum_reconstruction_abs"],
            }
        )
    return records, timing


def _score_map(
    runtime: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    map_name: str,
    coefficients: Mapping[str, np.ndarray | float],
    coefficient_source: str,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    volumes = np.asarray(
        runtime.geometry_by_resolution[a32.NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    rows = []
    for record in records:
        prediction = predict_response(record["x"], map_name, coefficients)
        predicted_field = _field_from_active(prediction, runtime)
        accepted_prediction = record["shadow_prediction"] + predicted_field
        offset, audit = _tether_offset(
            runtime,
            accepted=record["accepted"],
            shadow_prediction=record["shadow_prediction"],
            accepted_prediction=accepted_prediction,
            input_call=int(record["input_call"]),
            position_selected=bool(record["active_case"]),
            fine_on_native=record["fine_on_native"],
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
        )
        response_error = prediction - record["z"]
        update_error = offset - record["exact_offset"]
        exact_offset_rms = weighted_scaled_rms(
            record["exact_offset"], volumes=volumes, component_scale=state_scale
        )
        update_error_rms = weighted_scaled_rms(
            update_error, volumes=volumes, component_scale=state_scale
        )
        rows.append(
            {
                "map": map_name,
                "coefficient_source": coefficient_source,
                "case_id": record["case_id"],
                "group": record["group"],
                "input_call": record["input_call"],
                "band": record["band"],
                "active_case": record["active_case"],
                "correction_active": record["correction_active"],
                "response_error_sse": float(np.sum(np.square(response_error))),
                "response_zero_sse": float(np.sum(np.square(record["z"]))),
                "response_dot": float(prediction @ record["z"]),
                "response_prediction_norm2": float(prediction @ prediction),
                "response_target_norm2": float(record["z"] @ record["z"]),
                "update_error_sse": update_error_rms**2,
                "update_zero_sse": None,
                "exact_offset_rms": exact_offset_rms,
                "update_error_rms": update_error_rms,
                "update_error_to_exact_offset": (
                    update_error_rms / exact_offset_rms
                    if exact_offset_rms > DENOMINATOR_FLOOR
                    else None
                ),
                "update_error_to_exact_offset_status": (
                    "ok"
                    if exact_offset_rms > DENOMINATOR_FLOOR
                    else "small_denominator"
                ),
                "cap_active": audit["cap_active"],
                "exact_cap_active": record["exact_audit"]["cap_active"],
                "cap_agreement": audit["cap_active"]
                == record["exact_audit"]["cap_active"],
                "correction_to_native_increment": audit[
                    "correction_to_native_increment"
                ],
                "exact_correction_to_native_increment": record["exact_audit"][
                    "correction_to_native_increment"
                ],
                "maximum_integral_difference_abs": audit[
                    "maximum_integral_difference_abs"
                ],
                "maximum_boundary_difference_abs": audit[
                    "maximum_boundary_difference_abs"
                ],
                "maximum_projection_idempotence_abs": audit[
                    "maximum_projection_idempotence_abs"
                ],
                "maximum_update_identity_abs": audit["maximum_update_identity_abs"],
            }
        )
    return rows


def _attach_update_baseline(rows: list[dict[str, Any]]) -> None:
    zero = {
        (row["case_id"], row["input_call"]): row
        for row in rows
        if row["map"] == "zero" and row["coefficient_source"] == "e12_final"
    }
    for row in rows:
        key = (row["case_id"], row["input_call"])
        baseline = zero.get(key)
        if baseline is None:
            raise ValueError("response score lacks its zero-map update baseline")
        row["update_zero_sse"] = float(baseline["update_error_sse"])


def scope_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
) -> dict[str, Any]:
    selected = [row for row in rows if row["case_id"] in case_ids]
    if not selected or {row["case_id"] for row in selected} != set(case_ids):
        raise ValueError("scope rows do not contain the requested cases")
    case_values = []
    for case_id in case_ids:
        case_rows = [row for row in selected if row["case_id"] == case_id]
        response_error = float(
            np.mean([row["response_error_sse"] for row in case_rows])
        )
        response_zero = float(np.mean([row["response_zero_sse"] for row in case_rows]))
        update_error = float(np.mean([row["update_error_sse"] for row in case_rows]))
        update_zero = float(np.mean([row["update_zero_sse"] for row in case_rows]))
        dot = float(sum(row["response_dot"] for row in case_rows))
        prediction_norm2 = float(
            sum(row["response_prediction_norm2"] for row in case_rows)
        )
        target_norm2 = float(sum(row["response_target_norm2"] for row in case_rows))
        prediction_norm = float(np.sqrt(prediction_norm2))
        target_norm = float(np.sqrt(target_norm2))
        denominator = prediction_norm * target_norm
        response_resolved = response_zero > DENOMINATOR_FLOOR**2
        update_resolved = update_zero > DENOMINATOR_FLOOR**2
        cosine_resolved = (
            prediction_norm > DENOMINATOR_FLOOR and target_norm > DENOMINATOR_FLOOR
        )
        norm_ratio_resolved = target_norm > DENOMINATOR_FLOOR
        case_values.append(
            {
                "case_id": case_id,
                "response_error": response_error,
                "response_zero": response_zero,
                "update_error": update_error,
                "update_zero": update_zero,
                "response_skill": (
                    1.0 - response_error / response_zero if response_resolved else None
                ),
                "response_skill_status": (
                    "ok" if response_resolved else "small_denominator"
                ),
                "response_relative_rms": (
                    float(np.sqrt(response_error / response_zero))
                    if response_resolved
                    else None
                ),
                "update_skill": (
                    1.0 - update_error / update_zero if update_resolved else None
                ),
                "update_skill_status": (
                    "ok" if update_resolved else "small_denominator"
                ),
                "update_relative_rms": (
                    float(np.sqrt(update_error / update_zero))
                    if update_resolved
                    else None
                ),
                "cosine": (dot / denominator if cosine_resolved else None),
                "cosine_status": ("ok" if cosine_resolved else "small_denominator"),
                "norm_ratio": (
                    prediction_norm / target_norm if norm_ratio_resolved else None
                ),
                "norm_ratio_status": (
                    "ok" if norm_ratio_resolved else "small_denominator"
                ),
            }
        )
    response_zero = float(np.mean([row["response_zero"] for row in case_values]))
    update_zero = float(np.mean([row["update_zero"] for row in case_values]))
    cosines = [row["cosine"] for row in case_values]
    ratios = [row["norm_ratio"] for row in case_values]
    response_error = float(np.mean([row["response_error"] for row in case_values]))
    update_error = float(np.mean([row["update_error"] for row in case_values]))
    response_resolved = response_zero > DENOMINATOR_FLOOR**2
    update_resolved = update_zero > DENOMINATOR_FLOOR**2
    cosine_resolved = all(value is not None for value in cosines)
    norm_ratio_resolved = all(value is not None for value in ratios)
    return {
        "response_skill": (
            1.0 - response_error / response_zero if response_resolved else None
        ),
        "response_skill_status": ("ok" if response_resolved else "small_denominator"),
        "response_relative_rms": (
            float(np.sqrt(response_error / response_zero))
            if response_resolved
            else None
        ),
        "update_skill": (1.0 - update_error / update_zero if update_resolved else None),
        "update_skill_status": ("ok" if update_resolved else "small_denominator"),
        "update_relative_rms": (
            float(np.sqrt(update_error / update_zero)) if update_resolved else None
        ),
        "median_response_cosine": (
            float(np.median(cosines)) if cosine_resolved else None
        ),
        "median_response_cosine_status": (
            "ok" if cosine_resolved else "small_denominator"
        ),
        "median_response_norm_ratio": (
            float(np.median(ratios)) if norm_ratio_resolved else None
        ),
        "median_response_norm_ratio_status": (
            "ok" if norm_ratio_resolved else "small_denominator"
        ),
        "case_metrics": case_values,
    }


def _score_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = {(row["map"], row["coefficient_source"]) for row in rows}
    for map_name, coefficient_source in sorted(keys):
        subset = [
            row
            for row in rows
            if row["map"] == map_name
            and row["coefficient_source"] == coefficient_source
        ]
        for scope, case_ids in (
            ("e12", CALIBRATION_CASES),
            ("e14", TRANSFER_CASES),
        ):
            available = [
                case_id
                for case_id in case_ids
                if any(row["case_id"] == case_id for row in subset)
            ]
            if available:
                output.append(
                    {
                        "map": map_name,
                        "coefficient_source": coefficient_source,
                        "scope": scope,
                        "band": "overall",
                        **scope_metrics(subset, case_ids=available),
                    }
                )
                for band in FIXED_BANDS:
                    band_rows = [row for row in subset if row["band"] == band]
                    output.append(
                        {
                            "map": map_name,
                            "coefficient_source": coefficient_source,
                            "scope": scope,
                            "band": band,
                            **scope_metrics(band_rows, case_ids=available),
                        }
                    )
    return output


def _coefficient_stability(folds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scalar = [float(folds[case]["scalar"]) for case in CALIBRATION_CASES]
    diagonal = [
        np.asarray(folds[case]["diagonal"], dtype=np.float64)
        for case in CALIBRATION_CASES
    ]
    scalar_minimum = min(abs(value) for value in scalar)
    diagonal_norms = [float(np.linalg.norm(value)) for value in diagonal]
    diagonal_denominator = diagonal_norms[0] * diagonal_norms[1]
    return {
        "scalar_values": scalar,
        "scalar_all_positive": all(value > 0.0 for value in scalar),
        "scalar_magnitude_ratio": (
            max(abs(value) for value in scalar) / scalar_minimum
            if scalar_minimum > DENOMINATOR_FLOOR
            else None
        ),
        "scalar_stability_status": (
            "ok" if scalar_minimum > DENOMINATOR_FLOOR else "small_denominator"
        ),
        "diagonal_cosine": (
            float(diagonal[0] @ diagonal[1] / diagonal_denominator)
            if diagonal_denominator > DENOMINATOR_FLOOR**2
            else None
        ),
        "diagonal_norm_ratio": (
            max(diagonal_norms) / min(diagonal_norms)
            if min(diagonal_norms) > DENOMINATOR_FLOOR
            else None
        ),
        "diagonal_stability_status": (
            "ok" if min(diagonal_norms) > DENOMINATOR_FLOOR else "small_denominator"
        ),
        "diagonal_all_positive": all(bool(np.all(value > 0.0)) for value in diagonal),
        "diagonal_values": [value.tolist() for value in diagonal],
    }


def _candidate_gate(
    map_name: str,
    score_rows: Sequence[Mapping[str, Any]],
    row_scores: Sequence[Mapping[str, Any]],
    *,
    stability: Mapping[str, Any],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    final = [
        row
        for row in score_rows
        if row["map"] == map_name and row["coefficient_source"] == "e12_final"
    ]
    cross = [
        row
        for row in score_rows
        if row["map"] == map_name and row["coefficient_source"].startswith("fit_")
    ]

    def strong(row: Mapping[str, Any], threshold: float) -> bool:
        return (
            row["response_skill_status"] == "ok"
            and row["response_skill"] is not None
            and float(row["response_skill"]) > threshold
            and row["update_skill_status"] == "ok"
            and row["update_skill"] is not None
            and float(row["update_skill"]) > threshold
        )

    overall = [row for row in final if row["band"] == "overall"]
    bands = [row for row in final if row["band"] != "overall"]
    per_case_ok = all(
        case[f"{metric}_status"] == "ok"
        and case[metric] is not None
        and float(case[metric]) > 0.99
        for row in overall
        for case in row["case_metrics"]
        for metric in ("response_skill", "update_skill", "cosine")
    )
    per_case_band_ok = all(
        case[f"{metric}_status"] == "ok"
        and case[metric] is not None
        and float(case[metric]) > 0.95
        for row in bands
        for case in row["case_metrics"]
        for metric in ("response_skill", "update_skill")
    )
    row_subset = [
        row
        for row in row_scores
        if row["map"] == map_name
        and row["coefficient_source"] == "e12_final"
        and row["active_case"]
    ]
    checks = {
        **structural_checks,
        "overall_response_and_update_skill": len(overall) == 2
        and all(strong(row, 0.99) for row in overall),
        "band_response_and_update_skill": len(bands) == 8
        and all(strong(row, 0.95) for row in bands)
        and per_case_band_ok,
        "case_and_scope_response_cosine": per_case_ok
        and all(
            row["median_response_cosine_status"] == "ok"
            and row["median_response_cosine"] is not None
            and float(row["median_response_cosine"]) > 0.99
            for row in overall
        ),
        "scope_response_norm_ratio": all(
            row["median_response_norm_ratio_status"] == "ok"
            and row["median_response_norm_ratio"] is not None
            and 0.90 < float(row["median_response_norm_ratio"]) < 1.10
            for row in overall
        ),
        "per_row_tether_error_ratio": bool(row_subset)
        and all(
            (
                row["update_error_to_exact_offset_status"] == "ok"
                and row["update_error_to_exact_offset"] is not None
                and float(row["update_error_to_exact_offset"]) < 0.50
            )
            or (
                row["update_error_to_exact_offset_status"] == "small_denominator"
                and float(row["exact_offset_rms"]) <= DENOMINATOR_FLOOR
                and float(row["update_error_rms"]) <= DENOMINATOR_FLOOR
            )
            for row in row_subset
        ),
        "cap_agreement_and_bound": bool(row_subset)
        and all(
            bool(row["cap_agreement"])
            and (
                row["correction_to_native_increment"] is None
                or float(row["correction_to_native_increment"])
                <= 0.05 + STRICT_TOLERANCE
            )
            and (
                row["exact_correction_to_native_increment"] is None
                or float(row["exact_correction_to_native_increment"])
                <= 0.05 + STRICT_TOLERANCE
            )
            for row in row_subset
        ),
        "cross_case_response_and_update_skill": (
            True
            if map_name == "identity"
            else len(cross) == 10
            and all(
                strong(row, 0.99 if row["band"] == "overall" else 0.95) for row in cross
            )
        ),
        "coefficient_stability": (
            True
            if map_name == "identity"
            else (
                bool(stability["scalar_all_positive"])
                and stability["scalar_stability_status"] == "ok"
                and stability["scalar_magnitude_ratio"] is not None
                and float(stability["scalar_magnitude_ratio"]) < 1.5
                if map_name == "scalar"
                else bool(stability["diagonal_all_positive"])
                and stability["diagonal_stability_status"] == "ok"
                and stability["diagonal_cosine"] is not None
                and float(stability["diagonal_cosine"]) > 0.90
                and stability["diagonal_norm_ratio"] is not None
                and float(stability["diagonal_norm_ratio"]) < 1.5
            )
        ),
    }
    return {
        "status": "qualified" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"A41 requires CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
        )
    preflight = _verify_preflight(args.preflight)
    _, manifest, _ = a40._verify_a32_contract(args.a32_result, args.bundle_manifest)
    _verify_a40(args.a40_result)
    start, end, _ = a32.a31._verify_a28(args.a28_map)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = base._build_runtime(a29._base_args(args))
    started = perf_counter()
    try:
        if (
            runtime.source_manifest
            != preflight["source_manifest"]["base_runtime_source_manifest"]
        ):
            raise ValueError("A41 runtime source differs from preflight")
        pairs = a40._load_state_pairs(args.bundle_manifest, manifest)
        records, execution = _collect_records(
            runtime,
            pairs,
            start_coefficients=start,
            end_coefficients=end,
        )
        calibration = [
            record for record in records if record["case_id"] in CALIBRATION_CASES
        ]
        final_coefficients = fit_response_maps(
            np.stack([record["x"] for record in calibration]),
            np.stack([record["z"] for record in calibration]),
        )
        fold_coefficients = {
            held: fit_response_maps(
                np.stack(
                    [record["x"] for record in calibration if record["case_id"] != held]
                ),
                np.stack(
                    [record["z"] for record in calibration if record["case_id"] != held]
                ),
            )
            for held in CALIBRATION_CASES
        }
        stability = _coefficient_stability(fold_coefficients)
        scored_rows: list[dict[str, Any]] = []
        for map_name in MAP_NAMES:
            scored_rows.extend(
                _score_map(
                    runtime,
                    records,
                    map_name=map_name,
                    coefficients=final_coefficients,
                    coefficient_source="e12_final",
                    start_coefficients=start,
                    end_coefficients=end,
                )
            )
            if map_name in {"scalar", "diagonal"}:
                for held in CALIBRATION_CASES:
                    scored_rows.extend(
                        _score_map(
                            runtime,
                            [record for record in records if record["case_id"] == held],
                            map_name=map_name,
                            coefficients=fold_coefficients[held],
                            coefficient_source=f"fit_other_hold_{held}",
                            start_coefficients=start,
                            end_coefficients=end,
                        )
                    )
        _attach_update_baseline(scored_rows)
        score_rows = _score_inventory(scored_rows)
        zero_records = [record for record in records if record["case_id"] in ZERO_CASES]
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "pair_inventory_exact": len(records) == a40.EXPECTED_PAIR_COUNT,
            "native_call_inventory_exact": execution["native_logical_predictions"]
            == 150,
            "fine_call_inventory_exact": execution["fine_logical_predictions"] == 32,
            "repeatability": execution["maximum_repeat_abs"] <= REPEAT_TOLERANCE,
            "finite_state_and_response": all(
                np.isfinite(np.asarray(record[key])).all()
                for record in records
                for key in (
                    "shadow",
                    "accepted",
                    "shadow_prediction",
                    "x",
                    "z",
                    "exact_offset",
                )
            )
            and all(
                record["fine_on_native"] is None
                or np.isfinite(np.asarray(record["fine_on_native"])).all()
                for record in records
            )
            and all(
                np.isfinite(float(row[key]))
                for row in scored_rows
                for key in (
                    "response_error_sse",
                    "response_zero_sse",
                    "update_error_sse",
                    "exact_offset_rms",
                    "update_error_rms",
                )
            ),
            "common_source_fine_floor": execution[
                "maximum_pre_model_fine_to_native_abs"
            ]
            <= STRICT_TOLERANCE
            and execution["maximum_post_fp32_fine_to_native_abs"] <= REPEAT_TOLERANCE,
            "zero_rows_exact": all(
                np.max(np.abs(record["x"])) <= REPEAT_TOLERANCE
                and np.max(np.abs(record["z"])) <= REPEAT_TOLERANCE
                for record in zero_records
            ),
            "split_closure": max(
                record["split_reconstruction_max_abs"] for record in records
            )
            <= STRICT_TOLERANCE,
            "tether_closure": max(
                max(
                    float(row["maximum_integral_difference_abs"]),
                    float(row["maximum_boundary_difference_abs"]),
                    float(row["maximum_projection_idempotence_abs"]),
                    float(row["maximum_update_identity_abs"]),
                )
                for row in scored_rows
            )
            <= STRICT_TOLERANCE,
            "truth_arrays_not_loaded": True,
            "recurrence_not_executed": True,
            "fine_truth_not_loaded": True,
        }
        gates = {
            map_name: _candidate_gate(
                map_name,
                score_rows,
                scored_rows,
                stability=stability,
                structural_checks=structural_checks,
            )
            for map_name in ("identity", "scalar", "diagonal")
        }
        selected = next(
            (
                map_name
                for map_name in ("identity", "scalar", "diagonal")
                if gates[map_name]["status"] == "qualified"
            ),
            None,
        )
        execution.update(
            {
                "wall_seconds": perf_counter() - started,
                "environment": runtime_environment(runtime.device),
            }
        )
        write_csv(
            output_dir / "shadow_response_records.csv",
            [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "shadow",
                        "accepted",
                        "shadow_prediction",
                        "fine_on_native",
                        "x",
                        "z",
                        "exact_offset",
                        "exact_audit",
                    }
                }
                for record in records
            ],
        )
        write_csv(
            output_dir / "shadow_response_modal_coordinates.csv",
            [
                {
                    "case_id": record["case_id"],
                    "input_call": record["input_call"],
                    "mode_index": cell[0],
                    "component": cell[1],
                    "input_displacement_coordinate": float(record["x"][index]),
                    "output_displacement_coordinate": float(record["z"][index]),
                }
                for record in records
                for index, cell in enumerate(FROZEN_ACTIVE_CELLS)
            ],
        )
        write_csv(output_dir / "shadow_response_map_rows.csv", scored_rows)
        write_csv(
            output_dir / "shadow_response_scores.csv",
            [
                {key: value for key, value in row.items() if key != "case_metrics"}
                for row in score_rows
            ],
        )
        atomic_write_json(
            output_dir / "source_manifest.json", preflight["source_manifest"]
        )
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        result = with_payload_sha256(
            {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "qualified_response_geometry"
                    if selected is not None
                    else "stopped_response_geometry"
                ),
                "selected_map": selected,
                "gates": gates,
                "structural_checks": structural_checks,
                "final_coefficients": {
                    "scalar": final_coefficients["scalar"],
                    "diagonal": np.asarray(final_coefficients["diagonal"]).tolist(),
                },
                "coefficient_stability": stability,
                "scores": score_rows,
                "execution": execution,
                "mechanism": {
                    "maximum_residual_response_full_rms": max(
                        record["residual_response_full_rms"] for record in records
                    ),
                    "maximum_residual_response_parallel_rms": max(
                        record["residual_response_parallel_rms"] for record in records
                    ),
                    "maximum_residual_response_orthogonal_rms": max(
                        record["residual_response_orthogonal_rms"] for record in records
                    ),
                    "maximum_residual_response_excluded_rms": max(
                        record["residual_response_excluded_rms"] for record in records
                    ),
                },
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": preflight["source_manifest"][
                    "payload_sha256"
                ],
                "artifact_inventory_before_summary": artifacts,
                "bundle_state_arrays_loaded": True,
                "truth_arrays_loaded": False,
                "fine_truth_loaded": False,
                "recurrence_executed": False,
                "accuracy_scored": False,
                "new_population_opened": False,
                "claim_boundary": (
                    "Target-free model-response geometry on saved float32 A32 states only; "
                    "no PDE accuracy, independent-data, family-transfer, conservation, "
                    "convergence, direct-off-grid, or Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "shadow_response_geometry.json", result)
        return result, 0 if selected is not None else 4
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-readiness", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--d063-run-contract", type=Path, required=True)
    parser.add_argument("--d063-summary", type=Path, required=True)
    parser.add_argument("--a28-map", type=Path, required=True)
    parser.add_argument("--a32-result", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--a40-result", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    _add_external_arguments(analyze)
    analyze.add_argument("--preflight", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload, exit_code = run_analysis(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
