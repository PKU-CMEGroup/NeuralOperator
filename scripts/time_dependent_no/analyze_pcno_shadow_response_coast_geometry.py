"""Test the frozen A41 response map on every correction-inactive A32 call."""

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

from scripts.time_dependent_no import (
    analyze_pcno_shadow_response_geometry as a41,
)
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
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    as_model_state,
    conservative_admissibility_summary,
    predict_resolution_sample,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    binary_position_phase_correction_active,
    synchronized_shadow_tethered_binary_affine_step,
)

WORKING_ID = "W26-L5-P6-RFB19-A42-COAST-RESPONSE-GEOMETRY"
PREFLIGHT_SCHEMA = "pcno_shadow_response_coast_geometry_preflight_v1"
RESULT_SCHEMA = "pcno_shadow_response_coast_geometry_v1"
SOURCE_SCHEMA = "pcno_shadow_response_coast_geometry_source_v1"
A41_RESULT_SHA256 = "7e041fd8264810a83f4470241e7a087d3de316839610a636847f6d6010916aae"
A41_PAYLOAD_SHA256 = "d60df92750aac4580ade724964fe42066a70fe710bd9ec3dd174ad0567f78daa"
FROZEN_DIAGONAL = (
    0.944340727071158,
    0.9121302905235246,
    0.9111307821869262,
    0.915623610609988,
    0.7860447848257047,
    0.9164140318142833,
    0.9376252623913839,
    0.9303139276790172,
    0.9211448625310777,
    0.9700725688489519,
    0.9017901650293275,
    1.0228484831912932,
    0.8422580299229786,
    0.8751168452189039,
    0.8445208751369352,
    0.9010140381730256,
    0.909978988502174,
    0.8021087571824961,
    0.8153880928004269,
)
CASE_IDS = a41.ACTIVE_CASES
SEGMENT_START_CALLS = tuple(range(8, 22, 2))
COAST_CALLS = tuple(range(8, 22))
COAST_BANDS = {
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
}
EXPECTED_SEGMENTS = len(CASE_IDS) * len(SEGMENT_START_CALLS)
EXPECTED_ROWS = len(CASE_IDS) * len(COAST_CALLS)
NATIVE_LOGICAL_PREDICTIONS = EXPECTED_ROWS * 2
REPEATS = 2
REPEAT_TOLERANCE = 1.0e-6
STRICT_TOLERANCE = 1.0e-12
SERIALIZATION_TOLERANCE = 1.0e-6
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/analyze_pcno_shadow_response_coast_geometry.py",
    "tests/time_dependent_no/test_pcno_shadow_response_coast_geometry.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/analyze_pcno_shadow_response_geometry.py",
    "scripts/time_dependent_no/benchmark_pcno_paired_native_batch.py",
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _coast_band(input_call: int) -> str:
    matches = [name for name, calls in COAST_BANDS.items() if input_call in calls]
    if len(matches) != 1:
        raise ValueError("input call is outside the registered A42 coast")
    return matches[0]


def _verify_a41(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A41_RESULT_SHA256:
        raise ValueError("A41 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    diagonal = np.asarray(
        payload.get("final_coefficients", {}).get("diagonal"), dtype=np.float64
    )
    if (
        payload.get("schema") != a41.RESULT_SCHEMA
        or payload.get("working_id") != a41.WORKING_ID
        or payload.get("payload_sha256") != A41_PAYLOAD_SHA256
        or payload.get("status") != "stopped_response_geometry"
        or payload.get("selected_map") is not None
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("fine_truth_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or not all(payload.get("structural_checks", {}).values())
        or any(
            gate.get("status") != "failed" for gate in payload.get("gates", {}).values()
        )
        or not np.array_equal(diagonal, np.asarray(FROZEN_DIAGONAL))
    ):
        raise ValueError("A41 is not the exact stopped response result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A41 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A41 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A41 artifact differs: {relative}")
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
    a41_payload = _verify_a41(args.a41_result)
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a41_result_sha256": A41_RESULT_SHA256,
                "a41_payload_sha256": a41_payload["payload_sha256"],
                "a32_result_sha256": a40.A32_RESULT_SHA256,
                "a32_payload_sha256": a40.A32_PAYLOAD_SHA256,
                "a32_bundle_manifest_sha256": a40.A32_BUNDLE_MANIFEST_SHA256,
                "a32_bundle_manifest_payload_sha256": bundle_payload_sha256,
                "a28_result_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": a32.a31.A28_PAYLOAD_SHA256,
            },
            "base_runtime_source_manifest": dict(base_source),
            "a32_source_manifest": dict(a32_source),
            "coast_contract": {
                "case_ids": list(CASE_IDS),
                "segment_start_calls": list(SEGMENT_START_CALLS),
                "input_calls": list(COAST_CALLS),
                "bands": {name: list(calls) for name, calls in COAST_BANDS.items()},
                "segments": EXPECTED_SEGMENTS,
                "rows": EXPECTED_ROWS,
                "native_logical_predictions": NATIVE_LOGICAL_PREDICTIONS,
                "native_actual_forward_passes": NATIVE_LOGICAL_PREDICTIONS * REPEATS,
                "fine_predictions": 0,
                "repeats": REPEATS,
                "amp": "none",
                "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
                "frozen_diagonal": list(FROZEN_DIAGONAL),
                "truth_array_indexed": False,
                "surrogate_recurrence": False,
                "h30_rollout": False,
            },
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "case_inventory_4": len(CASE_IDS) == 4,
        "segment_starts_exact": SEGMENT_START_CALLS == tuple(range(8, 22, 2)),
        "coast_calls_exact": COAST_CALLS == tuple(range(8, 22)),
        "segment_inventory_28": EXPECTED_SEGMENTS == 28,
        "row_inventory_56": EXPECTED_ROWS == 56,
        "native_logical_inventory_112": NATIVE_LOGICAL_PREDICTIONS == 112,
        "diagonal_inventory_19": len(FROZEN_DIAGONAL) == 19,
        "diagonal_positive": all(value > 0.0 for value in FROZEN_DIAGONAL),
        "owned_source_inventory_3": len(OWNED_SOURCE_PATHS) == 3,
    }
    return {
        "schema": "pcno_shadow_response_coast_geometry_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _verify_a41(args.a41_result)
    _, manifest, a32_source = a40._verify_a32_contract(
        args.a32_result, args.bundle_manifest
    )
    a32.a31._verify_a28(args.a28_map)
    _, _, _, store, base_source = base._open_contract(a29._base_args(args))
    try:
        source = _source_manifest(
            args,
            base_source=base_source,
            a32_source=a32_source,
            bundle_payload_sha256=manifest["payload_sha256"],
        )
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "a32_base_runtime_exact": a32_source["base_runtime_source_manifest"]
            == base_source,
            "a41_lineage_exact": source["lineage"]["a41_result_sha256"]
            == A41_RESULT_SHA256
            and source["lineage"]["a41_payload_sha256"] == A41_PAYLOAD_SHA256,
            "bundle_inventory_exact": tuple(manifest["contract"]["case_ids"])
            == a32.ANIMATION_CASE_IDS
            and tuple(manifest["contract"]["output_calls"]) == a32.ANIMATION_CALLS,
            "coast_inventory_exact": EXPECTED_SEGMENTS == 28
            and EXPECTED_ROWS == 56
            and NATIVE_LOGICAL_PREDICTIONS == 112,
            "coefficient_identity_exact": source["coast_contract"]["frozen_diagonal"]
            == list(FROZEN_DIAGONAL),
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
                "local_exact_segments_executed": False,
                "surrogate_recurrence_executed": False,
                "h30_rollout_executed": False,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A42 preflight checks did not all pass")
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
        or payload.get("local_exact_segments_executed") is not False
        or payload.get("surrogate_recurrence_executed") is not False
        or payload.get("h30_rollout_executed") is not False
        or payload.get("source_manifest", {}).get("source_sha256") != _source_hashes()
    ):
        raise ValueError("A42 preflight differs from the frozen contract")
    return payload


def _load_segments(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for row in manifest["bundles"]:
        if row["case_id"] not in CASE_IDS:
            continue
        path = (manifest_path.parent / str(row["path"])).resolve()
        if (
            path.parent != manifest_path.parent.resolve()
            or sha256_file(path) != row["sha256"]
            or row.get("storage_dtype") != "float32_visualization_only"
        ):
            raise ValueError("A32 bundle path, hash, or storage dtype differs")
        with np.load(path, allow_pickle=False) as bundle:
            case_id, output_calls, raw, corrected = a40._state_arrays_from_bundle(
                bundle
            )
        if case_id != row["case_id"]:
            raise ValueError("A32 bundle case ID differs")
        indices = {int(call): index for index, call in enumerate(output_calls)}
        for start in SEGMENT_START_CALLS:
            if start not in indices or start + 2 not in indices:
                raise ValueError("A42 bundle lacks a registered coast segment")
            first = indices[start]
            last = indices[start + 2]
            segments.append(
                {
                    "case_id": case_id,
                    "start_call": start,
                    "shadow": np.array(raw[first], copy=True),
                    "accepted": np.array(corrected[first], copy=True),
                    "expected_shadow_end": np.array(raw[last], copy=True),
                    "expected_accepted_end": np.array(corrected[last], copy=True),
                }
            )
    if (
        len(segments) != EXPECTED_SEGMENTS
        or tuple(dict.fromkeys(row["case_id"] for row in segments)) != CASE_IDS
        or {(row["case_id"], int(row["start_call"])) for row in segments}
        != {(case_id, call) for case_id in CASE_IDS for call in SEGMENT_START_CALLS}
    ):
        raise ValueError("A42 segment inventory differs")
    return segments


def _new_execution() -> dict[str, Any]:
    return {
        "segments": 0,
        "response_rows": 0,
        "native_logical_predictions": 0,
        "native_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_logical_predictions": 0,
        "fine_actual_forward_passes": 0,
        "fine_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "maximum_repeat_abs": 0.0,
        "maximum_exact_tether_reconstruction_abs": 0.0,
        "maximum_shadow_end_serialization_abs": 0.0,
        "maximum_accepted_end_serialization_abs": 0.0,
    }


def _collect_records(
    runtime: Any,
    segments: Sequence[Mapping[str, Any]],
    *,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    execution = _new_execution()
    volumes = np.asarray(
        runtime.geometry_by_resolution[a32.NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    for segment_index, segment in enumerate(segments, start=1):
        case_id = str(segment["case_id"])
        start_call = int(segment["start_call"])
        print(
            f"A42 segment {segment_index}/{len(segments)}: "
            f"{case_id} calls {start_call}-{start_call + 1}",
            flush=True,
        )
        shadow = np.asarray(segment["shadow"], dtype=np.float64)
        accepted = np.asarray(segment["accepted"], dtype=np.float64)
        for local_step in range(2):
            input_call = start_call + local_step
            if binary_position_phase_correction_active(
                position_selected=True, input_call=input_call
            ):
                raise ValueError("A42 includes a correction-active call")
            call_order: list[tuple[int, int]] = []

            def predictor(resolution, value, call_order=call_order):
                if resolution != a32.NATIVE_RESOLUTION:
                    raise ValueError("A42 predictor received a non-native call")
                call_order.append(resolution)
                prediction, timing = predict_resolution_sample(
                    runtime.model,
                    runtime.sample_by_resolution[resolution],
                    value,
                    device=runtime.device,
                    amp="none",
                    repeats=REPEATS,
                )
                execution["native_logical_predictions"] += 1
                execution["native_actual_forward_passes"] += len(
                    timing["forward_seconds"]
                )
                execution["native_forward_seconds"] += float(
                    sum(timing["forward_seconds"])
                )
                execution["maximum_peak_gpu_memory_bytes"] = max(
                    execution["maximum_peak_gpu_memory_bytes"],
                    int(timing["peak_gpu_memory_bytes"]),
                )
                execution["maximum_repeat_abs"] = max(
                    execution["maximum_repeat_abs"],
                    float(timing["repeat_max_abs"]),
                )
                return prediction

            step = synchronized_shadow_tethered_binary_affine_step(
                accepted,
                shadow,
                position_selected=True,
                input_call=input_call,
                contract=a32.CONTRACT,
                projector=runtime.native_projector,
                predictor=predictor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                volumes=volumes,
                residual_scale=residual_scale,
                state_scale=state_scale,
            )
            accepted_prediction = np.asarray(
                step.proposal.predictions[a32.NATIVE_RESOLUTION], dtype=np.float64
            )
            shadow_prediction = np.asarray(step.shadow_prediction, dtype=np.float64)
            exact_offset, exact_audit = a41._tether_offset(
                runtime,
                accepted=accepted,
                shadow_prediction=shadow_prediction,
                accepted_prediction=accepted_prediction,
                input_call=input_call,
                position_selected=True,
                fine_on_native=None,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
            )
            tether_reconstruction = _maximum_abs(
                exact_offset - step.retained_displacement
            )
            execution["maximum_exact_tether_reconstruction_abs"] = max(
                execution["maximum_exact_tether_reconstruction_abs"],
                tether_reconstruction,
            )
            exact_next = np.asarray(step.next_native_state, dtype=np.float64)
            exact_admissibility = conservative_admissibility_summary(
                exact_next, gamma=runtime.normalization.gamma
            )
            records.append(
                {
                    "case_id": case_id,
                    "group": "e12" if case_id.startswith("sv_e12_") else "e14",
                    "segment_start_call": start_call,
                    "input_call": input_call,
                    "band": _coast_band(input_call),
                    "active_case": True,
                    "correction_active": False,
                    "call_order_exact": call_order
                    == [a32.NATIVE_RESOLUTION, a32.NATIVE_RESOLUTION],
                    "logical_call_count": step.logical_call_count,
                    "shadow": np.array(shadow, copy=True),
                    "accepted": np.array(accepted, copy=True),
                    "shadow_prediction": np.array(shadow_prediction, copy=True),
                    "fine_on_native": None,
                    "x": a41._active_coordinates(accepted - shadow, runtime),
                    "z": a41._active_coordinates(
                        accepted_prediction - shadow_prediction, runtime
                    ),
                    "exact_offset": np.array(exact_offset, copy=True),
                    "exact_audit": exact_audit,
                    "exact_next_finite": bool(np.isfinite(exact_next).all()),
                    "exact_next_admissible": bool(exact_admissibility["admissible"]),
                    "exact_minimum_density": exact_admissibility["minimum_density"],
                    "exact_minimum_pressure": exact_admissibility["minimum_pressure"],
                    "exact_minimum_internal_energy": exact_admissibility[
                        "minimum_internal_energy"
                    ],
                    "exact_tether_reconstruction_abs": tether_reconstruction,
                    "exact_maximum_integral_difference_abs": (
                        step.tether_audit.maximum_integral_difference_abs
                    ),
                    "exact_maximum_boundary_difference_abs": (
                        step.tether_audit.maximum_boundary_difference_abs
                    ),
                    "exact_maximum_projection_idempotence_abs": (
                        step.tether_audit.maximum_projection_idempotence_abs
                    ),
                    "exact_maximum_update_identity_abs": (
                        step.maximum_update_identity_abs
                    ),
                }
            )
            shadow = shadow_prediction
            accepted = exact_next
        shadow_end_abs = _maximum_abs(
            as_model_state(shadow)
            - np.asarray(segment["expected_shadow_end"], dtype=np.float64)
        )
        accepted_end_abs = _maximum_abs(
            as_model_state(accepted)
            - np.asarray(segment["expected_accepted_end"], dtype=np.float64)
        )
        execution["maximum_shadow_end_serialization_abs"] = max(
            execution["maximum_shadow_end_serialization_abs"], shadow_end_abs
        )
        execution["maximum_accepted_end_serialization_abs"] = max(
            execution["maximum_accepted_end_serialization_abs"], accepted_end_abs
        )
        execution["segments"] += 1
        segment_rows.append(
            {
                "case_id": case_id,
                "segment_start_call": start_call,
                "segment_end_call": start_call + 2,
                "shadow_end_serialization_abs": shadow_end_abs,
                "accepted_end_serialization_abs": accepted_end_abs,
            }
        )
    execution["response_rows"] = len(records)
    return records, segment_rows, execution


def _attach_update_baseline(
    zero_rows: Sequence[Mapping[str, Any]], candidate_rows: list[dict[str, Any]]
) -> None:
    lookup = {
        (str(row["case_id"]), int(row["input_call"])): float(row["update_error_sse"])
        for row in zero_rows
    }
    if len(lookup) != EXPECTED_ROWS:
        raise ValueError("A42 zero-map baseline inventory differs")
    for row in candidate_rows:
        key = (str(row["case_id"]), int(row["input_call"]))
        if key not in lookup:
            raise ValueError("A42 candidate lacks its zero-map baseline")
        row["update_zero_sse"] = lookup[key]


def _score_candidate(
    runtime: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficients = {"diagonal": np.asarray(FROZEN_DIAGONAL, dtype=np.float64)}
    zero_rows = a41._score_map(
        runtime,
        records,
        map_name="zero",
        coefficients=coefficients,
        coefficient_source="a42_zero_baseline",
        start_coefficients=start_coefficients,
        end_coefficients=end_coefficients,
    )
    candidate_rows = a41._score_map(
        runtime,
        records,
        map_name="diagonal",
        coefficients=coefficients,
        coefficient_source="a41_frozen_diagonal",
        start_coefficients=start_coefficients,
        end_coefficients=end_coefficients,
    )
    for row in zero_rows:
        row["update_zero_sse"] = row["update_error_sse"]
    _attach_update_baseline(zero_rows, candidate_rows)
    for record, row in zip(records, candidate_rows, strict=True):
        prediction = a41.predict_response(record["x"], "diagonal", coefficients)
        accepted_prediction = record["shadow_prediction"] + a41._field_from_active(
            prediction, runtime
        )
        offset, _ = a41._tether_offset(
            runtime,
            accepted=record["accepted"],
            shadow_prediction=record["shadow_prediction"],
            accepted_prediction=accepted_prediction,
            input_call=int(record["input_call"]),
            position_selected=True,
            fine_on_native=None,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
        )
        surrogate_next = record["shadow_prediction"] + offset
        admissibility = conservative_admissibility_summary(
            surrogate_next, gamma=runtime.normalization.gamma
        )
        row.update(
            {
                "surrogate_next_finite": bool(np.isfinite(surrogate_next).all()),
                "surrogate_next_admissible": bool(admissibility["admissible"]),
                "surrogate_minimum_density": admissibility["minimum_density"],
                "surrogate_minimum_pressure": admissibility["minimum_pressure"],
                "surrogate_minimum_internal_energy": admissibility[
                    "minimum_internal_energy"
                ],
            }
        )
    return zero_rows, candidate_rows


def _score_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scope, case_ids in (
        ("e12", a41.CALIBRATION_CASES),
        ("e14", a41.TRANSFER_CASES),
    ):
        output.append(
            {
                "scope": scope,
                "band": "overall",
                **a41.scope_metrics(rows, case_ids=case_ids),
            }
        )
        for band in COAST_BANDS:
            output.append(
                {
                    "scope": scope,
                    "band": band,
                    **a41.scope_metrics(
                        [row for row in rows if row["band"] == band],
                        case_ids=case_ids,
                    ),
                }
            )
    return output


def _gate(
    scores: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    def strong(row: Mapping[str, Any], threshold: float) -> bool:
        return (
            row["response_skill_status"] == "ok"
            and row["response_skill"] is not None
            and float(row["response_skill"]) > threshold
            and row["update_skill_status"] == "ok"
            and row["update_skill"] is not None
            and float(row["update_skill"]) > threshold
        )

    overall = [row for row in scores if row["band"] == "overall"]
    bands = [row for row in scores if row["band"] != "overall"]
    case_overall = all(
        case[f"{metric}_status"] == "ok"
        and case[metric] is not None
        and float(case[metric]) > 0.99
        for row in overall
        for case in row["case_metrics"]
        for metric in ("response_skill", "update_skill", "cosine")
    )
    case_bands = all(
        case[f"{metric}_status"] == "ok"
        and case[metric] is not None
        and float(case[metric]) > 0.95
        for row in bands
        for case in row["case_metrics"]
        for metric in ("response_skill", "update_skill")
    )
    checks = {
        **structural_checks,
        "overall_response_and_update_skill": len(overall) == 2
        and all(strong(row, 0.99) for row in overall),
        "band_response_and_update_skill": len(bands) == 4
        and all(strong(row, 0.95) for row in bands)
        and case_bands,
        "case_and_scope_response_cosine": case_overall
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
        "per_row_tether_error_ratio": len(candidate_rows) == EXPECTED_ROWS
        and all(
            (
                row["update_error_to_exact_offset_status"] == "ok"
                and row["update_error_to_exact_offset"] is not None
                and float(row["update_error_to_exact_offset"]) < 0.50
            )
            or (
                row["update_error_to_exact_offset_status"] == "small_denominator"
                and float(row["exact_offset_rms"]) <= a41.DENOMINATOR_FLOOR
                and float(row["update_error_rms"]) <= a41.DENOMINATOR_FLOOR
            )
            for row in candidate_rows
        ),
    }
    return {
        "status": "qualified" if all(checks.values()) else "failed",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
    }


def run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"A42 requires CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
        )
    preflight = _verify_preflight(args.preflight)
    _verify_a41(args.a41_result)
    _, manifest, _ = a40._verify_a32_contract(args.a32_result, args.bundle_manifest)
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
            raise ValueError("A42 runtime source differs from preflight")
        segments = _load_segments(args.bundle_manifest, manifest)
        records, segment_rows, execution = _collect_records(
            runtime,
            segments,
            start_coefficients=start,
            end_coefficients=end,
        )
        zero_rows, candidate_rows = _score_candidate(
            runtime,
            records,
            start_coefficients=start,
            end_coefficients=end,
        )
        scores = _score_inventory(candidate_rows)
        input_pairs = {
            (str(record["case_id"]), int(record["input_call"])) for record in records
        }
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "coefficient_identity_exact": preflight["source_manifest"][
                "coast_contract"
            ]["frozen_diagonal"]
            == list(FROZEN_DIAGONAL),
            "segment_inventory_exact": execution["segments"] == EXPECTED_SEGMENTS,
            "row_inventory_exact": execution["response_rows"] == EXPECTED_ROWS
            and input_pairs
            == {(case_id, call) for case_id in CASE_IDS for call in COAST_CALLS},
            "native_call_inventory_exact": execution["native_logical_predictions"]
            == NATIVE_LOGICAL_PREDICTIONS
            and execution["native_actual_forward_passes"]
            == NATIVE_LOGICAL_PREDICTIONS * REPEATS,
            "no_fine_calls": execution["fine_logical_predictions"] == 0
            and execution["fine_actual_forward_passes"] == 0,
            "call_order_and_correction_inactive": all(
                bool(record["call_order_exact"])
                and int(record["logical_call_count"]) == 2
                and record["correction_active"] is False
                for record in records
            ),
            "repeatability": execution["maximum_repeat_abs"] <= REPEAT_TOLERANCE,
            "exact_next_bundle_serialization": execution[
                "maximum_shadow_end_serialization_abs"
            ]
            <= SERIALIZATION_TOLERANCE
            and execution["maximum_accepted_end_serialization_abs"]
            <= SERIALIZATION_TOLERANCE,
            "exact_tether_reconstruction": execution[
                "maximum_exact_tether_reconstruction_abs"
            ]
            <= STRICT_TOLERANCE,
            "finite_and_admissible_states": all(
                bool(record["exact_next_finite"])
                and bool(record["exact_next_admissible"])
                for record in records
            )
            and all(
                bool(row["surrogate_next_finite"])
                and bool(row["surrogate_next_admissible"])
                for row in candidate_rows
            ),
            "integral_boundary_projection_update_closure": all(
                max(
                    float(record["exact_maximum_integral_difference_abs"]),
                    float(record["exact_maximum_boundary_difference_abs"]),
                    float(record["exact_maximum_projection_idempotence_abs"]),
                    float(record["exact_maximum_update_identity_abs"]),
                )
                <= STRICT_TOLERANCE
                for record in records
            )
            and all(
                max(
                    float(row["maximum_integral_difference_abs"]),
                    float(row["maximum_boundary_difference_abs"]),
                    float(row["maximum_projection_idempotence_abs"]),
                    float(row["maximum_update_identity_abs"]),
                )
                <= STRICT_TOLERANCE
                for row in candidate_rows
            ),
            "truth_arrays_not_loaded": True,
            "surrogate_recurrence_not_executed": True,
            "h30_rollout_not_executed": True,
        }
        gate = _gate(scores, candidate_rows, structural_checks=structural_checks)
        execution.update(
            {
                "wall_seconds": perf_counter() - started,
                "environment": runtime_environment(runtime.device),
            }
        )
        write_csv(
            output_dir / "coast_response_records.csv",
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
        write_csv(output_dir / "coast_segment_closure.csv", segment_rows)
        write_csv(output_dir / "coast_response_zero_rows.csv", zero_rows)
        write_csv(output_dir / "coast_response_candidate_rows.csv", candidate_rows)
        write_csv(
            output_dir / "coast_response_scores.csv",
            [
                {key: value for key, value in row.items() if key != "case_metrics"}
                for row in scores
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
                    "qualified_coast_response_geometry"
                    if gate["status"] == "qualified"
                    else "stopped_coast_response_geometry"
                ),
                "gate": gate,
                "structural_checks": structural_checks,
                "frozen_diagonal": list(FROZEN_DIAGONAL),
                "scores": scores,
                "execution": execution,
                "maximums": {
                    "per_row_tether_error_ratio": max(
                        float(row["update_error_to_exact_offset"])
                        for row in candidate_rows
                        if row["update_error_to_exact_offset"] is not None
                    ),
                    "exact_minimum_density": min(
                        float(record["exact_minimum_density"]) for record in records
                    ),
                    "exact_minimum_pressure": min(
                        float(record["exact_minimum_pressure"]) for record in records
                    ),
                    "surrogate_minimum_density": min(
                        float(row["surrogate_minimum_density"])
                        for row in candidate_rows
                    ),
                    "surrogate_minimum_pressure": min(
                        float(row["surrogate_minimum_pressure"])
                        for row in candidate_rows
                    ),
                },
                "a41_result_sha256": sha256_file(args.a41_result),
                "a41_payload_sha256": A41_PAYLOAD_SHA256,
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
                "fine_predictions_executed": False,
                "local_exact_two_step_segments_executed": True,
                "surrogate_recurrence_executed": False,
                "h30_rollout_executed": False,
                "accuracy_scored": False,
                "new_population_opened": False,
                "claim_boundary": (
                    "Adaptive-open target-free local coast-response geometry on "
                    "saved float32 A32 states only; no H30, PDE-accuracy, independent, "
                    "cross-family, conservation, convergence, off-grid, or Richardson "
                    "claim."
                ),
            }
        )
        atomic_write_json(output_dir / "shadow_response_coast_geometry.json", result)
        return result, 0 if gate["status"] == "qualified" else 4
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
    parser.add_argument("--a41-result", type=Path, required=True)


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
