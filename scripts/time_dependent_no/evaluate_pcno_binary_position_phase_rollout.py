#!/usr/bin/env python3
"""Evaluate the frozen A31 binary position/phase affine correction recurrently."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as base,
)
from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as fine_eval,
)
from scripts.time_dependent_no import evaluate_pcno_modal_affine_transfer as a29
from scripts.time_dependent_no import evaluate_pcno_response_filtered_block as response
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    runtime_environment,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    NativeIncrementBasis,
    case_first_statistics,
    score_scalar_correction,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    VIEW_SPECS,
    _statistics_for_view,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    load_resolution_reference,
    predict_resolution_sample,
    reference_at_resolution,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    transverse_velocity_position_descriptor,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    BINARY_POSITION_PHASE_CALLS,
    BINARY_POSITION_PHASE_POLICY,
    synchronized_binary_affine_modal_step,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A31-SP19-BINARY-POSITION-PHASE-ROLLOUT"
PREFLIGHT_SCHEMA = "pcno_sp19_binary_position_phase_rollout_preflight_v1"
RESULT_SCHEMA = "pcno_sp19_binary_position_phase_rollout_v1"
SOURCE_SCHEMA = "pcno_sp19_binary_position_phase_rollout_source_v1"
ANIMATION_SCHEMA = "pcno_sp19_binary_position_phase_animation_bundles_v1"

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_binary_position_phase_rollout.py",
    "scripts/time_dependent_no/visualize_pcno_binary_position_phase_rollout.py",
    "tests/time_dependent_no/test_pcno_sparse_modal_correction.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/analyze_pcno_binary_position_phase_rescore.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_fine_discrepancy_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "scripts/time_dependent_no/evaluate_pcno_response_filtered_block.py",
    "scripts/time_dependent_no/visualize_pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_cross_resolution_correction.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "utility/time_dependent_no/pcno_fine_discrepancy_correction.py",
    "utility/time_dependent_no/pcno_propagated_sensitivity.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
)

A30_RESULT_SHA256 = "1ad13e2a1b797f335e342fabab295bff84fd458ee89401d2868530839c9c03c6"
A30_PAYLOAD_SHA256 = "793cc29a3d2deb0295f121c07992a6e468a8c4ee33ad734918c78ef744332a3f"
A28_RESULT_SHA256 = "dfe21dd7cb97e7b7fde4a97707cca2acf811f618d43e5100bbb439c9cfcb9d2d"
A28_PAYLOAD_SHA256 = "4ddf57745878058171bef82b2055373a57e1e4a5612e3430b05b1cc225a36c62"
A29_REFERENCE_SHA256 = "8fc26a6be7ba39866257c71c4930a9068cfb5a1438f3a634a90026c75060be73"

CONTRACT = a29.CONTRACT
NATIVE_RESOLUTION = a29.NATIVE_RESOLUTION
FINE_RESOLUTION = a29.FINE_RESOLUTION
INPUT_CALLS = tuple(range(30))
BANDS = {
    "overall": INPUT_CALLS,
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
    "endpoint_29": (29,),
}
GROUP_CASES = a29.GROUP_CASES
CASE_IDS = a29.CASE_IDS
CASE_TO_GROUP = {
    case_id: group for group, cases in GROUP_CASES.items() for case_id in cases
}
EXPECTED_ACTIVE_CASES = (
    "sv_e12_y01",
    "sv_e12_y07",
    "sv_e14_y01",
    "sv_e14_y07",
)
INACTIVE_CASES = tuple(case_id for case_id in CASE_IDS if case_id not in EXPECTED_ACTIVE_CASES)
ANIMATION_CASE_IDS = (
    "sv_e12_y01",
    "sv_e12_y04",
    "sv_e12_y07",
    "sv_e14_y01",
    "sv_e14_y07",
)
ANIMATION_CALLS = tuple(range(0, 31, 2))
ANIMATION_CONTRACT = {
    "purpose": "post_run_diagnostic_only_not_a_selector_or_gate",
    "case_ids": list(ANIMATION_CASE_IDS),
    "output_calls": list(ANIMATION_CALLS),
    "physical_times": [0.02 * call for call in ANIMATION_CALLS],
    "native_resolution": list(NATIVE_RESOLUTION),
    "scored_output_calls": list(range(31)),
    "visualization_only_frame_stride": 2,
    "gamma": 1.4,
    "field_limits": {"density": [0.75, 1.25], "pressure": [0.65, 1.35]},
    "absolute_error_limits": [1.0e-5, 0.25],
    "relative_improvement_limits": [-1.0, 1.0],
    "relative_improvement_denominator_floor": 1.0e-5,
    "candidate_label": "binary position-phase correction",
    "output_stem": "binary_position_phase_h30",
}
CONTROL_LIMIT = 1.05
STRICT_TOLERANCE = 1.0e-12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _maximum_abs(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _verify_a30(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A30_RESULT_SHA256:
        raise ValueError("A30 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("payload_sha256") != A30_PAYLOAD_SHA256
        or payload.get("status") != "retrospective_protocol_ready"
        or payload.get("gate", {}).get("status") != "passed"
        or not all(payload.get("gate", {}).get("checks", {}).values())
    ):
        raise ValueError("A30 did not pass the exact registered gate")
    return payload


def _verify_a28(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if sha256_file(path) != A28_RESULT_SHA256:
        raise ValueError("A28 result file SHA-256 mismatch")
    start, end, payload = a29._verify_affine_map(path)
    if payload.get("payload_sha256") != A28_PAYLOAD_SHA256:
        raise ValueError("A28 result payload SHA-256 mismatch")
    return start, end, payload


def _reference_rows(path: Path) -> dict[str, dict[str, str]]:
    if sha256_file(path) != A29_REFERENCE_SHA256:
        raise ValueError("A29 reference-check SHA-256 mismatch")
    rows = _read_csv(path)
    if len(rows) != len(CASE_IDS) or {row.get("case_id") for row in rows} != set(CASE_IDS):
        raise ValueError("A29 reference-check inventory is not exact")
    return {str(row["case_id"]): row for row in rows}


def _source_hashes() -> dict[str, Any]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace, *, base_source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _verify_a30(args.a30_result)
    _verify_a28(args.a28_map)
    _reference_rows(args.a29_reference_checks)
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a30_result_sha256": sha256_file(args.a30_result),
                "a30_payload_sha256": A30_PAYLOAD_SHA256,
                "a28_result_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": A28_PAYLOAD_SHA256,
                "a29_reference_checks_sha256": sha256_file(
                    args.a29_reference_checks
                ),
            },
            "base_runtime_source_manifest": dict(base_source_manifest),
            "population": {
                "case_ids": list(CASE_IDS),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "status": "already_open_e12_e14_interiors_adaptive",
                "new_population_opened": False,
            },
            "inference_contract": {
                "position_descriptor": "call_zero_transverse_velocity_centroid",
                "position_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "active_input_calls": list(BINARY_POSITION_PHASE_CALLS),
                "active_call_order": ["native", "fine"],
                "inactive_call_order": ["native"],
                "one_native_state_retained": True,
                "fine_truth_loaded": False,
                "coefficient_refit": False,
                "truth_or_case_id_in_selector": False,
            },
            "animation_contract": ANIMATION_CONTRACT,
        }
    )


def synthetic_summary() -> dict[str, Any]:
    distances = {
        "near": np.nextafter(FROZEN_POSITION_BUFFER_THRESHOLD, 0.0),
        "edge": FROZEN_POSITION_BUFFER_THRESHOLD,
        "far": np.nextafter(FROZEN_POSITION_BUFFER_THRESHOLD, 1.0),
    }
    calls = list(BINARY_POSITION_PHASE_CALLS)
    checks = {
        "active_calls_exact": calls == [*range(8), *range(22, 30)],
        "active_call_count_16": len(calls) == 16,
        "threshold_inclusive": bool(
            distances["edge"] <= FROZEN_POSITION_BUFFER_THRESHOLD
        ),
        "above_threshold_inactive": bool(
            distances["far"] > FROZEN_POSITION_BUFFER_THRESHOLD
        ),
        "population_inventory_14": len(CASE_IDS) == 14,
        "expected_active_inventory_4": len(EXPECTED_ACTIVE_CASES) == 4,
        "logical_main_inventory": 420 + 420 + 64 == 904,
        "animation_calls_exact": ANIMATION_CALLS == tuple(range(0, 31, 2)),
        "owned_source_inventory": len(OWNED_SOURCE_PATHS) == 5,
    }
    return {
        "schema": "pcno_sp19_binary_position_phase_rollout_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, base_source = base._open_contract(a29._base_args(args))
    try:
        provenance = {
            case_id: family_case_provenance(manifest, case_id) for case_id in CASE_IDS
        }
        source = _source_manifest(args, base_source_manifest=base_source)
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"])
            == set(DEPENDENCY_PATHS),
            "lineage_exact": source["lineage"]
            == {
                "a30_result_sha256": A30_RESULT_SHA256,
                "a30_payload_sha256": A30_PAYLOAD_SHA256,
                "a28_result_sha256": A28_RESULT_SHA256,
                "a28_payload_sha256": A28_PAYLOAD_SHA256,
                "a29_reference_checks_sha256": A29_REFERENCE_SHA256,
            },
            "case_inventory_present": all(case_id in store.keys for case_id in CASE_IDS),
            "population_exact": set(provenance) == set(CASE_IDS),
            "groups_exact": all(
                provenance[case_id].get("split") == "test"
                and provenance[case_id].get("split_group_id") == a29.GROUP_IDS[group]
                for group, cases in GROUP_CASES.items()
                for case_id in cases
            ),
            "checkpoint_stride_exact": base.checkpoint_step_stride(checkpoint) == 2,
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest": source,
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "truth_arrays_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
                "population": provenance,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A31 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("working_id") != WORKING_ID
        or payload.get("status") != "passed"
        or not all(payload.get("checks", {}).values())
        or payload.get("checkpoint_model_built") is not False
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("model_calls") != 0
        or payload.get("recurrence_executed") is not False
        or payload.get("source_manifest")
        != _source_manifest(
            args,
            base_source_manifest=payload["source_manifest"][
                "base_runtime_source_manifest"
            ],
        )
    ):
        raise ValueError("A31 preflight differs from the frozen contract")
    return payload


def _new_execution() -> dict[str, Any]:
    return {
        "logical_model_calls": 0,
        "actual_forward_passes": 0,
        "total_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "native_logical_calls": 0,
        "fine_logical_calls": 0,
        "native_actual_forward_passes": 0,
        "fine_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_forward_seconds": 0.0,
    }


def _load_reference(
    runtime: Any,
    case_id: str,
    expected: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case_id.startswith("sv_e12_"):
        reference, check = load_resolution_reference(
            runtime.args.family_root,
            None,
            runtime.store,
            runtime.manifest,
            case_id,
            training_resolution=NATIVE_RESOLUTION,
        )
    elif case_id.startswith("sv_e14_"):
        reference, check = a29._load_shard_native_reference(runtime, case_id)
    else:  # pragma: no cover - fixed population
        raise ValueError(f"unsupported A31 case: {case_id}")
    a29._require_reference_row(check, expected)
    states = np.asarray(reference["conservative_states"], dtype=np.float64)
    if states.shape != (61, NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1], 4):
        raise ValueError("A31 native reference shape differs")
    return reference, check


def _descriptor(runtime: Any, initial: np.ndarray) -> Any:
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    return transverse_velocity_position_descriptor(
        initial,
        nodes=np.asarray(geometry.nodes, dtype=np.float64),
        volumes=np.asarray(geometry.node_measures, dtype=np.float64),
        y_min=float(runtime.first_config.y_min),
        y_max=float(runtime.first_config.y_max),
    )


def _run_arm(
    runtime: Any,
    *,
    case_id: str,
    reference: Mapping[str, Any],
    policy: str,
    position_selected: bool,
    descriptor: Any,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    if policy not in {"zero", BINARY_POSITION_PHASE_POLICY}:
        raise ValueError("unsupported A31 rollout policy")
    reference_resolution = tuple(int(value) for value in reference["retained_resolution"])
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("A31 native initial state is unavailable")
    state = np.asarray(initial, dtype=np.float64)
    states = [np.array(state, copy=True)]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    execution = _new_execution()
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    cumulative = np.zeros_like(state)
    started = perf_counter()
    for input_call in range(horizon):
        current_reference = reference_at_resolution(
            reference["conservative_states"][2 * input_call],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        target = reference_at_resolution(
            reference["conservative_states"][2 * (input_call + 1)],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if current_reference is None or target is None:
            raise ValueError("A31 native rollout truth is unavailable")
        current_reference = np.asarray(current_reference, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        call_order: list[tuple[int, int]] = []

        def predictor(resolution, value, call_order=call_order):
            call_order.append(resolution)
            prediction, timing = predict_resolution_sample(
                runtime.model,
                runtime.sample_by_resolution[resolution],
                value,
                device=runtime.device,
                amp="none",
                repeats=1,
            )
            fine_eval._account_execution(execution, timing, resolution=resolution)
            return prediction

        if policy == "zero":
            next_state = np.asarray(predictor(NATIVE_RESOLUTION, state), dtype=np.float64)
            correction = np.zeros_like(state)
            correction_active = False
            cap_active = False
            correction_status = "raw_native"
            update_closure = 0.0
            pre_fine_floor = 0.0
            post_fine_floor = 0.0
        else:
            step = synchronized_binary_affine_modal_step(
                state,
                position_selected=position_selected,
                input_call=input_call,
                contract=CONTRACT,
                projector=runtime.native_projector,
                predictor=predictor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                volumes=volumes,
                component_scale=residual_scale,
            )
            next_state = np.asarray(step.next_native_state, dtype=np.float64)
            correction = np.asarray(step.correction, dtype=np.float64)
            correction_active = bool(step.correction_active)
            cap_active = bool(step.audit.cap_active)
            correction_status = str(step.audit.status)
            native_prediction = np.asarray(
                step.predictions[NATIVE_RESOLUTION], dtype=np.float64
            )
            update_closure = _maximum_abs(next_state - native_prediction - correction)
            if step.prepared_inputs is None:
                pre_fine_floor = 0.0
                post_fine_floor = 0.0
            else:
                pre_fine_floor = float(
                    step.prepared_inputs.nesting_floors[
                        "pre_model_fine_to_native_max_abs"
                    ]
                )
                post_fine_floor = float(
                    step.prepared_inputs.nesting_floors[
                        "post_fp32_fine_to_native_max_abs"
                    ]
                )
            expected_order = (
                [NATIVE_RESOLUTION, FINE_RESOLUTION]
                if correction_active
                else [NATIVE_RESOLUTION]
            )
            audit_rows.append(
                {
                    "case_id": case_id,
                    "input_call": input_call,
                    "descriptor_status": descriptor.status,
                    "normalized_wall_distance": descriptor.normalized_wall_distance,
                    "position_selected": position_selected,
                    "correction_active": correction_active,
                    "call_order": ">".join(f"{x}x{y}" for x, y in call_order),
                    "call_order_exact": call_order == expected_order,
                    "logical_call_count": step.logical_call_count,
                    "pre_model_fine_to_native_max_abs": pre_fine_floor,
                    "post_fp32_fine_to_native_max_abs": post_fine_floor,
                    "state_update_closure_max_abs": update_closure,
                    **asdict(step.audit),
                }
            )
        row, cumulative = response._block_metric_row(
            runtime,
            case_id=case_id,
            policy=policy,
            input_call=input_call,
            previous_state=state,
            next_state=next_state,
            current_reference=current_reference,
            target=target,
            cumulative_defect=cumulative,
            intervention=correction,
            cap_active=cap_active,
            correction_status=correction_status,
        )
        row.update(
            {
                "position_selected": position_selected,
                "correction_active": correction_active,
                "logical_calls_this_step": len(call_order),
                "state_update_closure_max_abs": update_closure,
                "pre_model_fine_to_native_max_abs": pre_fine_floor,
                "post_fp32_fine_to_native_max_abs": post_fine_floor,
            }
        )
        rows.append(row)
        state = next_state
        states.append(np.array(state, copy=True))
        if not row["finite"] or not row["admissible"]:
            break
    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = len(rows)
    return {
        "case_id": case_id,
        "policy": policy,
        "position_selected": position_selected,
        "descriptor": asdict(descriptor),
        "rows": rows,
        "states": states,
        "audit_rows": audit_rows,
        "execution": execution,
    }


def _pair_view_statistics(
    runtime: Any,
    *,
    case_id: str,
    reference: Mapping[str, Any],
    raw_states: Sequence[np.ndarray],
    candidate_states: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(raw_states) != 31 or len(candidate_states) != 31:
        raise ValueError("A31 paired trajectory is incomplete")
    reference_resolution = tuple(int(value) for value in reference["retained_resolution"])
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    output: list[dict[str, Any]] = []
    maxima = {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
    for input_call in INPUT_CALLS:
        target = reference_at_resolution(
            reference["conservative_states"][2 * (input_call + 1)],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if target is None:
            raise ValueError("A31 view target is unavailable")
        target = np.asarray(target, dtype=np.float64)
        raw = np.asarray(raw_states[input_call + 1], dtype=np.float64)
        candidate = np.asarray(candidate_states[input_call + 1], dtype=np.float64)
        correction = candidate - raw
        target_correction = target - raw
        zero = np.zeros_like(correction)
        masks, _, _ = a29.shock_vortex_regions(
            target,
            geometry.nodes,
            resolution=NATIVE_RESOLUTION,
            gamma=runtime.normalization.gamma,
        )
        snapshot = DiagnosticSnapshot(
            case_id=case_id,
            group_id=CASE_TO_GROUP[case_id],
            input_call=input_call,
            basis=NativeIncrementBasis(
                native_increment=zero,
                coarse_on_native=zero,
                fine_on_native=correction,
                native_minus_coarse=zero,
                fine_minus_native=correction,
            ),
            target_correction=target_correction,
            volumes=volumes,
            component_scale=state_scale,
            masks=masks,
        )
        for view in VIEW_SPECS:
            statistics, closure = _statistics_for_view(
                snapshot,
                view,
                resolution=NATIVE_RESOLUTION,
                projector=runtime.native_projector,
            )
            maxima["band"] = max(
                maxima["band"],
                float(closure["maximum_reconstruction_abs_residual_scaled"]),
                float(closure["maximum_instantaneous_energy_relative_closure"]),
            )
            maxima["subspace_reconstruction"] = max(
                maxima["subspace_reconstruction"],
                float(closure["maximum_subspace_reconstruction_abs"]),
            )
            maxima["orthogonality"] = max(
                maxima["orthogonality"],
                abs(float(closure["maximum_parallel_orthogonal_weighted_inner"])),
            )
            output.append(
                {
                    "case_id": case_id,
                    "group_id": CASE_TO_GROUP[case_id],
                    "input_call": input_call,
                    "view": view.key,
                    "statistics": statistics,
                }
            )
    return output, maxima


def _population_specs() -> dict[str, tuple[str, ...]]:
    return {
        "population": CASE_IDS,
        "active_population": EXPECTED_ACTIVE_CASES,
        "group_e12": GROUP_CASES["e12"],
        "group_e14": GROUP_CASES["e14"],
        "active_group_e12": ("sv_e12_y01", "sv_e12_y07"),
        "active_group_e14": ("sv_e14_y01", "sv_e14_y07"),
    }


def _score_views(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, calls in BANDS.items():
        call_set = set(calls)
        for view in VIEW_SPECS:
            view_records = [
                row
                for row in records
                if row["view"] == view.key and row["input_call"] in call_set
            ]
            for scope, cases in _population_specs().items():
                selected = [
                    (str(row["case_id"]), row["statistics"])
                    for row in view_records
                    if row["case_id"] in cases
                ]
                score = score_scalar_correction(case_first_statistics(selected), (0.0, 1.0))
                rows.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "scope": scope,
                        "case_id": None,
                        **asdict(score),
                    }
                )
            for case_id in CASE_IDS:
                selected = [
                    (str(row["case_id"]), row["statistics"])
                    for row in view_records
                    if row["case_id"] == case_id
                ]
                score = score_scalar_correction(case_first_statistics(selected), (0.0, 1.0))
                rows.append(
                    {
                        "cell": cell,
                        "view": view.key,
                        "scope": "case",
                        "case_id": case_id,
                        **asdict(score),
                    }
                )
    return rows


def _score_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell: str,
    view: str,
    scope: str,
    case_id: str | None = None,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["cell"] == cell
        and row["view"] == view
        and row["scope"] == scope
        and (case_id is None or row.get("case_id") == case_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"missing A31 score row {(cell, view, scope, case_id)!r}")
    return matches[0]


def _field_controls(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    scopes = ("population", "active_population", "active_group_e12", "active_group_e14")
    for cell in ("overall", "endpoint_29"):
        for view in VIEW_SPECS:
            for scope in scopes:
                row = _score_row(score_rows, cell=cell, view=view.key, scope=scope)
                output.append(
                    {
                        "key": f"field::{view.key}",
                        "scope": scope,
                        "cell": cell,
                        "zero_rms": row["target_rms"],
                        "corrected_rms": (
                            float(row["target_rms"])
                            * float(row["rms_ratio_vs_zero"])
                            if row["rms_ratio_vs_zero"] is not None
                            else None
                        ),
                        "ratio": row["rms_ratio_vs_zero"],
                        "status": "ok" if row["skill_status"] == "ok" else row["skill_status"],
                        "policy": BINARY_POSITION_PHASE_POLICY,
                    }
                )
            for case_id in CASE_IDS:
                row = _score_row(
                    score_rows,
                    cell=cell,
                    view=view.key,
                    scope="case",
                    case_id=case_id,
                )
                output.append(
                    {
                        "key": f"field::{view.key}",
                        "scope": case_id,
                        "cell": cell,
                        "zero_rms": row["target_rms"],
                        "corrected_rms": (
                            float(row["target_rms"])
                            * float(row["rms_ratio_vs_zero"])
                            if row["rms_ratio_vs_zero"] is not None
                            else None
                        ),
                        "ratio": row["rms_ratio_vs_zero"],
                        "status": "ok" if row["skill_status"] == "ok" else row["skill_status"],
                        "policy": BINARY_POSITION_PHASE_POLICY,
                    }
                )
    return output


def _control_passed(row: Mapping[str, Any]) -> bool:
    return fine_eval._control_passed(row)


def _write_animation_bundle(
    output_dir: Path,
    *,
    runtime: Any,
    case_id: str,
    reference: Mapping[str, Any],
    raw_states: Sequence[np.ndarray],
    candidate_states: Sequence[np.ndarray],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_resolution = tuple(int(value) for value in reference["retained_resolution"])
    truth_states = []
    for output_call in range(31):
        value = reference_at_resolution(
            reference["conservative_states"][2 * output_call],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if value is None:
            raise ValueError("A31 animation truth is unavailable")
        truth_states.append(np.asarray(value, dtype=np.float64))
    frame_indices = np.asarray(ANIMATION_CALLS, dtype=np.int64)
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    path = output_dir / f"{case_id}_binary_position_phase_h30.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        split_group_id=np.asarray(a29.GROUP_IDS[CASE_TO_GROUP[case_id]]),
        output_calls=frame_indices,
        physical_times=np.asarray(ANIMATION_CONTRACT["physical_times"], dtype=np.float64),
        native_resolution=np.asarray(NATIVE_RESOLUTION, dtype=np.int64),
        nodes=np.asarray(geometry.nodes, dtype=np.float32),
        volumes=np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1),
        state_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float32),
        residual_scale=np.asarray(runtime.normalization.residual_scale, dtype=np.float32),
        gamma=np.asarray(runtime.normalization.gamma, dtype=np.float64),
        truth_conservative=np.asarray(truth_states, dtype=np.float64)[frame_indices].astype(np.float32),
        raw_shadow_conservative=np.asarray(raw_states, dtype=np.float64)[frame_indices].astype(np.float32),
        corrected_conservative=np.asarray(candidate_states, dtype=np.float64)[frame_indices].astype(np.float32),
    )
    return {
        "case_id": case_id,
        "split_group_id": a29.GROUP_IDS[CASE_TO_GROUP[case_id]],
        "path": path.name,
        "sha256": sha256_file(path),
        "frames": len(frame_indices),
        "storage_dtype": "float32_visualization_only",
    }


def _gate(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    inactive_max_abs: Mapping[str, float],
    prefix_repeat_abs: float,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    for scope in ("population", "active_population", "active_group_e12", "active_group_e14"):
        checks[f"{scope}_full_rank8_trajectory_strict"] = all(
            _score_row(score_rows, cell="overall", view=view, scope=scope)["skill_status"] == "ok"
            and float(
                _score_row(score_rows, cell="overall", view=view, scope=scope)[
                    "rms_ratio_vs_zero"
                ]
            )
            < 1.0
            for view in ("full", "rank8_parallel")
        )
    checks["active_population_endpoint_strict"] = (
        float(
            _score_row(
                score_rows,
                cell="endpoint_29",
                view="full",
                scope="active_population",
            )["rms_ratio_vs_zero"]
        )
        < 1.0
    )
    paired = {str(row["case_id"]): row for row in case_rows}
    active_wins = 0
    active_bounds = True
    for case_id in EXPECTED_ACTIVE_CASES:
        trajectory = float(
            _score_row(
                score_rows, cell="overall", view="full", scope="case", case_id=case_id
            )["rms_ratio_vs_zero"]
        )
        endpoint = float(
            _score_row(
                score_rows,
                cell="endpoint_29",
                view="full",
                scope="case",
                case_id=case_id,
            )["rms_ratio_vs_zero"]
        )
        active_wins += trajectory < 1.0 and endpoint < 1.0
        active_bounds = active_bounds and all(
            float(value) <= CONTROL_LIMIT
            for value in (
                trajectory,
                endpoint,
                paired[case_id]["increment_defect_rms_ratio"],
                paired[case_id]["endpoint_cumulative_defect_ratio"],
            )
        )
    checks["at_least_three_active_case_trajectory_endpoint_wins"] = active_wins >= 3
    checks["every_active_case_primary_no_harm"] = active_bounds
    checks["all_controls_no_harm"] = bool(controls) and all(
        _control_passed(row) for row in controls
    )
    checks["inactive_states_bitwise_raw"] = set(inactive_max_abs) == set(INACTIVE_CASES) and all(
        value == 0.0 for value in inactive_max_abs.values()
    )
    checks["all_rollouts_complete_finite_admissible"] = len(rollouts) == 2 * len(CASE_IDS) and all(
        row["execution"]["completed_calls"] == 30
        and all(metric["finite"] and metric["admissible"] for metric in row["rows"])
        for row in rollouts
    )
    active_audits = [row for row in audit_rows if row["correction_active"]]
    inactive_audits = [row for row in audit_rows if not row["correction_active"]]
    checks["active_audit_inventory_64"] = len(active_audits) == 64
    checks["inactive_audit_inventory_356"] = len(inactive_audits) == 356
    checks["active_correction_closure"] = all(
        row["status"] == "ok"
        and bool(row["call_order_exact"])
        and float(row["correction_to_native_increment"]) <= 0.05 + STRICT_TOLERANCE
        and float(row["maximum_scaled_component_mean_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_excluded_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_inactive_coordinate_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_modal_reconstruction_abs"]) <= STRICT_TOLERANCE
        and float(row["state_update_closure_max_abs"]) <= STRICT_TOLERANCE
        and float(row["pre_model_fine_to_native_max_abs"]) <= STRICT_TOLERANCE
        for row in active_audits
    )
    checks["inactive_calls_exact_one_native_no_correction"] = all(
        row["status"] == "inactive_exact_raw_native"
        and bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and float(row["correction_rms"]) == 0.0
        and float(row["state_update_closure_max_abs"]) == 0.0
        for row in inactive_audits
    )
    checks["deterministic_prefix"] = prefix_repeat_abs <= 1.0e-6
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "active_case_joint_win_count": active_wins,
        "strict_improvement_equality_fails": True,
        "control_ratio_limit": CONTROL_LIMIT,
    }


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight, args)
    start_coefficients, end_coefficients, _ = _verify_a28(args.a28_map)
    expected_references = _reference_rows(args.a29_reference_checks)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = base._build_runtime(a29._base_args(args))
    started = perf_counter()
    try:
        if runtime.source_manifest != preflight["source_manifest"]["base_runtime_source_manifest"]:
            raise ValueError("A31 runtime source differs from preflight")
        prefix_reference, _ = _load_reference(
            runtime,
            EXPECTED_ACTIVE_CASES[0],
            expected_references[EXPECTED_ACTIVE_CASES[0]],
        )
        prefix_initial = np.asarray(prefix_reference["conservative_states"][0], dtype=np.float64)
        prefix_descriptor = _descriptor(runtime, prefix_initial)
        prefix_selected = bool(
            prefix_descriptor.status == "ok"
            and prefix_descriptor.normalized_wall_distance is not None
            and prefix_descriptor.normalized_wall_distance
            <= FROZEN_POSITION_BUFFER_THRESHOLD
        )
        prefix_a = _run_arm(
            runtime,
            case_id=EXPECTED_ACTIVE_CASES[0],
            reference=prefix_reference,
            policy=BINARY_POSITION_PHASE_POLICY,
            position_selected=prefix_selected,
            descriptor=prefix_descriptor,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
            horizon=2,
        )
        prefix_b = _run_arm(
            runtime,
            case_id=EXPECTED_ACTIVE_CASES[0],
            reference=prefix_reference,
            policy=BINARY_POSITION_PHASE_POLICY,
            position_selected=prefix_selected,
            descriptor=prefix_descriptor,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
            horizon=2,
        )
        prefix_repeat_abs = max(
            _maximum_abs(left - right)
            for left, right in zip(prefix_a["states"], prefix_b["states"], strict=True)
        )
        prefix_execution = {
            "first": prefix_a["execution"],
            "second": prefix_b["execution"],
            "maximum_state_abs_difference": prefix_repeat_abs,
            "excluded_from_main_scientific_cost": True,
        }

        rollouts: list[dict[str, Any]] = []
        view_records: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        descriptor_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        inactive_max_abs: dict[str, float] = {}
        view_maxima = {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
        animation_rows = []
        animation_dir = output_dir / "animation_bundles"
        for case_id in CASE_IDS:
            print(f"A31 case {case_id}: loading authenticated native truth", flush=True)
            reference, reference_check = _load_reference(
                runtime, case_id, expected_references[case_id]
            )
            initial = np.asarray(reference["conservative_states"][0], dtype=np.float64)
            descriptor = _descriptor(runtime, initial)
            selected = bool(
                descriptor.status == "ok"
                and descriptor.normalized_wall_distance is not None
                and descriptor.normalized_wall_distance
                <= FROZEN_POSITION_BUFFER_THRESHOLD
            )
            descriptor_rows.append(
                {
                    "case_id": case_id,
                    "group_id": CASE_TO_GROUP[case_id],
                    **asdict(descriptor),
                    "position_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                    "position_selected": selected,
                    "expected_selected": case_id in EXPECTED_ACTIVE_CASES,
                }
            )
            print(f"A31 case {case_id}: raw H30", flush=True)
            raw = _run_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                policy="zero",
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            print(f"A31 case {case_id}: binary H30 selected={selected}", flush=True)
            candidate = _run_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                policy=BINARY_POSITION_PHASE_POLICY,
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            statistics, maxima = _pair_view_statistics(
                runtime,
                case_id=case_id,
                reference=reference,
                raw_states=raw["states"],
                candidate_states=candidate["states"],
            )
            view_records.extend(statistics)
            for key, value in maxima.items():
                view_maxima[key] = max(view_maxima[key], value)
            if case_id in INACTIVE_CASES:
                inactive_max_abs[case_id] = max(
                    _maximum_abs(left - right)
                    for left, right in zip(raw["states"], candidate["states"], strict=True)
                )
            if case_id in ANIMATION_CASE_IDS:
                animation_rows.append(
                    _write_animation_bundle(
                        animation_dir,
                        runtime=runtime,
                        case_id=case_id,
                        reference=reference,
                        raw_states=raw["states"],
                        candidate_states=candidate["states"],
                    )
                )
            reference_rows.append({"case_id": case_id, **reference_check})
            audit_rows.extend(candidate["audit_rows"])
            for rollout in (raw, candidate):
                rollout.pop("states")
                rollout.pop("audit_rows")
                rollouts.append(rollout)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        selected_cases = tuple(
            row["case_id"] for row in descriptor_rows if row["position_selected"]
        )
        score_rows = _score_views(view_records)
        all_metric_rows = [metric for rollout in rollouts for metric in rollout["rows"]]
        case_rows, paired_controls, paired_population = response._paired_rollout_controls_for_cases(
            all_metric_rows, case_ids=CASE_IDS
        )
        for row in paired_controls:
            row["cell"] = "overall"
            row["policy"] = BINARY_POSITION_PHASE_POLICY
        controls = [*_field_controls(score_rows), *paired_controls]
        animation_manifest = with_payload_sha256(
            {
                "schema": ANIMATION_SCHEMA,
                "working_id": WORKING_ID,
                "contract": ANIMATION_CONTRACT,
                "bundles": animation_rows,
                "inference_or_gate_input": False,
            }
        )
        atomic_write_json(animation_dir / "bundle_manifest.json", animation_manifest)
        main_execution = {
            "raw": {
                key: sum(float(row["execution"][key]) for row in rollouts if row["policy"] == "zero")
                for key in (
                    "logical_model_calls",
                    "actual_forward_passes",
                    "total_forward_seconds",
                    "native_logical_calls",
                    "fine_logical_calls",
                    "native_actual_forward_passes",
                    "fine_actual_forward_passes",
                    "native_forward_seconds",
                    "fine_forward_seconds",
                    "wall_seconds",
                )
            },
            "candidate": {
                key: sum(
                    float(row["execution"][key])
                    for row in rollouts
                    if row["policy"] == BINARY_POSITION_PHASE_POLICY
                )
                for key in (
                    "logical_model_calls",
                    "actual_forward_passes",
                    "total_forward_seconds",
                    "native_logical_calls",
                    "fine_logical_calls",
                    "native_actual_forward_passes",
                    "fine_actual_forward_passes",
                    "native_forward_seconds",
                    "fine_forward_seconds",
                    "wall_seconds",
                )
            },
            "maximum_peak_gpu_memory_bytes": max(
                int(row["execution"]["maximum_peak_gpu_memory_bytes"])
                for row in rollouts
            ),
            "wall_seconds": perf_counter() - started,
            "environment": runtime_environment(runtime.device),
        }
        main_execution["logical_cost_ratio_vs_raw"] = (
            main_execution["candidate"]["logical_model_calls"]
            / main_execution["raw"]["logical_model_calls"]
        )
        main_execution["forward_time_ratio_vs_raw"] = (
            main_execution["candidate"]["total_forward_seconds"]
            / main_execution["raw"]["total_forward_seconds"]
        )
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"] == _source_hashes(),
            "population_exact": set(selected_cases) == set(EXPECTED_ACTIVE_CASES),
            "descriptor_inventory_exact": len(descriptor_rows) == len(CASE_IDS),
            "reference_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "paired_metric_inventory_exact": len(all_metric_rows) == 2 * len(CASE_IDS) * len(INPUT_CALLS),
            "main_logical_call_inventory_exact": (
                main_execution["raw"]["native_logical_calls"] == 420
                and main_execution["raw"]["fine_logical_calls"] == 0
                and main_execution["candidate"]["native_logical_calls"] == 420
                and main_execution["candidate"]["fine_logical_calls"] == 64
            ),
            "view_closure": view_maxima["band"] <= 1.0e-10
            and view_maxima["subspace_reconstruction"] <= 1.0e-10
            and view_maxima["orthogonality"] <= 1.0e-10,
            "animation_inventory_exact": len(animation_rows) == len(ANIMATION_CASE_IDS),
            "fine_truth_not_loaded": True,
            "coefficient_refit_not_run": True,
            "one_native_state_per_arm": True,
        }
        gate = _gate(
            score_rows=score_rows,
            case_rows=case_rows,
            controls=controls,
            rollouts=rollouts,
            audit_rows=audit_rows,
            inactive_max_abs=inactive_max_abs,
            prefix_repeat_abs=prefix_repeat_abs,
            structural_checks=structural_checks,
        )
        write_csv(output_dir / "rollout_call_metrics.csv", all_metric_rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "view_scores.csv", score_rows)
        write_csv(output_dir / "correction_audits.csv", audit_rows)
        write_csv(output_dir / "position_decisions.csv", descriptor_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        write_csv(
            output_dir / "rollout_execution.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    "position_selected": row["position_selected"],
                    **row["execution"],
                }
                for row in rollouts
            ],
        )
        atomic_write_json(output_dir / "source_manifest.json", preflight["source_manifest"])
        artifacts = {
            str(path.relative_to(output_dir)).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }
        result = with_payload_sha256(
            {
                "schema": RESULT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "qualified_adaptive_open_rollout" if gate["status"] == "qualified" else "stopped_recurrent",
                "population_status": "already_open_e12_e14_adaptive_mechanism",
                "gate": gate,
                "population": paired_population,
                "primary_scores": {
                    scope: {
                        view: dict(
                            _score_row(
                                score_rows,
                                cell="overall",
                                view=view,
                                scope=scope,
                            )
                        )
                        for view in ("full", "rank8_parallel")
                    }
                    for scope in (
                        "population",
                        "active_population",
                        "active_group_e12",
                        "active_group_e14",
                    )
                },
                "failed_controls": [dict(row) for row in controls if not _control_passed(row)],
                "maximum_control_ratio": max(
                    float(row["ratio"]) for row in controls if row.get("ratio") is not None
                ),
                "inactive_case_maximum_state_abs_difference": inactive_max_abs,
                "view_closure_maxima": view_maxima,
                "main_execution": main_execution,
                "prefix_replay": prefix_execution,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
                "source_manifest_payload_sha256": preflight["source_manifest"]["payload_sha256"],
                "animation_bundle_manifest": animation_manifest,
                "animation_bundle_manifest_sha256": sha256_file(
                    animation_dir / "bundle_manifest.json"
                ),
                "artifact_inventory_before_summary": artifacts,
                "recurrence_executed": True,
                "animations_rendered": False,
                "fine_reference_loaded": False,
                "new_population_opened": False,
                "claim_boundary": (
                    "Adaptive-open E12/E14 recurrent cross-resolution correction evidence only; "
                    "no independent confirmation, cross-family coefficient transfer, direct "
                    "off-grid comparison, conservation, convergence-order, or Richardson claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(output_dir / "binary_position_phase_rollout.json", result)
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
    parser.add_argument("--a30-result", type=Path, required=True)
    parser.add_argument("--a29-reference-checks", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    rollout = commands.add_parser("rollout")
    _add_external_arguments(rollout)
    rollout.add_argument("--preflight", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    rollout.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    rollout.add_argument("--allow-cpu", action="store_true")
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
    payload, exit_code = run_rollout(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
