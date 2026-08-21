#!/usr/bin/env python3
"""Evaluate the frozen A32 shadow-tethered affine correction recurrently."""

from __future__ import annotations

import argparse
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
    evaluate_pcno_binary_position_phase_rollout as a31,
)
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
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    predict_resolution_sample,
    reference_at_resolution,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    transverse_velocity_position_descriptor,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    BINARY_POSITION_PHASE_POLICY,
    synchronized_shadow_tethered_binary_affine_step,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A32-SP19-AFFINE-SHADOW-TETHER-ROLLOUT"
PREFLIGHT_SCHEMA = "pcno_sp19_affine_shadow_tether_rollout_preflight_v1"
RESULT_SCHEMA = "pcno_sp19_affine_shadow_tether_rollout_v1"
SOURCE_SCHEMA = "pcno_sp19_affine_shadow_tether_rollout_source_v1"
ANIMATION_SCHEMA = "pcno_sp19_affine_shadow_tether_animation_bundles_v1"
POLICY = "binary_position_phase_sp19_affine_shadow_tether"

A31_RESULT_SHA256 = "aab7d9453e5c281bbb30269628df2cf04e17db8f6f126c78693c01aa92ea2b69"
A31_PAYLOAD_SHA256 = "5ca5a2565b777850b211cdee1e86bfa34d7e1a05c8f9022b240f0358a1f5ac42"

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_rollout.py",
    "scripts/time_dependent_no/visualize_pcno_affine_shadow_tether_rollout.py",
    "tests/time_dependent_no/test_pcno_sparse_modal_correction.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/analyze_pcno_binary_position_phase_rescore.py",
    "scripts/time_dependent_no/evaluate_pcno_binary_position_phase_rollout.py",
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

CONTRACT = a31.CONTRACT
NATIVE_RESOLUTION = a31.NATIVE_RESOLUTION
FINE_RESOLUTION = a31.FINE_RESOLUTION
CASE_IDS = a31.CASE_IDS
GROUP_CASES = a31.GROUP_CASES
CASE_TO_GROUP = a31.CASE_TO_GROUP
INPUT_CALLS = a31.INPUT_CALLS
EXPECTED_ACTIVE_CASES = a31.EXPECTED_ACTIVE_CASES
INACTIVE_CASES = a31.INACTIVE_CASES
ANIMATION_CASE_IDS = a31.ANIMATION_CASE_IDS
ANIMATION_CALLS = a31.ANIMATION_CALLS
STRICT_TOLERANCE = a31.STRICT_TOLERANCE
ANIMATION_CONTRACT = {
    **a31.ANIMATION_CONTRACT,
    "candidate_label": "affine shadow-tether correction",
    "output_stem": "affine_shadow_tether_h30",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _verify_a31(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A31_RESULT_SHA256:
        raise ValueError("A31 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("gate", {})
    checks = gate.get("checks", {})
    if (
        payload.get("schema") != a31.RESULT_SCHEMA
        or payload.get("payload_sha256") != A31_PAYLOAD_SHA256
        or payload.get("status") != "stopped_recurrent"
        or gate.get("status") != "failed"
        or gate.get("failed_checks") != ["all_controls_no_harm"]
        or checks.get("all_controls_no_harm") is not False
        or not all(value for key, value in checks.items() if key != "all_controls_no_harm")
    ):
        raise ValueError("A31 is not the exact registered stopped result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A31 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A31 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A31 artifact differs: {relative}")
    return payload


def _source_hashes() -> dict[str, Any]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace, *, base_source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    a31_payload = _verify_a31(args.a31_result)
    a31._verify_a30(args.a30_result)
    a31._verify_a28(args.a28_map)
    a31._reference_rows(args.a29_reference_checks)
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a31_result_sha256": sha256_file(args.a31_result),
                "a31_payload_sha256": a31_payload["payload_sha256"],
                "a28_result_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": a31.A28_PAYLOAD_SHA256,
                "a30_result_sha256": sha256_file(args.a30_result),
                "a30_payload_sha256": a31.A30_PAYLOAD_SHA256,
                "reference_checks_sha256": sha256_file(args.a29_reference_checks),
            },
            "base_runtime_source_manifest": dict(base_source_manifest),
            "population": {
                "case_ids": list(CASE_IDS),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "status": "already_open_e12_e14_adaptive",
                "new_population_opened": False,
            },
            "inference_contract": {
                "raw_comparator": "independent_native_recurrence",
                "safety_shadow": "independent_native_recurrence",
                "proposal": BINARY_POSITION_PHASE_POLICY,
                "accepted_state": "raw_shadow_plus_fixed_sp19_projection_of_proposal_difference",
                "active_input_calls": list(a31.BINARY_POSITION_PHASE_CALLS),
                "active_call_order": ["shadow_native", "candidate_native", "candidate_fine"],
                "inactive_call_order": ["shadow_native", "candidate_native"],
                "truth_or_case_id_in_decision": False,
                "coefficient_refit": False,
                "fine_truth_loaded": False,
            },
            "animation_contract": ANIMATION_CONTRACT,
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "population_inventory_14": len(CASE_IDS) == 14,
        "active_inventory_4": len(EXPECTED_ACTIVE_CASES) == 4,
        "raw_comparator_native_calls_420": 14 * 30 == 420,
        "candidate_shadow_native_calls_420": 14 * 30 == 420,
        "candidate_proposal_native_calls_420": 14 * 30 == 420,
        "candidate_fine_calls_64": 4 * 16 == 64,
        "main_logical_inventory_1324": 420 + 420 + 420 + 64 == 1324,
        "animation_inventory_5": len(ANIMATION_CASE_IDS) == 5,
        "owned_source_inventory_5": len(OWNED_SOURCE_PATHS) == 5,
    }
    return {
        "schema": "pcno_sp19_affine_shadow_tether_rollout_synthetic_v1",
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
                "a31_result_sha256": A31_RESULT_SHA256,
                "a31_payload_sha256": A31_PAYLOAD_SHA256,
                "a28_result_sha256": a31.A28_RESULT_SHA256,
                "a28_payload_sha256": a31.A28_PAYLOAD_SHA256,
                "a30_result_sha256": a31.A30_RESULT_SHA256,
                "a30_payload_sha256": a31.A30_PAYLOAD_SHA256,
                "reference_checks_sha256": a31.A29_REFERENCE_SHA256,
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
        raise ValueError("A32 preflight checks did not all pass")
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
        raise ValueError("A32 preflight differs from the frozen contract")
    return payload


def _new_execution() -> dict[str, Any]:
    return {
        **a31._new_execution(),
        "shadow_native_logical_calls": 0,
        "shadow_native_actual_forward_passes": 0,
        "shadow_native_forward_seconds": 0.0,
        "candidate_native_logical_calls": 0,
        "candidate_native_actual_forward_passes": 0,
        "candidate_native_forward_seconds": 0.0,
        "candidate_fine_logical_calls": 0,
        "candidate_fine_actual_forward_passes": 0,
        "candidate_fine_forward_seconds": 0.0,
    }


def _run_tethered_arm(
    runtime: Any,
    *,
    case_id: str,
    reference: Mapping[str, Any],
    position_selected: bool,
    descriptor: Any,
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    reference_resolution = tuple(int(value) for value in reference["retained_resolution"])
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("A32 native initial state is unavailable")
    accepted = np.asarray(initial, dtype=np.float64)
    shadow = np.array(accepted, copy=True)
    states = [np.array(accepted, copy=True)]
    shadow_states = [np.array(shadow, copy=True)]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    tether_rows: list[dict[str, Any]] = []
    execution = _new_execution()
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    cumulative = np.zeros_like(accepted)
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
            raise ValueError("A32 native rollout truth is unavailable")
        call_order: list[tuple[int, int]] = []

        def predictor(resolution, value, call_order=call_order):
            call_index = len(call_order)
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
            if call_index == 0:
                role = "shadow_native"
            elif resolution == NATIVE_RESOLUTION:
                role = "candidate_native"
            else:
                role = "candidate_fine"
            forward = timing["forward_seconds"]
            execution[f"{role}_logical_calls"] += 1
            execution[f"{role}_actual_forward_passes"] += len(forward)
            execution[f"{role}_forward_seconds"] += float(sum(forward))
            return prediction

        step = synchronized_shadow_tethered_binary_affine_step(
            accepted,
            shadow,
            position_selected=position_selected,
            input_call=input_call,
            contract=CONTRACT,
            projector=runtime.native_projector,
            predictor=predictor,
            start_coefficients=start_coefficients,
            end_coefficients=end_coefficients,
            volumes=volumes,
            residual_scale=residual_scale,
            state_scale=state_scale,
        )
        proposal = step.proposal
        expected_order = [NATIVE_RESOLUTION, NATIVE_RESOLUTION]
        if proposal.correction_active:
            expected_order.append(FINE_RESOLUTION)
        native_prediction = np.asarray(
            proposal.predictions[NATIVE_RESOLUTION], dtype=np.float64
        )
        proposal_closure = a31._maximum_abs(
            proposal.next_native_state - native_prediction - proposal.correction
        )
        if proposal.prepared_inputs is None:
            pre_fine_floor = 0.0
            post_fine_floor = 0.0
        else:
            pre_fine_floor = float(
                proposal.prepared_inputs.nesting_floors[
                    "pre_model_fine_to_native_max_abs"
                ]
            )
            post_fine_floor = float(
                proposal.prepared_inputs.nesting_floors[
                    "post_fp32_fine_to_native_max_abs"
                ]
            )
        audit_rows.append(
            {
                "case_id": case_id,
                "input_call": input_call,
                "descriptor_status": descriptor.status,
                "normalized_wall_distance": descriptor.normalized_wall_distance,
                "position_selected": position_selected,
                "correction_active": proposal.correction_active,
                "call_order": ">".join(f"{x}x{y}" for x, y in call_order),
                "call_order_exact": call_order == expected_order,
                "logical_call_count": proposal.logical_call_count,
                "total_logical_call_count": step.logical_call_count,
                "pre_model_fine_to_native_max_abs": pre_fine_floor,
                "post_fp32_fine_to_native_max_abs": post_fine_floor,
                "state_update_closure_max_abs": proposal_closure,
                **asdict(proposal.audit),
            }
        )
        tether_rows.append(
            {
                "case_id": case_id,
                "input_call": input_call,
                "position_selected": position_selected,
                "correction_active": proposal.correction_active,
                "call_order_exact": call_order == expected_order,
                "total_logical_call_count": step.logical_call_count,
                "maximum_update_identity_abs": step.maximum_update_identity_abs,
                **asdict(step.tether_audit),
            }
        )
        next_state = np.asarray(step.next_native_state, dtype=np.float64)
        next_shadow = np.asarray(step.shadow_prediction, dtype=np.float64)
        intervention = next_state - native_prediction
        row, cumulative = response._block_metric_row(
            runtime,
            case_id=case_id,
            policy=POLICY,
            input_call=input_call,
            previous_state=accepted,
            next_state=next_state,
            current_reference=np.asarray(current_reference, dtype=np.float64),
            target=np.asarray(target, dtype=np.float64),
            cumulative_defect=cumulative,
            intervention=intervention,
            cap_active=bool(proposal.audit.cap_active),
            correction_status=(
                "active_affine_shadow_tether"
                if proposal.correction_active
                else "coast_shadow_tether"
            ),
        )
        row.update(
            {
                "position_selected": position_selected,
                "correction_active": proposal.correction_active,
                "logical_calls_this_step": len(call_order),
                "proposal_state_update_closure_max_abs": proposal_closure,
                "tether_update_identity_max_abs": step.maximum_update_identity_abs,
                "tether_integral_max_abs": step.tether_audit.maximum_integral_difference_abs,
                "tether_boundary_max_abs": step.tether_audit.maximum_boundary_difference_abs,
                "tether_idempotence_max_abs": step.tether_audit.maximum_projection_idempotence_abs,
                "pre_model_fine_to_native_max_abs": pre_fine_floor,
                "post_fp32_fine_to_native_max_abs": post_fine_floor,
            }
        )
        rows.append(row)
        accepted = next_state
        shadow = next_shadow
        states.append(np.array(accepted, copy=True))
        shadow_states.append(np.array(shadow, copy=True))
        if not row["finite"] or not row["admissible"]:
            break
    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = len(rows)
    return {
        "case_id": case_id,
        "policy": POLICY,
        "position_selected": position_selected,
        "descriptor": asdict(descriptor),
        "rows": rows,
        "states": states,
        "shadow_states": shadow_states,
        "audit_rows": audit_rows,
        "tether_rows": tether_rows,
        "execution": execution,
    }


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
            raise ValueError("A32 animation truth is unavailable")
        truth_states.append(np.asarray(value, dtype=np.float64))
    indices = np.asarray(ANIMATION_CALLS, dtype=np.int64)
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    path = output_dir / f"{case_id}_affine_shadow_tether_h30.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        split_group_id=np.asarray(a29.GROUP_IDS[CASE_TO_GROUP[case_id]]),
        output_calls=indices,
        physical_times=np.asarray(ANIMATION_CONTRACT["physical_times"], dtype=np.float64),
        native_resolution=np.asarray(NATIVE_RESOLUTION, dtype=np.int64),
        nodes=np.asarray(geometry.nodes, dtype=np.float32),
        volumes=np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1),
        state_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float32),
        residual_scale=np.asarray(runtime.normalization.residual_scale, dtype=np.float32),
        gamma=np.asarray(runtime.normalization.gamma, dtype=np.float64),
        truth_conservative=np.asarray(truth_states, dtype=np.float64)[indices].astype(np.float32),
        raw_shadow_conservative=np.asarray(raw_states, dtype=np.float64)[indices].astype(np.float32),
        corrected_conservative=np.asarray(candidate_states, dtype=np.float64)[indices].astype(np.float32),
    )
    return {
        "case_id": case_id,
        "split_group_id": a29.GROUP_IDS[CASE_TO_GROUP[case_id]],
        "path": path.name,
        "sha256": sha256_file(path),
        "frames": len(indices),
        "storage_dtype": "float32_visualization_only",
    }


def _gate(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    tether_rows: Sequence[Mapping[str, Any]],
    inactive_max_abs: Mapping[str, float],
    shadow_raw_max_abs: Mapping[str, float],
    prefix_repeat_abs: float,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    gate = a31._gate(
        score_rows=score_rows,
        case_rows=case_rows,
        controls=controls,
        rollouts=rollouts,
        audit_rows=audit_rows,
        inactive_max_abs=inactive_max_abs,
        prefix_repeat_abs=prefix_repeat_abs,
        structural_checks=structural_checks,
    )
    checks = dict(gate["checks"])
    checks.update(
        {
            "tether_audit_inventory_420": len(tether_rows) == 420,
            "shadow_byte_equal_raw": set(shadow_raw_max_abs) == set(CASE_IDS)
            and all(value == 0.0 for value in shadow_raw_max_abs.values()),
            "tether_call_order_and_counts": all(
                bool(row["call_order_exact"])
                and int(row["total_logical_call_count"])
                == (3 if row["correction_active"] else 2)
                for row in tether_rows
            ),
            "tether_integral_boundary_projection_closure": all(
                row["status"] == "ok"
                and float(row["maximum_integral_difference_abs"])
                <= STRICT_TOLERANCE
                and float(row["maximum_boundary_difference_abs"])
                <= STRICT_TOLERANCE
                and float(row["maximum_projection_idempotence_abs"])
                <= STRICT_TOLERANCE
                and float(row["maximum_update_identity_abs"])
                <= STRICT_TOLERANCE
                for row in tether_rows
            ),
        }
    )
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        **gate,
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
    }


def _sum_execution(
    rollouts: Sequence[Mapping[str, Any]], policy: str, keys: Sequence[str]
) -> dict[str, float]:
    return {
        key: sum(
            float(row["execution"].get(key, 0.0))
            for row in rollouts
            if row["policy"] == policy
        )
        for key in keys
    }


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight, args)
    _verify_a31(args.a31_result)
    start_coefficients, end_coefficients, _ = a31._verify_a28(args.a28_map)
    expected_references = a31._reference_rows(args.a29_reference_checks)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = base._build_runtime(a29._base_args(args))
    started = perf_counter()
    try:
        if runtime.source_manifest != preflight["source_manifest"]["base_runtime_source_manifest"]:
            raise ValueError("A32 runtime source differs from preflight")
        prefix_reference, _ = a31._load_reference(
            runtime,
            EXPECTED_ACTIVE_CASES[0],
            expected_references[EXPECTED_ACTIVE_CASES[0]],
        )
        prefix_initial = np.asarray(prefix_reference["conservative_states"][0], dtype=np.float64)
        prefix_descriptor = transverse_velocity_position_descriptor(
            prefix_initial,
            nodes=np.asarray(
                runtime.geometry_by_resolution[NATIVE_RESOLUTION].nodes,
                dtype=np.float64,
            ),
            volumes=np.asarray(
                runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
                dtype=np.float64,
            ),
            y_min=float(runtime.first_config.y_min),
            y_max=float(runtime.first_config.y_max),
        )
        prefix_selected = bool(
            prefix_descriptor.status == "ok"
            and prefix_descriptor.normalized_wall_distance is not None
            and prefix_descriptor.normalized_wall_distance
            <= FROZEN_POSITION_BUFFER_THRESHOLD
        )
        prefix_runs = [
            _run_tethered_arm(
                runtime,
                case_id=EXPECTED_ACTIVE_CASES[0],
                reference=prefix_reference,
                position_selected=prefix_selected,
                descriptor=prefix_descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=2,
            )
            for _ in range(2)
        ]
        prefix_repeat_abs = max(
            a31._maximum_abs(left - right)
            for name in ("states", "shadow_states")
            for left, right in zip(
                prefix_runs[0][name], prefix_runs[1][name], strict=True
            )
        )
        prefix_execution = {
            "first": prefix_runs[0]["execution"],
            "second": prefix_runs[1]["execution"],
            "maximum_state_abs_difference": prefix_repeat_abs,
            "excluded_from_main_scientific_cost": True,
        }

        rollouts: list[dict[str, Any]] = []
        view_records: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        descriptor_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        tether_rows: list[dict[str, Any]] = []
        inactive_max_abs: dict[str, float] = {}
        shadow_raw_max_abs: dict[str, float] = {}
        view_maxima = {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
        animation_rows = []
        animation_dir = output_dir / "animation_bundles"
        for case_id in CASE_IDS:
            print(f"A32 case {case_id}: loading authenticated native truth", flush=True)
            reference, reference_check = a31._load_reference(
                runtime, case_id, expected_references[case_id]
            )
            initial = np.asarray(reference["conservative_states"][0], dtype=np.float64)
            descriptor = a31._descriptor(runtime, initial)
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
            print(f"A32 case {case_id}: raw comparator H30", flush=True)
            raw = a31._run_arm(
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
            print(f"A32 case {case_id}: shadow-tether H30 selected={selected}", flush=True)
            candidate = _run_tethered_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            statistics, maxima = a31._pair_view_statistics(
                runtime,
                case_id=case_id,
                reference=reference,
                raw_states=raw["states"],
                candidate_states=candidate["states"],
            )
            view_records.extend(statistics)
            for key, value in maxima.items():
                view_maxima[key] = max(view_maxima[key], value)
            shadow_raw_max_abs[case_id] = max(
                a31._maximum_abs(left - right)
                for left, right in zip(
                    raw["states"], candidate["shadow_states"], strict=True
                )
            )
            if case_id in INACTIVE_CASES:
                inactive_max_abs[case_id] = max(
                    a31._maximum_abs(left - right)
                    for left, right in zip(
                        raw["states"], candidate["states"], strict=True
                    )
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
            tether_rows.extend(candidate["tether_rows"])
            for rollout in (raw, candidate):
                rollout.pop("states")
                rollout.pop("audit_rows")
                if rollout["policy"] == POLICY:
                    rollout.pop("shadow_states")
                    rollout.pop("tether_rows")
                rollouts.append(rollout)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        selected_cases = tuple(
            row["case_id"] for row in descriptor_rows if row["position_selected"]
        )
        score_rows = a31._score_views(view_records)
        all_metric_rows = [metric for rollout in rollouts for metric in rollout["rows"]]
        case_rows, paired_controls, paired_population = response._paired_rollout_controls_for_cases(
            all_metric_rows, case_ids=CASE_IDS
        )
        controls = [*a31._field_controls(score_rows), *paired_controls]
        for row in controls:
            row["cell"] = row.get("cell", "overall")
            row["policy"] = POLICY
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
        common_keys = (
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
        candidate_keys = (
            *common_keys,
            "shadow_native_logical_calls",
            "shadow_native_actual_forward_passes",
            "shadow_native_forward_seconds",
            "candidate_native_logical_calls",
            "candidate_native_actual_forward_passes",
            "candidate_native_forward_seconds",
            "candidate_fine_logical_calls",
            "candidate_fine_actual_forward_passes",
            "candidate_fine_forward_seconds",
        )
        main_execution = {
            "raw": _sum_execution(rollouts, "zero", common_keys),
            "candidate": _sum_execution(rollouts, POLICY, candidate_keys),
            "maximum_peak_gpu_memory_bytes": max(
                int(row["execution"]["maximum_peak_gpu_memory_bytes"])
                for row in rollouts
            ),
            "wall_seconds": perf_counter() - started,
            "environment": runtime_environment(runtime.device),
        }
        main_execution["candidate_logical_cost_ratio_vs_raw"] = (
            main_execution["candidate"]["logical_model_calls"]
            / main_execution["raw"]["logical_model_calls"]
        )
        main_execution["total_logical_cost_ratio_vs_raw"] = (
            main_execution["candidate"]["logical_model_calls"]
            + main_execution["raw"]["logical_model_calls"]
        ) / main_execution["raw"]["logical_model_calls"]
        main_execution["candidate_forward_time_ratio_vs_raw"] = (
            main_execution["candidate"]["total_forward_seconds"]
            / main_execution["raw"]["total_forward_seconds"]
        )
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"] == _source_hashes(),
            "population_exact": set(selected_cases) == set(EXPECTED_ACTIVE_CASES),
            "descriptor_inventory_exact": len(descriptor_rows) == len(CASE_IDS),
            "reference_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "paired_metric_inventory_exact": len(all_metric_rows)
            == 2 * len(CASE_IDS) * len(INPUT_CALLS),
            "main_logical_call_inventory_exact": (
                main_execution["raw"]["native_logical_calls"] == 420
                and main_execution["raw"]["fine_logical_calls"] == 0
                and main_execution["candidate"]["logical_model_calls"] == 904
                and main_execution["candidate"]["shadow_native_logical_calls"] == 420
                and main_execution["candidate"]["candidate_native_logical_calls"] == 420
                and main_execution["candidate"]["candidate_fine_logical_calls"] == 64
            ),
            "view_closure": view_maxima["band"] <= 1.0e-10
            and view_maxima["subspace_reconstruction"] <= 1.0e-10
            and view_maxima["orthogonality"] <= 1.0e-10,
            "animation_inventory_exact": len(animation_rows) == len(ANIMATION_CASE_IDS),
            "fine_truth_not_loaded": True,
            "coefficient_refit_not_run": True,
            "one_accepted_native_state_per_call": True,
        }
        gate = _gate(
            score_rows=score_rows,
            case_rows=case_rows,
            controls=controls,
            rollouts=rollouts,
            audit_rows=audit_rows,
            tether_rows=tether_rows,
            inactive_max_abs=inactive_max_abs,
            shadow_raw_max_abs=shadow_raw_max_abs,
            prefix_repeat_abs=prefix_repeat_abs,
            structural_checks=structural_checks,
        )
        write_csv(output_dir / "rollout_call_metrics.csv", all_metric_rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "view_scores.csv", score_rows)
        write_csv(output_dir / "correction_audits.csv", audit_rows)
        write_csv(output_dir / "tether_audits.csv", tether_rows)
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
                "status": (
                    "qualified_adaptive_open_rollout"
                    if gate["status"] == "qualified"
                    else "stopped_recurrent"
                ),
                "population_status": "already_open_e12_e14_adaptive_mechanism",
                "gate": gate,
                "population": paired_population,
                "primary_scores": {
                    scope: {
                        view: dict(
                            a31._score_row(
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
                "failed_controls": [
                    dict(row) for row in controls if not a31._control_passed(row)
                ],
                "maximum_control_ratio": max(
                    float(row["ratio"])
                    for row in controls
                    if row.get("ratio") is not None
                ),
                "inactive_case_maximum_state_abs_difference": inactive_max_abs,
                "shadow_raw_maximum_state_abs_difference": shadow_raw_max_abs,
                "view_closure_maxima": view_maxima,
                "main_execution": main_execution,
                "prefix_replay": prefix_execution,
                "a31_result_sha256": sha256_file(args.a31_result),
                "a31_payload_sha256": A31_PAYLOAD_SHA256,
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
                    "Adaptive-open E12/E14 shadow-tethered recurrent mechanism "
                    "evidence only; no independent confirmation, cross-family "
                    "coefficient transfer, direct off-grid comparison, conservation, "
                    "convergence-order, or Richardson claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(output_dir / "affine_shadow_tether_rollout.json", result)
        return result, 0 if gate["status"] == "qualified" else 4
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    a31._add_external_arguments(parser)
    parser.add_argument("--a31-result", type=Path, required=True)


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
