#!/usr/bin/env python3
"""Replay A32 with the frozen A42 one-native surrogate only during coast."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    analyze_pcno_shadow_response_coast_geometry as a42,
)
from scripts.time_dependent_no import benchmark_pcno_paired_native_batch as a40
from scripts.time_dependent_no import evaluate_pcno_affine_shadow_tether_rollout as a32
from scripts.time_dependent_no import evaluate_pcno_binary_position_phase_rollout as a31
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
    as_model_state,
    predict_resolution_sample,
    reference_at_resolution,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    projected_shadow_tether,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    synchronized_shadow_tethered_binary_affine_step,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A43-PHASE-SELECTIVE-COAST-REPLAY"
PREFLIGHT_SCHEMA = "pcno_phase_selective_coast_replay_preflight_v1"
RESULT_SCHEMA = "pcno_phase_selective_coast_replay_v1"
SOURCE_SCHEMA = "pcno_phase_selective_coast_replay_source_v1"
ANIMATION_SCHEMA = "pcno_phase_selective_coast_replay_animation_bundles_v1"
POLICY = "phase_selective_one_native_coast_shadow_tether"

A32_RESULT_SHA256 = a40.A32_RESULT_SHA256
A32_PAYLOAD_SHA256 = a40.A32_PAYLOAD_SHA256
A42_RESULT_SHA256 = "eb7ecc4e5546c0749c620a1091ef8ccaed29fd5a86d7ba9d7c54e000f7b9ef05"
A42_PAYLOAD_SHA256 = "438582404fd6d8525b482301407dec47259b86f957548ef9f34aaab288ed30d5"
A32_ACTIVE_FULL = 0.9792855041700079
A32_ACTIVE_RANK8 = 0.9328702171441363
A32_ACTIVE_ENDPOINT = 0.9627016536922104
A32_CASE_FULL = {
    "sv_e12_y01": 0.9759675250638704,
    "sv_e12_y07": 0.9869167146391814,
    "sv_e14_y01": 0.9721483662791086,
    "sv_e14_y07": 0.9872315394865617,
}
CASE_RETENTION_ALLOWANCE = 0.005
OPTIMIZED_A32_CALLS = 604
CANDIDATE_CALLS = 548
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

CONTRACT = a32.CONTRACT
NATIVE_RESOLUTION = a32.NATIVE_RESOLUTION
FINE_RESOLUTION = a32.FINE_RESOLUTION
CASE_IDS = a32.CASE_IDS
GROUP_CASES = a32.GROUP_CASES
CASE_TO_GROUP = a32.CASE_TO_GROUP
INPUT_CALLS = a32.INPUT_CALLS
EXPECTED_ACTIVE_CASES = a32.EXPECTED_ACTIVE_CASES
INACTIVE_CASES = a32.INACTIVE_CASES
ANIMATION_CASE_IDS = a32.ANIMATION_CASE_IDS
ANIMATION_CALLS = a32.ANIMATION_CALLS
COAST_CALLS = a42.COAST_CALLS
STRICT_TOLERANCE = a32.STRICT_TOLERANCE
CONTROL_LIMIT = a31.CONTROL_LIMIT
FROZEN_DIAGONAL = a42.FROZEN_DIAGONAL
ANIMATION_CONTRACT = {
    **a31.ANIMATION_CONTRACT,
    "candidate_label": "phase-selective one-native coast correction",
    "output_stem": "phase_selective_coast_h30",
}

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_phase_selective_coast_replay.py",
    "scripts/time_dependent_no/visualize_pcno_phase_selective_coast_replay.py",
    "tests/time_dependent_no/test_pcno_phase_selective_coast_replay.py",
)
DEPENDENCY_PATHS = (
    "scripts/time_dependent_no/analyze_pcno_shadow_response_coast_geometry.py",
    "scripts/time_dependent_no/analyze_pcno_shadow_response_geometry.py",
    "scripts/time_dependent_no/benchmark_pcno_paired_native_batch.py",
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_binary_position_phase_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_fine_discrepancy_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_modal_affine_transfer.py",
    "scripts/time_dependent_no/evaluate_pcno_response_filtered_block.py",
    "scripts/time_dependent_no/visualize_pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/pcno_cross_resolution_correction.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "utility/time_dependent_no/pcno_fine_discrepancy_correction.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
)


@dataclass(frozen=True)
class PhaseSelectiveStep:
    route: str
    shadow_prediction: np.ndarray
    accepted_prediction: np.ndarray
    next_native_state: np.ndarray
    retained_displacement: np.ndarray
    metric_intervention: np.ndarray
    correction: np.ndarray
    correction_active: bool
    cap_active: bool
    logical_call_count: int
    pre_model_fine_to_native_max_abs: float
    post_fp32_fine_to_native_max_abs: float
    correction_to_native_increment: float
    maximum_scaled_component_mean_abs: float
    maximum_excluded_abs: float
    maximum_inactive_coordinate_abs: float
    maximum_modal_reconstruction_abs: float
    maximum_proposal_update_abs: float
    maximum_integral_difference_abs: float
    maximum_boundary_difference_abs: float
    maximum_projection_idempotence_abs: float
    maximum_update_identity_abs: float


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _verify_a32(path: Path) -> dict[str, Any]:
    return a40._verify_a32_result(path)


def _verify_a42(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A42_RESULT_SHA256:
        raise ValueError("A42 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != a42.RESULT_SCHEMA
        or payload.get("working_id") != a42.WORKING_ID
        or payload.get("payload_sha256") != A42_PAYLOAD_SHA256
        or payload.get("status") != "qualified_coast_response_geometry"
        or payload.get("gate", {}).get("status") != "qualified"
        or payload.get("gate", {}).get("failed_checks") != []
        or not all(payload.get("gate", {}).get("checks", {}).values())
        or payload.get("truth_arrays_loaded") is not False
        or payload.get("fine_predictions_executed") is not False
        or payload.get("surrogate_recurrence_executed") is not False
        or payload.get("h30_rollout_executed") is not False
        or payload.get("frozen_diagonal") != list(FROZEN_DIAGONAL)
    ):
        raise ValueError("A42 is not the exact qualified coast response result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A42 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A42 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A42 artifact differs: {relative}")
    return payload


def _source_hashes() -> dict[str, Any]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace, *, base_source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    a32_payload = _verify_a32(args.a32_result)
    a42_payload = _verify_a42(args.a42_result)
    a31._verify_a30(args.a30_result)
    a31._verify_a28(args.a28_map)
    a31._reference_rows(args.a29_reference_checks)
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a32_result_sha256": sha256_file(args.a32_result),
                "a32_payload_sha256": a32_payload["payload_sha256"],
                "a42_result_sha256": sha256_file(args.a42_result),
                "a42_payload_sha256": a42_payload["payload_sha256"],
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
                "inactive_route": "single_accepted_raw_native_recurrence",
                "exact_active_calls": [
                    call for call in INPUT_CALLS if call not in COAST_CALLS
                ],
                "coast_calls": list(COAST_CALLS),
                "exact_active_call_order": [
                    "shadow_native",
                    "accepted_native",
                    "accepted_fine",
                ],
                "coast_call_order": ["shadow_native"],
                "frozen_response_diagonal": list(FROZEN_DIAGONAL),
                "candidate_logical_calls": CANDIDATE_CALLS,
                "optimized_a32_logical_calls": OPTIMIZED_A32_CALLS,
                "raw_logical_calls": 420,
                "truth_or_case_id_in_decision": False,
                "coefficient_refit": False,
                "one_accepted_native_state_per_call": True,
                "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            },
            "animation_contract": ANIMATION_CONTRACT,
        }
    )


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "population_inventory_14": len(CASE_IDS) == 14,
        "active_inventory_4": len(EXPECTED_ACTIVE_CASES) == 4,
        "coast_inventory_14": COAST_CALLS == tuple(range(8, 22)),
        "inactive_native_calls_300": len(INACTIVE_CASES) * 30 == 300,
        "active_shadow_native_calls_120": len(EXPECTED_ACTIVE_CASES) * 30 == 120,
        "active_accepted_native_calls_64": len(EXPECTED_ACTIVE_CASES) * 16 == 64,
        "active_fine_calls_64": len(EXPECTED_ACTIVE_CASES) * 16 == 64,
        "candidate_logical_calls_548": 300 + 120 + 64 + 64 == CANDIDATE_CALLS,
        "candidate_below_optimized_a32": CANDIDATE_CALLS < OPTIMIZED_A32_CALLS,
        "animation_inventory_5": len(ANIMATION_CASE_IDS) == 5,
        "owned_source_inventory_4": len(OWNED_SOURCE_PATHS) == 4,
    }
    return {
        "schema": "pcno_phase_selective_coast_replay_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, base_source = base._open_contract(
        a29._base_args(args)
    )
    try:
        provenance = {
            case_id: family_case_provenance(manifest, case_id) for case_id in CASE_IDS
        }
        source = _source_manifest(args, base_source_manifest=base_source)
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "lineage_exact": source["lineage"]
            == {
                "a32_result_sha256": A32_RESULT_SHA256,
                "a32_payload_sha256": A32_PAYLOAD_SHA256,
                "a42_result_sha256": A42_RESULT_SHA256,
                "a42_payload_sha256": A42_PAYLOAD_SHA256,
                "a28_result_sha256": a31.A28_RESULT_SHA256,
                "a28_payload_sha256": a31.A28_PAYLOAD_SHA256,
                "a30_result_sha256": a31.A30_RESULT_SHA256,
                "a30_payload_sha256": a31.A30_PAYLOAD_SHA256,
                "reference_checks_sha256": a31.A29_REFERENCE_SHA256,
            },
            "case_inventory_present": all(
                case_id in store.keys for case_id in CASE_IDS
            ),
            "population_exact": set(provenance) == set(CASE_IDS),
            "groups_exact": all(
                provenance[case_id].get("split") == "test"
                and provenance[case_id].get("split_group_id") == a29.GROUP_IDS[group]
                for group, cases in GROUP_CASES.items()
                for case_id in cases
            ),
            "checkpoint_stride_exact": base.checkpoint_step_stride(checkpoint) == 2,
            "route_inventory_exact": source["inference_contract"][
                "candidate_logical_calls"
            ]
            == CANDIDATE_CALLS
            and source["inference_contract"]["coast_calls"] == list(COAST_CALLS),
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
        raise ValueError("A43 preflight checks did not all pass")
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
        raise ValueError("A43 preflight differs from the frozen contract")
    return payload


def phase_selective_active_step(
    accepted_native_state: np.ndarray,
    shadow_native_state: np.ndarray,
    *,
    input_call: int,
    projector: Any,
    predictor: Callable[[tuple[int, int], np.ndarray], np.ndarray],
    start_coefficients: np.ndarray,
    end_coefficients: np.ndarray,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    state_scale: np.ndarray,
) -> PhaseSelectiveStep:
    if input_call not in INPUT_CALLS:
        raise ValueError("A43 input call is outside 0..29")
    if input_call in COAST_CALLS:
        shadow_input = as_model_state(shadow_native_state)
        shadow_prediction = np.asarray(
            predictor(NATIVE_RESOLUTION, shadow_input), dtype=np.float64
        )
        runtime_view = SimpleNamespace(
            native_projector=projector,
            normalization=SimpleNamespace(state_scale=state_scale),
        )
        x = a42.a41._active_coordinates(
            np.asarray(accepted_native_state, dtype=np.float64)
            - np.asarray(shadow_native_state, dtype=np.float64),
            runtime_view,
        )
        z_hat = np.asarray(FROZEN_DIAGONAL, dtype=np.float64) * x
        predicted_displacement = a42.a41._field_from_active(z_hat, runtime_view)
        accepted_prediction = shadow_prediction + predicted_displacement
        next_state, retained, tether_audit = projected_shadow_tether(
            accepted_prediction,
            shadow_prediction,
            projector=projector,
            volumes=volumes,
            component_scale=state_scale,
        )
        zero = np.zeros_like(next_state)
        return PhaseSelectiveStep(
            route="surrogate_coast",
            shadow_prediction=np.array(shadow_prediction, copy=True),
            accepted_prediction=np.array(accepted_prediction, copy=True),
            next_native_state=np.array(next_state, copy=True),
            retained_displacement=np.array(retained, copy=True),
            metric_intervention=np.array(retained, copy=True),
            correction=zero,
            correction_active=False,
            cap_active=False,
            logical_call_count=1,
            pre_model_fine_to_native_max_abs=0.0,
            post_fp32_fine_to_native_max_abs=0.0,
            correction_to_native_increment=0.0,
            maximum_scaled_component_mean_abs=0.0,
            maximum_excluded_abs=0.0,
            maximum_inactive_coordinate_abs=0.0,
            maximum_modal_reconstruction_abs=0.0,
            maximum_proposal_update_abs=_maximum_abs(next_state - accepted_prediction),
            maximum_integral_difference_abs=(
                tether_audit.maximum_integral_difference_abs
            ),
            maximum_boundary_difference_abs=(
                tether_audit.maximum_boundary_difference_abs
            ),
            maximum_projection_idempotence_abs=(
                tether_audit.maximum_projection_idempotence_abs
            ),
            maximum_update_identity_abs=_maximum_abs(
                next_state - shadow_prediction - retained
            ),
        )

    exact = synchronized_shadow_tethered_binary_affine_step(
        accepted_native_state,
        shadow_native_state,
        position_selected=True,
        input_call=input_call,
        contract=CONTRACT,
        projector=projector,
        predictor=predictor,
        start_coefficients=start_coefficients,
        end_coefficients=end_coefficients,
        volumes=volumes,
        residual_scale=residual_scale,
        state_scale=state_scale,
    )
    proposal = exact.proposal
    if not proposal.correction_active or proposal.prepared_inputs is None:
        raise ValueError("A43 exact window did not activate the A32 fine correction")
    correction_audit = proposal.audit
    floors = proposal.prepared_inputs.nesting_floors
    accepted_prediction = np.asarray(
        proposal.predictions[NATIVE_RESOLUTION], dtype=np.float64
    )
    return PhaseSelectiveStep(
        route="exact_a32_window",
        shadow_prediction=np.array(exact.shadow_prediction, copy=True),
        accepted_prediction=np.array(accepted_prediction, copy=True),
        next_native_state=np.array(exact.next_native_state, copy=True),
        retained_displacement=np.array(exact.retained_displacement, copy=True),
        metric_intervention=np.array(
            exact.next_native_state - accepted_prediction, copy=True
        ),
        correction=np.array(proposal.correction, copy=True),
        correction_active=True,
        cap_active=bool(correction_audit.cap_active),
        logical_call_count=exact.logical_call_count,
        pre_model_fine_to_native_max_abs=float(
            floors["pre_model_fine_to_native_max_abs"]
        ),
        post_fp32_fine_to_native_max_abs=float(
            floors["post_fp32_fine_to_native_max_abs"]
        ),
        correction_to_native_increment=float(
            correction_audit.correction_to_native_increment
        ),
        maximum_scaled_component_mean_abs=float(
            correction_audit.maximum_scaled_component_mean_abs
        ),
        maximum_excluded_abs=float(correction_audit.maximum_excluded_abs),
        maximum_inactive_coordinate_abs=float(
            correction_audit.maximum_inactive_coordinate_abs
        ),
        maximum_modal_reconstruction_abs=float(
            correction_audit.maximum_modal_reconstruction_abs
        ),
        maximum_proposal_update_abs=_maximum_abs(
            proposal.next_native_state - accepted_prediction - proposal.correction
        ),
        maximum_integral_difference_abs=(
            exact.tether_audit.maximum_integral_difference_abs
        ),
        maximum_boundary_difference_abs=(
            exact.tether_audit.maximum_boundary_difference_abs
        ),
        maximum_projection_idempotence_abs=(
            exact.tether_audit.maximum_projection_idempotence_abs
        ),
        maximum_update_identity_abs=exact.maximum_update_identity_abs,
    )


def _new_execution() -> dict[str, Any]:
    return {
        **a31._new_execution(),
        "inactive_raw_native_logical_calls": 0,
        "inactive_raw_native_actual_forward_passes": 0,
        "inactive_raw_native_forward_seconds": 0.0,
        "shadow_native_logical_calls": 0,
        "shadow_native_actual_forward_passes": 0,
        "shadow_native_forward_seconds": 0.0,
        "candidate_native_logical_calls": 0,
        "candidate_native_actual_forward_passes": 0,
        "candidate_native_forward_seconds": 0.0,
        "candidate_fine_logical_calls": 0,
        "candidate_fine_actual_forward_passes": 0,
        "candidate_fine_forward_seconds": 0.0,
        "inactive_raw_steps": 0,
        "exact_a32_steps": 0,
        "surrogate_coast_steps": 0,
    }


def _account_role(
    execution: dict[str, Any], timing: Mapping[str, Any], *, role: str
) -> None:
    forward = timing["forward_seconds"]
    execution[f"{role}_logical_calls"] += 1
    execution[f"{role}_actual_forward_passes"] += len(forward)
    execution[f"{role}_forward_seconds"] += float(sum(forward))


def _run_candidate_arm(
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
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("A43 native initial state is unavailable")
    accepted = np.asarray(initial, dtype=np.float64)
    shadow = np.array(accepted, copy=True)
    states = [np.array(accepted, copy=True)]
    shadow_states = [np.array(shadow, copy=True)]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
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
            raise ValueError("A43 native rollout truth is unavailable")
        call_order: list[tuple[int, int]] = []
        route = (
            "inactive_raw"
            if not position_selected
            else (
                "surrogate_coast" if input_call in COAST_CALLS else "exact_a32_window"
            )
        )

        def predictor(resolution, value, call_order=call_order, route=route):
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
            if route == "inactive_raw":
                role = "inactive_raw_native"
            elif call_index == 0:
                role = "shadow_native"
            elif resolution == NATIVE_RESOLUTION:
                role = "candidate_native"
            else:
                role = "candidate_fine"
            _account_role(execution, timing, role=role)
            return prediction

        if route == "inactive_raw":
            next_state = np.asarray(
                predictor(NATIVE_RESOLUTION, accepted), dtype=np.float64
            )
            next_shadow = np.array(next_state, copy=True)
            intervention = np.zeros_like(next_state)
            retained = np.zeros_like(next_state)
            correction_active = False
            cap_active = False
            expected_order = [NATIVE_RESOLUTION]
            audit = {
                "pre_model_fine_to_native_max_abs": 0.0,
                "post_fp32_fine_to_native_max_abs": 0.0,
                "correction_to_native_increment": 0.0,
                "maximum_scaled_component_mean_abs": 0.0,
                "maximum_excluded_abs": 0.0,
                "maximum_inactive_coordinate_abs": 0.0,
                "maximum_modal_reconstruction_abs": 0.0,
                "maximum_proposal_update_abs": 0.0,
                "maximum_integral_difference_abs": 0.0,
                "maximum_boundary_difference_abs": 0.0,
                "maximum_projection_idempotence_abs": 0.0,
                "maximum_update_identity_abs": 0.0,
            }
            logical_count = 1
            execution["inactive_raw_steps"] += 1
        else:
            step = phase_selective_active_step(
                accepted,
                shadow,
                input_call=input_call,
                projector=runtime.native_projector,
                predictor=predictor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                volumes=volumes,
                residual_scale=residual_scale,
                state_scale=state_scale,
            )
            next_state = np.asarray(step.next_native_state, dtype=np.float64)
            next_shadow = np.asarray(step.shadow_prediction, dtype=np.float64)
            retained = np.asarray(step.retained_displacement, dtype=np.float64)
            intervention = np.asarray(step.metric_intervention, dtype=np.float64)
            correction_active = step.correction_active
            cap_active = step.cap_active
            expected_order = (
                [NATIVE_RESOLUTION]
                if route == "surrogate_coast"
                else [NATIVE_RESOLUTION, NATIVE_RESOLUTION, FINE_RESOLUTION]
            )
            audit = {
                key: value
                for key, value in asdict(step).items()
                if key
                not in {
                    "route",
                    "shadow_prediction",
                    "accepted_prediction",
                    "next_native_state",
                    "retained_displacement",
                    "metric_intervention",
                    "correction",
                    "correction_active",
                    "cap_active",
                    "logical_call_count",
                }
            }
            logical_count = step.logical_call_count
            execution[
                "surrogate_coast_steps"
                if route == "surrogate_coast"
                else "exact_a32_steps"
            ] += 1
        audit_rows.append(
            {
                "case_id": case_id,
                "input_call": input_call,
                "descriptor_status": descriptor.status,
                "normalized_wall_distance": descriptor.normalized_wall_distance,
                "position_selected": position_selected,
                "route": route,
                "correction_active": correction_active,
                "call_order": ">".join(f"{x}x{y}" for x, y in call_order),
                "call_order_exact": call_order == expected_order,
                "logical_call_count": logical_count,
                "retained_displacement_rms": weighted_scaled_rms(
                    retained, volumes=volumes, component_scale=state_scale
                ),
                **audit,
            }
        )
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
            cap_active=cap_active,
            correction_status=route,
        )
        row.update(
            {
                "position_selected": position_selected,
                "route": route,
                "correction_active": correction_active,
                "logical_calls_this_step": len(call_order),
                "retained_displacement_rms": audit_rows[-1][
                    "retained_displacement_rms"
                ],
                "pre_model_fine_to_native_max_abs": audit[
                    "pre_model_fine_to_native_max_abs"
                ],
                "post_fp32_fine_to_native_max_abs": audit[
                    "post_fp32_fine_to_native_max_abs"
                ],
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
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    truth_states = []
    for output_call in range(31):
        value = reference_at_resolution(
            reference["conservative_states"][2 * output_call],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if value is None:
            raise ValueError("A43 animation truth is unavailable")
        truth_states.append(np.asarray(value, dtype=np.float64))
    indices = np.asarray(ANIMATION_CALLS, dtype=np.int64)
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    path = output_dir / f"{case_id}_phase_selective_coast_h30.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        split_group_id=np.asarray(a29.GROUP_IDS[CASE_TO_GROUP[case_id]]),
        output_calls=indices,
        physical_times=np.asarray(
            ANIMATION_CONTRACT["physical_times"], dtype=np.float64
        ),
        native_resolution=np.asarray(NATIVE_RESOLUTION, dtype=np.int64),
        nodes=np.asarray(geometry.nodes, dtype=np.float32),
        volumes=np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1),
        state_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float32),
        residual_scale=np.asarray(
            runtime.normalization.residual_scale, dtype=np.float32
        ),
        gamma=np.asarray(runtime.normalization.gamma, dtype=np.float64),
        truth_conservative=np.asarray(truth_states, dtype=np.float64)[indices].astype(
            np.float32
        ),
        raw_shadow_conservative=np.asarray(raw_states, dtype=np.float64)[
            indices
        ].astype(np.float32),
        corrected_conservative=np.asarray(candidate_states, dtype=np.float64)[
            indices
        ].astype(np.float32),
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
    inactive_max_abs: Mapping[str, float],
    shadow_raw_max_abs: Mapping[str, float],
    prefix_repeat_abs: float,
    main_execution: Mapping[str, Any],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    for scope in (
        "population",
        "active_population",
        "active_group_e12",
        "active_group_e14",
    ):
        checks[f"{scope}_full_rank8_trajectory_strict"] = all(
            a31._score_row(score_rows, cell="overall", view=view, scope=scope)[
                "skill_status"
            ]
            == "ok"
            and float(
                a31._score_row(score_rows, cell="overall", view=view, scope=scope)[
                    "rms_ratio_vs_zero"
                ]
            )
            < 1.0
            for view in ("full", "rank8_parallel")
        )
    active_endpoint = float(
        a31._score_row(
            score_rows,
            cell="endpoint_29",
            view="full",
            scope="active_population",
        )["rms_ratio_vs_zero"]
    )
    checks["active_population_endpoint_strict"] = active_endpoint < 1.0
    paired = {str(row["case_id"]): row for row in case_rows}
    active_wins = 0
    active_bounds = True
    active_case_ratios = {}
    for case_id in EXPECTED_ACTIVE_CASES:
        trajectory = float(
            a31._score_row(
                score_rows,
                cell="overall",
                view="full",
                scope="case",
                case_id=case_id,
            )["rms_ratio_vs_zero"]
        )
        endpoint = float(
            a31._score_row(
                score_rows,
                cell="endpoint_29",
                view="full",
                scope="case",
                case_id=case_id,
            )["rms_ratio_vs_zero"]
        )
        active_case_ratios[case_id] = trajectory
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
        a31._control_passed(row) for row in controls
    )
    checks["inactive_states_bitwise_raw"] = set(inactive_max_abs) == set(
        INACTIVE_CASES
    ) and all(value == 0.0 for value in inactive_max_abs.values())
    checks["active_shadow_byte_equal_raw"] = set(shadow_raw_max_abs) == set(
        EXPECTED_ACTIVE_CASES
    ) and all(value == 0.0 for value in shadow_raw_max_abs.values())
    checks["all_rollouts_complete_finite_admissible"] = len(rollouts) == 2 * len(
        CASE_IDS
    ) and all(
        row["execution"]["completed_calls"] == 30
        and all(metric["finite"] and metric["admissible"] for metric in row["rows"])
        for row in rollouts
    )
    exact = [row for row in audit_rows if row["route"] == "exact_a32_window"]
    coast = [row for row in audit_rows if row["route"] == "surrogate_coast"]
    inactive = [row for row in audit_rows if row["route"] == "inactive_raw"]
    checks["route_audit_inventory_exact"] = (
        len(exact) == 64 and len(coast) == 56 and len(inactive) == 300
    )
    checks["exact_a32_window_closure"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 3
        and bool(row["correction_active"])
        and float(row["correction_to_native_increment"]) <= 0.05 + STRICT_TOLERANCE
        and max(
            float(row["pre_model_fine_to_native_max_abs"]),
            float(row["maximum_scaled_component_mean_abs"]),
            float(row["maximum_excluded_abs"]),
            float(row["maximum_inactive_coordinate_abs"]),
            float(row["maximum_modal_reconstruction_abs"]),
            float(row["maximum_proposal_update_abs"]),
            float(row["maximum_integral_difference_abs"]),
            float(row["maximum_boundary_difference_abs"]),
            float(row["maximum_projection_idempotence_abs"]),
            float(row["maximum_update_identity_abs"]),
        )
        <= STRICT_TOLERANCE
        and float(row["post_fp32_fine_to_native_max_abs"]) <= 1.0e-6
        for row in exact
    )
    checks["surrogate_coast_one_native_closure"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and not bool(row["correction_active"])
        and max(
            float(row["pre_model_fine_to_native_max_abs"]),
            float(row["post_fp32_fine_to_native_max_abs"]),
            float(row["maximum_proposal_update_abs"]),
            float(row["maximum_integral_difference_abs"]),
            float(row["maximum_boundary_difference_abs"]),
            float(row["maximum_projection_idempotence_abs"]),
            float(row["maximum_update_identity_abs"]),
        )
        <= STRICT_TOLERANCE
        for row in coast
    )
    checks["inactive_exact_raw_route"] = all(
        bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and float(row["retained_displacement_rms"]) == 0.0
        for row in inactive
    )
    checks["deterministic_prefix"] = prefix_repeat_abs <= 1.0e-6
    active_full = float(
        a31._score_row(
            score_rows, cell="overall", view="full", scope="active_population"
        )["rms_ratio_vs_zero"]
    )
    active_rank8 = float(
        a31._score_row(
            score_rows,
            cell="overall",
            view="rank8_parallel",
            scope="active_population",
        )["rms_ratio_vs_zero"]
    )
    checks["a32_active_population_benefit_retained"] = (
        active_full <= A32_ACTIVE_FULL
        and active_rank8 <= A32_ACTIVE_RANK8
        and active_endpoint <= A32_ACTIVE_ENDPOINT
    )
    checks["a32_active_case_benefit_retained"] = all(
        active_case_ratios[case_id] <= A32_CASE_FULL[case_id] + CASE_RETENTION_ALLOWANCE
        and active_case_ratios[case_id] < 1.0
        for case_id in EXPECTED_ACTIVE_CASES
    )
    checks["logical_cost_reduction"] = (
        main_execution["candidate"]["logical_model_calls"] == CANDIDATE_CALLS
        and CANDIDATE_CALLS < OPTIMIZED_A32_CALLS
    )
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "active_case_joint_win_count": active_wins,
        "active_case_full_trajectory_ratios": active_case_ratios,
        "strict_improvement_equality_fails": True,
        "control_ratio_limit": CONTROL_LIMIT,
        "a32_retention": {
            "active_full_limit": A32_ACTIVE_FULL,
            "active_rank8_limit": A32_ACTIVE_RANK8,
            "active_endpoint_limit": A32_ACTIVE_ENDPOINT,
            "case_allowance": CASE_RETENTION_ALLOWANCE,
        },
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
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"A43 requires CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
        )
    preflight = _verify_preflight(args.preflight, args)
    _verify_a32(args.a32_result)
    _verify_a42(args.a42_result)
    start_coefficients, end_coefficients, _ = a31._verify_a28(args.a28_map)
    expected_references = a31._reference_rows(args.a29_reference_checks)
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
            raise ValueError("A43 runtime source differs from preflight")
        prefix_reference, _ = a31._load_reference(
            runtime,
            EXPECTED_ACTIVE_CASES[0],
            expected_references[EXPECTED_ACTIVE_CASES[0]],
        )
        prefix_initial = np.asarray(
            prefix_reference["conservative_states"][0], dtype=np.float64
        )
        prefix_descriptor = a31._descriptor(runtime, prefix_initial)
        prefix_selected = bool(
            prefix_descriptor.status == "ok"
            and prefix_descriptor.normalized_wall_distance is not None
            and prefix_descriptor.normalized_wall_distance
            <= FROZEN_POSITION_BUFFER_THRESHOLD
        )
        prefix_runs = [
            _run_candidate_arm(
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
        inactive_max_abs: dict[str, float] = {}
        shadow_raw_max_abs: dict[str, float] = {}
        view_maxima = {
            "band": 0.0,
            "subspace_reconstruction": 0.0,
            "orthogonality": 0.0,
        }
        animation_rows = []
        animation_dir = output_dir / "animation_bundles"
        for case_id in CASE_IDS:
            print(f"A43 case {case_id}: loading authenticated native truth", flush=True)
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
            print(f"A43 case {case_id}: raw comparator H30", flush=True)
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
            print(
                f"A43 case {case_id}: phase-selective H30 selected={selected}",
                flush=True,
            )
            candidate = _run_candidate_arm(
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
            if selected:
                shadow_raw_max_abs[case_id] = max(
                    a31._maximum_abs(left - right)
                    for left, right in zip(
                        raw["states"], candidate["shadow_states"], strict=True
                    )
                )
            else:
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
            for rollout in (raw, candidate):
                rollout.pop("states")
                rollout.pop("audit_rows")
                if rollout["policy"] == POLICY:
                    rollout.pop("shadow_states")
                rollouts.append(rollout)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        selected_cases = tuple(
            row["case_id"] for row in descriptor_rows if row["position_selected"]
        )
        score_rows = a31._score_views(view_records)
        all_metric_rows = [metric for rollout in rollouts for metric in rollout["rows"]]
        case_rows, paired_controls, paired_population = (
            response._paired_rollout_controls_for_cases(
                all_metric_rows, case_ids=CASE_IDS
            )
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
            "inactive_raw_native_logical_calls",
            "inactive_raw_native_actual_forward_passes",
            "inactive_raw_native_forward_seconds",
            "shadow_native_logical_calls",
            "shadow_native_actual_forward_passes",
            "shadow_native_forward_seconds",
            "candidate_native_logical_calls",
            "candidate_native_actual_forward_passes",
            "candidate_native_forward_seconds",
            "candidate_fine_logical_calls",
            "candidate_fine_actual_forward_passes",
            "candidate_fine_forward_seconds",
            "inactive_raw_steps",
            "exact_a32_steps",
            "surrogate_coast_steps",
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
        main_execution["candidate_logical_cost_ratio_vs_optimized_a32"] = (
            main_execution["candidate"]["logical_model_calls"] / OPTIMIZED_A32_CALLS
        )
        main_execution["candidate_forward_time_ratio_vs_raw"] = (
            main_execution["candidate"]["total_forward_seconds"]
            / main_execution["raw"]["total_forward_seconds"]
        )
        structural_checks = {
            "source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "population_exact": set(selected_cases) == set(EXPECTED_ACTIVE_CASES),
            "descriptor_inventory_exact": len(descriptor_rows) == len(CASE_IDS),
            "reference_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "paired_metric_inventory_exact": len(all_metric_rows)
            == 2 * len(CASE_IDS) * len(INPUT_CALLS),
            "main_logical_call_inventory_exact": (
                main_execution["raw"]["native_logical_calls"] == 420
                and main_execution["raw"]["fine_logical_calls"] == 0
                and main_execution["candidate"]["logical_model_calls"]
                == CANDIDATE_CALLS
                and main_execution["candidate"]["native_logical_calls"] == 484
                and main_execution["candidate"]["fine_logical_calls"] == 64
                and main_execution["candidate"]["inactive_raw_native_logical_calls"]
                == 300
                and main_execution["candidate"]["shadow_native_logical_calls"] == 120
                and main_execution["candidate"]["candidate_native_logical_calls"] == 64
                and main_execution["candidate"]["candidate_fine_logical_calls"] == 64
            ),
            "route_step_inventory_exact": (
                main_execution["candidate"]["inactive_raw_steps"] == 300
                and main_execution["candidate"]["exact_a32_steps"] == 64
                and main_execution["candidate"]["surrogate_coast_steps"] == 56
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
            inactive_max_abs=inactive_max_abs,
            shadow_raw_max_abs=shadow_raw_max_abs,
            prefix_repeat_abs=prefix_repeat_abs,
            main_execution=main_execution,
            structural_checks=structural_checks,
        )
        write_csv(output_dir / "rollout_call_metrics.csv", all_metric_rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "view_scores.csv", score_rows)
        write_csv(output_dir / "route_audits.csv", audit_rows)
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
        atomic_write_json(
            output_dir / "source_manifest.json", preflight["source_manifest"]
        )
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
                    "qualified_phase_selective_replay"
                    if gate["status"] == "qualified"
                    else "stopped_phase_selective_replay"
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
                "active_shadow_raw_maximum_state_abs_difference": shadow_raw_max_abs,
                "view_closure_maxima": view_maxima,
                "main_execution": main_execution,
                "prefix_replay": prefix_execution,
                "a32_result_sha256": sha256_file(args.a32_result),
                "a32_payload_sha256": A32_PAYLOAD_SHA256,
                "a42_result_sha256": sha256_file(args.a42_result),
                "a42_payload_sha256": A42_PAYLOAD_SHA256,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": preflight["source_manifest"][
                    "payload_sha256"
                ],
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
                    "Adaptive-open E12/E14 phase-selective shadow-response replay "
                    "only; no independent, cross-family, conservation, convergence, "
                    "direct-off-grid, asymptotic-order, or Richardson claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(output_dir / "phase_selective_coast_replay.json", result)
        return result, 0 if gate["status"] == "qualified" else 4
    finally:
        base._close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    a31._add_external_arguments(parser)
    parser.add_argument("--a32-result", type=Path, required=True)
    parser.add_argument("--a42-result", type=Path, required=True)


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
    rollout.add_argument("--device", choices=("cuda",), default="cuda")
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
