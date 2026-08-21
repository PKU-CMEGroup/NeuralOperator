#!/usr/bin/env python3
"""Compare direct and A32 transfer rollouts on retained 500x200 truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    evaluate_pcno_affine_shadow_tether_rollout as a32,
)
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
    DENOMINATOR_FLOOR,
    DiagnosticSnapshot,
    NativeIncrementBasis,
    case_first_statistics,
    prolong_nested_state,
    score_scalar_correction,
    transfer_floor_fields,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    EVALUATION_CASE_IDS,
    INTEGRAL_COMPONENT_NAMES,
    VIEW_SPECS,
    _statistics_for_view,
    build_fixed_cosine_projector,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    predict_resolution_sample,
    pressure_profile_shock_metrics,
    reference_at_resolution,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    transverse_velocity_position_descriptor,
)
from utility.time_dependent_no.shock_vortex_family import family_case_provenance

WORKING_ID = "W26-L5-P6-RFB19-A33-SP19-AFFINE-SHADOW-TETHER-QUERY-GRID"
PREFLIGHT_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_preflight_v1"
RESULT_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_v1"
SOURCE_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_source_v1"
ANIMATION_SCHEMA = "pcno_sp19_affine_shadow_tether_query_grid_bundles_v1"

A32_RESULT_SHA256 = "58eb364fc2e1dfd8ea00a3e4312ae4ace4c03dd9234bfb92fc48da737180cc0f"
A32_PAYLOAD_SHA256 = "208ac069aee27d021e2506377dbd586c072c9a64353d866dc8e296b71b123e89"
D074_SUMMARY_SHA256 = "bb7b1c38097c92c6866b3f8853ceeab28668fe0bf38e4fa469cc6033c36a57ab"
D074_REFERENCE_CHECKS_SHA256 = (
    "f685f5b01462a609e3bf590f6997566803016c50143cc5a3dcabbac7ff73eeeb"
)

NATIVE_RESOLUTION = (250, 100)
QUERY_RESOLUTION = (500, 200)
INPUT_CALLS = tuple(range(30))
OUTPUT_CALLS = tuple(range(31))
BANDS = {
    "overall": INPUT_CALLS,
    "band_0_7": tuple(range(8)),
    "band_8_14": tuple(range(8, 15)),
    "band_15_21": tuple(range(15, 22)),
    "band_22_29": tuple(range(22, 30)),
    "endpoint_29": (29,),
}
GROUP_CASES = {
    f"e{strength:02d}": (
        f"sv_e{strength:02d}_y00",
        f"sv_e{strength:02d}_y08",
    )
    for strength in (0, 6, 11)
}
CASE_IDS = tuple(case_id for cases in GROUP_CASES.values() for case_id in cases)
CASE_TO_GROUP = {
    case_id: group for group, cases in GROUP_CASES.items() for case_id in cases
}
EXPECTED_SELECTED_CASES = CASE_IDS

DIRECT_QUERY_POLICY = "direct_query_raw"
MATCHED_DIRECT_POLICY = "matched_information_direct"
RAW_TRANSFER_POLICY = "raw_transfer_native"
CORRECTED_TRANSFER_POLICY = "a32_transfer_native"
POLICIES = (
    DIRECT_QUERY_POLICY,
    MATCHED_DIRECT_POLICY,
    RAW_TRANSFER_POLICY,
    CORRECTED_TRANSFER_POLICY,
)
PAIR_CORRECTION = "corrected_transfer_vs_raw_transfer"
PAIR_DIRECT_RAW = "direct_query_vs_raw_transfer"
PAIR_DIRECT_CORRECTED = "direct_query_vs_corrected_transfer"
PAIR_MATCHED_DIRECT = "matched_information_vs_direct_query"

CONTROL_LIMIT = 1.05
STRICT_TOLERANCE = 1.0e-12
REPEAT_TOLERANCE = 1.0e-6
ANIMATION_CASE_IDS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
ANIMATION_CALLS = tuple(range(0, 31, 2))
ANIMATION_CONTRACT = {
    "purpose": "post_run_diagnostic_only_not_a_selector_or_gate",
    "case_ids": list(ANIMATION_CASE_IDS),
    "output_calls": list(ANIMATION_CALLS),
    "physical_times": [0.02 * call for call in ANIMATION_CALLS],
    "query_resolution": list(QUERY_RESOLUTION),
    "scored_output_calls": list(OUTPUT_CALLS),
    "visualization_only_frame_stride": 2,
    "gamma": 1.4,
    "field_limits": {"density": [0.75, 1.25], "pressure": [0.65, 1.35]},
    "absolute_error_limits": [1.0e-5, 0.25],
    "relative_improvement_limits": [-1.0, 1.0],
    "relative_improvement_denominator_floor": 1.0e-5,
    "candidate_label": "A32 transfer",
    "output_stem": "affine_shadow_tether_query_grid_h30",
}

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_affine_shadow_tether_query_grid.py",
    "scripts/time_dependent_no/visualize_pcno_affine_shadow_tether_query_grid.py",
    "tests/time_dependent_no/test_pcno_affine_shadow_tether_query_grid.py",
)
DEPENDENCY_PATHS = (
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
    "utility/time_dependent_no/pcno_residual_structure.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_sparse_modal_correction.py",
    "utility/time_dependent_no/shock_vortex_family.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _verify_a32(path: Path) -> dict[str, Any]:
    if sha256_file(path) != A32_RESULT_SHA256:
        raise ValueError("A32 result file SHA-256 mismatch")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != a32.RESULT_SCHEMA
        or payload.get("working_id") != a32.WORKING_ID
        or payload.get("payload_sha256") != A32_PAYLOAD_SHA256
        or payload.get("status") != "qualified_adaptive_open_rollout"
        or gate.get("status") != "qualified"
        or gate.get("failed_checks") != []
        or not all(gate.get("checks", {}).values())
    ):
        raise ValueError("A32 is not the exact qualified result")
    inventory = payload.get("artifact_inventory_before_summary")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("A32 artifact inventory is unavailable")
    root = path.parent.resolve()
    for relative, expected in inventory.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents or not isinstance(expected, Mapping):
            raise ValueError("A32 artifact inventory contains an unsafe path")
        if (
            not candidate.is_file()
            or sha256_file(candidate) != expected.get("sha256")
            or candidate.stat().st_size != int(expected.get("bytes", -1))
        ):
            raise ValueError(f"A32 artifact differs: {relative}")
    return payload


def _verify_d074(
    summary_path: Path, reference_checks_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    if sha256_file(summary_path) != D074_SUMMARY_SHA256:
        raise ValueError("D074 H30 summary SHA-256 mismatch")
    if sha256_file(reference_checks_path) != D074_REFERENCE_CHECKS_SHA256:
        raise ValueError("D074 reference-check SHA-256 mismatch")
    summary = _read_json(summary_path)
    population = summary.get("population", {})
    if (
        summary.get("schema") != "pcno_native_residual_correction_diagnostic_v1"
        or summary.get("status") != "complete"
        or tuple(population.get("evaluation_case_ids", ())) != CASE_IDS
        or population.get("sealed_populations_accessed") is not False
        or population.get("split") != "validation"
    ):
        raise ValueError("D074 population record differs from A33")
    rows = _read_csv(reference_checks_path)
    indexed = {str(row.get("case_id")): row for row in rows}
    if len(rows) != len(CASE_IDS) or tuple(indexed) != CASE_IDS:
        raise ValueError("D074 reference inventory differs from A33")
    for case_id, row in indexed.items():
        if (
            row.get("retained_resolution") != "500x200"
            or row.get("state_dtype") != "float64"
            or json.loads(row.get("state_shape", "null")) != [61, 100000, 4]
            or float(row.get("restriction_crosscheck_max_abs", "inf"))
            > STRICT_TOLERANCE
            or len(row.get("active_reference_artifact_sha256", "")) != 64
        ):
            raise ValueError(f"D074 reference row differs for {case_id}")
    return summary, indexed


def _source_hashes() -> dict[str, Any]:
    return {
        "owned": sha256_files(OWNED_SOURCE_PATHS, root=ROOT),
        "dependencies": sha256_files(DEPENDENCY_PATHS, root=ROOT),
    }


def _source_manifest(
    args: argparse.Namespace, *, base_source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    a32_payload = _verify_a32(args.a32_result)
    a31._verify_a28(args.a28_map)
    d074_summary, reference_rows = _verify_d074(
        args.d074_summary, args.d074_reference_checks
    )
    return with_payload_sha256(
        {
            "schema": SOURCE_SCHEMA,
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "lineage": {
                "a32_result_sha256": sha256_file(args.a32_result),
                "a32_payload_sha256": a32_payload["payload_sha256"],
                "a28_map_sha256": sha256_file(args.a28_map),
                "a28_payload_sha256": a31.A28_PAYLOAD_SHA256,
                "d074_summary_sha256": sha256_file(args.d074_summary),
                "d074_reference_checks_sha256": sha256_file(args.d074_reference_checks),
            },
            "base_runtime_source_manifest": dict(base_source_manifest),
            "population": {
                "case_ids": list(CASE_IDS),
                "groups": {key: list(value) for key, value in GROUP_CASES.items()},
                "status": "already_open_d074_a2_evaluation_adaptive_reuse",
                "new_population_opened": False,
                "sealed_population_opened": False,
            },
            "reference_artifact_sha256": {
                case_id: row["active_reference_artifact_sha256"]
                for case_id, row in reference_rows.items()
            },
            "d074_population": d074_summary["population"],
            "inference_contract": {
                "query_resolution": list(QUERY_RESOLUTION),
                "native_resolution": list(NATIVE_RESOLUTION),
                "arms": list(POLICIES),
                "query_to_native": "float64_exact_2x2_conservative_restriction",
                "native_to_query": "float64_piecewise_constant_injection",
                "corrected_policy": a32.POLICY,
                "active_input_calls": list(a31.BINARY_POSITION_PHASE_CALLS),
                "truth_case_id_or_error_in_routing": False,
                "coefficient_refit": False,
                "physical_radius_arm": "omitted_no_D073_B_wrapper",
            },
            "animation_contract": ANIMATION_CONTRACT,
        }
    )


def _verify_multires_metadata(
    root: Path, expected_rows: Mapping[str, Mapping[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for case_id in CASE_IDS:
        case_root = root / case_id
        reference_path = case_root / "reference.npz"
        summary_path = case_root / "summary.json"
        summary = _read_json(summary_path)
        actual = sha256_file(reference_path)
        expected = expected_rows[case_id]["active_reference_artifact_sha256"]
        if (
            summary.get("status") != "passed"
            or summary.get("reference_artifact_sha256") != actual
            or actual != expected
        ):
            raise ValueError(f"retained multires reference differs for {case_id}")
        output.append(
            {
                "case_id": case_id,
                "reference_sha256": actual,
                "summary_sha256": sha256_file(summary_path),
                "reference_bytes": reference_path.stat().st_size,
                "arrays_loaded": False,
            }
        )
    return output


def synthetic_summary() -> dict[str, Any]:
    checks = {
        "population_exact_6": CASE_IDS == tuple(EVALUATION_CASE_IDS),
        "group_inventory_3x2": len(GROUP_CASES) == 3
        and all(len(cases) == 2 for cases in GROUP_CASES.values()),
        "direct_query_calls_180": len(CASE_IDS) * len(INPUT_CALLS) == 180,
        "matched_direct_calls_180": len(CASE_IDS) * len(INPUT_CALLS) == 180,
        "raw_transfer_calls_180": len(CASE_IDS) * len(INPUT_CALLS) == 180,
        "corrected_native_calls_360": 2 * len(CASE_IDS) * len(INPUT_CALLS) == 360,
        "corrected_fine_calls_96": len(CASE_IDS) * len(a31.BINARY_POSITION_PHASE_CALLS)
        == 96,
        "corrected_total_calls_456": 360 + 96 == 456,
        "main_total_calls_996": 180 + 180 + 180 + 456 == 996,
        "shared_shadow_all_arm_lower_bound_816": 180 + 180 + 456 == 816,
        "shared_shadow_corrected_marginal_276": 456 - 180 == 276,
        "animation_inventory_4": len(ANIMATION_CASE_IDS) == 4,
        "owned_source_inventory_4": len(OWNED_SOURCE_PATHS) == 4,
    }
    return {
        "schema": "pcno_sp19_affine_shadow_tether_query_grid_synthetic_v1",
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, base_source = base._open_contract(
        a29._base_args(args)
    )
    try:
        _, d074_rows = _verify_d074(args.d074_summary, args.d074_reference_checks)
        source = _source_manifest(args, base_source_manifest=base_source)
        metadata = _verify_multires_metadata(args.multires_reference_root, d074_rows)
        provenance = {
            case_id: family_case_provenance(manifest, case_id) for case_id in CASE_IDS
        }
        checks = {
            "source_inventory_exact": set(source["source_sha256"]["owned"])
            == set(OWNED_SOURCE_PATHS)
            and set(source["source_sha256"]["dependencies"]) == set(DEPENDENCY_PATHS),
            "lineage_exact": source["lineage"]
            == {
                "a32_result_sha256": A32_RESULT_SHA256,
                "a32_payload_sha256": A32_PAYLOAD_SHA256,
                "a28_map_sha256": a31.A28_RESULT_SHA256,
                "a28_payload_sha256": a31.A28_PAYLOAD_SHA256,
                "d074_summary_sha256": D074_SUMMARY_SHA256,
                "d074_reference_checks_sha256": D074_REFERENCE_CHECKS_SHA256,
            },
            "population_exact": tuple(CASE_IDS) == tuple(EVALUATION_CASE_IDS),
            "case_inventory_present": all(
                case_id in store.keys for case_id in CASE_IDS
            ),
            "reference_metadata_exact": len(metadata) == len(CASE_IDS),
            "checkpoint_stride_exact": base.checkpoint_step_stride(checkpoint) == 2,
            "new_population_not_opened": True,
            "D073_B_not_implemented_or_run": True,
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "source_manifest": source,
                "population": provenance,
                "reference_metadata": metadata,
                "expected_selected_cases": list(EXPECTED_SELECTED_CASES),
                "selector_recomputed_before_model_calls_in_rollout": True,
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "truth_arrays_loaded": False,
                "model_calls": 0,
                "recurrence_executed": False,
            }
        )
    finally:
        store.close()
    if payload["status"] != "passed":
        raise ValueError("A33 preflight checks did not all pass")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
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
        or source
        != _source_manifest(
            args,
            base_source_manifest=source.get("base_runtime_source_manifest", {}),
        )
    ):
        raise ValueError("A33 preflight differs from the frozen contract")
    return payload


def _require_reference_row(
    observed: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    required = (
        "case_id",
        "frozen_training_reference_sha256",
        "active_reference_artifact_sha256",
        "retained_resolution",
        "state_dtype",
        "state_shape",
        "restriction_crosscheck_max_abs",
    )
    for key in required:
        if key not in observed or key not in expected:
            raise ValueError(f"reference row is missing {key}")
    if (
        str(observed["case_id"]) != expected["case_id"]
        or str(observed["frozen_training_reference_sha256"])
        != expected["frozen_training_reference_sha256"]
        or str(observed["active_reference_artifact_sha256"])
        != expected["active_reference_artifact_sha256"]
        or str(observed["retained_resolution"]) != expected["retained_resolution"]
        or str(observed["state_dtype"]) != expected["state_dtype"]
        or list(observed["state_shape"]) != json.loads(expected["state_shape"])
        or abs(
            float(observed["restriction_crosscheck_max_abs"])
            - float(expected["restriction_crosscheck_max_abs"])
        )
        > np.finfo(np.float64).eps
    ):
        raise ValueError(f"reference row differs for {expected['case_id']}")


def _query_projector(runtime: Any) -> Any:
    geometry = runtime.geometry_by_resolution[QUERY_RESOLUTION]
    node_type = (
        runtime.sample_by_resolution[QUERY_RESOLUTION]["node_type"]
        .detach()
        .cpu()
        .numpy()[0]
    )
    return build_fixed_cosine_projector(
        geometry.nodes,
        geometry.node_measures,
        node_type,
        rank=8,
        domain_bounds=(
            runtime.first_config.x_min,
            runtime.first_config.x_max,
            runtime.first_config.y_min,
            runtime.first_config.y_max,
        ),
    )


def _run_direct_arm(
    runtime: Any, *, policy: str, initial_state: np.ndarray, horizon: int
) -> dict[str, Any]:
    if policy not in {DIRECT_QUERY_POLICY, MATCHED_DIRECT_POLICY}:
        raise ValueError("unsupported direct query-grid policy")
    state = np.asarray(initial_state, dtype=np.float64)
    expected_shape = (QUERY_RESOLUTION[0] * QUERY_RESOLUTION[1], 4)
    if state.shape != expected_shape or not np.isfinite(state).all():
        raise ValueError("direct initial state differs from query-grid contract")
    states = [np.array(state, copy=True)]
    execution = a31._new_execution()
    started = perf_counter()
    for _ in range(horizon):
        prediction, timing = predict_resolution_sample(
            runtime.model,
            runtime.sample_by_resolution[QUERY_RESOLUTION],
            state,
            device=runtime.device,
            amp="none",
            repeats=1,
        )
        fine_eval._account_execution(execution, timing, resolution=QUERY_RESOLUTION)
        state = np.asarray(prediction, dtype=np.float64)
        states.append(np.array(state, copy=True))
        status = conservative_admissibility_summary(
            state, gamma=runtime.normalization.gamma
        )
        if not status["finite"] or not status["admissible"]:
            break
    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = len(states) - 1
    return {"policy": policy, "states": states, "execution": execution}


def _map_native_states(states: Sequence[np.ndarray]) -> tuple[list[np.ndarray], float]:
    started = perf_counter()
    mapped = [
        prolong_nested_state(
            state,
            coarse_resolution=NATIVE_RESOLUTION,
            fine_resolution=QUERY_RESOLUTION,
        )
        for state in states
    ]
    return mapped, perf_counter() - started


def _front_errors(
    prediction: np.ndarray, target: np.ndarray, runtime: Any
) -> dict[str, float | None]:
    metrics = pressure_profile_shock_metrics(
        prediction,
        target,
        resolution=QUERY_RESOLUTION,
        x_min=runtime.first_config.x_min,
        x_max=runtime.first_config.x_max,
        gamma=runtime.normalization.gamma,
        shock_center_x=runtime.first_config.shock_x,
    )
    output: dict[str, float | None] = {
        "front_position": metrics["shock_position_absolute_error"],
        "front_strength_log_ratio": None,
        "front_thickness_log_ratio": None,
    }
    for source, target_key in (
        ("pressure_profile_shock_strength_ratio", "front_strength_log_ratio"),
        ("pressure_profile_shock_thickness_ratio", "front_thickness_log_ratio"),
    ):
        value = metrics[source]
        if value is not None and math.isfinite(float(value)) and float(value) > 0.0:
            output[target_key] = abs(math.log(float(value)))
    return output


def _physical_integral(value: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if field.shape != (mass.size, 4):
        raise ValueError("field and physical volumes do not align")
    return np.einsum("n,nc->c", mass, field, optimize=True)


def _query_metric_rows(
    runtime: Any,
    *,
    query_projector: Any,
    case_id: str,
    policy: str,
    states: Sequence[np.ndarray],
    truth_states: np.ndarray,
    shadow_states: Sequence[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if len(states) != 31 or truth_states.shape != (31, 100000, 4):
        raise ValueError("query-grid trajectory inventory differs")
    if shadow_states is not None and len(shadow_states) != len(states):
        raise ValueError("query-grid shadow trajectory differs")
    geometry = runtime.geometry_by_resolution[QUERY_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    cumulative = np.zeros_like(states[0], dtype=np.float64)
    rows = []
    for input_call in INPUT_CALLS:
        previous = np.asarray(states[input_call], dtype=np.float64)
        next_state = np.asarray(states[input_call + 1], dtype=np.float64)
        current_truth = np.asarray(truth_states[input_call], dtype=np.float64)
        target = np.asarray(truth_states[input_call + 1], dtype=np.float64)
        applied_increment = next_state - previous
        true_increment = target - current_truth
        defect = applied_increment - true_increment
        cumulative = cumulative + defect
        state_error = next_state - target
        parts, _ = query_projector.split(state_error)
        masks, _, _ = shock_vortex_regions(
            target,
            geometry.nodes,
            resolution=QUERY_RESOLUTION,
            gamma=runtime.normalization.gamma,
        )
        admissibility = conservative_admissibility_summary(
            next_state, gamma=runtime.normalization.gamma
        )
        front = _front_errors(next_state, target, runtime)
        integral = _physical_integral(state_error, volumes)
        intervention = (
            np.zeros_like(next_state)
            if shadow_states is None
            else next_state
            - np.asarray(shadow_states[input_call + 1], dtype=np.float64)
        )
        intervention_rms = weighted_scaled_rms(
            intervention,
            volumes=volumes,
            component_scale=residual_scale,
        )
        increment_rms = weighted_scaled_rms(
            applied_increment,
            volumes=volumes,
            component_scale=residual_scale,
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "group_id": CASE_TO_GROUP[case_id],
            "policy": policy,
            "input_call": input_call,
            "output_call": input_call + 1,
            "state_error": weighted_scaled_rms(
                state_error, volumes=volumes, component_scale=state_scale
            ),
            "rank8_state_error": weighted_scaled_rms(
                parts["parallel"], volumes=volumes, component_scale=state_scale
            ),
            "increment_defect": weighted_scaled_rms(
                defect, volumes=volumes, component_scale=residual_scale
            ),
            "cumulative_defect": weighted_scaled_rms(
                cumulative, volumes=volumes, component_scale=residual_scale
            ),
            "intervention_rms": intervention_rms,
            "intervention_to_increment": (
                intervention_rms / increment_rms
                if increment_rms > DENOMINATOR_FLOOR
                else None
            ),
            "finite": bool(admissibility["finite"]),
            "admissible": bool(admissibility["admissible"]),
            "minimum_density": admissibility["minimum_density"],
            "minimum_pressure": admissibility["minimum_pressure"],
            "minimum_internal_energy": admissibility["minimum_internal_energy"],
            **front,
        }
        for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
            row[f"component_{name}_state_error"] = weighted_scaled_rms(
                state_error[:, component : component + 1],
                volumes=volumes,
                component_scale=state_scale[component : component + 1],
            )
            row[f"integral_{name}_error"] = float(integral[component])
        for key, mask_name in (
            ("boundary_state_error", "boundary_le_0.05"),
            ("shock_state_error", "partition_shock"),
            ("vortex_state_error", "partition_vortex"),
            ("smooth_state_error", "partition_smooth"),
        ):
            row[key] = weighted_scaled_rms(
                state_error,
                volumes=volumes,
                component_scale=state_scale,
                mask=masks[mask_name],
            )
        rows.append(row)
    return rows


def _pair_view_records(
    runtime: Any,
    *,
    query_projector: Any,
    case_id: str,
    pair: str,
    baseline_states: Sequence[np.ndarray],
    candidate_states: Sequence[np.ndarray],
    truth_states: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(baseline_states) != 31 or len(candidate_states) != 31:
        raise ValueError("paired query-grid trajectory is incomplete")
    geometry = runtime.geometry_by_resolution[QUERY_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    maxima = {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
    output = []
    for input_call in INPUT_CALLS:
        target = np.asarray(truth_states[input_call + 1], dtype=np.float64)
        baseline = np.asarray(baseline_states[input_call + 1], dtype=np.float64)
        candidate = np.asarray(candidate_states[input_call + 1], dtype=np.float64)
        correction = candidate - baseline
        zero = np.zeros_like(correction)
        masks, _, _ = shock_vortex_regions(
            target,
            geometry.nodes,
            resolution=QUERY_RESOLUTION,
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
            target_correction=target - baseline,
            volumes=volumes,
            component_scale=state_scale,
            masks=masks,
        )
        for view in VIEW_SPECS:
            statistics, closure = _statistics_for_view(
                snapshot,
                view,
                resolution=QUERY_RESOLUTION,
                projector=query_projector,
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
                    "pair": pair,
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
        **{f"group_{group}": cases for group, cases in GROUP_CASES.items()},
    }


def _score_views(
    records: Sequence[Mapping[str, Any]], *, pair: str
) -> list[dict[str, Any]]:
    rows = []
    pair_records = [row for row in records if row["pair"] == pair]
    expected = {
        (case_id, input_call, view.key)
        for case_id in CASE_IDS
        for input_call in INPUT_CALLS
        for view in VIEW_SPECS
    }
    actual = {
        (str(row["case_id"]), int(row["input_call"]), str(row["view"]))
        for row in pair_records
    }
    if len(pair_records) != len(expected) or actual != expected:
        raise ValueError(f"A33 view-record inventory differs for {pair}")
    for cell, calls in BANDS.items():
        call_set = set(calls)
        for view in VIEW_SPECS:
            view_records = [
                row
                for row in pair_records
                if row["view"] == view.key and row["input_call"] in call_set
            ]
            for scope, cases in _population_specs().items():
                selected = [
                    (str(row["case_id"]), row["statistics"])
                    for row in view_records
                    if row["case_id"] in cases
                ]
                score = score_scalar_correction(
                    case_first_statistics(selected), (0.0, 1.0)
                )
                rows.append(
                    {
                        "pair": pair,
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
                score = score_scalar_correction(
                    case_first_statistics(selected), (0.0, 1.0)
                )
                rows.append(
                    {
                        "pair": pair,
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
    pair: str,
    cell: str,
    view: str,
    scope: str,
    case_id: str | None = None,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["pair"] == pair
        and row["cell"] == cell
        and row["view"] == view
        and row["scope"] == scope
        and (case_id is None or row.get("case_id") == case_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing A33 score row {(pair, cell, view, scope, case_id)!r}"
        )
    return matches[0]


def _control_passed(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") == "ok"
        and row.get("ratio") is not None
        and np.isfinite(float(row["ratio"]))
        and float(row["ratio"]) <= CONTROL_LIMIT
    )


def _field_controls(
    score_rows: Sequence[Mapping[str, Any]], *, pair: str
) -> list[dict[str, Any]]:
    output = []
    scopes = (*_population_specs(), "case")
    for cell in ("overall", "endpoint_29"):
        for view in VIEW_SPECS:
            for scope in scopes:
                if scope == "case":
                    cases: Sequence[str | None] = CASE_IDS
                else:
                    cases = (None,)
                for case_id in cases:
                    row = _score_row(
                        score_rows,
                        pair=pair,
                        cell=cell,
                        view=view.key,
                        scope=scope,
                        case_id=case_id,
                    )
                    output.append(
                        {
                            "pair": pair,
                            "kind": "field",
                            "key": view.key,
                            "scope": case_id if case_id is not None else scope,
                            "cell": cell,
                            "zero_rms": row["target_rms"],
                            "corrected_rms": (
                                float(row["target_rms"])
                                * float(row["rms_ratio_vs_zero"])
                                if row["rms_ratio_vs_zero"] is not None
                                else None
                            ),
                            "ratio": row["rms_ratio_vs_zero"],
                            "status": (
                                "ok"
                                if row["skill_status"] == "ok"
                                else row["skill_status"]
                            ),
                        }
                    )
    return output


def _metric_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair: str,
    baseline_policy: str,
    candidate_policy: str,
) -> list[dict[str, Any]]:
    baseline = {
        (row["case_id"], int(row["input_call"])): row
        for row in rows
        if row["policy"] == baseline_policy
    }
    candidate = {
        (row["case_id"], int(row["input_call"])): row
        for row in rows
        if row["policy"] == candidate_policy
    }
    expected = {(case_id, call) for case_id in CASE_IDS for call in INPUT_CALLS}
    baseline_count = sum(row["policy"] == baseline_policy for row in rows)
    candidate_count = sum(row["policy"] == candidate_policy for row in rows)
    if (
        set(baseline) != expected
        or set(candidate) != expected
        or baseline_count != len(expected)
        or candidate_count != len(expected)
    ):
        raise ValueError("A33 paired metric inventory differs")
    scopes = [("population", CASE_IDS)]
    scopes.extend((f"group_{group}", cases) for group, cases in GROUP_CASES.items())
    scopes.extend((case_id, (case_id,)) for case_id in CASE_IDS)
    endpoint_keys = (
        "rank8_state_error",
        "boundary_state_error",
        "shock_state_error",
        "vortex_state_error",
        "smooth_state_error",
        "front_position",
        "front_strength_log_ratio",
        "front_thickness_log_ratio",
        *(f"component_{name}_state_error" for name in INTEGRAL_COMPONENT_NAMES),
    )
    output = []
    for key in endpoint_keys:
        for scope, cases in scopes:
            for cell, calls in (
                ("overall", INPUT_CALLS),
                ("endpoint_29", (INPUT_CALLS[-1],)),
            ):
                control = fine_eval._rms_control(
                    key=f"{cell}::{key}",
                    scope=scope,
                    rows=[
                        {
                            "zero_error": baseline[(case_id, call)][key],
                            "corrected_error": candidate[(case_id, call)][key],
                        }
                        for case_id in cases
                        for call in calls
                    ],
                )
                output.append(
                    {"pair": pair, "kind": "structure", "cell": cell, **control}
                )
    for name in INTEGRAL_COMPONENT_NAMES:
        key = f"integral_{name}_error"
        for scope, cases in scopes:
            for cell, calls in (
                ("overall", INPUT_CALLS),
                ("endpoint_29", (INPUT_CALLS[-1],)),
            ):
                control = fine_eval._rms_control(
                    key=f"{cell}::integral::{name}",
                    scope=scope,
                    rows=[
                        {
                            "zero_error": baseline[(case_id, call)][key],
                            "corrected_error": candidate[(case_id, call)][key],
                        }
                        for case_id in cases
                        for call in calls
                    ],
                )
                output.append(
                    {"pair": pair, "kind": "integral", "cell": cell, **control}
                )
    return output


def _ratio_from_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    baseline_policy: str,
    candidate_policy: str,
    key: str,
    endpoint: bool = False,
) -> dict[str, Any]:
    calls = (INPUT_CALLS[-1],) if endpoint else INPUT_CALLS
    selected = [
        {
            "zero_error": next(
                row[key]
                for row in rows
                if row["case_id"] == case_id
                and row["policy"] == baseline_policy
                and row["input_call"] == call
            ),
            "corrected_error": next(
                row[key]
                for row in rows
                if row["case_id"] == case_id
                and row["policy"] == candidate_policy
                and row["input_call"] == call
            ),
        }
        for call in calls
    ]
    return fine_eval._rms_control(key=key, scope=case_id, rows=selected)


def _case_pair_rows(
    score_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    pair: str,
    baseline_policy: str,
    candidate_policy: str,
) -> list[dict[str, Any]]:
    output = []
    for case_id in CASE_IDS:
        output.append(
            {
                "pair": pair,
                "case_id": case_id,
                "trajectory_full_ratio": _score_row(
                    score_rows,
                    pair=pair,
                    cell="overall",
                    view="full",
                    scope="case",
                    case_id=case_id,
                )["rms_ratio_vs_zero"],
                "trajectory_rank8_ratio": _score_row(
                    score_rows,
                    pair=pair,
                    cell="overall",
                    view="rank8_parallel",
                    scope="case",
                    case_id=case_id,
                )["rms_ratio_vs_zero"],
                "endpoint_full_ratio": _score_row(
                    score_rows,
                    pair=pair,
                    cell="endpoint_29",
                    view="full",
                    scope="case",
                    case_id=case_id,
                )["rms_ratio_vs_zero"],
                "endpoint_rank8_ratio": _score_row(
                    score_rows,
                    pair=pair,
                    cell="endpoint_29",
                    view="rank8_parallel",
                    scope="case",
                    case_id=case_id,
                )["rms_ratio_vs_zero"],
                "increment_defect_ratio": _ratio_from_metric_rows(
                    metric_rows,
                    case_id=case_id,
                    baseline_policy=baseline_policy,
                    candidate_policy=candidate_policy,
                    key="increment_defect",
                )["ratio"],
                "endpoint_cumulative_defect_ratio": _ratio_from_metric_rows(
                    metric_rows,
                    case_id=case_id,
                    baseline_policy=baseline_policy,
                    candidate_policy=candidate_policy,
                    key="cumulative_defect",
                    endpoint=True,
                )["ratio"],
            }
        )
    return output


def _weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    a = np.asarray(left, dtype=np.float64) / scale[None, :]
    b = np.asarray(right, dtype=np.float64) / scale[None, :]
    return float(np.einsum("n,nc,nc->", mass, a, b, optimize=True) / mass.sum())


def _transfer_floor_rows(
    runtime: Any,
    *,
    case_id: str,
    query_truth: np.ndarray,
    native_truth: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], float]:
    query_volumes = np.asarray(
        runtime.geometry_by_resolution[QUERY_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    native_volumes = np.asarray(
        runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    rows = []
    maximum = 0.0
    for output_call in OUTPUT_CALLS:
        fields = transfer_floor_fields(
            query_truth[output_call],
            native_truth[output_call],
            query_resolution=QUERY_RESOLUTION,
            native_resolution=NATIVE_RESOLUTION,
        )
        row = base._floor_row(
            case_id=case_id,
            input_call=max(output_call - 1, 0),
            object_kind="state",
            direction="500x200_to_250x100",
            fields=fields,
            query_volumes=query_volumes,
            native_volumes=native_volumes,
            component_scale=state_scale,
        )
        row["output_call"] = output_call
        rows.append(row)
        maximum = max(maximum, float(row["closure_max_abs"]))
    for input_call in INPUT_CALLS:
        fields = transfer_floor_fields(
            query_truth[input_call + 1] - query_truth[input_call],
            native_truth[input_call + 1] - native_truth[input_call],
            query_resolution=QUERY_RESOLUTION,
            native_resolution=NATIVE_RESOLUTION,
        )
        row = base._floor_row(
            case_id=case_id,
            input_call=input_call,
            object_kind="true_increment",
            direction="500x200_to_250x100",
            fields=fields,
            query_volumes=query_volumes,
            native_volumes=native_volumes,
            component_scale=residual_scale,
        )
        row["output_call"] = input_call + 1
        rows.append(row)
        maximum = max(maximum, float(row["closure_max_abs"]))
    return rows, maximum


def _trajectory_identity_rows(
    *,
    case_id: str,
    policy: str,
    states: Sequence[np.ndarray],
    truth_states: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    if len(states) != len(truth_states):
        raise ValueError("trajectory identity requires aligned states and truth")
    initial_error = np.asarray(states[0]) - np.asarray(truth_states[0])
    cumulative = np.zeros_like(initial_error, dtype=np.float64)
    output = []
    maximum = 0.0
    for output_call in OUTPUT_CALLS:
        if output_call:
            cumulative = cumulative + (
                np.asarray(states[output_call])
                - np.asarray(states[output_call - 1])
                - np.asarray(truth_states[output_call])
                + np.asarray(truth_states[output_call - 1])
            )
        error = np.asarray(states[output_call]) - np.asarray(truth_states[output_call])
        closure = error - initial_error - cumulative
        value = _maximum_abs(closure)
        maximum = max(maximum, value)
        output.append(
            {
                "case_id": case_id,
                "policy": policy,
                "output_call": output_call,
                "closure_max_abs": value,
            }
        )
    return output, maximum


def _transfer_decomposition_rows(
    runtime: Any,
    *,
    case_id: str,
    policy: str,
    native_states: Sequence[np.ndarray],
    query_states: Sequence[np.ndarray],
    query_truth: np.ndarray,
    native_truth: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], float]:
    query_volumes = np.asarray(
        runtime.geometry_by_resolution[QUERY_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    output = []
    maximum = 0.0
    for output_call in OUTPUT_CALLS:
        evolution_native = np.asarray(native_states[output_call]) - np.asarray(
            native_truth[output_call]
        )
        evolution_query = prolong_nested_state(
            evolution_native,
            coarse_resolution=NATIVE_RESOLUTION,
            fine_resolution=QUERY_RESOLUTION,
        )
        round_trip = prolong_nested_state(
            native_truth[output_call],
            coarse_resolution=NATIVE_RESOLUTION,
            fine_resolution=QUERY_RESOLUTION,
        ) - np.asarray(query_truth[output_call])
        total = np.asarray(query_states[output_call]) - np.asarray(
            query_truth[output_call]
        )
        closure = total - evolution_query - round_trip
        closure_max = _maximum_abs(closure)
        maximum = max(maximum, closure_max)
        evolution_norm = weighted_scaled_rms(
            evolution_query,
            volumes=query_volumes,
            component_scale=state_scale,
        )
        round_trip_norm = weighted_scaled_rms(
            round_trip,
            volumes=query_volumes,
            component_scale=state_scale,
        )
        inner = _weighted_inner(
            evolution_query,
            round_trip,
            volumes=query_volumes,
            component_scale=state_scale,
        )
        denominator = evolution_norm * round_trip_norm
        output.append(
            {
                "case_id": case_id,
                "policy": policy,
                "output_call": output_call,
                "query_error_scaled_rms": weighted_scaled_rms(
                    total,
                    volumes=query_volumes,
                    component_scale=state_scale,
                ),
                "mapped_evolution_error_scaled_rms": evolution_norm,
                "round_trip_floor_scaled_rms": round_trip_norm,
                "evolution_round_trip_inner": inner,
                "evolution_round_trip_cosine": (
                    inner / denominator if denominator > DENOMINATOR_FLOOR**2 else None
                ),
                "evolution_round_trip_cosine_denominator": denominator,
                "evolution_round_trip_cosine_status": (
                    "ok"
                    if denominator > DENOMINATOR_FLOOR**2
                    else "unresolved_small_denominator"
                ),
                "closure_max_abs": closure_max,
            }
        )
    return output, maximum


def _correction_gate(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    tether_rows: Sequence[Mapping[str, Any]],
    shadow_raw_max_abs: Mapping[str, float],
    prefix_repeat_abs: float,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    checks = dict(structural_checks)
    checks["case_ratio_inventory_exact"] = len(case_rows) == len(CASE_IDS) and {
        str(row.get("case_id")) for row in case_rows
    } == set(CASE_IDS)
    checks["population_full_rank8_trajectory_strict"] = all(
        _score_row(
            score_rows,
            pair=PAIR_CORRECTION,
            cell="overall",
            view=view,
            scope="population",
        )["skill_status"]
        == "ok"
        and float(
            _score_row(
                score_rows,
                pair=PAIR_CORRECTION,
                cell="overall",
                view=view,
                scope="population",
            )["rms_ratio_vs_zero"]
        )
        < 1.0
        for view in ("full", "rank8_parallel")
    )
    checks["population_full_rank8_endpoint_strict"] = all(
        _score_row(
            score_rows,
            pair=PAIR_CORRECTION,
            cell="endpoint_29",
            view=view,
            scope="population",
        )["skill_status"]
        == "ok"
        and float(
            _score_row(
                score_rows,
                pair=PAIR_CORRECTION,
                cell="endpoint_29",
                view=view,
                scope="population",
            )["rms_ratio_vs_zero"]
        )
        < 1.0
        for view in ("full", "rank8_parallel")
    )

    def strict_win(value: Any) -> bool:
        return value is not None and np.isfinite(float(value)) and float(value) < 1.0

    joint_wins = sum(
        strict_win(row["trajectory_full_ratio"])
        and strict_win(row["endpoint_full_ratio"])
        for row in case_rows
    )
    checks["at_least_four_case_trajectory_endpoint_wins"] = joint_wins >= 4
    checks["every_case_full_rank8_primary_no_harm"] = all(
        row[key] is not None and float(row[key]) <= CONTROL_LIMIT
        for row in case_rows
        for key in (
            "trajectory_full_ratio",
            "trajectory_rank8_ratio",
            "endpoint_full_ratio",
            "endpoint_rank8_ratio",
        )
    )
    checks["all_controls_no_harm"] = bool(controls) and all(
        _control_passed(row) for row in controls
    )
    checks["all_transfer_rollouts_complete_finite_admissible"] = (
        len(rollouts) == 2 * len(CASE_IDS)
        and {(str(row.get("case_id")), str(row.get("policy"))) for row in rollouts}
        == {
            (case_id, policy)
            for case_id in CASE_IDS
            for policy in (RAW_TRANSFER_POLICY, CORRECTED_TRANSFER_POLICY)
        }
        and all(
            row["execution"]["completed_calls"] == 30
            and all(metric["finite"] and metric["admissible"] for metric in row["rows"])
            for row in rollouts
        )
    )
    active_audits = [row for row in audit_rows if row["correction_active"]]
    coast_audits = [row for row in audit_rows if not row["correction_active"]]
    checks["active_audit_inventory_96"] = len(active_audits) == 96
    checks["coast_audit_inventory_84"] = len(coast_audits) == 84
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
    checks["coast_proposal_closure"] = all(
        row["status"] == "inactive_exact_raw_native"
        and bool(row["call_order_exact"])
        and int(row["logical_call_count"]) == 1
        and float(row["correction_rms"]) == 0.0
        and float(row["state_update_closure_max_abs"]) == 0.0
        for row in coast_audits
    )
    checks["tether_inventory_180"] = len(tether_rows) == 180
    checks["tether_closure"] = all(
        row["status"] == "ok"
        and bool(row["call_order_exact"])
        and int(row["total_logical_call_count"])
        == (3 if row["correction_active"] else 2)
        and float(row["maximum_integral_difference_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_boundary_difference_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_projection_idempotence_abs"]) <= STRICT_TOLERANCE
        and float(row["maximum_update_identity_abs"]) <= STRICT_TOLERANCE
        for row in tether_rows
    )
    checks["shadow_bitwise_raw"] = set(shadow_raw_max_abs) == set(CASE_IDS) and all(
        value == 0.0 for value in shadow_raw_max_abs.values()
    )
    checks["deterministic_prefix"] = prefix_repeat_abs <= REPEAT_TOLERANCE
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "case_joint_win_count": joint_wins,
        "strict_improvement_equality_fails": True,
        "control_ratio_limit": CONTROL_LIMIT,
    }


def _direct_gate(
    *,
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    raw_ratios = [row.get("endpoint_full_ratio") for row in case_rows]
    resolved = all(
        value is not None and np.isfinite(float(value)) for value in raw_ratios
    )
    ratios = [float(value) for value in raw_ratios] if resolved else []
    median = float(np.median(ratios)) if resolved else None
    wins = sum(value < 1.0 for value in ratios)
    loo = {
        row["case_id"]: (
            float(
                np.median(
                    [value for index, value in enumerate(ratios) if index != held]
                )
            )
            if resolved
            else None
        )
        for held, row in enumerate(case_rows)
    }
    checks = {
        **structural_checks,
        "case_ratio_inventory_exact": len(case_rows) == len(CASE_IDS)
        and {str(row.get("case_id")) for row in case_rows} == set(CASE_IDS),
        "median_endpoint_ratio_at_most_0p95": median is not None and median <= 0.95,
        "at_least_four_case_wins": wins >= 4,
        "all_controls_no_harm": bool(controls)
        and all(_control_passed(row) for row in controls),
        "endpoint_denominators_resolved": resolved,
    }
    failed = sorted(key for key, value in checks.items() if not bool(value))
    return {
        "status": "qualified" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "median_endpoint_state_ratio": median,
        "case_win_count": wins,
        "case_ratios": {
            row["case_id"]: (
                float(row["endpoint_full_ratio"])
                if row.get("endpoint_full_ratio") is not None
                and np.isfinite(float(row["endpoint_full_ratio"]))
                else None
            )
            for row in case_rows
        },
        "leave_one_case_out_medians": loo,
        "median_threshold": 0.95,
        "control_ratio_limit": CONTROL_LIMIT,
    }


def _write_animation_bundle(
    output_dir: Path,
    *,
    runtime: Any,
    case_id: str,
    truth_states: np.ndarray,
    direct_states: Sequence[np.ndarray],
    matched_states: Sequence[np.ndarray],
    raw_transfer_states: Sequence[np.ndarray],
    corrected_transfer_states: Sequence[np.ndarray],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = np.asarray(ANIMATION_CALLS, dtype=np.int64)
    geometry = runtime.geometry_by_resolution[QUERY_RESOLUTION]
    path = output_dir / f"{case_id}_query_grid_h30.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        split_group_id=np.asarray(f"strength_{CASE_TO_GROUP[case_id]}"),
        output_calls=frame_indices,
        physical_times=np.asarray(
            ANIMATION_CONTRACT["physical_times"], dtype=np.float64
        ),
        query_resolution=np.asarray(QUERY_RESOLUTION, dtype=np.int64),
        nodes=np.asarray(geometry.nodes, dtype=np.float32),
        volumes=np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1),
        state_scale=np.asarray(runtime.normalization.state_scale, dtype=np.float32),
        residual_scale=np.asarray(
            runtime.normalization.residual_scale, dtype=np.float32
        ),
        gamma=np.asarray(runtime.normalization.gamma, dtype=np.float64),
        truth_conservative=np.asarray(truth_states, dtype=np.float64)[
            frame_indices
        ].astype(np.float32),
        direct_query_conservative=np.asarray(direct_states, dtype=np.float64)[
            frame_indices
        ].astype(np.float32),
        matched_direct_conservative=np.asarray(matched_states, dtype=np.float64)[
            frame_indices
        ].astype(np.float32),
        raw_transfer_conservative=np.asarray(raw_transfer_states, dtype=np.float64)[
            frame_indices
        ].astype(np.float32),
        corrected_transfer_conservative=np.asarray(
            corrected_transfer_states, dtype=np.float64
        )[frame_indices].astype(np.float32),
    )
    return {
        "case_id": case_id,
        "split_group_id": f"strength_{CASE_TO_GROUP[case_id]}",
        "path": path.name,
        "sha256": sha256_file(path),
        "frames": len(frame_indices),
        "storage_dtype": "float32_visualization_only",
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


def _arm_error_cost(
    metric_rows: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        case_trajectory = []
        case_endpoint = []
        for case_id in CASE_IDS:
            values = [
                float(row["state_error"])
                for row in metric_rows
                if row["policy"] == policy and row["case_id"] == case_id
            ]
            if len(values) != len(INPUT_CALLS) or not np.isfinite(values).all():
                raise ValueError("A33 arm error-cost inventory differs")
            case_trajectory.append(float(np.mean(np.square(values))))
            case_endpoint.append(values[-1] ** 2)
        mapping_seconds = {
            DIRECT_QUERY_POLICY: 0.0,
            MATCHED_DIRECT_POLICY: float(
                execution["mapping"]["matched_initialization_seconds"]
            ),
            RAW_TRANSFER_POLICY: float(
                execution["mapping"]["raw_transfer_output_seconds"]
            ),
            CORRECTED_TRANSFER_POLICY: float(
                execution["mapping"]["corrected_transfer_output_seconds"]
            ),
        }[policy]
        model_seconds = float(execution[policy]["total_forward_seconds"])
        output[policy] = {
            "case_first_trajectory_state_rms": float(np.sqrt(np.mean(case_trajectory))),
            "case_first_h30_state_rms": float(np.sqrt(np.mean(case_endpoint))),
            "logical_model_calls": int(execution[policy]["logical_model_calls"]),
            "actual_forward_passes": int(execution[policy]["actual_forward_passes"]),
            "model_forward_seconds": model_seconds,
            "required_mapping_seconds": mapping_seconds,
            "model_plus_required_mapping_seconds": model_seconds + mapping_seconds,
            "arm_wall_seconds": float(execution[policy]["wall_seconds"]),
        }
    raw = output[RAW_TRANSFER_POLICY]
    for row in output.values():
        row["trajectory_error_ratio_vs_raw_transfer"] = (
            row["case_first_trajectory_state_rms"]
            / raw["case_first_trajectory_state_rms"]
        )
        row["h30_error_ratio_vs_raw_transfer"] = (
            row["case_first_h30_state_rms"] / raw["case_first_h30_state_rms"]
        )
        row["logical_call_ratio_vs_raw_transfer"] = (
            row["logical_model_calls"] / raw["logical_model_calls"]
        )
        row["model_forward_ratio_vs_raw_transfer"] = (
            row["model_forward_seconds"] / raw["model_forward_seconds"]
        )
    return output


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_preflight(args.preflight, args)
    _verify_a32(args.a32_result)
    start_coefficients, end_coefficients, _ = a31._verify_a28(args.a28_map)
    _, expected_reference_rows = _verify_d074(
        args.d074_summary, args.d074_reference_checks
    )
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
            raise ValueError("A33 runtime source differs from preflight")
        query_projector = _query_projector(runtime)
        native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]

        descriptor_rows = []
        descriptors = {}
        for case_id in CASE_IDS:
            routing_reference, routing_check = load_resolution_reference(
                args.family_root,
                args.multires_reference_root,
                runtime.store,
                runtime.manifest,
                case_id,
                training_resolution=NATIVE_RESOLUTION,
            )
            _require_reference_row(routing_check, expected_reference_rows[case_id])
            initial = reference_at_resolution(
                routing_reference["conservative_states"][0],
                reference_resolution=QUERY_RESOLUTION,
                target_resolution=NATIVE_RESOLUTION,
            )
            if initial is None:
                raise ValueError("A33 routing state cannot restrict to native")
            initial = np.asarray(initial, dtype=np.float64)
            shard_initial = np.asarray(
                runtime.store.states(case_id)[0], dtype=np.float64
            )
            routing_crosscheck = _maximum_abs(initial - shard_initial)
            descriptor = transverse_velocity_position_descriptor(
                initial,
                nodes=np.asarray(native_geometry.nodes, dtype=np.float64),
                volumes=np.asarray(native_geometry.node_measures, dtype=np.float64),
                y_min=float(runtime.first_config.y_min),
                y_max=float(runtime.first_config.y_max),
            )
            selected = bool(
                descriptor.status == "ok"
                and descriptor.normalized_wall_distance is not None
                and descriptor.normalized_wall_distance
                <= FROZEN_POSITION_BUFFER_THRESHOLD
            )
            descriptors[case_id] = (descriptor, selected)
            descriptor_rows.append(
                {
                    "case_id": case_id,
                    "group_id": CASE_TO_GROUP[case_id],
                    **asdict(descriptor),
                    "position_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                    "position_selected": selected,
                    "expected_selected": case_id in EXPECTED_SELECTED_CASES,
                    "routing_native_vs_shard_max_abs": routing_crosscheck,
                    "source": (
                        "authenticated_query_call_zero_conservative_restriction_"
                        "before_model_calls"
                    ),
                }
            )
            del routing_reference, initial, shard_initial
        selected_cases = tuple(
            row["case_id"] for row in descriptor_rows if row["position_selected"]
        )
        if selected_cases != EXPECTED_SELECTED_CASES:
            raise ValueError("A33 selector inventory differs before model calls")

        prefix_reference, prefix_check = load_resolution_reference(
            args.family_root,
            args.multires_reference_root,
            runtime.store,
            runtime.manifest,
            CASE_IDS[0],
            training_resolution=NATIVE_RESOLUTION,
        )
        _require_reference_row(prefix_check, expected_reference_rows[CASE_IDS[0]])
        prefix_descriptor, prefix_selected = descriptors[CASE_IDS[0]]
        prefix_runs = [
            a32._run_tethered_arm(
                runtime,
                case_id=CASE_IDS[0],
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
            _maximum_abs(left - right)
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
        del prefix_reference, prefix_runs

        rollouts = []
        metric_rows = []
        view_records = []
        view_maxima = {
            pair: {"band": 0.0, "subspace_reconstruction": 0.0, "orthogonality": 0.0}
            for pair in (
                PAIR_CORRECTION,
                PAIR_DIRECT_RAW,
                PAIR_DIRECT_CORRECTED,
                PAIR_MATCHED_DIRECT,
            )
        }
        reference_rows = []
        floor_rows = []
        identity_rows = []
        decomposition_rows = []
        audit_rows = []
        tether_rows = []
        shadow_raw_max_abs = {}
        mapping_rows = []
        animation_rows = []
        animation_dir = output_dir / "animation_bundles"
        maximum_floor_closure = 0.0
        maximum_identity_closure = 0.0
        maximum_decomposition_closure = 0.0

        for case_index, case_id in enumerate(CASE_IDS, start=1):
            print(f"A33 case {case_index}/{len(CASE_IDS)}: {case_id}", flush=True)
            reference, reference_check = load_resolution_reference(
                args.family_root,
                args.multires_reference_root,
                runtime.store,
                runtime.manifest,
                case_id,
                training_resolution=NATIVE_RESOLUTION,
            )
            _require_reference_row(reference_check, expected_reference_rows[case_id])
            query_truth = np.asarray(
                reference["conservative_states"][::2], dtype=np.float64
            )
            if query_truth.shape != (31, 100000, 4):
                raise ValueError("A33 query truth shape differs")
            native_reference, native_reference_check = load_resolution_reference(
                args.family_root,
                None,
                runtime.store,
                runtime.manifest,
                case_id,
                training_resolution=NATIVE_RESOLUTION,
            )
            restricted_query_truth = [
                np.asarray(
                    reference_at_resolution(
                        state,
                        reference_resolution=QUERY_RESOLUTION,
                        target_resolution=NATIVE_RESOLUTION,
                    ),
                    dtype=np.float64,
                )
                for state in query_truth
            ]
            evolved_native_truth = np.asarray(
                native_reference["conservative_states"][::2], dtype=np.float64
            )
            if evolved_native_truth.shape != (31, 25000, 4):
                raise ValueError("A33 evolved native truth shape differs")
            native_reference_gap = _maximum_abs(
                np.asarray(restricted_query_truth) - evolved_native_truth
            )
            if (
                native_reference_check["case_id"] != case_id
                or native_reference_check["frozen_training_reference_sha256"]
                != reference_check["frozen_training_reference_sha256"]
                or native_reference_check["active_reference_artifact_sha256"]
                != reference_check["frozen_training_reference_sha256"]
                or native_reference_check["retained_resolution"] != "250x100"
                or native_reference_check["restriction_crosscheck_max_abs"] != 0.0
                or native_reference_gap > STRICT_TOLERANCE
            ):
                raise ValueError("query restriction does not reproduce native truth")
            descriptor, selected = descriptors[case_id]
            restricted_descriptor = transverse_velocity_position_descriptor(
                restricted_query_truth[0],
                nodes=np.asarray(native_geometry.nodes, dtype=np.float64),
                volumes=np.asarray(native_geometry.node_measures, dtype=np.float64),
                y_min=float(runtime.first_config.y_min),
                y_max=float(runtime.first_config.y_max),
            )
            restricted_selected = bool(
                restricted_descriptor.status == "ok"
                and restricted_descriptor.normalized_wall_distance is not None
                and restricted_descriptor.normalized_wall_distance
                <= FROZEN_POSITION_BUFFER_THRESHOLD
            )
            if restricted_selected != selected:
                raise ValueError("query-restricted and shard selector decisions differ")

            map_started = perf_counter()
            matched_initial = prolong_nested_state(
                restricted_query_truth[0],
                coarse_resolution=NATIVE_RESOLUTION,
                fine_resolution=QUERY_RESOLUTION,
            )
            initialization_map_seconds = perf_counter() - map_started
            direct = _run_direct_arm(
                runtime,
                policy=DIRECT_QUERY_POLICY,
                initial_state=query_truth[0],
                horizon=30,
            )
            matched = _run_direct_arm(
                runtime,
                policy=MATCHED_DIRECT_POLICY,
                initial_state=matched_initial,
                horizon=30,
            )
            raw_native = a31._run_arm(
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
            corrected_native = a32._run_tethered_arm(
                runtime,
                case_id=case_id,
                reference=reference,
                position_selected=selected,
                descriptor=descriptor,
                start_coefficients=start_coefficients,
                end_coefficients=end_coefficients,
                horizon=30,
            )
            raw_query, raw_map_seconds = _map_native_states(raw_native["states"])
            corrected_query, corrected_map_seconds = _map_native_states(
                corrected_native["states"]
            )
            shadow_query, shadow_map_seconds = _map_native_states(
                corrected_native["shadow_states"]
            )
            mapping_rows.append(
                {
                    "case_id": case_id,
                    "matched_initialization_seconds": initialization_map_seconds,
                    "raw_transfer_output_seconds": raw_map_seconds,
                    "corrected_transfer_output_seconds": corrected_map_seconds,
                    "corrected_shadow_audit_seconds": shadow_map_seconds,
                }
            )

            policy_states = {
                DIRECT_QUERY_POLICY: direct["states"],
                MATCHED_DIRECT_POLICY: matched["states"],
                RAW_TRANSFER_POLICY: raw_query,
                CORRECTED_TRANSFER_POLICY: corrected_query,
            }
            for policy, states in policy_states.items():
                rows = _query_metric_rows(
                    runtime,
                    query_projector=query_projector,
                    case_id=case_id,
                    policy=policy,
                    states=states,
                    truth_states=query_truth,
                    shadow_states=(
                        shadow_query if policy == CORRECTED_TRANSFER_POLICY else None
                    ),
                )
                metric_rows.extend(rows)
                identity, closure = _trajectory_identity_rows(
                    case_id=case_id,
                    policy=policy,
                    states=states,
                    truth_states=query_truth,
                )
                identity_rows.extend(identity)
                maximum_identity_closure = max(maximum_identity_closure, closure)

            pairs = {
                PAIR_CORRECTION: (raw_query, corrected_query),
                PAIR_DIRECT_RAW: (raw_query, direct["states"]),
                PAIR_DIRECT_CORRECTED: (corrected_query, direct["states"]),
                PAIR_MATCHED_DIRECT: (direct["states"], matched["states"]),
            }
            for pair, (baseline_states, candidate_states) in pairs.items():
                records, maxima = _pair_view_records(
                    runtime,
                    query_projector=query_projector,
                    case_id=case_id,
                    pair=pair,
                    baseline_states=baseline_states,
                    candidate_states=candidate_states,
                    truth_states=query_truth,
                )
                view_records.extend(records)
                for key, value in maxima.items():
                    view_maxima[pair][key] = max(view_maxima[pair][key], value)

            floors, floor_closure = _transfer_floor_rows(
                runtime,
                case_id=case_id,
                query_truth=query_truth,
                native_truth=evolved_native_truth,
            )
            floor_rows.extend(floors)
            maximum_floor_closure = max(maximum_floor_closure, floor_closure)
            for policy, native_states, query_states in (
                (RAW_TRANSFER_POLICY, raw_native["states"], raw_query),
                (
                    CORRECTED_TRANSFER_POLICY,
                    corrected_native["states"],
                    corrected_query,
                ),
            ):
                rows, closure = _transfer_decomposition_rows(
                    runtime,
                    case_id=case_id,
                    policy=policy,
                    native_states=native_states,
                    query_states=query_states,
                    query_truth=query_truth,
                    native_truth=restricted_query_truth,
                )
                decomposition_rows.extend(rows)
                maximum_decomposition_closure = max(
                    maximum_decomposition_closure, closure
                )

            shadow_raw_max_abs[case_id] = max(
                _maximum_abs(left - right)
                for left, right in zip(
                    raw_native["states"],
                    corrected_native["shadow_states"],
                    strict=True,
                )
            )
            if case_id in ANIMATION_CASE_IDS:
                animation_rows.append(
                    _write_animation_bundle(
                        animation_dir,
                        runtime=runtime,
                        case_id=case_id,
                        truth_states=query_truth,
                        direct_states=direct["states"],
                        matched_states=matched["states"],
                        raw_transfer_states=raw_query,
                        corrected_transfer_states=corrected_query,
                    )
                )
            reference_rows.append(
                {
                    "case_id": case_id,
                    **reference_check,
                    "loaded_native_reference_crosscheck_max_abs": native_reference_gap,
                    "loaded_native_reference_sha256": native_reference_check[
                        "active_reference_artifact_sha256"
                    ],
                }
            )
            audit_rows.extend(corrected_native["audit_rows"])
            tether_rows.extend(corrected_native["tether_rows"])
            raw_native["policy"] = RAW_TRANSFER_POLICY
            corrected_native["policy"] = CORRECTED_TRANSFER_POLICY
            raw_native["rows"] = [
                row
                for row in metric_rows
                if row["case_id"] == case_id and row["policy"] == RAW_TRANSFER_POLICY
            ]
            corrected_native["rows"] = [
                row
                for row in metric_rows
                if row["case_id"] == case_id
                and row["policy"] == CORRECTED_TRANSFER_POLICY
            ]
            direct["rows"] = [
                row
                for row in metric_rows
                if row["case_id"] == case_id and row["policy"] == DIRECT_QUERY_POLICY
            ]
            matched["rows"] = [
                row
                for row in metric_rows
                if row["case_id"] == case_id and row["policy"] == MATCHED_DIRECT_POLICY
            ]
            for rollout in (direct, matched, raw_native, corrected_native):
                rollout["case_id"] = case_id
                rollout.pop("states")
                rollout.pop("audit_rows", None)
                rollout.pop("shadow_states", None)
                rollout.pop("tether_rows", None)
                rollouts.append(rollout)
            del (
                reference,
                native_reference,
                query_truth,
                restricted_query_truth,
                evolved_native_truth,
                direct,
                matched,
                raw_native,
                corrected_native,
                raw_query,
                corrected_query,
                shadow_query,
            )
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        score_rows = []
        controls_by_pair = {}
        case_rows_by_pair = {}
        policy_by_pair = {
            PAIR_CORRECTION: (RAW_TRANSFER_POLICY, CORRECTED_TRANSFER_POLICY),
            PAIR_DIRECT_RAW: (RAW_TRANSFER_POLICY, DIRECT_QUERY_POLICY),
            PAIR_DIRECT_CORRECTED: (
                CORRECTED_TRANSFER_POLICY,
                DIRECT_QUERY_POLICY,
            ),
            PAIR_MATCHED_DIRECT: (DIRECT_QUERY_POLICY, MATCHED_DIRECT_POLICY),
        }
        for pair, (baseline_policy, candidate_policy) in policy_by_pair.items():
            pair_scores = _score_views(view_records, pair=pair)
            score_rows.extend(pair_scores)
            controls = [
                *_field_controls(pair_scores, pair=pair),
                *_metric_controls(
                    metric_rows,
                    pair=pair,
                    baseline_policy=baseline_policy,
                    candidate_policy=candidate_policy,
                ),
            ]
            controls_by_pair[pair] = controls
            case_rows_by_pair[pair] = _case_pair_rows(
                pair_scores,
                metric_rows,
                pair=pair,
                baseline_policy=baseline_policy,
                candidate_policy=candidate_policy,
            )

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
        corrected_keys = (
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
        execution = {
            policy: _sum_execution(
                rollouts,
                policy,
                corrected_keys if policy == CORRECTED_TRANSFER_POLICY else common_keys,
            )
            for policy in POLICIES
        }
        execution["mapping"] = {
            key: sum(float(row[key]) for row in mapping_rows)
            for key in (
                "matched_initialization_seconds",
                "raw_transfer_output_seconds",
                "corrected_transfer_output_seconds",
                "corrected_shadow_audit_seconds",
            )
        }
        execution["maximum_peak_gpu_memory_bytes"] = max(
            int(row["execution"]["maximum_peak_gpu_memory_bytes"]) for row in rollouts
        )
        execution["wall_seconds"] = perf_counter() - started
        execution["environment"] = runtime_environment(runtime.device)
        execution["logical_model_calls_total"] = sum(
            execution[policy]["logical_model_calls"] for policy in POLICIES
        )
        execution["corrected_to_raw_transfer_logical_ratio"] = (
            execution[CORRECTED_TRANSFER_POLICY]["logical_model_calls"]
            / execution[RAW_TRANSFER_POLICY]["logical_model_calls"]
        )
        execution["cached_raw_shadow_sharing_lower_bound"] = {
            "primary_cost_accounting_uses_sharing": False,
            "assumption": (
                "reuse the separately computed raw-transfer native predictions as "
                "the bitwise-equal A32 safety shadow"
            ),
            "all_arm_logical_calls": 816,
            "raw_plus_corrected_unique_logical_calls": 456,
            "corrected_marginal_logical_calls_after_raw_cache": 276,
            "measured_time_not_imputed": True,
        }
        error_cost = _arm_error_cost(metric_rows, execution)

        structural = {
            "source_exact": preflight["source_manifest"]["source_sha256"]
            == _source_hashes(),
            "population_exact": selected_cases == EXPECTED_SELECTED_CASES,
            "reference_inventory_exact": len(reference_rows) == len(CASE_IDS),
            "descriptor_inventory_exact": len(descriptor_rows) == len(CASE_IDS),
            "routing_native_reference_crosscheck": max(
                float(row["routing_native_vs_shard_max_abs"]) for row in descriptor_rows
            )
            <= STRICT_TOLERANCE,
            "query_metric_inventory_exact": len(metric_rows)
            == len(POLICIES) * len(CASE_IDS) * len(INPUT_CALLS),
            "transfer_floor_inventory_exact": len(floor_rows)
            == len(CASE_IDS) * (len(OUTPUT_CALLS) + len(INPUT_CALLS)),
            "animation_inventory_exact": len(animation_rows) == len(ANIMATION_CASE_IDS),
            "view_closure": all(
                value <= 1.0e-10
                for maxima in view_maxima.values()
                for value in maxima.values()
            ),
            "transfer_floor_closure": maximum_floor_closure <= STRICT_TOLERANCE,
            "trajectory_identity_closure": maximum_identity_closure <= STRICT_TOLERANCE,
            "transfer_decomposition_closure": maximum_decomposition_closure
            <= STRICT_TOLERANCE,
            "main_logical_call_inventory_exact": (
                execution[DIRECT_QUERY_POLICY]["logical_model_calls"] == 180
                and execution[MATCHED_DIRECT_POLICY]["logical_model_calls"] == 180
                and execution[RAW_TRANSFER_POLICY]["logical_model_calls"] == 180
                and execution[CORRECTED_TRANSFER_POLICY]["logical_model_calls"] == 456
                and execution["logical_model_calls_total"] == 996
            ),
            "coefficient_refit_not_run": True,
            "truth_or_case_id_not_in_routing": True,
            "D073_B_not_run": True,
        }
        correction_gate = _correction_gate(
            score_rows=score_rows,
            case_rows=case_rows_by_pair[PAIR_CORRECTION],
            controls=controls_by_pair[PAIR_CORRECTION],
            rollouts=[
                row
                for row in rollouts
                if row["policy"] in {RAW_TRANSFER_POLICY, CORRECTED_TRANSFER_POLICY}
            ],
            audit_rows=audit_rows,
            tether_rows=tether_rows,
            shadow_raw_max_abs=shadow_raw_max_abs,
            prefix_repeat_abs=prefix_repeat_abs,
            structural_checks=structural,
        )
        if correction_gate["status"] == "qualified":
            primary_transfer_policy = CORRECTED_TRANSFER_POLICY
            direct_pair = PAIR_DIRECT_CORRECTED
        else:
            primary_transfer_policy = RAW_TRANSFER_POLICY
            direct_pair = PAIR_DIRECT_RAW
        direct_gate = _direct_gate(
            case_rows=case_rows_by_pair[direct_pair],
            controls=controls_by_pair[direct_pair],
            structural_checks={
                **structural,
                "all_four_arms_complete_finite_admissible": len(rollouts)
                == len(POLICIES) * len(CASE_IDS)
                and all(
                    row["execution"]["completed_calls"] == 30
                    and all(
                        metric["finite"] and metric["admissible"]
                        for metric in row["rows"]
                    )
                    for row in rollouts
                ),
            },
        )

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
        write_csv(output_dir / "query_call_metrics.csv", metric_rows)
        write_csv(
            output_dir / "query_view_scores.csv",
            [
                {key: value for key, value in row.items() if key != "statistics"}
                for row in score_rows
            ],
        )
        write_csv(
            output_dir / "query_controls.csv",
            [row for pair in controls_by_pair.values() for row in pair],
        )
        write_csv(
            output_dir / "query_case_ratios.csv",
            [row for pair in case_rows_by_pair.values() for row in pair],
        )
        write_csv(output_dir / "transfer_floors.csv", floor_rows)
        write_csv(output_dir / "trajectory_identity.csv", identity_rows)
        write_csv(output_dir / "transfer_decomposition.csv", decomposition_rows)
        write_csv(output_dir / "correction_audits.csv", audit_rows)
        write_csv(output_dir / "tether_audits.csv", tether_rows)
        write_csv(output_dir / "position_decisions.csv", descriptor_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        write_csv(output_dir / "mapping_execution.csv", mapping_rows)
        write_csv(
            output_dir / "error_cost_comparison.csv",
            [{"policy": policy, **values} for policy, values in error_cost.items()],
        )
        write_csv(
            output_dir / "rollout_execution.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
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
                    "qualified_correction_and_direct_query"
                    if correction_gate["status"] == "qualified"
                    and direct_gate["status"] == "qualified"
                    else "completed_query_grid_comparator"
                ),
                "population_status": "already_open_d074_a2_adaptive_reuse",
                "correction_survival_gate": correction_gate,
                "direct_vs_transfer_gate": direct_gate,
                "primary_transfer_policy": primary_transfer_policy,
                "direct_gate_pair": direct_pair,
                "case_ratios": case_rows_by_pair,
                "primary_scores": {
                    pair: {
                        cell: {
                            view: dict(
                                _score_row(
                                    score_rows,
                                    pair=pair,
                                    cell=cell,
                                    view=view,
                                    scope="population",
                                )
                            )
                            for view in ("full", "rank8_parallel")
                        }
                        for cell in ("overall", "endpoint_29")
                    }
                    for pair in (
                        PAIR_CORRECTION,
                        direct_pair,
                        PAIR_MATCHED_DIRECT,
                    )
                },
                "failed_controls": {
                    pair: [dict(row) for row in controls if not _control_passed(row)]
                    for pair, controls in controls_by_pair.items()
                },
                "maximum_control_ratio": {
                    pair: max(
                        float(row["ratio"])
                        for row in controls
                        if row.get("ratio") is not None
                    )
                    for pair, controls in controls_by_pair.items()
                },
                "closure_maxima": {
                    "views": view_maxima,
                    "transfer_floor": maximum_floor_closure,
                    "trajectory_identity": maximum_identity_closure,
                    "transfer_decomposition": maximum_decomposition_closure,
                },
                "shadow_raw_maximum_state_abs_difference": shadow_raw_max_abs,
                "execution": execution,
                "error_cost_comparison": error_cost,
                "prefix_replay": prefix_execution,
                "a32_result_sha256": sha256_file(args.a32_result),
                "a32_payload_sha256": A32_PAYLOAD_SHA256,
                "d074_summary_sha256": sha256_file(args.d074_summary),
                "d074_reference_checks_sha256": sha256_file(args.d074_reference_checks),
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
                "fine_reference_loaded": True,
                "new_population_opened": False,
                "sealed_population_opened": False,
                "physical_radius_arm_run": False,
                "claim_boundary": (
                    "Adaptive reuse of the six already-open D074/A2 dynamic-FV "
                    "cases on retained 500x200 truth. No independent confirmation, "
                    "cross-family coefficient transfer, conservation, resolution "
                    "convergence, operator-learning, D073-B, asymptotic-order, or "
                    "Richardson claim."
                ),
                "git": git_state(ROOT),
            }
        )
        atomic_write_json(output_dir / "affine_shadow_tether_query_grid.json", result)
        passed = (
            correction_gate["status"] == "qualified"
            and direct_gate["status"] == "qualified"
        )
        return result, 0 if passed else 4
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
    parser.add_argument("--d074-summary", type=Path, required=True)
    parser.add_argument("--d074-reference-checks", type=Path, required=True)


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
