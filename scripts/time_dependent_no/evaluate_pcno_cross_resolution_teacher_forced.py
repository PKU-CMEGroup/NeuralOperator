#!/usr/bin/env python3
"""Run W26-L5 A2 calibration-first teacher-forced evaluation.

``calibrate`` can open only the frozen 18 calibration cases at input calls
0--19.  ``evaluate`` verifies a hashed, fully qualified calibration artifact
before constructing any reference loader, then reports the registered open
case/time cells and prospective gate.  This script never runs recurrence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_cross_resolution_correction import (
    synthetic_smoke_summary as a1_synthetic_smoke_summary,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    sha256_file,
    sha256_files,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    DiagnosticSnapshot,
    ResolutionContract,
    common_native_increment_basis,
    corrected_native_prediction,
    grouped_crossfit,
    mapped_increment_errors,
    prepare_common_native_inputs,
    prolong_nested_state,
    transfer_floor_fields,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    ALL_INPUT_CALLS,
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    CELL_CALLS,
    CELL_CASES,
    EVALUATION_CASE_IDS,
    FIT_INPUT_CALLS,
    FRONT_CONTROL_KEYS,
    HELD_OUT_INPUT_CALLS,
    INTEGRAL_COMPONENT_NAMES,
    REQUIRED_FIELD_CONTROL_VIEWS,
    CalibrationClosureEvidence,
    ControlRatio,
    ProposalEvidence,
    build_fixed_cosine_projector,
    error_relation_rows,
    prospective_gate,
    qualify_calibration,
    require_qualified_calibration,
    score_cell,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    parameter_count,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    checkpoint_step_stride,
    conservative_admissibility_summary,
    load_resolution_checkpoint,
    load_resolution_reference,
    make_model_sample,
    node_types_for_protocol,
    predict_resolution_sample,
    pressure_profile_shock_metrics,
    reference_at_resolution,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_runtime import select_device
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

WORKING_ID = "W26-L5-P1-A2"
READINESS_SCHEMA = "pcno_cross_resolution_teacher_forced_readiness_v1"
SOURCE_MANIFEST_SCHEMA = "pcno_cross_resolution_teacher_forced_source_manifest_v1"
CALIBRATION_SCHEMA = "pcno_cross_resolution_teacher_forced_calibration_v1"
EVALUATION_SCHEMA = "pcno_cross_resolution_teacher_forced_evaluation_v1"
RESOLUTION_CONTRACT = ResolutionContract(
    coarse=(125, 50),
    native=(250, 100),
    fine=(500, 200),
)
EXPECTED_UPSTREAM_SHA256 = {
    "checkpoint": "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9",
    "normalization_file": "4c931c813d318f9a3012814c9803cf85fbb30ce4c68739535047b2aff6f0faf4",
    "normalization_digest": "9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a",
    "split": "1be17494eaac3902159763a9e9a6c562d39c31957739899f50c28e37b48a921d",
    "data": "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c",
    "family_manifest": "2c0d0c516d38dc826c23edda76e1a0ef668ff4692caf72f1bd710150629c7dc8",
    "d063_run_contract": "09d331bc376788f4a038fcb7d82389951d61e5a1eb67bb8e85b3fd39360fd9fc",
    "d063_summary": "e44238902348f6517412844220d71332e4ed05803138a255d50cb2f35f781e0b",
}
EXPECTED_A1_SHA256 = {
    "docs/time_dependent_no/W26_L5_CROSS_RESOLUTION_PREREGISTRATION.md": (
        "157f02824b05eb346576fc90845edbd7fcb8b4fe2247b48e53d9168bc13fc2b7"
    ),
    "utility/time_dependent_no/pcno_cross_resolution_correction.py": (
        "1b5038b7b52f6eeb810263e194779a85039f97bef29220fcce49f60d7e04509d"
    ),
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_correction.py": (
        "3cf3b13b3312ece83f2163b67efd30e433ca2fa2e68dd416bbd0d7a762fcab70"
    ),
    "tests/time_dependent_no/test_pcno_cross_resolution_correction.py": (
        "9d2aa04cf8c955e4bfc02507e31a416239198dce126bdcb36595b6e93fbae9af"
    ),
}
A2_SOURCE_PATHS = (
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py",
    "tests/time_dependent_no/test_pcno_cross_resolution_teacher_forced.py",
)
SOURCE_PATHS = (*EXPECTED_A1_SHA256, *A2_SOURCE_PATHS)


@dataclass
class RuntimeContext:
    args: argparse.Namespace
    checkpoint: Mapping[str, Any]
    manifest: Mapping[str, Any]
    store: PCNOEuler2DShardStore
    model: torch.nn.Module
    normalization: Any
    device: torch.device
    geometry_by_resolution: Mapping[Resolution, Any]
    sample_by_resolution: Mapping[Resolution, Mapping[str, torch.Tensor]]
    native_projector: Any
    first_config: Any
    source_manifest: Mapping[str, Any]


@dataclass
class CollectionResult:
    snapshots: list[DiagnosticSnapshot]
    floor_rows: list[dict[str, Any]]
    front_rows: list[dict[str, Any]]
    integral_rows: list[dict[str, Any]]
    proposal_rows: list[dict[str, Any]]
    reference_rows: list[dict[str, Any]]
    execution: dict[str, Any]
    maxima: dict[str, float | int]


def _git_status_short(paths: Sequence[str] | None = None) -> list[str]:
    command = ["git", "status", "--short"]
    if paths is not None:
        command.extend(("--", *paths))
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path.name}")
    return value


def _maximum_absolute(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _physical_integral(value: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if field.ndim != 2 or field.shape[0] != mass.size:
        raise ValueError("field and physical volumes do not align")
    return np.einsum("n,nc->c", mass, field, optimize=True)


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def create_readiness(args: argparse.Namespace) -> dict[str, Any]:
    synthetic = a1_synthetic_smoke_summary(args.synthetic_seed)
    verify_payload_sha256(synthetic)
    source_hashes = _source_hashes()
    checks = {
        "a1_source_identity_exact": {
            path: source_hashes.get(path) for path in EXPECTED_A1_SHA256
        }
        == EXPECTED_A1_SHA256,
        "a2_source_inventory_exact": set(source_hashes) == set(SOURCE_PATHS),
        "focused_cpu_tests_passed": args.focused_test_result.startswith("passed"),
        "a1_synthetic_passed": synthetic.get("status") == "passed",
        "a1_synthetic_checkpoint_closed": synthetic.get("checkpoint_loaded") is False,
        "a1_synthetic_dataset_closed": synthetic.get("dataset_loaded") is False,
    }
    payload = with_payload_sha256(
        {
            "schema": READINESS_SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "git": git_state(ROOT),
            "global_status_short": _git_status_short(),
            "relevant_status_short": _git_status_short(SOURCE_PATHS),
            "source_sha256": source_hashes,
            "focused_cpu_test_command": args.focused_test_command,
            "focused_cpu_test_result": args.focused_test_result,
            "synthetic_payload_sha256": synthetic["payload_sha256"],
            "synthetic_seed": int(args.synthetic_seed),
            "expected_upstream_sha256": EXPECTED_UPSTREAM_SHA256,
            "authorization": {
                "a2_open_teacher_forced": True,
                "training": False,
                "sealed_population": False,
                "recurrent_p2": False,
                "d073_b": False,
                "bump": False,
            },
        }
    )
    atomic_write_json(args.output, payload)
    return payload


def _verify_readiness(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != READINESS_SCHEMA or payload.get("status") != "passed":
        raise ValueError("A2 readiness artifact is absent or did not pass")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not all(value is True for value in checks.values()):
        raise ValueError("A2 readiness checks are incomplete")
    current_hashes = _source_hashes()
    if payload.get("source_sha256") != current_hashes:
        raise ValueError("A2 source changed after readiness was recorded")
    if {path: current_hashes[path] for path in EXPECTED_A1_SHA256} != EXPECTED_A1_SHA256:
        raise ValueError("frozen A1 source identity mismatch")
    return payload


def _external_hashes(args: argparse.Namespace, store: PCNOEuler2DShardStore) -> dict[str, str]:
    return {
        "checkpoint": sha256_file(args.checkpoint),
        "normalization_file": sha256_file(args.normalization_file),
        "split": sha256_file(args.split_file),
        "data": store.manifest_digest,
        "family_manifest": sha256_file(args.family_root / "family_manifest.json"),
        "d063_run_contract": sha256_file(args.d063_run_contract),
        "d063_summary": sha256_file(args.d063_summary),
    }


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> None:
    if checkpoint.get("normalization_digest") != EXPECTED_UPSTREAM_SHA256["normalization_digest"]:
        raise ValueError("checkpoint normalization mapping digest mismatch")
    if checkpoint.get("data_manifest_digest") != EXPECTED_UPSTREAM_SHA256["data"]:
        raise ValueError("checkpoint data-manifest digest mismatch")
    if store.manifest_digest != checkpoint.get("data_manifest_digest"):
        raise ValueError("checkpoint and shard manifest differ")
    data_contract = checkpoint.get("data_contract", {})
    if data_contract.get("source_family_id") != manifest.get("family_id"):
        raise ValueError("checkpoint and family identifier differ")
    if data_contract.get("source_family_manifest_digest") != manifest.get(
        "manifest_digest_sha256"
    ):
        raise ValueError("checkpoint and family mapping digest differ")
    config = checkpoint.get("model_config", {})
    if int(config.get("k_max", -1)) != 8:
        raise ValueError("checkpoint k_max differs from the frozen contract")
    if tuple(float(value) for value in config.get("domain_lengths", ())) != (2.0, 1.0):
        raise ValueError("checkpoint Fourier periods differ from the physical domain")
    if checkpoint_step_stride(checkpoint) != 2:
        raise ValueError("checkpoint stride differs from the registered two-frame map")
    if checkpoint.get("boundary_mode") != "model_all_nodes":
        raise ValueError("checkpoint boundary policy mismatch")
    if checkpoint.get("raw_recurrence") is not True:
        raise ValueError("checkpoint raw-recurrence declaration mismatch")
    if any(bool(value) for value in checkpoint.get("inference_interventions", {}).values()):
        raise ValueError("checkpoint-time intervention is forbidden")


def _build_source_manifest(
    args: argparse.Namespace,
    readiness: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> dict[str, Any]:
    external_hashes = _external_hashes(args, store)
    if external_hashes != {
        key: EXPECTED_UPSTREAM_SHA256[key]
        for key in external_hashes
    }:
        raise ValueError(
            f"external source mismatch: expected={EXPECTED_UPSTREAM_SHA256}, "
            f"actual={external_hashes}"
        )
    _validate_checkpoint_contract(checkpoint, manifest, store)
    d063_contract = _read_json(args.d063_run_contract)
    if d063_contract.get("schema") != "pcno_resolution_allval_teacher_contract_v1":
        raise ValueError("D063 run-contract schema mismatch")
    if d063_contract.get("checkpoint", {}).get("sha256") != external_hashes["checkpoint"]:
        raise ValueError("D063 run contract and checkpoint differ")
    return with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "readiness_file_sha256": sha256_file(args.readiness),
            "readiness_payload_sha256": readiness["payload_sha256"],
            "source_sha256": _source_hashes(),
            "external_sha256": external_hashes,
            "checkpoint_contract": {
                "normalization_digest": checkpoint["normalization_digest"],
                "data_manifest_digest": checkpoint["data_manifest_digest"],
                "config_digest": checkpoint["config_digest"],
                "step_stride": checkpoint_step_stride(checkpoint),
                "boundary_mode": checkpoint["boundary_mode"],
                "raw_recurrence": checkpoint["raw_recurrence"],
                "model_config": checkpoint["model_config"],
            },
            "family_contract": {
                "family_id": manifest["family_id"],
                "mapping_digest": manifest["manifest_digest_sha256"],
                "source_resolution": manifest["reference_fidelity"]["evolution_grid"],
                "native_resolution": manifest["reference_fidelity"]["stored_model_grid"],
            },
            "population": {
                "calibration_cases": list(CALIBRATION_CASE_IDS),
                "evaluation_cases": list(EVALUATION_CASE_IDS),
                "fit_input_calls": list(FIT_INPUT_CALLS),
                "held_out_input_calls": list(HELD_OUT_INPUT_CALLS),
                "strength_ood_and_test": "sealed",
            },
            "resolution_contract": {
                "coarse": list(RESOLUTION_CONTRACT.coarse),
                "native": list(RESOLUTION_CONTRACT.native),
                "fine": list(RESOLUTION_CONTRACT.fine),
                "coarse_input": "float64 restriction of native physical state",
                "fine_input": "float64 piecewise-constant prolongation of native physical state",
                "model_boundary": "one explicit FP32 rounding per actual grid",
                "amp": "none",
                "reference_label_only": True,
            },
        }
    )


def _open_contract(args: argparse.Namespace):
    readiness = _verify_readiness(args.readiness)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    manifest = load_shock_vortex_family_manifest(args.family_root / "family_manifest.json")
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=1)
    try:
        source_manifest = _build_source_manifest(
            args,
            readiness,
            checkpoint,
            manifest,
            store,
        )
    except Exception:
        store.close()
        raise
    return readiness, checkpoint, manifest, store, source_manifest


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint, manifest, store, source_manifest = _open_contract(args)
    try:
        payload = with_payload_sha256(
            {
                "schema": "pcno_cross_resolution_teacher_forced_preflight_v1",
                "working_id": WORKING_ID,
                "status": "passed",
                "checkpoint_or_reference_arrays_loaded": False,
                "evaluation_targets_loaded": False,
                "source_manifest": source_manifest,
                "case_inventory_present": {
                    "calibration": all(case_id in store.keys for case_id in CALIBRATION_CASE_IDS),
                    "evaluation": all(case_id in store.keys for case_id in EVALUATION_CASE_IDS),
                },
                "family_splits": {
                    case_id: family_case_provenance(manifest, case_id)["split"]
                    for case_id in (*CALIBRATION_CASE_IDS, *EVALUATION_CASE_IDS)
                },
                "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
            }
        )
    finally:
        store.close()
    atomic_write_json(args.output, payload)
    return payload


def _build_runtime(args: argparse.Namespace) -> RuntimeContext:
    _, checkpoint, manifest, store, source_manifest = _open_contract(args)
    try:
        device = select_device(args.device)
        if device.type != "cuda" and not args.allow_cpu:
            raise ValueError("scientific A2 execution requires CUDA unless --allow-cpu is explicit")
        model, normalization = build_resolution_checkpoint_model(checkpoint, device)
        model.eval()
        first_config = config_for_family_case(manifest, CALIBRATION_CASE_IDS[0])
        geometry_by_resolution = {}
        sample_by_resolution = {}
        node_type_by_resolution = {}
        for resolution in (
            RESOLUTION_CONTRACT.coarse,
            RESOLUTION_CONTRACT.native,
            RESOLUTION_CONTRACT.fine,
        ):
            config, geometry = build_resolution_geometry(first_config, resolution)
            node_type = node_types_for_protocol(
                geometry,
                config,
                "physical",
                training_resolution=RESOLUTION_CONTRACT.native,
            )
            geometry_by_resolution[resolution] = geometry
            node_type_by_resolution[resolution] = node_type
            sample_by_resolution[resolution] = make_model_sample(
                geometry,
                node_type,
                mach=first_config.shock_mach,
                device=device,
            )
        native_geometry = geometry_by_resolution[RESOLUTION_CONTRACT.native]
        projector = build_fixed_cosine_projector(
            native_geometry.nodes,
            native_geometry.node_measures,
            node_type_by_resolution[RESOLUTION_CONTRACT.native],
            rank=8,
            domain_bounds=(
                first_config.x_min,
                first_config.x_max,
                first_config.y_min,
                first_config.y_max,
            ),
        )
        return RuntimeContext(
            args=args,
            checkpoint=checkpoint,
            manifest=manifest,
            store=store,
            model=model,
            normalization=normalization,
            device=device,
            geometry_by_resolution=geometry_by_resolution,
            sample_by_resolution=sample_by_resolution,
            native_projector=projector,
            first_config=first_config,
            source_manifest=source_manifest,
        )
    except Exception:
        store.close()
        raise


def _close_runtime(runtime: RuntimeContext) -> None:
    runtime.store.close()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()


def _floor_row(
    *,
    case_id: str,
    input_call: int,
    object_kind: str,
    direction: str,
    fields: Any,
    query_volumes: np.ndarray,
    native_volumes: np.ndarray,
    component_scale: np.ndarray,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_call": input_call,
        "object_kind": object_kind,
        "direction": direction,
        "query_round_trip_max_abs": _maximum_absolute(fields.query_round_trip),
        "query_round_trip_scaled_rms": weighted_scaled_rms(
            fields.query_round_trip,
            volumes=query_volumes,
            component_scale=component_scale,
        ),
        "native_information_mismatch_max_abs": _maximum_absolute(
            fields.native_information_mismatch
        ),
        "native_information_mismatch_scaled_rms": weighted_scaled_rms(
            fields.native_information_mismatch,
            volumes=native_volumes,
            component_scale=component_scale,
        ),
        "pipeline_truth_floor_max_abs": _maximum_absolute(fields.pipeline_truth_floor),
        "pipeline_truth_floor_scaled_rms": weighted_scaled_rms(
            fields.pipeline_truth_floor,
            volumes=query_volumes,
            component_scale=component_scale,
        ),
        "closure_max_abs": _maximum_absolute(fields.closure_residual),
    }


def _front_errors(prediction: np.ndarray, reference: np.ndarray, runtime: RuntimeContext):
    metrics = pressure_profile_shock_metrics(
        prediction,
        reference,
        resolution=RESOLUTION_CONTRACT.native,
        x_min=runtime.first_config.x_min,
        x_max=runtime.first_config.x_max,
        gamma=runtime.normalization.gamma,
        shock_center_x=runtime.first_config.shock_x,
    )
    errors: dict[str, float | None] = {
        "front_position": metrics["shock_position_absolute_error"],
        "front_strength_log_ratio": None,
        "front_thickness_log_ratio": None,
    }
    for source, target in (
        ("pressure_profile_shock_strength_ratio", "front_strength_log_ratio"),
        ("pressure_profile_shock_thickness_ratio", "front_thickness_log_ratio"),
    ):
        ratio = metrics[source]
        if ratio is not None and np.isfinite(float(ratio)) and float(ratio) > 0.0:
            errors[target] = abs(float(np.log(float(ratio))))
    return errors


def _collect(
    runtime: RuntimeContext,
    *,
    case_ids: Sequence[str],
    input_calls: Sequence[int],
    coefficients: tuple[float, float] | None,
    phase: str,
) -> CollectionResult:
    started = perf_counter()
    expected_cases = tuple(case_ids)
    expected_calls = tuple(input_calls)
    if len(set(expected_cases)) != len(expected_cases) or len(set(expected_calls)) != len(
        expected_calls
    ):
        raise ValueError("case/call collection inventory must be unique")
    snapshots: list[DiagnosticSnapshot] = []
    floor_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    integral_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    native_geometry = runtime.geometry_by_resolution[RESOLUTION_CONTRACT.native]
    coarse_geometry = runtime.geometry_by_resolution[RESOLUTION_CONTRACT.coarse]
    fine_geometry = runtime.geometry_by_resolution[RESOLUTION_CONTRACT.fine]
    execution = {
        "phase": phase,
        "logical_model_calls": 0,
        "actual_forward_passes_including_repeats": 0,
        "total_forward_seconds": 0.0,
        "maximum_repeat_abs_difference": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
    }
    maxima: dict[str, float | int] = {
        "pre_model_nesting": 0.0,
        "post_fp32_nesting": 0.0,
        "sign_closure": 0.0,
        "increment_integral_closure": 0.0,
        "transfer_floor_closure": 0.0,
        "region_partition_error": 0,
    }

    def account(timing: Mapping[str, Any]) -> None:
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

    for case_index, case_id in enumerate(expected_cases, start=1):
        provenance = family_case_provenance(runtime.manifest, case_id)
        if provenance["split"] != "validation":
            raise ValueError(f"A2 case is outside the open validation split: {case_id}")
        if case_id not in runtime.store.keys:
            raise ValueError(f"A2 case is absent from checkpoint-bound shards: {case_id}")
        reference, reference_check = load_resolution_reference(
            runtime.args.family_root,
            runtime.args.multires_reference_root,
            runtime.store,
            runtime.manifest,
            case_id,
            training_resolution=RESOLUTION_CONTRACT.native,
        )
        reference_rows.append(reference_check)
        reference_resolution = tuple(int(value) for value in reference["retained_resolution"])
        if reference_resolution != RESOLUTION_CONTRACT.fine:
            raise ValueError("A2 requires the retained source-rich 500x200 reference")
        print(
            f"{phase}: case {case_index}/{len(expected_cases)} {case_id} "
            f"calls={expected_calls[0]}..{expected_calls[-1]}",
            flush=True,
        )
        for input_call in expected_calls:
            input_frame = int(input_call) * 2
            output_frame = (int(input_call) + 1) * 2
            native_input = reference_at_resolution(
                reference["conservative_states"][input_frame],
                reference_resolution=reference_resolution,
                target_resolution=RESOLUTION_CONTRACT.native,
            )
            native_target = reference_at_resolution(
                reference["conservative_states"][output_frame],
                reference_resolution=reference_resolution,
                target_resolution=RESOLUTION_CONTRACT.native,
            )
            if native_input is None or native_target is None:
                raise ValueError("native common-source reference is unavailable")
            prepared = prepare_common_native_inputs(
                native_input,
                contract=RESOLUTION_CONTRACT,
            )
            predictions: dict[Resolution, np.ndarray] = {}
            for resolution in (
                RESOLUTION_CONTRACT.coarse,
                RESOLUTION_CONTRACT.native,
                RESOLUTION_CONTRACT.fine,
            ):
                repeat = runtime.args.repeat_forward if input_call == expected_calls[0] else 1
                prediction, timing = predict_resolution_sample(
                    runtime.model,
                    runtime.sample_by_resolution[resolution],
                    prepared.model_inputs[resolution],
                    device=runtime.device,
                    amp="none",
                    repeats=repeat,
                )
                account(timing)
                predictions[resolution] = prediction
            basis = common_native_increment_basis(
                prepared,
                predictions,
                contract=RESOLUTION_CONTRACT,
            )
            native_reference_increment = (
                np.asarray(native_target, dtype=np.float64)
                - prepared.model_inputs[RESOLUTION_CONTRACT.native]
            )
            native_true_increment = np.asarray(native_target, dtype=np.float64) - np.asarray(
                native_input, dtype=np.float64
            )
            target_correction = native_reference_increment - basis.native_increment
            errors = mapped_increment_errors(
                basis,
                native_reference_increment=native_reference_increment,
                contract=RESOLUTION_CONTRACT,
            )
            maxima["sign_closure"] = max(
                maxima["sign_closure"],
                _maximum_absolute(
                    basis.native_minus_coarse
                    - (errors["native"] - errors["coarse_on_native"])
                ),
                _maximum_absolute(
                    basis.fine_minus_native
                    - (errors["fine_on_native"] - errors["native"])
                ),
                _maximum_absolute(target_correction + errors["native"]),
            )
            maxima["pre_model_nesting"] = max(
                maxima["pre_model_nesting"],
                prepared.nesting_floors["pre_model_coarse_from_native_max_abs"],
                prepared.nesting_floors["pre_model_fine_to_native_max_abs"],
            )
            maxima["post_fp32_nesting"] = max(
                maxima["post_fp32_nesting"],
                prepared.nesting_floors["post_fp32_coarse_from_native_max_abs"],
                prepared.nesting_floors["post_fp32_fine_to_native_max_abs"],
            )

            coarse_increment = (
                predictions[RESOLUTION_CONTRACT.coarse]
                - prepared.model_inputs[RESOLUTION_CONTRACT.coarse]
            )
            fine_increment = (
                predictions[RESOLUTION_CONTRACT.fine]
                - prepared.model_inputs[RESOLUTION_CONTRACT.fine]
            )
            maxima["increment_integral_closure"] = max(
                maxima["increment_integral_closure"],
                _maximum_absolute(
                    _physical_integral(coarse_increment, coarse_geometry.node_measures)
                    - _physical_integral(basis.coarse_on_native, native_geometry.node_measures)
                ),
                _maximum_absolute(
                    _physical_integral(fine_increment, fine_geometry.node_measures)
                    - _physical_integral(basis.fine_on_native, native_geometry.node_measures)
                ),
            )

            coarse_state = prepared.pre_model_inputs[RESOLUTION_CONTRACT.coarse]
            matched_fine_state = prepared.pre_model_inputs[RESOLUTION_CONTRACT.fine]
            source_fine_state = np.asarray(
                reference["conservative_states"][input_frame], dtype=np.float64
            )
            coarse_true_increment = (
                reference_at_resolution(
                    reference["conservative_states"][output_frame],
                    reference_resolution=reference_resolution,
                    target_resolution=RESOLUTION_CONTRACT.coarse,
                )
                - reference_at_resolution(
                    reference["conservative_states"][input_frame],
                    reference_resolution=reference_resolution,
                    target_resolution=RESOLUTION_CONTRACT.coarse,
                )
            )
            source_fine_increment = (
                np.asarray(reference["conservative_states"][output_frame], dtype=np.float64)
                - source_fine_state
            )
            matched_fine_increment = prolong_nested_state(
                native_true_increment,
                coarse_resolution=RESOLUTION_CONTRACT.native,
                fine_resolution=RESOLUTION_CONTRACT.fine,
            )
            floor_specs = (
                (
                    "state",
                    "coarse_query_to_native",
                    coarse_state,
                    native_input,
                    RESOLUTION_CONTRACT.coarse,
                    coarse_geometry.node_measures,
                ),
                (
                    "state",
                    "matched_fine_query_to_native",
                    matched_fine_state,
                    native_input,
                    RESOLUTION_CONTRACT.fine,
                    fine_geometry.node_measures,
                ),
                (
                    "state",
                    "source_rich_fine_query_to_native",
                    source_fine_state,
                    native_input,
                    RESOLUTION_CONTRACT.fine,
                    fine_geometry.node_measures,
                ),
                (
                    "true_increment",
                    "coarse_query_to_native",
                    coarse_true_increment,
                    native_true_increment,
                    RESOLUTION_CONTRACT.coarse,
                    coarse_geometry.node_measures,
                ),
                (
                    "true_increment",
                    "matched_fine_query_to_native",
                    matched_fine_increment,
                    native_true_increment,
                    RESOLUTION_CONTRACT.fine,
                    fine_geometry.node_measures,
                ),
                (
                    "true_increment",
                    "source_rich_fine_query_to_native",
                    source_fine_increment,
                    native_true_increment,
                    RESOLUTION_CONTRACT.fine,
                    fine_geometry.node_measures,
                ),
            )
            for object_kind, direction, query_value, native_value, query_res, query_vol in floor_specs:
                floors = transfer_floor_fields(
                    query_value,
                    native_value,
                    query_resolution=query_res,
                    native_resolution=RESOLUTION_CONTRACT.native,
                )
                row = _floor_row(
                    case_id=case_id,
                    input_call=int(input_call),
                    object_kind=object_kind,
                    direction=direction,
                    fields=floors,
                    query_volumes=query_vol,
                    native_volumes=native_geometry.node_measures,
                    component_scale=runtime.normalization.residual_scale
                    if object_kind == "true_increment"
                    else runtime.normalization.state_scale,
                )
                floor_rows.append(row)
                maxima["transfer_floor_closure"] = max(
                    maxima["transfer_floor_closure"], row["closure_max_abs"]
                )

            masks, _, _ = shock_vortex_regions(
                native_target,
                native_geometry.nodes,
                resolution=RESOLUTION_CONTRACT.native,
                gamma=runtime.normalization.gamma,
            )
            partition_count = sum(
                masks[name].astype(np.int8)
                for name in (
                    "partition_boundary",
                    "partition_shock",
                    "partition_vortex",
                    "partition_smooth",
                )
            )
            maxima["region_partition_error"] = max(
                maxima["region_partition_error"],
                int(np.max(np.abs(partition_count - 1))),
            )
            snapshots.append(
                DiagnosticSnapshot(
                    case_id=case_id,
                    group_id=case_id.split("_")[1],
                    input_call=int(input_call),
                    basis=basis,
                    target_correction=np.asarray(target_correction, dtype=np.float64),
                    volumes=np.asarray(native_geometry.node_measures, dtype=np.float64),
                    component_scale=np.asarray(
                        runtime.normalization.residual_scale, dtype=np.float64
                    ),
                    masks=masks,
                )
            )

            if coefficients is not None:
                raw_prediction = predictions[RESOLUTION_CONTRACT.native]
                corrected = corrected_native_prediction(
                    raw_prediction,
                    basis,
                    alpha=coefficients[0],
                    beta=coefficients[1],
                )
                raw_front = _front_errors(raw_prediction, native_target, runtime)
                corrected_front = _front_errors(corrected, native_target, runtime)
                for key in FRONT_CONTROL_KEYS:
                    front_rows.append(
                        {
                            "case_id": case_id,
                            "input_call": int(input_call),
                            "key": key,
                            "zero_error": raw_front[key],
                            "corrected_error": corrected_front[key],
                        }
                    )
                raw_integral = _physical_integral(
                    raw_prediction - native_target, native_geometry.node_measures
                )
                corrected_integral = _physical_integral(
                    corrected - native_target, native_geometry.node_measures
                )
                for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
                    integral_rows.append(
                        {
                            "case_id": case_id,
                            "input_call": int(input_call),
                            "component": name,
                            "zero_error": float(raw_integral[component]),
                            "corrected_error": float(corrected_integral[component]),
                        }
                    )
                corrected_admissibility = conservative_admissibility_summary(
                    corrected,
                    gamma=runtime.normalization.gamma,
                )
                proposal_rows.append(
                    {
                        "case_id": case_id,
                        "input_call": int(input_call),
                        "finite": bool(np.isfinite(corrected).all()),
                        "admissible": bool(corrected_admissibility["admissible"]),
                        "minimum_density": corrected_admissibility["minimum_density"],
                        "minimum_pressure": corrected_admissibility["minimum_pressure"],
                        "minimum_internal_energy": corrected_admissibility[
                            "minimum_internal_energy"
                        ],
                    }
                )
        del reference
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
    execution["wall_seconds"] = perf_counter() - started
    execution["snapshot_count"] = len(snapshots)
    execution["parameter_count"] = parameter_count(runtime.model)
    execution["device"] = str(runtime.device)
    execution["amp"] = "none"
    return CollectionResult(
        snapshots=snapshots,
        floor_rows=floor_rows,
        front_rows=front_rows,
        integral_rows=integral_rows,
        proposal_rows=proposal_rows,
        reference_rows=reference_rows,
        execution=execution,
        maxima=maxima,
    )


def _select_cell(
    snapshots: Sequence[DiagnosticSnapshot], cell: str
) -> list[DiagnosticSnapshot]:
    cases = set(CELL_CASES[cell])
    calls = set(CELL_CALLS[cell])
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.case_id in cases and snapshot.input_call in calls
    ]


def _write_source_manifest(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = output_dir / "source_manifest.json"
    atomic_write_json(path, payload)
    return path


def run_calibration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    runtime = _build_runtime(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    source_manifest_path = _write_source_manifest(output_dir, runtime.source_manifest)
    try:
        collected = _collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=FIT_INPUT_CALLS,
            coefficients=None,
            phase="calibration",
        )
        crossfit = grouped_crossfit(
            collected.snapshots,
            resolution=RESOLUTION_CONTRACT.native,
            expected_groups=CALIBRATION_GROUPS,
            expected_input_calls=FIT_INPUT_CALLS,
        )
        fit_cell = score_cell(
            collected.snapshots,
            cell="calibration_fit",
            coefficients=tuple(crossfit["selected_coefficients"] or (0.0, 0.0)),
            resolution=RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        relations = error_relation_rows(
            collected.snapshots,
            cell="calibration_fit",
            resolution=RESOLUTION_CONTRACT.native,
            projector=runtime.native_projector,
        )
        closure = CalibrationClosureEvidence(
            checkpoint_contract=True,
            reference_contract=len(collected.reference_rows) == len(CALIBRATION_CASE_IDS),
            common_source_inventory=len(collected.snapshots)
            == len(CALIBRATION_CASE_IDS) * len(FIT_INPUT_CALLS),
            prediction_inventory=collected.execution["logical_model_calls"]
            == 3 * len(collected.snapshots),
            exact_source_identity=True,
            maximum_pre_model_nesting_floor=float(collected.maxima["pre_model_nesting"]),
            maximum_post_fp32_nesting_floor=float(collected.maxima["post_fp32_nesting"]),
            maximum_sign_closure=float(collected.maxima["sign_closure"]),
            maximum_increment_integral_closure=float(
                collected.maxima["increment_integral_closure"]
            ),
            maximum_transfer_floor_closure=float(
                collected.maxima["transfer_floor_closure"]
            ),
            maximum_band_closure=max(
                float(crossfit["maximum_band_closure"]),
                float(fit_cell["maximum_closure"]["band"]),
            ),
            maximum_region_partition_error=int(
                collected.maxima["region_partition_error"]
            ),
            maximum_repeat_abs_difference=float(
                collected.execution["maximum_repeat_abs_difference"]
            ),
        )
        qualification = qualify_calibration(crossfit, closure)
        write_csv(output_dir / "transfer_floors.csv", collected.floor_rows)
        write_csv(output_dir / "fit_cell_scores.csv", fit_cell["rows"])
        write_csv(output_dir / "error_relations.csv", relations)
        write_csv(output_dir / "reference_checks.csv", collected.reference_rows)
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        payload = with_payload_sha256(
            {
                "schema": CALIBRATION_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "complete_qualified"
                    if qualification["status"] == "qualified"
                    else "complete_not_qualified"
                ),
                "evaluation_targets_loaded": False,
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "source_manifest_payload_sha256": runtime.source_manifest[
                    "payload_sha256"
                ],
                "crossfit": crossfit,
                "qualification": qualification,
                "calibration_fit_cell": fit_cell,
                "execution": collected.execution,
                "transfer_floor_maxima": collected.maxima,
                "artifact_inventory_before_summary": artifacts,
                "claim_boundary": (
                    "Calibration-only open validation evidence. True error is an "
                    "offline label; no recurrent, sealed, bump, or Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "calibration.json", payload)
        return payload, 0 if qualification["status"] == "qualified" else 3
    finally:
        _close_runtime(runtime)


def _rms_control(
    *,
    key: str,
    scope: str,
    rows: Sequence[Mapping[str, Any]],
) -> ControlRatio:
    if not rows or any(
        row.get("zero_error") is None
        or row.get("corrected_error") is None
        or not np.isfinite(float(row["zero_error"]))
        or not np.isfinite(float(row["corrected_error"]))
        for row in rows
    ):
        return ControlRatio(
            key=key,
            scope=scope,
            zero_rms=0.0,
            corrected_rms=0.0,
            ratio=None,
            status="incomplete",
        )
    zero_rms = float(np.sqrt(np.mean([float(row["zero_error"]) ** 2 for row in rows])))
    corrected_rms = float(
        np.sqrt(np.mean([float(row["corrected_error"]) ** 2 for row in rows]))
    )
    ratio = corrected_rms / zero_rms if zero_rms > 1.0e-8 else None
    return ControlRatio(
        key=key,
        scope=scope,
        zero_rms=zero_rms,
        corrected_rms=corrected_rms,
        ratio=ratio,
        status="ok" if ratio is not None and np.isfinite(ratio) else "zero_denominator",
    )


def _build_controls(
    joint_cell: Mapping[str, Any],
    front_rows: Sequence[Mapping[str, Any]],
    integral_rows: Sequence[Mapping[str, Any]],
) -> list[ControlRatio]:
    controls: list[ControlRatio] = []
    score_rows = joint_cell["rows"]
    for view in REQUIRED_FIELD_CONTROL_VIEWS:
        for scope in ("population", *EVALUATION_CASE_IDS):
            matches = [
                row
                for row in score_rows
                if row.get("view") == view
                and (
                    (scope == "population" and row.get("scope") == "population")
                    or (row.get("scope") == "case" and row.get("case_id") == scope)
                )
            ]
            if len(matches) != 1:
                controls.append(
                    ControlRatio(
                        key=f"field::{view}",
                        scope=scope,
                        zero_rms=0.0,
                        corrected_rms=0.0,
                        ratio=None,
                        status="incomplete",
                    )
                )
                continue
            row = matches[0]
            target_rms = float(row["target_rms"])
            ratio = row.get("rms_ratio_vs_zero")
            controls.append(
                ControlRatio(
                    key=f"field::{view}",
                    scope=scope,
                    zero_rms=target_rms,
                    corrected_rms=(
                        target_rms * float(ratio) if ratio is not None else 0.0
                    ),
                    ratio=None if ratio is None else float(ratio),
                    status="ok" if row.get("skill_status") == "ok" else "unresolved",
                )
            )

    primary_front = [
        row
        for row in front_rows
        if row["case_id"] in EVALUATION_CASE_IDS
        and int(row["input_call"]) in HELD_OUT_INPUT_CALLS
    ]
    for key in FRONT_CONTROL_KEYS:
        for scope in ("population", *EVALUATION_CASE_IDS):
            rows = [
                row
                for row in primary_front
                if row["key"] == key
                and (scope == "population" or row["case_id"] == scope)
            ]
            controls.append(_rms_control(key=key, scope=scope, rows=rows))

    primary_integrals = [
        row
        for row in integral_rows
        if row["case_id"] in EVALUATION_CASE_IDS
        and int(row["input_call"]) in HELD_OUT_INPUT_CALLS
    ]
    for component in INTEGRAL_COMPONENT_NAMES:
        for scope in ("population", *EVALUATION_CASE_IDS):
            rms_rows = [
                row
                for row in primary_integrals
                if row["component"] == component
                and (scope == "population" or row["case_id"] == scope)
            ]
            controls.append(
                _rms_control(
                    key=f"integral_rms::{component}",
                    scope=scope,
                    rows=rms_rows,
                )
            )
            endpoint_rows = [
                row
                for row in rms_rows
                if int(row["input_call"]) == HELD_OUT_INPUT_CALLS[-1]
            ]
            controls.append(
                _rms_control(
                    key=f"integral_endpoint::{component}",
                    scope=scope,
                    rows=endpoint_rows,
                )
            )
    return controls


def _verify_frozen_source_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported source-manifest schema")
    if payload.get("source_sha256") != _source_hashes():
        raise ValueError("evaluator source differs from the frozen calibration source")
    readiness = _verify_readiness(args.readiness)
    if payload.get("readiness_payload_sha256") != readiness["payload_sha256"]:
        raise ValueError("readiness identity differs from the calibration source")
    return payload


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    # The firewall is intentionally first: no manifest, shard, or reference loader
    # exists before these two immutable artifacts qualify.
    frozen_source = _verify_frozen_source_manifest(args.source_manifest, args)
    calibration = require_qualified_calibration(
        args.calibration,
        expected_source_manifest_sha256=sha256_file(args.source_manifest),
    )
    coefficients = tuple(
        float(value) for value in calibration["qualification"]["selected_coefficients"]
    )
    runtime = _build_runtime(args)
    if runtime.source_manifest != frozen_source:
        _close_runtime(runtime)
        raise ValueError("live external/source contract differs from calibration")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        time_only = _collect(
            runtime,
            case_ids=CALIBRATION_CASE_IDS,
            input_calls=HELD_OUT_INPUT_CALLS,
            coefficients=coefficients,
            phase="time_only",
        )
        evaluation = _collect(
            runtime,
            case_ids=EVALUATION_CASE_IDS,
            input_calls=ALL_INPUT_CALLS,
            coefficients=coefficients,
            phase="evaluation",
        )
        all_snapshots = [*time_only.snapshots, *evaluation.snapshots]
        cell_payloads = {}
        relation_rows: list[dict[str, Any]] = []
        for cell in (
            "time_only",
            "case_only",
            "joint_held_out",
            "joint_late_1",
            "joint_late_2",
        ):
            selected = _select_cell(all_snapshots, cell)
            cell_payloads[cell] = score_cell(
                selected,
                cell=cell,
                coefficients=coefficients,
                resolution=RESOLUTION_CONTRACT.native,
                projector=runtime.native_projector,
            )
            relation_rows.extend(
                error_relation_rows(
                    selected,
                    cell=cell,
                    resolution=RESOLUTION_CONTRACT.native,
                    projector=runtime.native_projector,
                )
            )
        front_rows = [*time_only.front_rows, *evaluation.front_rows]
        integral_rows = [*time_only.integral_rows, *evaluation.integral_rows]
        controls = _build_controls(
            cell_payloads["joint_held_out"],
            front_rows,
            integral_rows,
        )
        proposal_evidence = [
            ProposalEvidence(
                case_id=str(row["case_id"]),
                input_call=int(row["input_call"]),
                finite=bool(row["finite"]),
                admissible=bool(row["admissible"]),
            )
            for row in evaluation.proposal_rows
            if int(row["input_call"]) in HELD_OUT_INPUT_CALLS
        ]
        gate = prospective_gate(
            cell_payloads=cell_payloads,
            controls=controls,
            proposals=proposal_evidence,
        )
        score_rows = [
            row for payload in cell_payloads.values() for row in payload["rows"]
        ]
        oracle_rows = [
            row for payload in cell_payloads.values() for row in payload["oracle_rows"]
        ]
        floor_rows = [*time_only.floor_rows, *evaluation.floor_rows]
        proposal_rows = [*time_only.proposal_rows, *evaluation.proposal_rows]
        reference_rows = [*time_only.reference_rows, *evaluation.reference_rows]
        write_csv(output_dir / "cell_scores.csv", score_rows)
        write_csv(output_dir / "error_relations.csv", relation_rows)
        write_csv(output_dir / "front_controls.csv", front_rows)
        write_csv(output_dir / "integral_controls.csv", integral_rows)
        write_csv(output_dir / "proposal_admissibility.csv", proposal_rows)
        write_csv(output_dir / "transfer_floors.csv", floor_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        write_csv(
            output_dir / "prospective_control_ratios.csv",
            [asdict(row) for row in controls],
        )
        artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        }
        payload = with_payload_sha256(
            {
                "schema": EVALUATION_SCHEMA,
                "working_id": WORKING_ID,
                "status": (
                    "complete_gate_passed"
                    if gate["status"] == "passed"
                    else "complete_gate_failed"
                ),
                "calibration_file_sha256": sha256_file(args.calibration),
                "source_manifest_sha256": sha256_file(args.source_manifest),
                "selected_model": calibration["qualification"]["selected_model"],
                "frozen_coefficients": list(coefficients),
                "prospective_gate": gate,
                "cell_payloads": cell_payloads,
                "oracle_rows": oracle_rows,
                "controls": [asdict(row) for row in controls],
                "execution": {
                    "time_only": time_only.execution,
                    "evaluation": evaluation.execution,
                },
                "closure_maxima": {
                    "time_only": time_only.maxima,
                    "evaluation": evaluation.maxima,
                    "cell_scoring": {
                        cell: payload["maximum_closure"]
                        for cell, payload in cell_payloads.items()
                    },
                },
                "artifact_inventory_before_summary": artifacts,
                "p2_executed": False,
                "claim_boundary": (
                    "Open-population teacher-forced fixed-checkpoint evidence only. "
                    "True error was used only for offline labels and diagnostics. "
                    "No recurrent benefit, resolution invariance, conservation-by-"
                    "construction, bump transfer, or Richardson claim."
                ),
            }
        )
        atomic_write_json(output_dir / "evaluation.json", payload)
        return payload, 0 if gate["status"] == "passed" else 4
    finally:
        _close_runtime(runtime)


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--d063-run-contract", type=Path, required=True)
    parser.add_argument("--d063-summary", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    readiness = commands.add_parser("readiness")
    readiness.add_argument("--output", type=Path, required=True)
    readiness.add_argument("--synthetic-seed", type=int, default=0)
    readiness.add_argument("--focused-test-command", required=True)
    readiness.add_argument("--focused-test-result", required=True)

    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser("calibrate")
    _add_external_arguments(calibration)
    calibration.add_argument("--output-dir", type=Path, required=True)
    calibration.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    calibration.add_argument("--allow-cpu", action="store_true")
    calibration.add_argument("--repeat-forward", type=int, default=2)

    evaluation = commands.add_parser("evaluate")
    _add_external_arguments(evaluation)
    evaluation.add_argument("--source-manifest", type=Path, required=True)
    evaluation.add_argument("--calibration", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    evaluation.add_argument("--allow-cpu", action="store_true")
    evaluation.add_argument("--repeat-forward", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "readiness":
        payload = create_readiness(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "preflight":
        payload = run_preflight(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.repeat_forward < 1:
        raise ValueError("repeat-forward must be positive")
    if args.command == "calibrate":
        payload, exit_code = run_calibration(args)
    elif args.command == "evaluate":
        payload, exit_code = run_evaluation(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
