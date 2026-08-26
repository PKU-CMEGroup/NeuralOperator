#!/usr/bin/env python3
"""Compare fixed D044/D060 maps only after the P0-A2 solver pair is frozen."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.p0_restart_sufficiency import (  # noqa: E402
    A0_ARTIFACT_MANIFEST_SHA256,
    CASE_MANIFEST_SHA256,
    CHANNEL_SCALES,
    CHECKPOINT_RECORDS,
    DATASET_MANIFEST_SHA256,
    FIXED_CASES,
    NATIVE_NODES,
    SPLIT_DIGEST,
    canonical_json_sha256,
    load_bound_json,
    load_json_object,
    raw_array_sha256,
    resolve_manifest_member,
    sha256_file,
    validate_artifact_manifest,
    validate_case_manifest,
    validate_execution_contract,
    verify_artifact_member,
)
from utility.time_dependent_no.p0_restart_sufficiency_a2 import (  # noqa: E402
    A1_CLOSEOUT_MANIFEST_SHA256,
    A2_MODEL_ARTIFACT_SCHEMA,
    A2_MODEL_SUMMARY_SCHEMA,
    A2_SOLVER_ARTIFACT_SCHEMA,
    HISTORICAL_EVALUATOR_MAPPING_SHA256,
    admissibility_summary,
    atomic_save_npy,
    atomic_write_json,
    bias_gate_row,
    bind_payload_sha256,
    build_artifact_manifest,
    compare_solver_process_summaries,
    integrated_conservative_change,
    prepare_fresh_directory,
    primitive_relative_l2_metrics,
    require_a2_cublas_workspace_config,
    scaled_rms_distance,
    validate_a1_closeout,
    validate_a2_source_manifest,
    validate_artifact_tree,
    validated_a2_physical_volumes,
    validate_solver_process_summary,
    weighted_relative_l2_metrics,
)

D044_CHECKPOINT_SHA256 = CHECKPOINT_RECORDS["checkpoints/d044/best.pt"]["sha256"]
D060_CHECKPOINT_SHA256 = CHECKPOINT_RECORDS["checkpoints/d060/best.pt"]["sha256"]
D044_COMPATIBILITY_SHA256 = (
    "66a7d30197e861820acad2ff178949e2301b4c66830c779f5b0575070bac8a14"
)
D060_COMPATIBILITY_SHA256 = (
    "625a07b91d8cd93354375e08493aaf2efdd62f00ad9d0308d1d8436032ff1dee"
)
GEOMETRY_ARRAYS = (
    "nodes",
    "node_measures",
    "node_weights",
    "node_rhos",
    "directed_edges",
    "edge_gradient_weights",
    "node_type",
    "edges",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path, required=True)
    parser.add_argument("--solver-primary-dir", type=Path, required=True)
    parser.add_argument("--solver-repeat-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--owner-authorized-a2",
        action="store_true",
        help="Required acknowledgement of the owner's explicit P0-A2 authorization.",
    )
    return parser.parse_args(argv)


def _verified_member(
    inputs_root: Path, artifact_members: Mapping[str, Any], relative: str
) -> Path:
    return verify_artifact_member(
        inputs_root=inputs_root,
        artifact_members=artifact_members,
        relative_path=relative,
    )


def _load_solver_process(
    root: Path, *, process_role: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    validate_artifact_tree(
        root,
        expected_schema=A2_SOLVER_ARTIFACT_SCHEMA,
        require_read_only=True,
    )
    summary = load_json_object(root / "summary.json")
    rows = validate_solver_process_summary(summary, process_role=process_role)
    states: dict[str, np.ndarray] = {}
    exchanges: dict[str, np.ndarray] = {}
    for row in rows:
        case_id = str(row["trajectory_id"])
        state_path = resolve_manifest_member(root, str(row["solver_states_path"]))
        exchange_path = resolve_manifest_member(
            root, str(row["boundary_exchange_path"])
        )
        retained = np.load(state_path, allow_pickle=False)
        exchange = np.load(exchange_path, allow_pickle=False)
        if retained.shape != (2, NATIVE_NODES, 4) or retained.dtype != np.float64:
            raise ValueError("frozen solver states have the wrong shape or dtype")
        if exchange.shape != (4,) or exchange.dtype != np.float64:
            raise ValueError("frozen solver exchange has the wrong shape or dtype")
        if raw_array_sha256(retained) != row["solver_states_raw_sha256"]:
            raise ValueError("frozen solver state raw hash drifted")
        if raw_array_sha256(exchange) != row["boundary_exchange_raw_sha256"]:
            raise ValueError("frozen solver exchange raw hash drifted")
        states[case_id] = retained[-1]
        exchanges[case_id] = exchange
    return summary, states, exchanges


def _validate_historical_base_manifest(
    payload: Mapping[str, Any], source_payload: Mapping[str, Any]
) -> None:
    if payload.get("schema") != "p0_historical_evaluator_source_manifest_v1":
        raise ValueError("unexpected historical evaluator source schema")
    members = payload.get("recovery_time_transitive_members")
    if members != source_payload.get("base_members"):
        raise ValueError("A2 evaluator base members differ from recovered A0")
    if (
        payload.get("canonical_transitive_mapping_sha256")
        != HISTORICAL_EVALUATOR_MAPPING_SHA256
    ):
        raise ValueError("historical evaluator mapping digest drifted")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any], *, label: str, stride: int
) -> None:
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        checkpoint_model_node_type_input,
    )

    if int(checkpoint.get("step_stride", -1)) != stride:
        raise ValueError(f"{label} checkpoint stride drifted")
    if checkpoint.get("data_manifest_digest") != DATASET_MANIFEST_SHA256:
        raise ValueError(f"{label} checkpoint data manifest drifted")
    data_contract = checkpoint.get("data_contract", {})
    if data_contract.get("data_manifest_digest") != DATASET_MANIFEST_SHA256:
        raise ValueError(f"{label} checkpoint data contract drifted")
    if checkpoint_model_node_type_input(checkpoint) != "physical":
        raise ValueError(f"{label} checkpoint node-type contract drifted")
    config = checkpoint.get("model_config", {})
    if (
        int(config.get("k_max", -1)) != 8
        or list(config.get("domain_lengths", [])) != [2.0, 1.0]
        or list(config.get("layers", [])) != [128, 128, 128, 128, 128]
        or int(config.get("fc_dim", -1)) != 128
        or int(config.get("nmeasures", -1)) != 1
    ):
        raise ValueError(f"{label} checkpoint model configuration drifted")
    state_scale = np.asarray(
        checkpoint.get("normalization", {}).get("state_scale"), dtype=np.float64
    )
    if state_scale.shape != (4,) or not np.allclose(
        state_scale,
        np.asarray(CHANNEL_SCALES),
        rtol=0.0,
        atol=5.0e-11,
    ):
        raise ValueError(f"{label} checkpoint state scales drifted")


def _compatibility_contract(payload: Mapping[str, Any], *, stride: int) -> None:
    execution = payload.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("amp") != "none"
        or execution.get("device") != "cuda"
    ):
        raise ValueError("compatibility receipt precision/device drifted")
    checkpoint = payload.get("checkpoint", {})
    if checkpoint.get("boundary_mode") != "model_all_nodes":
        raise ValueError("compatibility receipt boundary mode drifted")
    if checkpoint.get("raw_recurrence") is not True:
        raise ValueError("compatibility receipt recurrence drifted")
    if payload.get("translation_contract", {}).get("step_stride") != stride:
        raise ValueError("compatibility receipt stride drifted")


def _run_map(
    *,
    label: str,
    checkpoint: Mapping[str, Any],
    store: Any,
    device: torch.device,
    stride: int,
    calls: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        build_model,
        timed_model_call,
    )

    model, normalization = build_model(checkpoint, device)
    if not np.array_equal(
        np.asarray(normalization.state_scale, dtype=np.float64),
        np.asarray(checkpoint["normalization"]["state_scale"], dtype=np.float64),
    ):
        raise ValueError(f"{label} model normalization was not reconstructed exactly")
    predictions: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, Any]] = []
    for case in FIXED_CASES:
        case_id = str(case["trajectory_id"])
        frame = int(case["frame"])
        sample = store.tensor_sample(
            case_id,
            frame,
            step_stride=stride,
            device=device,
        )
        observed_input = sample["current"][0].detach().cpu().numpy()
        if raw_array_sha256(observed_input) != case["input_frame_sha256"]:
            raise ValueError(f"{label} model input bytes differ from the fixed frame")
        current = sample["current"]
        logical_rows = []
        for logical_call in range(1, calls + 1):
            prediction, timing = timed_model_call(
                model,
                sample,
                current,
                device=device,
                amp="none",
                repeats=2,
            )
            logical_rows.append({"logical_call": logical_call, **timing})
            current = torch.as_tensor(
                np.asarray(prediction, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        predictions[case_id] = np.asarray(prediction, dtype=np.float64)
        timing_rows.append({"trajectory_id": case_id, "calls": logical_rows})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, {"label": label, "cases": timing_rows}


def _variant_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    initial: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, Any]:
    from utility.time_dependent_no.shock_vortex_metrics import endpoint_metrics

    delta = integrated_conservative_change(prediction, initial, volumes=volumes)
    return {
        "scaled_physical_volume_rms": scaled_rms_distance(
            prediction,
            reference,
            volumes=volumes,
        ),
        **weighted_relative_l2_metrics(
            prediction,
            reference,
            volumes=volumes,
        ),
        "primitive_error": primitive_relative_l2_metrics(
            prediction,
            reference,
            volumes=volumes,
        ),
        "state": admissibility_summary(prediction),
        "integrated_conservative_change": delta,
        "state_implied_outward_boundary_exchange": -delta,
        "structure": endpoint_metrics(
            prediction,
            reference,
            positions=positions,
            edges=edges,
            volumes=volumes,
            component_scale=np.asarray(CHANNEL_SCALES),
            gamma=1.4,
            shock_quantile=0.9,
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.owner_authorized_a2:
        raise ValueError("P0-A2 requires explicit owner authorization")
    inputs_root = args.inputs_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    a1_receipt = validate_a1_closeout(args.a1_root)

    # This comparison is deliberately complete before a checkpoint path is opened.
    primary_summary, primary_states, primary_exchanges = _load_solver_process(
        args.solver_primary_dir, process_role="primary"
    )
    repeat_summary, repeat_states, repeat_exchanges = _load_solver_process(
        args.solver_repeat_dir, process_role="fresh_repeat"
    )
    uniform_volumes = np.full(NATIVE_NODES, 2.0 / NATIVE_NODES)
    repeatability = compare_solver_process_summaries(
        primary_summary,
        repeat_summary,
        primary_states=primary_states,
        repeat_states=repeat_states,
        volumes=uniform_volumes,
    )
    require_a2_cublas_workspace_config(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    # Historical evaluator imports are deferred until the frozen solver gate passes.
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        load_checkpoint,
        select_device,
    )
    from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore

    if sha256_file(args.source_manifest) != args.expected_source_manifest_sha256:
        raise ValueError("A2 evaluator source-manifest SHA-256 mismatch")
    source_payload = load_json_object(args.source_manifest)
    source_members = validate_a2_source_manifest(
        source_payload,
        source_root=source_root,
        source_role="historical_evaluator",
    )

    artifact_path = resolve_manifest_member(
        inputs_root, "manifests/artifact_manifest.json"
    )
    artifact = load_bound_json(artifact_path, A0_ARTIFACT_MANIFEST_SHA256)
    artifact_members = validate_artifact_manifest(artifact)
    execution_path = _verified_member(
        inputs_root, artifact_members, "manifests/execution_contract.json"
    )
    validate_execution_contract(load_json_object(execution_path))
    case_path = _verified_member(
        inputs_root, artifact_members, "manifests/case_manifest.json"
    )
    if sha256_file(case_path) != CASE_MANIFEST_SHA256:
        raise ValueError("case manifest does not match the fixed A0 selection")
    validate_case_manifest(load_json_object(case_path))
    historical_path = _verified_member(
        inputs_root,
        artifact_members,
        "manifests/historical_evaluator_source_manifest.json",
    )
    _validate_historical_base_manifest(
        load_json_object(historical_path), source_payload
    )

    data_root = inputs_root / "data/full_trajectory_root"
    data_manifest_path = _verified_member(
        inputs_root,
        artifact_members,
        "data/full_trajectory_root/manifest.json",
    )
    if sha256_file(data_manifest_path) != DATASET_MANIFEST_SHA256:
        raise ValueError("dataset manifest does not match the fixed checkpoints")
    store = PCNOEuler2DShardStore(data_root, max_cached_trajectories=2)
    if store.manifest_digest != DATASET_MANIFEST_SHA256:
        raise ValueError("shard-store manifest digest drifted")
    if store.manifest.get("open_grouped_split_digest") not in (None, SPLIT_DIGEST):
        raise ValueError("shard-store split digest drifted")

    for case in FIXED_CASES:
        entry = store.entry(str(case["trajectory_id"]))
        if entry.get("split") != "validation":
            raise ValueError("A2 case is no longer open validation")
        folder = str(entry["folder"])
        expected_state = f"{folder}/states_conservative.npy"
        if expected_state != case["state_relative_path"]:
            raise ValueError("A2 case folder differs from the fixed case manifest")
        for name in ("states_conservative", *GEOMETRY_ARRAYS):
            _verified_member(
                inputs_root,
                artifact_members,
                f"data/full_trajectory_root/{folder}/{name}.npy",
            )

    checkpoint_paths = {
        "d044": _verified_member(
            inputs_root, artifact_members, "checkpoints/d044/best.pt"
        ),
        "d060": _verified_member(
            inputs_root, artifact_members, "checkpoints/d060/best.pt"
        ),
    }
    compatibility_paths = {
        "d044": _verified_member(
            inputs_root,
            artifact_members,
            "receipts/d044_compatibility_summary.json",
        ),
        "d060": _verified_member(
            inputs_root,
            artifact_members,
            "receipts/d060_compatibility_summary.json",
        ),
    }
    if sha256_file(compatibility_paths["d044"]) != D044_COMPATIBILITY_SHA256:
        raise ValueError("D044 compatibility receipt drifted")
    if sha256_file(compatibility_paths["d060"]) != D060_COMPATIBILITY_SHA256:
        raise ValueError("D060 compatibility receipt drifted")
    _compatibility_contract(load_json_object(compatibility_paths["d044"]), stride=1)
    _compatibility_contract(load_json_object(compatibility_paths["d060"]), stride=2)
    if sha256_file(checkpoint_paths["d044"]) != D044_CHECKPOINT_SHA256:
        raise ValueError("D044 checkpoint bytes drifted")
    if sha256_file(checkpoint_paths["d060"]) != D060_CHECKPOINT_SHA256:
        raise ValueError("D060 checkpoint bytes drifted")

    # Checkpoint deserialization starts only after both frozen solver processes passed.
    checkpoints = {
        "d044": load_checkpoint(checkpoint_paths["d044"]),
        "d060": load_checkpoint(checkpoint_paths["d060"]),
    }
    _validate_checkpoint(checkpoints["d044"], label="D044", stride=1)
    _validate_checkpoint(checkpoints["d060"], label="D060", stride=2)

    device = select_device(args.device)
    if device.type != "cuda":
        raise ValueError("A2 fixed-checkpoint compatibility requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = perf_counter()
    d044, d044_timing = _run_map(
        label="D044",
        checkpoint=checkpoints["d044"],
        store=store,
        device=device,
        stride=1,
        calls=2,
    )
    d060, d060_timing = _run_map(
        label="D060",
        checkpoint=checkpoints["d060"],
        store=store,
        device=device,
        stride=2,
        calls=1,
    )

    case_rows: list[dict[str, Any]] = []
    for case in FIXED_CASES:
        case_id = str(case["trajectory_id"])
        frame = int(case["frame"])
        stored = store.states(case_id)
        stored_initial = np.asarray(stored[frame])
        stored_reference = np.asarray(stored[frame + 2])
        if raw_array_sha256(stored_reference) != case["reference_f_plus_2_sha256"]:
            raise ValueError("fixed reference frame bytes drifted")
        initial = np.asarray(stored_initial, dtype=np.float64)
        reference = np.asarray(stored_reference, dtype=np.float64)
        volumes = validated_a2_physical_volumes(
            store.array(case_id, "node_measures")
        )
        positions = np.asarray(store.array(case_id, "nodes"), dtype=np.float64)
        edges = np.asarray(store.array(case_id, "edges"), dtype=np.int64)
        solver_metrics = _variant_metrics(
            primary_states[case_id],
            reference,
            initial,
            positions=positions,
            edges=edges,
            volumes=volumes,
        )
        solver_delta = np.asarray(
            solver_metrics["integrated_conservative_change"], dtype=np.float64
        )
        solver_metrics["native_solver_recorded_outward_boundary_exchange"] = (
            primary_exchanges[case_id]
        )
        solver_metrics["native_solver_own_balance_residual"] = float(
            np.linalg.norm(solver_delta + primary_exchanges[case_id])
            / max(
                np.linalg.norm(solver_delta),
                np.linalg.norm(primary_exchanges[case_id]),
                1.0,
            )
        )
        d044_metrics = _variant_metrics(
            d044[case_id],
            reference,
            initial,
            positions=positions,
            edges=edges,
            volumes=volumes,
        )
        d060_metrics = _variant_metrics(
            d060[case_id],
            reference,
            initial,
            positions=positions,
            edges=edges,
            volumes=volumes,
        )
        separation = scaled_rms_distance(d044[case_id], d060[case_id], volumes=volumes)
        gate = bias_gate_row(
            baseline=float(solver_metrics["scaled_physical_volume_rms"]),
            d044=float(d044_metrics["scaled_physical_volume_rms"]),
            d060=float(d060_metrics["scaled_physical_volume_rms"]),
            separation=separation,
        )
        case_label = f"{case_id}_f{frame}"
        d044_path = f"cases/{case_label}/d044_proposal.npy"
        d060_path = f"cases/{case_label}/d060_proposal.npy"
        atomic_save_npy(args.output_dir / d044_path, d044[case_id])
        atomic_save_npy(args.output_dir / d060_path, d060[case_id])
        case_rows.append(
            {
                "trajectory_id": case_id,
                "frame": frame,
                "split": "validation",
                "input_frame_sha256": case["input_frame_sha256"],
                "reference_frame_sha256": case["reference_f_plus_2_sha256"],
                "fresh_repeat_exchange_max_abs_difference": float(
                    np.max(
                        np.abs(primary_exchanges[case_id] - repeat_exchanges[case_id])
                    )
                ),
                "d044_proposal_path": d044_path,
                "d044_proposal_raw_sha256": raw_array_sha256(d044[case_id]),
                "d060_proposal_path": d060_path,
                "d060_proposal_raw_sha256": raw_array_sha256(d060[case_id]),
                "solver": solver_metrics,
                "d044": d044_metrics,
                "d060": d060_metrics,
                "gate": gate,
            }
        )

    proxy_pass = all(bool(row["gate"]["pass"]) for row in case_rows)
    summary = {
        "schema": A2_MODEL_SUMMARY_SCHEMA,
        "stage": "P0-A2",
        "status": "complete_pass" if proxy_pass else "complete_fail_bias_gate",
        "proxy_gate_pass": proxy_pass,
        "claim_boundary": (
            "Three fixed open-validation stored-state restarts and two immutable "
            "historical checkpoints at one common 0.02 horizon. This is only a "
            "native-coarse-solver proxy qualification result, not A3, training, "
            "checkpoint selection, test evidence, or a model ranking."
        ),
        "authorization": {"a2": True, "a3": False},
        "activity": {
            "solver_processes_consumed": 2,
            "checkpoint_files_opened": 2,
            "checkpoint_deserialized": 2,
            "models_constructed": 2,
            "logical_d044_calls": 2 * len(FIXED_CASES),
            "logical_d060_calls": len(FIXED_CASES),
            "training_executed": False,
            "checkpoint_selection_performed": False,
            "test_or_strength_ood_arrays_opened": False,
        },
        "bindings": {
            "a0_artifact_manifest_sha256": A0_ARTIFACT_MANIFEST_SHA256,
            "a1_closeout_manifest_sha256": A1_CLOSEOUT_MANIFEST_SHA256,
            "a1_summary_payload_sha256": a1_receipt["summary_payload_sha256"],
            "evaluator_source_manifest_sha256": sha256_file(args.source_manifest),
            "evaluator_source_mapping_sha256": canonical_json_sha256(source_members),
            "historical_base_mapping_sha256": HISTORICAL_EVALUATOR_MAPPING_SHA256,
            "solver_primary_artifact_manifest_sha256": sha256_file(
                args.solver_primary_dir / "artifact_manifest.json"
            ),
            "solver_repeat_artifact_manifest_sha256": sha256_file(
                args.solver_repeat_dir / "artifact_manifest.json"
            ),
            "d044_checkpoint_sha256": D044_CHECKPOINT_SHA256,
            "d060_checkpoint_sha256": D060_CHECKPOINT_SHA256,
            "d044_compatibility_receipt_sha256": D044_COMPATIBILITY_SHA256,
            "d060_compatibility_receipt_sha256": D060_COMPATIBILITY_SHA256,
        },
        "selection": {
            "checkpoint_selection_performed": False,
            "d044": "fixed historical best.pt before A2",
            "d060": "fixed historical best.pt before A2",
        },
        "metric": {
            "channel_scales": list(CHANNEL_SCALES),
            "bias_ratio_limit": 0.25,
            "zero_or_unresolved_denominator_policy": "fail_without_epsilon",
            "shock_quantile": 0.9,
        },
        "solver_process_repeatability": repeatability,
        "cases": case_rows,
        "timing": {
            "d044": d044_timing,
            "d060": d060_timing,
            "model_and_metric_elapsed_seconds": perf_counter() - started,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "amp": "none",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "gpu_name": torch.cuda.get_device_name(device),
        },
    }
    summary = bind_payload_sha256(summary)
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(
        args.output_dir / "artifact_manifest.json",
        build_artifact_manifest(args.output_dir, schema=A2_MODEL_ARTIFACT_SCHEMA),
    )
    store.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_fresh_directory(args.output_dir)
    try:
        summary = _run(args)
    except Exception as exc:
        failure = {
            "schema": A2_MODEL_SUMMARY_SCHEMA,
            "stage": "P0-A2",
            "status": "failed_infrastructure_or_solver_gate",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "authorization": {"a2": bool(args.owner_authorized_a2), "a3": False},
        }
        atomic_write_json(args.output_dir / "summary.json", failure)
        atomic_write_json(
            args.output_dir / "artifact_manifest.json",
            build_artifact_manifest(args.output_dir, schema=A2_MODEL_ARTIFACT_SCHEMA),
        )
        print(f"P0-A2 evaluator failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"P0-A2 {summary['status']}: proxy_gate_pass={summary['proxy_gate_pass']} "
        f"payload={summary['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
