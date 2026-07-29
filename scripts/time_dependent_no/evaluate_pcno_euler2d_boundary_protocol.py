#!/usr/bin/env python3
"""Compare a checkpoint's native recurrence with minimum-change boundaries.

This is a validation-only protocol-selection evaluator. It never selects or
opens the manifest test split. Final test reporting remains the responsibility
of the official rollout evaluator after a protocol is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    build_model,
    endpoint_diagnostics,
    load_checkpoint,
    sha256_file,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (  # noqa: E402
    CAUSAL_BOUNDARY_MODE,
    LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    MINIMUM_CHANGE_BOUNDARY_MODE,
    NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    RAW_ALL_NODES_PRIMARY_OBJECTIVE,
    autocast_context,
    boundary_outflow_normal_mach,
    close_boundary,
    contract_forward_sample,
    evaluate_pairs,
    evaluate_rollouts,
    failure_cause,
    optional_region_relative_l2,
    select_device,
    write_json,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    balanced_presentations,
    boundary_band_normal_node_mask,
    build_graph_causal_boundary_policy,
    build_graph_minimum_change_boundary_policy,
    conservative_to_primitive_torch,
    normal_node_mask,
)

SCHEMA = "pcno_euler2d_boundary_protocol_validation_v1"
STRUCTURE_ERROR_FIELDS = (
    "smooth_highpass_energy_reconstructed_weight_proxy",
    "front_centroid_distance",
    "shock_thickness_log_error",
    "shock_strength_log_error",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--one-step-presentations", type=int, default=128)
    parser.add_argument("--presentation-seed", type=int, default=20260729)
    parser.add_argument("--rollout-count", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--rollout-checkpoints", type=int, nargs="+", default=(1, 5, 10, 20)
    )
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--boundary-rho-inf", type=float, default=1.4)
    parser.add_argument("--boundary-p-inf", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.one_step_presentations < 1 or args.rollout_count < 1:
        raise ValueError("presentation and rollout counts must be positive")
    if args.rollout_steps < 1 or args.start_frame < 0:
        raise ValueError("rollout steps must be positive and start frame nonnegative")
    checkpoints = sorted({int(value) for value in args.rollout_checkpoints})
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > args.rollout_steps:
        raise ValueError("rollout checkpoints must lie inside the requested horizon")
    args.rollout_checkpoints = checkpoints
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie inside (0, 1)")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")


def _policy_set(
    mode: str,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    device: torch.device,
    rho_inf: float,
    p_inf: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if mode == "model_all_nodes":
        return {}, {}
    policies: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    contract = checkpoint.get("boundary_contract", {})
    for key in keys:
        if mode == CAUSAL_BOUNDARY_MODE:
            policy, record = build_graph_causal_boundary_policy(
                store,
                key,
                device=device,
                max_source_hops=int(contract.get("max_source_hops", 3)),
                rho_inf=float(contract.get("rho_inf", rho_inf)),
                p_inf=float(contract.get("p_inf", p_inf)),
            )
        elif mode == MINIMUM_CHANGE_BOUNDARY_MODE:
            policy, record = build_graph_minimum_change_boundary_policy(
                store,
                key,
                device=device,
                rho_inf=float(contract.get("rho_inf", rho_inf)),
                p_inf=float(contract.get("p_inf", p_inf)),
            )
        else:
            raise ValueError(f"unsupported boundary mode: {mode}")
        expected = contract.get("policy_digests", {}).get(key)
        if expected is not None and record["policy_digest"] != expected:
            raise ValueError(f"trajectory {key} native boundary policy digest changed")
        policies[key] = policy
        metadata[key] = record
    return policies, metadata


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _structure_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    rollout_checkpoints: Sequence[int],
) -> dict[str, Any]:
    fields = (
        *STRUCTURE_ERROR_FIELDS,
        "front_iou",
        "front_symmetric_chamfer",
        "smooth_highpass_energy_equal_node_proxy",
    )
    endpoints: dict[str, dict[str, Any]] = {}
    for checkpoint in rollout_checkpoints:
        selected = [row for row in rows if row["call_index"] == checkpoint]
        endpoint: dict[str, Any] = {"population_count": len(selected)}
        for field in fields:
            values = [
                float(row[field]) for row in selected if row.get(field) is not None
            ]
            endpoint[field] = {
                "count": len(values),
                "mean": None if not values else float(np.mean(values)),
                "median": None if not values else float(np.median(values)),
            }
        endpoints[str(checkpoint)] = endpoint
    return {"rows": list(rows), "endpoints": endpoints}


def _paired_structure_ratios(
    native: Mapping[str, Any],
    minimum_change: Mapping[str, Any],
    *,
    rollout_checkpoints: Sequence[int],
) -> dict[str, Any]:
    native_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row for row in native["rows"]
    }
    minimum_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row
        for row in minimum_change["rows"]
    }
    result: dict[str, Any] = {}
    for checkpoint in rollout_checkpoints:
        endpoint: dict[str, Any] = {}
        for field in STRUCTURE_ERROR_FIELDS:
            ratios = []
            for key, native_row in native_rows.items():
                if key[1] != checkpoint or key not in minimum_rows:
                    continue
                baseline = native_row.get(field)
                candidate = minimum_rows[key].get(field)
                if (
                    baseline is not None
                    and candidate is not None
                    and float(baseline) > 0.0
                ):
                    ratios.append(float(candidate) / float(baseline))
            endpoint[field] = {
                "count": len(ratios),
                "mean_ratio": None if not ratios else float(np.mean(ratios)),
                "median_ratio": None if not ratios else float(np.median(ratios)),
            }
        result[str(checkpoint)] = endpoint
    return {
        "ratio": "minimum_change_over_native",
        "direction": "less_than_or_equal_to_one_is_nonworse",
        "paired_by": ["trajectory", "call_index"],
        "endpoints": result,
    }


@torch.no_grad()
def rollout_structure_diagnostics(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    policies: Mapping[str, Mapping[str, Any]],
    *,
    step_stride: int,
    start_frame: int,
    num_steps: int,
    rollout_checkpoints: Sequence[int],
    shock_quantile: float,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    component_scale = model.state_scale.detach().float().cpu().numpy()
    model.eval()
    for key in keys:
        states = store.states(key)
        available = (states.shape[0] - 1 - start_frame) // step_stride
        if available < num_steps:
            raise ValueError(
                f"trajectory {key} exposes only {available} structural calls; "
                f"{num_steps} were requested"
            )
        sample = store.tensor_sample(
            key, start_frame, step_stride=step_stride, device=device
        )
        policy = policies.get(key)
        current = sample["current"]
        positions = np.array(store.array(key, "nodes"), copy=True)
        edges = np.array(store.array(key, "edges"), copy=True)
        node_type = np.array(store.array(key, "node_type"), copy=True).reshape(-1)
        proxy_weights = np.array(store.array(key, "node_weights"), copy=True).sum(
            axis=-1
        )
        valid_length = 0
        stopped_for = "completed"
        for call_index in range(1, num_steps + 1):
            with autocast_context(device, amp):
                prediction, _, _ = contract_forward_sample(
                    model,
                    sample,
                    current,
                    boundary_policy=policy,
                )
            stopped_for, _ = failure_cause(prediction, gamma=model.gamma)
            if stopped_for is not None:
                break
            if policy is not None:
                outflow = boundary_outflow_normal_mach(prediction.float(), policy)
                if (
                    outflow is None
                    or not bool(torch.isfinite(outflow).all())
                    or float(outflow.min().cpu()) <= 1.0
                ):
                    stopped_for = "invalid_outflow_regime"
                    break
            valid_length = call_index
            if call_index in rollout_checkpoints:
                target_index = start_frame + call_index * step_stride
                target = np.array(states[target_index], copy=True)
                rows.append(
                    {
                        "trajectory": key,
                        "call_index": call_index,
                        **endpoint_diagnostics(
                            prediction[0].float().cpu().numpy(),
                            target,
                            positions=positions,
                            edges=edges,
                            node_type=node_type,
                            proxy_weights=proxy_weights,
                            component_scale=component_scale,
                            gamma=model.gamma,
                            shock_quantile=shock_quantile,
                        ),
                    }
                )
            current = prediction
        trajectories.append(
            {
                "trajectory": key,
                "requested_steps": num_steps,
                "valid_length": valid_length,
                "completed": valid_length == num_steps,
                "failure_cause": (
                    "completed" if valid_length == num_steps else stopped_for
                ),
            }
        )
    return {
        "trajectories": trajectories,
        "completion_rate": (
            sum(bool(row["completed"]) for row in trajectories) / len(trajectories)
        ),
        **_structure_summary(rows, rollout_checkpoints=rollout_checkpoints),
    }


@torch.no_grad()
def projection_decomposition(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    policies: Mapping[str, Mapping[str, Any]],
    *,
    step_stride: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {
        "reference_closure_bias_all_relative_l2": [],
        "reference_closure_bias_boundary_relative_l2": [],
        "model_under_closure_all_relative_l2": [],
        "model_under_closure_boundary_relative_l2": [],
        "raw_to_corrected_intervention_rms": [],
        "near_boundary_1_reference_relative_l2": [],
        "near_boundary_2_reference_relative_l2": [],
        "near_boundary_3_reference_relative_l2": [],
        "wall_constraint_rms": [],
        "inflow_constraint_rms": [],
        "outflow_normal_mach_min": [],
    }
    wall_constraint_max = 0.0
    inflow_constraint_max = 0.0
    model.eval()
    for key, time_index in pairs:
        sample = store.tensor_batch(
            key, [time_index], step_stride=step_stride, device=device
        )
        policy = policies[key]
        with autocast_context(device, amp):
            prediction, raw_prediction, _ = contract_forward_sample(
                model,
                sample,
                sample["current"],
                boundary_policy=policy,
            )
            projected_target = close_boundary(
                sample["target"], policy, gamma=model.gamma
            )
        normal = normal_node_mask(sample["node_type"], sample["node_mask"])
        boundary = sample["node_mask"] - normal

        def append_relative(
            name: str,
            left: torch.Tensor,
            right: torch.Tensor,
            mask: torch.Tensor,
            pair_sample: Mapping[str, torch.Tensor] = sample,
            evaluated_model: PCNOEuler2DResidual = model,
        ) -> None:
            value = optional_region_relative_l2(
                left, right, pair_sample, mask, evaluated_model
            )
            if value is not None:
                metrics[name].append(value)

        append_relative(
            "reference_closure_bias_all_relative_l2",
            projected_target.float(),
            sample["target"],
            sample["node_mask"],
        )
        append_relative(
            "reference_closure_bias_boundary_relative_l2",
            projected_target.float(),
            sample["target"],
            boundary,
        )
        append_relative(
            "model_under_closure_all_relative_l2",
            prediction.float(),
            projected_target.float(),
            sample["node_mask"],
        )
        append_relative(
            "model_under_closure_boundary_relative_l2",
            prediction.float(),
            projected_target.float(),
            boundary,
        )
        scale = model.state_scale.to(dtype=prediction.dtype, device=device)
        intervention = ((prediction - raw_prediction) / scale).square() * boundary
        denominator = boundary.sum() * prediction.shape[-1]
        metrics["raw_to_corrected_intervention_rms"].append(
            float(torch.sqrt(intervention.sum() / denominator.clamp_min(1.0)).cpu())
        )
        for hops in (1, 2, 3):
            band = boundary_band_normal_node_mask(
                sample["node_type"],
                sample["node_mask"],
                sample["directed_edges"],
                max_hops=hops,
            )
            append_relative(
                f"near_boundary_{hops}_reference_relative_l2",
                prediction.float(),
                sample["target"],
                band,
            )

        primitive = conservative_to_primitive_torch(
            prediction.float(), gamma=model.gamma
        )
        wall_nodes = policy["wall_constrained_nodes"]
        projectors = policy["wall_velocity_projectors"].to(
            dtype=primitive.dtype, device=device
        )
        wall_velocity = primitive[:, wall_nodes, 1:3]
        forbidden = wall_velocity - torch.einsum(
            "nij,bnj->bni", projectors, wall_velocity
        )
        metrics["wall_constraint_rms"].append(
            float(torch.sqrt(forbidden.square().mean()).cpu())
        )
        wall_constraint_max = max(
            wall_constraint_max, float(forbidden.abs().max().cpu())
        )
        inflow = primitive[:, policy["inflow_nodes"]]
        stream = policy["freestream"].to(dtype=inflow.dtype, device=device)
        inflow_difference = inflow - stream
        metrics["inflow_constraint_rms"].append(
            float(torch.sqrt(inflow_difference.square().mean()).cpu())
        )
        inflow_constraint_max = max(
            inflow_constraint_max, float(inflow_difference.abs().max().cpu())
        )
        outflow = boundary_outflow_normal_mach(prediction.float(), policy)
        if outflow is not None:
            metrics["outflow_normal_mach_min"].append(float(outflow.min().cpu()))
    result = {name: _mean(values) for name, values in metrics.items()}
    result["wall_constraint_max_abs"] = wall_constraint_max
    result["inflow_constraint_max_abs"] = inflow_constraint_max
    result["outflow_normal_mach_min"] = (
        min(metrics["outflow_normal_mach_min"])
        if metrics["outflow_normal_mach_min"]
        else None
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    validate_args(args, device)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = load_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir)
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and data manifest differ")
    if (
        args.expected_manifest_sha256
        and args.expected_manifest_sha256 != store.manifest_digest
    ):
        raise ValueError("manifest SHA-256 mismatch")
    validation_keys = [str(key) for key in checkpoint["val_keys"]]
    if not validation_keys or any(key not in store.keys for key in validation_keys):
        raise ValueError(
            "checkpoint validation split is unavailable in the shard store"
        )
    if args.rollout_count > len(validation_keys):
        raise ValueError("rollout count exceeds the checkpoint validation split")
    model = build_model(checkpoint, device)
    step_stride = int(checkpoint["step_stride"])
    pairs = balanced_presentations(
        store,
        validation_keys,
        step_stride=step_stride,
        count=args.one_step_presentations,
        rng=np.random.default_rng(args.presentation_seed),
    )
    rollout_keys = validation_keys[: args.rollout_count]

    native_mode = str(checkpoint["boundary_mode"])
    native_policies, native_metadata = _policy_set(
        native_mode,
        checkpoint,
        store,
        validation_keys,
        device=device,
        rho_inf=args.boundary_rho_inf,
        p_inf=args.boundary_p_inf,
    )
    minimum_policies, minimum_metadata = _policy_set(
        MINIMUM_CHANGE_BOUNDARY_MODE,
        {},
        store,
        validation_keys,
        device=device,
        rho_inf=args.boundary_rho_inf,
        p_inf=args.boundary_p_inf,
    )
    saved_primary = str(
        checkpoint.get("training_args", {}).get(
            "primary_objective", NORMAL_CLOSED_PRIMARY_OBJECTIVE
        )
    )
    if saved_primary not in {
        NORMAL_CLOSED_PRIMARY_OBJECTIVE,
        RAW_ALL_NODES_PRIMARY_OBJECTIVE,
        LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    }:
        raise ValueError(f"unsupported checkpoint primary objective: {saved_primary}")
    native_one_step = evaluate_pairs(
        model,
        store,
        pairs,
        step_stride=step_stride,
        batch_size=1,
        device=device,
        amp=args.amp,
        primary_objective=saved_primary,
        boundary_policies=native_policies,
    )
    minimum_one_step = evaluate_pairs(
        model,
        store,
        pairs,
        step_stride=step_stride,
        batch_size=1,
        device=device,
        amp=args.amp,
        primary_objective=LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
        boundary_policies=minimum_policies,
    )
    native_rollout = evaluate_rollouts(
        model,
        store,
        rollout_keys,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
        boundary_policies=native_policies,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    minimum_rollout = evaluate_rollouts(
        model,
        store,
        rollout_keys,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
        boundary_policies=minimum_policies,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    native_structure = rollout_structure_diagnostics(
        model,
        store,
        rollout_keys,
        native_policies,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        rollout_checkpoints=args.rollout_checkpoints,
        shock_quantile=args.shock_quantile,
        device=device,
        amp=args.amp,
    )
    minimum_structure = rollout_structure_diagnostics(
        model,
        store,
        rollout_keys,
        minimum_policies,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        rollout_checkpoints=args.rollout_checkpoints,
        shock_quantile=args.shock_quantile,
        device=device,
        amp=args.amp,
    )
    structure_ratios = _paired_structure_ratios(
        native_structure,
        minimum_structure,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    decomposition = projection_decomposition(
        model,
        store,
        pairs,
        minimum_policies,
        step_stride=step_stride,
        device=device,
        amp=args.amp,
    )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
        ROOT / "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
        ROOT / "utility/time_dependent_no/pcno_euler2d.py",
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "selection_population": "checkpoint_validation_split_only",
        "test_split_opened": False,
        "checkpoint": {
            "path_name": args.checkpoint.name,
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "boundary_mode": native_mode,
            "primary_objective": saved_primary,
            "data_manifest_digest": checkpoint["data_manifest_digest"],
        },
        "evaluation": {
            "validation_keys": validation_keys,
            "one_step_pairs": [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in pairs
            ],
            "presentation_seed": args.presentation_seed,
            "rollout_keys": rollout_keys,
            "rollout_steps": args.rollout_steps,
            "rollout_checkpoints": args.rollout_checkpoints,
            "shock_quantile": args.shock_quantile,
            "step_stride": step_stride,
            "device": str(device),
            "amp": args.amp,
        },
        "native": {
            "one_step": native_one_step,
            "rollout": native_rollout,
            "structure": native_structure,
            "policy_metadata": native_metadata,
        },
        "minimum_change": {
            "one_step": minimum_one_step,
            "rollout": minimum_rollout,
            "structure": minimum_structure,
            "projection_decomposition": decomposition,
            "policy_metadata": minimum_metadata,
        },
        "paired_structure_ratios": structure_ratios,
        "claim_boundary": {
            "verified": "frozen-checkpoint validation behavior under two causal recurrences",
            "inference": "whether minimum-change enforcement is promising enough for matched training",
            "not_supported": [
                "exact DG boundary replay",
                "physical conservation",
                "test or distribution-shift performance",
                "causal attribution to training before a matched checkpoint exists",
            ],
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "native_horizon": native_rollout["mean_endpoint_relative_l2"],
                "minimum_change_horizon": minimum_rollout["mean_endpoint_relative_l2"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    store.close()


if __name__ == "__main__":
    main()
