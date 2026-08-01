#!/usr/bin/env python3
"""Decompose one causal-boundary PCNO rollout into fresh and propagated error.

For the deployed map ``G`` this validation-only diagnostic records

``G(uhat_t) - u_(t+1) =
  [G(uhat_t) - G(u_t)] + [G(u_t) - u_(t+1)]``.

The first bracket measures the response to incoming rollout error and the
second is the fresh teacher-forced defect.  Reference states are used only for
diagnostic model calls, targets, and masks; they never alter the autonomous
rollout.  Bump node weights remain reconstructed quadrature proxies and are not
physical finite-volume measures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    CAUSAL_BOUNDARY_MODE,
    build_model,
    load_checkpoint,
    select_device,
    sha256_file,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (  # noqa: E402
    contract_forward_sample,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
    build_graph_causal_boundary_policy,
    reference_smooth_region_mask,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    node_highpass_field,
    normalized_node_weights,
    raw_admissibility_summary,
)


ERROR_SOURCE_SCHEMA = "pcno_euler2d_causal_error_source_v1"
IDENTITY_RELATIVE_TOLERANCE = 1.0e-6
CALL1_PROPAGATED_SHARE_TOLERANCE = 1.0e-6
DEFAULT_ENDPOINT_CALLS = (1, 5, 10, 20, 40, 60, 79)
REGION_NAMES = (
    "all_nodes_full",
    "normal_nodes_full",
    "boundary_nodes_full",
    "smooth_full",
    "front_support_full",
    "smooth_highpass",
    "front_support_highpass",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-keys", nargs="*", default=None)
    parser.add_argument("--expected-trajectory-count", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=79)
    parser.add_argument(
        "--endpoint-calls",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENDPOINT_CALLS),
    )
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--shock-dilation-hops", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.expected_trajectory_count <= 0:
        raise ValueError("--expected-trajectory-count must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be nonnegative")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    if args.shock_dilation_hops < 0:
        raise ValueError("--shock-dilation-hops must be nonnegative")
    endpoints = tuple(int(value) for value in args.endpoint_calls)
    if len(set(endpoints)) != len(endpoints) or any(
        value < 1 or value > args.num_steps for value in endpoints
    ):
        raise ValueError("--endpoint-calls must be unique calls inside the horizon")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _state_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.array(state, copy=True), dtype=torch.float32, device=device
    ).unsqueeze(0)


def _weighted_norm(
    field: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    values = np.asarray(field, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or selected.shape != (values.shape[0],):
        raise ValueError("field and mask must have shapes [N,C] and [N]")
    if not np.any(selected):
        return None
    mass = normalized_node_weights(weights, name="rollout-error decomposition")
    energy = float(np.sum(mass[selected, None] * values[selected] ** 2))
    return float(np.sqrt(max(energy, 0.0)))


def weighted_decomposition_metrics(
    total: np.ndarray,
    propagated: np.ndarray,
    fresh_defect: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure an additive vector decomposition under one fixed node measure."""

    total_value = np.asarray(total, dtype=np.float64)
    propagated_value = np.asarray(propagated, dtype=np.float64)
    fresh_value = np.asarray(fresh_defect, dtype=np.float64)
    if (
        total_value.ndim != 2
        or propagated_value.shape != total_value.shape
        or fresh_value.shape != total_value.shape
    ):
        raise ValueError("decomposition fields must share shape [N,C]")
    selected = (
        np.ones(total_value.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if selected.shape != (total_value.shape[0],):
        raise ValueError("decomposition mask must have shape [N]")
    if not np.any(selected):
        return {"status": "empty_region", "selected_node_count": 0}
    mass = normalized_node_weights(weights, name="rollout-error decomposition")
    chosen_mass = mass[selected, None]

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(chosen_mass * left[selected] * right[selected]))

    total_energy = inner(total_value, total_value)
    propagated_energy = inner(propagated_value, propagated_value)
    fresh_energy = inner(fresh_value, fresh_value)
    cross_inner = inner(propagated_value, fresh_value)
    residual = total_value - propagated_value - fresh_value
    residual_energy = inner(residual, residual)
    total_norm = math.sqrt(max(total_energy, 0.0))
    propagated_norm = math.sqrt(max(propagated_energy, 0.0))
    fresh_norm = math.sqrt(max(fresh_energy, 0.0))
    residual_norm = math.sqrt(max(residual_energy, 0.0))
    magnitude_sum = propagated_norm + fresh_norm
    identity_energy = propagated_energy + fresh_energy + 2.0 * cross_inner
    norm_scale = max(total_norm, magnitude_sum, 1.0e-30)
    energy_scale = max(
        abs(total_energy),
        propagated_energy + fresh_energy + 2.0 * abs(cross_inner),
        1.0e-30,
    )
    cosine = None
    if propagated_norm > 1.0e-30 and fresh_norm > 1.0e-30:
        cosine = cross_inner / (propagated_norm * fresh_norm)
    total_fraction_scale = max(abs(total_energy), 1.0e-30)
    return {
        "status": "available",
        "selected_node_count": int(np.count_nonzero(selected)),
        "selected_weight": float(mass[selected].sum()),
        "total_norm": total_norm,
        "propagated_norm": propagated_norm,
        "fresh_defect_norm": fresh_norm,
        "propagated_magnitude_share": (
            None if magnitude_sum <= 1.0e-30 else propagated_norm / magnitude_sum
        ),
        "propagated_fresh_cosine": cosine,
        "propagated_energy_fraction_of_total": (
            propagated_energy / total_fraction_scale
        ),
        "fresh_defect_energy_fraction_of_total": fresh_energy / total_fraction_scale,
        "cross_energy_fraction_of_total": 2.0 * cross_inner / total_fraction_scale,
        "relative_reconstruction_residual": residual_norm / norm_scale,
        "relative_energy_identity_residual": (
            abs(total_energy - identity_energy) / energy_scale
        ),
    }


def _reference_masks(
    current: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    *,
    shock_quantile: float,
    dilation_hops: int,
    gamma: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    node_type = sample["node_type"].reshape(1, -1, 1)
    valid = sample["node_mask"].to(dtype=torch.bool)
    normal = (node_type == 0) & valid
    boundary = (node_type != 0) & valid
    if not bool(normal.any()):
        raise ValueError("decomposition requires normal nodes")
    current_smooth = reference_smooth_region_mask(
        current,
        sample["directed_edges"],
        sample["node_mask"],
        interior_mask=normal,
        shock_quantile=shock_quantile,
        dilation_hops=dilation_hops,
        gamma=gamma,
    )
    target_smooth = reference_smooth_region_mask(
        target,
        sample["directed_edges"],
        sample["node_mask"],
        interior_mask=normal,
        shock_quantile=shock_quantile,
        dilation_hops=dilation_hops,
        gamma=gamma,
    )
    normal_np = normal[0, :, 0].cpu().numpy().astype(bool)
    boundary_np = boundary[0, :, 0].cpu().numpy().astype(bool)
    all_nodes_np = valid[0, :, 0].cpu().numpy().astype(bool)
    smooth = (
        (current_smooth[0, :, 0] & target_smooth[0, :, 0]).cpu().numpy().astype(bool)
    )
    smooth_fallback = not bool(np.any(smooth))
    if smooth_fallback:
        smooth = normal_np.copy()
    front = normal_np & ~smooth
    masks = {
        "all_nodes": all_nodes_np,
        "normal_nodes": normal_np,
        "boundary_nodes": boundary_np,
        "smooth": smooth,
        "front_support": front,
    }
    contract = {
        "source": "reference_current_and_target_shock_union",
        "shock_quantile": float(shock_quantile),
        "dilation_hops": int(dilation_hops),
        "smooth_fallback_to_normal_nodes": smooth_fallback,
        **{
            f"{name}_node_count": int(np.count_nonzero(mask))
            for name, mask in masks.items()
        },
    }
    return masks, contract


def _gain(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 1.0e-30:
        return None
    return float(numerator / denominator)


def _finite_values(values: Sequence[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _aggregate(values: Sequence[Any]) -> dict[str, float | int | None]:
    finite = _finite_values(values)
    return {
        "count": len(finite),
        "mean": None if not finite else float(np.mean(finite)),
        "median": None if not finite else float(np.median(finite)),
        "minimum": None if not finite else float(np.min(finite)),
        "maximum": None if not finite else float(np.max(finite)),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], endpoint_calls: Sequence[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scalar_fields = (
        "total_norm",
        "propagated_norm",
        "fresh_defect_norm",
        "propagated_magnitude_share",
        "propagated_fresh_cosine",
        "propagated_energy_fraction_of_total",
        "fresh_defect_energy_fraction_of_total",
        "cross_energy_fraction_of_total",
    )
    for call_index in endpoint_calls:
        selected = [row for row in rows if int(row["call_index"]) == int(call_index)]
        regions: dict[str, Any] = {}
        for region_name in REGION_NAMES:
            regions[region_name] = {
                field: _aggregate(
                    [row["regions"].get(region_name, {}).get(field) for row in selected]
                )
                for field in scalar_fields
            }
        result[str(call_index)] = {
            "row_count": len(selected),
            "regions": regions,
            "input_error": {
                field: _aggregate([row["input_error"].get(field) for row in selected])
                for field in (
                    "normal_full_norm",
                    "smooth_highpass_norm",
                    "normal_full_propagation_gain",
                    "smooth_highpass_propagation_gain",
                )
            },
        }
    return result


def _selected_keys(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> list[str]:
    checkpoint_val = [str(value) for value in checkpoint.get("val_keys", [])]
    if not checkpoint_val:
        raise ValueError("checkpoint has no validation-key declaration")
    keys = (
        checkpoint_val
        if args.trajectory_keys is None
        else [str(value) for value in args.trajectory_keys]
    )
    if len(keys) != len(set(keys)):
        raise ValueError("trajectory keys must be unique")
    if len(keys) != args.expected_trajectory_count:
        raise ValueError(
            f"expected {args.expected_trajectory_count} trajectories, got {len(keys)}"
        )
    if not set(keys).issubset(checkpoint_val):
        raise ValueError("selected trajectories are not all checkpoint validation keys")
    if not set(keys).issubset(set(store.keys)):
        raise ValueError("selected trajectories are missing from the shard store")
    checkpoint_test = {str(value) for value in checkpoint.get("test_keys", [])}
    if set(keys) & checkpoint_test:
        raise ValueError("selected validation keys overlap checkpoint test keys")
    return keys


def _source_hashes() -> dict[str, str]:
    files = (
        Path(__file__),
        ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
        ROOT / "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
        ROOT / "utility/time_dependent_no/pcno_euler2d.py",
        ROOT / "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    )
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in files
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["boundary_mode"] != CAUSAL_BOUNDARY_MODE:
        raise ValueError(
            "this diagnostic requires the checkpoint-native causal boundary map"
        )
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if store.manifest_digest != checkpoint["data_manifest_digest"]:
            raise ValueError("checkpoint and shard manifest digests differ")
        keys = _selected_keys(args, checkpoint, store)
        step_stride = int(checkpoint["step_stride"])
        boundary_contract = checkpoint["boundary_contract"]
        model = build_model(checkpoint, device)
        component_scale = np.asarray(
            checkpoint["normalization"]["state_scale"], dtype=np.float64
        ).reshape(1, -1)
        weight_provenance = str(store.manifest.get("weight_provenance"))

        args.output_dir.mkdir(parents=True)
        rows_path = args.output_dir / "error_sources.jsonl"
        rows: list[dict[str, Any]] = []
        trajectory_records: list[dict[str, Any]] = []
        policy_records: dict[str, Any] = {}

        for trajectory_index, key in enumerate(keys):
            states = store.states(key)
            final_index = args.start_frame + args.num_steps * step_stride
            if final_index >= states.shape[0]:
                raise ValueError(
                    f"trajectory {key} has {states.shape[0]} frames; "
                    f"target index {final_index} is unavailable"
                )
            sample = store.tensor_sample(
                key, args.start_frame, step_stride=step_stride, device=device
            )
            edges = np.asarray(store.array(key, "edges"), dtype=np.int64)
            weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64)
            weights = weights.sum(axis=-1)
            policy, policy_record = build_graph_causal_boundary_policy(
                store,
                key,
                device=device,
                max_source_hops=int(boundary_contract["max_source_hops"]),
                rho_inf=float(boundary_contract["rho_inf"]),
                p_inf=float(boundary_contract["p_inf"]),
            )
            expected_digest = boundary_contract.get("policy_digests", {}).get(key)
            if expected_digest is None:
                raise ValueError(
                    f"trajectory {key} has no checkpoint boundary-policy digest"
                )
            if str(policy_record["policy_digest"]) != str(expected_digest):
                raise ValueError(f"trajectory {key} boundary-policy digest changed")
            policy_records[key] = {
                "policy_digest": policy_record["policy_digest"],
                "expected_policy_digest": expected_digest,
                "matches_checkpoint": (
                    str(policy_record["policy_digest"]) == str(expected_digest)
                ),
            }

            current = _state_tensor(states[args.start_frame], device)
            failure_cause = "completed"
            completed_calls = 0
            for call_index in range(1, args.num_steps + 1):
                current_index = args.start_frame + (call_index - 1) * step_stride
                target_index = current_index + step_stride
                reference_current = _state_tensor(states[current_index], device)
                target = _state_tensor(states[target_index], device)
                with torch.no_grad():
                    rollout_prediction, _, _ = contract_forward_sample(
                        model,
                        sample,
                        current,
                        boundary_policy=policy,
                    )
                    if call_index == 1 and torch.equal(current, reference_current):
                        teacher_prediction = rollout_prediction
                    else:
                        teacher_prediction, _, _ = contract_forward_sample(
                            model,
                            sample,
                            reference_current,
                            boundary_policy=policy,
                        )
                rollout_np = rollout_prediction[0].float().cpu().numpy()
                teacher_np = teacher_prediction[0].float().cpu().numpy()
                target_np = target[0].float().cpu().numpy()
                current_np = current[0].float().cpu().numpy()
                reference_current_np = reference_current[0].float().cpu().numpy()
                rollout_admissibility = raw_admissibility_summary(
                    rollout_np, gamma=model.gamma
                )
                teacher_admissibility = raw_admissibility_summary(
                    teacher_np, gamma=model.gamma
                )
                if not rollout_admissibility["all_finite"]:
                    failure_cause = "nonfinite_rollout_prediction"
                    break
                if not rollout_admissibility["all_admissible"]:
                    failure_cause = "inadmissible_rollout_prediction"
                    break

                total = (
                    np.asarray(rollout_np, dtype=np.float64)
                    - np.asarray(target_np, dtype=np.float64)
                ) / component_scale
                propagated = (
                    np.asarray(rollout_np, dtype=np.float64)
                    - np.asarray(teacher_np, dtype=np.float64)
                ) / component_scale
                fresh = (
                    np.asarray(teacher_np, dtype=np.float64)
                    - np.asarray(target_np, dtype=np.float64)
                ) / component_scale
                input_error = (
                    np.asarray(current_np, dtype=np.float64)
                    - np.asarray(reference_current_np, dtype=np.float64)
                ) / component_scale
                total_highpass = node_highpass_field(total, edges)
                propagated_highpass = node_highpass_field(propagated, edges)
                fresh_highpass = node_highpass_field(fresh, edges)
                input_highpass = node_highpass_field(input_error, edges)
                masks, mask_contract = _reference_masks(
                    reference_current,
                    target,
                    sample,
                    shock_quantile=args.shock_quantile,
                    dilation_hops=args.shock_dilation_hops,
                    gamma=model.gamma,
                )
                regions = {
                    "all_nodes_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=masks["all_nodes"]
                    ),
                    "normal_nodes_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=masks["normal_nodes"]
                    ),
                    "boundary_nodes_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=masks["boundary_nodes"]
                    ),
                    "smooth_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=masks["smooth"]
                    ),
                    "front_support_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=masks["front_support"]
                    ),
                    "smooth_highpass": weighted_decomposition_metrics(
                        total_highpass,
                        propagated_highpass,
                        fresh_highpass,
                        weights,
                        mask=masks["smooth"],
                    ),
                    "front_support_highpass": weighted_decomposition_metrics(
                        total_highpass,
                        propagated_highpass,
                        fresh_highpass,
                        weights,
                        mask=masks["front_support"],
                    ),
                }
                input_normal_norm = _weighted_norm(
                    input_error, weights, masks["normal_nodes"]
                )
                input_smooth_highpass_norm = _weighted_norm(
                    input_highpass, weights, masks["smooth"]
                )
                row = {
                    "schema": ERROR_SOURCE_SCHEMA,
                    "trajectory": key,
                    "trajectory_index": trajectory_index,
                    "call_index": call_index,
                    "current_frame": current_index,
                    "target_frame": target_index,
                    "regions": regions,
                    "input_error": {
                        "normal_full_norm": input_normal_norm,
                        "smooth_highpass_norm": input_smooth_highpass_norm,
                        "normal_full_propagation_gain": _gain(
                            regions["normal_nodes_full"].get("propagated_norm"),
                            input_normal_norm,
                        ),
                        "smooth_highpass_propagation_gain": _gain(
                            regions["smooth_highpass"].get("propagated_norm"),
                            input_smooth_highpass_norm,
                        ),
                    },
                    "mask_contract": mask_contract,
                    "admissibility": {
                        "rollout_prediction": rollout_admissibility,
                        "teacher_prediction": teacher_admissibility,
                    },
                    "claim_boundary": (
                        "exact frozen-map validation identity under checkpoint-native "
                        "causal closure; proxy-weighted bump diagnostic, not physical "
                        "conservation"
                    ),
                }
                rows.append(row)
                _append_jsonl(rows_path, row)
                completed_calls = call_index
                current = rollout_prediction

            trajectory_records.append(
                {
                    "trajectory": key,
                    "requested_calls": args.num_steps,
                    "completed_calls": completed_calls,
                    "completed": completed_calls == args.num_steps,
                    "failure_cause": failure_cause,
                }
            )
            print(
                json.dumps(
                    {
                        "trajectory": key,
                        "completed_calls": completed_calls,
                        "failure_cause": failure_cause,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        identity_residuals = [
            float(row["regions"][region][metric])
            for row in rows
            for region in REGION_NAMES
            if row["regions"][region].get("status") == "available"
            for metric in (
                "relative_reconstruction_residual",
                "relative_energy_identity_residual",
            )
        ]
        call1_shares = [
            float(row["regions"][region]["propagated_magnitude_share"])
            for row in rows
            if int(row["call_index"]) == 1
            for region in ("normal_nodes_full", "smooth_highpass")
            if row["regions"][region].get("propagated_magnitude_share") is not None
        ]
        exact_row_count = len(keys) * args.num_steps
        checks = {
            "validation_population_only": set(keys).issubset(
                {str(value) for value in checkpoint.get("val_keys", [])}
            ),
            "no_test_keys_selected": not bool(
                set(keys) & {str(value) for value in checkpoint.get("test_keys", [])}
            ),
            "exact_row_count": len(rows) == exact_row_count,
            "all_rollouts_completed": all(
                record["completed"] for record in trajectory_records
            ),
            "identity_closure": bool(identity_residuals)
            and max(identity_residuals) <= IDENTITY_RELATIVE_TOLERANCE,
            "call1_zero_propagation": bool(call1_shares)
            and max(call1_shares) <= CALL1_PROPAGATED_SHARE_TOLERANCE,
            "boundary_policy_identity": all(
                record["matches_checkpoint"] for record in policy_records.values()
            ),
            "no_smooth_mask_fallback": all(
                not bool(row["mask_contract"]["smooth_fallback_to_normal_nodes"])
                for row in rows
            ),
        }
        summary = {
            "schema": ERROR_SOURCE_SCHEMA,
            "status": "complete",
            "checkpoint": {
                "path_name": args.checkpoint.name,
                "sha256": sha256_file(args.checkpoint),
                "config_digest": str(checkpoint["config_digest"]),
                "data_manifest_digest": str(checkpoint["data_manifest_digest"]),
                "boundary_mode": str(checkpoint["boundary_mode"]),
                "step_stride": step_stride,
            },
            "source_files": _source_hashes(),
            "evaluation": {
                "device": str(device),
                "trajectory_keys": keys,
                "trajectory_count": len(keys),
                "start_frame": args.start_frame,
                "num_steps": args.num_steps,
                "row_count": len(rows),
                "endpoint_calls": [int(value) for value in args.endpoint_calls],
                "precision": "fp32",
                "weight_provenance": weight_provenance,
                "boundary_policy_records": policy_records,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "decomposition_contract": {
                "identity": (
                    "G(uhat_t)-u_(t+1) = [G(uhat_t)-G(u_t)] + [G(u_t)-u_(t+1)]"
                ),
                "map": "checkpoint-native P_B o G_theta o P_B",
                "component_scaling": "checkpoint state_scale",
                "node_measure": weight_provenance,
                "highpass": "linear self-plus-neighbor graph high-pass",
                "smooth_mask": (
                    "intersection of reference-current and reference-target "
                    "normal-node smooth masks"
                ),
                "maximum_identity_relative_residual": (
                    max(identity_residuals) if identity_residuals else None
                ),
                "maximum_call1_propagated_share": (
                    max(call1_shares) if call1_shares else None
                ),
                "physical_conservation": False,
            },
            "contract_checks": checks,
            "contract_complete": all(checks.values()),
            "trajectories": trajectory_records,
            "endpoints": aggregate_rows(rows, args.endpoint_calls),
            "claim_boundary": {
                "verified": (
                    "exact additive attribution for this checkpoint, validation "
                    "population, evaluator source, boundary map, and horizon"
                ),
                "plausible": (
                    "differences between matched checkpoints may diagnose fresh-defect "
                    "versus propagation shaping"
                ),
                "unsupported": (
                    "physical conservation, causal Jacobian identification, holdout "
                    "or sealed performance, seed robustness, or cross-family transfer"
                ),
            },
        }
        _write_json(args.output_dir / "summary.json", summary)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "contract_complete": summary["contract_complete"],
                    "contract_checks": summary["contract_checks"],
                    "checkpoint_sha256": summary["checkpoint"]["sha256"],
                    "row_count": len(rows),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
