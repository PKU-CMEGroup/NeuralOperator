"""Shared bump rollout, boundary, and structure diagnostics for Euler2D PCNOs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.euler2d_metrics import (
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    shock_front_masks,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_causal_boundary_conservative_batch,
    apply_minimum_change_boundary_conservative_batch,
    boundary_band_normal_node_mask,
    conservative_admissibility,
    conservative_to_primitive_torch,
    homogeneous_presentation_batches,
    normal_node_mask,
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    induced_subgraph,
    node_highpass_amplitude,
    normalized_node_weights,
    weighted_relative_l2_numpy,
)
from utility.time_dependent_no.pcno_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    autocast_context,
    build_checkpoint_model,
    forward_sample,
    load_checkpoint_payload,
)

CAUSAL_BOUNDARY_MODE = "causal_nodal_physical"
MINIMUM_CHANGE_BOUNDARY_MODE = "minimum_change_nodal_physical"
NORMAL_CLOSED_PRIMARY_OBJECTIVE = "normal_closed"
RAW_ALL_NODES_PRIMARY_OBJECTIVE = "raw_all_nodes"
LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE = "learned_dofs_closed"
STRICT_PHYSICAL_ROLLOUT_POLICY = "strict_physical"
FINITE_ONLY_ROLLOUT_POLICY = "finite_only"
ROLLOUT_FAILURE_POLICIES = (
    STRICT_PHYSICAL_ROLLOUT_POLICY,
    FINITE_ONLY_ROLLOUT_POLICY,
)

STRUCTURE_ERROR_FIELDS = (
    "smooth_highpass_energy_reconstructed_weight_proxy",
    "front_centroid_distance",
    "shock_thickness_log_error",
    "shock_strength_log_error",
)


def load_bump_checkpoint(path: Path) -> dict[str, Any]:
    """Load a schema-4 bump checkpoint and enforce its recurrence declaration."""

    checkpoint = load_checkpoint_payload(path)
    required = {
        "checkpoint_schema_version",
        "model_state",
        "model_config",
        "normalization",
        "data_manifest_digest",
        "step_stride",
        "config_digest",
        "boundary_mode",
        "raw_recurrence",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    if int(checkpoint["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported PCNO checkpoint schema")
    boundary_mode = str(checkpoint["boundary_mode"])
    if boundary_mode not in {
        "model_all_nodes",
        CAUSAL_BOUNDARY_MODE,
        MINIMUM_CHANGE_BOUNDARY_MODE,
    }:
        raise ValueError(f"unsupported checkpoint boundary mode: {boundary_mode}")
    expected_raw = boundary_mode == "model_all_nodes"
    if checkpoint["raw_recurrence"] is not expected_raw:
        raise ValueError("checkpoint recurrence declaration contradicts boundary mode")
    if boundary_mode in {CAUSAL_BOUNDARY_MODE, MINIMUM_CHANGE_BOUNDARY_MODE}:
        boundary_contract = checkpoint.get("boundary_contract")
        if not isinstance(boundary_contract, Mapping):
            raise ValueError("hard-boundary checkpoint lacks its boundary contract")
        if str(boundary_contract.get("mode")) != boundary_mode:
            raise ValueError("checkpoint boundary contract mode is inconsistent")
    return checkpoint


def build_bump_checkpoint_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> PCNOEuler2DResidual:
    """Build the historical bump evaluator model with physical node types."""

    model, _ = build_checkpoint_model(
        checkpoint,
        device,
        model_node_type_input="physical",
    )
    return model


def boundary_outflow_normal_mach(
    conservative: torch.Tensor,
    policy: Mapping[str, Any],
) -> torch.Tensor | None:
    """Return per-sample minimum outward Mach on causal outflow nodes."""

    outflow_rows = policy["outflow_rows"]
    if outflow_rows.numel() == 0:
        return None
    primitive = conservative_to_primitive_torch(
        conservative, gamma=float(policy["gamma"])
    )
    nodes = policy["target_nodes"][outflow_rows]
    normals = policy["target_normals"][outflow_rows].to(
        dtype=primitive.dtype, device=primitive.device
    )
    values = primitive[:, nodes]
    normal_speed = torch.sum(values[..., 1:3] * normals.unsqueeze(0), dim=-1)
    sound_speed = torch.sqrt(float(policy["gamma"]) * values[..., 3] / values[..., 0])
    return (normal_speed / sound_speed).min(dim=1).values


def resolve_boundary_policy(
    boundary_policies: Mapping[str, Mapping[str, Any]] | None,
    key: str,
) -> Mapping[str, Any] | None:
    if not boundary_policies:
        return None
    if key not in boundary_policies:
        raise KeyError(f"boundary contract lacks trajectory {key}")
    return boundary_policies[key]


def close_boundary(
    state: torch.Tensor,
    policy: Mapping[str, Any] | None,
    *,
    gamma: float,
) -> torch.Tensor:
    if policy is None:
        return state
    kind = str(policy.get("closure_kind", "causal_interior_reconstruction"))
    if kind == "causal_interior_reconstruction":
        return apply_causal_boundary_conservative_batch(state, policy, gamma=gamma)
    if kind == "minimum_change_primitive_projection":
        return apply_minimum_change_boundary_conservative_batch(
            state, policy, gamma=gamma
        )
    raise ValueError(f"unsupported boundary closure kind: {kind}")


def contract_forward_sample(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    *,
    boundary_policy: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_current = close_boundary(current, boundary_policy, gamma=model.gamma)
    raw_prediction = forward_sample(model, sample, model_current)
    prediction = close_boundary(
        raw_prediction,
        boundary_policy,
        gamma=model.gamma,
    )
    return prediction, raw_prediction, model_current


def pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    *,
    loss_node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_loss_mask = (
        sample["node_mask"] if loss_node_mask is None else loss_node_mask
    )
    loss = weighted_scaled_mse(
        prediction,
        target,
        sample["node_weights"],
        resolved_loss_mask,
        model.state_scale,
    )
    relative_l2 = weighted_scaled_relative_l2(
        prediction,
        target,
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    return loss, relative_l2


def primary_training_metrics(
    prediction: torch.Tensor,
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    *,
    boundary_policy: Mapping[str, Any] | None,
    primary_objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the selected clean one-step objective without changing recurrence."""

    if primary_objective == RAW_ALL_NODES_PRIMARY_OBJECTIVE:
        if boundary_policy is None:
            raise ValueError(
                "raw all-node supervision requires a causal boundary policy"
            )
        objective_prediction = raw_prediction
        objective_target = target
        loss_node_mask = sample["node_mask"]
    elif primary_objective == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE:
        if (
            boundary_policy is None
            or str(boundary_policy.get("closure_kind"))
            != "minimum_change_primitive_projection"
        ):
            raise ValueError("learned-DOF supervision requires a minimum-change policy")
        objective_prediction = prediction
        objective_target = close_boundary(target, boundary_policy, gamma=model.gamma)
        loss_node_mask = sample["node_mask"]
    elif primary_objective == NORMAL_CLOSED_PRIMARY_OBJECTIVE:
        objective_prediction = prediction
        objective_target = target
        loss_node_mask = (
            normal_node_mask(sample["node_type"], sample["node_mask"])
            if boundary_policy is not None
            else sample["node_mask"]
        )
    else:
        raise ValueError(f"unsupported primary objective: {primary_objective}")
    return pair_metrics(
        objective_prediction,
        objective_target,
        sample,
        model,
        loss_node_mask=loss_node_mask,
    )


def optional_region_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    node_mask: torch.Tensor,
    model: torch.nn.Module,
) -> float | None:
    """Return a regional metric, or ``None`` when that region is absent."""

    support = (sample["node_weights"].sum(dim=-1, keepdim=True) * node_mask).sum()
    if not bool(support > 0.0):
        return None
    value = weighted_scaled_relative_l2(
        prediction,
        target,
        sample["node_weights"],
        node_mask,
        model.state_scale,
    )
    return float(value.detach().cpu())


def evaluate_pairs(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    step_stride: int,
    batch_size: int,
    device: torch.device,
    amp: str,
    primary_objective: str = NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    boundary_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    relative_errors = []
    all_relative_errors = []
    normal_relative_errors = []
    boundary_relative_errors = []
    raw_all_relative_errors = []
    raw_normal_relative_errors = []
    raw_boundary_relative_errors = []
    boundary_correction_rms = []
    outflow_mach_min = math.inf
    admissible = 0
    batches = homogeneous_presentation_batches(pairs, batch_size=batch_size)
    for key, time_indices in batches:
        sample = store.tensor_batch(
            key,
            time_indices,
            step_stride=step_stride,
            device=device,
        )
        boundary_policy = resolve_boundary_policy(boundary_policies, key)
        with autocast_context(device, amp):
            prediction, raw_prediction, _ = contract_forward_sample(
                model,
                sample,
                sample["current"],
                boundary_policy=boundary_policy,
            )
        batch_normal_mask = normal_node_mask(sample["node_type"], sample["node_mask"])
        batch_boundary_mask = sample["node_mask"] - batch_normal_mask
        if boundary_policy is not None:
            scale = model.state_scale.to(
                dtype=prediction.dtype, device=prediction.device
            )
            correction = ((prediction - raw_prediction) / scale).square()
            denominator = batch_boundary_mask.sum(dim=1).squeeze(-1) * 4.0
            per_sample_rms = torch.sqrt(
                (correction * batch_boundary_mask).sum(dim=(1, 2))
                / denominator.clamp_min(1.0)
            )
            boundary_correction_rms.extend(
                float(value) for value in per_sample_rms.detach().cpu()
            )
            outflow = boundary_outflow_normal_mach(prediction.float(), boundary_policy)
            if outflow is not None:
                outflow_mach_min = min(
                    outflow_mach_min, float(outflow.min().detach().cpu())
                )
        for batch_index in range(prediction.shape[0]):
            sample_slice = {
                name: value[batch_index : batch_index + 1]
                for name, value in sample.items()
            }
            loss, relative_l2 = primary_training_metrics(
                prediction[batch_index : batch_index + 1],
                raw_prediction[batch_index : batch_index + 1],
                sample["target"][batch_index : batch_index + 1],
                sample_slice,
                model,
                boundary_policy=boundary_policy,
                primary_objective=primary_objective,
            )
            losses.append(float(loss.detach().cpu()))
            relative_errors.append(float(relative_l2.detach().cpu()))
            all_relative_l2 = optional_region_relative_l2(
                prediction[batch_index : batch_index + 1],
                sample["target"][batch_index : batch_index + 1],
                sample_slice,
                sample_slice["node_mask"],
                model,
            )
            if all_relative_l2 is None:
                raise RuntimeError("a validation presentation has no valid nodes")
            all_relative_errors.append(all_relative_l2)
            normal_mask = batch_normal_mask[batch_index : batch_index + 1]
            boundary_mask = batch_boundary_mask[batch_index : batch_index + 1]
            normal_relative_l2 = optional_region_relative_l2(
                prediction[batch_index : batch_index + 1],
                sample["target"][batch_index : batch_index + 1],
                sample_slice,
                normal_mask,
                model,
            )
            boundary_relative_l2 = optional_region_relative_l2(
                prediction[batch_index : batch_index + 1],
                sample["target"][batch_index : batch_index + 1],
                sample_slice,
                boundary_mask,
                model,
            )
            if normal_relative_l2 is not None:
                normal_relative_errors.append(normal_relative_l2)
            if boundary_relative_l2 is not None:
                boundary_relative_errors.append(boundary_relative_l2)
            if boundary_policy is not None:
                raw_all_relative_l2 = optional_region_relative_l2(
                    raw_prediction[batch_index : batch_index + 1],
                    sample["target"][batch_index : batch_index + 1],
                    sample_slice,
                    sample_slice["node_mask"],
                    model,
                )
                raw_normal_relative_l2 = optional_region_relative_l2(
                    raw_prediction[batch_index : batch_index + 1],
                    sample["target"][batch_index : batch_index + 1],
                    sample_slice,
                    normal_mask,
                    model,
                )
                raw_boundary_relative_l2 = optional_region_relative_l2(
                    raw_prediction[batch_index : batch_index + 1],
                    sample["target"][batch_index : batch_index + 1],
                    sample_slice,
                    boundary_mask,
                    model,
                )
                if raw_all_relative_l2 is None:
                    raise RuntimeError(
                        "a validation presentation has no valid raw nodes"
                    )
                raw_all_relative_errors.append(raw_all_relative_l2)
                if raw_normal_relative_l2 is not None:
                    raw_normal_relative_errors.append(raw_normal_relative_l2)
                if raw_boundary_relative_l2 is not None:
                    raw_boundary_relative_errors.append(raw_boundary_relative_l2)
        diagnostics = conservative_admissibility(prediction.float(), gamma=model.gamma)
        per_sample_admissible = (
            diagnostics["admissible"].reshape(prediction.shape[0], -1).all(dim=1)
        )
        admissible += int(per_sample_admissible.sum().cpu())
    return {
        "loss": float(np.mean(losses)),
        "relative_l2": float(np.mean(relative_errors)),
        "all_relative_l2": float(np.mean(all_relative_errors)),
        "normal_relative_l2": (
            None
            if not normal_relative_errors
            else float(np.mean(normal_relative_errors))
        ),
        "boundary_relative_l2": (
            None
            if not boundary_relative_errors
            else float(np.mean(boundary_relative_errors))
        ),
        "raw_all_relative_l2": (
            None
            if not raw_all_relative_errors
            else float(np.mean(raw_all_relative_errors))
        ),
        "raw_normal_relative_l2": (
            None
            if not raw_normal_relative_errors
            else float(np.mean(raw_normal_relative_errors))
        ),
        "raw_boundary_relative_l2": (
            None
            if not raw_boundary_relative_errors
            else float(np.mean(raw_boundary_relative_errors))
        ),
        "boundary_correction_rms": (
            None
            if not boundary_correction_rms
            else float(np.mean(boundary_correction_rms))
        ),
        "outflow_normal_mach_min": (
            None if not math.isfinite(outflow_mach_min) else outflow_mach_min
        ),
        "admissible_fraction": admissible / len(pairs),
        "presentations": len(pairs),
    }


def finite_minimum(value: torch.Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.min().cpu())


def failure_cause(
    prediction: torch.Tensor,
    *,
    gamma: float,
) -> tuple[str | None, dict[str, float | None]]:
    diagnostics = conservative_admissibility(prediction.float(), gamma=gamma)
    finite = bool(diagnostics["finite_components"].all())
    density = diagnostics["density"]
    internal_energy = diagnostics["internal_energy"]
    pressure = diagnostics["pressure"]
    summary = {
        "min_density": finite_minimum(density),
        "min_internal_energy": finite_minimum(internal_energy),
        "min_pressure": finite_minimum(pressure),
    }
    if (
        not finite
        or not bool(torch.isfinite(internal_energy).all())
        or not bool(torch.isfinite(pressure).all())
    ):
        return "nonfinite_state", summary
    if not bool((density > 0.0).all()):
        return "nonpositive_density", summary
    if not bool((internal_energy > 0.0).all()):
        return "nonpositive_internal_energy", summary
    if not bool((pressure > 0.0).all()):
        return "nonpositive_pressure", summary
    return None, summary


@torch.no_grad()
def rollout_trajectory(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    step_stride: int,
    start_frame: int,
    num_steps: int,
    device: torch.device,
    amp: str,
    boundary_policy: Mapping[str, Any] | None = None,
    rollout_checkpoints: Sequence[int] = (),
    failure_policy: str = STRICT_PHYSICAL_ROLLOUT_POLICY,
) -> dict[str, Any]:
    if failure_policy not in ROLLOUT_FAILURE_POLICIES:
        raise ValueError(f"unsupported rollout failure policy: {failure_policy}")
    states = store.states(key)
    available = (states.shape[0] - 1 - start_frame) // step_stride
    requested = int(num_steps)
    if available < requested:
        raise ValueError(
            f"trajectory {key} exposes only {available} calls from frame "
            f"{start_frame} at stride {step_stride}; {requested} were requested"
        )
    sample = store.tensor_sample(
        key, start_frame, step_stride=step_stride, device=device
    )
    current = sample["current"]
    errors = []
    normal_errors = []
    boundary_errors = []
    normal_mask = normal_node_mask(sample["node_type"], sample["node_mask"])
    boundary_mask = sample["node_mask"] - normal_mask
    minimums = {
        "min_density": math.inf,
        "min_internal_energy": math.inf,
        "min_pressure": math.inf,
    }
    termination_cause = None
    hard_failure_cause = None
    first_physical_violation: dict[str, Any] | None = None
    first_physical_violation_by_cause: dict[str, int] = {}
    physical_violation_counts: dict[str, int] = {}
    endpoint_errors: dict[str, float] = {}
    endpoint_normal_errors: dict[str, float] = {}
    endpoint_boundary_errors: dict[str, float] = {}
    maximum_boundary_correction_rms = 0.0
    minimum_outflow_normal_mach = math.inf

    def record_physical_violation(cause: str, call_number: int) -> None:
        nonlocal first_physical_violation
        physical_violation_counts[cause] = (
            physical_violation_counts.get(cause, 0) + 1
        )
        first_physical_violation_by_cause.setdefault(cause, call_number)
        if first_physical_violation is None:
            first_physical_violation = {"call": call_number, "cause": cause}

    model.eval()
    for call_index in range(requested):
        call_number = call_index + 1
        with autocast_context(device, amp):
            proposal, raw_proposal, _ = contract_forward_sample(
                model,
                sample,
                current,
                boundary_policy=boundary_policy,
            )
        if boundary_policy is not None:
            scale = model.state_scale.to(dtype=proposal.dtype, device=proposal.device)
            correction = ((proposal - raw_proposal) / scale).square()
            denominator = boundary_mask.sum() * 4.0
            correction_rms = torch.sqrt(
                (correction * boundary_mask).sum() / denominator.clamp_min(1.0)
            )
            maximum_boundary_correction_rms = max(
                maximum_boundary_correction_rms,
                float(correction_rms.detach().cpu()),
            )
        state_failure, current_minimums = failure_cause(proposal, gamma=model.gamma)
        for name, value in current_minimums.items():
            if value is not None:
                minimums[name] = min(minimums[name], value)
        if state_failure == "nonfinite_state":
            hard_failure_cause = state_failure
            termination_cause = state_failure
            break
        if state_failure is not None:
            record_physical_violation(state_failure, call_number)
            if failure_policy == STRICT_PHYSICAL_ROLLOUT_POLICY:
                termination_cause = state_failure
                break
        if boundary_policy is not None:
            outflow = boundary_outflow_normal_mach(proposal.float(), boundary_policy)
            if outflow is None:
                raise ValueError("boundary policy has no outflow support")
            if not bool(torch.isfinite(outflow).all()):
                outflow_failure = "nonfinite_outflow_normal_mach"
            else:
                current_outflow = float(outflow.min().cpu())
                minimum_outflow_normal_mach = min(
                    minimum_outflow_normal_mach, current_outflow
                )
                outflow_failure = (
                    "non_supersonic_outflow" if current_outflow <= 1.0 else None
                )
            if outflow_failure is not None:
                record_physical_violation(outflow_failure, call_number)
                if failure_policy == STRICT_PHYSICAL_ROLLOUT_POLICY:
                    termination_cause = outflow_failure
                    break
        target_index = start_frame + (call_index + 1) * step_stride
        target = torch.as_tensor(
            np.array(states[target_index], copy=True),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        relative_l2 = weighted_scaled_relative_l2(
            proposal.float(),
            target,
            sample["node_weights"],
            sample["node_mask"],
            model.state_scale,
        )
        relative_l2_value = float(relative_l2.cpu())
        normal_relative_l2 = optional_region_relative_l2(
            proposal.float(), target, sample, normal_mask, model
        )
        boundary_relative_l2 = optional_region_relative_l2(
            proposal.float(), target, sample, boundary_mask, model
        )
        regional_errors = (normal_relative_l2, boundary_relative_l2)
        if not math.isfinite(relative_l2_value) or any(
            value is not None and not math.isfinite(value)
            for value in regional_errors
        ):
            hard_failure_cause = "nonfinite_error_metric"
            termination_cause = hard_failure_cause
            break
        errors.append(relative_l2_value)
        if normal_relative_l2 is not None:
            normal_errors.append(normal_relative_l2)
        if boundary_relative_l2 is not None:
            boundary_errors.append(boundary_relative_l2)
        if call_number in rollout_checkpoints:
            endpoint_errors[str(call_number)] = errors[-1]
            if normal_relative_l2 is not None:
                endpoint_normal_errors[str(call_number)] = normal_relative_l2
            if boundary_relative_l2 is not None:
                endpoint_boundary_errors[str(call_number)] = boundary_relative_l2
        current = proposal
    valid_length = len(errors)
    minimum_summary = {
        name: None if not math.isfinite(value) else value
        for name, value in minimums.items()
    }
    return {
        "trajectory": key,
        "requested_steps": requested,
        "valid_length": valid_length,
        "completed": valid_length == requested,
        "failure_cause": (
            "completed" if valid_length == requested else termination_cause
        ),
        "hard_failure_cause": hard_failure_cause,
        "first_physical_violation": first_physical_violation,
        "first_physical_violation_by_cause": first_physical_violation_by_cause,
        "physical_violation_counts": physical_violation_counts,
        "physically_admissible": first_physical_violation is None,
        "rollout_failure_policy": failure_policy,
        "survival_fraction": valid_length / requested,
        "final_relative_l2": errors[-1] if errors else None,
        "mean_prefix_relative_l2": float(np.mean(errors)) if errors else None,
        "endpoint_relative_l2": endpoint_errors,
        "final_normal_relative_l2": normal_errors[-1] if normal_errors else None,
        "mean_prefix_normal_relative_l2": (
            float(np.mean(normal_errors)) if normal_errors else None
        ),
        "endpoint_normal_relative_l2": endpoint_normal_errors,
        "final_boundary_relative_l2": (
            boundary_errors[-1] if boundary_errors else None
        ),
        "mean_prefix_boundary_relative_l2": (
            float(np.mean(boundary_errors)) if boundary_errors else None
        ),
        "endpoint_boundary_relative_l2": endpoint_boundary_errors,
        "maximum_boundary_correction_rms": (
            maximum_boundary_correction_rms if boundary_policy is not None else None
        ),
        "minimum_outflow_normal_mach": (
            None
            if not math.isfinite(minimum_outflow_normal_mach)
            else minimum_outflow_normal_mach
        ),
        **minimum_summary,
    }


@torch.no_grad()
def evaluate_rollouts(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    start_frame: int,
    num_steps: int,
    device: torch.device,
    amp: str,
    boundary_policies: Mapping[str, Mapping[str, Any]] | None = None,
    rollout_checkpoints: Sequence[int] = (),
    parity_keys: Sequence[str] = (),
    parity_horizon: int = 20,
    failure_policy: str = STRICT_PHYSICAL_ROLLOUT_POLICY,
) -> dict[str, Any]:
    rows = [
        rollout_trajectory(
            model,
            store,
            key,
            step_stride=step_stride,
            start_frame=start_frame,
            num_steps=num_steps,
            device=device,
            amp=amp,
            boundary_policy=resolve_boundary_policy(boundary_policies, key),
            rollout_checkpoints=rollout_checkpoints,
            failure_policy=failure_policy,
        )
        for key in keys
    ]
    completed = [row for row in rows if row["completed"]]
    all_completed = len(completed) == len(rows)
    if all_completed:
        selection_values = [float(row["final_relative_l2"]) for row in rows]
        selection_population = "all_trajectories_completed_final"
    else:
        selection_values = [
            float(row["mean_prefix_relative_l2"])
            for row in rows
            if row["mean_prefix_relative_l2"] is not None
        ]
        selection_population = "mixed_valid_prefix"
    mean_error = None if not selection_values else float(np.mean(selection_values))
    full_horizon_values = [
        float(row["mean_prefix_relative_l2"])
        for row in rows
        if row["completed"] and row["mean_prefix_relative_l2"] is not None
    ]
    mean_full_horizon_error = (
        float(np.mean(full_horizon_values))
        if len(full_horizon_values) == len(rows)
        else None
    )

    def mean_available(field: str) -> float | None:
        values = [float(row[field]) for row in rows if row[field] is not None]
        return None if not values else float(np.mean(values))

    def aggregate_endpoints(
        field: str,
    ) -> tuple[dict[str, float | None], dict[str, int]]:
        means: dict[str, float | None] = {}
        counts: dict[str, int] = {}
        for checkpoint in rollout_checkpoints:
            values = [
                row[field][str(checkpoint)]
                for row in rows
                if str(checkpoint) in row[field]
            ]
            counts[str(checkpoint)] = len(values)
            means[str(checkpoint)] = None if not values else float(np.mean(values))
        return means, counts

    endpoint_means, endpoint_counts = aggregate_endpoints("endpoint_relative_l2")
    normal_endpoint_means, normal_endpoint_counts = aggregate_endpoints(
        "endpoint_normal_relative_l2"
    )
    boundary_endpoint_means, boundary_endpoint_counts = aggregate_endpoints(
        "endpoint_boundary_relative_l2"
    )
    parity = None
    if parity_keys:
        row_by_key = {str(row["trajectory"]): row for row in rows}
        missing = sorted({str(key) for key in parity_keys} - set(row_by_key))
        if missing:
            raise ValueError(f"parity cohort is absent from rollout rows: {missing}")
        parity_rows = [row_by_key[str(key)] for key in parity_keys]
        parity_values = [
            float(row["endpoint_relative_l2"][str(parity_horizon)])
            for row in parity_rows
            if str(parity_horizon) in row["endpoint_relative_l2"]
        ]
        parity = {
            "keys": [str(key) for key in parity_keys],
            "horizon": int(parity_horizon),
            "completed": len(parity_values),
            "completion_rate": len(parity_values) / len(parity_rows),
            "mean_relative_l2": (
                None if not parity_values else float(np.mean(parity_values))
            ),
        }
    return {
        "trajectories": rows,
        "num_trajectories": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "mean_survival_fraction": float(
            np.mean([row["survival_fraction"] for row in rows])
        ),
        "mean_selection_relative_l2": mean_error,
        "mean_full_horizon_relative_l2": mean_full_horizon_error,
        "mean_final_relative_l2": mean_available("final_relative_l2"),
        "hard_failure_count": sum(
            row["hard_failure_cause"] is not None for row in rows
        ),
        "physically_admissible_count": sum(
            bool(row["physically_admissible"]) for row in rows
        ),
        "physical_admissibility_rate": float(
            np.mean([bool(row["physically_admissible"]) for row in rows])
        ),
        "mean_final_normal_relative_l2": mean_available("final_normal_relative_l2"),
        "mean_final_boundary_relative_l2": mean_available("final_boundary_relative_l2"),
        "mean_endpoint_relative_l2": endpoint_means,
        "endpoint_population_count": endpoint_counts,
        "mean_endpoint_normal_relative_l2": normal_endpoint_means,
        "normal_endpoint_population_count": normal_endpoint_counts,
        "mean_endpoint_boundary_relative_l2": boundary_endpoint_means,
        "boundary_endpoint_population_count": boundary_endpoint_counts,
        "selection_population": selection_population,
        "rollout_failure_policy": failure_policy,
        "parity": parity,
    }


def _dilate_mask(mask: np.ndarray, edges: np.ndarray, hops: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    edge_index = np.asarray(edges, dtype=np.int64)
    for _ in range(hops):
        expanded = result.copy()
        left = edge_index[:, 0]
        right = edge_index[:, 1]
        np.logical_or.at(expanded, left, result[right])
        np.logical_or.at(expanded, right, result[left])
        result = expanded
    return result


def _scalar(value: Any) -> float | None:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar diagnostic")
    numeric = float(array.reshape(-1)[0])
    return numeric if math.isfinite(numeric) else None


def endpoint_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    proxy_weights: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
) -> dict[str, Any]:
    pred_primitive = conservative_to_primitive_raw(prediction, gamma=gamma)
    target_primitive = conservative_to_primitive_raw(target, gamma=gamma)
    interior = np.asarray(node_type).reshape(-1) == 0
    if np.count_nonzero(interior) < 3:
        interior = np.ones_like(interior)
    fronts = shock_front_masks(
        pred_primitive,
        target_primitive,
        edges,
        quantile=shock_quantile,
        node_mask=interior,
    )
    overlap = front_overlap_metrics(fronts["prediction_mask"], fronts["target_mask"])
    distance = front_distance_metrics(
        fronts["prediction_mask"], fronts["target_mask"], positions
    )
    pred_interior, interior_edges, _ = induced_subgraph(
        pred_primitive, edges, np.ones(prediction.shape[0]), interior
    )
    target_interior, _, _ = induced_subgraph(
        target_primitive, edges, np.ones(prediction.shape[0]), interior
    )
    smearing = shock_smearing_metrics(
        pred_interior, target_interior, interior_edges, scalar_index=3
    )

    target_front = _dilate_mask(fronts["target_mask"], edges, hops=1)
    smooth = interior & ~target_front
    smooth_contract = "target_pressure_front_dilated_one_hop"
    if not np.any(smooth):
        smooth = interior & ~fronts["target_mask"]
        smooth_contract = "target_pressure_front_without_dilation_fallback"
    scaled_error = (prediction - target) / component_scale.reshape(1, -1)
    highpass = node_highpass_amplitude(scaled_error, edges)
    proxy_mass = normalized_node_weights(proxy_weights, name="endpoint high-pass")
    equal_mass = np.full(prediction.shape[0], 1.0 / prediction.shape[0])

    def smooth_energy(mass: np.ndarray) -> float:
        selected = mass * smooth
        return float(np.sum(selected * np.square(highpass)) / np.sum(selected))

    thickness_ratio = _scalar(smearing["thickness_ratio"])
    strength_ratio = _scalar(smearing["strength_ratio"])

    def log_error(value: float | None) -> float | None:
        if value is None or value <= 0.0:
            return None
        return abs(math.log(value))

    error_energy = proxy_mass * np.sum(np.square(scaled_error), axis=-1)
    return {
        "scaled_relative_l2_reconstructed_weight_proxy": weighted_relative_l2_numpy(
            prediction, target, proxy_weights, component_scale
        ),
        "scaled_relative_l2_equal_node_proxy": weighted_relative_l2_numpy(
            prediction, target, np.ones(prediction.shape[0]), component_scale
        ),
        "smooth_region_contract": smooth_contract,
        "smooth_region_node_fraction": float(np.mean(smooth)),
        "smooth_highpass_energy_reconstructed_weight_proxy": smooth_energy(proxy_mass),
        "smooth_highpass_energy_equal_node_proxy": smooth_energy(equal_mass),
        "scaled_error_energy_fraction_target_front_dilated_proxy": float(
            error_energy[target_front].sum() / max(float(error_energy.sum()), 1e-30)
        ),
        "front_iou": _scalar(overlap["iou"]),
        "front_symmetric_chamfer": _scalar(distance["symmetric_chamfer_mean"]),
        "front_centroid_distance": _scalar(
            front_centroid_distance(
                fronts["prediction_mask"], fronts["target_mask"], positions
            )
        ),
        "shock_thickness_ratio": thickness_ratio,
        "shock_strength_ratio": strength_ratio,
        "shock_thickness_log_error": log_error(thickness_ratio),
        "shock_strength_log_error": log_error(strength_ratio),
    }


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


@torch.no_grad()
def rollout_structure_diagnostics(
    model: torch.nn.Module,
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
    model: torch.nn.Module,
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
            evaluated_model: torch.nn.Module = model,
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
        if "wall_velocity_projectors" in policy:
            wall_nodes = policy["wall_constrained_nodes"]
            projectors = policy["wall_velocity_projectors"].to(
                dtype=primitive.dtype, device=device
            )
            wall_velocity = primitive[:, wall_nodes, 1:3]
            forbidden = wall_velocity - torch.einsum(
                "nij,bnj->bni", projectors, wall_velocity
            )
        else:
            wall_rows = policy["wall_rows"]
            wall_nodes = policy["target_nodes"][wall_rows]
            wall_normals = policy["target_normals"][wall_rows].to(
                dtype=primitive.dtype, device=device
            )
            wall_velocity = primitive[:, wall_nodes, 1:3]
            wall_normal_speed = torch.sum(
                wall_velocity * wall_normals.unsqueeze(0), dim=-1, keepdim=True
            )
            forbidden = wall_normal_speed * wall_normals.unsqueeze(0)
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
