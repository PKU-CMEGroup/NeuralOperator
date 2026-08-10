"""Pure diagnostics for finite inadmissibility in PCNO Euler rollouts.

These helpers do not alter a model state.  They distinguish physical
admissibility from numerical finiteness and keep proxy-weighted diagnostics
explicit because the bump shards do not contain validated control volumes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from utility.time_dependent_no.pcno_euler2d import conservative_admissibility
from utility.time_dependent_no.pcno_rollout import failure_cause

BLOWUP_AMPLITUDE_RATIO = 100.0
BLOWUP_RELATIVE_L2 = 10.0
EXCURSION_THRESHOLDS = (6.0, 10.0)


def _masked_values(value: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    mask = node_mask
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    if mask.shape[-1] == 1 and value.shape[-1] != 1:
        mask = mask.expand_as(value)
    return value[mask > 0.0]


def _finite_minimum(value: torch.Tensor, node_mask: torch.Tensor) -> float | None:
    selected = _masked_values(value, node_mask)
    finite = selected[torch.isfinite(selected)]
    return None if finite.numel() == 0 else float(finite.min().detach().cpu())


def weighted_rms(
    value: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    component_scale: torch.Tensor | None = None,
) -> float | None:
    """Return a proxy-weighted component RMS, or ``None`` when nonfinite."""

    value64 = value.detach().double()
    if not bool(torch.isfinite(value64).all()):
        return None
    weights = node_weights.detach().double() * node_mask.detach().double()
    if component_scale is not None:
        scale = component_scale.detach().double().to(value64.device)
        value64 = value64 / scale
    numerator = (value64.square() * weights).sum()
    denominator = weights.sum() * value64.shape[-1]
    if not bool(torch.isfinite(numerator)) or float(denominator) <= 0.0:
        return None
    return float(torch.sqrt(numerator / denominator).cpu())


def weighted_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> float | None:
    """Return the established proxy-weighted, state-scaled relative L2."""

    prediction64 = prediction.detach().double()
    target64 = target.detach().double()
    if not bool(torch.isfinite(prediction64).all()) or not bool(
        torch.isfinite(target64).all()
    ):
        return None
    weights = node_weights.detach().double() * node_mask.detach().double()
    scale = component_scale.detach().double().to(prediction64.device)
    difference = (prediction64 - target64) / scale
    reference = target64 / scale
    numerator = (difference.square() * weights).sum()
    denominator = (reference.square() * weights).sum()
    if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(torch.sqrt(numerator / denominator).cpu())


def state_diagnostics(
    state: torch.Tensor,
    *,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    state_mean: torch.Tensor,
    state_scale: torch.Tensor,
    reference_max_abs: float,
    gamma: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Measure one conservative state without repairing it.

    The returned Boolean mask marks nodes that fail the same density/internal-
    energy/pressure/finiteness contract used by the maintained rollout code.
    Normalizer excursions are descriptive diagnostics, not proof that a state
    lies outside the unknown training distribution.
    """

    if state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError("state must have shape [B,N,4]")
    if reference_max_abs <= 0.0 or not math.isfinite(reference_max_abs):
        raise ValueError("reference_max_abs must be finite and positive")
    admissibility = conservative_admissibility(state.float(), gamma=gamma)
    active = node_mask[..., 0] > 0.0
    finite_node = admissibility["finite_components"]
    finite_component_mask = torch.isfinite(state)
    density = admissibility["density"]
    internal_energy = admissibility["internal_energy"]
    pressure = admissibility["pressure"]
    invalid = active & (
        ~finite_node
        | ~torch.isfinite(internal_energy)
        | ~torch.isfinite(pressure)
        | (density <= 0.0)
        | (internal_energy <= 0.0)
        | (pressure <= 0.0)
    )
    cause, _ = failure_cause(state, gamma=gamma)
    active_components = active.unsqueeze(-1).expand_as(state)
    active_component_count = int(active_components.sum().item())
    finite_component_count = int(
        (finite_component_mask & active_components).sum().item()
    )
    active_node_count = int(active.sum().item())
    finite_node_count = int((finite_node & active).sum().item())

    finite_state = state.detach()[active_components & torch.isfinite(state)]
    max_abs = (
        None if finite_state.numel() == 0 else float(finite_state.abs().max().cpu())
    )
    centered = (state.detach().float() - state_mean) / state_scale
    finite_centered = centered[active_components & torch.isfinite(centered)]
    maximum_excursion = (
        None
        if finite_centered.numel() == 0
        else float(finite_centered.abs().max().cpu())
    )
    excursion: dict[str, float | None] = {}
    weights = node_weights.detach().double() * node_mask.detach().double()
    weight_denominator = float(weights.sum().cpu()) * state.shape[-1]
    for threshold in EXCURSION_THRESHOLDS:
        beyond = (centered.abs() > threshold) & active_components
        unweighted = (
            None
            if active_component_count == 0
            else float(beyond.sum().cpu()) / active_component_count
        )
        weighted_numerator = float(
            (beyond.to(dtype=torch.float64) * weights).sum().cpu()
        )
        weighted = (
            None
            if weight_denominator <= 0.0
            else weighted_numerator / weight_denominator
        )
        label = str(int(threshold))
        excursion[f"normalizer_excursion_fraction_gt_{label}"] = unweighted
        excursion[f"proxy_normalizer_excursion_fraction_gt_{label}"] = weighted

    metrics: dict[str, Any] = {
        "failure_cause": "admissible" if cause is None else cause,
        "admissible": cause is None,
        "active_node_count": active_node_count,
        "finite_component_fraction": (
            None
            if active_component_count == 0
            else finite_component_count / active_component_count
        ),
        "finite_node_fraction": (
            None if active_node_count == 0 else finite_node_count / active_node_count
        ),
        "invalid_node_count": int(invalid.sum().item()),
        "nonpositive_density_node_count": int(
            (active & torch.isfinite(density) & (density <= 0.0)).sum().item()
        ),
        "nonpositive_internal_energy_node_count": int(
            (active & torch.isfinite(internal_energy) & (internal_energy <= 0.0))
            .sum()
            .item()
        ),
        "nonpositive_pressure_node_count": int(
            (active & torch.isfinite(pressure) & (pressure <= 0.0)).sum().item()
        ),
        "min_density": _finite_minimum(density, active),
        "min_internal_energy": _finite_minimum(internal_energy, active),
        "min_pressure": _finite_minimum(pressure, active),
        "max_abs_conservative": max_abs,
        "max_abs_to_reference_max_ratio": (
            None if max_abs is None else max_abs / reference_max_abs
        ),
        "proxy_scaled_relative_l2": weighted_relative_l2(
            state,
            target,
            node_weights,
            node_mask,
            state_scale,
        ),
        "proxy_physical_error_rms": weighted_rms(
            state - target,
            node_weights,
            node_mask,
        ),
        "maximum_normalizer_excursion": maximum_excursion,
        **excursion,
    }
    return metrics, invalid


def stage_transition_label(
    before_failure: str,
    after_failure: str,
    *,
    stage: str,
) -> str:
    """Classify whether one frozen stage introduces, preserves, or repairs failure."""

    before_valid = before_failure == "admissible"
    after_valid = after_failure == "admissible"
    if before_valid and after_valid:
        return f"{stage}_admissible"
    if before_valid and not after_valid:
        return f"{stage}_introduced_inadmissibility"
    if not before_valid and after_valid:
        return f"{stage}_recovery"
    return f"{stage}_persistent_inadmissibility"


def inadmissibility_episodes(flags: Sequence[bool]) -> list[dict[str, Any]]:
    """Return maximal one-indexed call intervals of deployed inadmissibility."""

    values = [bool(value) for value in flags]
    episodes: list[dict[str, Any]] = []
    start: int | None = None
    for offset, invalid in enumerate(values, start=1):
        if invalid and start is None:
            start = offset
        if not invalid and start is not None:
            episodes.append(
                {
                    "start_call": start,
                    "end_call": offset - 1,
                    "duration_calls": offset - start,
                    "recovered": True,
                    "recovery_call": offset,
                }
            )
            start = None
    if start is not None:
        episodes.append(
            {
                "start_call": start,
                "end_call": len(values),
                "duration_calls": len(values) - start + 1,
                "recovered": False,
                "recovery_call": None,
            }
        )
    return episodes


def first_true_call(values: Sequence[bool]) -> int | None:
    for call, value in enumerate(values, start=1):
        if value:
            return call
    return None


def trajectory_event_summary(
    call_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one continuation without equating invalidity and blow-up."""

    if not call_rows:
        raise ValueError("call_rows cannot be empty")
    deployed_invalid = [
        row["deployed_failure_cause"] != "admissible" for row in call_rows
    ]
    nonfinite = [
        row["deployed_failure_cause"] == "nonfinite_state" for row in call_rows
    ]
    amplitude = [
        row["deployed_max_abs_to_reference_max_ratio"] is not None
        and row["deployed_max_abs_to_reference_max_ratio"] >= BLOWUP_AMPLITUDE_RATIO
        for row in call_rows
    ]
    global_error = [
        row["deployed_proxy_scaled_relative_l2"] is not None
        and row["deployed_proxy_scaled_relative_l2"] >= BLOWUP_RELATIVE_L2
        for row in call_rows
    ]
    episodes = inadmissibility_episodes(deployed_invalid)
    first_inadmissible = first_true_call(deployed_invalid)
    first_nonfinite = first_true_call(nonfinite)
    first_amplitude = first_true_call(amplitude)
    first_global_error = first_true_call(global_error)
    blowup_calls = [
        value
        for value in (first_nonfinite, first_amplitude, first_global_error)
        if value is not None
    ]
    first_blowup = min(blowup_calls) if blowup_calls else None
    return {
        "executed_calls": len(call_rows),
        "ever_inadmissible": first_inadmissible is not None,
        "first_inadmissible_call": first_inadmissible,
        "terminal_admissible": not deployed_invalid[-1],
        "inadmissibility_episode_count": len(episodes),
        "recovered_episode_count": sum(bool(row["recovered"]) for row in episodes),
        "longest_inadmissible_episode_calls": max(
            (int(row["duration_calls"]) for row in episodes), default=0
        ),
        "episodes": episodes,
        "first_nonfinite_call": first_nonfinite,
        "first_100x_reference_amplitude_call": first_amplitude,
        "first_global_proxy_l2_at_least_10_call": first_global_error,
        "first_registered_blowup_call": first_blowup,
        "registered_blowup": first_blowup is not None,
        "lag_first_inadmissible_to_blowup": (
            None
            if first_inadmissible is None or first_blowup is None
            else first_blowup - first_inadmissible
        ),
        "input_projection_recovery_calls": sum(
            row["input_projection_transition"] == "input_projection_recovery"
            for row in call_rows
        ),
        "model_recovery_calls": sum(
            row["model_transition"] == "model_recovery" for row in call_rows
        ),
        "output_projection_recovery_calls": sum(
            row["output_projection_transition"] == "output_projection_recovery"
            for row in call_rows
        ),
        "final_proxy_scaled_relative_l2": call_rows[-1][
            "deployed_proxy_scaled_relative_l2"
        ],
        "maximum_proxy_scaled_relative_l2": max(
            (
                float(row["deployed_proxy_scaled_relative_l2"])
                for row in call_rows
                if row["deployed_proxy_scaled_relative_l2"] is not None
            ),
            default=None,
        ),
        "maximum_amplitude_ratio": max(
            (
                float(row["deployed_max_abs_to_reference_max_ratio"])
                for row in call_rows
                if row["deployed_max_abs_to_reference_max_ratio"] is not None
            ),
            default=None,
        ),
    }
