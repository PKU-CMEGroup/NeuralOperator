#!/usr/bin/env python3
"""Render matched PCNO boundary-recurrence animations on validation trajectories.

The script replays one frozen checkpoint under three inference contracts:

1. its native recurrence;
2. full causal nodal boundary reconstruction ``P_B``; and
3. minimum-change hard projection ``P_B*``.

It selects two explanatory cases from an existing all-validation protocol
summary: the earliest native admissibility failure and the upper-median H79
``P_B*`` improvement among native completers.  Selection therefore precedes
rendering and does not depend on visual appeal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol import (
    SCHEMA as PROTOCOL_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol import (
    _policy_set,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    build_model,
    load_checkpoint,
    sha256_file,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    CAUSAL_BOUNDARY_MODE,
    MINIMUM_CHANGE_BOUNDARY_MODE,
    autocast_context,
    boundary_outflow_normal_mach,
    contract_forward_sample,
    failure_cause,
    optional_region_relative_l2,
    select_device,
    write_json,
)
from utility.time_dependent_no.euler2d import (
    conservative_to_primitive,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    normal_node_mask,
    weighted_scaled_relative_l2,
)

SCHEMA = "pcno_euler2d_boundary_protocol_animation_v1"
ARM_ORDER = ("native", "causal", "minimum_change")
ARM_LABELS = {
    "native": "No hard boundary",
    "causal": r"Full reconstruction $P_B$",
    "minimum_change": r"Minimum-change $P_B^*$",
}
ARM_COLORS = {
    "native": "#4D4D4D",
    "causal": "#D55E00",
    "minimum_change": "#0072B2",
}
CAPTURE_PARITY_RTOL = 2.0e-2
CAPTURE_PARITY_ATOL = 1.0e-5
CAPTURE_PARITY_BASIS = (
    "The retained same-source BF16 one-case preflight and all-30 final run differ "
    "by as much as 0.5585% relatively at call 1, while an independent H79 "
    "animation replays differed by -1.0811% and +2.7840% at H79. The "
    "visualization records a 2% relative plus 1e-5 absolute diagnostic flag, "
    "requires the same completion class and failure cause, and labels the "
    "replay-specific first excluded call."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-protocol-summary-sha256", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=79)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--error-quantile", type=float, default=0.995)
    parser.add_argument("--boundary-rho-inf", type=float, default=1.4)
    parser.add_argument("--boundary-p-inf", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def _natural_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _trajectory_rows(
    summary: Mapping[str, Any], arm: str
) -> dict[str, Mapping[str, Any]]:
    rows = summary.get(arm, {}).get("rollout", {}).get("trajectories", [])
    result = {str(row["trajectory"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{arm} rollout contains duplicate trajectory rows")
    return result


def select_visual_cases(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select one stability-rescue and one representative accuracy case."""

    native = _trajectory_rows(summary, "native")
    minimum = _trajectory_rows(summary, "minimum_change")
    causal = _trajectory_rows(summary, "causal")
    common = set(native) & set(minimum) & set(causal)
    if common != set(native) or common != set(minimum) or common != set(causal):
        raise ValueError("all three arms must expose the same trajectory population")

    failures = [row for row in native.values() if not bool(row["completed"])]
    if not failures:
        raise ValueError("the protocol summary contains no native failure to visualize")
    failure = min(
        failures,
        key=lambda row: (
            int(row["valid_length"]),
            _natural_key(str(row["trajectory"])),
        ),
    )

    completers: list[dict[str, Any]] = []
    horizon = str(int(summary["evaluation"]["rollout_steps"]))
    for key in common:
        native_row = native[key]
        minimum_row = minimum[key]
        if not bool(native_row["completed"]) or not bool(minimum_row["completed"]):
            continue
        native_error = native_row["endpoint_relative_l2"].get(horizon)
        minimum_error = minimum_row["endpoint_relative_l2"].get(horizon)
        if native_error is None or minimum_error is None or float(native_error) <= 0.0:
            continue
        completers.append(
            {
                "trajectory": key,
                "native_horizon_error": float(native_error),
                "minimum_change_horizon_error": float(minimum_error),
                "relative_improvement": 1.0
                - float(minimum_error) / float(native_error),
            }
        )
    if not completers:
        raise ValueError("no common completed H79 cases are available")
    completers.sort(
        key=lambda row: (
            float(row["relative_improvement"]),
            _natural_key(str(row["trajectory"])),
        )
    )
    representative = completers[len(completers) // 2]

    return [
        {
            "role": "stability_rescue",
            "trajectory": str(failure["trajectory"]),
            "selection_rule": (
                "minimum native valid_length; trajectory key breaks ties"
            ),
            "native_valid_length": int(failure["valid_length"]),
            "first_excluded_call": int(failure["valid_length"]) + 1,
            "native_failure_cause": str(failure["failure_cause"]),
        },
        {
            "role": "typical_accuracy",
            **representative,
            "selection_rule": (
                "upper median rank of P_B* relative H79 improvement among "
                "native completers"
            ),
            "completed_population": len(completers),
            "rank_zero_based": len(completers) // 2,
        },
    ]


def animation_frame_calls(
    *,
    rollout_steps: int,
    frame_stride: int,
    first_excluded_calls: Sequence[int] = (),
) -> list[int]:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    calls = set(range(0, rollout_steps + 1, frame_stride))
    calls.add(rollout_steps)
    for excluded in first_excluded_calls:
        if 1 <= int(excluded) <= rollout_steps:
            calls.add(int(excluded))
            calls.add(int(excluded) - 1)
    return sorted(calls)


def _boundary_mask(sample: Mapping[str, torch.Tensor]) -> torch.Tensor:
    normal = normal_node_mask(sample["node_type"], sample["node_mask"])
    return sample["node_mask"] - normal


@torch.no_grad()
def capture_rollout(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    boundary_policy: Mapping[str, Any] | None,
    step_stride: int,
    start_frame: int,
    rollout_steps: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    """Capture only the admissible rollout prefix and its evaluator metrics."""

    states = store.states(key)
    available = (states.shape[0] - 1 - start_frame) // step_stride
    if available < rollout_steps:
        raise ValueError(
            f"trajectory {key} has {available} calls, but {rollout_steps} were requested"
        )
    sample = store.tensor_sample(
        key,
        start_frame,
        step_stride=step_stride,
        device=device,
    )
    current = sample["current"]
    boundary = _boundary_mask(sample)
    captured = [current[0].float().cpu().numpy().copy()]
    all_errors = [0.0]
    boundary_errors = [0.0]
    failure = None
    model.eval()
    for call_index in range(1, rollout_steps + 1):
        with autocast_context(device, amp):
            prediction, _, _ = contract_forward_sample(
                model,
                sample,
                current,
                boundary_policy=boundary_policy,
            )
        failure, _ = failure_cause(prediction, gamma=model.gamma)
        if failure is not None:
            break
        if boundary_policy is not None:
            outflow = boundary_outflow_normal_mach(prediction.float(), boundary_policy)
            if outflow is None or not bool(torch.isfinite(outflow).all()):
                failure = "nonfinite_outflow_normal_mach"
                break
            if float(outflow.min().cpu()) <= 1.0:
                failure = "non_supersonic_outflow"
                break
        target_index = start_frame + call_index * step_stride
        target = torch.as_tensor(
            np.array(states[target_index], copy=True),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        all_error = weighted_scaled_relative_l2(
            prediction.float(),
            target,
            sample["node_weights"],
            sample["node_mask"],
            model.state_scale,
        )
        boundary_error = optional_region_relative_l2(
            prediction.float(),
            target,
            sample,
            boundary,
            model,
        )
        captured.append(prediction[0].float().cpu().numpy().copy())
        all_errors.append(float(all_error.cpu()))
        boundary_errors.append(
            float("nan") if boundary_error is None else float(boundary_error)
        )
        current = prediction

    valid_length = len(captured) - 1
    return {
        "states": np.stack(captured).astype(np.float32, copy=False),
        "all_relative_l2": np.asarray(all_errors, dtype=np.float64),
        "boundary_relative_l2": np.asarray(boundary_errors, dtype=np.float64),
        "valid_length": valid_length,
        "completed": valid_length == rollout_steps,
        "failure_cause": "completed" if valid_length == rollout_steps else failure,
        "first_excluded_call": (
            None if valid_length == rollout_steps else valid_length + 1
        ),
    }


def assert_capture_matches_summary(
    captured: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    rtol: float = CAPTURE_PARITY_RTOL,
    atol: float = CAPTURE_PARITY_ATOL,
) -> dict[str, Any]:
    captured_completed = bool(captured["completed"])
    expected_completed = bool(expected["completed"])
    if captured_completed != expected_completed:
        raise ValueError(
            f"captured completion {captured_completed} differs from protocol "
            f"completion {expected_completed}"
        )
    if str(captured["failure_cause"]) != str(expected["failure_cause"]):
        raise ValueError(
            f"captured failure cause {captured['failure_cause']!r} differs from "
            f"protocol value {expected['failure_cause']!r}"
        )
    comparisons = []
    for raw_call, raw_expected in expected["endpoint_relative_l2"].items():
        call = int(raw_call)
        if call > int(captured["valid_length"]):
            continue
        actual = float(captured["all_relative_l2"][call])
        expected_value = float(raw_expected)
        within_tolerance = bool(
            np.isclose(actual, expected_value, rtol=rtol, atol=atol)
        )
        comparisons.append(
            {
                "call_index": call,
                "captured": actual,
                "summary": expected_value,
                "absolute_difference": abs(actual - expected_value),
                "relative_difference": (
                    None if expected_value == 0.0 else actual / expected_value - 1.0
                ),
                "within_diagnostic_tolerance": within_tolerance,
            }
        )
    return {
        "rtol": rtol,
        "atol": atol,
        "captured_valid_length": int(captured["valid_length"]),
        "summary_valid_length": int(expected["valid_length"]),
        "valid_length_difference": int(captured["valid_length"])
        - int(expected["valid_length"]),
        "completion_match": True,
        "failure_cause_match": True,
        "comparisons": comparisons,
        "all_endpoint_values_within_diagnostic_tolerance": all(
            row["within_diagnostic_tolerance"] for row in comparisons
        ),
        "maximum_absolute_difference": max(
            (row["absolute_difference"] for row in comparisons),
            default=0.0,
        ),
    }


def _validate_render_inputs(
    nodes: np.ndarray,
    node_type: np.ndarray,
    reference_states: np.ndarray,
    rollouts: Mapping[str, Mapping[str, Any]],
    *,
    rollout_steps: int,
) -> None:
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (num_nodes, 2)")
    node_type = np.asarray(node_type).reshape(-1)
    if node_type.shape[0] != nodes.shape[0]:
        raise ValueError("node_type and nodes disagree")
    if reference_states.shape != (rollout_steps + 1, nodes.shape[0], 4):
        raise ValueError("reference states have the wrong shape")
    if set(rollouts) != set(ARM_ORDER):
        raise ValueError(f"rollouts must contain exactly {ARM_ORDER}")
    for arm in ARM_ORDER:
        states = np.asarray(rollouts[arm]["states"])
        valid_length = int(rollouts[arm]["valid_length"])
        if states.shape != (valid_length + 1, nodes.shape[0], 4):
            raise ValueError(f"{arm} captured states have the wrong shape")


def _scatter(
    ax: plt.Axes,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    point_size: float,
    cmap: str,
    vmin: float,
    vmax: float,
):
    artist = ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def render_boundary_protocol_animation(
    *,
    nodes: np.ndarray,
    node_type: np.ndarray,
    reference_states: np.ndarray,
    rollouts: Mapping[str, Mapping[str, Any]],
    trajectory_key: str,
    role: str,
    rollout_steps: int,
    step_stride: int,
    start_frame: int,
    amp: str,
    frame_stride: int,
    fps: int,
    dpi: int,
    error_quantile: float,
    output_path: Path,
    snapshot_path: Path,
    gamma: float = 1.4,
) -> dict[str, Any]:
    """Save a synchronized state/error animation for the three policies."""

    nodes = np.asarray(nodes, dtype=np.float64)
    node_type = np.asarray(node_type).reshape(-1)
    reference_states = np.asarray(reference_states, dtype=np.float32)
    _validate_render_inputs(
        nodes,
        node_type,
        reference_states,
        rollouts,
        rollout_steps=rollout_steps,
    )
    if not 0.0 < error_quantile <= 1.0:
        raise ValueError("error_quantile must lie in (0, 1]")
    if fps < 1 or dpi < 1:
        raise ValueError("fps and dpi must be positive")

    reference_pressure = conservative_to_primitive(
        reference_states,
        gamma=gamma,
    )[..., 3].astype(np.float32)
    arm_pressures = {
        arm: conservative_to_primitive(
            np.asarray(rollouts[arm]["states"]),
            gamma=gamma,
        )[..., 3].astype(np.float32)
        for arm in ARM_ORDER
    }
    field_values = [reference_pressure.reshape(-1)]
    field_values.extend(values.reshape(-1) for values in arm_pressures.values())
    finite_fields = np.concatenate(field_values)
    finite_fields = finite_fields[np.isfinite(finite_fields)]
    if finite_fields.size == 0:
        raise ValueError("no finite pressure values are available")
    field_limits = (float(finite_fields.min()), float(finite_fields.max()))
    if field_limits[0] >= field_limits[1]:
        raise ValueError("pressure color limits are degenerate")

    pressure_errors = []
    for arm in ARM_ORDER:
        valid_length = int(rollouts[arm]["valid_length"])
        pressure_errors.append(
            np.abs(
                arm_pressures[arm][1 : valid_length + 1]
                - reference_pressure[1 : valid_length + 1]
            ).reshape(-1)
        )
    finite_errors = np.concatenate(pressure_errors)
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    error_vmax = (
        float(np.quantile(finite_errors, error_quantile)) if finite_errors.size else 1.0
    )
    if not np.isfinite(error_vmax) or error_vmax <= 0.0:
        error_vmax = 1.0
    error_limits = (0.0, error_vmax)

    first_excluded = [
        int(row["first_excluded_call"])
        for row in rollouts.values()
        if row["first_excluded_call"] is not None
    ]
    frame_calls = animation_frame_calls(
        rollout_steps=rollout_steps,
        frame_stride=frame_stride,
        first_excluded_calls=first_excluded,
    )
    point_size = float(np.clip(45_000.0 / max(nodes.shape[0], 1), 0.4, 4.0))
    boundary_nodes = node_type != 0

    fig = plt.figure(figsize=(16.0, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    top_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    metric_ax = fig.add_subplot(grid[1, 0])
    error_axes = [fig.add_subplot(grid[1, column]) for column in range(1, 4)]

    top_artists = [
        _scatter(
            top_axes[0],
            nodes,
            reference_pressure[0],
            point_size=point_size,
            cmap="viridis",
            vmin=field_limits[0],
            vmax=field_limits[1],
        )
    ]
    for arm, ax in zip(ARM_ORDER, top_axes[1:]):
        top_artists.append(
            _scatter(
                ax,
                nodes,
                arm_pressures[arm][0],
                point_size=point_size,
                cmap="viridis",
                vmin=field_limits[0],
                vmax=field_limits[1],
            )
        )
    error_artists = [
        _scatter(
            ax,
            nodes,
            np.zeros(nodes.shape[0], dtype=np.float32),
            point_size=point_size,
            cmap="magma",
            vmin=error_limits[0],
            vmax=error_limits[1],
        )
        for ax in error_axes
    ]

    for ax in [*top_axes, *error_axes]:
        ax.scatter(
            nodes[boundary_nodes, 0],
            nodes[boundary_nodes, 1],
            s=max(1.8 * point_size, 2.0),
            facecolors="none",
            edgecolors="black",
            linewidths=0.18,
            alpha=0.75,
            rasterized=True,
        )
    fig.colorbar(
        top_artists[0],
        ax=top_axes,
        fraction=0.018,
        pad=0.01,
        label="pressure",
    )
    fig.colorbar(
        error_artists[0],
        ax=error_axes,
        fraction=0.024,
        pad=0.01,
        label="absolute pressure error",
    )

    lines = {}
    failure_markers = {}
    for arm in ARM_ORDER:
        (lines[arm],) = metric_ax.plot(
            [],
            [],
            color=ARM_COLORS[arm],
            linewidth=1.8,
            label=ARM_LABELS[arm],
        )
        (failure_markers[arm],) = metric_ax.plot(
            [],
            [],
            color=ARM_COLORS[arm],
            marker="x",
            markersize=8,
            markeredgewidth=2,
            linestyle="none",
        )
    max_error = max(
        float(np.nanmax(np.asarray(rollouts[arm]["all_relative_l2"])))
        for arm in ARM_ORDER
    )
    metric_ax.set_xlim(0, rollout_steps)
    metric_ax.set_ylim(0.0, max(1.0e-6, 1.08 * max_error))
    metric_ax.set_xlabel("learned call")
    metric_ax.set_ylabel("weighted scaled relative L2")
    metric_ax.set_title("All-node rollout error")
    metric_ax.grid(True, alpha=0.25)
    metric_ax.legend(loc="upper left", fontsize=8)

    stopped_top = {}
    stopped_error = {}
    for arm, ax in zip(ARM_ORDER, top_axes[1:]):
        stopped_top[arm] = ax.text(
            0.5,
            0.5,
            "",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "black", "alpha": 0.78},
            visible=False,
        )
    for arm, ax in zip(ARM_ORDER, error_axes):
        stopped_error[arm] = ax.text(
            0.5,
            0.5,
            "",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "black", "alpha": 0.78},
            visible=False,
        )

    def update(call_index: int):
        top_artists[0].set_array(reference_pressure[call_index])
        top_axes[0].set_title(f"Reference pressure\ncall {call_index}")
        for position, (arm, ax) in enumerate(zip(ARM_ORDER, top_axes[1:]), start=1):
            row = rollouts[arm]
            valid_length = int(row["valid_length"])
            shown_call = min(call_index, valid_length)
            top_artists[position].set_array(arm_pressures[arm][shown_call])
            is_stopped = call_index > valid_length
            stopped_top[arm].set_visible(is_stopped)
            stopped_error[arm].set_visible(is_stopped)
            if is_stopped:
                message = (
                    f"STOPPED before call {row['first_excluded_call']}\n"
                    f"{row['failure_cause']}"
                )
                stopped_top[arm].set_text(
                    f"{message}\nshowing last admissible call {valid_length}"
                )
                stopped_error[arm].set_text(message)
                error_artists[position - 1].set_array(
                    np.full(nodes.shape[0], np.nan, dtype=np.float32)
                )
                ax.set_title(f"{ARM_LABELS[arm]}\nlast admissible call {valid_length}")
                error_axes[position - 1].set_title(
                    f"{ARM_LABELS[arm]} pressure error\nexcluded"
                )
            else:
                all_error = float(row["all_relative_l2"][call_index])
                boundary_error = float(row["boundary_relative_l2"][call_index])
                ax.set_title(
                    f"{ARM_LABELS[arm]}\n"
                    f"all={all_error:.4f}, boundary={boundary_error:.4f}"
                )
                error_artists[position - 1].set_array(
                    np.abs(
                        arm_pressures[arm][call_index] - reference_pressure[call_index]
                    )
                )
                error_axes[position - 1].set_title(
                    f"{ARM_LABELS[arm]} pressure error\ncall {call_index}"
                )

            curve_end = min(call_index, valid_length)
            curve_calls = np.arange(1, curve_end + 1)
            lines[arm].set_data(
                curve_calls,
                np.asarray(row["all_relative_l2"])[1 : curve_end + 1],
            )
            excluded = row["first_excluded_call"]
            if excluded is not None and call_index >= int(excluded):
                failure_markers[arm].set_data(
                    [int(excluded)],
                    [float(row["all_relative_l2"][valid_length])],
                )
            else:
                failure_markers[arm].set_data([], [])

        fig.suptitle(
            f"Boundary protocol comparison | validation trajectory {trajectory_key} "
            f"({role}) | H{rollout_steps} | call {call_index}/{rollout_steps} | "
            f"stride {step_stride} | {amp.upper()}",
            fontsize=13,
        )
        return [
            *top_artists,
            *error_artists,
            *lines.values(),
            *failure_markers.values(),
            *stopped_top.values(),
            *stopped_error.values(),
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    update(rollout_steps)
    fig.savefig(snapshot_path, dpi=dpi)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frame_calls,
        blit=False,
    )
    movie.save(
        output_path,
        writer=animation.PillowWriter(fps=fps),
        dpi=dpi,
    )
    plt.close(fig)

    clipped_fraction = {}
    for arm, values in zip(ARM_ORDER, pressure_errors):
        clipped_fraction[arm] = (
            float(np.mean(values > error_vmax)) if values.size else 0.0
        )
    return {
        "animation_path": str(output_path),
        "animation_sha256": sha256_file(output_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "trajectory": str(trajectory_key),
        "role": role,
        "requested_horizon": int(rollout_steps),
        "step_stride": int(step_stride),
        "start_frame": int(start_frame),
        "encoded_state_count": int(rollout_steps + 1),
        "visual_frame_stride": int(frame_stride),
        "rendered_calls": frame_calls,
        "rendered_frame_count": len(frame_calls),
        "field": "pressure",
        "field_limits": list(field_limits),
        "error_kind": "absolute_pressure",
        "error_quantile": float(error_quantile),
        "error_limits": list(error_limits),
        "error_clipped_fraction": clipped_fraction,
        "node_count": int(nodes.shape[0]),
        "boundary_node_count": int(boundary_nodes.sum()),
        "visual_node_subsampling": False,
        "failure_marker_semantics": (
            "first excluded inadmissible proposal; no error is plotted after failure"
        ),
    }


def _validate_protocol_contract(
    summary: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    start_frame: int,
    rollout_steps: int,
    amp: str,
) -> None:
    if summary.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected boundary protocol summary schema")
    if summary.get("status") != "complete":
        raise ValueError("boundary protocol summary is incomplete")
    if summary.get("selection_population") != "checkpoint_validation_split_only":
        raise ValueError("the protocol summary is not validation-only")
    if bool(summary.get("test_split_opened")):
        raise ValueError("test-opened protocol summaries are not accepted")
    if summary["checkpoint"]["sha256"] != checkpoint_sha256:
        raise ValueError("protocol and checkpoint SHA-256 differ")
    if summary["checkpoint"]["data_manifest_digest"] != manifest_sha256:
        raise ValueError("protocol and data manifest SHA-256 differ")
    evaluation = summary["evaluation"]
    if int(evaluation["start_frame"]) != int(start_frame):
        raise ValueError("start_frame differs from the protocol summary")
    if int(evaluation["rollout_steps"]) != int(rollout_steps):
        raise ValueError("rollout horizon differs from the protocol summary")
    if str(evaluation["amp"]) != str(amp):
        raise ValueError("AMP contract differs from the protocol summary")

    for relative_path, expected_hash in summary["source_sha256"].items():
        source_path = ROOT / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(
                f"current source does not match protocol-bound {relative_path}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    device = select_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")

    protocol_sha256 = sha256_file(args.protocol_summary)
    if protocol_sha256 != args.expected_protocol_summary_sha256:
        raise ValueError("protocol summary SHA-256 mismatch")
    protocol = json.loads(args.protocol_summary.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = load_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if store.manifest_digest != args.expected_manifest_sha256:
            raise ValueError("data manifest SHA-256 mismatch")
        if checkpoint["data_manifest_digest"] != store.manifest_digest:
            raise ValueError("checkpoint and data manifest differ")
        _validate_protocol_contract(
            protocol,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=store.manifest_digest,
            start_frame=args.start_frame,
            rollout_steps=args.rollout_steps,
            amp=args.amp,
        )
        selections = select_visual_cases(protocol)
        selected_keys = [str(row["trajectory"]) for row in selections]
        validation_keys = {str(key) for key in checkpoint["val_keys"]}
        if any(key not in validation_keys for key in selected_keys):
            raise ValueError("selected animation trajectory is outside validation")

        model = build_model(checkpoint, device)
        step_stride = int(checkpoint["step_stride"])
        if step_stride != int(protocol["evaluation"]["step_stride"]):
            raise ValueError("checkpoint stride differs from the protocol summary")
        native_policies, _ = _policy_set(
            str(checkpoint["boundary_mode"]),
            checkpoint,
            store,
            selected_keys,
            device=device,
            rho_inf=args.boundary_rho_inf,
            p_inf=args.boundary_p_inf,
        )
        causal_policies, causal_metadata = _policy_set(
            CAUSAL_BOUNDARY_MODE,
            {},
            store,
            selected_keys,
            device=device,
            rho_inf=args.boundary_rho_inf,
            p_inf=args.boundary_p_inf,
        )
        minimum_policies, minimum_metadata = _policy_set(
            MINIMUM_CHANGE_BOUNDARY_MODE,
            {},
            store,
            selected_keys,
            device=device,
            rho_inf=args.boundary_rho_inf,
            p_inf=args.boundary_p_inf,
        )
        policy_sets = {
            "native": native_policies,
            "causal": causal_policies,
            "minimum_change": minimum_policies,
        }
        expected_rows = {arm: _trajectory_rows(protocol, arm) for arm in ARM_ORDER}

        args.output_dir.mkdir(parents=True)
        outputs = []
        captures = []
        for selection in selections:
            key = str(selection["trajectory"])
            rollouts = {}
            parity = {}
            for arm in ARM_ORDER:
                rollouts[arm] = capture_rollout(
                    model,
                    store,
                    key,
                    boundary_policy=policy_sets[arm].get(key),
                    step_stride=step_stride,
                    start_frame=args.start_frame,
                    rollout_steps=args.rollout_steps,
                    device=device,
                    amp=args.amp,
                )
                parity[arm] = assert_capture_matches_summary(
                    rollouts[arm],
                    expected_rows[arm][key],
                )
            horizon_errors = {
                arm: float(rollouts[arm]["all_relative_l2"][-1]) for arm in ARM_ORDER
            }
            if selection["role"] == "stability_rescue":
                direction_check = {
                    "contract": (
                        "native incomplete; causal and minimum-change complete"
                    ),
                    "passed": (
                        not bool(rollouts["native"]["completed"])
                        and bool(rollouts["causal"]["completed"])
                        and bool(rollouts["minimum_change"]["completed"])
                    ),
                }
            else:
                direction_check = {
                    "contract": (
                        "minimum-change H79 error below native; causal H79 error "
                        "above native"
                    ),
                    "passed": (
                        horizon_errors["minimum_change"]
                        < horizon_errors["native"]
                        < horizon_errors["causal"]
                    ),
                }
            direction_check["captured_horizon_errors"] = horizon_errors
            if not direction_check["passed"]:
                raise ValueError(
                    f"trajectory {key} does not preserve its visual mechanism "
                    "under the fresh BF16 replay"
                )
            reference_indices = (
                args.start_frame
                + np.arange(args.rollout_steps + 1, dtype=np.int64) * step_stride
            )
            reference = np.array(
                store.states(key)[reference_indices],
                dtype=np.float32,
                copy=True,
            )
            nodes = np.array(store.array(key, "nodes"), dtype=np.float32, copy=True)
            node_type = np.array(
                store.array(key, "node_type"),
                dtype=np.int64,
                copy=True,
            ).reshape(-1)
            stem = (
                f"boundary_protocol_{selection['role']}_traj{key}_h{args.rollout_steps}"
            )
            output = render_boundary_protocol_animation(
                nodes=nodes,
                node_type=node_type,
                reference_states=reference,
                rollouts=rollouts,
                trajectory_key=key,
                role=str(selection["role"]),
                rollout_steps=args.rollout_steps,
                step_stride=step_stride,
                start_frame=args.start_frame,
                amp=args.amp,
                frame_stride=args.frame_stride,
                fps=args.fps,
                dpi=args.dpi,
                error_quantile=args.error_quantile,
                output_path=args.output_dir / f"{stem}.gif",
                snapshot_path=args.output_dir / f"{stem}_final.png",
                gamma=model.gamma,
            )
            outputs.append(output)
            captures.append(
                {
                    "selection": selection,
                    "trajectory": key,
                    "parity_to_protocol_summary": parity,
                    "fresh_replay_direction_check": direction_check,
                    "arms": {
                        arm: {
                            "all_relative_l2": rollouts[arm][
                                "all_relative_l2"
                            ].tolist(),
                            "boundary_relative_l2": rollouts[arm][
                                "boundary_relative_l2"
                            ].tolist(),
                            "valid_length": int(rollouts[arm]["valid_length"]),
                            "completed": bool(rollouts[arm]["completed"]),
                            "failure_cause": str(rollouts[arm]["failure_cause"]),
                            "first_excluded_call": rollouts[arm]["first_excluded_call"],
                        }
                        for arm in ARM_ORDER
                    },
                    "causal_policy_digest": causal_metadata[key]["policy_digest"],
                    "minimum_change_policy_digest": minimum_metadata[key][
                        "policy_digest"
                    ],
                }
            )

        source_paths = (
            Path(__file__).resolve(),
            ROOT
            / "scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_protocol.py",
            ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
            ROOT / "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
            ROOT / "utility/time_dependent_no/cpg_mesh_contract.py",
            ROOT / "utility/time_dependent_no/euler2d.py",
            ROOT / "utility/time_dependent_no/pcno_euler2d.py",
        )
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "selection_population": "checkpoint_validation_split_only",
            "test_split_opened": False,
            "checkpoint": {
                "path_name": args.checkpoint.name,
                "sha256": checkpoint_sha256,
                "epoch": int(checkpoint["epoch"]),
                "boundary_mode": str(checkpoint["boundary_mode"]),
                "data_manifest_digest": store.manifest_digest,
            },
            "protocol_summary": {
                "path_name": args.protocol_summary.name,
                "sha256": protocol_sha256,
                "schema": protocol["schema"],
            },
            "evaluation": {
                "start_frame": int(args.start_frame),
                "rollout_steps": int(args.rollout_steps),
                "step_stride": step_stride,
                "device": str(device),
                "amp": args.amp,
                "frame_stride": int(args.frame_stride),
                "fps": int(args.fps),
                "dpi": int(args.dpi),
                "capture_parity_contract": {
                    "rtol": CAPTURE_PARITY_RTOL,
                    "atol": CAPTURE_PARITY_ATOL,
                    "basis": CAPTURE_PARITY_BASIS,
                    "endpoint_values": "diagnostic_flag_not_execution_gate",
                    "completion_class": "exact",
                    "failure_cause_for_incomplete_rollouts": "exact",
                    "first_excluded_call": "replay_specific_and_explicitly_labeled",
                },
            },
            "selections": selections,
            "captures": captures,
            "animations": outputs,
            "source_sha256": {
                path.relative_to(ROOT).as_posix(): sha256_file(path)
                for path in source_paths
            },
            "claim_boundary": {
                "verified": (
                    "two selection-rule-bound validation visualizations of the "
                    "matched frozen-checkpoint recurrences"
                ),
                "not_supported": [
                    "population-average improvement from either single animation",
                    "exact DG boundary replay",
                    "physical conservation",
                    "test or distribution-shift performance",
                    "causal attribution to training",
                ],
            },
        }
        write_json(args.output_dir / "animation_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(args.output_dir),
                    "selections": selections,
                    "animations": [
                        {
                            "trajectory": row["trajectory"],
                            "sha256": row["animation_sha256"],
                        }
                        for row in outputs
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
