#!/usr/bin/env python3
"""Continue the frozen bump PCNO after its first inadmissible proposal.

This is a deliberately narrow diagnostic for trajectory 05 of D041. It uses
the exact raw conservative recurrence and does not floor, clip, smooth, replace
boundaries, or stop for nonpositive density/internal energy/pressure. The run
is accepted only if calls 1--33 replay the retained D041 prefix within the
declared tolerance.

The resulting sequence is not an admissible Euler rollout. It is evidence
about what the learned numerical map does after its physical validity contract
has already failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    build_model,
    load_checkpoint,
    model_call,
    preprocessing_contract_audit,
    select_device,
    synchronize,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    conservative_to_primitive_raw,
    normalized_node_weights,
)

SCHEMA = "pcno_euler2d_unchecked_post_admissibility_v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964"
)
EXPECTED_D041_TRAJECTORY05_SHA256 = (
    "4d32a1ea9869a4eb565e3eb1769f5864f3247704857140410c5e43ea67fbb5ed"
)
EXPECTED_TRAJECTORY = "05"
EXPECTED_SOURCE_FRAMES = 80
EXPECTED_RECURRENT_CALLS = 79
EXPECTED_FIRST_EXCLUDED_CALL = 33
REPLAY_MAX_ABS_TOLERANCE = 1.0e-5
REPLAY_RELATIVE_L2_TOLERANCE = 1.0e-6
AMPLITUDE_EXPLOSION_RATIO = 100.0
GLOBAL_ERROR_EXPLOSION_THRESHOLD = 10.0
NODE_TYPE_NAMES = {0: "interior", 1: "wall", 2: "outflow", 3: "inflow"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d041-trajectory-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-key", default=EXPECTED_TRAJECTORY)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=EXPECTED_RECURRENT_CALLS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--animation-dpi", type=int, default=105)
    parser.add_argument("--animation-format", choices=("mp4", "gif"), default="mp4")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(str(name))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json_safe(row.get(name)) for name in fields})


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if str(args.trajectory_key) != EXPECTED_TRAJECTORY:
        raise ValueError("this registered diagnostic is restricted to trajectory 05")
    if args.start_frame != 0 or args.num_steps != EXPECTED_RECURRENT_CALLS:
        raise ValueError("this diagnostic requires source frame 0 and exactly 79 calls")
    if args.fps <= 0.0 or args.animation_dpi < 50:
        raise ValueError("fps must be positive and animation dpi at least 50")
    for path in (
        args.checkpoint,
        args.d041_trajectory_artifact,
        args.data_dir,
        args.training_data_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_digest = sha256_file(args.checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint digest {checkpoint_digest} does not match the D041 checkpoint"
        )
    replay_digest = sha256_file(args.d041_trajectory_artifact)
    if replay_digest != EXPECTED_D041_TRAJECTORY05_SHA256:
        raise ValueError(
            f"D041 artifact digest {replay_digest} does not match the retained bundle"
        )


def finite_minimum(value: np.ndarray) -> float | None:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.min(finite))


def finite_maximum_absolute(value: np.ndarray) -> float | None:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.max(np.abs(finite)))


def raw_inadmissible_node_mask(state: np.ndarray, *, gamma: float) -> np.ndarray:
    """Return the exact raw Euler-admissibility failure mask for one frame."""

    conservative = np.asarray(state, dtype=np.float64)
    if conservative.ndim != 2 or conservative.shape[-1] != 4:
        raise ValueError("state must have shape [N,4]")
    primitive = conservative_to_primitive_raw(conservative, gamma=gamma)
    rho = conservative[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = (
            conservative[:, 3]
            - 0.5 * (conservative[:, 1] ** 2 + conservative[:, 2] ** 2) / rho
        )
    pressure = primitive[:, 3]
    finite = (
        np.isfinite(conservative).all(axis=-1)
        & np.isfinite(primitive).all(axis=-1)
        & np.isfinite(internal)
    )
    return ~(finite & (rho > 0.0) & (internal > 0.0) & (pressure > 0.0))


def stable_weighted_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
) -> float | None:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[-1] != 4:
        raise ValueError("prediction and target must align as [N,4]")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        return None
    mass = normalized_node_weights(weights, name="unchecked rollout relative L2")
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, 4)
    error_energy = np.sum(mass[:, None] * ((pred - truth) / scale) ** 2)
    target_energy = np.sum(mass[:, None] * (truth / scale) ** 2)
    if not math.isfinite(float(error_energy)) or target_energy <= 0.0:
        return None
    return float(np.sqrt(error_energy / target_energy))


def raw_frame_metrics(state: np.ndarray, *, gamma: float) -> dict[str, Any]:
    conservative = np.asarray(state, dtype=np.float64)
    if conservative.ndim != 2 or conservative.shape[-1] != 4:
        raise ValueError("state must have shape [N,4]")
    primitive = conservative_to_primitive_raw(conservative, gamma=gamma)
    rho = conservative[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = (
            conservative[:, 3]
            - 0.5 * (conservative[:, 1] ** 2 + conservative[:, 2] ** 2) / rho
        )
    pressure = primitive[:, 3]
    finite_state = np.isfinite(conservative).all(axis=-1)
    finite_primitive = np.isfinite(primitive).all(axis=-1) & np.isfinite(internal)
    admissible = (
        finite_state
        & finite_primitive
        & (rho > 0.0)
        & (internal > 0.0)
        & (pressure > 0.0)
    )
    return {
        "all_conservative_finite": bool(np.all(finite_state)),
        "all_primitive_finite": bool(np.all(finite_primitive)),
        "finite_conservative_fraction": float(np.mean(finite_state)),
        "finite_primitive_fraction": float(np.mean(finite_primitive)),
        "nonfinite_conservative_node_count": int(np.count_nonzero(~finite_state)),
        "nonfinite_primitive_node_count": int(np.count_nonzero(~finite_primitive)),
        "inadmissible_node_count": int(np.count_nonzero(~admissible)),
        "nonpositive_density_node_count": int(np.count_nonzero(rho <= 0.0)),
        "nonpositive_internal_energy_node_count": int(
            np.count_nonzero(internal <= 0.0)
        ),
        "nonpositive_pressure_node_count": int(np.count_nonzero(pressure <= 0.0)),
        "min_density": finite_minimum(rho),
        "min_internal_energy": finite_minimum(internal),
        "min_pressure": finite_minimum(pressure),
        "max_abs_conservative": finite_maximum_absolute(conservative),
        "max_abs_pressure": finite_maximum_absolute(pressure),
    }


def load_d041_replay_prefix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "schema",
            "trajectory_key",
            "checkpoint_sha256",
            "baseline_valid_length",
            "baseline_failure_call",
            "pcno_baseline_predictions_conservative",
            "baseline_failed_proposal",
        }
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"D041 replay artifact is missing fields: {missing}")
        if str(bundle["schema"].item()) != "pcno_euler2d_official_rollout_v1":
            raise ValueError("unsupported D041 trajectory artifact schema")
        if str(bundle["trajectory_key"].item()) != EXPECTED_TRAJECTORY:
            raise ValueError("D041 replay artifact is not trajectory 05")
        if str(bundle["checkpoint_sha256"].item()) != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("D041 replay artifact binds a different checkpoint")
        valid_length = int(bundle["baseline_valid_length"].item())
        failure_call = int(bundle["baseline_failure_call"].item())
        accepted = np.asarray(
            bundle["pcno_baseline_predictions_conservative"], dtype=np.float32
        )
        failed = np.asarray(bundle["baseline_failed_proposal"], dtype=np.float32)
    if valid_length != 32 or failure_call != EXPECTED_FIRST_EXCLUDED_CALL:
        raise ValueError("D041 trajectory 05 must have 32 accepted calls then call 33")
    if accepted.shape[0] != valid_length or failed.shape != accepted.shape[1:]:
        raise ValueError("D041 replay arrays do not align")
    prefix = np.concatenate((accepted, failed[None]), axis=0)
    return prefix, {
        "artifact_sha256": sha256_file(path),
        "accepted_calls": valid_length,
        "first_excluded_call": failure_call,
        "prefix_calls": int(prefix.shape[0]),
    }


@torch.no_grad()
def unchecked_rollout(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    *,
    num_steps: int,
    device: torch.device,
    replay_prefix: np.ndarray | None = None,
    call_fn: (
        Callable[[Any, Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor] | None
    ) = None,
) -> dict[str, Any]:
    """Run every requested raw conservative call, ignoring admissibility only."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    step = model_call if call_fn is None else call_fn
    current = sample["current"]
    predictions: list[np.ndarray] = []
    call_seconds: list[float] = []
    replay_rows: list[dict[str, float | int]] = []
    for call_index in range(1, num_steps + 1):
        synchronize(device)
        started = perf_counter()
        proposal = step(model, sample, current)
        synchronize(device)
        call_seconds.append(perf_counter() - started)
        if proposal.shape != current.shape:
            raise ValueError("model proposal shape changed during unchecked recurrence")
        proposal_np = proposal[0].detach().float().cpu().numpy().copy()
        if replay_prefix is not None and call_index <= replay_prefix.shape[0]:
            expected = np.asarray(replay_prefix[call_index - 1], dtype=np.float64)
            observed = np.asarray(proposal_np, dtype=np.float64)
            delta = observed - expected
            max_abs = float(np.max(np.abs(delta)))
            denominator = max(float(np.linalg.norm(expected)), 1.0e-12)
            relative_l2 = float(np.linalg.norm(delta) / denominator)
            replay_rows.append(
                {
                    "call": call_index,
                    "max_abs": max_abs,
                    "relative_l2": relative_l2,
                }
            )
            if (
                max_abs > REPLAY_MAX_ABS_TOLERANCE
                or relative_l2 > REPLAY_RELATIVE_L2_TOLERANCE
            ):
                raise ValueError(
                    "D041 replay mismatch at call "
                    f"{call_index}: max_abs={max_abs:.3e}, "
                    f"relative_l2={relative_l2:.3e}"
                )
        predictions.append(proposal_np)
        current = proposal
    return {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "call_seconds": call_seconds,
        "replay_rows": replay_rows,
    }


def first_call(rows: Sequence[Mapping[str, Any]], field: str) -> int | None:
    for row in rows:
        if int(row[field]) > 0:
            return int(row["call"])
    return None


def build_call_rows(
    predictions: np.ndarray,
    references: np.ndarray,
    initial: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
    *,
    gamma: float,
    dt: float,
    node_index: int,
    call_seconds: Sequence[float],
) -> list[dict[str, Any]]:
    frames = np.concatenate((initial[None], predictions), axis=0)
    if frames.shape != references.shape:
        raise ValueError("prediction and reference frame stacks do not align")
    reference_max = finite_maximum_absolute(references)
    if reference_max is None or reference_max <= 0.0:
        raise ValueError("reference sequence has no positive finite amplitude")
    rows: list[dict[str, Any]] = []
    for call, (state, truth) in enumerate(zip(frames, references)):
        metrics = raw_frame_metrics(state, gamma=gamma)
        primitive = conservative_to_primitive_raw(state, gamma=gamma)
        rho = float(state[node_index, 0])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            kinetic = float(
                0.5
                * (state[node_index, 1] ** 2 + state[node_index, 2] ** 2)
                / state[node_index, 0]
            )
        internal = float(state[node_index, 3] - kinetic)
        global_error = stable_weighted_relative_l2(
            state, truth, weights, component_scale
        )
        max_abs = metrics["max_abs_conservative"]
        rows.append(
            {
                "call": call,
                "source_frame": call,
                "physical_time": call * dt,
                "model_call_seconds": None if call == 0 else call_seconds[call - 1],
                "scaled_relative_l2_reconstructed_weight_proxy": global_error,
                "max_abs_conservative_to_reference_max_ratio": (
                    None if max_abs is None else float(max_abs / reference_max)
                ),
                "upper_left_node_index": node_index,
                "upper_left_density": rho,
                "upper_left_velocity_x": float(primitive[node_index, 1]),
                "upper_left_velocity_y": float(primitive[node_index, 2]),
                "upper_left_total_energy": float(state[node_index, 3]),
                "upper_left_kinetic_energy": kinetic,
                "upper_left_internal_energy": internal,
                "upper_left_pressure": float(primitive[node_index, 3]),
                **metrics,
            }
        )
    return rows


def percentile_at_or_below(values: np.ndarray, value: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite <= value))


def corner_geometry_summary(
    positions: np.ndarray,
    node_type: np.ndarray,
    node_weights: np.ndarray,
    directed_edges: np.ndarray,
    gradient_weights: np.ndarray,
) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=np.float64)
    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    weights = (
        np.asarray(node_weights, dtype=np.float64)
        .reshape(pos.shape[0], -1)
        .sum(axis=-1)
    )
    edges = np.asarray(directed_edges, dtype=np.int64)
    gradients = np.asarray(gradient_weights, dtype=np.float64)
    upper_left = int(
        np.argmin(
            (pos[:, 0] - np.min(pos[:, 0])) ** 2 + (pos[:, 1] - np.max(pos[:, 1])) ** 2
        )
    )
    target = edges[:, 0]
    degree = np.bincount(target, minlength=pos.shape[0])
    aggregate_gradient = np.zeros(pos.shape[0], dtype=np.float64)
    np.add.at(aggregate_gradient, target, np.sum(gradients**2, axis=1))
    aggregate_gradient = np.sqrt(aggregate_gradient)
    condition = np.full(pos.shape[0], np.nan, dtype=np.float64)
    for index in range(pos.shape[0]):
        neighbors = edges[target == index, 1]
        if neighbors.size:
            singular = np.linalg.svd(pos[neighbors] - pos[index], compute_uv=False)
            condition[index] = (
                math.inf if singular[-1] <= 0.0 else singular[0] / singular[-1]
            )
    selected = target == upper_left
    neighbors = []
    for source, gradient in zip(edges[selected, 1], gradients[selected]):
        neighbors.append(
            {
                "index": int(source),
                "position": pos[source],
                "node_type": int(types[source]),
                "node_type_name": NODE_TYPE_NAMES[int(types[source])],
                "distance": float(np.linalg.norm(pos[source] - pos[upper_left])),
                "gradient_weight": gradient,
                "gradient_weight_norm": float(np.linalg.norm(gradient)),
            }
        )
    type_mask = types == types[upper_left]
    return {
        "index": upper_left,
        "position": pos[upper_left],
        "node_type": int(types[upper_left]),
        "node_type_name": NODE_TYPE_NAMES[int(types[upper_left])],
        "directed_stencil_degree": int(degree[upper_left]),
        "least_squares_stencil_condition": float(condition[upper_left]),
        "condition_percentile_all_nodes": percentile_at_or_below(
            condition, condition[upper_left]
        ),
        "condition_percentile_same_node_type": percentile_at_or_below(
            condition[type_mask], condition[upper_left]
        ),
        "aggregate_gradient_weight_norm": float(aggregate_gradient[upper_left]),
        "gradient_norm_percentile_all_nodes": percentile_at_or_below(
            aggregate_gradient, aggregate_gradient[upper_left]
        ),
        "gradient_norm_percentile_same_node_type": percentile_at_or_below(
            aggregate_gradient[type_mask], aggregate_gradient[upper_left]
        ),
        "reconstructed_weight_proxy": float(weights[upper_left]),
        "weight_percentile_all_nodes": percentile_at_or_below(
            weights, weights[upper_left]
        ),
        "neighbors": neighbors,
    }


def save_figure(figure: Any, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def make_visualizations(
    output_dir: Path,
    predictions: np.ndarray,
    references: np.ndarray,
    initial: np.ndarray,
    positions: np.ndarray,
    node_type: np.ndarray,
    directed_edges: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float,
    fps: float,
    animation_dpi: int,
    animation_format: str,
    upper_left_index: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.facecolor": "white",
        }
    )
    predicted_frames = np.concatenate((initial[None], predictions), axis=0)
    predicted_primitive = conservative_to_primitive_raw(predicted_frames, gamma=gamma)
    reference_primitive = conservative_to_primitive_raw(references, gamma=gamma)
    predicted_pressure = predicted_primitive[..., 3]
    reference_pressure = reference_primitive[..., 3]
    predicted_inadmissible = np.stack(
        [raw_inadmissible_node_mask(frame, gamma=gamma) for frame in predicted_frames]
    )
    pressure_error = predicted_pressure - reference_pressure
    finite_reference = reference_pressure[np.isfinite(reference_pressure)]
    pressure_limits = tuple(np.quantile(finite_reference, (0.002, 0.998)).tolist())
    reference_scale = max(float(np.median(np.abs(finite_reference))), 1.0e-12)
    with np.errstate(invalid="ignore", over="ignore"):
        signed_log_error = np.sign(pressure_error) * np.log10(
            1.0 + np.abs(pressure_error) / reference_scale
        )
    finite_log_error = np.abs(signed_log_error[np.isfinite(signed_log_error)])
    error_limit = max(float(np.quantile(finite_log_error, 0.999)), 1.0)
    calls = np.arange(predicted_frames.shape[0])
    global_error = np.asarray(
        [
            (
                np.nan
                if row["scaled_relative_l2_reconstructed_weight_proxy"] is None
                else float(row["scaled_relative_l2_reconstructed_weight_proxy"])
            )
            for row in rows
        ]
    )
    amplitude_ratio = np.asarray(
        [
            (
                np.nan
                if row["max_abs_conservative_to_reference_max_ratio"] is None
                else float(row["max_abs_conservative_to_reference_max_ratio"])
            )
            for row in rows
        ]
    )
    first_invalid = first_call(rows, "inadmissible_node_count")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#cc00cc")
    cmap.set_over("#111111")
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.1), constrained_layout=False)
    ax_reference, ax_prediction, ax_error, ax_trace = axes.flat
    reference_scatter = ax_reference.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.ma.masked_invalid(reference_pressure[0]),
        s=0.9,
        cmap=cmap,
        vmin=pressure_limits[0],
        vmax=pressure_limits[1],
        linewidths=0,
    )
    prediction_scatter = ax_prediction.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.ma.masked_invalid(predicted_pressure[0]),
        s=0.9,
        cmap=cmap,
        vmin=pressure_limits[0],
        vmax=pressure_limits[1],
        linewidths=0,
    )
    error_scatter = ax_error.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.ma.masked_invalid(signed_log_error[0]),
        s=0.9,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
        linewidths=0,
    )
    failure_marker = ax_prediction.scatter(
        [], [], s=95, facecolors="none", edgecolors="#d62728", linewidths=1.8
    )
    for axis in (ax_reference, ax_prediction, ax_error):
        axis.set_aspect("equal")
        axis.set_xlim(float(np.min(positions[:, 0])), float(np.max(positions[:, 0])))
        axis.set_ylim(float(np.min(positions[:, 1])), float(np.max(positions[:, 1])))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    ax_reference.set_title("Reference pressure")
    ax_prediction.set_title("Unchecked predicted pressure")
    ax_error.set_title("Signed log pressure error")
    fig.colorbar(
        reference_scatter,
        ax=[ax_reference, ax_prediction],
        shrink=0.72,
        label="p (reference scale; magenta/black are under/over)",
    )
    fig.colorbar(error_scatter, ax=ax_error, shrink=0.72, label="signed log10 error")

    positive_error = np.where(global_error > 0.0, global_error, np.nan)
    ax_trace.semilogy(calls, positive_error, color="#1f77b4", label="global proxy L2")
    ax_trace.semilogy(
        calls, amplitude_ratio, color="#ff7f0e", label="max amplitude ratio"
    )
    ax_trace.axhline(GLOBAL_ERROR_EXPLOSION_THRESHOLD, color="#1f77b4", linestyle=":")
    ax_trace.axhline(AMPLITUDE_EXPLOSION_RATIO, color="#ff7f0e", linestyle=":")
    if first_invalid is not None:
        ax_trace.axvline(first_invalid, color="#d62728", linestyle="--", linewidth=1.0)
    current_line = ax_trace.axvline(0, color="#111111", linestyle="--")
    ax_trace.set_xlabel("recurrent call (0 is the source state)")
    ax_trace.set_ylabel("diagnostic magnitude (log scale)")
    ax_trace.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(
        left=0.07, right=0.93, bottom=0.08, top=0.84, wspace=0.34, hspace=0.34
    )

    def update(call: int) -> list[Any]:
        reference_scatter.set_array(np.ma.masked_invalid(reference_pressure[call]))
        prediction_scatter.set_array(np.ma.masked_invalid(predicted_pressure[call]))
        error_scatter.set_array(np.ma.masked_invalid(signed_log_error[call]))
        current_line.set_xdata([call, call])
        row = rows[call]
        failure_marker.set_offsets(positions[predicted_inadmissible[call]])
        fig.suptitle(
            f"trajectory 05: encoded state {call + 1}/80; recurrent call {call}/79; "
            f"t={float(row['physical_time']):.3f}\n"
            f"min rho={row['min_density']!s}, min e={row['min_internal_energy']!s}, "
            f"min p={row['min_pressure']!s}; invalid nodes={row['inadmissible_node_count']}",
            fontsize=9.5,
        )
        return [
            reference_scatter,
            prediction_scatter,
            error_scatter,
            failure_marker,
            current_line,
        ]

    update(0)
    animation_path = output_dir / f"trajectory05_unchecked_all80.{animation_format}"
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=predicted_frames.shape[0],
        interval=1000.0 / fps,
        blit=False,
    )
    if animation_format == "mp4":
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            bitrate=2500,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    elif animation_format == "gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"unsupported animation format: {animation_format}")
    movie.save(animation_path, writer=writer, dpi=animation_dpi)
    plt.close(fig)

    diagnostic_fig, diagnostic_axes = plt.subplots(
        2, 2, figsize=(10.2, 6.0), constrained_layout=True
    )
    ax_margin, ax_counts, ax_growth, ax_finite = diagnostic_axes.flat
    for name, color, label in (
        ("min_density", "#1f77b4", "min density"),
        ("min_internal_energy", "#ff7f0e", "min internal energy"),
        ("min_pressure", "#d62728", "min pressure"),
    ):
        ax_margin.plot(calls, [row[name] for row in rows], color=color, label=label)
    ax_margin.axhline(0.0, color="#111111", linewidth=0.8)
    ax_margin.set_yscale("symlog", linthresh=1.0e-2)
    ax_margin.set_xlabel("recurrent call")
    ax_margin.set_ylabel("raw local margin (symlog)")
    ax_margin.legend(frameon=False)
    for name, color, label in (
        ("nonpositive_density_node_count", "#1f77b4", "rho <= 0"),
        ("nonpositive_internal_energy_node_count", "#ff7f0e", "internal <= 0"),
        ("nonfinite_conservative_node_count", "#9467bd", "nonfinite state"),
    ):
        ax_counts.plot(calls, [row[name] for row in rows], color=color, label=label)
    ax_counts.set_xlabel("recurrent call")
    ax_counts.set_ylabel("node count")
    ax_counts.legend(frameon=False)
    ax_growth.semilogy(calls, positive_error, color="#1f77b4", label="global proxy L2")
    ax_growth.semilogy(
        calls, amplitude_ratio, color="#ff7f0e", label="max amplitude ratio"
    )
    ax_growth.axhline(GLOBAL_ERROR_EXPLOSION_THRESHOLD, color="#1f77b4", linestyle=":")
    ax_growth.axhline(AMPLITUDE_EXPLOSION_RATIO, color="#ff7f0e", linestyle=":")
    ax_growth.set_xlabel("recurrent call")
    ax_growth.set_ylabel("growth diagnostic (log)")
    ax_growth.legend(frameon=False)
    ax_finite.plot(
        calls,
        [row["finite_conservative_fraction"] for row in rows],
        color="#2ca02c",
        label="finite conservative fraction",
    )
    ax_finite.plot(
        calls,
        [row["finite_primitive_fraction"] for row in rows],
        color="#9467bd",
        label="finite primitive fraction",
    )
    ax_finite.set_ylim(-0.02, 1.02)
    ax_finite.set_xlabel("recurrent call")
    ax_finite.set_ylabel("node fraction")
    ax_finite.legend(frameon=False)
    if first_invalid is not None:
        for axis in diagnostic_axes.flat:
            axis.axvline(first_invalid, color="#d62728", linestyle="--", linewidth=0.9)
    diagnostic_paths = save_figure(
        diagnostic_fig, output_dir / "trajectory05_unchecked_diagnostics"
    )
    plt.close(diagnostic_fig)

    selected_calls = [call for call in (0, 32, 33, 40, 60, 79) if call < len(rows)]
    snapshot_fig, snapshot_axes = plt.subplots(
        2, 3, figsize=(12.0, 5.3), constrained_layout=True
    )
    scatter = None
    for axis, call in zip(snapshot_axes.flat, selected_calls):
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=np.ma.masked_invalid(predicted_pressure[call]),
            s=1.0,
            cmap=cmap,
            vmin=pressure_limits[0],
            vmax=pressure_limits[1],
            linewidths=0,
            rasterized=True,
        )
        axis.set_aspect("equal")
        axis.set_title(
            f"call {call}: min p={rows[call]['min_pressure']!s}\n"
            f"invalid={rows[call]['inadmissible_node_count']}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if scatter is None:
        raise ValueError("no snapshot call is available")
    snapshot_fig.colorbar(
        scatter,
        ax=list(snapshot_axes.flat),
        shrink=0.76,
        label="predicted pressure (fixed reference scale)",
    )
    snapshot_paths = save_figure(
        snapshot_fig, output_dir / "trajectory05_unchecked_pressure_snapshots"
    )
    plt.close(snapshot_fig)

    lower_left = int(
        np.argmin(
            (positions[:, 0] - np.min(positions[:, 0])) ** 2
            + (positions[:, 1] - np.min(positions[:, 1])) ** 2
        )
    )
    directed = np.asarray(directed_edges)
    outgoing = directed[directed[:, 0] == upper_left_index]
    neighbor_indices = outgoing[:, 1].astype(int)
    corner_fig, corner_axes = plt.subplots(
        2, 2, figsize=(10.4, 6.0), constrained_layout=True
    )
    ax_pressure, ax_energy, ax_increment, ax_geometry = corner_axes.flat
    ax_pressure.plot(
        calls, predicted_pressure[:, upper_left_index], label="upper-left inflow"
    )
    ax_pressure.plot(
        calls, predicted_pressure[:, lower_left], label="lower-left inflow"
    )
    for neighbor in neighbor_indices:
        ax_pressure.plot(
            calls,
            predicted_pressure[:, neighbor],
            linewidth=0.9,
            alpha=0.7,
            label=f"neighbor {neighbor} ({NODE_TYPE_NAMES[int(node_type[neighbor])]})",
        )
    ax_pressure.axhline(0.0, color="#111111", linewidth=0.8)
    ax_pressure.set_xlabel("recurrent call")
    ax_pressure.set_ylabel("raw pressure")
    ax_pressure.legend(frameon=False, fontsize=7)
    total = predicted_frames[:, upper_left_index, 3]
    density = predicted_frames[:, upper_left_index, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kinetic = (
            0.5
            * np.sum(predicted_frames[:, upper_left_index, 1:3] ** 2, axis=-1)
            / density
        )
    internal = total - kinetic
    ax_energy.plot(calls, total, label="total energy E")
    ax_energy.plot(calls, kinetic, label="kinetic energy")
    ax_energy.plot(calls, internal, label="internal E - K")
    ax_energy.axhline(0.0, color="#111111", linewidth=0.8)
    ax_energy.set_xlabel("recurrent call")
    ax_energy.set_ylabel("upper-left energy density")
    ax_energy.legend(frameon=False)
    ax_increment.plot(calls[1:], np.diff(total), label="delta total")
    ax_increment.plot(calls[1:], np.diff(kinetic), label="delta kinetic")
    ax_increment.plot(calls[1:], np.diff(internal), label="delta internal")
    ax_increment.axhline(0.0, color="#111111", linewidth=0.8)
    ax_increment.set_xlabel("recurrent call")
    ax_increment.set_ylabel("per-call increment")
    ax_increment.legend(frameon=False)
    local = (positions[:, 0] <= positions[upper_left_index, 0] + 0.08) & (
        positions[:, 1] >= positions[upper_left_index, 1] - 0.08
    )
    for code, color in ((0, "#7f7f7f"), (1, "#2ca02c"), (2, "#ff7f0e"), (3, "#1f77b4")):
        mask = local & (node_type == code)
        ax_geometry.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=18,
            color=color,
            label=NODE_TYPE_NAMES[code],
        )
    for neighbor in neighbor_indices:
        ax_geometry.plot(
            [positions[upper_left_index, 0], positions[neighbor, 0]],
            [positions[upper_left_index, 1], positions[neighbor, 1]],
            color="#d62728",
            linewidth=1.2,
        )
    ax_geometry.scatter(
        [positions[upper_left_index, 0]],
        [positions[upper_left_index, 1]],
        s=110,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2.0,
    )
    ax_geometry.set_aspect("equal")
    ax_geometry.set_xlabel("x")
    ax_geometry.set_ylabel("y")
    ax_geometry.set_title("Upper-left one-hot label and PCNO stencil")
    ax_geometry.legend(frameon=False, fontsize=7)
    corner_paths = save_figure(
        corner_fig, output_dir / "trajectory05_upper_left_failure_mechanism"
    )
    plt.close(corner_fig)
    return [animation_path, *diagnostic_paths, *snapshot_paths, *corner_paths]


def runtime_identity(device: torch.device) -> dict[str, Any]:
    source_paths = [
        Path(__file__),
        ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
        ROOT / "utility/time_dependent_no/pcno_euler2d.py",
        ROOT / "utility/time_dependent_no/pcno_ripple_diagnostics.py",
        ROOT / "pcno/pcno.py",
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        status = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "git_commit": commit,
        "git_status_short": status,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    preprocessing = preprocessing_contract_audit(checkpoint, training_store, test_store)
    if int(checkpoint["step_stride"]) != 1:
        raise ValueError("the D041 checkpoint must use model step stride 1")
    key = str(args.trajectory_key)
    if key not in test_store.keys:
        raise KeyError(f"test shard is missing trajectory {key}")
    states = np.asarray(test_store.states(key), dtype=np.float32)
    if states.shape[0] != EXPECTED_SOURCE_FRAMES:
        raise ValueError("trajectory 05 must contain exactly 80 source states")
    target_indices = args.start_frame + np.arange(args.num_steps + 1)
    references = np.array(states[target_indices], copy=True)
    initial = references[0].copy()
    positions = np.array(test_store.array(key, "nodes"), copy=True)
    edges = np.array(test_store.array(key, "edges"), copy=True)
    node_type = np.array(test_store.array(key, "node_type"), copy=True).reshape(-1)
    node_weights = np.array(test_store.array(key, "node_weights"), copy=True)
    directed_edges = np.array(test_store.array(key, "directed_edges"), copy=True)
    gradient_weights = np.array(
        test_store.array(key, "edge_gradient_weights"), copy=True
    )
    geometry = corner_geometry_summary(
        positions, node_type, node_weights, directed_edges, gradient_weights
    )
    if geometry["index"] != 6 or geometry["node_type"] != 3:
        raise ValueError("trajectory 05 upper-left corner contract has changed")
    replay_prefix, replay_metadata = load_d041_replay_prefix(
        args.d041_trajectory_artifact
    )
    model = build_model(checkpoint, device)
    sample = test_store.tensor_sample(
        key, args.start_frame, step_stride=1, device=device
    )
    rollout = unchecked_rollout(
        model,
        sample,
        num_steps=args.num_steps,
        device=device,
        replay_prefix=replay_prefix,
    )
    predictions = rollout["predictions"]
    if predictions.shape[0] != EXPECTED_RECURRENT_CALLS:
        raise ValueError("unchecked recurrence did not produce all 79 calls")
    component_scale = np.asarray(
        checkpoint["normalization"]["state_scale"], dtype=np.float64
    )
    dt = float(test_store.manifest["dt"])
    rows = build_call_rows(
        predictions,
        references,
        initial,
        node_weights,
        component_scale,
        gamma=float(checkpoint["normalization"]["gamma"]),
        dt=dt,
        node_index=int(geometry["index"]),
        call_seconds=rollout["call_seconds"],
    )
    first_inadmissible = first_call(rows, "inadmissible_node_count")
    if first_inadmissible != EXPECTED_FIRST_EXCLUDED_CALL:
        raise ValueError(
            f"unchecked replay first becomes inadmissible at {first_inadmissible}, not 33"
        )
    first_nonfinite = first_call(rows, "nonfinite_conservative_node_count")
    first_negative_density = first_call(rows, "nonpositive_density_node_count")
    first_negative_internal = first_call(rows, "nonpositive_internal_energy_node_count")
    first_amplitude_explosion = next(
        (
            int(row["call"])
            for row in rows
            if row["max_abs_conservative_to_reference_max_ratio"] is not None
            and float(row["max_abs_conservative_to_reference_max_ratio"])
            >= AMPLITUDE_EXPLOSION_RATIO
        ),
        None,
    )
    first_global_error_explosion = next(
        (
            int(row["call"])
            for row in rows
            if row["scaled_relative_l2_reconstructed_weight_proxy"] is not None
            and float(row["scaled_relative_l2_reconstructed_weight_proxy"])
            >= GLOBAL_ERROR_EXPLOSION_THRESHOLD
        ),
        None,
    )

    args.output_dir.mkdir(parents=True)
    trajectory_path = args.output_dir / "trajectory05_unchecked_raw.npz"
    np.savez_compressed(
        trajectory_path,
        schema=np.asarray(SCHEMA),
        trajectory_key=np.asarray(key),
        initial_conservative=initial,
        reference_states_conservative=references,
        unchecked_predictions_conservative=predictions,
        positions=positions,
        edges=edges,
        node_type=node_type,
        reconstructed_node_weights_proxy=node_weights,
        physical_times=np.arange(EXPECTED_SOURCE_FRAMES, dtype=np.float64) * dt,
        checkpoint_sha256=np.asarray(EXPECTED_CHECKPOINT_SHA256),
        source_state_frames=np.asarray(EXPECTED_SOURCE_FRAMES, dtype=np.int64),
        recurrent_calls=np.asarray(EXPECTED_RECURRENT_CALLS, dtype=np.int64),
        encoded_animation_frames=np.asarray(EXPECTED_SOURCE_FRAMES, dtype=np.int64),
    )
    call_metrics_path = args.output_dir / "call_metrics.csv"
    replay_path = args.output_dir / "d041_prefix_replay.csv"
    write_csv(call_metrics_path, rows)
    write_csv(replay_path, rollout["replay_rows"])
    visual_paths = make_visualizations(
        args.output_dir,
        predictions,
        references,
        initial,
        positions,
        node_type,
        directed_edges,
        rows,
        gamma=float(checkpoint["normalization"]["gamma"]),
        fps=args.fps,
        animation_dpi=args.animation_dpi,
        animation_format=args.animation_format,
        upper_left_index=int(geometry["index"]),
    )
    replay_rows = rollout["replay_rows"]
    replay_metadata.update(
        {
            "max_abs": max(float(row["max_abs"]) for row in replay_rows),
            "max_relative_l2": max(float(row["relative_l2"]) for row in replay_rows),
            "max_abs_tolerance": REPLAY_MAX_ABS_TOLERANCE,
            "relative_l2_tolerance": REPLAY_RELATIVE_L2_TOLERANCE,
            "status": "passed",
        }
    )
    summary = {
        "schema": SCHEMA,
        "scientific_role": (
            "qualitative frozen-checkpoint continuation after physical invalidity; "
            "not an admissible Euler rollout, method comparison, or conservation claim"
        ),
        "trajectory": key,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_config_digest": checkpoint["config_digest"],
        "boundary_mode": "model_all_nodes",
        "raw_conservative_recurrence": True,
        "state_modifications": {
            "positivity_stop": False,
            "floor": False,
            "clipping": False,
            "smoothing": False,
            "boundary_replacement": False,
            "future_reference_boundary": False,
        },
        "frame_contract": {
            "source_state_frames": EXPECTED_SOURCE_FRAMES,
            "requested_physical_horizon_transitions": EXPECTED_RECURRENT_CALLS,
            "model_step_stride": 1,
            "executed_recurrent_calls": int(predictions.shape[0]),
            "encoded_animation_frames": int(predictions.shape[0] + 1),
            "animation_format": args.animation_format,
            "visualization_subsampling": False,
            "includes_initial_state": True,
        },
        "replay_gate": replay_metadata,
        "events": {
            "first_inadmissible_call": first_inadmissible,
            "first_nonpositive_internal_energy_call": first_negative_internal,
            "first_nonpositive_pressure_call": first_call(
                rows, "nonpositive_pressure_node_count"
            ),
            "first_nonpositive_density_call": first_negative_density,
            "first_nonfinite_conservative_call": first_nonfinite,
            "first_100x_reference_amplitude_call": first_amplitude_explosion,
            "first_global_proxy_l2_at_least_10_call": first_global_error_explosion,
        },
        "blowup_contract": {
            "definitive_numerical_blowup": "any nonfinite conservative component",
            "severe_amplitude_explosion": (
                "finite max absolute conservative component reaches 100 times the "
                "maximum absolute reference component"
            ),
            "severe_global_error_explosion": (
                "reconstructed-weight scaled relative L2 reaches 10"
            ),
            "thresholds_are_diagnostic_not_physical": True,
        },
        "final_frame": rows[-1],
        "upper_left_corner": geometry,
        "preprocessing_contract": preprocessing,
        "runtime": runtime_identity(device),
        "artifacts": {},
    }
    output_paths = [trajectory_path, call_metrics_path, replay_path, *visual_paths]
    summary["artifacts"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_paths
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "first_inadmissible_call": first_inadmissible,
                "first_nonpositive_density_call": first_negative_density,
                "first_nonfinite_call": first_nonfinite,
                "first_amplitude_explosion_call": first_amplitude_explosion,
                "first_global_error_explosion_call": first_global_error_explosion,
                "final": json_safe(rows[-1]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
