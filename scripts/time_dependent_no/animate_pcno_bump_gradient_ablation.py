#!/usr/bin/env python3
"""Render fixed-scale full/no-gradient bump rollout diagnostics.

This is a visualization-only postprocessor for the W26-L2 bump gradient
ablation.  It consumes the maintained native-rollout trajectory artifacts and
never evaluates a checkpoint.  Accepted recurrent states are shown through the
last valid call; if an arm fails admissibility, its rejected proposal is shown
once at the registered failure call and all later frames remain explicitly
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_gradient_ablation import (
    _load_trajectory_artifact,
    _pressure,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    DIFFERENTIAL_BRANCH_MODES,
    sha256_file,
    write_json,
)

SCHEMA = "w26_l2_bump_gradient_ablation_animation_v1"
DEFAULT_FPS = 5
DEFAULT_DPI = 120


def _configure_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    return plt, animation


def _failure_location(state: np.ndarray, node_type: np.ndarray) -> dict[str, Any]:
    value = np.asarray(state, dtype=np.float64)
    rho = value[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = value[:, 3] - 0.5 * (
            np.square(value[:, 1]) + np.square(value[:, 2])
        ) / rho
        pressure = 0.4 * internal
    nonfinite = np.flatnonzero(~np.isfinite(value).all(axis=1))
    if nonfinite.size:
        index = int(nonfinite[0])
        quantity = "nonfinite_components"
        quantity_value = None
    else:
        candidates = [
            (float(np.min(values)), name, int(np.argmin(values)))
            for name, values in (
                ("density", rho),
                ("internal_energy", internal),
                ("pressure", pressure),
            )
        ]
        quantity_value, quantity, index = min(candidates, key=lambda item: item[0])
    return {
        "node_index": index,
        "node_type": int(np.asarray(node_type).reshape(-1)[index]),
        "quantity": quantity,
        "value": quantity_value,
    }


def _state_for_call(
    arrays: Mapping[str, np.ndarray], call: int
) -> tuple[np.ndarray | None, bool]:
    predictions = np.asarray(
        arrays["pcno_baseline_predictions_conservative"], dtype=np.float64
    )
    valid_length = int(arrays["baseline_valid_length"].item())
    failure_call = int(arrays.get("baseline_failure_call", np.asarray(-1)).item())
    if call <= valid_length:
        return predictions[call - 1], False
    if call == failure_call and "baseline_failed_proposal" in arrays:
        return np.asarray(arrays["baseline_failed_proposal"], dtype=np.float64), True
    return None, False


def _pressure_unchecked(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    rho = value[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kinetic = 0.5 * (
            np.square(value[..., 1]) + np.square(value[..., 2])
        ) / rho
        return 0.4 * (value[..., 3] - kinetic)


def _predicted_increment(
    arrays: Mapping[str, np.ndarray], call: int, state: np.ndarray
) -> np.ndarray:
    if call == 1:
        previous = np.asarray(arrays["initial_conservative"], dtype=np.float64)
    else:
        previous, rejected = _state_for_call(arrays, call - 1)
        if previous is None or rejected:
            raise ValueError("a retained predecessor is required for an increment")
    return _pressure_unchecked(state) - _pressure_unchecked(previous)


def _truth_fields(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(arrays["initial_conservative"], dtype=np.float64)
    targets = np.asarray(arrays["reference_targets_conservative"], dtype=np.float64)
    target_pressure = _pressure(targets)
    previous_pressure = np.concatenate(
        (_pressure(initial)[None], target_pressure[:-1]), axis=0
    )
    return target_pressure, target_pressure - previous_pressure


def _finite_abs_quantile(values: Sequence[np.ndarray], quantile: float) -> float:
    flat = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    )
    finite = np.abs(flat[np.isfinite(flat)])
    if not finite.size:
        return 1.0
    return max(float(np.quantile(finite, quantile)), np.finfo(float).tiny)


def _artifact_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def render_rollout_animation(
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
    *,
    seed: int,
    trajectory: str,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Any]:
    """Render one paired rollout and return its strict provenance payload."""

    if set(artifacts) != set(DIFFERENTIAL_BRANCH_MODES):
        raise ValueError("both full and no_gradient artifacts are required")
    full = artifacts["full"]
    no_gradient = artifacts["no_gradient"]
    for name in (
        "positions",
        "node_type",
        "initial_conservative",
        "reference_targets_conservative",
        "physical_target_times",
    ):
        if name not in full or name not in no_gradient:
            raise ValueError(f"trajectory artifacts are missing {name}")
        if not np.array_equal(full[name], no_gradient[name]):
            raise ValueError(f"paired trajectory artifacts disagree on {name}")

    positions = np.asarray(full["positions"], dtype=np.float64)
    node_type = np.asarray(full["node_type"], dtype=np.int64)
    times = np.asarray(full["physical_target_times"], dtype=np.float64)
    target_pressure, true_increment = _truth_fields(full)
    num_calls = int(target_pressure.shape[0])

    state_pressure: dict[str, dict[int, np.ndarray]] = {}
    increments: dict[str, dict[int, np.ndarray]] = {}
    increment_errors: dict[str, dict[int, np.ndarray]] = {}
    rejected: dict[str, dict[int, bool]] = {}
    failure: dict[str, dict[str, Any] | None] = {}
    for mode in DIFFERENTIAL_BRANCH_MODES:
        state_pressure[mode] = {}
        increments[mode] = {}
        increment_errors[mode] = {}
        rejected[mode] = {}
        arrays = artifacts[mode]
        failure_call = int(arrays.get("baseline_failure_call", np.asarray(-1)).item())
        failure[mode] = None
        for call in range(1, num_calls + 1):
            state, is_rejected = _state_for_call(arrays, call)
            if state is None:
                continue
            state_pressure[mode][call] = _pressure_unchecked(state)
            increment = _predicted_increment(arrays, call, state)
            increments[mode][call] = increment
            increment_errors[mode][call] = increment - true_increment[call - 1]
            rejected[mode][call] = is_rejected
            if is_rejected:
                location = _failure_location(state, node_type)
                failure[mode] = {
                    "call": failure_call,
                    "cause": str(
                        arrays.get("baseline_failure_cause", np.asarray("unknown")).item()
                    ),
                    **location,
                }

    pressure_values = [target_pressure]
    increment_values = [true_increment]
    error_values: list[np.ndarray] = []
    for mode in DIFFERENTIAL_BRANCH_MODES:
        pressure_values.extend(state_pressure[mode].values())
        increment_values.extend(increments[mode].values())
        error_values.extend(increment_errors[mode].values())
    pressure_flat = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in pressure_values]
    )
    pressure_finite = pressure_flat[np.isfinite(pressure_flat)]
    if not pressure_finite.size:
        raise ValueError("pressure visualization has no finite value")
    pressure_min, pressure_max = (
        float(value) for value in np.quantile(pressure_finite, (0.005, 0.995))
    )
    if pressure_min == pressure_max:
        pressure_max = pressure_min + np.finfo(float).eps
    increment_limit = _finite_abs_quantile(increment_values, 0.995)
    error_limit = _finite_abs_quantile(error_values, 0.995)

    plt, animation = _configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), constrained_layout=True)
    panels = (
        ("target pressure", "pressure", None),
        ("full pressure", "pressure", "full"),
        ("no-gradient pressure", "pressure", "no_gradient"),
        ("exact pressure increment", "increment", None),
        ("full increment", "increment", "full"),
        ("full increment error", "error", "full"),
        ("no-gradient increment", "increment", "no_gradient"),
        ("no-gradient increment error", "error", "no_gradient"),
    )
    scatters = []
    annotations = []
    failure_markers = []
    for axis, (title, scale, _) in zip(axes.flat, panels, strict=True):
        if scale == "pressure":
            vmin, vmax, cmap = pressure_min, pressure_max, "viridis"
        elif scale == "increment":
            vmin, vmax, cmap = -increment_limit, increment_limit, "coolwarm"
        else:
            vmin, vmax, cmap = -error_limit, error_limit, "coolwarm"
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=np.zeros(positions.shape[0]),
            s=0.5,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
            rasterized=True,
        )
        annotation = axis.text(
            0.5,
            0.5,
            "",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        (failure_marker,) = axis.plot(
            [],
            [],
            marker="X",
            markersize=8,
            markeredgecolor="black",
            markerfacecolor="#ffd166",
            linestyle="none",
            zorder=5,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        scatters.append(scatter)
        annotations.append(annotation)
        failure_markers.append(failure_marker)

    fig.colorbar(
        scatters[0], ax=[axes[0, 0], axes[0, 1], axes[0, 2]], shrink=0.78,
        label="pressure",
    )
    fig.colorbar(
        scatters[3], ax=[axes[0, 3], axes[1, 0], axes[1, 2]], shrink=0.78,
        label="pressure increment",
    )
    fig.colorbar(
        scatters[5], ax=[axes[1, 1], axes[1, 3]], shrink=0.78,
        label="pressure-increment error",
    )

    def frame_values(kind: str, mode: str | None, call: int) -> np.ndarray | None:
        if kind == "pressure" and mode is None:
            return target_pressure[call - 1]
        if kind == "increment" and mode is None:
            return true_increment[call - 1]
        if kind == "pressure":
            return state_pressure[str(mode)].get(call)
        if kind == "increment":
            return increments[str(mode)].get(call)
        return increment_errors[str(mode)].get(call)

    def update(frame_index: int) -> list[Any]:
        call = frame_index + 1
        artists: list[Any] = []
        for scatter, annotation, failure_marker, (_, kind, mode) in zip(
            scatters, annotations, failure_markers, panels, strict=True
        ):
            values = frame_values(kind, mode, call)
            annotation.set_text("")
            failure_marker.set_visible(False)
            if values is None:
                scatter.set_array(np.full(positions.shape[0], np.nan))
                annotation.set_text("not retained after inadmissibility")
            else:
                scatter.set_array(np.asarray(values, dtype=np.float64))
                if mode is not None and rejected[str(mode)].get(call, False):
                    detail = failure[str(mode)]
                    assert detail is not None
                    value = detail["value"]
                    value_text = "nonfinite" if value is None else f"{value:.3e}"
                    annotation.set_text(
                        "REJECTED PROPOSAL\n"
                        f"{detail['quantity']}={value_text}; type={detail['node_type']}"
                    )
                    node_index = int(detail["node_index"])
                    failure_marker.set_data(
                        [positions[node_index, 0]], [positions[node_index, 1]]
                    )
                    failure_marker.set_visible(True)
            artists.extend((scatter, annotation, failure_marker))
        fig.suptitle(
            f"W26-L2 bump gradient ablation | seed {seed} | trajectory {trajectory} | "
            f"call {call}/{num_calls} | t={times[call - 1]:.3f}"
        )
        return artists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2400,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        raise ValueError("output path must end in .gif or .mp4")
    movie = animation.FuncAnimation(
        fig, update, frames=num_calls, interval=1000 / fps, blit=False
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)

    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "trajectory": str(trajectory),
        "frames": num_calls,
        "fps": int(fps),
        "physical_delta_t": float(np.asarray(full["physical_delta_t"]).item()),
        "physical_horizon": float(times[-1]),
        "scales": {
            "pressure": [pressure_min, pressure_max],
            "pressure_increment": [-increment_limit, increment_limit],
            "pressure_increment_error": [-error_limit, error_limit],
            "pressure_quantiles": [0.005, 0.995],
            "symmetric_absolute_quantile": 0.995,
            "fixed_across_arms_and_calls": True,
        },
        "failure": failure,
        "artifact_payload_sha256": {
            mode: _artifact_digest(artifacts[mode])
            for mode in DIFFERENTIAL_BRANCH_MODES
        },
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": {
            "visualization_only_subsampling": "all retained graph nodes; no temporal subsampling",
            "rejected_proposal_is_not_recurred": True,
            "missing_post_failure_states_are_not_imputed": True,
            "bump_conservation_claim": False,
            "strict_gibbs_claim": False,
        },
    }


def load_paired_artifacts(
    input_root: Path,
    *,
    seed: int,
    trajectory: str,
    checkpoint_sha256: Mapping[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    artifacts = {}
    for mode in DIFFERENTIAL_BRANCH_MODES:
        path = (
            input_root
            / f"seed_{seed}"
            / mode
            / "rollout_open_validation"
            / "trajectories"
            / f"trajectory_{trajectory}.npz"
        )
        artifacts[mode] = _load_trajectory_artifact(
            path,
            trajectory=str(trajectory),
            checkpoint_sha256=checkpoint_sha256[mode],
        )
    return artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--full-checkpoint-sha256", required=True)
    parser.add_argument("--no-gradient-checkpoint-sha256", required=True)
    parser.add_argument("--format", choices=("gif", "mp4"), default="mp4")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = {
        "full": args.full_checkpoint_sha256,
        "no_gradient": args.no_gradient_checkpoint_sha256,
    }
    records = []
    for trajectory in args.trajectories:
        artifacts = load_paired_artifacts(
            args.input_root,
            seed=args.seed,
            trajectory=str(trajectory),
            checkpoint_sha256=checkpoint_sha256,
        )
        output_path = args.output_dir / (
            f"bump_gradient_ablation_seed_{args.seed}_trajectory_{trajectory}."
            f"{args.format}"
        )
        records.append(
            render_rollout_animation(
                artifacts,
                output_path,
                seed=args.seed,
                trajectory=str(trajectory),
                fps=args.fps,
                dpi=args.dpi,
            )
        )
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "records": records,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
