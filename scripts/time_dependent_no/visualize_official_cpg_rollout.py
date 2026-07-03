"""Visualize official CPGNet rollout HDF5 outputs for 2D Euler.

The official cpgGNSpdes rollout writer stores one file per test trajectory with
``predicteds`` and ``targets`` arrays. This script pairs those arrays with the
trajectory geometry in ``test.h5`` and saves side-by-side truth, prediction, and
error animations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


FEATURE_NAMES = ("rho", "v1", "v2", "pres")
FEATURE_ALIASES = {
    "density": "rho",
    "u": "v1",
    "x_velocity": "v1",
    "v": "v2",
    "y_velocity": "v2",
    "p": "pres",
    "pressure": "pres",
}
NODE_FILTERS = ("all", "normal", "boundary")


def parse_feature(feature: str | int) -> int:
    if isinstance(feature, str):
        key = FEATURE_ALIASES.get(feature.strip(), feature.strip())
        if key in FEATURE_NAMES:
            return FEATURE_NAMES.index(key)
        try:
            feature = int(key)
        except ValueError as exc:
            raise ValueError(
                f"Unknown feature {feature!r}; use one of {FEATURE_NAMES}, "
                f"aliases {sorted(FEATURE_ALIASES)}, or an integer index."
            ) from exc

    feature_index = int(feature)
    if feature_index < 0 or feature_index >= len(FEATURE_NAMES):
        raise IndexError(
            f"feature index {feature_index} is outside [0, {len(FEATURE_NAMES)})."
        )
    return feature_index


def trajectory_keys(handle: h5py.File) -> list[str]:
    """Return HDF5 trajectory keys in the same order used by the official loader."""

    return list(handle.keys())


def select_trajectory_key(
    handle: h5py.File,
    *,
    trajectory_index: int,
    trajectory_key: str | None = None,
) -> str:
    if trajectory_key is not None:
        if trajectory_key not in handle:
            raise KeyError(f"trajectory key {trajectory_key!r} not found in dataset")
        return trajectory_key

    keys = trajectory_keys(handle)
    if trajectory_index < 0 or trajectory_index >= len(keys):
        raise IndexError(
            f"trajectory_index={trajectory_index} is outside [0, {len(keys)})."
        )
    return keys[trajectory_index]


def infer_result_file(run_dir: Path, trajectory_index: int) -> Path:
    return run_dir / "result" / f"{trajectory_index + 1}.h5"


def load_rollout_arrays(result_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(result_file, "r") as handle:
        missing = [key for key in ("predicteds", "targets") if key not in handle]
        if missing:
            raise KeyError(f"{result_file} is missing datasets: {missing}")
        prediction = np.asarray(handle["predicteds"], dtype=np.float32)
        truth = np.asarray(handle["targets"], dtype=np.float32)

    if prediction.shape != truth.shape:
        raise ValueError(
            f"predicteds and targets must have the same shape, got "
            f"{prediction.shape} and {truth.shape}"
        )
    if prediction.ndim != 3 or prediction.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            "rollout arrays must have shape (steps, nodes, 4), got "
            f"{prediction.shape}"
        )
    return truth, prediction


def _read_frame_or_static(dataset: h5py.Dataset, frame: int) -> np.ndarray:
    shape = dataset.shape
    if len(shape) >= 3:
        if frame >= shape[0]:
            raise IndexError(f"frame {frame} is outside dataset shape {shape}")
        return np.asarray(dataset[frame])
    return np.asarray(dataset)


def _squeeze_node_column(array: np.ndarray) -> np.ndarray:
    if array.ndim > 1 and array.shape[-1] == 1:
        return np.squeeze(array, axis=-1)
    return array


def load_geometry(
    dataset_file: Path,
    *,
    trajectory_index: int,
    trajectory_key: str | None,
    frame: int,
    expected_nodes: int,
) -> dict[str, np.ndarray | str]:
    with h5py.File(dataset_file, "r") as handle:
        key = select_trajectory_key(
            handle,
            trajectory_index=trajectory_index,
            trajectory_key=trajectory_key,
        )
        group = handle[key]
        for required in ("pos", "node_type"):
            if required not in group:
                raise KeyError(f"trajectory {key!r} is missing {required!r}")
        nodes = np.asarray(_read_frame_or_static(group["pos"], frame), dtype=np.float32)
        node_type = _squeeze_node_column(_read_frame_or_static(group["node_type"], frame))
        node_type = np.asarray(node_type, dtype=np.int64)

    if nodes.ndim != 2 or nodes.shape[-1] != 2:
        raise ValueError(f"pos must have shape (nodes, 2), got {nodes.shape}")
    if node_type.ndim != 1:
        raise ValueError(f"node_type must reduce to shape (nodes,), got {node_type.shape}")
    if nodes.shape[0] != expected_nodes or node_type.shape[0] != expected_nodes:
        raise ValueError(
            "geometry and rollout arrays disagree on node count: "
            f"pos={nodes.shape[0]}, node_type={node_type.shape[0]}, "
            f"rollout={expected_nodes}"
        )
    return {"nodes": nodes, "node_type": node_type, "trajectory_key": key}


def node_filter_mask(node_type: np.ndarray, node_filter: str) -> np.ndarray:
    if node_filter not in NODE_FILTERS:
        raise ValueError(f"node_filter must be one of {NODE_FILTERS}")
    if node_filter == "all":
        return np.ones(node_type.shape[0], dtype=bool)
    normal = node_type == 0
    return normal if node_filter == "normal" else ~normal


def parse_steps(raw_steps: str | None, *, max_step: int) -> list[int]:
    if not raw_steps:
        return sorted(set([1, max(1, max_step // 4), max(1, max_step // 2), max_step]))

    steps: list[int] = []
    for part in raw_steps.replace(",", " ").split():
        step = int(part)
        if step < 1 or step > max_step:
            raise ValueError(f"snapshot step {step} is outside [1, {max_step}]")
        steps.append(step)
    return sorted(set(steps))


def relative_l2_by_time(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - truth, axis=1)
    denominator = np.linalg.norm(truth, axis=1)
    relative_error = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1.0e-12
    relative_error[valid] = numerator[valid] / denominator[valid]
    relative_error[~valid & (numerator <= 1.0e-12)] = 0.0
    return relative_error


def _auto_point_size(num_nodes: int) -> float:
    return float(np.clip(50_000.0 / max(num_nodes, 1), 0.25, 4.0))


def _limits(values: np.ndarray, user_limits: Sequence[float] | None = None) -> tuple[float, float]:
    if user_limits is not None:
        if len(user_limits) != 2:
            raise ValueError("color limits must contain exactly two values")
        vmin, vmax = float(user_limits[0]), float(user_limits[1])
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"invalid color limits: {(vmin, vmax)}")
    return vmin, vmax


def _error_limits(
    error: np.ndarray,
    *,
    error_kind: str,
    error_quantile: float,
    user_limits: Sequence[float] | None,
) -> tuple[float, float]:
    if user_limits is not None:
        return _limits(np.asarray(user_limits), user_limits)
    if not 0.0 < error_quantile <= 1.0:
        raise ValueError("error_quantile must be in (0, 1]")

    max_abs = float(np.nanquantile(np.abs(error), error_quantile))
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        max_abs = 1.0
    if error_kind == "absolute":
        return 0.0, max_abs
    if error_kind == "signed":
        return -max_abs, max_abs
    raise ValueError("error_kind must be 'absolute' or 'signed'")


def _writer(output_path: Path, fps: int):
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        return animation.FFMpegWriter(fps=fps, bitrate=2200)
    raise ValueError("animation output must end with .gif or .mp4")


def save_relative_error_plot(
    relative_error: np.ndarray,
    *,
    feature_name: str,
    output_path: Path,
) -> None:
    steps = np.arange(1, relative_error.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(steps, relative_error, linewidth=1.6)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"{feature_name} relative error")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _scatter_field(
    ax,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    point_size: float,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
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
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def save_snapshot(
    nodes: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    step: int,
    feature_name: str,
    value_limits: tuple[float, float],
    error_limits: tuple[float, float],
    error_kind: str,
    point_size: float,
    output_path: Path,
    field_cmap: str,
    abs_error_cmap: str,
    signed_error_cmap: str,
) -> None:
    index = step - 1
    signed_error = prediction[index] - truth[index]
    if error_kind == "absolute":
        error = np.abs(signed_error)
        error_title = f"Abs error {feature_name}, step {step}"
        error_cmap = abs_error_cmap
    else:
        error = signed_error
        error_title = f"Signed error {feature_name}, step {step}"
        error_cmap = signed_error_cmap

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    specs = (
        (truth[index], value_limits, field_cmap, f"Truth {feature_name}, step {step}"),
        (
            prediction[index],
            value_limits,
            field_cmap,
            f"Prediction {feature_name}, step {step}",
        ),
        (error, error_limits, error_cmap, error_title),
    )
    for ax, (values, limits, cmap, title) in zip(axes, specs):
        artist = _scatter_field(
            ax,
            nodes,
            values,
            point_size=point_size,
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            title=title,
        )
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(
    nodes: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    feature_name: str,
    frame_steps: Sequence[int],
    value_limits: tuple[float, float],
    error_limits: tuple[float, float],
    error_kind: str,
    point_size: float,
    output_path: Path,
    fps: int,
    field_cmap: str,
    abs_error_cmap: str,
    signed_error_cmap: str,
) -> None:
    first_index = frame_steps[0] - 1
    first_error = prediction[first_index] - truth[first_index]
    first_error_values = np.abs(first_error) if error_kind == "absolute" else first_error
    error_cmap = abs_error_cmap if error_kind == "absolute" else signed_error_cmap

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    artists = [
        _scatter_field(
            axes[0],
            nodes,
            truth[first_index],
            point_size=point_size,
            cmap=field_cmap,
            vmin=value_limits[0],
            vmax=value_limits[1],
            title="",
        ),
        _scatter_field(
            axes[1],
            nodes,
            prediction[first_index],
            point_size=point_size,
            cmap=field_cmap,
            vmin=value_limits[0],
            vmax=value_limits[1],
            title="",
        ),
        _scatter_field(
            axes[2],
            nodes,
            first_error_values,
            point_size=point_size,
            cmap=error_cmap,
            vmin=error_limits[0],
            vmax=error_limits[1],
            title="",
        ),
    ]
    for ax, artist in zip(axes, artists):
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

    def update(step: int):
        index = step - 1
        signed_error = prediction[index] - truth[index]
        error_values = np.abs(signed_error) if error_kind == "absolute" else signed_error
        artists[0].set_array(truth[index])
        artists[1].set_array(prediction[index])
        artists[2].set_array(error_values)
        axes[0].set_title(f"Truth {feature_name}, step {step}")
        axes[1].set_title(f"Prediction {feature_name}, step {step}")
        axes[2].set_title(
            f"{'Abs' if error_kind == 'absolute' else 'Signed'} error "
            f"{feature_name}, step {step}"
        )
        return artists

    ani = animation.FuncAnimation(fig, update, frames=frame_steps, blit=False)
    ani.save(output_path, writer=_writer(output_path, fps=fps))
    plt.close(fig)


def save_official_rollout_visualization(
    *,
    dataset_file: Path,
    result_file: Path,
    output_dir: Path,
    trajectory_index: int = 0,
    trajectory_key: str | None = None,
    geometry_frame: int = 0,
    feature: str | int = "pres",
    node_filter: str = "all",
    animation_name: str | None = None,
    no_animation: bool = False,
    fps: int = 5,
    frame_stride: int = 1,
    snapshot_steps: str | None = None,
    error_kind: str = "absolute",
    field_color_limits: Sequence[float] | None = None,
    error_color_limits: Sequence[float] | None = None,
    error_quantile: float = 1.0,
    point_size: float | None = None,
    field_cmap: str = "viridis",
    abs_error_cmap: str = "magma",
    signed_error_cmap: str = "coolwarm",
) -> dict[str, object]:
    truth, prediction = load_rollout_arrays(result_file)
    feature_index = parse_feature(feature)
    feature_name = FEATURE_NAMES[feature_index]
    geometry = load_geometry(
        dataset_file,
        trajectory_index=trajectory_index,
        trajectory_key=trajectory_key,
        frame=geometry_frame,
        expected_nodes=truth.shape[1],
    )
    nodes = np.asarray(geometry["nodes"])
    node_type = np.asarray(geometry["node_type"])
    mask = node_filter_mask(node_type, node_filter)
    if not np.any(mask):
        raise ValueError(f"node_filter={node_filter!r} selected no nodes")

    nodes = nodes[mask]
    truth_feature = truth[:, mask, feature_index]
    pred_feature = prediction[:, mask, feature_index]
    error_feature = pred_feature - truth_feature

    output_dir.mkdir(parents=True, exist_ok=True)
    value_limits = _limits(
        np.concatenate((truth_feature.reshape(-1), pred_feature.reshape(-1))),
        field_color_limits,
    )
    error_limits = _error_limits(
        error_feature,
        error_kind=error_kind,
        error_quantile=error_quantile,
        user_limits=error_color_limits,
    )
    if point_size is None:
        point_size = _auto_point_size(nodes.shape[0])

    max_step = truth_feature.shape[0]
    snapshots = parse_steps(snapshot_steps, max_step=max_step)
    frame_steps = list(range(1, max_step + 1, frame_stride))
    if frame_steps[-1] != max_step:
        frame_steps.append(max_step)

    stem = (
        f"traj_{trajectory_index:02d}_{feature_name}_{node_filter}"
        f"_stride{frame_stride}"
    )
    if animation_name is None:
        animation_name = f"{stem}.gif"
    animation_path = output_dir / animation_name
    rel_error_path = output_dir / f"{stem}_relative_l2.png"

    relative_error = relative_l2_by_time(truth_feature, pred_feature)
    save_relative_error_plot(
        relative_error,
        feature_name=feature_name,
        output_path=rel_error_path,
    )

    snapshot_paths = []
    for step in snapshots:
        path = output_dir / f"{stem}_step_{step:04d}.png"
        save_snapshot(
            nodes,
            truth_feature,
            pred_feature,
            step=step,
            feature_name=feature_name,
            value_limits=value_limits,
            error_limits=error_limits,
            error_kind=error_kind,
            point_size=float(point_size),
            output_path=path,
            field_cmap=field_cmap,
            abs_error_cmap=abs_error_cmap,
            signed_error_cmap=signed_error_cmap,
        )
        snapshot_paths.append(path)

    if not no_animation:
        save_animation(
            nodes,
            truth_feature,
            pred_feature,
            feature_name=feature_name,
            frame_steps=frame_steps,
            value_limits=value_limits,
            error_limits=error_limits,
            error_kind=error_kind,
            point_size=float(point_size),
            output_path=animation_path,
            fps=fps,
            field_cmap=field_cmap,
            abs_error_cmap=abs_error_cmap,
            signed_error_cmap=signed_error_cmap,
        )
    else:
        animation_path = None

    summary = {
        "animation_path": str(animation_path) if animation_path is not None else None,
        "relative_error_path": str(rel_error_path),
        "snapshot_paths": [str(path) for path in snapshot_paths],
        "dataset_file": str(dataset_file),
        "result_file": str(result_file),
        "trajectory_index": int(trajectory_index),
        "trajectory_key": str(geometry["trajectory_key"]),
        "geometry_frame": int(geometry_frame),
        "feature": feature_name,
        "node_filter": node_filter,
        "num_nodes_plotted": int(nodes.shape[0]),
        "num_steps": int(max_step),
        "frame_steps": [int(step) for step in frame_steps],
        "snapshot_steps": [int(step) for step in snapshots],
        "value_limits": [float(value_limits[0]), float(value_limits[1])],
        "error_limits": [float(error_limits[0]), float(error_limits[1])],
        "error_kind": error_kind,
        "error_quantile": float(error_quantile),
        "relative_l2_final": float(relative_error[-1]),
        "relative_l2_mean": float(np.nanmean(relative_error)),
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Official run dir containing result/*.h5")
    source.add_argument("--result-file", type=Path, help="Specific rollout result HDF5 file")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--trajectory-key")
    parser.add_argument("--geometry-frame", type=int, default=0)
    parser.add_argument("--feature", default="pres")
    parser.add_argument("--node-filter", choices=NODE_FILTERS, default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--animation-name")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--snapshot-steps", help="Comma- or space-separated 1-based rollout steps")
    parser.add_argument("--error-kind", choices=("absolute", "signed"), default="absolute")
    parser.add_argument("--field-color-limits", nargs=2, type=float)
    parser.add_argument("--error-color-limits", nargs=2, type=float)
    parser.add_argument("--error-quantile", type=float, default=1.0)
    parser.add_argument("--point-size", type=float)
    parser.add_argument("--field-cmap", default="viridis")
    parser.add_argument("--abs-error-cmap", default="magma")
    parser.add_argument("--signed-error-cmap", default="coolwarm")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.fps < 1:
        raise SystemExit("--fps must be at least 1")

    dataset_file = args.dataset_root / f"{args.split}.h5"
    if args.result_file is None:
        result_file = infer_result_file(args.run_dir, args.trajectory_index)
        default_output_dir = args.run_dir / "visualizations"
    else:
        result_file = args.result_file
        default_output_dir = result_file.parent / "visualizations"
    output_dir = args.output_dir or default_output_dir

    summary = save_official_rollout_visualization(
        dataset_file=dataset_file,
        result_file=result_file,
        output_dir=output_dir,
        trajectory_index=args.trajectory_index,
        trajectory_key=args.trajectory_key,
        geometry_frame=args.geometry_frame,
        feature=args.feature,
        node_filter=args.node_filter,
        animation_name=args.animation_name,
        no_animation=args.no_animation,
        fps=args.fps,
        frame_stride=args.frame_stride,
        snapshot_steps=args.snapshot_steps,
        error_kind=args.error_kind,
        field_color_limits=args.field_color_limits,
        error_color_limits=args.error_color_limits,
        error_quantile=args.error_quantile,
        point_size=args.point_size,
        field_cmap=args.field_cmap,
        abs_error_cmap=args.abs_error_cmap,
        signed_error_cmap=args.signed_error_cmap,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

