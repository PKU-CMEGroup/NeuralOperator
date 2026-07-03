from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pcno.pcno import compute_Fourier_modes, euler2d_PCNO


FEATURE_NAMES = ("rho", "u", "v", "p")
DEFAULT_DATA_PATH = "/root/autodl-tmp/data/Euler_eq_2d/npy_forward_300/"


def _as_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _load_euler_forward_data(
    data_path: Union[str, Path],
    equal_weights: bool = False,
):
    data_path = _as_path(data_path)
    data_file = data_path / "pcno_Euler_forward_data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find preprocessed data file: {data_file}")

    data = np.load(data_file)
    nnodes = data["nnodes"]
    node_mask = data["node_mask"].astype(np.float32)
    if node_mask.ndim == 2:
        node_mask = node_mask[..., None]
    nodes = data["nodes"].astype(np.float32)
    node_weights = (
        data["node_equal_weights"] if equal_weights else data["node_weights"]
    ).astype(np.float32)
    node_measures = data["node_measures"]
    node_measures_raw = data["node_measures_raw"]
    directed_edges = data["directed_edges"].astype(np.int64)
    edge_gradient_weights = data["edge_gradient_weights"].astype(np.float32)
    features = data["features"].astype(np.float32)

    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]
    node_rhos = node_rhos.astype(np.float32)

    return {
        "data_path": data_path,
        "nnodes": nnodes,
        "node_mask": node_mask,
        "nodes": nodes,
        "node_weights": node_weights,
        "node_rhos": node_rhos,
        "features": features,
        "directed_edges": directed_edges,
        "edge_gradient_weights": edge_gradient_weights,
    }


def _valid_node_count(nnodes: np.ndarray, node_mask: np.ndarray, index: int) -> int:
    count = np.asarray(nnodes[index]).reshape(-1)[0]
    if np.isfinite(count) and count > 0:
        return int(count)

    sample_mask = node_mask[index]
    if sample_mask.ndim > 1:
        sample_mask = sample_mask[..., 0]
    return int(np.count_nonzero(sample_mask))


def _parse_feature_index(feature: Union[int, str], nfeatures: int) -> int:
    if isinstance(feature, str):
        feature = feature.strip()
        if feature in FEATURE_NAMES:
            return FEATURE_NAMES.index(feature)
        try:
            feature = int(feature)
        except ValueError as exc:
            raise ValueError(
                f"Unknown feature {feature!r}; use one of {FEATURE_NAMES} or an integer."
            ) from exc

    feature_index = int(feature)
    if feature_index < 0 or feature_index >= nfeatures:
        raise IndexError(
            f"feature_index={feature_index} is outside available feature range [0, {nfeatures})."
        )
    return feature_index


def _load_state_dict(model_path: Union[str, Path], device: torch.device):
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def build_euler2d_pcno_from_data(
    data: dict,
    model_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    k_max: int = 12,
    domain_lengths: Sequence[float] = (6.0, 2.0),
    layers: Sequence[int] = (128, 128, 128, 128, 128),
    fc_dim: int = 128,
    act: str = "gelu",
) -> torch.nn.Module:
    """Rebuild the PCNO architecture used by forward_train.py and load weights."""

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    nodes = data["nodes"]
    node_weights = data["node_weights"]
    node_rhos = data["node_rhos"]
    features = data["features"]
    ndim = nodes.shape[-1]
    nmeasures = node_weights.shape[-1]
    if len(domain_lengths) < ndim:
        raise ValueError(
            f"domain_lengths must have at least {ndim} entries, got {domain_lengths}."
        )

    modes = compute_Fourier_modes(
        ndim,
        [k_max] * (ndim * nmeasures),
        list(domain_lengths[:ndim]) * nmeasures,
    )
    modes = torch.tensor(modes, dtype=torch.float32, device=device)

    model = euler2d_PCNO(
        ndim,
        modes,
        nmeasures=nmeasures,
        layers=list(layers),
        fc_dim=fc_dim,
        in_dim=nodes.shape[-1] + node_rhos.shape[-1] + features.shape[-1],
        out_dim=features.shape[-1],
        act=act,
    ).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.eval()
    return model


def _make_sample_tensors(data: dict, index: int, device: torch.device):
    return {
        "nodes": torch.from_numpy(data["nodes"][[index]]).to(device),
        "node_rhos": torch.from_numpy(data["node_rhos"][[index]]).to(device),
        "node_mask": torch.from_numpy(data["node_mask"][[index]]).to(device),
        "node_weights": torch.from_numpy(data["node_weights"][[index]]).to(device),
        "directed_edges": torch.from_numpy(data["directed_edges"][[index]]).to(device),
        "edge_gradient_weights": torch.from_numpy(
            data["edge_gradient_weights"][[index]]
        ).to(device),
    }


def rollout_euler2d_sample(
    model: torch.nn.Module,
    data: dict,
    index: int,
    n_time: int,
    start_time: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one sample autoregressively.

    Returns:
        truth:      [n_time + 1, nnodes, nfeatures], including the initial frame.
        prediction: [n_time + 1, nnodes, nfeatures], prediction[0] equals truth[0].
    """

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    features = data["features"]
    nsamples, ntotal = features.shape[:2]
    if n_time < 1:
        raise ValueError(f"n_time must be at least 1, got {n_time}.")
    if index < 0 or index >= nsamples:
        raise IndexError(f"index={index} is outside available sample range [0, {nsamples}).")
    if start_time < 0 or start_time >= ntotal:
        raise IndexError(
            f"start_time={start_time} is outside available time range [0, {ntotal})."
        )
    if start_time + n_time >= ntotal:
        raise ValueError(
            f"Need truth frames through time {start_time + n_time}, but dataset has "
            f"only {ntotal} frames. Choose n_time <= {ntotal - start_time - 1}."
        )

    count = _valid_node_count(data["nnodes"], data["node_mask"], index)
    truth = features[index, start_time : start_time + n_time + 1, :count, :].copy()
    prediction = np.empty_like(truth)
    prediction[0] = truth[0]

    sample = _make_sample_tensors(data, index, device)
    aux = (
        sample["node_mask"],
        sample["nodes"],
        sample["node_weights"],
        sample["directed_edges"],
        sample["edge_gradient_weights"],
    )
    current = torch.from_numpy(features[index, start_time : start_time + 1]).to(device)

    model.eval()
    with torch.no_grad():
        for step in range(1, n_time + 1):
            x = torch.cat((sample["nodes"], sample["node_rhos"], current), dim=-1)
            current = model(x, aux)
            current = current * sample["node_mask"]
            prediction[step] = current.detach().cpu().numpy()[0, :count, :]

    return truth, prediction


def _load_plot_triangles(data_path: Path, sample_index: int, nnodes: int):
    elems_file = data_path / str(sample_index) / "elems.npy"
    if not elems_file.exists():
        return None

    elems = np.load(elems_file).astype(np.int64)
    if elems.ndim != 2:
        return None

    triangles = []
    has_elem_dim = elems.shape[1] >= 4 and np.all(np.isin(elems[:, 0], [1, 2, 3]))
    for elem in elems:
        node_ids = elem[1:] if has_elem_dim else elem
        node_ids = node_ids[node_ids >= 0]
        if len(node_ids) == 3:
            triangles.append(node_ids)
        elif len(node_ids) == 4:
            triangles.append(node_ids[[0, 1, 2]])
            triangles.append(node_ids[[1, 2, 3]])

    if not triangles:
        return None

    triangles = np.asarray(triangles, dtype=np.int64)
    valid = np.all((triangles >= 0) & (triangles < nnodes), axis=1)
    triangles = triangles[valid]
    return triangles if len(triangles) else None


def _plot_scalar_field(ax, nodes, triangles, values, vmin, vmax, cmap, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if triangles is not None:
        artist = ax.tripcolor(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        artist = ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=values,
            s=4,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
    return artist


def _use_color_limits(auto_limits, user_limits):
    if user_limits is None:
        return auto_limits
    if len(user_limits) != 2:
        raise ValueError(f"Color limits must be (vmin, vmax), got {user_limits}.")
    vmin, vmax = float(user_limits[0]), float(user_limits[1])
    if vmin >= vmax:
        raise ValueError(f"Color limits require vmin < vmax, got {user_limits}.")
    return vmin, vmax


def _relative_l2_by_time(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - truth, axis=1)
    denominator = np.linalg.norm(truth, axis=1)
    relative_error = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator > 1.0e-12
    relative_error[valid] = numerator[valid] / denominator[valid]
    relative_error[~valid & (numerator <= 1.0e-12)] = 0.0
    return relative_error


def _save_relative_error_plot(
    relative_error: np.ndarray,
    feature_name: str,
    output_path: Union[str, Path],
):
    steps = np.arange(relative_error.shape[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(steps, relative_error, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"{feature_name} relative error over time")
    ax.grid(True, alpha=0.35)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_rollout_frame(
    nodes: np.ndarray,
    triangles: Optional[np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    step: int,
    feature_name: str,
    value_limits: tuple[float, float],
    error_limit: float,
    output_path: Union[str, Path],
    error_kind: str = "absolute",
    field_color_limits: Optional[tuple[float, float]] = None,
    error_color_limits: Optional[tuple[float, float]] = None,
    field_cmap: str = "viridis",
    abs_error_cmap: str = "magma",
    signed_error_cmap: str = "coolwarm",
):
    signed_error = prediction[step] - truth[step]
    if error_kind == "absolute":
        error = np.abs(signed_error)
        error_limits = _use_color_limits((0.0, error_limit), error_color_limits)
        error_cmap = abs_error_cmap
        error_title = f"Abs error {feature_name}, step {step}"
    elif error_kind == "signed":
        error = signed_error
        error_limits = _use_color_limits((-error_limit, error_limit), error_color_limits)
        error_cmap = signed_error_cmap
        error_title = f"Signed error {feature_name}, step {step}"
    else:
        raise ValueError("error_kind must be 'absolute' or 'signed'.")

    value_limits = _use_color_limits(value_limits, field_color_limits)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    titles = (
        f"Truth {feature_name}, step {step}",
        f"Prediction {feature_name}, step {step}",
        error_title,
    )
    plots = (
        (truth[step], value_limits[0], value_limits[1], field_cmap, titles[0]),
        (prediction[step], value_limits[0], value_limits[1], field_cmap, titles[1]),
        (error, error_limits[0], error_limits[1], error_cmap, titles[2]),
    )
    for ax, (values, vmin, vmax, cmap, title) in zip(axes, plots):
        artist = _plot_scalar_field(ax, nodes, triangles, values, vmin, vmax, cmap, title)
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _animation_writer(output_path: Path, fps: int):
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        return animation.FFMpegWriter(fps=fps, bitrate=1800)
    raise ValueError("animation_path must end with .gif or .mp4")


def save_euler2d_rollout_plots(
    data: dict,
    index: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_index: Union[int, str] = 0,
    output_dir: Union[str, Path] = "euler2d_rollout_outputs",
    snapshot_steps: Optional[Iterable[int]] = None,
    animation_name: Optional[str] = None,
    fps: int = 5,
    error_kind: str = "absolute",
    field_color_limits: Optional[tuple[float, float]] = None,
    error_color_limits: Optional[tuple[float, float]] = None,
    field_cmap: str = "viridis",
    abs_error_cmap: str = "magma",
    signed_error_cmap: str = "coolwarm",
    relative_error_name: Optional[str] = None,
) -> dict:
    """Save truth/prediction/error plots and relative error over time."""

    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nsteps = truth.shape[0] - 1
    feature_idx = _parse_feature_index(feature_index, truth.shape[-1])
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"f{feature_idx}"

    count = truth.shape[1]
    nodes = data["nodes"][index, :count, :]
    triangles = _load_plot_triangles(data["data_path"], index, count)

    truth_feature = truth[..., feature_idx]
    pred_feature = prediction[..., feature_idx]
    value_min = float(np.nanmin([truth_feature.min(), pred_feature.min()]))
    value_max = float(np.nanmax([truth_feature.max(), pred_feature.max()]))
    value_limits = _use_color_limits(
        (value_min, value_max),
        field_color_limits,
    )
    error_feature = pred_feature - truth_feature
    error_limit = float(np.nanmax(np.abs(error_feature)))
    if error_limit == 0.0:
        error_limit = 1.0
    if error_kind not in ("absolute", "signed"):
        raise ValueError("error_kind must be 'absolute' or 'signed'.")
    auto_error_limits = (
        (0.0, error_limit)
        if error_kind == "absolute"
        else (-error_limit, error_limit)
    )
    error_limits = _use_color_limits(auto_error_limits, error_color_limits)

    if animation_name is None:
        animation_name = f"sample_{index}_{feature_name}_rollout.gif"
    animation_path = output_dir / animation_name
    if relative_error_name is None:
        relative_error_name = f"sample_{index}_{feature_name}_relative_error.png"
    relative_error_path = output_dir / relative_error_name

    relative_error = _relative_l2_by_time(truth_feature, pred_feature)
    _save_relative_error_plot(relative_error, feature_name, relative_error_path)

    frames = list(range(1, nsteps + 1))
    fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)

    def update(step):
        fig.clear()
        axes = fig.subplots(1, 3)
        _save_artists = []
        titles = (
            f"Truth {feature_name}, step {step}",
            f"Prediction {feature_name}, step {step}",
            f"Error {feature_name}, step {step}",
        )
        plot_specs = (
            (truth_feature[step], value_limits[0], value_limits[1], field_cmap, titles[0]),
            (pred_feature[step], value_limits[0], value_limits[1], field_cmap, titles[1]),
        )
        if error_kind == "absolute":
            error_values = np.abs(error_feature[step])
            error_spec = (
                error_values,
                error_limits[0],
                error_limits[1],
                abs_error_cmap,
                f"Abs error {feature_name}, step {step}",
            )
        else:
            error_spec = (
                error_feature[step],
                error_limits[0],
                error_limits[1],
                signed_error_cmap,
                f"Signed error {feature_name}, step {step}",
            )
        plot_specs = plot_specs + (error_spec,)
        for ax, (values, vmin, vmax, cmap, title) in zip(axes, plot_specs):
            artist = _plot_scalar_field(ax, nodes, triangles, values, vmin, vmax, cmap, title)
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
            _save_artists.append(artist)
        return _save_artists

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(animation_path, writer=_animation_writer(animation_path, fps=fps))
    plt.close(fig)

    if snapshot_steps is None:
        snapshot_steps = sorted(set([1, max(1, nsteps // 4), max(1, nsteps // 2), nsteps]))

    snapshot_paths = []
    for step in snapshot_steps:
        step = int(step)
        if step < 0 or step > nsteps:
            raise ValueError(f"snapshot step {step} is outside [0, {nsteps}].")
        frame_path = output_dir / f"sample_{index}_{feature_name}_step_{step:04d}.png"
        _save_rollout_frame(
            nodes,
            triangles,
            truth_feature,
            pred_feature,
            step,
            feature_name,
            (value_min, value_max),
            error_limit,
            frame_path,
            error_kind=error_kind,
            field_color_limits=field_color_limits,
            error_color_limits=error_color_limits,
            field_cmap=field_cmap,
            abs_error_cmap=abs_error_cmap,
            signed_error_cmap=signed_error_cmap,
        )
        snapshot_paths.append(frame_path)

    return {
        "animation_path": animation_path,
        "snapshot_paths": snapshot_paths,
        "relative_error_path": relative_error_path,
        "relative_error": relative_error,
        "feature_index": feature_idx,
        "feature_name": feature_name,
    }


def rollout_and_plot_euler2d_sample(
    index: int,
    n_time: int,
    feature_index: Union[int, str] = 0,
    model_path: Union[str, Path] = "PCNO_forward_euler_exp_model.pth",
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    output_dir: Union[str, Path] = "euler2d_rollout_outputs",
    start_time: int = 0,
    snapshot_steps: Optional[Iterable[int]] = None,
    animation_name: Optional[str] = None,
    fps: int = 5,
    device: Optional[Union[str, torch.device]] = None,
    equal_weights: bool = False,
    error_kind: str = "absolute",
    field_color_limits: Optional[tuple[float, float]] = None,
    error_color_limits: Optional[tuple[float, float]] = None,
    field_cmap: str = "viridis",
    abs_error_cmap: str = "magma",
    signed_error_cmap: str = "coolwarm",
    relative_error_name: Optional[str] = None,
) -> dict:
    """Load a trained model, roll out one sample, and save rollout visualizations."""

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    data = _load_euler_forward_data(data_path, equal_weights=equal_weights)
    model = build_euler2d_pcno_from_data(data, model_path=model_path, device=device)
    truth, prediction = rollout_euler2d_sample(
        model,
        data,
        index=index,
        n_time=n_time,
        start_time=start_time,
        device=device,
    )
    plot_info = save_euler2d_rollout_plots(
        data,
        index=index,
        truth=truth,
        prediction=prediction,
        feature_index=feature_index,
        output_dir=output_dir,
        snapshot_steps=snapshot_steps,
        animation_name=animation_name,
        fps=fps,
        error_kind=error_kind,
        field_color_limits=field_color_limits,
        error_color_limits=error_color_limits,
        field_cmap=field_cmap,
        abs_error_cmap=abs_error_cmap,
        signed_error_cmap=signed_error_cmap,
        relative_error_name=relative_error_name,
    )

    return {
        "truth": truth,
        "prediction": prediction,
        "error": prediction - truth,
        **plot_info,
    }


def run_from_internal_config():
    # Modify the values in this block, then run this file directly.
    index = 0
    n_time = 60
    feature_index = "rho"  # one of: "rho", "u", "v", "p"; or use 0, 1, 2, 3
    model_path = "PCNO_forward_euler_exp_model.pth"
    data_path = DEFAULT_DATA_PATH
    output_dir = "euler2d_rollout_outputs"
    start_time = 0
    snapshot_steps = [1, 10, 30, 60]
    animation_name = None  # e.g. "sample_0_rho_rollout.gif" or "sample_0_rho_rollout.mp4"
    fps = 5
    device = None  # None uses cuda:0 if available, otherwise cpu
    equal_weights = False
    error_kind = "absolute"  # "absolute" or "signed"
    field_color_limits = None  # e.g. (0.8, 1.8), shared by truth and prediction
    error_color_limits = None  # e.g. (0.0, 0.2) for absolute, or (-0.2, 0.2) for signed
    field_cmap = "viridis"
    abs_error_cmap = "magma"
    signed_error_cmap = "coolwarm"
    relative_error_name = None  # e.g. "sample_0_rho_relative_error.png"

    result = rollout_and_plot_euler2d_sample(
        index=index,
        n_time=n_time,
        feature_index=feature_index,
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        start_time=start_time,
        snapshot_steps=snapshot_steps,
        animation_name=animation_name,
        fps=fps,
        device=device,
        equal_weights=equal_weights,
        error_kind=error_kind,
        field_color_limits=field_color_limits,
        error_color_limits=error_color_limits,
        field_cmap=field_cmap,
        abs_error_cmap=abs_error_cmap,
        signed_error_cmap=signed_error_cmap,
        relative_error_name=relative_error_name,
    )
    print(f"Animation saved to: {result['animation_path']}")
    print(f"Relative error plot saved to: {result['relative_error_path']}")
    for path in result["snapshot_paths"]:
        print(f"Frame saved to: {path}")
    return result


def main():
    run_from_internal_config()


if __name__ == "__main__":
    main()
