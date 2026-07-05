"""Dump and summarize CPGNet solver-facing interface latents.

This script imports the local author ``cpggnspdes`` checkout at runtime.  It
does not vendor author code into this repository; it only drives the official
model path and records compact diagnostics under ignored artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.cpg_interface_latents import (  # noqa: E402
    WAVE_TYPE_ORDER,
    admissibility_summary,
    classify_wave_edges,
    conservative_update_from_flux,
    edge_pressure_jump_scores,
    edge_shock_mask_from_scores,
    llf_flux_decomposition,
    node_mask_from_edge_mask,
    split_directed_reconstruct_prims,
    summarize_trace_likeness,
    trace_likeness_arrays,
    update_error_summary,
)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    CONSERVATIVE_NAMES,
    PRIMITIVE_NAMES,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid,
    shock_front_mask_from_scores,
    shock_front_scores,
)


EPS = 1.0e-12


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        return name.strip(), Path(value)
    path = Path(raw)
    return path.stem if path.is_file() else path.name, path


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(PRIMITIVE_NAMES)}


def by_cons(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[index]) for index, name in enumerate(CONSERVATIVE_NAMES)}


def finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=run_dir_or_checkpoint; checkpoint defaults to run/checkpoint/simulator.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"artifacts/time_dependent_no/cpg_interface_latent_diagnostic_{date}"),
    )
    parser.add_argument(
        "--trajectory-indices",
        nargs="+",
        type=int,
        default=[0, 6, 11, 13, 17],
    )
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 20, 40, 58, 78])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--boundary-mode",
        choices=["official-clamped", "current"],
        default="official-clamped",
        help="official-clamped matches rollout.py by replacing boundary current state with target",
    )
    parser.add_argument("--shock-quantile", type=float, default=0.90)
    parser.add_argument("--wave-high-quantile", type=float, default=0.75)
    parser.add_argument("--wave-smooth-quantile", type=float, default=0.50)
    parser.add_argument("--max-sampled-edges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-full-npz", action="store_true")
    parser.add_argument("--capture-hidden-edge", action="store_true")
    parser.add_argument("--no-capture-decoder-logits", action="store_true")
    return parser.parse_args(argv)


def select_device(args: argparse.Namespace):
    import torch

    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_checkpoint(path: Path) -> Path:
    return path if path.is_file() else path / "checkpoint" / "simulator.pth"


class ModelCapture:
    """Small hook collector for optional decoder/hidden-edge summaries."""

    def __init__(
        self,
        simulator: Any,
        *,
        capture_decoder_logits: bool,
        capture_hidden_edge: bool,
    ) -> None:
        self.data: dict[str, dict[str, float]] = {}
        self.handles = []
        model = simulator.model
        if capture_decoder_logits:
            self.handles.append(
                model.decoder.decode0.register_forward_hook(
                    self._stats_hook("decoder_raw_rho")
                )
            )
            self.handles.append(
                model.decoder.decode3.register_forward_hook(
                    self._stats_hook("decoder_raw_pres")
                )
            )
        if capture_hidden_edge:
            self.handles.append(
                model.convRecon.register_forward_hook(
                    self._stats_hook("hidden_edge_embedding")
                )
            )

    def clear(self) -> None:
        self.data.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _stats_hook(self, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output.detach()
            if tensor.numel() == 0:
                self.data[name] = {
                    "mean": math.nan,
                    "std": math.nan,
                    "min": math.nan,
                    "max": math.nan,
                    "negative_fraction": math.nan,
                }
                return
            self.data[name] = {
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "negative_fraction": float((tensor < 0).float().mean().item()),
            }

        return hook


def make_model(checkpoint: Path, device: Any):
    from modelEdgeUpd.simulator import Simulator

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Simulator(
        message_passing_num=12,
        node_input_size=6,
        edge_input_size=5,
        device=device,
        model_dir=str(checkpoint),
    ).to(device)
    model.load_checkpoint(str(checkpoint))
    model.eval()
    return model


def load_graphs_for_trajectory(
    dataset: Any,
    loader: Any,
    transformer: Any,
    trajectory_index: int,
    frames: Sequence[int],
) -> dict[int, Any]:
    from utils.to_undirected import make_edges_undirected

    wanted = set(frames)
    if not wanted:
        return {}
    dataset.change_file(trajectory_index)
    out = {}
    max_frame = max(wanted)
    for step, graph in enumerate(loader):
        if step > max_frame:
            break
        if step not in wanted:
            continue
        graph = make_edges_undirected(graph)
        graph = transformer(graph)
        out[step] = graph.cpu()
    missing = sorted(wanted.difference(out))
    if missing:
        raise RuntimeError(
            f"trajectory {trajectory_index} did not yield requested frames {missing}"
        )
    return out


def tensor_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def run_frame(
    model: Any,
    capture: ModelCapture,
    graph: Any,
    *,
    device: Any,
    node_type_normal_value: int,
    boundary_mode: str,
) -> dict[str, np.ndarray | dict[str, dict[str, float]]]:
    import torch

    graph = graph.clone().to(device)
    node_type = graph.x[:, 0].clone()
    normal = node_type == node_type_normal_value
    boundary = torch.logical_not(normal)
    target = graph.y.clone()
    if boundary_mode == "official-clamped":
        graph.x[boundary, 1:5] = target[boundary]
    current = graph.x[:, 1:5].clone()

    capture.clear()
    with torch.no_grad():
        prediction = model(graph, sequence_noise=None)
    prediction_clamped = prediction.clone()
    if boundary_mode == "official-clamped":
        prediction_clamped[boundary] = target[boundary]

    full_edge_index = tensor_to_numpy(graph.edge_index.T).astype(np.int64)
    unique_edge_index = tensor_to_numpy(graph.edge_ind_unique.T).astype(np.int64)
    edge_count = unique_edge_index.shape[0]
    return {
        "current_primitives": tensor_to_numpy(current),
        "target_primitives": tensor_to_numpy(target),
        "prediction_primitives": tensor_to_numpy(prediction),
        "prediction_clamped_primitives": tensor_to_numpy(prediction_clamped),
        "node_type": tensor_to_numpy(node_type).astype(np.int64),
        "normal_mask": tensor_to_numpy(normal).astype(bool),
        "pos": tensor_to_numpy(graph.pos)[:, :2],
        "full_edge_index": full_edge_index,
        "unique_edge_index": unique_edge_index,
        "reconstruct_prims": tensor_to_numpy(graph.reconstruct_prims),
        "edge_factor": tensor_to_numpy(graph.edge_factor),
        "edge_unit_normal": tensor_to_numpy(graph.edge_unit_normal[:edge_count]),
        "current_consers": tensor_to_numpy(graph.current_consers),
        "capture": dict(capture.data),
    }


def frame_diagnostics(
    arrays: dict[str, Any],
    *,
    run_name: str,
    trajectory_index: int,
    frame: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current = arrays["current_primitives"]
    target = arrays["target_primitives"]
    prediction = arrays["prediction_clamped_primitives"]
    node_type = arrays["node_type"]
    normal = arrays["normal_mask"]
    pos = arrays["pos"]
    unique_edges = arrays["unique_edge_index"]
    full_edges = arrays["full_edge_index"]
    reconstruct = arrays["reconstruct_prims"]
    normals = arrays["edge_unit_normal"]
    edge_factor = arrays["edge_factor"]
    num_nodes = current.shape[0]
    edge_normal_mask = normal[unique_edges[:, 0]] & normal[unique_edges[:, 1]]

    left, right = split_directed_reconstruct_prims(
        reconstruct, num_unique_edges=unique_edges.shape[0]
    )
    decomp = llf_flux_decomposition(left, right, normals)
    fv = conservative_update_from_flux(
        decomp["flux"], edge_factor, full_edges, num_nodes=num_nodes
    )
    true_delta = primitive_to_conservative(target) - primitive_to_conservative(current)
    induced_delta = fv["conservative_delta"]
    model_delta = primitive_to_conservative(prediction) - primitive_to_conservative(current)
    target_pressure_jumps = edge_pressure_jump_scores(target, unique_edges)
    shock_edges = edge_shock_mask_from_scores(
        target_pressure_jumps,
        quantile=args.shock_quantile,
        edge_mask=edge_normal_mask,
    )
    shock_nodes = node_mask_from_edge_mask(unique_edges, shock_edges, num_nodes=num_nodes)
    shock_nodes &= normal
    smooth_nodes = normal & ~shock_nodes
    trace_arrays = trace_likeness_arrays(left, right, current, unique_edges)
    wave = classify_wave_edges(
        current,
        unique_edges,
        normals,
        shock_edge_mask=shock_edges,
        edge_mask=edge_normal_mask,
        high_quantile=args.wave_high_quantile,
        smooth_quantile=args.wave_smooth_quantile,
    )
    diss_mag = np.linalg.norm(decomp["dissipation"], axis=-1)
    central_mag = np.linalg.norm(decomp["central"], axis=-1)
    flux_mag = np.linalg.norm(decomp["flux"], axis=-1)
    weighted_msg_mag = np.linalg.norm(fv["messages"][: unique_edges.shape[0]], axis=-1)
    wave_speed = decomp["wave_speed"]

    frame_summary: dict[str, Any] = {
        "run": run_name,
        "trajectory_index": int(trajectory_index),
        "frame": int(frame),
        "node_count": int(num_nodes),
        "normal_node_count": int(np.count_nonzero(normal)),
        "unique_edge_count": int(unique_edges.shape[0]),
        "normal_normal_edge_count": int(np.count_nonzero(edge_normal_mask)),
        "shock_edge_count": int(np.count_nonzero(shock_edges)),
        "shock_node_count": int(np.count_nonzero(shock_nodes)),
        "admissibility": admissibility_summary(left, right, current, node_mask=normal),
        "trace_all_normal_edges": summarize_trace_likeness(
            trace_arrays, edge_mask=edge_normal_mask
        ),
        "trace_shock_edges": summarize_trace_likeness(
            trace_arrays, edge_mask=shock_edges
        ),
        "trace_smooth_edges": summarize_trace_likeness(
            trace_arrays, edge_mask=edge_normal_mask & ~shock_edges
        ),
        "update_match_all_normal_nodes": update_error_summary(
            induced_delta, true_delta, node_mask=normal
        ),
        "update_match_shock_nodes": update_error_summary(
            induced_delta, true_delta, node_mask=shock_nodes
        ),
        "update_match_smooth_nodes": update_error_summary(
            induced_delta, true_delta, node_mask=smooth_nodes
        ),
        "model_delta_match": update_error_summary(
            induced_delta, model_delta, node_mask=normal
        ),
        "state_rmse_normal": primitive_rmse(prediction, target, normal),
        "dissipation": dissipation_summary(
            diss_mag,
            central_mag,
            flux_mag,
            weighted_msg_mag,
            wave_speed,
            edge_normal_mask=edge_normal_mask,
            shock_edges=shock_edges,
        ),
        "front_motion": front_motion_summary(
            current,
            prediction,
            target,
            pos,
            unique_edges,
            normal,
            quantile=args.shock_quantile,
        ),
        "decoder_capture": arrays.get("capture", {}),
    }

    wave_rows = []
    for label in WAVE_TYPE_ORDER:
        mask = wave["masks"][label]
        nodes = node_mask_from_edge_mask(unique_edges, mask, num_nodes=num_nodes) & normal
        wave_rows.append(
            wave_type_row(
                run_name=run_name,
                trajectory_index=trajectory_index,
                frame=frame,
                label=label,
                edge_mask=mask,
                trace_arrays=trace_arrays,
                diss_mag=diss_mag,
                central_mag=central_mag,
                wave_speed=wave_speed,
                induced_delta=induced_delta,
                true_delta=true_delta,
                node_mask=nodes,
            )
        )

    sample_rows = sampled_edge_rows(
        run_name=run_name,
        trajectory_index=trajectory_index,
        frame=frame,
        unique_edges=unique_edges,
        current=current,
        left=left,
        right=right,
        trace_arrays=trace_arrays,
        shock_scores=target_pressure_jumps,
        shock_edges=shock_edges,
        wave_masks=wave["masks"],
        diss_mag=diss_mag,
        central_mag=central_mag,
        wave_speed=wave_speed,
        edge_factor=edge_factor[: unique_edges.shape[0], 0],
        normals=normals,
        valid_edges=edge_normal_mask,
        max_rows=args.max_sampled_edges,
        seed=args.seed + 1000 * trajectory_index + frame,
    )

    if args.save_full_npz:
        frame_summary["_full_arrays"] = {
            "left_reconstruct_prims": left,
            "right_reconstruct_prims": right,
            "unique_edge_index": unique_edges,
            "edge_unit_normal": normals,
            "edge_factor": edge_factor,
            "llf_flux": decomp["flux"],
            "llf_central": decomp["central"],
            "llf_dissipation": decomp["dissipation"],
            "induced_conservative_delta": induced_delta,
            "true_conservative_delta": true_delta,
            "shock_edges": shock_edges,
            "shock_nodes": shock_nodes,
        }
    return frame_summary, wave_rows, sample_rows


def primitive_rmse(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    if not np.any(mask):
        return by_var(np.full(len(PRIMITIVE_NAMES), np.nan))
    err = prediction[mask] - target[mask]
    return by_var(np.sqrt(np.mean(err * err, axis=0)))


def dissipation_summary(
    diss_mag: np.ndarray,
    central_mag: np.ndarray,
    flux_mag: np.ndarray,
    weighted_msg_mag: np.ndarray,
    wave_speed: np.ndarray,
    *,
    edge_normal_mask: np.ndarray,
    shock_edges: np.ndarray,
) -> dict[str, Any]:
    smooth = edge_normal_mask & ~shock_edges
    top = top_fraction_mask(diss_mag, edge_normal_mask, fraction=0.10)
    return {
        "dissipation_norm_normal": finite_stats(diss_mag[edge_normal_mask]),
        "central_norm_normal": finite_stats(central_mag[edge_normal_mask]),
        "flux_norm_normal": finite_stats(flux_mag[edge_normal_mask]),
        "weighted_message_norm_normal": finite_stats(weighted_msg_mag[edge_normal_mask]),
        "wave_speed_normal": finite_stats(wave_speed[edge_normal_mask]),
        "dissipation_norm_shock": finite_stats(diss_mag[shock_edges]),
        "dissipation_norm_smooth": finite_stats(diss_mag[smooth]),
        "shock_to_smooth_dissipation_mean_ratio": safe_ratio(
            np.mean(diss_mag[shock_edges]) if np.any(shock_edges) else math.nan,
            np.mean(diss_mag[smooth]) if np.any(smooth) else math.nan,
        ),
        "top10_dissipation_edge_shock_fraction": safe_ratio(
            np.count_nonzero(top & shock_edges), np.count_nonzero(top)
        ),
    }


def front_motion_summary(
    current: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    pos: np.ndarray,
    edges: np.ndarray,
    normal: np.ndarray,
    *,
    quantile: float,
) -> dict[str, float]:
    current_front = shock_front_mask_from_scores(
        shock_front_scores(current, edges, scalar_index=3),
        quantile=quantile,
        node_mask=normal,
    )
    pred_front = shock_front_mask_from_scores(
        shock_front_scores(prediction, edges, scalar_index=3),
        quantile=quantile,
        node_mask=normal,
    )
    target_front = shock_front_mask_from_scores(
        shock_front_scores(target, edges, scalar_index=3),
        quantile=quantile,
        node_mask=normal,
    )
    c0 = front_centroid(current_front[None, :], pos).reshape(-1)
    cp = front_centroid(pred_front[None, :], pos).reshape(-1)
    ct = front_centroid(target_front[None, :], pos).reshape(-1)
    if not (np.all(np.isfinite(c0)) and np.all(np.isfinite(cp)) and np.all(np.isfinite(ct))):
        return {
            "target_motion_norm": math.nan,
            "prediction_motion_norm": math.nan,
            "speed_projection_ratio": math.nan,
            "signed_lag_distance": math.nan,
            "prediction_target_front_distance": math.nan,
        }
    target_motion = ct - c0
    pred_motion = cp - c0
    motion_norm = float(np.linalg.norm(target_motion))
    pred_norm = float(np.linalg.norm(pred_motion))
    if motion_norm <= EPS:
        projection = math.nan
        lag = math.nan
    else:
        unit = target_motion / motion_norm
        projection = float(np.dot(pred_motion, target_motion) / (motion_norm**2))
        lag = float(np.dot(ct - cp, unit))
    return {
        "target_motion_norm": motion_norm,
        "prediction_motion_norm": pred_norm,
        "speed_projection_ratio": projection,
        "signed_lag_distance": lag,
        "prediction_target_front_distance": float(np.linalg.norm(cp - ct)),
    }


def wave_type_row(
    *,
    run_name: str,
    trajectory_index: int,
    frame: int,
    label: str,
    edge_mask: np.ndarray,
    trace_arrays: dict[str, np.ndarray],
    diss_mag: np.ndarray,
    central_mag: np.ndarray,
    wave_speed: np.ndarray,
    induced_delta: np.ndarray,
    true_delta: np.ndarray,
    node_mask: np.ndarray,
) -> dict[str, Any]:
    trace = summarize_trace_likeness(trace_arrays, edge_mask=edge_mask)
    update = update_error_summary(induced_delta, true_delta, node_mask=node_mask)
    row: dict[str, Any] = {
        "run": run_name,
        "trajectory_index": int(trajectory_index),
        "frame": int(frame),
        "wave_type": label,
        "edge_count": int(np.count_nonzero(edge_mask)),
        "node_count": int(np.count_nonzero(node_mask)),
        "left_owner_l2_mean": trace["left_owner_l2_mean"],
        "right_owner_l2_mean": trace["right_owner_l2_mean"],
        "dissipation_norm_mean": float(np.mean(diss_mag[edge_mask]))
        if np.any(edge_mask)
        else math.nan,
        "central_norm_mean": float(np.mean(central_mag[edge_mask]))
        if np.any(edge_mask)
        else math.nan,
        "wave_speed_mean": float(np.mean(wave_speed[edge_mask]))
        if np.any(edge_mask)
        else math.nan,
        "update_energy_relative_l2": update["relative_l2"]["energy"],
    }
    for name in CONSERVATIVE_NAMES:
        row[f"update_rmse_{name}"] = update["rmse"][name]
    return row


def sampled_edge_rows(
    *,
    run_name: str,
    trajectory_index: int,
    frame: int,
    unique_edges: np.ndarray,
    current: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    trace_arrays: dict[str, np.ndarray],
    shock_scores: np.ndarray,
    shock_edges: np.ndarray,
    wave_masks: dict[str, np.ndarray],
    diss_mag: np.ndarray,
    central_mag: np.ndarray,
    wave_speed: np.ndarray,
    edge_factor: np.ndarray,
    normals: np.ndarray,
    valid_edges: np.ndarray,
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_rows <= 0 or not np.any(valid_edges):
        return []
    rng = random.Random(seed)
    candidates = set(np.flatnonzero(top_fraction_mask(shock_scores, valid_edges, 0.05)))
    candidates.update(np.flatnonzero(top_fraction_mask(diss_mag, valid_edges, 0.05)))
    remaining = [int(i) for i in np.flatnonzero(valid_edges) if int(i) not in candidates]
    rng.shuffle(remaining)
    for index in remaining[: max(0, max_rows - len(candidates))]:
        candidates.add(index)
    selected = sorted(candidates)[:max_rows]
    labels = edge_labels(wave_masks, unique_edges.shape[0])
    rows = []
    for index in selected:
        src, dst = unique_edges[index]
        row = {
            "run": run_name,
            "trajectory_index": int(trajectory_index),
            "frame": int(frame),
            "edge_index": int(index),
            "src": int(src),
            "dst": int(dst),
            "wave_type": labels[index],
            "is_shock_edge": bool(shock_edges[index]),
            "shock_pressure_jump": float(shock_scores[index]),
            "dissipation_norm": float(diss_mag[index]),
            "central_norm": float(central_mag[index]),
            "wave_speed": float(wave_speed[index]),
            "edge_factor": float(edge_factor[index]),
            "normal_x": float(normals[index, 0]),
            "normal_y": float(normals[index, 1]),
            "left_owner_l2": float(trace_arrays["left_owner_l2"][index]),
            "right_owner_l2": float(trace_arrays["right_owner_l2"][index]),
            "left_rho": float(left[index, 0]),
            "left_pres": float(left[index, 3]),
            "right_rho": float(right[index, 0]),
            "right_pres": float(right[index, 3]),
            "src_rho": float(current[src, 0]),
            "src_pres": float(current[src, 3]),
            "dst_rho": float(current[dst, 0]),
            "dst_pres": float(current[dst, 3]),
        }
        for var_index, name in enumerate(PRIMITIVE_NAMES):
            row[f"left_bounded_{name}"] = bool(
                trace_arrays["left_bounded"][index, var_index]
            )
            row[f"right_bounded_{name}"] = bool(
                trace_arrays["right_bounded"][index, var_index]
            )
        rows.append(row)
    return rows


def edge_labels(wave_masks: dict[str, np.ndarray], edge_count: int) -> list[str]:
    labels = ["unclassified"] * edge_count
    for label in WAVE_TYPE_ORDER:
        for index in np.flatnonzero(wave_masks[label]):
            labels[int(index)] = label
    return labels


def top_fraction_mask(values: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=bool)
    if not np.any(valid):
        return out
    count = max(1, int(math.ceil(float(fraction) * np.count_nonzero(valid))))
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(values[valid_indices])[-count:]]
    out[order] = True
    return out


def safe_ratio(num: float | int, den: float | int) -> float:
    num_f = float(num)
    den_f = float(den)
    if not np.isfinite(num_f) or not np.isfinite(den_f) or abs(den_f) <= EPS:
        return math.nan
    return num_f / den_f


def flatten_frame_row(frame_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run": frame_summary["run"],
        "trajectory_index": frame_summary["trajectory_index"],
        "frame": frame_summary["frame"],
        "node_count": frame_summary["node_count"],
        "normal_node_count": frame_summary["normal_node_count"],
        "unique_edge_count": frame_summary["unique_edge_count"],
        "normal_normal_edge_count": frame_summary["normal_normal_edge_count"],
        "shock_edge_count": frame_summary["shock_edge_count"],
        "shock_node_count": frame_summary["shock_node_count"],
    }
    for name in PRIMITIVE_NAMES:
        row[f"state_rmse_{name}"] = frame_summary["state_rmse_normal"][name]
    for region in ("all_normal_nodes", "shock_nodes", "smooth_nodes"):
        update = frame_summary[f"update_match_{region}"]
        for name in CONSERVATIVE_NAMES:
            row[f"{region}_update_rel_l2_{name}"] = update["relative_l2"][name]
    for key, value in frame_summary["front_motion"].items():
        row[f"front_{key}"] = value
    diss = frame_summary["dissipation"]
    row["shock_to_smooth_dissipation_mean_ratio"] = diss[
        "shock_to_smooth_dissipation_mean_ratio"
    ]
    row["top10_dissipation_edge_shock_fraction"] = diss[
        "top10_dissipation_edge_shock_fraction"
    ]
    trace = frame_summary["trace_all_normal_edges"]
    row["trace_left_owner_closer_fraction"] = trace["left_owner_closer_fraction"]
    row["trace_right_owner_closer_fraction"] = trace["right_owner_closer_fraction"]
    row["trace_left_owner_l2_mean"] = trace["left_owner_l2_mean"]
    row["trace_right_owner_l2_mean"] = trace["right_owner_l2_mean"]
    return row


def strip_large_arrays(frame_summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in frame_summary.items() if key != "_full_arrays"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CPGNet Interface Latent Diagnostic",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        f"- Runs: {', '.join(summary['runs'].keys())}",
        f"- Trajectories: {summary['trajectory_indices']}",
        f"- Frames: {summary['frames']}",
        f"- Boundary mode: `{summary['boundary_mode']}`",
        "",
        "## Frame Averages",
        "| Run | state pres RMSE | update energy rel L2 | shock/smooth diss | "
        "top-diss shock frac | speed ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name, aggregate in summary["run_aggregates"].items():
        lines.append(
            f"| {run_name} | "
            f"{aggregate['state_rmse_pres_mean']:.6g} | "
            f"{aggregate['update_energy_relative_l2_mean']:.6g} | "
            f"{aggregate['shock_to_smooth_dissipation_ratio_mean']:.6g} | "
            f"{aggregate['top10_dissipation_shock_fraction_mean']:.6g} | "
            f"{aggregate['speed_projection_ratio_mean']:.6g} |"
        )
    lines += [
        "",
        "## Interpretation Checklist",
        "- Trace-likeness: compare owner-closer fractions, bounded fractions, "
        "and overshoot statistics in `summary.json`.",
        "- Flux/update match: compare induced conservative-update relative L2 "
        "against normal-node state RMSE.",
        "- Dissipation localization: shock/smooth dissipation ratio and "
        "top-dissipation shock fraction indicate whether damping follows target fronts.",
        "- Wave-speed: speed projection ratio below 1 suggests lag/too-slow "
        "front motion; above 1 suggests over-advance.",
        "",
        "## Files",
        "- `summary.json`: nested per-frame diagnostics.",
        "- `per_frame_summary.csv`: compact frame-level table.",
        "- `per_wave_type_summary.csv`: smooth/compression/rarefaction/contact/shock strata.",
        "- `sampled_edges.csv`: small edge table for manual inspection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run"]), []).append(row)
    out = {}
    for run_name, items in grouped.items():
        out[run_name] = {
            "frame_count": int(len(items)),
            "state_rmse_pres_mean": mean_field(items, "state_rmse_pres"),
            "update_energy_relative_l2_mean": mean_field(
                items, "all_normal_nodes_update_rel_l2_energy"
            ),
            "shock_to_smooth_dissipation_ratio_mean": mean_field(
                items, "shock_to_smooth_dissipation_mean_ratio"
            ),
            "top10_dissipation_shock_fraction_mean": mean_field(
                items, "top10_dissipation_edge_shock_fraction"
            ),
            "speed_projection_ratio_mean": mean_field(
                items, "front_speed_projection_ratio"
            ),
        }
    return out


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    values = np.array([float(row[field]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.name
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_npz_arrays(output_dir: Path, frame_summary: dict[str, Any]) -> None:
    arrays = frame_summary.get("_full_arrays")
    if not arrays:
        return
    name = (
        f"{frame_summary['run']}_traj{frame_summary['trajectory_index']:02d}_"
        f"frame{frame_summary['frame']:04d}.npz"
    )
    np.savez_compressed(output_dir / name, **arrays)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    import torch_geometric.transforms as T
    from torch_geometric.loader import DataLoader

    from dataset.fpcMulti import FPC_ROLLOUT
    from utils.utils import NodeType

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args)
    dataset = FPC_ROLLOUT(str(args.dataset_root), split="test")
    loader = DataLoader(dataset=dataset, batch_size=1)
    transformer = T.Compose(
        [T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)]
    )

    frame_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    frame_summaries: dict[str, list[dict[str, Any]]] = {}
    run_metadata: dict[str, Any] = {}
    start = time.time()

    for raw_run in args.run:
        run_name, run_path = parse_named_path(raw_run)
        checkpoint = resolve_checkpoint(run_path)
        run_metadata[run_name] = {
            "checkpoint_name": checkpoint.name,
            "checkpoint_exists": bool(checkpoint.exists()),
        }
        model = make_model(checkpoint, device)
        capture = ModelCapture(
            model,
            capture_decoder_logits=not args.no_capture_decoder_logits,
            capture_hidden_edge=args.capture_hidden_edge,
        )
        try:
            frame_summaries[run_name] = []
            for trajectory_index in args.trajectory_indices:
                graphs = load_graphs_for_trajectory(
                    dataset, loader, transformer, trajectory_index, args.frames
                )
                for frame in args.frames:
                    arrays = run_frame(
                        model,
                        capture,
                        graphs[frame],
                        device=device,
                        node_type_normal_value=int(NodeType.NORMAL),
                        boundary_mode=args.boundary_mode,
                    )
                    frame_summary, current_wave_rows, current_sample_rows = (
                        frame_diagnostics(
                            arrays,
                            run_name=run_name,
                            trajectory_index=trajectory_index,
                            frame=frame,
                            args=args,
                        )
                    )
                    save_npz_arrays(args.output_dir, frame_summary)
                    frame_rows.append(flatten_frame_row(frame_summary))
                    wave_rows.extend(current_wave_rows)
                    sample_rows.extend(current_sample_rows)
                    frame_summaries[run_name].append(strip_large_arrays(frame_summary))
        finally:
            capture.close()

    summary = {
        "mode": "cpg_interface_latent_diagnostic",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(time.time() - start),
        "device": str(device),
        "boundary_mode": args.boundary_mode,
        "trajectory_indices": [int(value) for value in args.trajectory_indices],
        "frames": [int(value) for value in args.frames],
        "shock_quantile": float(args.shock_quantile),
        "wave_high_quantile": float(args.wave_high_quantile),
        "wave_smooth_quantile": float(args.wave_smooth_quantile),
        "runs": run_metadata,
        "run_aggregates": aggregate_rows(frame_rows),
        "frame_summaries": frame_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "per_frame_summary.csv", frame_rows)
    write_csv(args.output_dir / "per_wave_type_summary.csv", wave_rows)
    write_csv(args.output_dir / "sampled_edges.csv", sample_rows)
    write_report(args.output_dir / "report.md", summary)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "report": str(args.output_dir / "report.md"),
                "per_frame": str(args.output_dir / "per_frame_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
