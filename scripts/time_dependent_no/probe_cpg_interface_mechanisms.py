
"""Probe CPGNet interface latents as physical traces or flux-control states."""

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

from scripts.time_dependent_no.dump_cpg_interface_latents import (  # noqa: E402
    ModelCapture,
    dissipation_summary,
    make_model,
    parse_named_path,
    primitive_rmse,
    resolve_checkpoint,
    run_frame,
    select_device,
    tensor_to_numpy,
)
from utility.time_dependent_no.cpg_interface_latents import (  # noqa: E402
    clip_reconstruct_to_edge_box,
    conservative_update_from_flux,
    divergence_cancellation_summary,
    edge_pressure_jump_scores,
    edge_shock_mask_from_scores,
    flux_error_summary,
    llf_flux_decomposition,
    midpoint_reconstruct_prims,
    node_mask_from_edge_mask,
    owner_neighbor_reconstruct_prims,
    split_directed_reconstruct_prims,
    summarize_trace_likeness,
    time_midpoint_reconstruct_prims,
    trace_likeness_arrays,
    update_error_summary,
)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    CONSERVATIVE_NAMES,
    PRIMITIVE_NAMES,
    primitive_to_conservative,
)

EPS = 1.0e-12
SOURCE_MODES = ("teacher_forced", "autoregressive")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir")
    parser.add_argument("--output-dir", type=Path, default=Path(f"artifacts/time_dependent_no/cpg_interface_mechanism_probe_{date}"))
    parser.add_argument("--trajectory-indices", nargs="+", type=int, default=[0, 6, 11, 13, 17])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 20, 40, 58, 78])
    parser.add_argument("--source-modes", nargs="+", choices=SOURCE_MODES, default=list(SOURCE_MODES))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--boundary-mode", choices=["official-clamped", "current"], default="official-clamped")
    parser.add_argument("--shock-quantile", type=float, default=0.90)
    parser.add_argument("--projection-margin-factor", type=float, default=1.0)
    parser.add_argument("--projection-margin-abs", type=float, default=0.0)
    parser.add_argument("--inverse-sample-edges", type=int, default=128)
    parser.add_argument("--inverse-steps", type=int, default=160)
    parser.add_argument("--inverse-lr", type=float, default=0.08)
    parser.add_argument("--inverse-margin-factors", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--inverse-margin-abs", type=float, default=0.01)
    parser.add_argument("--no-inverse-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_graph_sequence(dataset: Any, loader: Any, transformer: Any, trajectory_index: int, max_frame: int) -> list[Any]:
    from utils.to_undirected import make_edges_undirected

    dataset.change_file(trajectory_index)
    graphs = []
    for step, graph in enumerate(loader):
        if step > max_frame:
            break
        graphs.append(transformer(make_edges_undirected(graph)).cpu())
    if len(graphs) <= max_frame:
        raise RuntimeError(f"trajectory {trajectory_index} yielded {len(graphs)} frames, need {max_frame}")
    return graphs


def run_model_frame(model: Any, capture: ModelCapture, graph: Any, *, device: Any, node_type_normal_value: int, boundary_mode: str, state_override: np.ndarray | None = None) -> dict[str, Any]:
    import torch

    graph = graph.clone()
    truth_current = tensor_to_numpy(graph.x[:, 1:5])
    if state_override is not None:
        override = torch.as_tensor(state_override, dtype=graph.x.dtype)
        if override.shape != graph.x[:, 1:5].shape:
            raise ValueError("state_override shape must match graph primitive state")
        graph.x[:, 1:5] = override
    arrays = run_frame(model, capture, graph, device=device, node_type_normal_value=node_type_normal_value, boundary_mode=boundary_mode)
    arrays["truth_current_primitives"] = truth_current
    return arrays


def collect_source_frames(model: Any, capture: ModelCapture, graphs: list[Any], frames: Sequence[int], *, source_mode: str, device: Any, node_type_normal_value: int, boundary_mode: str) -> dict[int, dict[str, Any]]:
    wanted = set(int(frame) for frame in frames)
    out: dict[int, dict[str, Any]] = {}
    if source_mode == "teacher_forced":
        for frame in sorted(wanted):
            out[frame] = run_model_frame(model, capture, graphs[frame], device=device, node_type_normal_value=node_type_normal_value, boundary_mode=boundary_mode)
        return out
    predicted_prims: np.ndarray | None = None
    for frame in range(max(wanted) + 1):
        arrays = run_model_frame(model, capture, graphs[frame], device=device, node_type_normal_value=node_type_normal_value, boundary_mode=boundary_mode, state_override=predicted_prims)
        predicted_prims = np.asarray(arrays["prediction_clamped_primitives"], dtype=np.float64)
        if frame in wanted:
            out[frame] = arrays
    return out


def frame_diagnostics(arrays: dict[str, Any], *, run_name: str, source_mode: str, trajectory_index: int, frame: int, args: argparse.Namespace, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current = arrays["current_primitives"]
    truth_current = arrays["truth_current_primitives"]
    target = arrays["target_primitives"]
    prediction = arrays["prediction_clamped_primitives"]
    normal = arrays["normal_mask"]
    unique_edges = arrays["unique_edge_index"]
    full_edges = arrays["full_edge_index"]
    normals = arrays["edge_unit_normal"]
    edge_factor = arrays["edge_factor"]
    edge_normal_mask = normal[unique_edges[:, 0]] & normal[unique_edges[:, 1]]
    left, right = split_directed_reconstruct_prims(arrays["reconstruct_prims"], num_unique_edges=unique_edges.shape[0])
    learned = llf_flux_decomposition(left, right, normals)
    learned_delta = conservative_update_from_flux(learned["flux"], edge_factor, full_edges, num_nodes=current.shape[0])["conservative_delta"]
    true_delta = primitive_to_conservative(target) - primitive_to_conservative(current)
    model_delta = primitive_to_conservative(prediction) - primitive_to_conservative(current)
    shock_scores = edge_pressure_jump_scores(target, unique_edges)
    shock_edges = edge_shock_mask_from_scores(shock_scores, quantile=args.shock_quantile, edge_mask=edge_normal_mask)
    shock_nodes = node_mask_from_edge_mask(unique_edges, shock_edges, num_nodes=current.shape[0]) & normal
    smooth_nodes = normal & ~shock_nodes
    trace_arrays = trace_likeness_arrays(left, right, current, unique_edges)
    trace = summarize_trace_likeness(trace_arrays, edge_mask=edge_normal_mask)
    learned_true = update_error_summary(learned_delta, true_delta, node_mask=normal)
    learned_shock = update_error_summary(learned_delta, true_delta, node_mask=shock_nodes)
    learned_smooth = update_error_summary(learned_delta, true_delta, node_mask=smooth_nodes)
    learned_model = update_error_summary(learned_delta, model_delta, node_mask=normal)
    messages = conservative_update_from_flux(learned["flux"], edge_factor, full_edges, num_nodes=current.shape[0])["messages"][: unique_edges.shape[0]]
    diss = dissipation_summary(
        np.linalg.norm(learned["dissipation"], axis=-1),
        np.linalg.norm(learned["central"], axis=-1),
        np.linalg.norm(learned["flux"], axis=-1),
        np.linalg.norm(messages, axis=-1),
        learned["wave_speed"],
        edge_normal_mask=edge_normal_mask,
        shock_edges=shock_edges,
    )
    row: dict[str, Any] = {
        "run": run_name,
        "source_mode": source_mode,
        "trajectory_index": int(trajectory_index),
        "frame": int(frame),
        "normal_node_count": int(np.count_nonzero(normal)),
        "normal_normal_edge_count": int(np.count_nonzero(edge_normal_mask)),
        "shock_edge_count": int(np.count_nonzero(shock_edges)),
        "shock_node_count": int(np.count_nonzero(shock_nodes)),
        "trace_left_owner_closer_fraction": trace["left_owner_closer_fraction"],
        "trace_right_owner_closer_fraction": trace["right_owner_closer_fraction"],
        "shock_to_smooth_dissipation_mean_ratio": diss["shock_to_smooth_dissipation_mean_ratio"],
        "top10_dissipation_edge_shock_fraction": diss["top10_dissipation_edge_shock_fraction"],
    }
    add_vars(row, "input_rmse_to_truth", primitive_rmse(current, truth_current, normal))
    add_vars(row, "state_rmse", primitive_rmse(prediction, target, normal))
    for name in PRIMITIVE_NAMES:
        row[f"trace_bounded_fraction_{name}"] = trace["bounded_fraction"][name]
        row[f"trace_overshoot_mean_{name}"] = trace["overshoot_mean"][name]
    for name in CONSERVATIVE_NAMES:
        row[f"learned_true_rel_l2_{name}"] = learned_true["relative_l2"][name]
        row[f"learned_shock_rel_l2_{name}"] = learned_shock["relative_l2"][name]
        row[f"learned_smooth_rel_l2_{name}"] = learned_smooth["relative_l2"][name]
        row[f"learned_model_rel_l2_{name}"] = learned_model["relative_l2"][name]
    proj = projection_rows(
        run_name=run_name,
        source_mode=source_mode,
        trajectory_index=trajectory_index,
        frame=frame,
        current=current,
        target=target,
        left=left,
        right=right,
        normals=normals,
        unique_edges=unique_edges,
        full_edges=full_edges,
        edge_factor=edge_factor,
        normal=normal,
        edge_normal_mask=edge_normal_mask,
        shock_edges=shock_edges,
        shock_nodes=shock_nodes,
        true_delta=true_delta,
        model_delta=model_delta,
        learned_flux=learned["flux"],
        learned_delta=learned_delta,
        args=args,
    )
    inv: list[dict[str, Any]] = []
    if not args.no_inverse_fit and args.inverse_sample_edges > 0:
        inv = inverse_rows(
            run_name=run_name,
            source_mode=source_mode,
            trajectory_index=trajectory_index,
            frame=frame,
            current=current,
            left=left,
            right=right,
            normals=normals,
            unique_edges=unique_edges,
            learned_flux=learned["flux"],
            edge_normal_mask=edge_normal_mask,
            shock_edges=shock_edges,
            shock_scores=shock_scores,
            args=args,
            device=device,
        )
    return row, proj, inv


def projection_rows(**kw: Any) -> list[dict[str, Any]]:
    current = kw["current"]
    target = kw["target"]
    left = kw["left"]
    right = kw["right"]
    edges = kw["unique_edges"]
    args = kw["args"]
    owner_left, owner_right = owner_neighbor_reconstruct_prims(current, edges)
    mid_left, mid_right = midpoint_reconstruct_prims(current, edges)
    target_left, target_right = owner_neighbor_reconstruct_prims(target, edges)
    time_left, time_right = time_midpoint_reconstruct_prims(current, target, edges)
    clip_left, clip_right = clip_reconstruct_to_edge_box(left, right, current, edges)
    exp_left, exp_right = clip_reconstruct_to_edge_box(left, right, current, edges, margin_factor=args.projection_margin_factor, margin_abs=args.projection_margin_abs)
    candidates = {
        "node_pair": (owner_left, owner_right),
        "midpoint": (mid_left, mid_right),
        "time_midpoint": (time_left, time_right),
        "target_pair": (target_left, target_right),
        "bounded_clip": (clip_left, clip_right),
        "expanded_clip": (exp_left, exp_right),
    }
    rows = []
    for name, (cl, cr) in candidates.items():
        decomp = llf_flux_decomposition(cl, cr, kw["normals"])
        fv = conservative_update_from_flux(decomp["flux"], kw["edge_factor"], kw["full_edges"], num_nodes=current.shape[0])
        delta = fv["conservative_delta"]
        update_model = update_error_summary(delta, kw["model_delta"], node_mask=kw["normal"])
        update_true = update_error_summary(delta, kw["true_delta"], node_mask=kw["normal"])
        update_learned = update_error_summary(delta, kw["learned_delta"], node_mask=kw["normal"])
        update_shock = update_error_summary(delta, kw["true_delta"], node_mask=kw["shock_nodes"])
        flux_all = flux_error_summary(decomp["flux"], kw["learned_flux"], edge_mask=kw["edge_normal_mask"])
        flux_shock = flux_error_summary(decomp["flux"], kw["learned_flux"], edge_mask=kw["shock_edges"])
        flux_smooth = flux_error_summary(decomp["flux"], kw["learned_flux"], edge_mask=kw["edge_normal_mask"] & ~kw["shock_edges"])
        cancel = divergence_cancellation_summary(decomp["flux"], kw["learned_flux"], kw["edge_factor"], kw["full_edges"], num_nodes=current.shape[0], node_mask=kw["normal"])
        trace = summarize_trace_likeness(trace_likeness_arrays(cl, cr, current, edges), edge_mask=kw["edge_normal_mask"])
        state_delta = np.concatenate((cl - left, cr - right), axis=0)
        row = {
            "run": kw["run_name"],
            "source_mode": kw["source_mode"],
            "trajectory_index": int(kw["trajectory_index"]),
            "frame": int(kw["frame"]),
            "candidate": name,
            "flux_rel_l2_normal": flux_all["edge_relative_l2"],
            "flux_rel_l2_shock": flux_shock["edge_relative_l2"],
            "flux_rel_l2_smooth": flux_smooth["edge_relative_l2"],
            "flux_cosine_normal": flux_all["edge_cosine_mean"],
            "update_vs_model_energy_rel_l2": update_model["relative_l2"]["energy"],
            "update_vs_true_energy_rel_l2": update_true["relative_l2"]["energy"],
            "update_vs_learned_energy_rel_l2": update_learned["relative_l2"]["energy"],
            "shock_update_vs_true_energy_rel_l2": update_shock["relative_l2"]["energy"],
            "divergence_update_to_message_l2_ratio": cancel["update_to_message_l2_ratio"],
            "state_l2_to_learned_mean": finite_mean(np.linalg.norm(state_delta, axis=-1)),
            "trace_pressure_bounded_fraction": trace["bounded_fraction"]["pres"],
            "trace_pressure_overshoot_mean": trace["overshoot_mean"]["pres"],
        }
        for cons in CONSERVATIVE_NAMES:
            row[f"update_vs_true_rel_l2_{cons}"] = update_true["relative_l2"][cons]
            row[f"update_vs_model_rel_l2_{cons}"] = update_model["relative_l2"][cons]
        rows.append(row)
    return rows


def inverse_rows(**kw: Any) -> list[dict[str, Any]]:
    selected = select_inverse_edges(
        valid_edges=kw["edge_normal_mask"],
        shock_edges=kw["shock_edges"],
        shock_scores=kw["shock_scores"],
        max_edges=kw["args"].inverse_sample_edges,
        seed=kw["args"].seed + 9176 * int(kw["trajectory_index"]) + int(kw["frame"]),
    )
    rows = []
    for margin in kw["args"].inverse_margin_factors:
        fit = fit_physical_flux_inverse(
            current=kw["current"],
            left=kw["left"],
            right=kw["right"],
            unique_edges=kw["unique_edges"],
            normals=kw["normals"],
            learned_flux=kw["learned_flux"],
            selected_edges=selected,
            shock_edges=kw["shock_edges"],
            margin_factor=float(margin),
            margin_abs=float(kw["args"].inverse_margin_abs),
            steps=int(kw["args"].inverse_steps),
            lr=float(kw["args"].inverse_lr),
            device=kw["device"],
        )
        rows.append({
            "run": kw["run_name"],
            "source_mode": kw["source_mode"],
            "trajectory_index": int(kw["trajectory_index"]),
            "frame": int(kw["frame"]),
            "margin_factor": float(margin),
            **fit,
        })
    return rows


def select_inverse_edges(*, valid_edges: np.ndarray, shock_edges: np.ndarray, shock_scores: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    if max_edges <= 0 or not np.any(valid_edges):
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    shock = np.flatnonzero(shock_edges)
    smooth = np.flatnonzero(valid_edges & ~shock_edges)
    if shock.size:
        shock = shock[np.argsort(shock_scores[shock])[::-1]]
    shock_take = min(shock.size, max_edges // 2)
    selected = list(map(int, shock[:shock_take]))
    remaining = max_edges - len(selected)
    if remaining > 0 and smooth.size:
        selected.extend(map(int, rng.permutation(smooth)[:remaining]))
    if len(selected) < max_edges:
        rest = [int(i) for i in np.flatnonzero(valid_edges) if int(i) not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: max_edges - len(selected)])
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def fit_physical_flux_inverse(*, current: np.ndarray, left: np.ndarray, right: np.ndarray, unique_edges: np.ndarray, normals: np.ndarray, learned_flux: np.ndarray, selected_edges: np.ndarray, shock_edges: np.ndarray, margin_factor: float, margin_abs: float, steps: int, lr: float, device: Any) -> dict[str, Any]:
    import torch

    if selected_edges.size == 0:
        return empty_inverse_summary()
    src = unique_edges[selected_edges, 0]
    dst = unique_edges[selected_edges, 1]
    owner_left = current[dst]
    owner_right = current[src]
    lo = np.minimum(owner_left, owner_right)
    hi = np.maximum(owner_left, owner_right)
    width = hi - lo
    abs_scale = np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
    margin = margin_factor * width + margin_abs * abs_scale
    lo = lo - margin
    hi = hi + margin
    lo[:, 0] = np.maximum(lo[:, 0], 1.0e-6)
    lo[:, 3] = np.maximum(lo[:, 3], 1.0e-6)
    hi[:, 0] = np.maximum(hi[:, 0], lo[:, 0] + 1.0e-8)
    hi[:, 3] = np.maximum(hi[:, 3], lo[:, 3] + 1.0e-8)
    low = np.stack((lo, lo), axis=1)
    high = np.stack((hi, hi), axis=1)
    init = np.stack((left[selected_edges], right[selected_edges]), axis=1)
    clipped = np.clip(init, low, high)
    denom = np.maximum(high - low, 1.0e-8)
    u = np.clip((clipped - low) / denom, 1.0e-4, 1.0 - 1.0e-4)
    z0 = np.log(u / (1.0 - u))
    target_flux = torch.as_tensor(learned_flux[selected_edges], dtype=torch.float64, device=device)
    normal_t = torch.as_tensor(normals[selected_edges], dtype=torch.float64, device=device)
    low_t = torch.as_tensor(low, dtype=torch.float64, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float64, device=device)
    z = torch.tensor(z0, dtype=torch.float64, device=device, requires_grad=True)
    rms = torch.sqrt(torch.mean(target_flux * target_flux, dim=0)).clamp_min(1.0e-6)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(max(0, steps)):
        opt.zero_grad(set_to_none=True)
        state = low_t + torch.sigmoid(z) * (high_t - low_t)
        flux = torch_llf_flux(state[:, 0, :], state[:, 1, :], normal_t)
        loss = torch.mean(((flux - target_flux) / rms) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        state = low_t + torch.sigmoid(z) * (high_t - low_t)
        flux = torch_llf_flux(state[:, 0, :], state[:, 1, :], normal_t)
        rel = torch.linalg.norm(flux - target_flux, dim=1) / torch.linalg.norm(target_flux, dim=1).clamp_min(EPS)
        fitted = state.detach().cpu().numpy()
        rel_np = rel.detach().cpu().numpy().astype(np.float64)
    selected_shock = shock_edges[selected_edges]
    owner_dist = np.concatenate((np.linalg.norm(fitted[:, 0, :] - owner_left, axis=-1), np.linalg.norm(fitted[:, 1, :] - owner_right, axis=-1)))
    learned_dist = np.concatenate((np.linalg.norm(fitted[:, 0, :] - left[selected_edges], axis=-1), np.linalg.norm(fitted[:, 1, :] - right[selected_edges], axis=-1)))
    return {
        "sample_count": int(selected_edges.size),
        "shock_sample_count": int(np.count_nonzero(selected_shock)),
        "smooth_sample_count": int(np.count_nonzero(~selected_shock)),
        "rel_l2_mean": finite_mean(rel_np),
        "rel_l2_median": finite_percentile(rel_np, 50),
        "rel_l2_p90": finite_percentile(rel_np, 90),
        "rel_l2_max": finite_percentile(rel_np, 100),
        "shock_rel_l2_mean": finite_mean(rel_np[selected_shock]),
        "smooth_rel_l2_mean": finite_mean(rel_np[~selected_shock]),
        "success_fraction_rel_l2_lt_0p05": finite_mean(rel_np < 0.05),
        "success_fraction_rel_l2_lt_0p10": finite_mean(rel_np < 0.10),
        "fitted_owner_l2_mean": finite_mean(owner_dist),
        "fitted_to_learned_state_l2_mean": finite_mean(learned_dist),
    }


def torch_llf_flux(left: Any, right: Any, normals: Any, gamma: float = 1.4) -> Any:
    import torch

    def prim_to_cons(prim: Any) -> Any:
        rho, v1, v2, pres = prim.unbind(-1)
        energy = pres / (gamma - 1.0) + 0.5 * rho * (v1 * v1 + v2 * v2)
        return torch.stack((rho, rho * v1, rho * v2, energy), dim=-1)

    def prim_flux(prim: Any) -> Any:
        rho, v1, v2, pres = prim.unbind(-1)
        nx, ny = normals.unbind(-1)
        vn = v1 * nx + v2 * ny
        rho_vn = rho * vn
        energy = pres / (gamma - 1.0) + 0.5 * rho * (v1 * v1 + v2 * v2)
        return torch.stack((rho_vn, rho_vn * v1 + pres * nx, rho_vn * v2 + pres * ny, (energy + pres) * vn), dim=-1)

    cons_left = prim_to_cons(left)
    cons_right = prim_to_cons(right)
    central = 0.5 * (prim_flux(left) + prim_flux(right))
    nx, ny = normals.unbind(-1)
    vn_left = left[:, 1] * nx + left[:, 2] * ny
    vn_right = right[:, 1] * nx + right[:, 2] * ny
    c_left = torch.sqrt(gamma * left[:, 3] / left[:, 0])
    c_right = torch.sqrt(gamma * right[:, 3] / right[:, 0])
    lam = torch.maximum(torch.abs(vn_left), torch.abs(vn_right)) + torch.maximum(c_left, c_right) * torch.linalg.norm(normals, dim=-1)
    return central - 0.5 * lam[:, None] * (cons_right - cons_left)


def add_vars(row: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    for name in PRIMITIVE_NAMES:
        row[f"{prefix}_{name}"] = values[name]


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def finite_percentile(values: Any, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else math.nan


def empty_inverse_summary() -> dict[str, Any]:
    keys = [
        "rel_l2_mean",
        "rel_l2_median",
        "rel_l2_p90",
        "rel_l2_max",
        "shock_rel_l2_mean",
        "smooth_rel_l2_mean",
        "success_fraction_rel_l2_lt_0p05",
        "success_fraction_rel_l2_lt_0p10",
        "fitted_owner_l2_mean",
        "fitted_to_learned_state_l2_mean",
    ]
    out = {key: math.nan for key in keys}
    out.update({"sample_count": 0, "shock_sample_count": 0, "smooth_sample_count": 0})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by(rows: list[dict[str, Any]], keys: Sequence[str], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for key_values, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, key_values, strict=True)}
        row["count"] = int(len(items))
        for field in fields:
            values = [item.get(field, math.nan) for item in items]
            row[f"{field}_mean"] = finite_mean(values)
            row[f"{field}_median"] = finite_percentile(values, 50)
        out.append(row)
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CPG Interface Mechanism Probe",
        "",
        f"Generated: {summary['generated_at']}",
        f"Elapsed seconds: {summary['elapsed_seconds']:.3f}",
        "",
        "## Learned Latent Frame Means",
        "| Run | Mode | input pres RMSE | learned energy rel L2 | pressure bounded | shock/smooth diss |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["frame_aggregate"]:
        lines.append(
            f"| {row['run']} | {row['source_mode']} | {row['input_rmse_to_truth_pres_mean']:.6g} | "
            f"{row['learned_true_rel_l2_energy_mean']:.6g} | {row['trace_bounded_fraction_pres_mean']:.6g} | "
            f"{row['shock_to_smooth_dissipation_mean_ratio_mean']:.6g} |"
        )
    lines += [
        "",
        "## Projection Means",
        "| Run | Mode | Candidate | flux rel L2 | update-vs-model energy | update-vs-true energy |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["projection_aggregate"]:
        lines.append(
            f"| {row['run']} | {row['source_mode']} | {row['candidate']} | "
            f"{row['flux_rel_l2_normal_mean']:.6g} | {row['update_vs_model_energy_rel_l2_mean']:.6g} | "
            f"{row['update_vs_true_energy_rel_l2_mean']:.6g} |"
        )
    lines += [
        "",
        "## Inverse Fit Means",
        "| Run | Mode | margin | rel L2 median | rel L2 p90 | <10% frac | shock rel L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["inverse_aggregate"]:
        lines.append(
            f"| {row['run']} | {row['source_mode']} | {row['margin_factor']} | "
            f"{row['rel_l2_median_mean']:.6g} | {row['rel_l2_p90_mean']:.6g} | "
            f"{row['success_fraction_rel_l2_lt_0p10_mean']:.6g} | {row['shock_rel_l2_mean_mean']:.6g} |"
        )
    lines += [
        "",
        "## Files",
        "- `summary.json`: run metadata and aggregate tables.",
        "- `per_frame_summary.csv`: learned-latent metrics by frame/source mode.",
        "- `projection_summary.csv`: physical replacement/projection tests.",
        "- `inverse_fit_summary.csv`: sampled constrained inverse flux fits.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.loader import DataLoader

    from dataset.fpcMulti import FPC_ROLLOUT
    from utils.utils import NodeType

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args)
    dataset = FPC_ROLLOUT(str(args.dataset_root), split="test")
    loader = DataLoader(dataset=dataset, batch_size=1)
    transformer = T.Compose([T.NormalizeScale(), T.Cartesian(norm=False), T.Distance(norm=False)])
    frame_rows: list[dict[str, Any]] = []
    proj_rows: list[dict[str, Any]] = []
    inv_rows: list[dict[str, Any]] = []
    run_meta: dict[str, Any] = {}
    start = time.time()
    max_frame = max(args.frames)

    for raw_run in args.run:
        run_name, run_path = parse_named_path(raw_run)
        checkpoint = resolve_checkpoint(run_path)
        run_meta[run_name] = {"checkpoint_name": checkpoint.name, "checkpoint_exists": bool(checkpoint.exists())}
        model = make_model(checkpoint, device)
        capture = ModelCapture(model, capture_decoder_logits=False, capture_hidden_edge=False)
        try:
            for trajectory_index in args.trajectory_indices:
                graphs = load_graph_sequence(dataset, loader, transformer, trajectory_index, max_frame)
                for source_mode in args.source_modes:
                    source_frames = collect_source_frames(
                        model,
                        capture,
                        graphs,
                        args.frames,
                        source_mode=source_mode,
                        device=device,
                        node_type_normal_value=int(NodeType.NORMAL),
                        boundary_mode=args.boundary_mode,
                    )
                    for frame in args.frames:
                        row, proj, inv = frame_diagnostics(
                            source_frames[frame],
                            run_name=run_name,
                            source_mode=source_mode,
                            trajectory_index=trajectory_index,
                            frame=frame,
                            args=args,
                            device=device,
                        )
                        frame_rows.append(row)
                        proj_rows.extend(proj)
                        inv_rows.extend(inv)
        finally:
            capture.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame_fields = [
        "input_rmse_to_truth_pres",
        "state_rmse_pres",
        "learned_true_rel_l2_energy",
        "learned_shock_rel_l2_energy",
        "learned_model_rel_l2_energy",
        "trace_bounded_fraction_pres",
        "trace_overshoot_mean_pres",
        "shock_to_smooth_dissipation_mean_ratio",
    ]
    proj_fields = [
        "flux_rel_l2_normal",
        "flux_rel_l2_shock",
        "update_vs_model_energy_rel_l2",
        "update_vs_true_energy_rel_l2",
        "update_vs_learned_energy_rel_l2",
        "divergence_update_to_message_l2_ratio",
    ]
    inv_fields = [
        "rel_l2_mean",
        "rel_l2_median",
        "rel_l2_p90",
        "shock_rel_l2_mean",
        "smooth_rel_l2_mean",
        "success_fraction_rel_l2_lt_0p10",
    ]
    summary = {
        "mode": "cpg_interface_mechanism_probe",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(time.time() - start),
        "device": str(device),
        "boundary_mode": args.boundary_mode,
        "trajectory_indices": [int(value) for value in args.trajectory_indices],
        "frames": [int(value) for value in args.frames],
        "source_modes": list(args.source_modes),
        "shock_quantile": float(args.shock_quantile),
        "inverse_sample_edges": int(args.inverse_sample_edges),
        "inverse_steps": int(args.inverse_steps),
        "runs": run_meta,
        "frame_aggregate": aggregate_by(frame_rows, ["run", "source_mode"], frame_fields),
        "projection_aggregate": aggregate_by(proj_rows, ["run", "source_mode", "candidate"], proj_fields),
        "inverse_aggregate": aggregate_by(inv_rows, ["run", "source_mode", "margin_factor"], inv_fields),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_dir / "per_frame_summary.csv", frame_rows)
    write_csv(args.output_dir / "projection_summary.csv", proj_rows)
    write_csv(args.output_dir / "inverse_fit_summary.csv", inv_rows)
    write_report(args.output_dir / "report.md", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "report": str(args.output_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
