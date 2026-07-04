"""Mechanistic diagnostics for CPG-style Euler rollout artifacts.

Consumes rollout HDF5 arrays named ``predicteds`` and ``targets`` plus geometry
(``pos``, ``edges``, ``node_type``) either in the same file or in a CPG test.h5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from utility.time_dependent_no.euler2d import (  # noqa: E402
    CONSERVATIVE_NAMES,
    PRIMITIVE_NAMES,
    EulerNodeType,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    front_region_masks,
    local_shift_alignment_metrics,
    median_edge_length,
    shift_grid,
    shock_front_masks,
)

EPS = 1.0e-12


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        return name.strip(), Path(value)
    path = Path(raw)
    return path.name, path


def by_var(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[i]) for i, name in enumerate(PRIMITIVE_NAMES)}


def by_cons(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(arr[i]) for i, name in enumerate(CONSERVATIVE_NAMES)}


def finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "final": math.nan,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "final": float(arr.reshape(-1)[-1]) if arr.size else math.nan,
    }


def read_frame_or_static(dataset: h5py.Dataset, frame: int) -> np.ndarray:
    if len(dataset.shape) >= 3:
        return np.asarray(dataset[frame])
    return np.asarray(dataset)


def squeeze_node_column(array: np.ndarray) -> np.ndarray:
    if array.ndim > 1 and array.shape[-1] == 1:
        return np.squeeze(array, axis=-1)
    return array


def normalize_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = np.asarray(edges)
    if edge_index.ndim == 3:
        edge_index = edge_index[0]
    if edge_index.shape[0] == 2 and edge_index.ndim == 2 and edge_index.shape[1] != 2:
        edge_index = edge_index.T
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {edge_index.shape}")
    edge_index = edge_index.astype(np.int64, copy=False)
    valid = (
        (edge_index[:, 0] >= 0)
        & (edge_index[:, 1] >= 0)
        & (edge_index[:, 0] < num_nodes)
        & (edge_index[:, 1] < num_nodes)
    )
    return edge_index[valid]


def result_files(path: Path) -> list[tuple[int, Path]]:
    if path.is_file():
        index = int(path.stem) - 1 if path.stem.isdigit() else 0
        return [(index, path)]
    directory = path / "result" if (path / "result").is_dir() else path
    files = sorted(
        directory.glob("*.h5"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )
    out = []
    for offset, file in enumerate(files):
        index = int(file.stem) - 1 if file.stem.isdigit() else offset
        out.append((index, file))
    return out


def load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        missing = [key for key in ("predicteds", "targets") if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing datasets {missing}")
        pred = np.asarray(handle["predicteds"], dtype=np.float64)
        target = np.asarray(handle["targets"], dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 4:
        raise ValueError(f"unexpected rollout shapes: {pred.shape}, {target.shape}")
    return pred, target


def validate_geometry(
    pos: np.ndarray, edges: np.ndarray, node_type: np.ndarray, nodes: int
) -> dict[str, Any]:
    if pos.ndim != 2 or pos.shape[0] != nodes or pos.shape[1] < 2:
        raise ValueError(f"pos must have shape ({nodes}, dim>=2), got {pos.shape}")
    if node_type.ndim != 1 or node_type.shape[0] != nodes:
        raise ValueError(f"node_type must have shape ({nodes},), got {node_type.shape}")
    return {"pos": pos[:, :2], "edges": edges, "node_type": node_type}


def load_geometry_from_result(
    path: Path, nodes: int, frame: int
) -> dict[str, Any] | None:
    with h5py.File(path, "r") as handle:
        if not all(key in handle for key in ("pos", "edges", "node_type")):
            return None
        pos = np.asarray(read_frame_or_static(handle["pos"], frame), dtype=np.float64)
        edges = normalize_edges(read_frame_or_static(handle["edges"], frame), nodes)
        node_type = squeeze_node_column(
            read_frame_or_static(handle["node_type"], frame)
        ).astype(np.int64)
    return validate_geometry(pos, edges, node_type, nodes)


def load_geometry_from_dataset(
    handle: h5py.File, index: int, nodes: int, frame: int
) -> dict[str, Any]:
    keys = list(handle.keys())
    if index < 0 or index >= len(keys):
        raise IndexError(
            f"trajectory index {index} outside dataset with {len(keys)} groups"
        )
    group = handle[keys[index]]
    pos = np.asarray(read_frame_or_static(group["pos"], frame), dtype=np.float64)
    edges = normalize_edges(read_frame_or_static(group["edges"], frame), nodes)
    node_type = squeeze_node_column(
        read_frame_or_static(group["node_type"], frame)
    ).astype(np.int64)
    geometry = validate_geometry(pos, edges, node_type, nodes)
    geometry["trajectory_key"] = keys[index]
    return geometry


def per_time_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = pred[:, mask, :] - target[:, mask, :]
    return np.sqrt(np.mean(err * err, axis=1))


def per_time_relative_l2(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    err = pred[:, mask, :] - target[:, mask, :]
    numerator = np.sqrt(np.sum(err * err, axis=1))
    denominator = np.sqrt(np.sum(target[:, mask, :] ** 2, axis=1))
    return numerator / np.maximum(denominator, EPS)


def vpt(
    relative_l2: np.ndarray, thresholds: Sequence[float], dt: float
) -> dict[str, dict[str, float]]:
    out = {}
    for threshold in thresholds:
        failed = relative_l2 > threshold
        by_feature = {}
        for i, name in enumerate(PRIMITIVE_NAMES):
            step = (
                int(np.argmax(failed[:, i]))
                if np.any(failed[:, i])
                else relative_l2.shape[0] - 1
            )
            by_feature[name] = float((step + 1) * dt)
        out[str(float(threshold))] = by_feature
    return out


def positivity(
    pred: np.ndarray, target: np.ndarray, normal: np.ndarray
) -> dict[str, Any]:
    p = pred[:, normal, :]
    t = target[:, normal, :]
    bad_rho = p[:, :, 0] <= 0.0
    bad_pres = p[:, :, 3] <= 0.0
    return {
        "all_positive": bool(not np.any(bad_rho) and not np.any(bad_pres)),
        "rho_nonpositive_count": int(np.count_nonzero(bad_rho)),
        "pres_nonpositive_count": int(np.count_nonzero(bad_pres)),
        "rho_nonpositive_rate": float(np.mean(bad_rho)) if bad_rho.size else 0.0,
        "pres_nonpositive_rate": float(np.mean(bad_pres)) if bad_pres.size else 0.0,
        "pred_rho_min": float(np.min(p[:, :, 0])),
        "pred_pres_min": float(np.min(p[:, :, 3])),
        "target_rho_min": float(np.min(t[:, :, 0])),
        "target_pres_min": float(np.min(t[:, :, 3])),
    }


def boundary(
    pred: np.ndarray, target: np.ndarray, boundary_mask: np.ndarray
) -> dict[str, Any]:
    if not np.any(boundary_mask):
        return {
            "boundary_node_count": 0,
            "boundary_max_abs_error": 0.0,
            "boundary_rmse": by_var(np.zeros(4)),
        }
    err = pred[:, boundary_mask, :] - target[:, boundary_mask, :]
    return {
        "boundary_node_count": int(np.count_nonzero(boundary_mask)),
        "boundary_max_abs_error": float(np.max(np.abs(err))),
        "boundary_rmse": by_var(np.sqrt(np.mean(err * err, axis=(0, 1)))),
    }


def equal_node_conservation(
    pred: np.ndarray, target: np.ndarray, normal: np.ndarray
) -> dict[str, Any]:
    pc = primitive_to_conservative(pred[:, normal, :])
    tc = primitive_to_conservative(target[:, normal, :])
    pt = np.sum(pc, axis=1)
    tt = np.sum(tc, axis=1)
    mismatch = np.abs(pt - tt) / np.maximum(np.abs(tt), EPS)
    pred_drift = np.abs(pt - pt[[0]]) / np.maximum(np.abs(pt[[0]]), EPS)
    target_drift = np.abs(tt - tt[[0]]) / np.maximum(np.abs(tt[[0]]), EPS)
    return {
        "note": "equal-node normal-node totals; no mesh/cell weights used",
        "mean_relative_mismatch_vs_target": by_cons(np.mean(mismatch, axis=0)),
        "final_relative_mismatch_vs_target": by_cons(mismatch[-1]),
        "max_pred_relative_temporal_drift": by_cons(np.max(pred_drift, axis=0)),
        "max_target_relative_temporal_drift": by_cons(np.max(target_drift, axis=0)),
    }


def region_errors(
    pred: np.ndarray, target: np.ndarray, regions: dict[str, np.ndarray]
) -> dict[str, Any]:
    err = pred - target
    out = {}
    for name, mask in regions.items():
        item: dict[str, Any] = {
            "mean_node_fraction": float(np.mean(mask)),
            "min_nodes_per_step": int(np.min(np.count_nonzero(mask, axis=1))),
            "max_nodes_per_step": int(np.max(np.count_nonzero(mask, axis=1))),
        }
        if np.any(mask):
            rmse = []
            rel = []
            for i in range(4):
                e = err[:, :, i][mask]
                t = target[:, :, i][mask]
                rmse.append(float(np.sqrt(np.mean(e * e))))
                rel.append(
                    float(np.sqrt(np.sum(e * e)) / max(np.sqrt(np.sum(t * t)), EPS))
                )
            item["rmse"] = by_var(rmse)
            item["relative_l2"] = by_var(rel)
        else:
            item["rmse"] = by_var(np.full(4, np.nan))
            item["relative_l2"] = by_var(np.full(4, np.nan))
        out[name] = item
    return out


def thickness_strength(
    pred_scores: np.ndarray, target_scores: np.ndarray, normal: np.ndarray
) -> dict[str, Any]:
    ps = pred_scores[:, normal]
    ts = target_scores[:, normal]
    pmax = np.max(ps, axis=1)
    tmax = np.max(ts, axis=1)
    threshold = 0.25
    pfrac = np.mean((ps >= threshold * pmax[:, None]) & (ps > 0.0), axis=1)
    tfrac = np.mean((ts >= threshold * tmax[:, None]) & (ts > 0.0), axis=1)
    return {
        "relative_threshold": threshold,
        "thickness_ratio": finite_stats(pfrac / np.maximum(tfrac, EPS)),
        "strength_ratio": finite_stats(pmax / np.maximum(tmax, EPS)),
        "prediction_active_fraction": finite_stats(pfrac),
        "target_active_fraction": finite_stats(tfrac),
    }


def shock_summary(
    pred: np.ndarray,
    target: np.ndarray,
    pos: np.ndarray,
    edges: np.ndarray,
    normal: np.ndarray,
    quantile: float,
    *,
    do_alignment: bool,
    alignment_grid_size: int,
    alignment_max_shift: float | None,
    alignment_edge_multiple: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fronts = shock_front_masks(
        pred, target, edges, scalar_index=3, quantile=quantile, node_mask=normal
    )
    pmask = fronts["prediction_mask"]
    tmask = fronts["target_mask"]
    overlap = front_overlap_metrics(pmask, tmask)
    distance = front_distance_metrics(pmask, tmask, pos)
    centroid = front_centroid_distance(
        pmask,
        tmask,
        pos,
        prediction_weights=fronts["prediction_scores"],
        target_weights=fronts["target_scores"],
    )
    regions = front_region_masks(pmask, tmask, node_mask=normal)
    summary = {
        "quantile": float(quantile),
        "overlap": {
            "iou": finite_stats(overlap["iou"]),
            "f1": finite_stats(overlap["f1"]),
        },
        "distance": {
            "symmetric_chamfer_mean": finite_stats(distance["symmetric_chamfer_mean"]),
            "hausdorff": finite_stats(distance["hausdorff"]),
            "centroid_distance": finite_stats(centroid),
        },
        "region_errors": region_errors(pred, target, regions),
        "thickness_strength": thickness_strength(
            fronts["prediction_scores"], fronts["target_scores"], normal
        ),
    }
    per_time = {
        "iou": overlap["iou"],
        "f1": overlap["f1"],
        "chamfer": distance["symmetric_chamfer_mean"],
        "centroid_distance": centroid,
    }
    if do_alignment:
        max_shift = alignment_max_shift
        if max_shift is None:
            max_shift = alignment_edge_multiple * median_edge_length(pos, edges)
        shifts = shift_grid(max_shift, grid_size=alignment_grid_size, dim=pos.shape[1])
        align = local_shift_alignment_metrics(
            pred, target, pos, regions["front_union"], shifts, scalar_index=3
        )
        summary["alignment"] = {
            "max_shift": float(max_shift),
            "candidate_shifts": int(shifts.shape[0]),
            "baseline_pressure_rmse": finite_stats(align["baseline_rmse"]),
            "best_shift_pressure_rmse": finite_stats(align["best_shift_rmse"]),
            "relative_rmse_reduction": finite_stats(align["relative_rmse_reduction"]),
            "best_shift_norm": finite_stats(align["best_shift_norm"]),
        }
        per_time["alignment_reduction"] = align["relative_rmse_reduction"]
        per_time["best_shift_norm"] = align["best_shift_norm"]
    return summary, per_time


def analyze_file(
    run: str,
    index: int,
    path: Path,
    dataset: h5py.File | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred, target = load_rollout(path)
    geometry = load_geometry_from_result(path, pred.shape[1], args.geometry_frame)
    if geometry is None:
        if dataset is None:
            raise ValueError(f"{path} has no geometry; pass --dataset-file")
        geometry = load_geometry_from_dataset(
            dataset, index, pred.shape[1], args.geometry_frame
        )
    pos = np.asarray(geometry["pos"], dtype=np.float64)
    edges = np.asarray(geometry["edges"], dtype=np.int64)
    node_type = np.asarray(geometry["node_type"], dtype=np.int64)
    normal = node_type == int(EulerNodeType.NORMAL)
    if not np.any(normal):
        normal = np.ones(node_type.shape[0], dtype=bool)
    bmask = ~normal

    step_rmse = per_time_rmse(pred, target, normal)
    step_rel = per_time_relative_l2(pred, target, normal)
    err = pred[:, normal, :] - target[:, normal, :]
    sse = np.sum(err * err, axis=(0, 1))
    count = int(err.shape[0] * err.shape[1])

    shock = {}
    shock_time = {}
    for q in args.quantiles:
        do_align = (not args.disable_alignment) and math.isclose(
            q, args.alignment_quantile
        )
        key = f"q{q:.2f}"
        shock[key], shock_time[key] = shock_summary(
            pred,
            target,
            pos,
            edges,
            normal,
            q,
            do_alignment=do_align,
            alignment_grid_size=args.alignment_grid_size,
            alignment_max_shift=args.alignment_max_shift,
            alignment_edge_multiple=args.alignment_edge_multiple,
        )

    trajectory = {
        "run": run,
        "file": str(path),
        "trajectory_index": int(index),
        "trajectory_key": str(geometry.get("trajectory_key", "")),
        "num_steps": int(pred.shape[0]),
        "num_nodes": int(pred.shape[1]),
        "normal_node_count": int(np.count_nonzero(normal)),
        "edge_count": int(edges.shape[0]),
        "finite_prediction": bool(np.isfinite(pred).all()),
        "finite_target": bool(np.isfinite(target).all()),
        "overall_rmse": by_var(np.sqrt(sse / max(count, 1))),
        "final_step_rmse": by_var(step_rmse[-1]),
        "mean_step_rmse": by_var(np.mean(step_rmse, axis=0)),
        "final_relative_l2": by_var(step_rel[-1]),
        "valid_prediction_time": vpt(step_rel, args.vpt_thresholds, args.dt),
        "positivity": positivity(pred, target, normal),
        "boundary": boundary(pred, target, bmask),
        "equal_node_conservation": equal_node_conservation(pred, target, normal),
        "shock_quantiles": shock,
        "_sse": sse,
        "_count": count,
    }

    rows = []
    qkey = f"q{args.alignment_quantile:.2f}"
    selected_shock_time = shock_time.get(qkey, {})
    for step in range(pred.shape[0]):
        row: dict[str, Any] = {
            "run": run,
            "trajectory_index": int(index),
            "step": step + 1,
            "time": float((step + 1) * args.dt),
        }
        for i, name in enumerate(PRIMITIVE_NAMES):
            row[f"rmse_{name}"] = float(step_rmse[step, i])
            row[f"relative_l2_{name}"] = float(step_rel[step, i])
        for name, values in selected_shock_time.items():
            row[f"shock_{qkey}_{name}"] = float(values[step])
        rows.append(row)
    return trajectory, rows


def strip_private_arrays(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def aggregate(run: str, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectories:
        return {"run": run, "trajectory_count": 0}
    sse = np.sum([item["_sse"] for item in trajectories], axis=0)
    count = int(np.sum([item["_count"] for item in trajectories]))
    overall = np.array(
        [
            [item["overall_rmse"][name] for name in PRIMITIVE_NAMES]
            for item in trajectories
        ]
    )
    final = np.array(
        [
            [item["final_step_rmse"][name] for name in PRIMITIVE_NAMES]
            for item in trajectories
        ]
    )
    return {
        "run": run,
        "trajectory_count": int(len(trajectories)),
        "node_time_weighted_overall_rmse": by_var(np.sqrt(sse / max(count, 1))),
        "mean_per_trajectory_overall_rmse": by_var(np.mean(overall, axis=0)),
        "std_per_trajectory_overall_rmse": by_var(np.std(overall, axis=0)),
        "mean_final_step_rmse": by_var(np.mean(final, axis=0)),
        "all_predictions_finite": bool(
            all(item["finite_prediction"] for item in trajectories)
        ),
        "all_targets_finite": bool(all(item["finite_target"] for item in trajectories)),
        "max_boundary_abs_error": float(
            max(item["boundary"]["boundary_max_abs_error"] for item in trajectories)
        ),
        "all_positive": bool(
            all(item["positivity"]["all_positive"] for item in trajectories)
        ),
        "max_rho_nonpositive_rate": float(
            max(item["positivity"]["rho_nonpositive_rate"] for item in trajectories)
        ),
        "max_pres_nonpositive_rate": float(
            max(item["positivity"]["pres_nonpositive_rate"] for item in trajectories)
        ),
    }


def add_teacher_forced(summary: dict[str, Any], specs: Sequence[str] | None) -> None:
    for raw in specs or []:
        name, path = parse_named_path(raw)
        data = json.loads(path.read_text(encoding="utf-8"))
        tf = data.get("normal_node_rmse") or data.get("all_node_rmse")
        run = summary["runs"].get(name)
        if not run or not isinstance(tf, dict):
            continue
        ar = run.get("mean_per_trajectory_overall_rmse", {})
        ratio = {}
        for feature in PRIMITIVE_NAMES:
            denom = float(tf.get(feature, math.nan))
            ratio[feature] = float(ar[feature] / denom) if denom > 0.0 else math.nan
        run["teacher_forced_reference_rmse"] = {
            feature: float(tf[feature]) for feature in PRIMITIVE_NAMES if feature in tf
        }
        run["autoregressive_to_teacher_forced_rmse_ratio"] = ratio
        run["teacher_forced_time_resolution"] = (
            "aggregate only; no teacher-forced per-time arrays were provided"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    qkey = summary["alignment_quantile_key"]
    lines = ["# CPGNet Mechanistic Diagnostic", "", "## Rollout RMSE"]
    lines += ["| Run | rho | v1 | v2 | pres |", "| --- | ---: | ---: | ---: | ---: |"]
    for run, data in summary["runs"].items():
        values = data.get("mean_per_trajectory_overall_rmse", {})
        cells = " | ".join(
            f"{values.get(name, math.nan):.6f}" for name in PRIMITIVE_NAMES
        )
        lines.append(f"| {run} | {cells} |")
    lines += [
        "",
        "## Shock Position",
        "| Run/Traj | IoU | F1 | Chamfer | Centroid | Align Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, trajectories in summary["trajectory_summaries"].items():
        for traj in trajectories:
            shock = traj["shock_quantiles"].get(qkey)
            if not shock:
                continue
            align = shock.get("alignment", {})
            reduction = align.get("relative_rmse_reduction", {}).get("mean", math.nan)
            lines.append(
                f"| {run}/traj{traj['trajectory_index']} | "
                f"{shock['overlap']['iou']['mean']:.4f} | {shock['overlap']['f1']['mean']:.4f} | "
                f"{shock['distance']['symmetric_chamfer_mean']['mean']:.6f} | "
                f"{shock['distance']['centroid_distance']['mean']:.6f} | {reduction:.4f} |"
            )
    lines += [
        "",
        "## Notes",
        "- Conservation metrics use equal-node normal-node totals; no cell or mesh weights are used.",
        "- Boundary leakage is measured directly and should be zero for clamped official rollout arrays.",
        "- Teacher-forced comparison is aggregate unless rollout-style teacher-forced arrays are analyzed as a separate run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="NAME=run_dir_or_result_h5"
    )
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/time_dependent_no/cpg_mechanistic_diagnostic"),
    )
    parser.add_argument("--trajectory-indices", nargs="*", type=int)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--geometry-frame", type=int, default=0)
    parser.add_argument(
        "--quantiles", nargs="+", type=float, default=[0.85, 0.90, 0.95]
    )
    parser.add_argument("--alignment-quantile", type=float, default=0.90)
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--alignment-grid-size", type=int, default=5)
    parser.add_argument("--alignment-max-shift", type=float)
    parser.add_argument("--alignment-edge-multiple", type=float, default=3.0)
    parser.add_argument("--vpt-thresholds", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument(
        "--teacher-forced-summary", action="append", help="NAME=summary.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.quantiles = sorted(
        set(float(q) for q in args.quantiles + [args.alignment_quantile])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = h5py.File(args.dataset_file, "r") if args.dataset_file else None
    time_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_summaries: dict[str, list[dict[str, Any]]] = {}
    run_summaries: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    try:
        for raw in args.run:
            run, path = parse_named_path(raw)
            files = result_files(path)
            if args.trajectory_indices:
                wanted = set(args.trajectory_indices)
                files = [(idx, file) for idx, file in files if idx in wanted]
            if args.max_trajectories is not None:
                files = files[: args.max_trajectories]
            audit[run] = {
                "path": str(path),
                "result_files_analyzed": len(files),
                "files": [str(file) for _, file in files],
            }
            trajectories = []
            for index, file in files:
                traj, rows = analyze_file(run, index, file, dataset, args)
                trajectories.append(traj)
                time_rows.extend(rows)
                trajectory_rows.append(
                    {
                        "run": run,
                        "trajectory_index": index,
                        **{
                            f"overall_rmse_{name}": traj["overall_rmse"][name]
                            for name in PRIMITIVE_NAMES
                        },
                        **{
                            f"final_rmse_{name}": traj["final_step_rmse"][name]
                            for name in PRIMITIVE_NAMES
                        },
                        "boundary_max_abs_error": traj["boundary"][
                            "boundary_max_abs_error"
                        ],
                        "all_positive": traj["positivity"]["all_positive"],
                    }
                )
            trajectory_summaries[run] = [
                strip_private_arrays(item) for item in trajectories
            ]
            run_summaries[run] = aggregate(run, trajectories)
    finally:
        if dataset is not None:
            dataset.close()

    summary = {
        "artifact_audit": audit,
        "runs": run_summaries,
        "trajectory_summaries": trajectory_summaries,
        "quantiles": args.quantiles,
        "alignment_quantile": float(args.alignment_quantile),
        "alignment_quantile_key": f"q{args.alignment_quantile:.2f}",
        "dt": float(args.dt),
        "vpt_thresholds": [float(x) for x in args.vpt_thresholds],
    }
    add_teacher_forced(summary, args.teacher_forced_summary)

    summary_path = args.output_dir / "analysis_summary.json"
    time_path = args.output_dir / "per_time_metrics.csv"
    traj_path = args.output_dir / "per_trajectory_metrics.csv"
    report_path = args.output_dir / "diagnostic_report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_csv(time_path, time_rows)
    write_csv(traj_path, trajectory_rows)
    write_report(report_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "per_time": str(time_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
