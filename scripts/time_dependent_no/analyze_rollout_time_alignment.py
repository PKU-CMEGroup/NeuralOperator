"""Align rollout frames to ground-truth times by minimum MSE.

For each predicted rollout time ``tr``, this script finds the ground-truth time
``tg`` in the same trajectory that minimizes MSE. A curve near
``tg = tr - c`` indicates an approximately fixed phase lag. A curve with slope
away from one indicates a speed/frequency mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.time_alignment import (  # noqa: E402
    compute_time_alignment,
    fit_best_time_curve,
)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Rollout directory containing result/*.h5")
    source.add_argument("--result-files", nargs="+", type=Path, help="Specific rollout result files")
    parser.add_argument("--trajectory-indices", nargs="+", type=int, default=[0, 6, 11, 13, 17])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--feature", default="pres", help="'all' or one of rho/v1/v2/pres")
    parser.add_argument("--node-filter", choices=NODE_FILTERS, default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"artifacts/time_dependent_no/rollout_time_alignment_{date}"),
    )
    parser.add_argument("--step-offset", type=int, default=1, help="Label first rollout array frame with this step.")
    return parser.parse_args(argv)


def parse_feature(feature: str) -> list[int]:
    key = FEATURE_ALIASES.get(feature.strip(), feature.strip())
    if key == "all":
        return list(range(len(FEATURE_NAMES)))
    if key in FEATURE_NAMES:
        return [FEATURE_NAMES.index(key)]
    raise ValueError(f"unknown feature {feature!r}; use 'all' or one of {FEATURE_NAMES}")


def result_file_for(run_dir: Path, trajectory_index: int) -> Path:
    return run_dir / "result" / f"{trajectory_index + 1}.h5"


def load_rollout_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        prediction = np.asarray(handle["predicteds"], dtype=np.float64)
        truth = np.asarray(handle["targets"], dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ValueError(f"{path}: predicteds and targets shapes differ: {prediction.shape} vs {truth.shape}")
    if prediction.ndim != 3 or prediction.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"{path}: expected shape [time, nodes, 4], got {prediction.shape}")
    return prediction, truth


def squeeze_node_column(array: np.ndarray) -> np.ndarray:
    if array.ndim > 1 and array.shape[-1] == 1:
        return np.squeeze(array, axis=-1)
    return array


def trajectory_key(handle: h5py.File, trajectory_index: int) -> str:
    keys = list(handle.keys())
    if trajectory_index < 0 or trajectory_index >= len(keys):
        raise IndexError(f"trajectory_index={trajectory_index} outside [0, {len(keys)})")
    return keys[trajectory_index]


def load_node_type_from_result(result_file: Path, expected_nodes: int) -> np.ndarray | None:
    with h5py.File(result_file, "r") as handle:
        if "node_type" not in handle:
            return None
        node_type = squeeze_node_column(np.asarray(handle["node_type"]))
    if node_type.shape != (expected_nodes,):
        raise ValueError(
            f"{result_file}: node_type shape {node_type.shape} does not match rollout node count {expected_nodes}"
        )
    return np.asarray(node_type, dtype=np.int64)


def load_node_mask(
    *,
    result_file: Path,
    dataset_root: Path | None,
    split: str,
    trajectory_index: int,
    expected_nodes: int,
    node_filter: str,
) -> np.ndarray | None:
    if node_filter == "all":
        return None
    result_node_type = load_node_type_from_result(result_file, expected_nodes)
    if result_node_type is not None:
        normal = result_node_type == 0
        return normal if node_filter == "normal" else ~normal
    if dataset_root is None:
        raise ValueError(
            "--dataset-root is required when --node-filter is not 'all' "
            "and result files do not include node_type"
        )
    dataset_file = dataset_root / f"{split}.h5"
    with h5py.File(dataset_file, "r") as handle:
        key = trajectory_key(handle, trajectory_index)
        node_type = squeeze_node_column(np.asarray(handle[key]["node_type"][0]))
    if node_type.shape != (expected_nodes,):
        raise ValueError(
            f"node_type shape {node_type.shape} does not match rollout node count {expected_nodes}"
        )
    normal = np.asarray(node_type, dtype=np.int64) == 0
    return normal if node_filter == "normal" else ~normal


def improvement_ratio(diagonal_mse: np.ndarray, best_mse: np.ndarray) -> np.ndarray:
    out = np.full(best_mse.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(diagonal_mse) & (diagonal_mse > 0.0)
    out[valid] = (diagonal_mse[valid] - best_mse[valid]) / diagonal_mse[valid]
    return out


def write_alignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_best_time(output_path: Path, summaries: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for item in summaries:
        rows = item["rows"]
        x = np.array([row["rollout_step"] for row in rows], dtype=float)
        y = np.array([row["best_target_step"] for row in rows], dtype=float)
        ax.plot(x, y, marker=".", linewidth=1.5, label=f"traj {item['trajectory_index']}")
    if summaries:
        max_step = max(max(row["rollout_step"] for row in item["rows"]) for item in summaries)
        ax.plot([1, max_step], [1, max_step], "k--", linewidth=1.0, label="tg = tr")
    ax.set_xlabel("rollout time tr")
    ax.set_ylabel("best ground-truth time tg")
    ax.set_title("Best matching ground-truth time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lag(output_path: Path, summaries: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for item in summaries:
        rows = item["rows"]
        x = np.array([row["rollout_step"] for row in rows], dtype=float)
        y = np.array([row["best_target_step"] - row["rollout_step"] for row in rows], dtype=float)
        ax.plot(x, y, marker=".", linewidth=1.5, label=f"traj {item['trajectory_index']}")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("rollout time tr")
    ax.set_ylabel("lag tg - tr")
    ax.set_title("Best-time lag")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def interpret_fit(fit: dict[str, float]) -> str:
    slope = fit["slope"]
    lag = fit["median_lag"]
    if abs(slope - 1.0) <= 0.1 and abs(lag) >= 2.0:
        return "fixed_phase_offset_like"
    if slope < 0.9:
        return "rollout_too_slow_or_saturating"
    if slope > 1.1:
        return "rollout_too_fast"
    return "near_diagonal_or_mixed"


def fit_subset(
    rollout_steps: np.ndarray,
    best_target_steps: np.ndarray,
    selector: np.ndarray,
) -> dict[str, float] | None:
    selector = np.asarray(selector, dtype=bool)
    if np.count_nonzero(selector) < 2:
        return None
    return fit_best_time_curve(rollout_steps[selector], best_target_steps[selector])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    feature_indices = parse_feature(args.feature)
    feature_label = "all" if len(feature_indices) > 1 else FEATURE_NAMES[feature_indices[0]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    result_files = args.result_files
    if result_files is None:
        result_files = [result_file_for(args.run_dir, idx) for idx in args.trajectory_indices]
        trajectory_indices = args.trajectory_indices
    else:
        trajectory_indices = args.trajectory_indices
        if len(trajectory_indices) != len(result_files):
            raise ValueError("--trajectory-indices must match --result-files length")

    for trajectory_index, result_file in zip(trajectory_indices, result_files):
        prediction, truth = load_rollout_arrays(result_file)
        node_mask = load_node_mask(
            result_file=result_file,
            dataset_root=args.dataset_root,
            split=args.split,
            trajectory_index=trajectory_index,
            expected_nodes=prediction.shape[1],
            node_filter=args.node_filter,
        )
        alignment = compute_time_alignment(
            prediction,
            truth,
            variables=feature_indices,
            node_mask=node_mask,
        )
        rollout_steps = np.arange(prediction.shape[0], dtype=np.int64) + int(args.step_offset)
        best_target_steps = alignment.best_target_indices.astype(np.int64) + int(args.step_offset)
        fit = fit_best_time_curve(rollout_steps, best_target_steps)
        fit["interpretation"] = interpret_fit(fit)
        min_target_step = int(args.step_offset)
        max_target_step = int(args.step_offset) + truth.shape[0] - 1
        edge_hit = (
            ((alignment.best_target_indices == 0) & (rollout_steps > min_target_step))
            | (
                (alignment.best_target_indices == truth.shape[0] - 1)
                & (rollout_steps < max_target_step)
            )
        )
        fit["edge_hit_fraction"] = float(np.mean(edge_hit))
        improv = improvement_ratio(alignment.diagonal_mse, alignment.best_mse)
        fit["mean_best_vs_diagonal_improvement"] = float(np.nanmean(improv))

        edge_indices = np.flatnonzero(edge_hit)
        first_edge_hit_step = (
            int(rollout_steps[int(edge_indices[0])]) if edge_indices.size else None
        )
        pre_edge_selector = np.ones_like(edge_hit, dtype=bool)
        if edge_indices.size:
            pre_edge_selector[int(edge_indices[0]) :] = False
        non_edge_fit = fit_subset(rollout_steps, best_target_steps, ~edge_hit)
        pre_edge_fit = fit_subset(rollout_steps, best_target_steps, pre_edge_selector)
        if non_edge_fit is not None:
            non_edge_fit["interpretation"] = interpret_fit(non_edge_fit)
        if pre_edge_fit is not None:
            pre_edge_fit["interpretation"] = interpret_fit(pre_edge_fit)

        rows = []
        for i, rollout_step in enumerate(rollout_steps):
            rows.append(
                {
                    "rollout_step": int(rollout_step),
                    "best_target_step": int(best_target_steps[i]),
                    "lag": int(best_target_steps[i] - rollout_step),
                    "best_mse": float(alignment.best_mse[i]),
                    "diagonal_mse": float(alignment.diagonal_mse[i]),
                    "best_vs_diagonal_improvement": float(improv[i]),
                }
            )
        csv_path = args.output_dir / f"traj_{trajectory_index:02d}_{feature_label}_{args.node_filter}_time_alignment.csv"
        write_alignment_csv(csv_path, rows)
        summaries.append(
            {
                "trajectory_index": int(trajectory_index),
                "result_file": str(result_file),
                "feature": feature_label,
                "node_filter": args.node_filter,
                "num_steps": int(prediction.shape[0]),
                "num_nodes": int(np.sum(node_mask) if node_mask is not None else prediction.shape[1]),
                "csv_path": str(csv_path),
                "fit": fit,
                "first_edge_hit_step": first_edge_hit_step,
                "non_edge_fit": non_edge_fit,
                "pre_edge_fit": pre_edge_fit,
                "rows": rows,
            }
        )

    best_time_plot = args.output_dir / f"{feature_label}_{args.node_filter}_best_time_curve.png"
    lag_plot = args.output_dir / f"{feature_label}_{args.node_filter}_lag_curve.png"
    plot_best_time(best_time_plot, summaries)
    plot_lag(lag_plot, summaries)

    payload = {
        "mode": "rollout_time_alignment",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature": feature_label,
        "node_filter": args.node_filter,
        "step_offset": int(args.step_offset),
        "best_time_plot": str(best_time_plot),
        "lag_plot": str(lag_plot),
        "trajectories": [
            {key: value for key, value in item.items() if key != "rows"} for item in summaries
        ],
    }
    summary_path = args.output_dir / f"{feature_label}_{args.node_filter}_time_alignment_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Rollout Time Alignment",
        "",
        f"Feature: `{feature_label}`",
        f"Node filter: `{args.node_filter}`",
        "",
        "| Traj | Slope | Intercept | Median lag | Edge-hit frac | Mean improvement | Interpretation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        fit = item["fit"]
        lines.append(
            f"| {item['trajectory_index']} | {fit['slope']:.4g} | {fit['intercept']:.4g} | "
            f"{fit['median_lag']:.4g} | {fit['edge_hit_fraction']:.3g} | "
            f"{fit['mean_best_vs_diagonal_improvement']:.3g} | {fit['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Pre-Edge Fit",
            "",
            "| Traj | First edge hit | Pre-edge slope | Pre-edge median lag | Non-edge slope | Non-edge median lag |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summaries:
        pre = item["pre_edge_fit"] or {}
        non = item["non_edge_fit"] or {}
        first_edge = item["first_edge_hit_step"]
        lines.append(
            f"| {item['trajectory_index']} | {first_edge if first_edge is not None else 'none'} | "
            f"{pre.get('slope', float('nan')):.4g} | {pre.get('median_lag', float('nan')):.4g} | "
            f"{non.get('slope', float('nan')):.4g} | {non.get('median_lag', float('nan')):.4g} |"
        )
    lines.extend(
        [
            "",
            f"Best-time plot: `{best_time_plot}`",
            f"Lag plot: `{lag_plot}`",
            "",
            "Interpretation rule: slope near 1 with nonzero lag suggests fixed phase offset; "
            "slope away from 1 suggests speed/frequency mismatch unless the curve is clipped "
            "at the first/last available target time.",
        ]
    )
    report_path = args.output_dir / f"{feature_label}_{args.node_filter}_time_alignment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
