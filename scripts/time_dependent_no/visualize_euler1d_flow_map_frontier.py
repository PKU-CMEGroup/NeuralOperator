"""Visualize and extend the frozen Euler large-step flow-map evidence.

The subcommands are deliberately training-free. ``summary`` turns the
registered D032--D034 tables into reproducible publication figures. ``closeout``
combines the registered D032--D038 tables into one operating-envelope and
global/local/runtime Pareto audit. ``rollout`` replays the four selected
residual-FNO checkpoints with exact conservative recurrence, writes norm/phase
diagnostics, and renders selected raw rollouts. ``ripple`` analyzes D013-style
frozen spectral tables at common physical endpoints. No clipping, primitive
floor, limiter, or checkpoint reselection is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, LogNorm

if __package__:
    from scripts.time_dependent_no.diagnose_euler1d_flow_map_solver_consistency import (
        _validate_checkpoint_contract,
    )
    from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (
        PHYSICAL_SCALES,
        _checkpoint_paths,
        _euler_state_is_admissible,
        _frozen_split,
        _predict_model_batch_state,
        _saved_time_sha256,
    )
    from scripts.time_dependent_no.evaluate_euler1d_resolution_transfer import (
        load_frozen_residual_checkpoint,
    )
    from scripts.time_dependent_no.train_euler1d_target_ladder import (
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )
else:
    from diagnose_euler1d_flow_map_solver_consistency import (
        _validate_checkpoint_contract,
    )
    from evaluate_euler1d_flow_map_frontier import (
        PHYSICAL_SCALES,
        _checkpoint_paths,
        _euler_state_is_admissible,
        _frozen_split,
        _predict_model_batch_state,
        _saved_time_sha256,
    )
    from evaluate_euler1d_resolution_transfer import (
        load_frozen_residual_checkpoint,
    )
    from train_euler1d_target_ladder import (
        json_ready,
        pressure_front_top2_metrics_np,
        sha256_file,
    )

from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    load_euler1d_npz,
    primitive_to_conservative_np,
)

EPS = 1.0e-12
REQUIRED_STRIDES = (1, 2, 4, 8)
COLORS = {
    1: "#0072B2",
    2: "#E69F00",
    4: "#009E73",
    8: "#CC79A7",
}
MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}
VARIABLE_LABELS = (r"$\rho$", r"$u$", r"$p$")
RIPPLE_METRICS = (
    "state_cons_scaled_rel_l2",
    "smooth_cons_scaled_rmse",
    "error_tail_65_nyquist_fraction",
    "state_tv_ratio",
    "smooth_state_tv_ratio",
    "smooth_error_d1_rms",
    "smooth_error_d2_rms",
)
SPECTRAL_MODE_BANDS = {
    "low_1_4": (1, 5),
    "mid_5_16": (5, 17),
    # Legacy artifact key: k=24 is the first mode beyond a 24-mode FNO cutoff.
    "resolved_17_24": (17, 25),
    "high_25_64": (25, 65),
    "tail_65_nyquist": (65, None),
}
MODAL_RELATIVE_LOG10_LIMITS = (-3.0, 1.0)
MODAL_SHARE_LOG10_LIMITS = (-6.0, -0.3)
MODAL_RATIO_LOG10_LIMITS = (-0.5, 0.5)
CLOSEOUT_HORIZONS = (8, 16, 32, 64, 96)
CLOSEOUT_ERROR_BUDGETS = (0.02, 0.05, 0.10)
CLOSEOUT_PRIMARY_MODELS = {1: "s1", 2: "s2", 4: "s4", 8: "s8_seed20260707"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary",
        help="Render D032--D034 tables without loading data or checkpoints.",
    )
    summary.add_argument("--d032-dir", type=Path, required=True)
    summary.add_argument("--d033-dir", type=Path, required=True)
    summary.add_argument("--d034-dir", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--error-budget", type=float, default=0.05)

    closeout = subparsers.add_parser(
        "closeout",
        help="Synthesize the frozen D032--D038 operating-envelope evidence.",
    )
    closeout.add_argument("--d032-dir", type=Path, required=True)
    closeout.add_argument("--d033-dir", type=Path, required=True)
    closeout.add_argument("--d034-dir", type=Path, required=True)
    closeout.add_argument("--d036-dir", type=Path, required=True)
    closeout.add_argument("--d038-dir", type=Path, required=True)
    closeout.add_argument("--output-dir", type=Path, required=True)
    closeout.add_argument("--bootstrap-replicates", type=int, default=5000)
    closeout.add_argument("--bootstrap-seed", type=int, default=20260720)

    ripple = subparsers.add_parser(
        "ripple",
        help="Analyze frozen D013-style ripple metrics at common endpoints.",
    )
    ripple.add_argument("--metrics-csv", type=Path, required=True)
    ripple.add_argument(
        "--mode-spectra-csv",
        type=Path,
        default=None,
        help=(
            "Optional per-mode table from diagnose_euler1d_scale_spectra.py; "
            "enables common-scale modal heatmaps and band summaries."
        ),
    )
    ripple.add_argument("--output-dir", type=Path, required=True)
    ripple.add_argument("--primary-s8-name", default="s8_seed20260707")
    ripple.add_argument("--horizon", type=int, default=96)
    ripple.add_argument("--common-frame-step", type=int, default=8)
    ripple.add_argument("--saved-frame-dt", type=float, default=0.005)
    ripple.add_argument(
        "--smooth-tv-threshold",
        type=float,
        action="append",
        default=None,
        help="Repeat for a sensitivity grid; defaults to 1.25, 1.5, and 2.0.",
    )
    ripple.add_argument("--bootstrap-replicates", type=int, default=5000)
    ripple.add_argument("--bootstrap-seed", type=int, default=20260719)

    rollout = subparsers.add_parser(
        "rollout",
        help="Replay frozen checkpoints and render raw exact-recurrence rollouts.",
    )
    rollout.add_argument("--data-path", type=Path, required=True)
    rollout.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("STRIDE", "PATH"),
        required=True,
    )
    rollout.add_argument("--output-dir", type=Path, required=True)
    rollout.add_argument("--horizon", type=int, default=96)
    rollout.add_argument("--common-frame-step", type=int, default=8)
    rollout.add_argument(
        "--animation-case-ids",
        type=int,
        nargs="+",
        default=[355, 34, 407, 427, 158],
    )
    rollout.add_argument("--animation-fps", type=float, default=2.0)
    rollout.add_argument("--shock-radius-cells", type=int, default=4)
    rollout.add_argument("--max-alignment-shift-cells", type=int, default=24)
    rollout.add_argument("--bootstrap-replicates", type=int, default=2000)
    rollout.add_argument("--bootstrap-seed", type=int, default=20260719)
    rollout.add_argument("--split", choices=("validation", "test"), default="test")
    rollout.add_argument("--split-seed", type=int, default=20260707)
    rollout.add_argument("--train-cases", type=int, default=384)
    rollout.add_argument("--val-cases", type=int, default=64)
    rollout.add_argument("--test-cases", type=int, default=64)
    rollout.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    rollout.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(figure)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(list(rows)))


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def direct_curve_statistics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate direct-path rows without treating failed paths as survivors."""

    direct = [
        row for row in rows if str(row.get("path")) == f"s{int(row['stride'])}_direct"
    ]
    case_counts = {
        int(stride): len(
            {int(row["case_id"]) for row in direct if int(row["stride"]) == stride}
        )
        for stride in sorted({int(row["stride"]) for row in direct})
    }
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in direct:
        groups[(int(row["stride"]), int(row["frame"]))].append(row)
    metrics = {
        "error": "fixed_scale_conservative_relative_l2",
        "time_mean_error": "time_mean_fixed_scale_conservative_relative_l2",
        "smooth_error": "smooth_region_relative_l2",
        "shock_position_error": "shock_top2_position_mae",
        "shock_strength_error": "shock_top2_strength_relative_l1",
    }
    output: list[dict[str, Any]] = []
    for (stride, frame), group in sorted(groups.items()):
        valid = [row for row in group if _as_bool(row.get("completed_step"))]
        record: dict[str, Any] = {
            "stride": stride,
            "frame": frame,
            "physical_time": float(
                np.mean([_as_float(row, "physical_elapsed_time") for row in group])
            ),
            "num_cases": case_counts[stride],
            "num_raw_admissible": len(valid),
            "raw_completion_fraction": len(valid) / max(case_counts[stride], 1),
        }
        for prefix, key in metrics.items():
            summary = _finite_summary(_as_float(row, key) for row in valid)
            record.update(
                {f"{prefix}_{name}": value for name, value in summary.items()}
            )
        output.append(record)
    return output


def first_call_scaling(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Recover first-jump error and truth-increment scales from D032 rows."""

    output: list[dict[str, Any]] = []
    for stride in sorted({int(row["stride"]) for row in rows}):
        first = [
            row
            for row in rows
            if int(row["stride"]) == stride
            and int(row["frame"]) == stride
            and _as_bool(row.get("completed_step"))
        ]
        learned = np.asarray(
            [_as_float(row, "fixed_scale_conservative_relative_l2") for row in first],
            dtype=np.float64,
        )
        increment_relative = np.asarray(
            [_as_float(row, "trajectory_increment_relative_l2") for row in first],
            dtype=np.float64,
        )
        increment_state_normalized = np.asarray(
            [
                _as_float(row, "trajectory_increment_state_normalized_l2")
                for row in first
            ],
            dtype=np.float64,
        )
        truth_increment = increment_state_normalized / np.maximum(
            increment_relative,
            EPS,
        )
        output.append(
            {
                "stride": stride,
                "physical_dt": float(
                    np.mean([_as_float(row, "physical_elapsed_time") for row in first])
                ),
                "learned_state_error_mean": float(np.mean(learned)),
                "learned_state_error_median": float(np.median(learned)),
                "increment_relative_error_mean": float(np.mean(increment_relative)),
                "increment_relative_error_median": float(np.median(increment_relative)),
                "truth_increment_state_normalized_mean": float(
                    np.mean(truth_increment)
                ),
                "truth_increment_state_normalized_median": float(
                    np.median(truth_increment)
                ),
            }
        )
    strides = np.asarray([row["stride"] for row in output], dtype=np.float64)
    learned = np.asarray(
        [row["learned_state_error_mean"] for row in output], dtype=np.float64
    )
    truth_increment = np.asarray(
        [row["truth_increment_state_normalized_mean"] for row in output],
        dtype=np.float64,
    )
    slopes = {
        "learned_state_error_log_stride_slope": float(
            np.polyfit(np.log(strides), np.log(learned), 1)[0]
        ),
        "truth_increment_log_stride_slope": float(
            np.polyfit(np.log(strides), np.log(truth_increment), 1)[0]
        ),
    }
    return output, slopes


def casewise_frontier_rows(
    rows: Sequence[dict[str, Any]],
    *,
    strides: Sequence[int] = (2, 4, 8),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return winner shares and per-frame oracle gaps on common valid cases."""

    wanted = tuple(sorted(set(int(value) for value in strides)))
    by_key = {
        (int(row["frame"]), int(row["stride"]), int(row["case_id"])): _as_float(
            row,
            "fixed_scale_conservative_relative_l2",
        )
        for row in rows
        if int(row["stride"]) in wanted and _as_bool(row.get("completed_step"))
    }
    frames = sorted({key[0] for key in by_key if key[0] % max(wanted) == 0})
    winner_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for frame in frames:
        per_stride_cases = [
            {
                case
                for (row_frame, stride, case) in by_key
                if row_frame == frame and stride == value
            }
            for value in wanted
        ]
        common = sorted(set.intersection(*per_stride_cases)) if per_stride_cases else []
        if not common:
            continue
        matrix = np.asarray(
            [[by_key[(frame, stride, case)] for stride in wanted] for case in common],
            dtype=np.float64,
        )
        winners = np.argmin(matrix, axis=1)
        means = np.mean(matrix, axis=0)
        population_index = int(np.argmin(means))
        oracle = np.min(matrix, axis=1)
        for stride_index, stride in enumerate(wanted):
            winner_rows.append(
                {
                    "frame": frame,
                    "stride": stride,
                    "num_common_cases": len(common),
                    "winner_count": int(np.sum(winners == stride_index)),
                    "winner_fraction": float(np.mean(winners == stride_index)),
                }
            )
        oracle_rows.append(
            {
                "frame": frame,
                "num_common_cases": len(common),
                "population_best_stride": wanted[population_index],
                "population_best_mean_error": float(means[population_index]),
                "per_case_oracle_mean_error": float(np.mean(oracle)),
                "oracle_relative_improvement": float(
                    1.0 - np.mean(oracle) / max(means[population_index], EPS)
                ),
            }
        )
    return winner_rows, oracle_rows


def budget_reliability_rows(
    rows: Sequence[dict[str, Any]],
    *,
    budget: float,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if math.isclose(
            _as_float(row, "error_budget"), budget, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    output: list[dict[str, Any]] = []
    for stride in sorted({int(row["stride"]) for row in selected}):
        group = [row for row in selected if int(row["stride"]) == stride]
        max_horizon = int(max(_as_float(row, "max_horizon") for row in group))
        for frame in range(stride, max_horizon + 1, stride):
            reliable = [
                (not _as_bool(row.get("event_observed")))
                or _as_float(row, "event_frame") > frame
                for row in group
            ]
            output.append(
                {
                    "stride": stride,
                    "frame": frame,
                    "reliability_fraction": float(np.mean(reliable)),
                    "num_cases": len(group),
                }
            )
    return output


def _stride_series(
    rows: Sequence[dict[str, Any]],
    stride: int,
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if int(row["stride"]) == stride),
        key=lambda row: int(row["frame"]),
    )


def _plot_error_accumulation(
    statistics: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    metric_panels = (
        (axes[0, 0], "error", "Fixed-scale conservative relative L2"),
        (axes[0, 1], "time_mean_error", "Time-mean relative L2"),
    )
    for axis, prefix, ylabel in metric_panels:
        for stride in REQUIRED_STRIDES:
            series = _stride_series(statistics, stride)
            time = np.asarray([row["physical_time"] for row in series])
            median = np.asarray([row[f"{prefix}_median"] for row in series])
            mean = np.asarray([row[f"{prefix}_mean"] for row in series])
            q25 = np.asarray([row[f"{prefix}_q25"] for row in series])
            q75 = np.asarray([row[f"{prefix}_q75"] for row in series])
            axis.fill_between(time, q25, q75, color=COLORS[stride], alpha=0.12)
            axis.plot(
                time,
                median,
                color=COLORS[stride],
                marker=MARKERS[stride],
                markevery=max(1, len(series) // 8),
                markersize=3.5,
                linewidth=1.4,
                label=f"stride {stride}",
            )
            axis.plot(time, mean, color=COLORS[stride], linestyle=":", linewidth=0.9)
        axis.set_xlabel("Physical time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.set_xlim(left=0.0)
    axes[0, 0].legend(ncol=2, frameon=False)
    axes[0, 0].text(
        0.98,
        0.04,
        "solid: median; dotted: mean; band: case IQR",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
    )

    for stride in REQUIRED_STRIDES:
        series = _stride_series(statistics, stride)
        axes[1, 0].plot(
            [row["physical_time"] for row in series],
            [row["raw_completion_fraction"] for row in series],
            color=COLORS[stride],
            marker=MARKERS[stride],
            markevery=max(1, len(series) // 8),
            markersize=3.5,
            linewidth=1.4,
            label=f"stride {stride}",
        )
        budget_series = _stride_series(reliability, stride)
        time_by_frame = {int(row["frame"]): row["physical_time"] for row in series}
        axes[1, 1].step(
            [time_by_frame[int(row["frame"])] for row in budget_series],
            [row["reliability_fraction"] for row in budget_series],
            where="post",
            color=COLORS[stride],
            linewidth=1.4,
            label=f"stride {stride}",
        )
    axes[1, 0].set_xlabel("Physical time")
    axes[1, 0].set_ylabel("Raw-admissible fraction")
    axes[1, 1].set_xlabel("Physical time")
    axes[1, 1].set_ylabel("Fraction still within error budget")
    for axis in axes[1]:
        axis.set_ylim(-0.025, 1.025)
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.2)
    axes[1, 1].text(
        0.03,
        0.06,
        "Drops at first observed exceedance or raw failure",
        transform=axes[1, 1].transAxes,
        fontsize=6.8,
    )
    for label, axis in zip("abcd", axes.flat):
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "error_accumulation")


def _plot_mechanism(
    same_state_rows: Sequence[dict[str, Any]],
    scaling: Sequence[dict[str, Any]],
    slopes: dict[str, float],
    saved_times: Sequence[float],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for stride in REQUIRED_STRIDES:
        truth_rows = sorted(
            (
                row
                for row in same_state_rows
                if str(row["state_source"]) == "truth" and int(row["stride"]) == stride
            ),
            key=lambda row: int(row["start_frame"]),
        )
        on_policy_rows = sorted(
            (
                row
                for row in same_state_rows
                if str(row["state_source"]) == "on_policy"
                and int(row["stride"]) == stride
            ),
            key=lambda row: int(row["start_frame"]),
        )
        axes[0, 0].plot(
            [saved_times[int(row["start_frame"])] for row in truth_rows],
            [
                _as_float(row, "model_reference_cons_scaled_rel_l2_mean")
                for row in truth_rows
            ],
            color=COLORS[stride],
            marker=MARKERS[stride],
            markersize=3.5,
            linewidth=1.4,
            label=f"stride {stride}",
        )
        axes[0, 1].plot(
            [saved_times[int(row["start_frame"])] for row in on_policy_rows],
            [
                _as_float(row, "current_truth_cons_scaled_rel_l2_mean")
                for row in on_policy_rows
            ],
            color=COLORS[stride],
            marker=MARKERS[stride],
            markersize=3.5,
            linewidth=1.4,
            label=f"stride {stride}",
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Physical start time")
    axes[0, 0].set_ylabel("Truth-state one-call error")
    axes[0, 0].legend(ncol=2, frameon=False)
    axes[0, 1].set_xlabel("Physical start time")
    axes[0, 1].set_ylabel("Accumulated on-policy error before call")

    strides = np.asarray([row["stride"] for row in scaling], dtype=np.float64)
    axes[1, 0].loglog(
        strides,
        [row["truth_increment_state_normalized_mean"] for row in scaling],
        color="#000000",
        marker="o",
        linewidth=1.4,
        label=(
            f"truth increment (slope {slopes['truth_increment_log_stride_slope']:.2f})"
        ),
    )
    axes[1, 0].loglog(
        strides,
        [row["learned_state_error_mean"] for row in scaling],
        color="#D55E00",
        marker="s",
        linestyle="--",
        linewidth=1.4,
        label=(
            "learned first-call error "
            f"(slope {slopes['learned_state_error_log_stride_slope']:.2f})"
        ),
    )
    axes[1, 0].set_xlabel("Stride")
    axes[1, 0].set_ylabel("State-normalized magnitude")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        strides,
        [row["increment_relative_error_mean"] for row in scaling],
        color="#D55E00",
        marker="s",
        linewidth=1.4,
    )
    axes[1, 1].set_xlabel("Stride")
    axes[1, 1].set_ylabel("First-call error / truth-increment norm")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[1, 0].set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
    axes[1, 1].set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
    for label, axis in zip("abcd", axes.flat):
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "large_step_mechanism")


def _plot_casewise_optimum(
    statistics: Sequence[dict[str, Any]],
    winner_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
    saved_times: Sequence[float],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    frames = sorted({int(row["frame"]) for row in winner_rows})
    times = np.asarray([saved_times[frame] for frame in frames])
    bottom = np.zeros(len(frames), dtype=np.float64)
    for stride in (2, 4, 8):
        values = np.asarray(
            [
                next(
                    float(row["winner_fraction"])
                    for row in winner_rows
                    if int(row["frame"]) == frame and int(row["stride"]) == stride
                )
                for frame in frames
            ]
        )
        axes[0, 0].bar(
            times,
            values,
            width=0.032,
            bottom=bottom,
            color=COLORS[stride],
            edgecolor="white",
            linewidth=0.3,
            label=f"stride {stride}",
        )
        bottom += values
    axes[0, 0].set_xlabel("Physical time")
    axes[0, 0].set_ylabel("Casewise winner fraction")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].legend(frameon=False, ncol=3)

    oracle_sorted = sorted(oracle_rows, key=lambda row: int(row["frame"]))
    axes[0, 1].plot(
        [saved_times[int(row["frame"])] for row in oracle_sorted],
        [float(row["population_best_mean_error"]) for row in oracle_sorted],
        color="#000000",
        marker="o",
        linewidth=1.4,
        label="one population stride",
    )
    axes[0, 1].plot(
        [saved_times[int(row["frame"])] for row in oracle_sorted],
        [float(row["per_case_oracle_mean_error"]) for row in oracle_sorted],
        color="#D55E00",
        marker="s",
        linestyle="--",
        linewidth=1.4,
        label="per-case oracle (diagnostic)",
    )
    axes[0, 1].set_xlabel("Physical time")
    axes[0, 1].set_ylabel("Mean fixed-scale relative L2")
    axes[0, 1].legend(frameon=False)

    for stride in REQUIRED_STRIDES:
        series = [
            row
            for row in _stride_series(statistics, stride)
            if int(row["frame"]) % 8 == 0
        ]
        for axis, prefix in (
            (axes[1, 0], "smooth_error_mean"),
            (axes[1, 1], "shock_position_error_mean"),
        ):
            axis.plot(
                [row["physical_time"] for row in series],
                [row[prefix] for row in series],
                color=COLORS[stride],
                marker=MARKERS[stride],
                markersize=3.5,
                linewidth=1.4,
                label=f"stride {stride}",
            )
    axes[1, 0].set_xlabel("Physical time")
    axes[1, 0].set_ylabel("Smooth-region relative L2")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 1].set_xlabel("Physical time")
    axes[1, 1].set_ylabel("Top-two front-position MAE")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for label, axis in zip("abcd", axes.flat):
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "casewise_optimum_and_metric_conflict")


def _plot_runtime_pareto(
    pareto_rows: Sequence[dict[str, Any]],
    saved_times: Sequence[float],
    output_dir: Path,
) -> None:
    direct = [
        row
        for row in pareto_rows
        if _as_bool(row.get("claim_eligible"))
        and str(row.get("path")) == f"s{int(row['call_stride'])}_direct"
        and int(row["call_stride"]) in (2, 4, 8)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for row in direct:
        stride = int(row["call_stride"])
        horizon = int(row["horizon"])
        learned_x = _as_float(row, "learned_host_to_host_wall_seconds")
        learned_y = _as_float(row, "learned_fixed_scale_conservative_relative_l2_mean")
        reference_x = _as_float(row, "reference_batch1_wall_seconds_median")
        reference_y = _as_float(
            row, "reference_fixed_scale_conservative_relative_l2_mean"
        )
        axes[0].plot(
            [learned_x, reference_x],
            [learned_y, reference_y],
            color=COLORS[stride],
            alpha=0.35,
            linewidth=0.8,
        )
        axes[0].scatter(
            learned_x,
            learned_y,
            color=COLORS[stride],
            marker=MARKERS[stride],
            s=18 + 0.2 * horizon,
        )
        axes[0].scatter(
            reference_x,
            reference_y,
            facecolors="none",
            edgecolors=COLORS[stride],
            marker=MARKERS[stride],
            s=18 + 0.2 * horizon,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Batch-1 host-to-host wall time (s)")
    axes[0].set_ylabel("Mean fixed-scale relative L2")
    axes[0].text(
        0.03,
        0.04,
        "filled: learned; hollow: matched reference\nmarker size increases with horizon",
        transform=axes[0].transAxes,
        fontsize=6.8,
    )

    for stride in (2, 4, 8):
        series = sorted(
            (row for row in direct if int(row["call_stride"]) == stride),
            key=lambda row: int(row["horizon"]),
        )
        axes[1].plot(
            [saved_times[int(row["horizon"])] for row in series],
            [
                _as_float(
                    row,
                    "host_to_host_speedup_vs_accuracy_matched_reference",
                )
                for row in series
            ],
            color=COLORS[stride],
            marker=MARKERS[stride],
            linewidth=1.4,
            label=f"stride {stride}",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Physical horizon")
    axes[1].set_ylabel("Accuracy-matched host-to-host speedup")
    axes[1].legend(frameon=False)
    for label, axis in zip("ab", axes):
        axis.grid(alpha=0.2)
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "runtime_accuracy_pareto")


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    if args.error_budget <= 0.0:
        raise ValueError("--error-budget must be positive")
    required = {
        "d032_curves": args.d032_dir / "direct_curves.jsonl",
        "d032_survival": args.d032_dir / "error_budget_survival.csv",
        "d032_contract": args.d032_dir / "contract.json",
        "d033_summary": args.d033_dir / "same_state_summary.csv",
        "d033_report": args.d033_dir / "report.json",
        "d034_pareto": args.d034_dir / "accuracy_matched_pareto.csv",
        "d034_contract": args.d034_dir / "contract.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direct_rows = _read_jsonl(required["d032_curves"])
    survival_rows = _read_csv(required["d032_survival"])
    same_state_rows = _read_csv(required["d033_summary"])
    pareto_rows = _read_csv(required["d034_pareto"])
    d032_contract = json.loads(required["d032_contract"].read_text(encoding="utf-8"))
    saved_times = [float(value) for value in d032_contract["saved_times"]]

    statistics = direct_curve_statistics(direct_rows)
    scaling, slopes = first_call_scaling(direct_rows)
    winner_rows, oracle_rows = casewise_frontier_rows(direct_rows)
    reliability = budget_reliability_rows(survival_rows, budget=args.error_budget)
    for row in reliability:
        row["physical_time"] = saved_times[int(row["frame"])] - saved_times[0]

    _write_csv(args.output_dir / "curve_statistics.csv", statistics)
    _write_csv(args.output_dir / "first_call_scaling.csv", scaling)
    _write_csv(args.output_dir / "winner_shares.csv", winner_rows)
    _write_csv(args.output_dir / "oracle_gap.csv", oracle_rows)
    _write_csv(args.output_dir / "error_budget_reliability.csv", reliability)
    _plot_error_accumulation(statistics, reliability, args.output_dir)
    _plot_mechanism(same_state_rows, scaling, slopes, saved_times, args.output_dir)
    _plot_casewise_optimum(
        statistics,
        winner_rows,
        oracle_rows,
        saved_times,
        args.output_dir,
    )
    _plot_runtime_pareto(pareto_rows, saved_times, args.output_dir)

    final_oracle = max(oracle_rows, key=lambda row: int(row["frame"]))
    final_winners = {
        str(row["stride"]): row["winner_fraction"]
        for row in winner_rows
        if int(row["frame"]) == int(final_oracle["frame"])
    }
    report = {
        "status": "complete",
        "analysis_role": "frozen_registered_artifact_visualization",
        "new_learned_training_runs": 0,
        "error_budget": args.error_budget,
        "first_call_scaling_slopes": slopes,
        "final_frame": int(final_oracle["frame"]),
        "final_physical_time": saved_times[int(final_oracle["frame"])],
        "final_population_best_stride": int(final_oracle["population_best_stride"]),
        "final_per_case_oracle_relative_improvement": float(
            final_oracle["oracle_relative_improvement"]
        ),
        "final_casewise_winner_fractions": final_winners,
        "figure_stems": [
            "error_accumulation",
            "large_step_mechanism",
            "casewise_optimum_and_metric_conflict",
            "runtime_accuracy_pareto",
        ],
        "input_sha256": {name: sha256_file(path) for name, path in required.items()},
        "definitions": {
            "case_band": "interquartile range across held-out cases, not seed uncertainty",
            "failure_accounting": "raw failures leave later error summaries missing and reduce completion",
            "oracle": "post-hoc per-case minimum over frozen strides 2, 4, and 8; diagnostic only",
            "first_call_truth_increment": (
                "recovered per case as state-normalized increment residual divided by increment-relative error"
            ),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return report


def operating_envelope_rows(
    statistics: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]],
    ripple_statistics: Sequence[dict[str, Any]],
    pareto_rows: Sequence[dict[str, Any]],
    *,
    horizons: Sequence[int] = CLOSEOUT_HORIZONS,
    primary_models: dict[int, str] = CLOSEOUT_PRIMARY_MODELS,
) -> list[dict[str, Any]]:
    """Select metric-specific winners without collapsing ties."""

    statistic_index = {
        (int(row["stride"]), int(row["frame"])): row for row in statistics
    }
    reliability_index = {
        (
            float(row["error_budget"]),
            int(row["stride"]),
            int(row["frame"]),
        ): row
        for row in reliability
    }
    ripple_index = {
        (str(row["model"]), int(row["frame"])): row for row in ripple_statistics
    }
    pareto_index = {
        (int(row["call_stride"]), int(row["horizon"])): row
        for row in pareto_rows
        if _as_bool(row.get("claim_eligible"))
        and str(row.get("path")) == f"s{int(row['call_stride'])}_direct"
    }

    output: list[dict[str, Any]] = []

    def add_objective(
        frame: int,
        objective: str,
        label: str,
        source: str,
        direction: str,
        values: dict[int, float],
        *,
        transformation: str = "identity",
    ) -> None:
        finite_values = {
            int(stride): float(value)
            for stride, value in values.items()
            if np.isfinite(value)
        }
        if not finite_values:
            return
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"unsupported objective direction: {direction}")
        best_value = (
            min(finite_values.values())
            if direction == "minimize"
            else max(finite_values.values())
        )
        winners = sorted(
            stride
            for stride, value in finite_values.items()
            if math.isclose(value, best_value, rel_tol=1.0e-9, abs_tol=1.0e-12)
        )
        remaining = [
            (stride, value)
            for stride, value in finite_values.items()
            if stride not in winners
        ]
        remaining.sort(key=lambda item: item[1], reverse=direction == "maximize")
        runner_stride, runner_value = (
            remaining[0] if remaining else (None, float("nan"))
        )
        signed_advantage = (
            runner_value - best_value
            if direction == "minimize"
            else best_value - runner_value
        )
        record: dict[str, Any] = {
            "frame": frame,
            "objective": objective,
            "objective_label": label,
            "source": source,
            "direction": direction,
            "transformation": transformation,
            "eligible_strides": "|".join(str(value) for value in sorted(finite_values)),
            "num_winners": len(winners),
            "winner_strides": "|".join(str(value) for value in winners),
            "winner_stride": winners[0] if len(winners) == 1 else None,
            "best_value": best_value,
            "runner_up_stride": runner_stride,
            "runner_up_value": runner_value,
            "signed_advantage_over_runner_up": signed_advantage,
            "winner_includes_tested_boundary": max(finite_values) in winners,
            "unique_winner_at_tested_boundary": (
                len(winners) == 1 and winners[0] == max(finite_values)
            ),
        }
        for stride in REQUIRED_STRIDES:
            record[f"stride_{stride}_value"] = finite_values.get(stride, float("nan"))
        output.append(record)

    direct_objectives = (
        (
            "endpoint_global_error",
            "Endpoint global error",
            "error_mean",
            "minimize",
        ),
        (
            "time_mean_global_error",
            "Time-mean global error",
            "time_mean_error_mean",
            "minimize",
        ),
        (
            "smooth_region_error",
            "Smooth-region error",
            "smooth_error_mean",
            "minimize",
        ),
        (
            "front_position_error",
            "Front-position error",
            "shock_position_error_mean",
            "minimize",
        ),
        (
            "raw_completion",
            "Raw-admissible fraction",
            "raw_completion_fraction",
            "maximize",
        ),
    )
    for frame in horizons:
        for objective, label, key, direction in direct_objectives:
            add_objective(
                frame,
                objective,
                label,
                "D032",
                direction,
                {
                    stride: _as_float(statistic_index.get((stride, frame), {}), key)
                    for stride in REQUIRED_STRIDES
                },
            )
        for budget in CLOSEOUT_ERROR_BUDGETS:
            add_objective(
                frame,
                f"error_budget_{budget:.2f}_reliability",
                f"Reliability at error budget {budget:.2f}",
                "D032",
                "maximize",
                {
                    stride: _as_float(
                        reliability_index.get((budget, stride, frame), {}),
                        "reliability_fraction",
                    )
                    for stride in REQUIRED_STRIDES
                },
            )
        add_objective(
            frame,
            "away_front_d2_error",
            "Away-front second-derivative error",
            "D036",
            "minimize",
            {
                stride: _as_float(
                    ripple_index.get((model, frame), {}),
                    "smooth_error_d2_rms_mean",
                )
                for stride, model in primary_models.items()
            },
        )
        add_objective(
            frame,
            "away_front_tv_deviation",
            "Away-front TV deviation from truth",
            "D036",
            "minimize",
            {
                stride: abs(
                    _as_float(
                        ripple_index.get((model, frame), {}),
                        "smooth_state_tv_ratio_mean",
                    )
                    - 1.0
                )
                for stride, model in primary_models.items()
            },
            transformation="absolute_distance_from_one",
        )
        add_objective(
            frame,
            "host_to_host_latency",
            "Host-to-host batch-1 latency",
            "D034",
            "minimize",
            {
                stride: _as_float(
                    pareto_index.get((stride, frame), {}),
                    "learned_host_to_host_wall_seconds",
                )
                for stride in (2, 4, 8)
            },
        )
    return output


def closeout_paired_statistics(
    direct_rows: Sequence[dict[str, Any]],
    envelope_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap each unique D032 winner against its aggregate runner-up."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    raw_metrics = {
        "endpoint_global_error": "fixed_scale_conservative_relative_l2",
        "time_mean_global_error": "time_mean_fixed_scale_conservative_relative_l2",
        "smooth_region_error": "smooth_region_relative_l2",
        "front_position_error": "shock_top2_position_mae",
    }
    indexed = {
        (int(row["stride"]), int(row["frame"]), int(row["case_id"])): row
        for row in direct_rows
        if str(row.get("path")) == f"s{int(row['stride'])}_direct"
        and _as_bool(row.get("completed_step"))
    }
    rng = np.random.default_rng(bootstrap_seed)
    output: list[dict[str, Any]] = []
    for envelope in envelope_rows:
        objective = str(envelope["objective"])
        if objective not in raw_metrics or int(envelope["num_winners"]) != 1:
            continue
        winner = int(envelope["winner_stride"])
        runner = envelope.get("runner_up_stride")
        if runner is None:
            continue
        runner = int(runner)
        frame = int(envelope["frame"])
        winner_cases = {
            case
            for stride, row_frame, case in indexed
            if stride == winner and row_frame == frame
        }
        runner_cases = {
            case
            for stride, row_frame, case in indexed
            if stride == runner and row_frame == frame
        }
        common_cases = sorted(winner_cases & runner_cases)
        metric = raw_metrics[objective]
        winner_values = np.asarray(
            [
                _as_float(indexed[(winner, frame, case)], metric)
                for case in common_cases
            ],
            dtype=np.float64,
        )
        runner_values = np.asarray(
            [
                _as_float(indexed[(runner, frame, case)], metric)
                for case in common_cases
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(winner_values) & np.isfinite(runner_values)
        winner_values = winner_values[finite]
        runner_values = runner_values[finite]
        if not winner_values.size:
            continue
        difference = winner_values - runner_values
        indices = rng.integers(
            0,
            winner_values.size,
            size=(bootstrap_replicates, winner_values.size),
        )
        bootstrap_difference = np.mean(difference[indices], axis=1)
        bootstrap_ratio = np.mean(winner_values[indices], axis=1) / np.maximum(
            np.mean(runner_values[indices], axis=1), EPS
        )
        winner_mean = float(np.mean(winner_values))
        runner_mean = float(np.mean(runner_values))
        output.append(
            {
                "frame": frame,
                "objective": objective,
                "winner_stride": winner,
                "runner_up_stride": runner,
                "num_common_cases": int(winner_values.size),
                "winner_common_case_mean": winner_mean,
                "runner_up_common_case_mean": runner_mean,
                "winner_over_runner_up_mean_ratio": winner_mean / max(runner_mean, EPS),
                "mean_ratio_bootstrap_95_lower": float(
                    np.quantile(bootstrap_ratio, 0.025)
                ),
                "mean_ratio_bootstrap_95_upper": float(
                    np.quantile(bootstrap_ratio, 0.975)
                ),
                "paired_mean_difference": float(np.mean(difference)),
                "paired_difference_bootstrap_95_lower": float(
                    np.quantile(bootstrap_difference, 0.025)
                ),
                "paired_difference_bootstrap_95_upper": float(
                    np.quantile(bootstrap_difference, 0.975)
                ),
                "winner_better_case_fraction": float(np.mean(difference < 0.0)),
                "aggregate_selection_preserved_on_common_cases": winner_mean
                <= runner_mean,
            }
        )
    return output


def closeout_crossover_rows(
    statistics: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Locate first and sustained aggregate crossovers at common endpoints."""

    indexed = {(int(row["stride"]), int(row["frame"])): row for row in statistics}
    output: list[dict[str, Any]] = []
    metrics = {
        "endpoint_global_error": "error_mean",
        "time_mean_global_error": "time_mean_error_mean",
    }
    for metric, key in metrics.items():
        for small_index, smaller_stride in enumerate(REQUIRED_STRIDES[:-1]):
            for larger_stride in REQUIRED_STRIDES[small_index + 1 :]:
                frames = sorted(
                    frame
                    for stride, frame in indexed
                    if stride == larger_stride
                    and (smaller_stride, frame) in indexed
                    and frame <= max(CLOSEOUT_HORIZONS)
                    and frame % larger_stride == 0
                )
                if not frames:
                    continue
                ratios = [
                    _as_float(indexed[(larger_stride, frame)], key)
                    / max(_as_float(indexed[(smaller_stride, frame)], key), EPS)
                    for frame in frames
                ]
                advantages = [ratio < 1.0 for ratio in ratios]
                first_index = next(
                    (index for index, advantage in enumerate(advantages) if advantage),
                    None,
                )
                sustained_index = next(
                    (
                        index
                        for index in range(len(advantages))
                        if advantages[index] and all(advantages[index:])
                    ),
                    None,
                )
                output.append(
                    {
                        "metric": metric,
                        "smaller_stride": smaller_stride,
                        "larger_stride": larger_stride,
                        "num_common_endpoints": len(frames),
                        "first_advantage_frame": (
                            frames[first_index] if first_index is not None else None
                        ),
                        "first_sustained_advantage_frame": (
                            frames[sustained_index]
                            if sustained_index is not None
                            else None
                        ),
                        "larger_calls_at_sustained_crossover": (
                            frames[sustained_index] // larger_stride
                            if sustained_index is not None
                            else None
                        ),
                        "smaller_calls_at_sustained_crossover": (
                            frames[sustained_index] // smaller_stride
                            if sustained_index is not None
                            else None
                        ),
                        "larger_over_smaller_ratio_at_final_common_endpoint": ratios[
                            -1
                        ],
                        "final_common_frame": frames[-1],
                    }
                )
    return output


def closeout_consistency_audit(
    d032_contract: dict[str, Any],
    wave_occupancy: dict[str, Any],
    d033_rows: Sequence[dict[str, Any]],
    d036_rows: Sequence[dict[str, Any]],
    d038_rows: Sequence[dict[str, Any]],
    teacher_modal_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Check the cross-artifact invariants used by the closeout claims."""

    checkpoint_strides = sorted(int(value) for value in d032_contract["checkpoints"])
    saved_times = [float(value) for value in d032_contract["saved_times"]]
    case_ids = [int(value) for value in d032_contract["case_ids"]]
    h96_wave = wave_occupancy["horizons"].get("96", {})

    duplicate_metrics = (
        "state_cons_scaled_rel_l2_mean",
        "smooth_error_d2_rms_mean",
        "smooth_state_tv_ratio_mean",
        "raw_completion_fraction",
    )
    d036_index = {(str(row["model"]), int(row["frame"])): row for row in d036_rows}
    d038_index = {(str(row["model"]), int(row["frame"])): row for row in d038_rows}
    common_duplicate_keys = sorted(set(d036_index) & set(d038_index))
    duplicate_differences = [
        abs(_as_float(d036_index[key], metric) - _as_float(d038_index[key], metric))
        for key in common_duplicate_keys
        for metric in duplicate_metrics
    ]
    duplicate_max_abs = max(duplicate_differences, default=float("inf"))

    def mechanism_values(state_source: str, metric: str) -> list[float]:
        index = {
            int(row["stride"]): _as_float(row, metric)
            for row in d033_rows
            if str(row["state_source"]) == state_source
            and int(row["start_frame"]) == 32
        }
        return [index.get(stride, float("nan")) for stride in REQUIRED_STRIDES]

    truth_one_call = mechanism_values(
        "truth", "model_reference_cons_scaled_rel_l2_mean"
    )
    on_policy_before_call = mechanism_values(
        "on_policy", "current_truth_cons_scaled_rel_l2_mean"
    )
    truth_monotone = all(np.isfinite(truth_one_call)) and all(
        first < second for first, second in zip(truth_one_call, truth_one_call[1:])
    )
    on_policy_monotone = all(np.isfinite(on_policy_before_call)) and all(
        first > second
        for first, second in zip(on_policy_before_call, on_policy_before_call[1:])
    )

    modal_index = {
        (str(row["model"]), str(row["band"])): _as_float(
            row, "modal_relative_error_rms"
        )
        for row in teacher_modal_rows
    }
    stride8_models = sorted(
        {model for model, _band in modal_index if model.startswith("s8_seed")}
    )
    modal_ratios = {
        f"{model}:{band}": value / max(modal_index[("s4", band)], EPS)
        for (model, band), value in modal_index.items()
        if model in stride8_models and ("s4", band) in modal_index
    }
    modal_injection_consistent = bool(modal_ratios) and all(
        value > 1.0 for value in modal_ratios.values()
    )

    checks = {
        "checkpoint_strides_are_1_2_4_8": checkpoint_strides == list(REQUIRED_STRIDES),
        "checkpoint_hashes_are_present_and_unique": len(
            {
                str(d032_contract["checkpoints"][str(stride)]["sha256"])
                for stride in REQUIRED_STRIDES
            }
        )
        == len(REQUIRED_STRIDES),
        "full_test_split_has_64_unique_cases": len(case_ids) == 64
        and len(set(case_ids)) == 64,
        "saved_time_contract_has_101_common_frames": len(saved_times) == 101
        and bool(d032_contract.get("saved_times_common_across_cases")),
        "h96_is_inside_saved_time_support": len(saved_times) > 96
        and saved_times[96] < saved_times[-1],
        "h96_waves_remain_active": _as_float(h96_wave, "interior_active_fraction")
        >= 0.5
        and _as_float(h96_wave, "state_change_relative_l2_median") > 0.0,
        "d036_d038_duplicate_rows_match": bool(common_duplicate_keys)
        and set(d036_index) == set(d038_index)
        and duplicate_max_abs <= 1.0e-12,
        "truth_state_one_call_error_increases_with_stride_at_h32": truth_monotone,
        "on_policy_error_before_call_decreases_with_stride_at_h32": on_policy_monotone,
        "stride8_teacher_modal_defect_exceeds_stride4_in_every_band_and_seed": modal_injection_consistent,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "details": {
            "checkpoint_strides": checkpoint_strides,
            "num_test_cases": len(case_ids),
            "num_saved_frames": len(saved_times),
            "h96_interior_active_fraction": _as_float(
                h96_wave, "interior_active_fraction"
            ),
            "h96_state_change_relative_l2_median": _as_float(
                h96_wave, "state_change_relative_l2_median"
            ),
            "num_duplicate_summary_rows": len(common_duplicate_keys),
            "duplicate_summary_max_abs_difference": duplicate_max_abs,
            "h32_truth_one_call_error_by_stride": dict(
                zip(REQUIRED_STRIDES, truth_one_call)
            ),
            "h32_on_policy_error_before_call_by_stride": dict(
                zip(REQUIRED_STRIDES, on_policy_before_call)
            ),
            "stride8_over_stride4_teacher_modal_ratio_range": [
                min(modal_ratios.values(), default=float("nan")),
                max(modal_ratios.values(), default=float("nan")),
            ],
        },
    }


def _plot_operating_envelope(
    envelope_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    objective_order = (
        "endpoint_global_error",
        "time_mean_global_error",
        "front_position_error",
        "away_front_d2_error",
        "away_front_tv_deviation",
        "error_budget_0.05_reliability",
        "host_to_host_latency",
    )
    labels = {
        str(row["objective"]): str(row["objective_label"]) for row in envelope_rows
    }
    indexed = {(str(row["objective"]), int(row["frame"])): row for row in envelope_rows}
    paired_index = {
        (str(row["objective"]), int(row["frame"])): row for row in paired_rows
    }
    stride_codes = {stride: index for index, stride in enumerate(REQUIRED_STRIDES)}
    matrix = np.full((len(objective_order), len(CLOSEOUT_HORIZONS)), np.nan)
    annotations: list[list[str]] = []
    for row_index, objective in enumerate(objective_order):
        row_annotations: list[str] = []
        for column_index, frame in enumerate(CLOSEOUT_HORIZONS):
            record = indexed.get((objective, frame))
            if record is None:
                row_annotations.append("--")
                continue
            winners = [int(value) for value in str(record["winner_strides"]).split("|")]
            if len(winners) == 1:
                matrix[row_index, column_index] = stride_codes[winners[0]]
                paired = paired_index.get((objective, frame))
                uncertain = paired is not None and (
                    float(paired["paired_difference_bootstrap_95_lower"])
                    <= 0.0
                    <= float(paired["paired_difference_bootstrap_95_upper"])
                )
                suffix = ""
                if bool(record["unique_winner_at_tested_boundary"]):
                    suffix += "*"
                if uncertain:
                    suffix += "†"
                row_annotations.append(f"s{winners[0]}{suffix}")
            else:
                row_annotations.append("/".join(str(stride) for stride in winners))
        annotations.append(row_annotations)

    colors = [COLORS[stride] for stride in REQUIRED_STRIDES]
    color_map = ListedColormap(colors)
    color_map.set_bad("#F0F0F0")
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap=color_map, vmin=-0.5, vmax=3.5
    )
    for row_index, row_annotations in enumerate(annotations):
        for column_index, annotation in enumerate(row_annotations):
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )
    axis.set_xticks(
        range(len(CLOSEOUT_HORIZONS)),
        labels=[f"H{frame}" for frame in CLOSEOUT_HORIZONS],
    )
    axis.set_yticks(
        range(len(objective_order)),
        labels=[labels.get(objective, objective) for objective in objective_order],
    )
    axis.set_xlabel("Common physical endpoint")
    axis.set_title("Metric-dependent learned flow-map operating envelope")
    colorbar = figure.colorbar(image, ax=axis, ticks=range(len(REQUIRED_STRIDES)))
    colorbar.ax.set_yticklabels([f"stride {stride}" for stride in REQUIRED_STRIDES])
    colorbar.set_label("Unique winning stride")
    axis.text(
        0.0,
        -0.17,
        "gray: aggregate tie; *: tested boundary; †: paired case-bootstrap CI crosses equality",
        transform=axis.transAxes,
        fontsize=7,
    )
    _save_figure(figure, output_dir, "operating_envelope_map")


def _plot_closeout_pareto(
    statistics: Sequence[dict[str, Any]],
    ripple_statistics: Sequence[dict[str, Any]],
    pareto_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    frame = max(CLOSEOUT_HORIZONS)
    statistic_index = {
        (int(row["stride"]), int(row["frame"])): row for row in statistics
    }
    ripple_index = {
        (str(row["model"]), int(row["frame"])): row for row in ripple_statistics
    }
    runtime_rows = {
        int(row["call_stride"]): row
        for row in pareto_rows
        if _as_bool(row.get("claim_eligible"))
        and str(row.get("path")) == f"s{int(row['call_stride'])}_direct"
        and int(row["horizon"]) == frame
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), constrained_layout=True)
    for stride, model in CLOSEOUT_PRIMARY_MODELS.items():
        global_error = _as_float(statistic_index[(stride, frame)], "error_mean")
        ripple_row = ripple_index[(model, frame)]
        d2_error = _as_float(ripple_row, "smooth_error_d2_rms_mean")
        tv_deviation = abs(_as_float(ripple_row, "smooth_state_tv_ratio_mean") - 1.0)
        for axis, y_value in ((axes[0], d2_error), (axes[1], tv_deviation)):
            axis.scatter(
                global_error,
                y_value,
                color=COLORS[stride],
                marker=MARKERS[stride],
                s=34,
            )
            axis.annotate(
                f"s{stride}",
                (global_error, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    for stride, row in sorted(runtime_rows.items()):
        axes[2].scatter(
            _as_float(row, "learned_host_to_host_wall_seconds"),
            _as_float(row, "learned_fixed_scale_conservative_relative_l2_mean"),
            color=COLORS[stride],
            marker=MARKERS[stride],
            s=34,
        )
        axes[2].annotate(
            f"s{stride}",
            (
                _as_float(row, "learned_host_to_host_wall_seconds"),
                _as_float(row, "learned_fixed_scale_conservative_relative_l2_mean"),
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_xlabel("H96 global relative L2")
    axes[0].set_ylabel("Away-front second-derivative error")
    axes[1].set_xlabel("H96 global relative L2")
    axes[1].set_ylabel("Away-front |TV ratio - 1|")
    axes[2].set_xlabel("H96 host-to-host batch-1 time (s)")
    axes[2].set_ylabel("H96 relative L2 (timed subset)")
    axes[2].set_xscale("log")
    for label, axis in zip("abc", axes):
        axis.grid(alpha=0.2)
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "global_local_runtime_pareto")


def run_closeout(args: argparse.Namespace) -> dict[str, Any]:
    """Build the final zero-training Line-1 synthesis from frozen tables."""

    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    required = {
        "d032_curves": args.d032_dir / "direct_curves.jsonl",
        "d032_survival": args.d032_dir / "error_budget_survival.csv",
        "d032_contract": args.d032_dir / "contract.json",
        "d032_wave_occupancy": args.d032_dir / "wave_occupancy.json",
        "d033_summary": args.d033_dir / "same_state_summary.csv",
        "d033_report": args.d033_dir / "report.json",
        "d034_pareto": args.d034_dir / "accuracy_matched_pareto.csv",
        "d034_contract": args.d034_dir / "contract.json",
        "d036_common": args.d036_dir / "common_endpoint_summary.csv",
        "d036_paired": args.d036_dir / "paired_s8_over_s4.csv",
        "d036_report": args.d036_dir / "report.json",
        "d038_common": args.d038_dir / "common_endpoint_summary.csv",
        "d038_shapes": args.d038_dir / "rollout_modal_shape_summary.csv",
        "d038_teacher": args.d038_dir / "teacher_pooled_modal_band_summary.csv",
        "d038_report": args.d038_dir / "report.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen closeout inputs: {missing}")

    direct_rows = _read_jsonl(required["d032_curves"])
    survival_rows = _read_csv(required["d032_survival"])
    d033_rows = _read_csv(required["d033_summary"])
    pareto_rows = _read_csv(required["d034_pareto"])
    d036_rows = _read_csv(required["d036_common"])
    d036_paired_rows = _read_csv(required["d036_paired"])
    d038_rows = _read_csv(required["d038_common"])
    modal_shape_rows = _read_csv(required["d038_shapes"])
    teacher_modal_rows = _read_csv(required["d038_teacher"])
    d032_contract = json.loads(required["d032_contract"].read_text(encoding="utf-8"))
    wave_occupancy = json.loads(
        required["d032_wave_occupancy"].read_text(encoding="utf-8")
    )
    d036_report = json.loads(required["d036_report"].read_text(encoding="utf-8"))
    d038_report = json.loads(required["d038_report"].read_text(encoding="utf-8"))
    saved_times = [float(value) for value in d032_contract["saved_times"]]

    statistics = direct_curve_statistics(direct_rows)
    reliability: list[dict[str, Any]] = []
    for budget in CLOSEOUT_ERROR_BUDGETS:
        budget_rows = budget_reliability_rows(survival_rows, budget=budget)
        for row in budget_rows:
            row["error_budget"] = budget
            row["physical_time"] = saved_times[int(row["frame"])] - saved_times[0]
        reliability.extend(budget_rows)
    envelope = operating_envelope_rows(
        statistics,
        reliability,
        d036_rows,
        pareto_rows,
    )
    paired = closeout_paired_statistics(
        direct_rows,
        envelope,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    crossovers = closeout_crossover_rows(statistics)
    audit = closeout_consistency_audit(
        d032_contract,
        wave_occupancy,
        d033_rows,
        d036_rows,
        d038_rows,
        teacher_modal_rows,
    )
    if audit["status"] != "pass":
        failed = [name for name, passed in audit["checks"].items() if not bool(passed)]
        raise RuntimeError(f"frozen closeout audit failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "operating_envelope.csv", envelope)
    _write_csv(args.output_dir / "paired_winner_uncertainty.csv", paired)
    _write_csv(args.output_dir / "crossover_summary.csv", crossovers)
    _write_csv(args.output_dir / "curve_statistics.csv", statistics)
    _write_csv(args.output_dir / "error_budget_reliability.csv", reliability)
    _plot_operating_envelope(envelope, paired, args.output_dir)
    _plot_closeout_pareto(statistics, d036_rows, pareto_rows, args.output_dir)

    envelope_index = {
        (str(row["objective"]), int(row["frame"])): row for row in envelope
    }
    endpoint_winners = {
        str(frame): envelope_index[("endpoint_global_error", frame)]["winner_strides"]
        for frame in CLOSEOUT_HORIZONS
    }
    time_mean_winners = {
        str(frame): envelope_index[("time_mean_global_error", frame)]["winner_strides"]
        for frame in CLOSEOUT_HORIZONS
    }
    h96_paired = {
        str(row["metric"]): row
        for row in d036_paired_rows
        if int(row["frame"]) == 96
        and str(row["numerator"]) == CLOSEOUT_PRIMARY_MODELS[8]
        and str(row["denominator"]) == CLOSEOUT_PRIMARY_MODELS[4]
    }
    modal_shape_index = {
        (str(row["model"]), int(row["frame"])): row for row in modal_shape_rows
    }
    modal_total_error_ratios = {
        str(frame): _as_float(
            modal_shape_index[(CLOSEOUT_PRIMARY_MODELS[8], frame)],
            "modal_relative_error_rms",
        )
        / max(
            _as_float(
                modal_shape_index[(CLOSEOUT_PRIMARY_MODELS[4], frame)],
                "modal_relative_error_rms",
            ),
            EPS,
        )
        for frame in (8, 32, 96)
    }
    h96_endpoint = envelope_index[("endpoint_global_error", 96)]
    h96_ripple = envelope_index[("away_front_d2_error", 96)]
    report = {
        "status": "complete",
        "experiment": "D039_frozen_line1_operating_envelope_closeout",
        "analysis_role": "registered_frozen_artifact_synthesis",
        "new_learned_training_runs": 0,
        "new_checkpoint_evaluations": 0,
        "new_solver_runs": 0,
        "audit": audit,
        "contract": {
            "horizons": list(CLOSEOUT_HORIZONS),
            "error_budgets": list(CLOSEOUT_ERROR_BUDGETS),
            "primary_models": CLOSEOUT_PRIMARY_MODELS,
            "bootstrap": {
                "unit": "paired held-out case",
                "replicates": args.bootstrap_replicates,
                "seed": args.bootstrap_seed,
                "is_seed_uncertainty": False,
            },
            "checkpoint_sha256_by_stride": {
                str(stride): d032_contract["checkpoints"][str(stride)]["sha256"]
                for stride in REQUIRED_STRIDES
            },
        },
        "operating_envelope": {
            "endpoint_global_error_winner_strides": endpoint_winners,
            "time_mean_global_error_winner_strides": time_mean_winners,
            "h96_endpoint_global_error": h96_endpoint,
            "h96_away_front_d2_error": h96_ripple,
            "h96_is_right_censored_at_stride8": bool(
                h96_endpoint["unique_winner_at_tested_boundary"]
            ),
        },
        "mechanism": {
            "crossover_rows": crossovers,
            "rollout_total_modal_error_s8_over_s4": modal_total_error_ratios,
            "teacher_modal_ratio_range": audit["details"][
                "stride8_over_stride4_teacher_modal_ratio_range"
            ],
        },
        "global_local_conflict_at_h96": {
            metric: h96_paired[metric]
            for metric in (
                "state_cons_scaled_rel_l2",
                "smooth_error_d2_rms",
                "smooth_state_tv_ratio",
            )
        },
        "source_classifications": {
            "d036": d036_report.get("classification"),
            "d038": d038_report.get("classification"),
        },
        "claim_boundary": [
            "The operating envelope is conditioned on this 1D Euler distribution, FNO architecture, training contract, horizon, metric, and hardware.",
            "Stride 8 is the tested boundary, so the medium-horizon optimum is right-censored.",
            "Only stride 8 has independent training-seed repeats; paired case bootstrap intervals are not seed uncertainty.",
            "Thresholded ripple lifetime is a diagnostic, not raw-admissible or universal physical lifetime.",
            "No universal neural CFL, timestep optimum, or rejection of autoregressive neural PDE prediction is claimed.",
        ],
        "theory_ready": True,
        "figure_stems": [
            "operating_envelope_map",
            "global_local_runtime_pareto",
        ],
        "input_sha256": {name: sha256_file(path) for name, path in required.items()},
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return report


def _ripple_frame(row: dict[str, Any]) -> int:
    return int(float(row["target_frame"]))


def _ripple_case(row: dict[str, Any]) -> int:
    return int(float(row["case_id"]))


def _is_ripple_rollout_row(
    row: dict[str, Any],
    *,
    horizon: int,
    common_frame_step: int,
) -> bool:
    if str(row.get("mode")) != "autoregressive":
        return False
    frame = _ripple_frame(row)
    return 0 < frame <= horizon and frame % common_frame_step == 0


def ripple_endpoint_statistics(
    rows: Sequence[dict[str, Any]],
    *,
    horizon: int,
    common_frame_step: int,
    saved_frame_dt: float,
) -> list[dict[str, Any]]:
    """Aggregate only raw-admissible cases at common physical endpoints."""

    selected = [
        row
        for row in rows
        if _is_ripple_rollout_row(
            row,
            horizon=horizon,
            common_frame_step=common_frame_step,
        )
    ]
    case_counts = {
        model: len({_ripple_case(row) for row in selected if row["model"] == model})
        for model in sorted({str(row["model"]) for row in selected})
    }
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(str(row["model"]), _ripple_frame(row))].append(row)
    output: list[dict[str, Any]] = []
    for (model, frame), group in sorted(groups.items()):
        valid = [row for row in group if _as_bool(row.get("proposal_valid"))]
        record: dict[str, Any] = {
            "model": model,
            "stride": int(float(group[0]["stride"])),
            "frame": frame,
            "physical_time": frame * saved_frame_dt,
            "num_cases": case_counts[model],
            "num_raw_admissible": len(valid),
            "raw_completion_fraction": len(valid) / max(case_counts[model], 1),
        }
        for metric in RIPPLE_METRICS:
            summary = _finite_summary(_as_float(row, metric) for row in valid)
            record.update(
                {f"{metric}_{name}": value for name, value in summary.items()}
            )
        output.append(record)
    return output


def ripple_paired_statistics(
    rows: Sequence[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    horizon: int,
    common_frame_step: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Return paired case contrasts with case-resampling intervals."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    selected = [
        row
        for row in rows
        if _is_ripple_rollout_row(
            row,
            horizon=horizon,
            common_frame_step=common_frame_step,
        )
        and _as_bool(row.get("proposal_valid"))
        and str(row["model"]) in {numerator, denominator}
    ]
    indexed = {
        (str(row["model"]), _ripple_frame(row), _ripple_case(row)): row
        for row in selected
    }
    rng = np.random.default_rng(bootstrap_seed)
    output: list[dict[str, Any]] = []
    for frame in range(common_frame_step, horizon + 1, common_frame_step):
        numerator_cases = {
            case
            for model, row_frame, case in indexed
            if model == numerator and row_frame == frame
        }
        denominator_cases = {
            case
            for model, row_frame, case in indexed
            if model == denominator and row_frame == frame
        }
        common_cases = sorted(numerator_cases & denominator_cases)
        for metric in RIPPLE_METRICS:
            numerator_values = np.asarray(
                [
                    _as_float(indexed[(numerator, frame, case)], metric)
                    for case in common_cases
                ],
                dtype=np.float64,
            )
            denominator_values = np.asarray(
                [
                    _as_float(indexed[(denominator, frame, case)], metric)
                    for case in common_cases
                ],
                dtype=np.float64,
            )
            finite = np.isfinite(numerator_values) & np.isfinite(denominator_values)
            numerator_values = numerator_values[finite]
            denominator_values = denominator_values[finite]
            if not numerator_values.size:
                continue
            difference = numerator_values - denominator_values
            indices = rng.integers(
                0,
                numerator_values.size,
                size=(bootstrap_replicates, numerator_values.size),
            )
            difference_bootstrap = np.mean(difference[indices], axis=1)
            ratio_bootstrap = np.mean(numerator_values[indices], axis=1) / np.maximum(
                np.mean(denominator_values[indices], axis=1),
                EPS,
            )
            output.append(
                {
                    "numerator": numerator,
                    "denominator": denominator,
                    "frame": frame,
                    "metric": metric,
                    "common_cases": int(numerator_values.size),
                    "numerator_mean": float(np.mean(numerator_values)),
                    "denominator_mean": float(np.mean(denominator_values)),
                    "mean_ratio": float(
                        np.mean(numerator_values)
                        / max(float(np.mean(denominator_values)), EPS)
                    ),
                    "mean_ratio_bootstrap_95_lower": float(
                        np.quantile(ratio_bootstrap, 0.025)
                    ),
                    "mean_ratio_bootstrap_95_upper": float(
                        np.quantile(ratio_bootstrap, 0.975)
                    ),
                    "paired_mean_difference": float(np.mean(difference)),
                    "paired_difference_bootstrap_95_lower": float(
                        np.quantile(difference_bootstrap, 0.025)
                    ),
                    "paired_difference_bootstrap_95_upper": float(
                        np.quantile(difference_bootstrap, 0.975)
                    ),
                    "numerator_worse_fraction": float(
                        np.mean(numerator_values > denominator_values)
                    ),
                }
            )
    return output


def ripple_budget_analysis(
    rows: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    thresholds: Sequence[float],
    *,
    horizon: int,
    common_frame_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute right-censored smooth-TV threshold survival at common frames."""

    frames = list(range(common_frame_step, horizon + 1, common_frame_step))
    selected = [
        row
        for row in rows
        if _is_ripple_rollout_row(
            row,
            horizon=horizon,
            common_frame_step=common_frame_step,
        )
        and str(row["model"]) in set(model_names)
    ]
    indexed = {
        (str(row["model"]), _ripple_case(row), _ripple_frame(row)): row
        for row in selected
    }
    survival_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if threshold <= 0.0:
            raise ValueError("smooth-TV thresholds must be positive")
        for model in model_names:
            cases = sorted(
                {case for row_model, case, _frame in indexed if row_model == model}
            )
            event_frames: list[int] = []
            for case in cases:
                event = horizon + common_frame_step
                for frame in frames:
                    row = indexed.get((model, case, frame))
                    value = (
                        _as_float(row, "smooth_state_tv_ratio")
                        if row is not None
                        else float("nan")
                    )
                    if (
                        row is None
                        or not _as_bool(row.get("proposal_valid"))
                        or not np.isfinite(value)
                        or value > threshold
                    ):
                        event = frame
                        break
                event_frames.append(event)
            events = np.asarray(event_frames, dtype=np.int64)
            for frame in frames:
                surviving = int(np.sum(events > frame))
                survival_rows.append(
                    {
                        "threshold": threshold,
                        "model": model,
                        "frame": frame,
                        "num_cases": len(cases),
                        "num_surviving": surviving,
                        "survival_fraction": surviving / max(len(cases), 1),
                    }
                )
            median_event = float(np.median(events)) if events.size else float("nan")
            summary_rows.append(
                {
                    "threshold": threshold,
                    "model": model,
                    "num_cases": len(cases),
                    "horizon_survivors": int(np.sum(events > horizon)),
                    "horizon_survival_fraction": float(
                        np.mean(events > horizon) if events.size else float("nan")
                    ),
                    "median_first_exceedance_frame": (
                        median_event if median_event <= horizon else float("nan")
                    ),
                    "median_right_censored": bool(median_event > horizon),
                    "restricted_mean_last_acceptable_frame": float(
                        np.mean(np.minimum(events - common_frame_step, horizon))
                        if events.size
                        else float("nan")
                    ),
                }
            )
    return survival_rows, summary_rows


def _ripple_endpoint_series(
    statistics: Sequence[dict[str, Any]],
    model: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in statistics if str(row["model"]) == model),
        key=lambda row: int(row["frame"]),
    )
    return (
        np.asarray([float(row["physical_time"]) for row in selected]),
        np.asarray([float(row[f"{metric}_mean"]) for row in selected]),
        np.asarray([float(row[f"{metric}_q25"]) for row in selected]),
        np.asarray([float(row[f"{metric}_q75"]) for row in selected]),
    )


def _plot_ripple_tradeoff(
    statistics: Sequence[dict[str, Any]],
    primary_models: dict[int, str],
    output_dir: Path,
) -> None:
    panels = (
        (
            "state_cons_scaled_rel_l2",
            "Global rollout error",
            "Fixed-scale conservative relative L2",
            1.0,
            True,
        ),
        (
            "smooth_error_d2_rms",
            "Away-front ripple amplitude",
            r"Smooth-region $\partial_{xx}$ error RMS",
            1.0,
            True,
        ),
        (
            "smooth_state_tv_ratio",
            "Away-front variation",
            "Smooth-region predicted / truth TV",
            1.0,
            False,
        ),
        (
            "error_tail_65_nyquist_fraction",
            "Unresolved error-energy share",
            "Modes 65-Nyquist (% of error energy)",
            100.0,
            False,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for panel_id, (axis, panel) in enumerate(zip(axes.flat, panels, strict=True)):
        metric, title, ylabel, scale, log_scale = panel
        for stride, model in sorted(primary_models.items()):
            x, mean, q25, q75 = _ripple_endpoint_series(statistics, model, metric)
            axis.fill_between(
                x,
                scale * q25,
                scale * q75,
                color=COLORS[stride],
                alpha=0.12,
            )
            axis.plot(
                x,
                scale * mean,
                color=COLORS[stride],
                marker=MARKERS[stride],
                markersize=3.0,
                linewidth=1.2,
                label=f"Stride {stride}",
            )
        if log_scale:
            axis.set_yscale("log")
        if metric == "smooth_state_tv_ratio":
            axis.axhline(1.0, color="#666666", linewidth=0.8, linestyle=":")
            axis.axhline(1.5, color="#666666", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Physical time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, linewidth=0.5)
        _panel_label(axis, chr(ord("a") + panel_id))
    axes[0, 0].legend(frameon=False, ncol=2)
    _save_figure(figure, output_dir, "ripple_error_tradeoff")


def _plot_ripple_lifetime(
    survival_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    *,
    primary_models: dict[int, str],
    repeat_s8_models: Sequence[str],
    thresholds: Sequence[float],
    saved_frame_dt: float,
    output_dir: Path,
) -> None:
    display_threshold = min(thresholds, key=lambda value: abs(value - 1.5))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    curve_models = [primary_models[4], primary_models[8], *repeat_s8_models]
    for model in curve_models:
        selected = sorted(
            (
                row
                for row in survival_rows
                if str(row["model"]) == model
                and math.isclose(float(row["threshold"]), display_threshold)
            ),
            key=lambda row: int(row["frame"]),
        )
        stride = 4 if model == primary_models[4] else 8
        is_repeat = model in repeat_s8_models
        axes[0].step(
            [float(row["frame"]) * saved_frame_dt for row in selected],
            [float(row["survival_fraction"]) for row in selected],
            where="post",
            color=COLORS[stride],
            linewidth=0.9 if is_repeat else 1.5,
            linestyle="--" if is_repeat else "-",
            alpha=0.55 if is_repeat else 1.0,
            label=(
                model.replace("_seed", ", seed ") if is_repeat else f"Stride {stride}"
            ),
        )
    axes[0].set_title(f"Smooth-TV budget: ratio <= {display_threshold:g}")
    axes[0].set_xlabel("Physical time")
    axes[0].set_ylabel("Acceptable and raw-admissible fraction")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.2, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=6.7)
    _panel_label(axes[0], "a")

    for model in curve_models:
        selected = sorted(
            (row for row in summary_rows if str(row["model"]) == model),
            key=lambda row: float(row["threshold"]),
        )
        stride = 4 if model == primary_models[4] else 8
        is_repeat = model in repeat_s8_models
        axes[1].plot(
            [float(row["threshold"]) for row in selected],
            [float(row["horizon_survival_fraction"]) for row in selected],
            color=COLORS[stride],
            marker=MARKERS[stride],
            markersize=3.2,
            linewidth=0.9 if is_repeat else 1.5,
            linestyle="--" if is_repeat else "-",
            alpha=0.55 if is_repeat else 1.0,
        )
    axes[1].set_title("Horizon survival sensitivity")
    axes[1].set_xlabel("Smooth-region predicted / truth TV budget")
    axes[1].set_ylabel("H96 survival fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2, linewidth=0.5)
    _panel_label(axes[1], "b")
    _save_figure(figure, output_dir, "ripple_lifetime_sensitivity")


def _modal_frame(row: dict[str, Any], evaluation_mode: str) -> int:
    key = (
        "target_frame" if evaluation_mode == "autoregressive_state" else "source_frame"
    )
    return int(float(row[key]))


def modal_heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    model: str,
    evaluation_mode: str,
    frames: Sequence[int],
    metric: str,
    *,
    modes: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mode-by-time matrix with truth-unresolved ratios masked."""

    frame_values = np.asarray(frames, dtype=np.int64)
    if frame_values.ndim != 1 or frame_values.size == 0:
        raise ValueError("frames must be a nonempty one-dimensional sequence")
    selected = [
        row
        for row in rows
        if str(row.get("model")) == model
        and str(row.get("evaluation_mode")) == evaluation_mode
    ]
    if modes is None:
        mode_values = np.asarray(
            sorted({int(float(row["mode_index"])) for row in selected}),
            dtype=np.int64,
        )
    else:
        mode_values = np.asarray(modes, dtype=np.int64)
    if mode_values.ndim != 1 or mode_values.size == 0:
        raise ValueError("modes must be a nonempty one-dimensional sequence")
    matrix = np.full((mode_values.size, frame_values.size), np.nan, dtype=np.float64)
    mode_positions = {int(mode): index for index, mode in enumerate(mode_values)}
    frame_positions = {int(frame): index for index, frame in enumerate(frame_values)}
    populated: set[tuple[int, int]] = set()
    for row in selected:
        mode = int(float(row["mode_index"]))
        frame = _modal_frame(row, evaluation_mode)
        if mode not in mode_positions or frame not in frame_positions:
            continue
        if metric == "modal_relative_error_rms" and not _as_bool(
            row.get("truth_resolved")
        ):
            continue
        position = (mode_positions[mode], frame_positions[frame])
        if position in populated:
            raise ValueError("duplicate modal row for one model, frame, and mode")
        populated.add(position)
        value = _as_float(row, metric)
        if math.isfinite(value):
            matrix[position] = value
    return mode_values, frame_values, matrix


def _common_teacher_source_frames(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    *,
    horizon: int,
) -> list[int]:
    frame_sets = []
    for model in models:
        frame_sets.append(
            {
                _modal_frame(row, "teacher_update")
                for row in rows
                if str(row.get("model")) == model
                and str(row.get("evaluation_mode")) == "teacher_update"
                and _modal_frame(row, "teacher_update") <= horizon
            }
        )
    common = set.intersection(*frame_sets) if frame_sets else set()
    return sorted(common)


def modal_band_statistics(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluation_mode: str,
    frames: Sequence[int],
    *,
    saved_frame_dt: float,
) -> list[dict[str, Any]]:
    """Aggregate modal powers before forming bandwise relative errors."""

    allowed_models = set(models)
    allowed_frames = set(int(frame) for frame in frames)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            str(row.get("evaluation_mode")) == evaluation_mode
            and str(row.get("model")) in allowed_models
        ):
            frame = _modal_frame(row, evaluation_mode)
            if frame in allowed_frames:
                groups[(str(row["model"]), frame)].append(row)

    output: list[dict[str, Any]] = []
    for (model, frame), group in sorted(groups.items()):
        total_error = sum(_as_float(row, "modal_error_power_mean") for row in group)
        stride = int(float(group[0]["stride"]))
        for band, (start, stop) in SPECTRAL_MODE_BANDS.items():
            band_rows = [
                row
                for row in group
                if int(float(row["mode_index"])) >= start
                and (stop is None or int(float(row["mode_index"])) < stop)
            ]
            if not band_rows:
                continue
            error_power = sum(
                _as_float(row, "modal_error_power_mean") for row in band_rows
            )
            target_power = sum(
                _as_float(row, "modal_target_power_mean") for row in band_rows
            )
            prediction_power = sum(
                _as_float(row, "modal_prediction_power_mean") for row in band_rows
            )
            output.append(
                {
                    "evaluation_mode": evaluation_mode,
                    "model": model,
                    "stride": stride,
                    "frame": frame,
                    "physical_time": frame * saved_frame_dt,
                    "band": band,
                    "num_modes": len(band_rows),
                    "num_truth_resolved_modes": sum(
                        _as_bool(row.get("truth_resolved")) for row in band_rows
                    ),
                    "num_samples": min(
                        int(float(row["num_samples"])) for row in band_rows
                    ),
                    "modal_error_power_sum": error_power,
                    "modal_target_power_sum": target_power,
                    "modal_relative_error_rms": (
                        math.sqrt(error_power / target_power)
                        if target_power > 0.0
                        else float("nan")
                    ),
                    "modal_prediction_to_truth_amplitude_ratio": (
                        math.sqrt(prediction_power / target_power)
                        if target_power > 0.0
                        else float("nan")
                    ),
                    "modal_error_energy_fraction": (
                        error_power / total_error if total_error > 0.0 else 0.0
                    ),
                }
            )
    return output


def modal_shape_statistics(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluation_mode: str,
    frames: Sequence[int],
    *,
    saved_frame_dt: float,
) -> list[dict[str, Any]]:
    """Summarize total modal error and its frequency shape at each time."""

    allowed_models = set(models)
    allowed_frames = set(int(frame) for frame in frames)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            str(row.get("evaluation_mode")) == evaluation_mode
            and str(row.get("model")) in allowed_models
        ):
            frame = _modal_frame(row, evaluation_mode)
            if frame in allowed_frames:
                groups[(str(row["model"]), frame)].append(row)

    output: list[dict[str, Any]] = []
    for (model, frame), group in sorted(groups.items()):
        modal_values = sorted(
            (
                int(float(row["mode_index"])),
                _as_float(row, "modal_error_power_mean"),
                _as_float(row, "modal_target_power_mean"),
            )
            for row in group
        )
        total_error = sum(error for _, error, _ in modal_values)
        total_target = sum(target for _, _, target in modal_values)
        cumulative = 0.0
        q90_mode = modal_values[-1][0]
        q95_mode = modal_values[-1][0]
        found_q90 = False
        for mode, error, _ in modal_values:
            cumulative += error / max(total_error, EPS)
            if not found_q90 and cumulative >= 0.9:
                q90_mode = mode
                found_q90 = True
            if cumulative >= 0.95:
                q95_mode = mode
                break
        output.append(
            {
                "evaluation_mode": evaluation_mode,
                "model": model,
                "stride": int(float(group[0]["stride"])),
                "frame": frame,
                "physical_time": frame * saved_frame_dt,
                "num_samples": min(int(float(row["num_samples"])) for row in group),
                "num_truth_resolved_modes": sum(
                    _as_bool(row.get("truth_resolved")) for row in group
                ),
                "modal_relative_error_rms": (
                    math.sqrt(total_error / total_target)
                    if total_target > 0.0
                    else float("nan")
                ),
                "error_spectral_centroid_mode": (
                    sum(mode * error for mode, error, _ in modal_values)
                    / max(total_error, EPS)
                ),
                "error_spectral_rms_mode": math.sqrt(
                    sum(mode**2 * error for mode, error, _ in modal_values)
                    / max(total_error, EPS)
                ),
                "error_spectral_d2_shape_factor": math.sqrt(
                    sum(mode**4 * error for mode, error, _ in modal_values)
                    / max(total_error, EPS)
                ),
                "tail_error_energy_fraction": (
                    sum(
                        error
                        for mode, error, _ in modal_values
                        if mode >= SPECTRAL_MODE_BANDS["tail_65_nyquist"][0]
                    )
                    / max(total_error, EPS)
                ),
                "error_energy_q90_mode": q90_mode,
                "error_energy_q95_mode": q95_mode,
            }
        )
    return output


def pooled_modal_band_statistics(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluation_mode: str,
    frames: Sequence[int],
) -> list[dict[str, Any]]:
    """Pool modal powers over a fixed frame set before forming band ratios."""

    allowed_models = set(models)
    allowed_frames = set(int(frame) for frame in frames)
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            str(row.get("evaluation_mode")) == evaluation_mode
            and str(row.get("model")) in allowed_models
            and _modal_frame(row, evaluation_mode) in allowed_frames
        ):
            selected[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(selected.items()):
        total_error = sum(
            _as_float(row, "modal_error_power_mean") for row in model_rows
        )
        for band, (start, stop) in SPECTRAL_MODE_BANDS.items():
            band_rows = [
                row
                for row in model_rows
                if int(float(row["mode_index"])) >= start
                and (stop is None or int(float(row["mode_index"])) < stop)
            ]
            if not band_rows:
                continue
            error_power = sum(
                _as_float(row, "modal_error_power_mean") for row in band_rows
            )
            target_power = sum(
                _as_float(row, "modal_target_power_mean") for row in band_rows
            )
            output.append(
                {
                    "evaluation_mode": evaluation_mode,
                    "model": model,
                    "stride": int(float(band_rows[0]["stride"])),
                    "band": band,
                    "first_frame": min(allowed_frames),
                    "last_frame": max(allowed_frames),
                    "num_frames": len(allowed_frames),
                    "modal_error_power_sum": error_power,
                    "modal_target_power_sum": target_power,
                    "modal_relative_error_rms": (
                        math.sqrt(error_power / target_power)
                        if target_power > 0.0
                        else float("nan")
                    ),
                    "modal_error_energy_fraction": (
                        error_power / max(total_error, EPS)
                    ),
                }
            )
    return output


def modal_ratio_statistics(
    rows: Sequence[dict[str, Any]],
    numerator: str,
    denominator: str,
    frames: Sequence[int],
) -> list[dict[str, Any]]:
    """Compare population-aggregate truth-normalized modal errors."""

    allowed_frames = set(int(frame) for frame in frames)
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("evaluation_mode")) != "autoregressive_state" or str(
            row.get("model")
        ) not in {numerator, denominator}:
            continue
        frame = _modal_frame(row, "autoregressive_state")
        if frame not in allowed_frames:
            continue
        key = (str(row["model"]), frame, int(float(row["mode_index"])))
        indexed[key] = row
    output: list[dict[str, Any]] = []
    modes = sorted({mode for _, _, mode in indexed})
    for frame in sorted(allowed_frames):
        for mode in modes:
            top = indexed.get((numerator, frame, mode))
            bottom = indexed.get((denominator, frame, mode))
            if top is None or bottom is None:
                continue
            top_error = _as_float(top, "modal_relative_error_rms")
            bottom_error = _as_float(bottom, "modal_relative_error_rms")
            jointly_resolved = _as_bool(top.get("truth_resolved")) and _as_bool(
                bottom.get("truth_resolved")
            )
            ratio = (
                top_error / bottom_error
                if jointly_resolved
                and math.isfinite(top_error)
                and math.isfinite(bottom_error)
                and bottom_error > 0.0
                else float("nan")
            )
            output.append(
                {
                    "numerator_model": numerator,
                    "denominator_model": denominator,
                    "frame": frame,
                    "mode_index": mode,
                    "jointly_truth_resolved": jointly_resolved,
                    "numerator_modal_relative_error_rms": top_error,
                    "denominator_modal_relative_error_rms": bottom_error,
                    "modal_relative_error_ratio": ratio,
                    "log10_modal_relative_error_ratio": (
                        math.log10(ratio)
                        if math.isfinite(ratio) and ratio > 0.0
                        else float("nan")
                    ),
                }
            )
    return output


def _nice_log_floor(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    mantissa = value / scale
    return (
        max(candidate for candidate in (1.0, 2.0, 5.0) if candidate <= mantissa) * scale
    )


def _nice_log_ceiling(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    mantissa = value / scale
    return next(
        candidate * scale
        for candidate in (1.0, 2.0, 5.0, 10.0)
        if candidate >= mantissa
    )


def _data_covering_log10_scale(
    values: np.ndarray,
    base_log10_limits: tuple[float, float],
) -> dict[str, Any]:
    """Expand a shared logarithmic scale to include every finite datum."""

    values = np.asarray(values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("logarithmic heatmap has no finite positive values")
    base_raw = (10.0 ** base_log10_limits[0], 10.0 ** base_log10_limits[1])
    data_min = float(np.min(positive))
    data_max = float(np.max(positive))
    raw_limits = (
        min(base_raw[0], _nice_log_floor(data_min)),
        max(base_raw[1], _nice_log_ceiling(data_max)),
    )
    return {
        "base_log10_limits": list(base_log10_limits),
        "rendered_log10_limits": [
            math.log10(raw_limits[0]),
            math.log10(raw_limits[1]),
        ],
        "raw_limits": list(raw_limits),
        "data_range": [data_min, data_max],
        "num_finite_positive_values": int(positive.size),
        "num_values_outside_base_scale": int(
            np.count_nonzero((positive < base_raw[0]) | (positive > base_raw[1]))
        ),
    }


def _log_colorbar_ticks(vmin: float, vmax: float) -> list[float]:
    integer_decades = range(
        math.ceil(math.log10(vmin)),
        math.floor(math.log10(vmax)) + 1,
    )
    ticks = [10.0**exponent for exponent in integer_decades]
    for edge in (vmin, vmax):
        if not any(math.isclose(edge, tick, rel_tol=1.0e-12) for tick in ticks):
            ticks.append(edge)
    return sorted(ticks)


def _format_positive_tick(value: float) -> str:
    return f"{value:g}"


def _plot_modal_panels(
    rows: Sequence[dict[str, Any]],
    primary_models: dict[int, str],
    *,
    evaluation_mode: str,
    frames: Sequence[int],
    metric: str,
    log10_limits: tuple[float, float],
    colorbar_label: str,
    stem: str,
    saved_frame_dt: float,
    output_dir: Path,
) -> dict[str, Any]:
    modes = list(
        range(
            1,
            1
            + max(
                int(float(row["mode_index"]))
                for row in rows
                if str(row.get("model")) in set(primary_models.values())
                and str(row.get("evaluation_mode")) == evaluation_mode
            ),
        )
    )
    frame_values = np.asarray(frames, dtype=np.float64)
    frame_step = (
        float(np.median(np.diff(frame_values))) if frame_values.size > 1 else 1.0
    )
    extent = (
        (frame_values[0] - 0.5 * frame_step) * saved_frame_dt,
        (frame_values[-1] + 0.5 * frame_step) * saved_frame_dt,
        0.5,
        max(modes) + 0.5,
    )
    panel_data = []
    for stride, model in primary_models.items():
        _, _, matrix = modal_heatmap_matrix(
            rows,
            model,
            evaluation_mode,
            frames,
            metric,
            modes=modes,
        )
        panel_data.append((stride, model, matrix))
    finite_positive = np.concatenate(
        [matrix[np.isfinite(matrix) & (matrix > 0.0)] for _, _, matrix in panel_data]
    )
    color_scale = _data_covering_log10_scale(finite_positive, log10_limits)
    raw_limits = color_scale["raw_limits"]
    norm = LogNorm(vmin=raw_limits[0], vmax=raw_limits[1], clip=False)
    figure, axes = plt.subplots(2, 2, figsize=(7.3, 5.2), sharex=True, sharey=True)
    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("#D9D9D9")
    image = None
    for panel, (stride, _model, matrix) in enumerate(panel_data):
        axis = axes.flat[panel]
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=colormap,
            norm=norm,
        )
        axis.set_title(f"Stride {stride}")
        axis.set_xlabel("Physical time")
        axis.set_ylabel("Mode index $k$")
        for boundary in (24.5, 64.5):
            if boundary < max(modes):
                axis.axhline(boundary, color="white", linewidth=0.55, alpha=0.8)
        _panel_label(axis, chr(ord("a") + panel))
    figure.subplots_adjust(left=0.09, right=0.87, bottom=0.09, top=0.94, hspace=0.24)
    colorbar_axis = figure.add_axes((0.9, 0.16, 0.018, 0.68))
    assert image is not None
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    ticks = _log_colorbar_ticks(raw_limits[0], raw_limits[1])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_positive_tick(value) for value in ticks])
    colorbar.set_label(colorbar_label)
    _save_figure(figure, output_dir, stem)
    return color_scale


def _plot_modal_ratio(
    ratio_rows: Sequence[dict[str, Any]],
    frames: Sequence[int],
    *,
    saved_frame_dt: float,
    output_dir: Path,
) -> dict[str, Any]:
    modes = sorted({int(float(row["mode_index"])) for row in ratio_rows})
    frame_values = np.asarray(frames, dtype=np.float64)
    matrix = np.full((len(modes), len(frames)), np.nan, dtype=np.float64)
    mode_positions = {mode: index for index, mode in enumerate(modes)}
    frame_positions = {int(frame): index for index, frame in enumerate(frames)}
    for row in ratio_rows:
        frame = int(float(row["frame"]))
        mode = int(float(row["mode_index"]))
        value = _as_float(row, "log10_modal_relative_error_ratio")
        if frame in frame_positions and mode in mode_positions and math.isfinite(value):
            matrix[mode_positions[mode], frame_positions[frame]] = value
    frame_step = (
        float(np.median(np.diff(frame_values))) if frame_values.size > 1 else 1.0
    )
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("modal-ratio heatmap has no finite values")
    base_limit = max(abs(value) for value in MODAL_RATIO_LOG10_LIMITS)
    data_limit = float(np.max(np.abs(finite)))
    rendered_limit = max(base_limit, math.ceil(data_limit / 0.25) * 0.25)
    figure, axis = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("#D9D9D9")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            (frame_values[0] - 0.5 * frame_step) * saved_frame_dt,
            (frame_values[-1] + 0.5 * frame_step) * saved_frame_dt,
            0.5,
            max(modes) + 0.5,
        ),
        cmap=colormap,
        vmin=-rendered_limit,
        vmax=rendered_limit,
    )
    for boundary in (24.5, 64.5):
        if boundary < max(modes):
            axis.axhline(boundary, color="black", linewidth=0.55, alpha=0.65)
    axis.set_xlabel("Physical time")
    axis.set_ylabel("Mode index $k$")
    axis.set_title("Stride 8 / stride 4 truth-normalized modal error")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    ratio_ticks = np.linspace(-rendered_limit, rendered_limit, 5)
    colorbar.set_ticks(ratio_ticks)
    colorbar.set_ticklabels([f"{10.0**value:.2g}x" for value in ratio_ticks])
    colorbar.set_label("Modal-error ratio (stride 8 / stride 4)")
    _save_figure(figure, output_dir, "rollout_modal_s8_over_s4_heatmap")
    return {
        "base_log10_limits": list(MODAL_RATIO_LOG10_LIMITS),
        "rendered_log10_limits": [-rendered_limit, rendered_limit],
        "rendered_raw_ratio_limits": [
            10.0 ** (-rendered_limit),
            10.0**rendered_limit,
        ],
        "data_log10_range": [float(np.min(finite)), float(np.max(finite))],
        "data_raw_ratio_range": [
            10.0 ** float(np.min(finite)),
            10.0 ** float(np.max(finite)),
        ],
    }


def run_ripple(args: argparse.Namespace) -> dict[str, Any]:
    if args.horizon < 1 or args.horizon % args.common_frame_step:
        raise ValueError("horizon must be positive and divisible by common-frame-step")
    if args.common_frame_step < 1:
        raise ValueError("common-frame-step must be positive")
    if args.saved_frame_dt <= 0.0:
        raise ValueError("saved-frame-dt must be positive")
    thresholds = sorted(set(args.smooth_tv_threshold or (1.25, 1.5, 2.0)))
    rows = _read_csv(args.metrics_csv)
    model_strides = {
        str(row["model"]): int(float(row["stride"]))
        for row in rows
        if str(row.get("mode")) == "autoregressive"
    }
    primary_models = {
        1: "s1",
        2: "s2",
        4: "s4",
        8: str(args.primary_s8_name),
    }
    missing = sorted(set(primary_models.values()) - set(model_strides))
    if missing:
        raise RuntimeError(f"missing primary ripple models: {missing}")
    if any(model_strides[name] != stride for stride, name in primary_models.items()):
        raise RuntimeError("a primary ripple model has the wrong native stride")
    repeat_s8_models = sorted(
        name
        for name, stride in model_strides.items()
        if stride == 8 and name != primary_models[8]
    )
    analyzed_models = [*primary_models.values(), *repeat_s8_models]
    endpoint_statistics = ripple_endpoint_statistics(
        rows,
        horizon=args.horizon,
        common_frame_step=args.common_frame_step,
        saved_frame_dt=args.saved_frame_dt,
    )
    paired_statistics = ripple_paired_statistics(
        rows,
        primary_models[8],
        primary_models[4],
        horizon=args.horizon,
        common_frame_step=args.common_frame_step,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    survival_rows, survival_summary = ripple_budget_analysis(
        rows,
        analyzed_models,
        thresholds,
        horizon=args.horizon,
        common_frame_step=args.common_frame_step,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "common_endpoint_summary.csv", endpoint_statistics)
    _write_csv(args.output_dir / "paired_s8_over_s4.csv", paired_statistics)
    _write_csv(args.output_dir / "ripple_budget_survival.csv", survival_rows)
    _write_csv(args.output_dir / "ripple_budget_summary.csv", survival_summary)
    figure_dir = args.output_dir / "figures"
    _plot_ripple_tradeoff(endpoint_statistics, primary_models, figure_dir)
    _plot_ripple_lifetime(
        survival_rows,
        survival_summary,
        primary_models=primary_models,
        repeat_s8_models=repeat_s8_models,
        thresholds=thresholds,
        saved_frame_dt=args.saved_frame_dt,
        output_dir=figure_dir,
    )
    figure_stems = [
        "ripple_error_tradeoff",
        "ripple_lifetime_sensitivity",
    ]
    modal_report: dict[str, Any] | None = None
    mode_spectra_path = getattr(args, "mode_spectra_csv", None)
    if mode_spectra_path is not None:
        modal_rows = _read_csv(mode_spectra_path)
        modal_models = {
            str(row["model"])
            for row in modal_rows
            if str(row.get("evaluation_mode")) == "autoregressive_state"
        }
        modal_missing = sorted(set(primary_models.values()) - modal_models)
        if modal_missing:
            raise RuntimeError(f"missing primary modal models: {modal_missing}")
        rollout_frames = list(
            range(args.common_frame_step, args.horizon + 1, args.common_frame_step)
        )
        teacher_frames = _common_teacher_source_frames(
            modal_rows,
            list(primary_models.values()),
            horizon=args.horizon,
        )
        if not teacher_frames:
            raise RuntimeError("no common teacher-update source frames")
        rollout_band_rows = modal_band_statistics(
            modal_rows,
            analyzed_models,
            "autoregressive_state",
            rollout_frames,
            saved_frame_dt=args.saved_frame_dt,
        )
        teacher_band_rows = modal_band_statistics(
            modal_rows,
            analyzed_models,
            "teacher_update",
            teacher_frames,
            saved_frame_dt=args.saved_frame_dt,
        )
        rollout_shape_rows = modal_shape_statistics(
            modal_rows,
            analyzed_models,
            "autoregressive_state",
            rollout_frames,
            saved_frame_dt=args.saved_frame_dt,
        )
        teacher_pooled_band_rows = pooled_modal_band_statistics(
            modal_rows,
            analyzed_models,
            "teacher_update",
            teacher_frames,
        )
        modal_ratio_rows = modal_ratio_statistics(
            modal_rows,
            primary_models[8],
            primary_models[4],
            rollout_frames,
        )
        _write_csv(
            args.output_dir / "rollout_modal_band_summary.csv", rollout_band_rows
        )
        _write_csv(
            args.output_dir / "teacher_modal_band_summary.csv", teacher_band_rows
        )
        _write_csv(
            args.output_dir / "rollout_modal_shape_summary.csv",
            rollout_shape_rows,
        )
        _write_csv(
            args.output_dir / "teacher_pooled_modal_band_summary.csv",
            teacher_pooled_band_rows,
        )
        _write_csv(args.output_dir / "modal_s8_over_s4.csv", modal_ratio_rows)
        rollout_relative_scale = _plot_modal_panels(
            modal_rows,
            primary_models,
            evaluation_mode="autoregressive_state",
            frames=rollout_frames,
            metric="modal_relative_error_rms",
            log10_limits=MODAL_RELATIVE_LOG10_LIMITS,
            colorbar_label="Truth-normalized modal RMS error",
            stem="rollout_modal_relative_error_heatmap",
            saved_frame_dt=args.saved_frame_dt,
            output_dir=figure_dir,
        )
        rollout_share_scale = _plot_modal_panels(
            modal_rows,
            primary_models,
            evaluation_mode="autoregressive_state",
            frames=rollout_frames,
            metric="modal_error_energy_fraction",
            log10_limits=MODAL_SHARE_LOG10_LIMITS,
            colorbar_label="Fraction of total error energy",
            stem="rollout_modal_error_share_heatmap",
            saved_frame_dt=args.saved_frame_dt,
            output_dir=figure_dir,
        )
        teacher_relative_scale = _plot_modal_panels(
            modal_rows,
            primary_models,
            evaluation_mode="teacher_update",
            frames=teacher_frames,
            metric="modal_relative_error_rms",
            log10_limits=MODAL_RELATIVE_LOG10_LIMITS,
            colorbar_label="Truth-normalized update error",
            stem="teacher_update_modal_relative_error_heatmap",
            saved_frame_dt=args.saved_frame_dt,
            output_dir=figure_dir,
        )
        modal_ratio_scale = _plot_modal_ratio(
            modal_ratio_rows,
            rollout_frames,
            saved_frame_dt=args.saved_frame_dt,
            output_dir=figure_dir,
        )
        figure_stems.extend(
            [
                "rollout_modal_relative_error_heatmap",
                "rollout_modal_error_share_heatmap",
                "teacher_update_modal_relative_error_heatmap",
                "rollout_modal_s8_over_s4_heatmap",
            ]
        )
        horizon_bands = {
            (str(row["model"]), str(row["band"])): row
            for row in rollout_band_rows
            if int(row["frame"]) == args.horizon
        }
        horizon_contrasts = {}
        for band in SPECTRAL_MODE_BANDS:
            stride8 = horizon_bands.get((primary_models[8], band))
            stride4 = horizon_bands.get((primary_models[4], band))
            if stride8 is None or stride4 is None:
                continue
            stride8_error = float(stride8["modal_relative_error_rms"])
            stride4_error = float(stride4["modal_relative_error_rms"])
            horizon_contrasts[band] = {
                "stride8_modal_relative_error_rms": stride8_error,
                "stride4_modal_relative_error_rms": stride4_error,
                "stride8_over_stride4": (
                    stride8_error / stride4_error
                    if stride4_error > 0.0
                    else float("nan")
                ),
                "stride8_error_energy_fraction": float(
                    stride8["modal_error_energy_fraction"]
                ),
                "stride4_error_energy_fraction": float(
                    stride4["modal_error_energy_fraction"]
                ),
            }
        shape_index = {
            (str(row["model"]), int(row["frame"])): row for row in rollout_shape_rows
        }
        shape_metrics = (
            "modal_relative_error_rms",
            "error_spectral_centroid_mode",
            "error_spectral_rms_mode",
            "error_spectral_d2_shape_factor",
            "tail_error_energy_fraction",
        )
        shape_contrasts = {}
        seed_consistency = {}
        for frame in sorted({args.common_frame_step, 32, args.horizon}):
            stride4_shape = shape_index[(primary_models[4], frame)]
            stride8_shape = shape_index[(primary_models[8], frame)]
            shape_contrasts[str(frame)] = {
                metric: (float(stride8_shape[metric]) / float(stride4_shape[metric]))
                for metric in shape_metrics
            }
            seed_consistency[str(frame)] = {
                metric: all(
                    float(shape_index[(model, frame)][metric])
                    > float(stride4_shape[metric])
                    for model in [primary_models[8], *repeat_s8_models]
                )
                for metric in shape_metrics[1:]
            }
        teacher_pooled_index = {
            (str(row["model"]), str(row["band"])): row
            for row in teacher_pooled_band_rows
        }
        teacher_contrasts = {}
        for band in SPECTRAL_MODE_BANDS:
            stride4_error = float(
                teacher_pooled_index[(primary_models[4], band)][
                    "modal_relative_error_rms"
                ]
            )
            stride8_errors = [
                float(teacher_pooled_index[(model, band)]["modal_relative_error_rms"])
                for model in [primary_models[8], *repeat_s8_models]
            ]
            teacher_contrasts[band] = {
                "stride4_modal_relative_error_rms": stride4_error,
                "stride8_range": [min(stride8_errors), max(stride8_errors)],
                "stride8_over_stride4_range": [
                    min(stride8_errors) / stride4_error,
                    max(stride8_errors) / stride4_error,
                ],
            }
        modal_report = {
            "mode_spectra_csv": str(mode_spectra_path),
            "mode_spectra_sha256": sha256_file(mode_spectra_path),
            "truth_floor_fraction": _as_float(modal_rows[0], "truth_floor_fraction"),
            "rollout_common_frames": rollout_frames,
            "teacher_common_source_frames": teacher_frames,
            "rendered_color_scales": {
                "rollout_relative_error": rollout_relative_scale,
                "rollout_error_share": rollout_share_scale,
                "teacher_relative_error": teacher_relative_scale,
                "rollout_s8_over_s4": modal_ratio_scale,
            },
            "horizon_band_contrasts": horizon_contrasts,
            "primary_s8_over_s4_shape_contrasts": shape_contrasts,
            "stride8_seed_consistent_higher_shape_metric": seed_consistency,
            "teacher_pooled_band_contrasts": teacher_contrasts,
            "comparison_semantics": (
                "Ratios compare population-aggregate modal RMS errors over "
                "raw-admissible cases; they are descriptive, not paired confidence "
                "intervals."
            ),
        }

    horizon_paired = {
        str(row["metric"]): row
        for row in paired_statistics
        if int(row["frame"]) == args.horizon
    }
    display_threshold = min(thresholds, key=lambda value: abs(value - 1.5))
    display_survival = {
        str(row["model"]): row
        for row in survival_summary
        if math.isclose(float(row["threshold"]), display_threshold)
    }
    global_error = horizon_paired["state_cons_scaled_rel_l2"]
    smooth_d2 = horizon_paired["smooth_error_d2_rms"]
    smooth_tv = horizon_paired["smooth_state_tv_ratio"]
    global_error_better = (
        float(global_error["paired_difference_bootstrap_95_upper"]) < 0.0
    )
    smooth_d2_worse = float(smooth_d2["paired_difference_bootstrap_95_lower"]) > 0.0
    smooth_tv_worse = float(smooth_tv["paired_difference_bootstrap_95_lower"]) > 0.0
    s4_lifetime = float(
        display_survival[primary_models[4]]["restricted_mean_last_acceptable_frame"]
    )
    primary_s8_lifetime = float(
        display_survival[primary_models[8]]["restricted_mean_last_acceptable_frame"]
    )
    repeat_lifetimes = [
        float(display_survival[model]["restricted_mean_last_acceptable_frame"])
        for model in repeat_s8_models
    ]
    shorter_ripple_lifetime = primary_s8_lifetime < s4_lifetime
    repeat_support = bool(repeat_lifetimes) and all(
        lifetime < s4_lifetime for lifetime in repeat_lifetimes
    )
    if (
        global_error_better
        and smooth_d2_worse
        and smooth_tv_worse
        and shorter_ripple_lifetime
        and repeat_support
    ):
        classification = (
            "lower_global_error_with_earlier_thresholded_ripple_degradation"
        )
    elif global_error_better and (smooth_d2_worse or smooth_tv_worse):
        classification = "lower_global_error_with_ripple_metric_conflict"
    else:
        classification = "no_confirmed_global_error_ripple_tradeoff"

    stride8_horizon_rows = [
        row
        for row in endpoint_statistics
        if int(row["stride"]) == 8 and int(row["frame"]) == args.horizon
    ]
    stride8_repeat_ranges = {
        metric: [
            float(min(row[f"{metric}_mean"] for row in stride8_horizon_rows)),
            float(max(row[f"{metric}_mean"] for row in stride8_horizon_rows)),
        ]
        for metric in (
            "state_cons_scaled_rel_l2",
            "smooth_error_d2_rms",
            "smooth_state_tv_ratio",
        )
    }
    input_report = {
        "metrics_csv": str(args.metrics_csv),
        "metrics_sha256": sha256_file(args.metrics_csv),
    }
    if mode_spectra_path is not None:
        input_report.update(
            {
                "mode_spectra_csv": str(mode_spectra_path),
                "mode_spectra_sha256": sha256_file(mode_spectra_path),
            }
        )
    report = {
        "experiment": "D036_frozen_large_step_ripple_tradeoff",
        "input": input_report,
        "contract": {
            "training_runs_added": 0,
            "horizon_frame": args.horizon,
            "horizon_physical_time": args.horizon * args.saved_frame_dt,
            "common_frame_step": args.common_frame_step,
            "primary_models": primary_models,
            "stride8_repeat_models": repeat_s8_models,
            "smooth_tv_threshold_sensitivity": thresholds,
            "threshold_semantics": (
                "first common endpoint where raw admissibility fails or the "
                "predicted/truth total-variation ratio outside two four-cell "
                "front neighborhoods exceeds the stated threshold"
            ),
            "bootstrap": {
                "unit": "paired held-out case",
                "replicates": args.bootstrap_replicates,
                "seed": args.bootstrap_seed,
            },
        },
        "horizon_s8_over_s4": horizon_paired,
        "display_threshold": display_threshold,
        "display_threshold_survival": display_survival,
        "stride8_horizon_ranges_across_three_seeds": stride8_repeat_ranges,
        "flags": {
            "stride8_lower_global_error_at_horizon": global_error_better,
            "stride8_higher_smooth_d2_error_at_horizon": smooth_d2_worse,
            "stride8_higher_smooth_tv_ratio_at_horizon": smooth_tv_worse,
            "primary_stride8_shorter_thresholded_ripple_lifetime": shorter_ripple_lifetime,
            "stride8_repeats_support_shorter_thresholded_ripple_lifetime": repeat_support,
        },
        "classification": classification,
        "modal_spectral_analysis": modal_report,
        "claim_boundary": (
            "The smooth-TV thresholds are a diagnostic sensitivity grid, not "
            "a universal physical tolerance. H96 is right-censored for raw "
            "stride-4/8 admissibility, and only stride 8 has seed repeats."
        ),
        "figure_stems": figure_stems,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return report


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _relative_l1(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sum(np.abs(prediction - truth)) / max(float(np.sum(np.abs(truth))), EPS)
    )


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm((prediction - truth).reshape(-1))
        / max(float(np.linalg.norm(truth.reshape(-1))), EPS)
    )


def _separated_front_faces(
    pressure: np.ndarray,
    *,
    num_fronts: int = 2,
    separation_cells: int = 8,
) -> np.ndarray:
    gradient = np.abs(np.diff(np.asarray(pressure, dtype=np.float64)))
    selected: list[int] = []
    for face in np.argsort(gradient)[::-1]:
        candidate = int(face)
        if all(abs(candidate - previous) >= separation_cells for previous in selected):
            selected.append(candidate)
        if len(selected) == num_fronts:
            break
    if len(selected) < num_fronts:
        selected.extend(
            [selected[-1] if selected else 0] * (num_fronts - len(selected))
        )
    return np.asarray(selected, dtype=np.int64)


def front_aligned_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    gamma: float,
    *,
    max_shift_cells: int,
) -> dict[str, Any]:
    """Measure error after an oracle global translation on a fixed interior.

    The diagnostic searches a predeclared integer-shift window and retains the
    shift that minimizes fixed-scale conservative relative L2.  It therefore
    estimates the error removable by one post-hoc global translation; it is not
    a new inference rule and cannot separately align multiple waves.
    """

    if max_shift_cells < 0:
        raise ValueError("max_shift_cells must be nonnegative")
    cells = int(truth.shape[0])
    if 2 * max_shift_cells >= cells:
        raise ValueError("alignment interior is empty")
    margin = max_shift_cells
    truth_indices = np.arange(margin, cells - margin, dtype=np.int64)
    scaled_prediction = (
        primitive_to_conservative_np(prediction, gamma) / PHYSICAL_SCALES
    )
    scaled_truth = primitive_to_conservative_np(truth, gamma) / PHYSICAL_SCALES
    central_truth = scaled_truth[truth_indices]
    unaligned = scaled_prediction[truth_indices]
    unaligned_l2 = _relative_l2(unaligned, central_truth)
    candidates = []
    for candidate_shift in range(-max_shift_cells, max_shift_cells + 1):
        candidate = scaled_prediction[truth_indices + candidate_shift]
        candidates.append(
            (
                _relative_l2(candidate, central_truth),
                abs(candidate_shift),
                candidate_shift,
            )
        )
    aligned_l2, _absolute_shift, shift = min(candidates)
    aligned = scaled_prediction[truth_indices + shift]
    return {
        "front_alignment_requested_shift_cells": shift,
        "front_alignment_shift_cells": shift,
        "front_alignment_shift_clipped": False,
        "front_alignment_interior_cells": int(truth_indices.size),
        "front_alignment_unaligned_fixed_scale_conservative_relative_l2": unaligned_l2,
        "front_aligned_fixed_scale_conservative_relative_l2": aligned_l2,
        "front_aligned_fixed_scale_conservative_relative_l1": _relative_l1(
            aligned,
            central_truth,
        ),
        "front_translation_removable_l2_fraction": float(
            1.0 - aligned_l2 / max(unaligned_l2, EPS)
        ),
    }


def rollout_state_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    x: np.ndarray,
    gamma: float,
    *,
    shock_radius_cells: int,
    max_alignment_shift_cells: int,
) -> dict[str, Any]:
    if shock_radius_cells < 0:
        raise ValueError("shock_radius_cells must be nonnegative")
    scaled_prediction = (
        primitive_to_conservative_np(prediction, gamma) / PHYSICAL_SCALES
    )
    scaled_truth = primitive_to_conservative_np(truth, gamma) / PHYSICAL_SCALES
    shock = np.zeros(truth.shape[0], dtype=bool)
    truth_faces = _separated_front_faces(truth[:, 2])
    for face in truth_faces:
        shock[
            max(0, int(face) - shock_radius_cells) : min(
                truth.shape[0],
                int(face) + shock_radius_cells + 2,
            )
        ] = True
    smooth = ~shock
    top2 = pressure_front_top2_metrics_np(
        prediction[None],
        truth[None],
        np.asarray(x, dtype=np.float64),
        min_separation_cells=8,
    )
    result: dict[str, Any] = {
        "fixed_scale_conservative_relative_l1": _relative_l1(
            scaled_prediction,
            scaled_truth,
        ),
        "fixed_scale_conservative_relative_l2": _relative_l2(
            scaled_prediction,
            scaled_truth,
        ),
        "shock_region_fixed_scale_conservative_relative_l1": _relative_l1(
            scaled_prediction[shock],
            scaled_truth[shock],
        ),
        "shock_region_fixed_scale_conservative_relative_l2": _relative_l2(
            scaled_prediction[shock],
            scaled_truth[shock],
        ),
        "smooth_region_fixed_scale_conservative_relative_l1": _relative_l1(
            scaled_prediction[smooth],
            scaled_truth[smooth],
        ),
        "smooth_region_fixed_scale_conservative_relative_l2": _relative_l2(
            scaled_prediction[smooth],
            scaled_truth[smooth],
        ),
        "shock_top2_position_mae": float(top2["position_assignment_mae"][0]),
        "shock_top2_strength_relative_l1": float(top2["strength_relative_l1"][0]),
        "min_density": float(np.min(prediction[:, 0])),
        "min_pressure": float(np.min(prediction[:, 2])),
    }
    result.update(
        front_aligned_metrics(
            prediction,
            truth,
            gamma,
            max_shift_cells=max_alignment_shift_cells,
        )
    )
    return result


def run_frozen_rollouts(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]],
    horizon: int,
    common_frame_step: int,
    device: torch.device,
    *,
    shock_radius_cells: int,
    max_alignment_shift_cells: int,
) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Replay every frozen stride and retain only common physical endpoints."""

    if horizon < 1 or horizon >= source.num_frames:
        raise ValueError("horizon must be a valid saved frame")
    if common_frame_step < 1 or horizon % common_frame_step:
        raise ValueError("horizon must be divisible by common_frame_step")
    if any(common_frame_step % stride for stride in models):
        raise ValueError("common_frame_step must be divisible by every stride")
    ids = np.asarray(case_ids, dtype=np.int64)
    common_frames = np.arange(0, horizon + 1, common_frame_step, dtype=np.int64)
    common_slot = {int(frame): position for position, frame in enumerate(common_frames)}
    predictions: dict[int, np.ndarray] = {}
    termination_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for stride, (model, adapter) in sorted(models.items()):
        current = np.asarray(source.data[ids, 0], dtype=np.float64).copy()
        conservative = primitive_to_conservative_np(current, source.gamma)
        active = np.ones(ids.size, dtype=bool)
        termination_frame = np.full(ids.size, -1, dtype=np.int64)
        termination_reason = np.full(ids.size, None, dtype=object)
        retained = np.full(
            (ids.size, common_frames.size, source.num_cells, 3),
            np.nan,
            dtype=np.float32,
        )
        retained[:, 0] = current.astype(np.float32)
        for target_frame in range(stride, horizon + 1, stride):
            positions = np.flatnonzero(active)
            if positions.size:
                proposed, proposed_conservative = _predict_model_batch_state(
                    source,
                    ids[positions],
                    current[positions],
                    target_frame - stride,
                    stride,
                    model,
                    adapter,
                    device,
                    conservative=conservative[positions],
                )
                for local, position in enumerate(positions):
                    candidate = proposed[local]
                    finite = bool(np.all(np.isfinite(candidate)))
                    admissible = finite and _euler_state_is_admissible(candidate)
                    if admissible:
                        current[position] = candidate
                        conservative[position] = proposed_conservative[local]
                    else:
                        active[position] = False
                        termination_frame[position] = target_frame
                        termination_reason[position] = (
                            "nonfinite_raw_state"
                            if not finite
                            else "nonpositive_raw_state"
                        )
            if target_frame in common_slot:
                slot = common_slot[target_frame]
                retained[active, slot] = current[active].astype(np.float32)
        predictions[stride] = retained
        for position, case_id_value in enumerate(ids):
            case_id = int(case_id_value)
            termination_rows.append(
                {
                    "case_id": case_id,
                    "stride": stride,
                    "requested_horizon": horizon,
                    "completed_horizon": bool(active[position]),
                    "termination_frame": (
                        int(termination_frame[position])
                        if termination_frame[position] >= 0
                        else float("nan")
                    ),
                    "termination_reason": termination_reason[position],
                }
            )
            for slot, frame in enumerate(common_frames[1:], start=1):
                prediction = retained[position, slot].astype(np.float64)
                completed = bool(np.all(np.isfinite(prediction)))
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "stride": stride,
                    "frame": int(frame),
                    "physical_time": float(source.t[case_id, frame]),
                    "completed_frame": completed,
                    "termination_frame": (
                        int(termination_frame[position])
                        if termination_frame[position] >= 0
                        else float("nan")
                    ),
                    "termination_reason": termination_reason[position],
                }
                if completed:
                    row.update(
                        rollout_state_metrics(
                            prediction,
                            np.asarray(source.data[case_id, frame], dtype=np.float64),
                            np.asarray(source.x[case_id], dtype=np.float64),
                            source.gamma,
                            shock_radius_cells=shock_radius_cells,
                            max_alignment_shift_cells=max_alignment_shift_cells,
                        )
                    )
                metric_rows.append(row)
    return common_frames, predictions, metric_rows, termination_rows


def rollout_metric_statistics(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "fixed_scale_conservative_relative_l1",
        "fixed_scale_conservative_relative_l2",
        "shock_region_fixed_scale_conservative_relative_l2",
        "smooth_region_fixed_scale_conservative_relative_l2",
        "front_aligned_fixed_scale_conservative_relative_l2",
        "front_translation_removable_l2_fraction",
        "shock_top2_position_mae",
    )
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["stride"]), int(row["frame"]))].append(row)
    output: list[dict[str, Any]] = []
    for (stride, frame), group in sorted(groups.items()):
        valid = [row for row in group if _as_bool(row.get("completed_frame"))]
        record: dict[str, Any] = {
            "stride": stride,
            "frame": frame,
            "physical_time": float(
                np.mean([_as_float(row, "physical_time") for row in group])
            ),
            "num_cases": len(group),
            "num_completed": len(valid),
            "completion_fraction": len(valid) / max(len(group), 1),
        }
        for metric in metrics:
            summary = _finite_summary(_as_float(row, metric) for row in valid)
            record.update(
                {f"{metric}_{name}": value for name, value in summary.items()}
            )
        output.append(record)
    return output


def _metadata_vector(
    source: Euler1DNPZ,
    names: Sequence[str],
) -> np.ndarray | None:
    for name in names:
        if name in source.metadata:
            values = np.asarray(source.metadata[name])
            if values.shape == (source.num_cases,):
                return values.astype(np.float64)
    return None


def initial_physical_descriptors(
    source: Euler1DNPZ,
    case_ids: np.ndarray,
) -> list[dict[str, Any]]:
    ids = np.asarray(case_ids, dtype=np.int64)
    left = np.asarray(source.left_states[ids], dtype=np.float64)
    right = np.asarray(source.right_states[ids], dtype=np.float64)
    left_sound = np.sqrt(source.gamma * left[:, 2] / left[:, 0])
    right_sound = np.sqrt(source.gamma * right[:, 2] / right[:, 0])
    characteristic = np.maximum(
        np.abs(left[:, 1]) + left_sound, np.abs(right[:, 1]) + right_sound
    )
    dx = np.mean(np.diff(source.x[ids], axis=1), axis=1)
    saved_dt = source.t[ids, 1] - source.t[ids, 0]
    domains = None
    for name in ("domains_exact", "domains"):
        values = np.asarray(source.metadata.get(name, ()))
        if values.shape == (source.num_cases, 2):
            domains = values.astype(np.float64)
            break
    if domains is None:
        domains = np.column_stack(
            (
                source.x[:, 0] - 0.5 * np.mean(np.diff(source.x, axis=1), axis=1),
                source.x[:, -1] + 0.5 * np.mean(np.diff(source.x, axis=1), axis=1),
            )
        )
    x_disc = _metadata_vector(source, ("x_disc_exact", "x_disc"))
    if x_disc is None:
        x_disc = np.empty(source.num_cases, dtype=np.float64)
        for case_id in range(source.num_cases):
            face = int(np.argmax(np.abs(np.diff(source.data[case_id, 0, :, 2]))))
            x_disc[case_id] = 0.5 * (
                source.x[case_id, face] + source.x[case_id, face + 1]
            )
    output: list[dict[str, Any]] = []
    for position, case_id_value in enumerate(ids):
        case_id = int(case_id_value)
        domain = domains[case_id]
        output.append(
            {
                "case_id": case_id,
                "max_initial_characteristic_speed": float(characteristic[position]),
                "initial_effective_cfl_stride8": float(
                    characteristic[position] * 8.0 * saved_dt[position] / dx[position]
                ),
                "absolute_log_density_ratio": float(
                    abs(np.log(left[position, 0] / right[position, 0]))
                ),
                "absolute_log_pressure_ratio": float(
                    abs(np.log(left[position, 2] / right[position, 2]))
                ),
                "velocity_jump_over_max_sound_speed": float(
                    abs(left[position, 1] - right[position, 1])
                    / max(left_sound[position], right_sound[position], EPS)
                ),
                "discontinuity_fraction": float(
                    (x_disc[case_id] - domain[0]) / max(domain[1] - domain[0], EPS)
                ),
                "normalized_pressure_jump": float(
                    abs(left[position, 2] - right[position, 2])
                    / max(0.5 * (left[position, 2] + right[position, 2]), EPS)
                ),
            }
        )
    return output


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def _bootstrap_spearman_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, first.size, size=first.size)
        estimate = _spearman(first[indices], second[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def nearest_centroid_loo(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    tie_preference: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed leave-one-out nearest-centroid classifier and majority control."""

    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("feature and label shapes are inconsistent")
    classes = np.unique(labels)
    predictions = np.empty(labels.size, dtype=labels.dtype)
    baseline = np.empty_like(predictions)
    for held_out in range(labels.size):
        train_mask = np.arange(labels.size) != held_out
        train_features = features[train_mask]
        train_labels = labels[train_mask]
        mean = np.mean(train_features, axis=0)
        scale = np.std(train_features, axis=0)
        scale[scale < EPS] = 1.0
        standardized_train = (train_features - mean) / scale
        standardized_test = (features[held_out] - mean) / scale
        distances: dict[int, float] = {}
        counts: dict[int, int] = {}
        for class_value in classes:
            class_rows = standardized_train[train_labels == class_value]
            counts[int(class_value)] = int(class_rows.shape[0])
            distances[int(class_value)] = float(
                np.linalg.norm(standardized_test - np.mean(class_rows, axis=0))
            )
        minimum = min(distances.values())
        tied = [
            value
            for value, distance in distances.items()
            if math.isclose(distance, minimum)
        ]
        predictions[held_out] = tie_preference if tie_preference in tied else min(tied)
        maximum_count = max(counts.values())
        majority = [value for value, count in counts.items() if count == maximum_count]
        baseline[held_out] = (
            tie_preference if tie_preference in majority else min(majority)
        )
    return predictions, baseline


def descriptor_outcome_analysis(
    descriptor_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    *,
    horizon: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    descriptor_by_case = {int(row["case_id"]): dict(row) for row in descriptor_rows}
    errors = {
        (int(row["case_id"]), int(row["stride"])): _as_float(
            row,
            "fixed_scale_conservative_relative_l2",
        )
        for row in metric_rows
        if int(row["frame"]) == horizon
        and int(row["stride"]) in (2, 4, 8)
        and _as_bool(row.get("completed_frame"))
    }
    eligible = sorted(
        case_id
        for case_id in descriptor_by_case
        if all((case_id, stride) in errors for stride in (2, 4, 8))
    )
    if not eligible:
        raise ValueError("descriptor analysis has no common completed cases")
    matrix = np.asarray(
        [[errors[(case_id, stride)] for stride in (2, 4, 8)] for case_id in eligible],
        dtype=np.float64,
    )
    population_index = int(np.argmin(np.mean(matrix, axis=0)))
    population_stride = (2, 4, 8)[population_index]
    labels = np.asarray(
        [(2, 4, 8)[index] for index in np.argmin(matrix, axis=1)], dtype=np.int64
    )
    oracle = np.min(matrix, axis=1)
    population_error = matrix[:, population_index]
    oracle_gain = 1.0 - oracle / np.maximum(population_error, EPS)
    outcome_rows: list[dict[str, Any]] = []
    for position, case_id in enumerate(eligible):
        row = dict(descriptor_by_case[case_id])
        row.update(
            {
                "stride2_horizon_error": matrix[position, 0],
                "stride4_horizon_error": matrix[position, 1],
                "stride8_horizon_error": matrix[position, 2],
                "casewise_best_stride": int(labels[position]),
                "population_best_stride": population_stride,
                "population_best_error": population_error[position],
                "per_case_oracle_error": oracle[position],
                "per_case_oracle_relative_gain": oracle_gain[position],
            }
        )
        outcome_rows.append(row)

    feature_names = (
        "initial_effective_cfl_stride8",
        "absolute_log_density_ratio",
        "absolute_log_pressure_ratio",
        "velocity_jump_over_max_sound_speed",
        "discontinuity_fraction",
        "normalized_pressure_jump",
    )
    feature_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in outcome_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    correlation_rows: list[dict[str, Any]] = []
    targets = {
        "log2_casewise_best_stride": np.log2(labels.astype(np.float64)),
        "per_case_oracle_relative_gain": oracle_gain,
    }
    for feature_index, name in enumerate(feature_names):
        values = feature_matrix[:, feature_index]
        for target_name, target in targets.items():
            estimate = _spearman(values, target)
            lower, upper = _bootstrap_spearman_interval(
                values,
                target,
                replicates=bootstrap_replicates,
                rng=rng,
            )
            correlation_rows.append(
                {
                    "descriptor": name,
                    "outcome": target_name,
                    "spearman_rho": estimate,
                    "bootstrap_95_lower": lower,
                    "bootstrap_95_upper": upper,
                    "num_cases": len(eligible),
                    "exploratory_multiple_descriptors": True,
                }
            )

    predicted, baseline = nearest_centroid_loo(
        feature_matrix,
        labels,
        tie_preference=population_stride,
    )
    correct = predicted == labels
    baseline_correct = baseline == labels
    differences = correct.astype(np.float64) - baseline_correct.astype(np.float64)
    bootstrap_difference = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, labels.size, size=labels.size)
        bootstrap_difference[replicate] = float(np.mean(differences[indices]))
    recalls = {
        str(class_value): float(
            np.mean(predicted[labels == class_value] == class_value)
        )
        for class_value in np.unique(labels)
    }
    confusion = {
        f"truth_{truth}_pred_{prediction}": int(
            np.sum((labels == truth) & (predicted == prediction))
        )
        for truth in (2, 4, 8)
        for prediction in (2, 4, 8)
    }
    accuracy = float(np.mean(correct))
    baseline_accuracy = float(np.mean(baseline_correct))
    difference_lower = float(np.quantile(bootstrap_difference, 0.025))
    difference_upper = float(np.quantile(bootstrap_difference, 0.975))
    balanced_accuracy = float(np.mean(list(recalls.values())))
    predictor_report = {
        "role": "exploratory_initial_descriptor_predictor_not_independent_validation",
        "features": list(feature_names),
        "predictor": "leave_one_out_standardized_nearest_class_centroid",
        "control": "leave_one_out_training_fold_majority_class",
        "num_cases": len(eligible),
        "population_best_stride": population_stride,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference": accuracy - baseline_accuracy,
        "accuracy_difference_bootstrap_95": [difference_lower, difference_upper],
        "balanced_accuracy": balanced_accuracy,
        "recall_by_stride": recalls,
        "confusion": confusion,
        "predeclared_signal_gate": {
            "accuracy_gain_at_least_0.10": accuracy >= baseline_accuracy + 0.10,
            "paired_bootstrap_lower_above_zero": difference_lower > 0.0,
            "balanced_accuracy_at_least_0.55": balanced_accuracy >= 0.55,
        },
    }
    predictor_report["signal_gate_passed"] = all(
        predictor_report["predeclared_signal_gate"].values()
    )
    predictor_report["routing"] = (
        "exploratory_signal_requires_independent_validation_before_any_policy"
        if predictor_report["signal_gate_passed"]
        else "stop_simple_descriptor_routing_and_do_not_build_adaptive_controller"
    )
    return outcome_rows, correlation_rows, predictor_report


def _plot_rollout_norms(
    statistics: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    panels = (
        ("fixed_scale_conservative_relative_l2", "Global relative L2", True),
        ("fixed_scale_conservative_relative_l1", "Global relative L1", True),
        (
            "shock_region_fixed_scale_conservative_relative_l2",
            "Separated-front region relative L2",
            True,
        ),
        (
            "smooth_region_fixed_scale_conservative_relative_l2",
            "Smooth-region relative L2",
            True,
        ),
        (
            "front_aligned_fixed_scale_conservative_relative_l2",
            "Oracle-translation-aligned relative L2",
            True,
        ),
        (
            "front_translation_removable_l2_fraction",
            "Fraction removable by one global translation",
            False,
        ),
    )
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 7.5), constrained_layout=True)
    for axis, (metric, ylabel, log_scale) in zip(axes.flat, panels):
        for stride in REQUIRED_STRIDES:
            series = _stride_series(statistics, stride)
            x = [row["physical_time"] for row in series]
            mean = [row[f"{metric}_mean"] for row in series]
            q25 = [row[f"{metric}_q25"] for row in series]
            q75 = [row[f"{metric}_q75"] for row in series]
            axis.fill_between(x, q25, q75, color=COLORS[stride], alpha=0.12)
            axis.plot(
                x,
                mean,
                color=COLORS[stride],
                marker=MARKERS[stride],
                markersize=3.5,
                linewidth=1.4,
                label=f"stride {stride}",
            )
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.axhline(0.0, color="#666666", linewidth=0.7)
        axis.set_xlabel("Physical time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[2, 0].text(
        0.02,
        0.04,
        "Post-hoc conservative-L2 translation oracle; fixed interior",
        transform=axes[2, 0].transAxes,
        fontsize=6.7,
    )
    for label, axis in zip("abcdef", axes.flat):
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "rollout_norm_and_phase_curves")


def _plot_descriptor_analysis(
    outcome_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
    predictor_report: dict[str, Any],
    output_dir: Path,
) -> None:
    oracle_correlations = [
        row
        for row in correlation_rows
        if row["outcome"] == "per_case_oracle_relative_gain"
    ]
    oracle_correlations.sort(key=lambda row: abs(float(row["spearman_rho"])))
    top_descriptor = max(
        oracle_correlations,
        key=lambda row: abs(float(row["spearman_rho"])),
    )["descriptor"]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)

    y = np.arange(len(oracle_correlations))
    estimates = np.asarray([float(row["spearman_rho"]) for row in oracle_correlations])
    lower = np.asarray(
        [float(row["bootstrap_95_lower"]) for row in oracle_correlations]
    )
    upper = np.asarray(
        [float(row["bootstrap_95_upper"]) for row in oracle_correlations]
    )
    axes[0, 0].errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        color="#0072B2",
        ecolor="#999999",
        capsize=2,
    )
    axes[0, 0].axvline(0.0, color="#555555", linewidth=0.8)
    axes[0, 0].set_yticks(
        y, [str(row["descriptor"]).replace("_", " ") for row in oracle_correlations]
    )
    axes[0, 0].set_xlabel("Spearman rho with H96 oracle gain")
    axes[0, 0].text(
        0.02,
        0.02,
        "Exploratory marginal intervals; no multiplicity claim",
        transform=axes[0, 0].transAxes,
        fontsize=6.7,
    )

    for stride in (2, 4, 8):
        selected = [
            row for row in outcome_rows if int(row["casewise_best_stride"]) == stride
        ]
        axes[0, 1].scatter(
            [float(row[top_descriptor]) for row in selected],
            [float(row["per_case_oracle_relative_gain"]) for row in selected],
            color=COLORS[stride],
            marker=MARKERS[stride],
            s=22,
            alpha=0.8,
            label=f"best stride {stride}",
        )
    axes[0, 1].set_xlabel(str(top_descriptor).replace("_", " "))
    axes[0, 1].set_ylabel("H96 per-case oracle relative gain")
    axes[0, 1].legend(frameon=False)

    jitter = {2: -0.08, 4: 0.0, 8: 0.08}
    for stride in (2, 4, 8):
        selected = [
            row for row in outcome_rows if int(row["casewise_best_stride"]) == stride
        ]
        axes[1, 0].scatter(
            [float(row["initial_effective_cfl_stride8"]) for row in selected],
            np.full(len(selected), np.log2(stride) + jitter[stride]),
            color=COLORS[stride],
            marker=MARKERS[stride],
            s=22,
            alpha=0.8,
        )
    axes[1, 0].set_xlabel("Initial effective CFL at stride 8")
    axes[1, 0].set_ylabel("Casewise H96 best stride")
    axes[1, 0].set_yticks([1, 2, 3], labels=["2", "4", "8"])

    confusion = np.asarray(
        [
            [
                predictor_report["confusion"][f"truth_{truth}_pred_{prediction}"]
                for prediction in (2, 4, 8)
            ]
            for truth in (2, 4, 8)
        ],
        dtype=np.int64,
    )
    image = axes[1, 1].imshow(confusion, cmap="Blues", vmin=0)
    for row in range(3):
        for column in range(3):
            axes[1, 1].text(
                column, row, str(confusion[row, column]), ha="center", va="center"
            )
    axes[1, 1].set_xticks([0, 1, 2], labels=["2", "4", "8"])
    axes[1, 1].set_yticks([0, 1, 2], labels=["2", "4", "8"])
    axes[1, 1].set_xlabel("Predicted best stride")
    axes[1, 1].set_ylabel("Observed best stride")
    axes[1, 1].text(
        0.02,
        -0.25,
        (
            f"LOO accuracy {predictor_report['accuracy']:.3f}; "
            f"majority {predictor_report['baseline_accuracy']:.3f}; "
            f"gate {'pass' if predictor_report['signal_gate_passed'] else 'fail'}"
        ),
        transform=axes[1, 1].transAxes,
        fontsize=6.8,
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for label, axis in zip("abcd", axes.flat):
        axis.grid(alpha=0.16)
        _panel_label(axis, label)
    _save_figure(figure, output_dir, "physical_descriptor_diagnostic")


def _case_variable_limits(
    truth: np.ndarray,
    predictions: dict[int, np.ndarray],
) -> list[tuple[float, float]]:
    limits: list[tuple[float, float]] = []
    for variable in range(3):
        values = [truth[..., variable].reshape(-1)]
        for prediction in predictions.values():
            finite = prediction[..., variable]
            values.append(finite[np.isfinite(finite)])
        merged = np.concatenate([value for value in values if value.size])
        lower = float(np.min(merged))
        upper = float(np.max(merged))
        span = max(upper - lower, 1.0e-6)
        limits.append((lower - 0.04 * span, upper + 0.04 * span))
    return limits


def _save_pressure_error_heatmap(
    case_id: int,
    x: np.ndarray,
    common_frames: np.ndarray,
    times: np.ndarray,
    truth: np.ndarray,
    predictions: dict[int, np.ndarray],
    terminations: dict[int, dict[str, Any]],
    output_dir: Path,
) -> None:
    errors = {
        stride: prediction[..., 2] - truth[..., 2]
        for stride, prediction in predictions.items()
    }
    finite_values = np.concatenate(
        [
            error[np.isfinite(error)]
            for error in errors.values()
            if np.any(np.isfinite(error))
        ]
    )
    maximum = max(float(np.max(np.abs(finite_values))), 1.0e-8)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#EEEEEE")
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(7.2, 2.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, stride in zip(axes, REQUIRED_STRIDES):
        image = axis.pcolormesh(
            x,
            times,
            errors[stride],
            shading="auto",
            cmap=cmap,
            vmin=-maximum,
            vmax=maximum,
            rasterized=True,
        )
        termination_frame = terminations[stride]["termination_frame"]
        if np.isfinite(termination_frame):
            frame = int(termination_frame)
            termination_time = float(np.interp(frame, common_frames, times))
            axis.axhline(
                termination_time,
                color="#000000",
                linestyle="--",
                linewidth=0.8,
            )
        axis.set_title(f"stride {stride}")
        axis.set_xlabel("x")
    axes[0].set_ylabel("Physical time")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Pressure error", fraction=0.022, pad=0.02)
    figure.suptitle(f"Case {case_id}: raw rollout pressure error")
    _save_figure(figure, output_dir, f"case_{case_id}_pressure_error_spacetime")


def save_case_rollout_artifacts(
    source: Euler1DNPZ,
    case_id: int,
    case_position: int,
    common_frames: np.ndarray,
    all_predictions: dict[int, np.ndarray],
    termination_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    fps: float,
) -> dict[str, Any]:
    """Write one self-contained array bundle, GIF, final frame, and heatmap."""

    output_dir.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(source.data[case_id, common_frames], dtype=np.float32)
    x = np.asarray(source.x[case_id], dtype=np.float32)
    times = np.asarray(source.t[case_id, common_frames], dtype=np.float32)
    predictions = {
        stride: np.asarray(values[case_position], dtype=np.float32)
        for stride, values in all_predictions.items()
    }
    termination_by_stride = {
        int(row["stride"]): dict(row)
        for row in termination_rows
        if int(row["case_id"]) == case_id
    }
    npz_path = output_dir / f"case_{case_id}_rollout.npz"
    np.savez_compressed(
        npz_path,
        case_id=np.asarray(case_id, dtype=np.int64),
        common_frames=common_frames,
        physical_times=times,
        x=x,
        truth=truth,
        **{
            f"prediction_stride{stride}": values
            for stride, values in predictions.items()
        },
    )

    limits = _case_variable_limits(truth, predictions)
    figure, axes = plt.subplots(
        3,
        4,
        figsize=(8.0, 5.2),
        sharex=True,
        constrained_layout=True,
    )
    truth_lines: dict[tuple[int, int], Any] = {}
    prediction_lines: dict[tuple[int, int], Any] = {}
    annotations: dict[tuple[int, int], Any] = {}
    for row, variable in enumerate(range(3)):
        for column, stride in enumerate(REQUIRED_STRIDES):
            axis = axes[row, column]
            (truth_line,) = axis.plot(
                x, truth[0, :, variable], color="#000000", linewidth=1.1, label="truth"
            )
            (prediction_line,) = axis.plot(
                x,
                predictions[stride][0, :, variable],
                color=COLORS[stride],
                linewidth=1.2,
                linestyle="--",
                label=f"stride {stride}",
            )
            truth_lines[(row, column)] = truth_line
            prediction_lines[(row, column)] = prediction_line
            annotations[(row, column)] = axis.text(
                0.5,
                0.5,
                "",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#D55E00",
                fontsize=7.5,
            )
            axis.set_ylim(*limits[variable])
            axis.set_xlim(float(x[0]), float(x[-1]))
            axis.grid(alpha=0.16)
            if row == 0:
                axis.set_title(f"stride {stride}")
            if column == 0:
                axis.set_ylabel(VARIABLE_LABELS[variable])
            if row == 2:
                axis.set_xlabel("x")
    axes[0, 0].legend(frameon=False, loc="best")
    title = figure.suptitle("")

    def update(slot: int) -> list[Any]:
        artists: list[Any] = [title]
        frame = int(common_frames[slot])
        title.set_text(f"Case {case_id} | H{frame} | t={times[slot]:.3f}")
        for row, variable in enumerate(range(3)):
            for column, stride in enumerate(REQUIRED_STRIDES):
                truth_line = truth_lines[(row, column)]
                prediction_line = prediction_lines[(row, column)]
                annotation = annotations[(row, column)]
                truth_line.set_ydata(truth[slot, :, variable])
                state = predictions[stride][slot]
                if np.all(np.isfinite(state)):
                    prediction_line.set_ydata(state[:, variable])
                    annotation.set_text("")
                else:
                    prediction_line.set_ydata(np.full(x.shape, np.nan))
                    termination_frame = termination_by_stride[stride][
                        "termination_frame"
                    ]
                    annotation.set_text(
                        "raw rollout terminated"
                        + (
                            f"\nat H{int(termination_frame)}"
                            if np.isfinite(termination_frame)
                            else ""
                        )
                    )
                artists.extend((truth_line, prediction_line, annotation))
        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=len(common_frames),
        interval=1000.0 / fps,
        blit=False,
    )
    gif_path = output_dir / f"case_{case_id}_rollout.gif"
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    update(len(common_frames) - 1)
    figure.savefig(
        output_dir / f"case_{case_id}_final_frame.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / f"case_{case_id}_final_frame.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(figure)
    _save_pressure_error_heatmap(
        case_id,
        x,
        common_frames,
        times,
        truth,
        predictions,
        termination_by_stride,
        output_dir,
    )
    return {
        "case_id": case_id,
        "npz": npz_path.name,
        "gif": gif_path.name,
        "final_frame_pdf": f"case_{case_id}_final_frame.pdf",
        "final_frame_png": f"case_{case_id}_final_frame.png",
        "pressure_error_spacetime_pdf": f"case_{case_id}_pressure_error_spacetime.pdf",
        "pressure_error_spacetime_png": f"case_{case_id}_pressure_error_spacetime.png",
        "termination_by_stride": termination_by_stride,
    }


def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if args.horizon < 1 or args.horizon % 8:
        raise ValueError("--horizon must be a positive multiple of eight")
    if args.common_frame_step < 1 or args.common_frame_step % 8:
        raise ValueError("--common-frame-step must be a positive multiple of eight")
    if args.animation_fps <= 0.0:
        raise ValueError("--animation-fps must be positive")
    if args.shock_radius_cells < 0 or args.max_alignment_shift_cells < 0:
        raise ValueError("shock radius and alignment shift must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    source = load_euler1d_npz(args.data_path)
    source.validate()
    if args.horizon >= source.num_frames:
        raise ValueError("--horizon exceeds the serialized trajectory")
    if not np.array_equal(source.t, np.broadcast_to(source.t[:1], source.t.shape)):
        raise ValueError("rollout visualization requires common saved times")
    case_ids = _frozen_split(
        source,
        split_seed=args.split_seed,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        test_cases=args.test_cases,
        split=args.split,
    )
    animation_ids = list(dict.fromkeys(int(value) for value in args.animation_case_ids))
    missing_animation_ids = sorted(set(animation_ids) - set(case_ids.tolist()))
    if missing_animation_ids:
        raise ValueError(
            f"animation cases are outside the frozen {args.split} split: {missing_animation_ids}"
        )
    checkpoint_paths = _checkpoint_paths(args.checkpoint)
    if tuple(sorted(checkpoint_paths)) != REQUIRED_STRIDES:
        raise ValueError(f"rollout visualization requires strides {REQUIRED_STRIDES}")
    data_sha256 = sha256_file(args.data_path)
    saved_time_sha256 = _saved_time_sha256(source)
    models: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}
    checkpoint_contract: dict[str, Any] = {}
    for stride, path in sorted(checkpoint_paths.items()):
        model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
            path,
            device,
        )
        frozen = _validate_checkpoint_contract(
            checkpoint,
            stride=stride,
            expected_cases=case_ids,
            data_sha256=data_sha256,
            saved_time_sha256=saved_time_sha256,
            split=args.split,
        )
        model.eval()
        adapter.eval()
        models[stride] = (model, adapter)
        checkpoint_contract[str(stride)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "best_epoch": checkpoint.get("best_epoch"),
            "frozen_coordinates": frozen,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "analysis": "frozen_exact_recurrence_norm_phase_and_visualization",
        "new_learned_training_runs": 0,
        "data_path": str(args.data_path),
        "data_sha256": data_sha256,
        "saved_time_sha256": saved_time_sha256,
        "data_shape": list(source.data.shape),
        "split": args.split,
        "case_ids": case_ids.tolist(),
        "animation_case_ids": animation_ids,
        "horizon": args.horizon,
        "physical_horizon": float(source.t[0, args.horizon] - source.t[0, 0]),
        "common_frame_step": args.common_frame_step,
        "strides": list(REQUIRED_STRIDES),
        "checkpoints": checkpoint_contract,
        "raw_recurrence": "decoded conservative state retained exactly between calls",
        "inference_modification": "none_no_clipping_no_floor_no_limiter_no_projection",
        "shock_region": (
            f"cells within radius {args.shock_radius_cells} of two separated strongest truth pressure fronts"
        ),
        "front_alignment": {
            "definition": (
                "integer global shift minimizing fixed-scale conservative relative L2 "
                "inside the predeclared search window"
            ),
            "maximum_shift_cells": args.max_alignment_shift_cells,
            "evaluation_interior": (
                "fixed central cells after removing maximum shift from both boundaries"
            ),
            "role": "diagnostic_only_not_inference_correction",
        },
        "descriptor_predictor": {
            "features_fixed_before_outcome_readout": [
                "initial_effective_cfl_stride8",
                "absolute_log_density_ratio",
                "absolute_log_pressure_ratio",
                "velocity_jump_over_max_sound_speed",
                "discontinuity_fraction",
                "normalized_pressure_jump",
            ],
            "method": "leave_one_out_standardized_nearest_class_centroid",
            "signal_gate": (
                "accuracy gain >=0.10 over fold-majority, paired bootstrap lower >0, balanced accuracy >=0.55"
            ),
            "claim_limit": "exploratory same-split diagnostic only",
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(json_ready(contract), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    common_frames, predictions, metric_rows, termination_rows = run_frozen_rollouts(
        source,
        case_ids,
        models,
        args.horizon,
        args.common_frame_step,
        device,
        shock_radius_cells=args.shock_radius_cells,
        max_alignment_shift_cells=args.max_alignment_shift_cells,
    )
    metric_statistics = rollout_metric_statistics(metric_rows)
    descriptor_rows = initial_physical_descriptors(source, case_ids)
    outcome_rows, correlation_rows, predictor_report = descriptor_outcome_analysis(
        descriptor_rows,
        metric_rows,
        horizon=args.horizon,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(args.output_dir / "rollout_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "rollout_metric_statistics.csv", metric_statistics)
    _write_csv(args.output_dir / "terminations.csv", termination_rows)
    _write_csv(args.output_dir / "casewise_horizon_descriptors.csv", outcome_rows)
    _write_csv(args.output_dir / "descriptor_correlations.csv", correlation_rows)
    (args.output_dir / "descriptor_predictor.json").write_text(
        json.dumps(json_ready(predictor_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    figure_dir = args.output_dir / "figures"
    _plot_rollout_norms(metric_statistics, figure_dir)
    _plot_descriptor_analysis(
        outcome_rows,
        correlation_rows,
        predictor_report,
        figure_dir,
    )
    position_by_case = {
        int(case_id): position for position, case_id in enumerate(case_ids)
    }
    case_dir = args.output_dir / "cases"
    animation_manifest = [
        save_case_rollout_artifacts(
            source,
            case_id,
            position_by_case[case_id],
            common_frames,
            predictions,
            termination_rows,
            case_dir,
            fps=args.animation_fps,
        )
        for case_id in animation_ids
    ]
    (args.output_dir / "animation_manifest.json").write_text(
        json.dumps(json_ready(animation_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    final_rows = [row for row in metric_statistics if int(row["frame"]) == args.horizon]
    final_by_stride = {int(row["stride"]): row for row in final_rows}
    metric_names = (
        "fixed_scale_conservative_relative_l1",
        "fixed_scale_conservative_relative_l2",
        "shock_region_fixed_scale_conservative_relative_l2",
        "smooth_region_fixed_scale_conservative_relative_l2",
        "front_aligned_fixed_scale_conservative_relative_l2",
        "shock_top2_position_mae",
    )
    best_stride_by_metric = {
        metric: min(
            REQUIRED_STRIDES,
            key=lambda stride: float(final_by_stride[stride][f"{metric}_mean"]),
        )
        for metric in metric_names
    }
    final_metrics = {
        str(stride): {
            "num_completed": int(final_by_stride[stride]["num_completed"]),
            **{
                metric: float(final_by_stride[stride][f"{metric}_mean"])
                for metric in metric_names
            },
        }
        for stride in REQUIRED_STRIDES
    }
    report = {
        "status": "complete",
        "new_learned_training_runs": 0,
        "horizon": args.horizon,
        "physical_horizon": contract["physical_horizon"],
        "final_metrics_by_stride": final_metrics,
        "best_stride_by_metric": best_stride_by_metric,
        "descriptor_predictor": predictor_report,
        "animations": animation_manifest,
        "runtime_seconds": time.perf_counter() - started,
        "claim_limits": [
            "separate fixed-step models, not one timestep-conditioned model",
            "front alignment is diagnostic and uses a post-hoc global translation",
            "descriptor analysis is exploratory on the same 64 held-out cases",
            "stride 8 is the tested right boundary, so no universal optimum is bracketed",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    _configure_plot_style()
    args = parse_args(argv)
    if args.command == "summary":
        run_summary(args)
    elif args.command == "closeout":
        run_closeout(args)
    elif args.command == "rollout":
        run_rollout(args)
    else:
        run_ripple(args)


if __name__ == "__main__":
    main()
