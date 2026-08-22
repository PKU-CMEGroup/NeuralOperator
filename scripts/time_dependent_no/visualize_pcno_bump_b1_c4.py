#!/usr/bin/env python3
"""Visualize D094 B1-C4 training dynamics and paired rollout behavior.

The primary loss figure compares fixed-bank seen-train one-step error, fixed
open-validation one-step error, and autonomous H79 rollout error.  The online
training trace is shown only as optimization context.  Animation bundles keep
two mechanisms separate: free recurrent pressure states contain accumulated
error, whereas one-step conservative residuals always receive exact U(t) and
therefore contain no accumulated rollout drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_scaling import _metric_snapshot
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    BUNDLE_SCHEMA,
    EXPECTED_SELECTED_STEP,
    TRAJECTORY_COUNTS,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    SCHEMA as AUDIT_SCHEMA,
)
from utility.time_dependent_no.pcno_artifacts import atomic_write_json, sha256_file

SCHEMA = "d094_b1_c4_visualization_v1"
DEFAULT_FPS = 5
DEFAULT_DPI = 110
COMPONENT_INDEX = {"density": 0, "momentum_x": 1, "momentum_y": 2, "energy": 3}
COUNT_COLORS = {128: "#0072B2", 256: "#D55E00"}
METRIC_COLORS = {
    "online": "#7A7A7A",
    "seen": "#0072B2",
    "validation": "#E69F00",
    "h79": "#009E73",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("gif", "mp4"), default="mp4")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"metric history must contain object rows: {path}")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _array_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _configure_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "lines.linewidth": 1.7,
        }
    )
    return plt, animation


def discover_training_curves(run_root: Path) -> dict[int, dict[str, Any]]:
    outputs = run_root / "outputs"
    if run_root.is_symlink() or not outputs.is_dir():
        raise ValueError("B1-C4 run root lacks a regular outputs directory")
    curves: dict[int, dict[str, Any]] = {}
    for run_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        contract = _load_json(run_dir / "bump_scaling_contract.json")
        count = int(contract["trajectory_count"])
        if count not in TRAJECTORY_COUNTS or count in curves:
            raise ValueError("B1-C4 curves must contain one n=128 and one n=256 run")
        summary = _load_json(run_dir / "summary.json")
        raw_rows = _load_jsonl(run_dir / "metrics.jsonl")
        if len(raw_rows) != 160:
            raise ValueError("B1-C4 curve requires all 160 epoch rows")
        rows = [_metric_snapshot(row) for row in raw_rows]
        comparable = [
            row
            for row in rows
            if row["fixed_seen_train_one_step_relative_l2"] is not None
            and row["rollout_h79_relative_l2"] is not None
        ]
        if not comparable or int(summary["best_epoch"]) != 149:
            raise ValueError("B1-C4 selected or comparable metric history changed")
        if int(comparable[-1]["optimizer_step"]) != 40_960:
            raise ValueError("B1-C4 comparable curve does not reach step 40,960")
        curves[count] = {
            "run": run_dir.name,
            "run_dir": run_dir,
            "rows": rows,
            "comparable_rows": comparable,
            "selected_epoch": int(summary["best_epoch"]),
            "selected_step": EXPECTED_SELECTED_STEP,
        }
    if set(curves) != set(TRAJECTORY_COUNTS):
        raise ValueError("B1-C4 curves do not expose the exact paired counts")
    return curves


def validate_audit(audit_dir: Path) -> dict[str, Any]:
    summary = _load_json(audit_dir / "summary.json")
    if (
        summary.get("schema") != AUDIT_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_on_outside_cases") is not False
    ):
        raise ValueError("B1-C4 audit is incomplete or violates its access contract")
    manifest = _load_json(audit_dir / "artifact_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("B1-C4 audit artifact manifest lacks files")
    for relative, record in files.items():
        path = audit_dir / str(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"B1-C4 audit artifact changed: {relative}")
    return summary


def audit_h79_by_count_step(
    audit_summary: Mapping[str, Any],
) -> dict[int, dict[int, float]]:
    values: dict[int, dict[int, float]] = {count: {} for count in TRAJECTORY_COUNTS}
    for cell in audit_summary["cells"]:
        count = int(cell["trajectory_count"])
        step = int(cell["optimizer_step"])
        value = float(
            cell["outside_selection_rollout"]["mean_endpoint_relative_l2"]["79"]
        )
        if count not in values or step in values[count] or not math.isfinite(value):
            raise ValueError("B1-C4 outside-audit H79 matrix is malformed")
        values[count][step] = value
    if any(len(values[count]) != 4 for count in TRAJECTORY_COUNTS):
        raise ValueError("B1-C4 outside-audit H79 matrix is incomplete")
    return values


def _curve_rows(curves: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in TRAJECTORY_COUNTS:
        for row in curves[count]["comparable_rows"]:
            rows.append(
                {
                    "trajectory_count": count,
                    "optimizer_step": row["optimizer_step"],
                    "epoch": row["epoch"],
                    "online_train_one_step_relative_l2": row[
                        "online_train_one_step_relative_l2"
                    ],
                    "fixed_seen_train_one_step_relative_l2": row[
                        "fixed_seen_train_one_step_relative_l2"
                    ],
                    "fixed_validation_one_step_relative_l2": row[
                        "fixed_validation_one_step_relative_l2"
                    ],
                    "selection_rollout_h79_relative_l2": row["rollout_h79_relative_l2"],
                    "is_selected_checkpoint": (
                        int(row["optimizer_step"]) == EXPECTED_SELECTED_STEP
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("curve CSV requires at least one row")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curves(
    curves: Mapping[int, Mapping[str, Any]],
    audit_h79: Mapping[int, Mapping[int, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write the side-by-side figure requested by the experiment owner."""

    plt, _ = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), sharex=True, sharey=True)
    for axis, count in zip(axes, TRAJECTORY_COUNTS, strict=True):
        rows = curves[count]["comparable_rows"]
        steps = np.asarray([row["optimizer_step"] for row in rows])
        online_by_step = {
            int(row["optimizer_step"]): float(row["online_train_one_step_relative_l2"])
            for row in curves[count]["rows"]
        }
        axis.plot(
            steps,
            [online_by_step[int(step)] for step in steps],
            color=METRIC_COLORS["online"],
            linewidth=1.0,
            alpha=0.58,
            linestyle=":",
            label="online train (context only)",
        )
        axis.plot(
            steps,
            [row["fixed_seen_train_one_step_relative_l2"] for row in rows],
            color=METRIC_COLORS["seen"],
            marker="o",
            markersize=2.4,
            label="fixed seen-train one-step",
        )
        axis.plot(
            steps,
            [row["fixed_validation_one_step_relative_l2"] for row in rows],
            color=METRIC_COLORS["validation"],
            marker="s",
            markersize=2.3,
            label="fixed validation one-step",
        )
        axis.plot(
            steps,
            [row["rollout_h79_relative_l2"] for row in rows],
            color=METRIC_COLORS["h79"],
            marker="D",
            markersize=2.3,
            label="H79 selection rollout (16)",
        )
        audit_steps = sorted(audit_h79[count])
        axis.scatter(
            audit_steps,
            [audit_h79[count][step] for step in audit_steps],
            color=METRIC_COLORS["h79"],
            marker="x",
            s=27,
            linewidths=1.3,
            zorder=6,
            label="H79 outside audit (28)",
        )
        selected = next(
            row for row in rows if int(row["optimizer_step"]) == EXPECTED_SELECTED_STEP
        )
        for field, color in (
            ("fixed_seen_train_one_step_relative_l2", METRIC_COLORS["seen"]),
            ("fixed_validation_one_step_relative_l2", METRIC_COLORS["validation"]),
            ("rollout_h79_relative_l2", METRIC_COLORS["h79"]),
        ):
            axis.scatter(
                [EXPECTED_SELECTED_STEP],
                [selected[field]],
                marker="*",
                s=68,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=7,
            )
        passes = 40_960 / (79 * count)
        axis.set_title(f"PCNO n={count} | final exposure={passes:.2f} passes")
        axis.set_yscale("log")
        axis.set_xlabel("Optimizer steps")
        axis.axvline(20_480, color="#999999", linestyle="--", linewidth=0.8)
        axis.axvline(
            EXPECTED_SELECTED_STEP, color="#444444", linestyle="-.", linewidth=0.8
        )
    axes[0].set_ylabel("Relative L2 (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "D094 B1-C4: one-step and recurrent errors are different observables",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"b1_c4_n128_n256_loss_curves_side_by_side.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    plt.close(fig)

    fields = (
        (
            "fixed_seen_train_one_step_relative_l2",
            "Fixed seen-train one-step",
        ),
        (
            "fixed_validation_one_step_relative_l2",
            "Fixed validation one-step",
        ),
        ("rollout_h79_relative_l2", "Autonomous H79 selection rollout"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharex=True)
    for axis, (field, title) in zip(axes, fields, strict=True):
        for count in TRAJECTORY_COUNTS:
            rows = curves[count]["comparable_rows"]
            axis.plot(
                [row["optimizer_step"] for row in rows],
                [row[field] for row in rows],
                color=COUNT_COLORS[count],
                label=f"n={count}",
            )
        if field == "rollout_h79_relative_l2":
            for count in TRAJECTORY_COUNTS:
                steps = sorted(audit_h79[count])
                axis.scatter(
                    steps,
                    [audit_h79[count][step] for step in steps],
                    marker="x",
                    color=COUNT_COLORS[count],
                    s=24,
                )
            axis.text(
                0.03,
                0.04,
                "x: 28-case outside audit",
                transform=axis.transAxes,
                fontsize=7,
            )
        axis.set_title(title)
        axis.set_yscale("log")
        axis.set_xlabel("Optimizer steps")
        axis.set_ylabel("Relative L2")
        axis.axvline(
            EXPECTED_SELECTED_STEP, color="#777777", linestyle="-.", linewidth=0.8
        )
    axes[0].legend()
    fig.suptitle("D094 B1-C4: paired trajectory-count comparison", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"b1_c4_n128_n256_loss_curves_by_metric.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    plt.close(fig)
    return outputs


def pressure(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    rho = value[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kinetic = 0.5 * (np.square(value[..., 1]) + np.square(value[..., 2])) / rho
        return (gamma - 1.0) * (value[..., 3] - kinetic)


def free_pressure_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return accumulated free-rollout pressure fields for calls 1..79."""

    reference = np.asarray(arrays["reference_states_conservative"], dtype=np.float64)
    n128 = np.asarray(arrays["n128_free_states_conservative"], dtype=np.float64)
    n256 = np.asarray(arrays["n256_free_states_conservative"], dtype=np.float64)
    if reference.shape != n128.shape or reference.shape != n256.shape:
        raise ValueError("free pressure states do not share one shape")
    if reference.shape[0] != 80 or reference.shape[-1] != 4:
        raise ValueError("free pressure states must contain H0 through H79")
    truth = pressure(reference[1:])
    pred128 = pressure(n128[1:])
    pred256 = pressure(n256[1:])
    return {
        "truth": truth,
        "n128": pred128,
        "n128_error": pred128 - truth,
        "n256": pred256,
        "n256_error": pred256 - truth,
    }


def one_step_residual_fields(
    arrays: Mapping[str, np.ndarray], component: str
) -> dict[str, np.ndarray]:
    """Return exact-input conservative residuals with no recurrent accumulation."""

    if component not in COMPONENT_INDEX:
        raise ValueError(f"unsupported conservative component: {component}")
    index = COMPONENT_INDEX[component]
    reference = np.asarray(arrays["reference_states_conservative"], dtype=np.float64)
    n128_next = np.asarray(
        arrays["n128_teacher_forced_next_conservative"], dtype=np.float64
    )
    n256_next = np.asarray(
        arrays["n256_teacher_forced_next_conservative"], dtype=np.float64
    )
    if (
        reference.shape[0] != 80
        or reference.shape[-1] != 4
        or n128_next.shape != reference[1:].shape
        or n256_next.shape != reference[1:].shape
    ):
        raise ValueError("one-step residual arrays do not share the H1-H79 shape")
    current = reference[:-1, :, index]
    truth = reference[1:, :, index] - current
    pred128 = n128_next[:, :, index] - current
    pred256 = n256_next[:, :, index] - current
    return {
        "truth": truth,
        "n128": pred128,
        "n128_error": pred128 - truth,
        "n256": pred256,
        "n256_error": pred256 - truth,
    }


def _finite_quantile(values: Sequence[np.ndarray], quantile: float) -> float:
    flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    finite = np.abs(flat[np.isfinite(flat)])
    if not finite.size:
        raise ValueError("visualization fields have no finite values")
    return max(float(np.quantile(finite, quantile)), np.finfo(float).tiny)


def _field_scales(fields: Mapping[str, np.ndarray], *, mode: str) -> dict[str, Any]:
    if mode == "free_pressure":
        states = np.concatenate(
            [fields[name].reshape(-1) for name in ("truth", "n128", "n256")]
        )
        finite = states[np.isfinite(states)]
        if not finite.size:
            raise ValueError("pressure fields have no finite values")
        state_min, state_max = (
            float(value) for value in np.quantile(finite, (0.005, 0.995))
        )
        if state_min == state_max:
            state_max = state_min + np.finfo(float).eps
        return {
            "field": [state_min, state_max],
            "error": [
                -_finite_quantile([fields["n128_error"], fields["n256_error"]], 0.995),
                _finite_quantile([fields["n128_error"], fields["n256_error"]], 0.995),
            ],
            "quantiles": [0.005, 0.995],
        }
    field_limit = _finite_quantile(
        [fields["truth"], fields["n128"], fields["n256"]], 0.995
    )
    error_limit = _finite_quantile([fields["n128_error"], fields["n256_error"]], 0.995)
    return {
        "field": [-field_limit, field_limit],
        "error": [-error_limit, error_limit],
        "quantiles": [0.005, 0.995],
    }


def render_comparison_animation(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    mode: str,
    component: str | None = None,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Any]:
    if mode == "free_pressure":
        fields = free_pressure_fields(arrays)
        quantity = "pressure"
        definition = "autoregressive free state; errors include accumulated drift"
        titles = (
            "reference pressure",
            "n=128 free pressure",
            "n=128 state error",
            "n=256 free pressure",
            "n=256 state error",
        )
    elif mode == "one_step_residual":
        if component is None:
            raise ValueError("one-step residual animation requires a component")
        fields = one_step_residual_fields(arrays, component)
        quantity = f"delta {component}"
        definition = (
            "teacher-forced one-step residual; every proposal receives exact U(t); "
            "errors contain no accumulated rollout drift"
        )
        titles = (
            f"true delta {component}",
            f"n=128 predicted delta {component}",
            "n=128 residual error",
            f"n=256 predicted delta {component}",
            "n=256 residual error",
        )
    else:
        raise ValueError(f"unsupported animation mode: {mode}")

    positions = np.asarray(arrays["positions"], dtype=np.float64)
    times = np.asarray(arrays["physical_times"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or times.shape != (80,):
        raise ValueError("visualization geometry or physical times changed")
    scales = _field_scales(fields, mode=mode)
    plt, animation = _configure_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.45), constrained_layout=True)
    names = ("truth", "n128", "n128_error", "n256", "n256_error")
    scatters = []
    for index, (axis, name, title) in enumerate(zip(axes, names, titles, strict=True)):
        is_error = "error" in name
        vmin, vmax = scales["error" if is_error else "field"]
        cmap = "RdBu_r" if is_error or mode == "one_step_residual" else "viridis"
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=fields[name][0],
            s=0.55,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        scatters.append(scatter)
    fig.colorbar(
        scatters[0], ax=[axes[0], axes[1], axes[3]], shrink=0.78, label=quantity
    )
    fig.colorbar(
        scatters[2],
        ax=[axes[2], axes[4]],
        shrink=0.78,
        label=f"{quantity} error",
    )
    trajectory = str(np.asarray(arrays["trajectory"]).item())

    def update(frame: int) -> list[Any]:
        for scatter, name in zip(scatters, names, strict=True):
            scatter.set_array(np.asarray(fields[name][frame], dtype=np.float64))
        fig.suptitle(
            f"D094 B1-C4 | trajectory {trajectory} | call {frame + 1}/79 | "
            f"t={times[frame + 1]:.3f}\n{definition}",
            fontsize=10.5,
        )
        return list(scatters)

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2600,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        raise ValueError("animation output must end in .gif or .mp4")
    movie = animation.FuncAnimation(
        fig, update, frames=79, interval=1000 / fps, blit=False
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)
    return {
        "mode": mode,
        "component": component,
        "trajectory": trajectory,
        "frames": 79,
        "fps": fps,
        "definition": definition,
        "fixed_scales": scales,
        "output": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "bundle_payload_sha256": _array_payload_sha256(arrays),
    }


def plot_endpoint_diagnostic(
    arrays: Mapping[str, np.ndarray], output_dir: Path
) -> list[dict[str, Any]]:
    pressure_fields = free_pressure_fields(arrays)
    residual_fields = one_step_residual_fields(arrays, "density")
    pressure_scales = _field_scales(pressure_fields, mode="free_pressure")
    residual_scales = _field_scales(residual_fields, mode="one_step_residual")
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    trajectory = str(np.asarray(arrays["trajectory"]).item())
    names = ("truth", "n128", "n128_error", "n256", "n256_error")
    titles = (
        "reference",
        "n=128 prediction",
        "n=128 error",
        "n=256 prediction",
        "n=256 error",
    )
    plt, _ = _configure_matplotlib()
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.2), constrained_layout=True)
    for row_index, (fields, scales, row_title, mode) in enumerate(
        (
            (pressure_fields, pressure_scales, "H79 free pressure", "pressure"),
            (
                residual_fields,
                residual_scales,
                "call-79 one-step density residual",
                "residual",
            ),
        )
    ):
        for column, (name, title) in enumerate(zip(names, titles, strict=True)):
            axis = axes[row_index, column]
            is_error = "error" in name
            vmin, vmax = scales["error" if is_error else "field"]
            cmap = "RdBu_r" if is_error or mode == "residual" else "viridis"
            scatter = axis.scatter(
                positions[:, 0],
                positions[:, 1],
                c=fields[name][-1],
                s=0.5,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.0,
                rasterized=True,
            )
            axis.set_title(title if row_index == 0 else "")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(row_title)
            if column in (1, 4):
                fig.colorbar(scatter, ax=axis, shrink=0.7)
    fig.suptitle(
        f"D094 B1-C4 trajectory {trajectory}: accumulated state vs one-step residual",
        fontsize=11,
    )
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"trajectory_{trajectory}_endpoint_diagnostic.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    plt.close(fig)
    return outputs


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = discover_training_curves(args.run_root)
    audit = validate_audit(args.audit_dir)
    audit_h79 = audit_h79_by_count_step(audit)
    curve_rows = _curve_rows(curves)
    _write_csv(args.output_dir / "loss_curve_data.csv", curve_rows)
    figures = plot_loss_curves(curves, audit_h79, args.output_dir)

    bundle_records = audit["visualization_case_selection"]["records"]
    animations = []
    endpoint_figures = []
    for bundle_record in bundle_records:
        path = args.audit_dir / str(bundle_record["path"])
        if sha256_file(path) != str(bundle_record["sha256"]):
            raise ValueError(f"visualization bundle changed: {path.name}")
        arrays = _load_npz(path)
        if str(np.asarray(arrays.get("schema")).item()) != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported visualization bundle: {path.name}")
        trajectory = str(np.asarray(arrays["trajectory"]).item())
        endpoint_figures.extend(plot_endpoint_diagnostic(arrays, args.output_dir))
        suffix = args.format
        animations.append(
            render_comparison_animation(
                arrays,
                args.output_dir / f"trajectory_{trajectory}_free_pressure.{suffix}",
                mode="free_pressure",
                fps=args.fps,
                dpi=args.dpi,
            )
        )
        for component in ("density", "energy"):
            animations.append(
                render_comparison_animation(
                    arrays,
                    args.output_dir
                    / f"trajectory_{trajectory}_one_step_{component}_residual.{suffix}",
                    mode="one_step_residual",
                    component=component,
                    fps=args.fps,
                    dpi=args.dpi,
                )
            )
        del arrays

    output = {
        "schema": SCHEMA,
        "status": "complete",
        "loss_metric_semantics": {
            "online_train_one_step_relative_l2": (
                "changing-parameter optimization trace; context only"
            ),
            "fixed_seen_train_one_step_relative_l2": (
                "frozen seen-trajectory pair bank; primary train curve"
            ),
            "fixed_validation_one_step_relative_l2": (
                "frozen 44-trajectory open-validation pair bank"
            ),
            "selection_rollout_h79_relative_l2": (
                "autonomous H79 on the frozen 16-trajectory selection cohort"
            ),
            "outside_audit_h79_relative_l2": (
                "autonomous H79 on 28 open-validation cases excluded from selection"
            ),
        },
        "animation_semantics": {
            "free_pressure": "autoregressive; state error includes accumulated drift",
            "one_step_residual": (
                "teacher-forced exact U(t) input at every call; predicted U(t+1)-U(t); "
                "no accumulated drift"
            ),
            "components_rendered": ["density", "energy"],
            "all_graph_nodes_rendered": True,
            "temporal_subsampling": False,
        },
        "loss_curve_data": {
            "path": "loss_curve_data.csv",
            "bytes": (args.output_dir / "loss_curve_data.csv").stat().st_size,
            "sha256": sha256_file(args.output_dir / "loss_curve_data.csv"),
        },
        "loss_figures": figures,
        "endpoint_figures": endpoint_figures,
        "animations": animations,
        "source_audit_summary_sha256": sha256_file(args.audit_dir / "summary.json"),
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": {
            "visualization_only": True,
            "diagnostic_cases_not_used_for_checkpoint_selection": True,
            "historical_test_population_accessed": False,
            "multi_seed_claim": False,
            "physical_conservation_claim": False,
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "loss_figures": len(figures),
                "endpoint_figures": len(endpoint_figures),
                "animations": len(animations),
            },
            sort_keys=True,
        )
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
