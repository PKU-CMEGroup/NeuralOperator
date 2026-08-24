#!/usr/bin/env python3
"""Visualize the observed D094 seed-0 data--compute response grid.

The figures contain measured cells only.  They do not interpolate between
trajectory counts or optimizer checkpoints.  The primary one-step training
metric is the fixed seen-pair bank, not the changing-parameter online trace.
Window-equivalent exposure is shown only as a derived diagonal because
``X = optimizer_step / (79 * trajectory_count)``.
"""

from __future__ import annotations

import argparse
import csv
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

from utility.time_dependent_no.pcno_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)

SCHEMA = "d094_bump_scaling_response_surface_visualization_v1"
ANALYSIS_SCHEMA = "d094_bump_scaling_ladder_analysis_v1"
ARTIFACT_SCHEMA = "d094_bump_scaling_ladder_analysis_artifacts_v1"
COUNTS = (8, 16, 32, 64, 128, 256)
ARCHITECTURES = ("pcno", "pcfno")
RAW_STEPS = tuple(range(256, 20_481, 256))
COMPARABLE_STEPS = (256, *range(1_280, 20_481, 1_280))
WINDOWS_PER_TRAJECTORY = 79
FIXED_UPDATE_SLICES = (5_120, 10_240, 15_360, 20_480)
EXPOSURE_MULTIPLIERS = (80, 160, 320)
METRICS = (
    (
        "fixed_seen_train_one_step_relative_l2",
        "Fixed seen-train one-step",
    ),
    (
        "fixed_validation_one_step_relative_l2",
        "Fixed open-validation one-step",
    ),
    (
        "rollout_h79_relative_l2",
        "Selection-cohort H79 rollout",
    ),
)
REQUIRED_COLUMNS = {
    "trajectory_count",
    "architecture",
    "epoch",
    "optimizer_step",
    "window_equivalent_exposure",
    "online_train_one_step_relative_l2",
    *(name for name, _ in METRICS),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def verify_analysis_artifacts(analysis_dir: Path) -> dict[str, Any]:
    """Verify the retained analysis packet before plotting any values."""

    if analysis_dir.is_symlink() or not analysis_dir.is_dir():
        raise ValueError("analysis directory must be a regular directory")
    manifest_path = analysis_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("historical_test_population_accessed") is not False
    ):
        raise ValueError("unsupported or unsafe D094 analysis manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TypeError("D094 analysis manifest lacks file records")
    for relative, record in files.items():
        if not isinstance(record, Mapping):
            raise TypeError("D094 analysis file record is malformed")
        path = analysis_dir / str(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"D094 analysis artifact changed: {relative}")

    analysis = _load_json(analysis_dir / "analysis.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("scientific_scope")
        != "single_seed_bump_development_scaling_analysis"
        or analysis.get("historical_test_population_accessed") is not False
        or tuple(analysis.get("counts", ())) != COUNTS
        or tuple(analysis.get("architectures", ())) != ARCHITECTURES
        or int(analysis.get("optimizer_steps", -1)) != RAW_STEPS[-1]
    ):
        raise ValueError("D094 analysis scope or grid changed")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "analysis_sha256": sha256_file(analysis_dir / "analysis.json"),
        "training_curves_sha256": sha256_file(
            analysis_dir / "training_curves.csv"
        ),
        "verified_files": len(files),
    }


def _finite_positive(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is missing or nonnumeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def load_observed_grid(
    path: Path,
) -> dict[tuple[str, int, int], dict[str, float]]:
    """Load the exact 2 x 6 x 17 common metric grid without interpolation."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError("training curve CSV lacks required columns")
        rows = list(reader)

    by_cell: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        architecture = str(row["architecture"])
        count = int(row["trajectory_count"])
        if architecture not in ARCHITECTURES or count not in COUNTS:
            raise ValueError("training curve CSV contains an unexpected cell")
        by_cell.setdefault((architecture, count), []).append(row)
    if set(by_cell) != {
        (architecture, count)
        for architecture in ARCHITECTURES
        for count in COUNTS
    }:
        raise ValueError("training curve CSV does not contain the full cell matrix")

    observed: dict[tuple[str, int, int], dict[str, float]] = {}
    for key, cell_rows in by_cell.items():
        ordered = sorted(cell_rows, key=lambda row: int(row["optimizer_step"]))
        steps = tuple(int(row["optimizer_step"]) for row in ordered)
        if steps != RAW_STEPS:
            raise ValueError(f"raw optimizer-step history changed for {key}")
        for row in ordered:
            step = int(row["optimizer_step"])
            if step not in COMPARABLE_STEPS:
                continue
            count = key[1]
            exposure = _finite_positive(
                row["window_equivalent_exposure"], field="window exposure"
            )
            expected_exposure = step / (WINDOWS_PER_TRAJECTORY * count)
            if not math.isclose(exposure, expected_exposure, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("window-equivalent exposure definition changed")
            metrics = {
                name: _finite_positive(row[name], field=name) for name, _ in METRICS
            }
            observed[(key[0], count, step)] = {
                "window_equivalent_exposure": exposure,
                **metrics,
            }
    expected_size = len(ARCHITECTURES) * len(COUNTS) * len(COMPARABLE_STEPS)
    if len(observed) != expected_size:
        raise ValueError("common observed metric grid is incomplete")
    return observed


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "axes.grid": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )
    return plt


def _cell_edges(values: Sequence[float]) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or np.any(np.diff(centers) <= 0.0):
        raise ValueError("cell centers must be strictly increasing")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        )
    )


def _matrix(
    observed: Mapping[tuple[str, int, int], Mapping[str, float]],
    architecture: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [observed[(architecture, count, step)][metric] for step in COMPARABLE_STEPS]
            for count in COUNTS
        ],
        dtype=np.float64,
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[dict[str, Any]]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return outputs


def _format_log_colorbar(colorbar: Any) -> None:
    from matplotlib.ticker import FuncFormatter

    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))


def plot_response_maps(
    observed: Mapping[tuple[str, int, int], Mapping[str, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Plot six measured-cell heatmaps with one shared scale per metric."""

    from matplotlib.colors import LogNorm

    plt = _configure_matplotlib()
    fig, axes = plt.subplots(
        len(METRICS),
        len(ARCHITECTURES),
        figsize=(10.2, 7.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    x = np.asarray(COMPARABLE_STEPS, dtype=np.float64) / 1_000.0
    y = np.log2(np.asarray(COUNTS, dtype=np.float64))
    x_edges = _cell_edges(x)
    y_edges = _cell_edges(y)
    for row_index, (metric, title) in enumerate(METRICS):
        matrices = [_matrix(observed, architecture, metric) for architecture in ARCHITECTURES]
        values = np.concatenate([matrix.reshape(-1) for matrix in matrices])
        norm = LogNorm(vmin=float(values.min()), vmax=float(values.max()))
        meshes = []
        for column, (architecture, matrix) in enumerate(
            zip(ARCHITECTURES, matrices, strict=True)
        ):
            axis = axes[row_index, column]
            mesh = axis.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap="viridis",
                norm=norm,
                shading="flat",
                edgecolors="white",
                linewidth=0.3,
                rasterized=True,
            )
            meshes.append(mesh)
            best_indices = np.argmin(matrix, axis=1)
            axis.scatter(
                x[best_indices],
                y,
                marker="o",
                s=17,
                facecolors="none",
                edgecolors="white",
                linewidths=0.75,
            )
            if row_index == 0:
                axis.set_title(architecture.upper())
            if column == 0:
                axis.set_ylabel(f"{title}\nTraining trajectories")
            if row_index == len(METRICS) - 1:
                axis.set_xlabel("Optimizer updates (thousands)")
        colorbar = fig.colorbar(meshes[-1], ax=list(axes[row_index]), shrink=0.82)
        colorbar.set_label("Relative L2 (log color scale)")
        _format_log_colorbar(colorbar)

    for axis in axes.reshape(-1):
        axis.set_yticks(y, labels=[str(count) for count in COUNTS])
        axis.set_xticks(
            np.asarray((256, 5_120, 10_240, 15_360, 20_480)) / 1_000.0,
            labels=("0.26", "5.12", "10.24", "15.36", "20.48"),
        )
    fig.suptitle(
        "D094 seed 0: observed data--compute response maps\n"
        "Measured cells only; circles mark the best observed checkpoint in each row",
        fontsize=11,
    )
    outputs = _save_figure(fig, output_dir, "d094_seed0_observed_response_maps")
    plt.close(fig)
    return outputs


def plot_architecture_ratio_maps(
    observed: Mapping[tuple[str, int, int], Mapping[str, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Plot matched-step PCNO/PCFNO ratios over the observed grid."""

    from matplotlib.colors import TwoSlopeNorm

    plt = _configure_matplotlib()
    ratio_matrices = [
        np.log2(_matrix(observed, "pcno", metric) / _matrix(observed, "pcfno", metric))
        for metric, _ in METRICS
    ]
    limit = max(float(np.abs(matrix).max()) for matrix in ratio_matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    x = np.asarray(COMPARABLE_STEPS, dtype=np.float64) / 1_000.0
    y = np.log2(np.asarray(COUNTS, dtype=np.float64))
    x_edges = _cell_edges(x)
    y_edges = _cell_edges(y)
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(11.2, 3.5), sharex=True, sharey=True, constrained_layout=True
    )
    mesh = None
    for axis, matrix, (_, title) in zip(axes, ratio_matrices, METRICS, strict=True):
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.3,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates (thousands)")
        axis.set_yticks(y, labels=[str(count) for count in COUNTS])
        axis.set_xticks(
            np.asarray((256, 5_120, 10_240, 15_360, 20_480)) / 1_000.0,
            labels=("0.26", "5.12", "10.24", "15.36", "20.48"),
        )
    axes[0].set_ylabel("Training trajectories")
    if mesh is None:
        raise AssertionError("ratio plot did not create a mesh")
    colorbar = fig.colorbar(mesh, ax=list(axes), shrink=0.82)
    colorbar.set_label("log2(PCNO / PCFNO); blue favors PCNO")
    fig.suptitle(
        "Architecture interaction over the measured grid (seed 0; no interpolation)",
        fontsize=11,
    )
    outputs = _save_figure(fig, output_dir, "d094_seed0_architecture_ratio_maps")
    plt.close(fig)
    return outputs


def plot_rollout_slices(
    observed: Mapping[tuple[str, int, int], Mapping[str, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Contrast fixed-update slices with confounded fixed-exposure diagonals."""

    plt = _configure_matplotlib()
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.3), sharey="row", constrained_layout=True)
    for row_index, architecture in enumerate(ARCHITECTURES):
        fixed_axis, exposure_axis = axes[row_index]
        for color, marker, step in zip(colors, markers, FIXED_UPDATE_SLICES, strict=True):
            values = [
                observed[(architecture, count, step)]["rollout_h79_relative_l2"]
                for count in COUNTS
            ]
            fixed_axis.plot(
                COUNTS,
                values,
                color=color,
                marker=marker,
                label=f"s={step:,}",
            )
        for color, marker, multiplier in zip(
            colors, markers, EXPOSURE_MULTIPLIERS, strict=False
        ):
            points = []
            for count in COUNTS:
                step = multiplier * count
                if (architecture, count, step) in observed:
                    points.append(
                        (
                            count,
                            observed[(architecture, count, step)][
                                "rollout_h79_relative_l2"
                            ],
                        )
                    )
            exposure_axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                marker=marker,
                label=f"X={multiplier / WINDOWS_PER_TRAJECTORY:.2f}",
            )
        for axis in (fixed_axis, exposure_axis):
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(COUNTS, labels=[str(count) for count in COUNTS])
            axis.grid(True, alpha=0.16)
            axis.set_xlabel("Training trajectories")
        fixed_axis.set_ylabel(f"{architecture.upper()} H79 relative L2")
        fixed_axis.legend(ncol=2)
        exposure_axis.legend(ncol=3)
    axes[0, 0].set_title("Fixed optimizer updates")
    axes[0, 1].set_title("Fixed window-equivalent exposure")
    axes[1, 1].text(
        0.02,
        0.04,
        "Exposure diagonals also change optimizer updates\nand learning-rate phase; not a pure data effect.",
        transform=axes[1, 1].transAxes,
        fontsize=7.2,
        va="bottom",
    )
    fig.suptitle("D094 seed 0: rollout slices through the observed response grid", fontsize=11)
    outputs = _save_figure(fig, output_dir, "d094_seed0_rollout_slices")
    plt.close(fig)
    return outputs


def _write_surface_csv(
    path: Path,
    observed: Mapping[tuple[str, int, int], Mapping[str, float]],
) -> None:
    fieldnames = [
        "architecture",
        "trajectory_count",
        "optimizer_step",
        "window_equivalent_exposure",
        *(name for name, _ in METRICS),
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for architecture in ARCHITECTURES:
            for count in COUNTS:
                for step in COMPARABLE_STEPS:
                    writer.writerow(
                        {
                            "architecture": architecture,
                            "trajectory_count": count,
                            "optimizer_step": step,
                            **observed[(architecture, count, step)],
                        }
                    )


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_analysis_artifacts(args.analysis_dir)
    observed = load_observed_grid(args.analysis_dir / "training_curves.csv")
    surface_csv = args.output_dir / "observed_surface_points.csv"
    _write_surface_csv(surface_csv, observed)
    figures = [
        *plot_response_maps(observed, args.output_dir),
        *plot_architecture_ratio_maps(observed, args.output_dir),
        *plot_rollout_slices(observed, args.output_dir),
    ]
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "input_verification": verification,
        "observed_grid": {
            "architectures": list(ARCHITECTURES),
            "trajectory_counts": list(COUNTS),
            "optimizer_steps": list(COMPARABLE_STEPS),
            "points_per_architecture": len(COUNTS) * len(COMPARABLE_STEPS),
            "total_points": len(observed),
            "independent_axes": ["trajectory_count", "optimizer_step"],
            "derived_exposure": "optimizer_step / (79 * trajectory_count)",
            "seed_count": 1,
        },
        "metric_semantics": {
            "fixed_seen_train_one_step_relative_l2": (
                "frozen seen-trajectory pair bank; primary train surface"
            ),
            "fixed_validation_one_step_relative_l2": (
                "frozen 44-trajectory open-validation pair bank"
            ),
            "rollout_h79_relative_l2": (
                "autonomous H79 on the frozen 16-trajectory checkpoint-selection cohort"
            ),
            "online_train_one_step_relative_l2": (
                "changing-parameter optimization trace; deliberately excluded from the primary surface"
            ),
        },
        "surface_points": {
            "path": surface_csv.name,
            "bytes": surface_csv.stat().st_size,
            "sha256": sha256_file(surface_csv),
        },
        "figures": figures,
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": {
            "observed_cells_only": True,
            "interpolation_used": False,
            "single_seed_development_evidence": True,
            "multi_seed_scaling_law": False,
            "causal_data_compute_architecture_interaction": False,
            "historical_test_population_accessed": False,
            "outside_selection_rollout_surface": False,
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "observed_points": len(observed),
                "figures": len(figures),
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
