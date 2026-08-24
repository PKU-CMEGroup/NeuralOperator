#!/usr/bin/env python3
"""Render D094 B1-C2 three-seed scaling and optimization diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_b1_c2 import (
    ARTIFACT_SCHEMA as ANALYSIS_ARTIFACT_SCHEMA,
    SCHEMA as ANALYSIS_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c2 import (
    ARCHITECTURES,
    CHECKPOINT_ROLES,
    SEEDS,
    TRAJECTORY_COUNTS,
)
from utility.time_dependent_no.pcno_artifacts import atomic_write_json, sha256_file

SCHEMA = "d094_b1_c2_three_seed_visualization_v1"
COLORS = {
    "pcno": "#0072B2",
    "pcfno": "#D55E00",
    "online": "#7A7A7A",
    "seen": "#0072B2",
    "validation": "#E69F00",
    "h79": "#009E73",
}
COUNT_COLORS = (
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
)
SURFACE_METRICS = (
    (
        "online_train_one_step_relative_l2",
        "Online train one-step",
    ),
    (
        "fixed_validation_one_step_relative_l2",
        "Fixed validation one-step",
    ),
    ("rollout_h79_relative_l2", "Selection-cohort H79"),
)
RATIO_METRICS = SURFACE_METRICS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("visualization metric must be finite or empty")
    return resolved


def validate_analysis(analysis_dir: Path) -> dict[str, Any]:
    analysis = _load_json(analysis_dir / "analysis.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("historical_test_population_accessed") is not False
        or analysis.get("checkpoint_reselection_performed") is not False
        or analysis.get("checkpoint_count") != 72
    ):
        raise ValueError("B1-C2 analysis is incomplete or violates its contract")
    manifest_path = analysis_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != ANALYSIS_ARTIFACT_SCHEMA:
        raise ValueError("B1-C2 analysis artifact schema changed")
    records = manifest.get("files")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("B1-C2 analysis artifact manifest is empty")
    for name, record in records.items():
        path = analysis_dir / str(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != str(record["sha256"])
        ):
            raise ValueError(f"B1-C2 analysis artifact changed: {name}")
    return {
        "analysis": analysis,
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "checked_analysis_files": len(records),
    }


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
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
            "grid.alpha": 0.15,
            "lines.linewidth": 1.7,
            "lines.markersize": 4,
        }
    )
    return plt


def _grid_edges(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.all(np.diff(array) > 0):
        raise ValueError("surface axes must be strictly increasing")
    midpoints = (array[:-1] + array[1:]) / 2.0
    return np.concatenate(
        (
            [array[0] - (midpoints[0] - array[0])],
            midpoints,
            [array[-1] + (array[-1] - midpoints[-1])],
        )
    )


def surface_matrix(
    rows: Sequence[Mapping[str, str]], architecture: str, metric: str
) -> tuple[list[int], np.ndarray]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["architecture"] != architecture:
            continue
        value = _optional_float(row.get(metric))
        if value is not None:
            grouped[(int(row["optimizer_step"]), int(row["trajectory_count"]))].append(
                value
            )
    steps = sorted(
        step
        for step in {key[0] for key in grouped}
        if all(len(grouped[(step, count)]) == len(SEEDS) for count in TRAJECTORY_COUNTS)
    )
    if not steps:
        raise ValueError(
            f"surface has no complete observations: {architecture}/{metric}"
        )
    matrix = np.asarray(
        [
            [statistics.mean(grouped[(step, count)]) for count in TRAJECTORY_COUNTS]
            for step in steps
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(matrix).all()) or bool((matrix <= 0.0).any()):
        raise ValueError("surface errors must be positive and finite")
    return steps, matrix


def ratio_surface_matrix(
    rows: Sequence[Mapping[str, str]], metric: str
) -> tuple[list[int], np.ndarray]:
    field = f"pcno_over_pcfno_{metric}"
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _optional_float(row.get(field))
        if value is not None:
            grouped[(int(row["optimizer_step"]), int(row["trajectory_count"]))].append(
                math.log2(value)
            )
    steps = sorted(
        step
        for step in {key[0] for key in grouped}
        if all(len(grouped[(step, count)]) == len(SEEDS) for count in TRAJECTORY_COUNTS)
    )
    if not steps:
        raise ValueError(f"ratio surface has no complete observations: {metric}")
    matrix = np.asarray(
        [
            [statistics.mean(grouped[(step, count)]) for count in TRAJECTORY_COUNTS]
            for step in steps
        ],
        dtype=np.float64,
    )
    return steps, matrix


def _overlay_one_pass(ax: Any) -> None:
    x = np.arange(len(TRAJECTORY_COUNTS), dtype=np.float64)
    y = np.asarray(TRAJECTORY_COUNTS, dtype=np.float64) * 79.0
    ax.plot(x, y, color="white", linewidth=2.4, linestyle="--", zorder=5)
    ax.plot(x, y, color="#222222", linewidth=0.8, linestyle="--", zorder=6)


def _save(fig: Any, output_dir: Path, stem: str, dpi: int) -> list[str]:
    names = [f"{stem}.pdf", f"{stem}.png"]
    fig.savefig(output_dir / names[0])
    fig.savefig(output_dir / names[1], dpi=dpi)
    return names


def render_training_surfaces(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    from matplotlib.colors import Normalize

    grids = {
        (architecture, metric): surface_matrix(rows, architecture, metric)
        for architecture in ARCHITECTURES
        for metric, _ in SURFACE_METRICS
    }
    norms = {}
    for metric, _ in SURFACE_METRICS:
        values = np.concatenate(
            [
                np.log10(grids[(architecture, metric)][1]).ravel()
                for architecture in ARCHITECTURES
            ]
        )
        norms[metric] = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), constrained_layout=True)
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, (metric, title) in enumerate(SURFACE_METRICS):
            ax = axes[row_index, column_index]
            steps, matrix = grids[(architecture, metric)]
            mesh = ax.pcolormesh(
                _grid_edges(range(len(TRAJECTORY_COUNTS))),
                _grid_edges(steps),
                np.log10(matrix),
                cmap="viridis_r",
                norm=norms[metric],
                shading="flat",
            )
            _overlay_one_pass(ax)
            ax.set_title(f"{architecture.upper()} — {title}")
            ax.set_xticks(range(len(TRAJECTORY_COUNTS)), TRAJECTORY_COUNTS)
            ax.set_xlabel("Training trajectories n")
            if column_index == 0:
                ax.set_ylabel("Optimizer steps")
            fig.colorbar(mesh, ax=ax, pad=0.01, label="log10 relative L2")
    fig.suptitle(
        "Observed data × compute response (three-seed mean; dashed = one dataset pass)",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c2_training_surfaces", dpi)
    plt.close(fig)
    return names


def render_ratio_surfaces(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    from matplotlib.colors import TwoSlopeNorm

    grids = {metric: ratio_surface_matrix(rows, metric) for metric, _ in RATIO_METRICS}
    maximum = max(float(np.max(np.abs(matrix))) for _, matrix in grids.values())
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    mesh = None
    for ax, (metric, title) in zip(axes, RATIO_METRICS):
        steps, matrix = grids[metric]
        mesh = ax.pcolormesh(
            _grid_edges(range(len(TRAJECTORY_COUNTS))),
            _grid_edges(steps),
            matrix,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
        )
        _overlay_one_pass(ax)
        ax.set_title(title)
        ax.set_xticks(range(len(TRAJECTORY_COUNTS)), TRAJECTORY_COUNTS)
        ax.set_xlabel("Training trajectories n")
    axes[0].set_ylabel("Optimizer steps")
    assert mesh is not None
    fig.colorbar(mesh, ax=axes, pad=0.015, label="log2(PCNO / PCFNO); blue favors PCNO")
    fig.suptitle(
        "Architecture ratio response across data and compute", fontweight="bold"
    )
    names = _save(fig, output_dir, "d094_b1_c2_architecture_ratio_surfaces", dpi)
    plt.close(fig)
    return names


def _mean_std_by_step(
    rows: Sequence[Mapping[str, str]],
    *,
    count: int,
    architecture: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["trajectory_count"]) != count or row["architecture"] != architecture:
            continue
        value = _optional_float(row.get(metric))
        if value is not None:
            grouped[int(row["optimizer_step"])].append(value)
    steps = sorted(
        step for step, values in grouped.items() if len(values) == len(SEEDS)
    )
    means = np.asarray([statistics.mean(grouped[step]) for step in steps])
    stds = np.asarray([statistics.stdev(grouped[step]) for step in steps])
    return np.asarray(steps), means, stds


def render_n128_n256_curves(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    metrics = (
        ("online_train_one_step_relative_l2", "online train", COLORS["online"]),
        ("fixed_seen_train_one_step_relative_l2", "fixed seen", COLORS["seen"]),
        (
            "fixed_validation_one_step_relative_l2",
            "fixed validation",
            COLORS["validation"],
        ),
        ("rollout_h79_relative_l2", "selection H79", COLORS["h79"]),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 6.0), sharex=True, constrained_layout=True
    )
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, count in enumerate((128, 256)):
            ax = axes[row_index, column_index]
            for metric, label, color in metrics:
                steps, means, stds = _mean_std_by_step(
                    rows,
                    count=count,
                    architecture=architecture,
                    metric=metric,
                )
                ax.plot(steps, means, color=color, label=label)
                lower = np.where(means > stds, means - stds, np.nan)
                ax.fill_between(
                    steps,
                    lower,
                    means + stds,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.axvline(79 * count, color="#444444", linestyle="--", linewidth=0.9)
            ax.set_yscale("log")
            ax.set_title(f"{architecture.upper()}, n={count}")
            ax.set_xticks(np.arange(0, 20_001, 5_000))
            ax.set_xlabel("Optimizer steps")
            if column_index == 0:
                ax.set_ylabel("Relative L2 (mean ± seed s.d.)")
    for ax in axes[-1, :]:
        ax.tick_params(labelbottom=True)
    axes[0, 1].legend(loc="best", ncol=2)
    fig.suptitle(
        "One-step and rollout dynamics at n=128 versus n=256", fontweight="bold"
    )
    names = _save(fig, output_dir, "d094_b1_c2_n128_n256_loss_curves", dpi)
    plt.close(fig)
    return names


def render_ratio_dynamics(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.3), sharex=True, constrained_layout=True
    )
    for ax, (metric, title) in zip(axes, RATIO_METRICS):
        field = f"pcno_over_pcfno_{metric}"
        for count, color in zip(TRAJECTORY_COUNTS, COUNT_COLORS):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if int(row["trajectory_count"]) != count:
                    continue
                value = _optional_float(row.get(field))
                if value is not None:
                    grouped[int(row["optimizer_step"])].append(math.log2(value))
            steps = sorted(
                step for step, values in grouped.items() if len(values) == len(SEEDS)
            )
            means = np.asarray([statistics.mean(grouped[step]) for step in steps])
            stds = np.asarray([statistics.stdev(grouped[step]) for step in steps])
            ax.plot(steps, means, color=color, label=f"n={count}")
            ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Optimizer steps")
    axes[0].set_ylabel("log2(PCNO / PCFNO), mean ± seed s.d.")
    axes[-1].legend(loc="best", ncol=2)
    fig.suptitle(
        "Does the architecture ratio remain stable during optimization?",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c2_ratio_dynamics", dpi)
    plt.close(fig)
    return names


def render_generalization_ratio_dynamics(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.3), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, architecture in zip(axes, ARCHITECTURES):
        for count, color in zip(TRAJECTORY_COUNTS, COUNT_COLORS):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if (
                    row["architecture"] != architecture
                    or int(row["trajectory_count"]) != count
                ):
                    continue
                value = _optional_float(row.get("fixed_validation_over_seen_ratio"))
                if value is not None:
                    grouped[int(row["optimizer_step"])].append(value)
            steps = sorted(
                step for step, values in grouped.items() if len(values) == len(SEEDS)
            )
            means = np.asarray([statistics.mean(grouped[step]) for step in steps])
            stds = np.asarray([statistics.stdev(grouped[step]) for step in steps])
            ax.plot(steps, means, color=color, label=f"n={count}")
            ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
        ax.axhline(1.0, color="#333333", linewidth=0.9)
        ax.set_title(architecture.upper())
        ax.set_xlabel("Optimizer steps")
    axes[0].set_ylabel("Fixed validation / fixed seen, mean +/- seed s.d.")
    axes[-1].legend(loc="best", ncol=2)
    fig.suptitle(
        "Does the one-step generalization ratio drift during optimization?",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c2_generalization_ratio_dynamics", dpi)
    plt.close(fig)
    return names


def render_checkpoint_scaling(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    cohorts = (
        ("seed_specific_outside_rollout_h79_relative_l2", "Seed-specific outside 28"),
        ("common_outside_rollout_h79_relative_l2", "Common outside 9"),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(8.6, 6.1), sharex=True, constrained_layout=True
    )
    for row_index, role in enumerate(CHECKPOINT_ROLES):
        for column_index, (field, cohort_title) in enumerate(cohorts):
            ax = axes[row_index, column_index]
            for architecture in ARCHITECTURES:
                means = []
                stds = []
                per_seed = {seed: [] for seed in SEEDS}
                for count in TRAJECTORY_COUNTS:
                    values = []
                    for seed in SEEDS:
                        matches = [
                            row
                            for row in rows
                            if int(row["seed"]) == seed
                            and int(row["trajectory_count"]) == count
                            and row["architecture"] == architecture
                            and row["checkpoint_role"] == role
                        ]
                        if len(matches) != 1:
                            raise ValueError(
                                "checkpoint visualization group is incomplete"
                            )
                        value = float(matches[0][field])
                        values.append(value)
                        per_seed[seed].append(value)
                    means.append(statistics.mean(values))
                    stds.append(statistics.stdev(values))
                x = np.arange(len(TRAJECTORY_COUNTS))
                for values in per_seed.values():
                    ax.plot(
                        x, values, color=COLORS[architecture], alpha=0.20, linewidth=0.8
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=stds,
                    color=COLORS[architecture],
                    marker="o" if architecture == "pcno" else "s",
                    capsize=2,
                    label=architecture.upper(),
                )
            ax.set_yscale("log")
            ax.set_title(f"{role.capitalize()} — {cohort_title}")
            ax.set_xticks(range(len(TRAJECTORY_COUNTS)), TRAJECTORY_COUNTS)
            ax.set_xlabel("Training trajectories n")
            if column_index == 0:
                ax.set_ylabel("H79 relative L2")
    for ax in axes[-1, :]:
        ax.tick_params(labelbottom=True)
    axes[0, 1].legend(loc="best")
    fig.suptitle(
        "Fixed-checkpoint rollout scaling: raw seeds and mean ± s.d.", fontweight="bold"
    )
    names = _save(fig, output_dir, "d094_b1_c2_checkpoint_rollout_scaling", dpi)
    plt.close(fig)
    return names


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C2 visualization output already exists")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    integrity = validate_analysis(args.analysis_dir)
    curves = _load_csv(args.analysis_dir / "training_curves.csv")
    ratios = _load_csv(args.analysis_dir / "training_architecture_ratios.csv")
    checkpoints = _load_csv(args.analysis_dir / "checkpoint_metrics.csv")
    args.output_dir.mkdir(parents=True)
    plt = _configure_matplotlib()
    figure_names = []
    figure_names.extend(
        render_training_surfaces(plt, curves, args.output_dir, args.dpi)
    )
    figure_names.extend(render_ratio_surfaces(plt, ratios, args.output_dir, args.dpi))
    figure_names.extend(render_n128_n256_curves(plt, curves, args.output_dir, args.dpi))
    figure_names.extend(render_ratio_dynamics(plt, ratios, args.output_dir, args.dpi))
    figure_names.extend(
        render_generalization_ratio_dynamics(
            plt, curves, args.output_dir, args.dpi
        )
    )
    figure_names.extend(
        render_checkpoint_scaling(plt, checkpoints, args.output_dir, args.dpi)
    )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "analysis_manifest_sha256": integrity["analysis_manifest_sha256"],
        "visualizer_sha256": sha256_file(Path(__file__)),
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in figure_names
        },
        "interpretation_limits": [
            (
                "seed-specific 28-case error bars combine initialization and "
                "cohort variation"
            ),
            "the common-nine cohort fixes cases but is small",
            "surface cells are observed means with no interpolation",
            (
                "the dashed one-pass boundary is exposure accounting, not a "
                "fitted transition"
            ),
        ],
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
