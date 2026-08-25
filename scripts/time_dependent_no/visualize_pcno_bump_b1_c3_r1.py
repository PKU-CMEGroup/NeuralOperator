#!/usr/bin/env python3
"""Render three-seed D094 exact-exposure, training-surface, and replay figures."""

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

from scripts.time_dependent_no.analyze_pcno_bump_b1_c3_r1 import (
    ANALYSIS_SCHEMA,
    ARCHITECTURES,
    ARTIFACT_SCHEMA,
    COUNTS,
    PRECISIONS,
    SEEDS,
)
from utility.time_dependent_no.pcno_artifacts import atomic_write_json, sha256_file

SCHEMA = "d094_b1_c3_three_seed_visualization_v1"
COLORS = {"pcno": "#0072B2", "pcfno": "#D55E00"}
COUNT_COLORS = {
    8: "#E69F00",
    16: "#56B4E9",
    32: "#009E73",
    64: "#F0E442",
    128: "#0072B2",
    256: "#D55E00",
}
PRECISION_STYLES = {
    "bf16": ("BF16", "-", "o", "#009E73"),
    "none": ("FP32", "--", "s", "#CC79A7"),
}


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


def validate_analysis(analysis_dir: Path) -> dict[str, Any]:
    if analysis_dir.is_symlink() or not analysis_dir.is_dir():
        raise ValueError("three-seed analysis must be a regular directory")
    analysis = _load_json(analysis_dir / "analysis.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("historical_test_population_accessed") is not False
        or analysis.get("checkpoint_reselection_performed") is not False
        or analysis.get("cell_count_per_precision") != 36
        or analysis.get("seeds") != list(SEEDS)
        or analysis.get("precision_arms") != list(PRECISIONS)
    ):
        raise ValueError("three-seed analysis contract changed")
    manifest_path = analysis_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    records = manifest.get("files")
    if manifest.get("schema") != ARTIFACT_SCHEMA or not isinstance(records, Mapping):
        raise ValueError("three-seed analysis artifact contract changed")
    for name, record in records.items():
        path = analysis_dir / str(name)
        if (
            not isinstance(record, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != str(record.get("sha256"))
        ):
            raise ValueError(f"three-seed analysis artifact changed: {name}")
    return {
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
            "legend.fontsize": 7.3,
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


def _save(fig: Any, output_dir: Path, stem: str, dpi: int) -> list[str]:
    names = [f"{stem}.pdf", f"{stem}.png"]
    fig.savefig(output_dir / names[0])
    fig.savefig(output_dir / names[1], dpi=dpi)
    return names


def grouped_values(
    rows: Sequence[Mapping[str, str]],
    *,
    precision: str,
    architecture: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    by_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
    by_count: dict[int, list[float]] = defaultdict(list)
    for count in COUNTS:
        for seed in SEEDS:
            matches = [
                row
                for row in rows
                if row["precision"] == precision
                and row["architecture"] == architecture
                and int(row["trajectory_count"]) == count
                and int(row["seed"]) == seed
            ]
            if len(matches) != 1:
                raise ValueError("three-seed exact-exposure matrix is incomplete")
            value = float(matches[0][metric])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("visualized errors must be positive and finite")
            by_seed[seed].append(value)
            by_count[count].append(value)
    return (
        np.asarray([statistics.mean(by_count[count]) for count in COUNTS]),
        np.asarray([statistics.stdev(by_count[count]) for count in COUNTS]),
        {seed: np.asarray(values) for seed, values in by_seed.items()},
    )


def architecture_ratio_values(
    rows: Sequence[Mapping[str, str]], metric: str, precision: str
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    raw: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        values = []
        for count in COUNTS:
            matches = [
                row
                for row in rows
                if row["precision"] == precision
                and int(row["seed"]) == seed
                and int(row["trajectory_count"]) == count
                and row["metric"] == metric
            ]
            if len(matches) != 1:
                raise ValueError("three-seed architecture-ratio matrix is incomplete")
            values.append(float(matches[0]["pcno_over_pcfno"]))
        raw[seed] = np.asarray(values)
    geometric = np.exp(np.mean(np.log(np.stack(list(raw.values()))), axis=0))
    return geometric, raw


def surface_matrix(
    rows: Sequence[Mapping[str, str]], architecture: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["architecture"] == architecture and row.get(metric) not in {None, ""}
    ]
    steps = np.asarray(sorted({int(row["optimizer_step"]) for row in selected}))
    matrix = np.empty((len(steps), len(COUNTS)), dtype=np.float64)
    for step_index, step in enumerate(steps):
        for count_index, count in enumerate(COUNTS):
            values = [
                float(row[metric])
                for row in selected
                if int(row["optimizer_step"]) == step
                and int(row["trajectory_count"]) == count
            ]
            if len(values) != len(SEEDS):
                raise ValueError("training-surface seed matrix is incomplete")
            matrix[step_index, count_index] = statistics.mean(values)
    if not np.all(np.isfinite(matrix)) or not np.all(matrix > 0.0):
        raise ValueError("training surface contains invalid errors")
    return steps, matrix


def render_exact_exposure_scaling(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    panels = (
        ("online_train_one_step_relative_l2", "Online train one-step"),
        ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
        ("outside_rollout_h79_relative_l2", "Outside-28 rollout H79"),
    )
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1, 3, figsize=(9.2, 3.0), sharex=True, constrained_layout=True
    )
    for ax, (metric, title) in zip(axes, panels):
        for architecture in ARCHITECTURES:
            means, stds, raw = grouped_values(
                rows, precision="bf16", architecture=architecture, metric=metric
            )
            for values in raw.values():
                ax.plot(x, values, color=COLORS[architecture], alpha=0.18, linewidth=0.7)
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
        ax.set_title(title)
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
        ax.set_ylabel("Relative L2")
    axes[-1].legend(loc="best")
    fig.suptitle(
        "Exact exposure: 64 presentations per trajectory (three seeds, BF16)",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_exact_exposure", dpi)
    plt.close(fig)
    return names


def render_architecture_ratios(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    panels = (
        ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
        ("selection_rollout_h79_relative_l2", "Selection-cohort H79"),
        ("outside_rollout_h79_relative_l2", "Outside-28 H79"),
    )
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1, 3, figsize=(9.2, 3.0), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (metric, title) in zip(axes, panels):
        for precision in PRECISIONS:
            geometric, raw = architecture_ratio_values(rows, metric, precision)
            label, linestyle, marker, color = PRECISION_STYLES[precision]
            for values in raw.values():
                ax.plot(x, values, color=color, alpha=0.18, linewidth=0.65)
            ax.plot(
                x,
                geometric,
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_yscale("log", base=2)
        ax.set_title(title)
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
    axes[0].set_ylabel("PCNO / PCFNO error\n(<1 favors PCNO)")
    axes[-1].legend(loc="best")
    fig.suptitle(
        "One-step advantage does not determine recurrent ranking",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_architecture_ratios", dpi)
    plt.close(fig)
    return names


def render_training_surfaces(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    metrics = (
        ("online_train_one_step_relative_l2", "Online train one-step"),
        ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
        ("rollout_h79_relative_l2", "Selection rollout H79"),
    )
    matrices = {
        (architecture, metric): surface_matrix(rows, architecture, metric)
        for architecture in ARCHITECTURES
        for metric, _ in metrics
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(9.2, 5.5), sharex=True, constrained_layout=True
    )
    x = np.arange(len(COUNTS))
    for column, (metric, title) in enumerate(metrics):
        logs = [
            np.log10(matrices[(architecture, metric)][1])
            for architecture in ARCHITECTURES
        ]
        vmin = min(float(values.min()) for values in logs)
        vmax = max(float(values.max()) for values in logs)
        images = []
        for row_index, architecture in enumerate(ARCHITECTURES):
            ax = axes[row_index, column]
            steps, matrix = matrices[(architecture, metric)]
            image = ax.pcolormesh(
                x,
                steps,
                np.log10(matrix),
                shading="nearest",
                cmap="viridis_r",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            images.append(image)
            ax.scatter(
                x,
                [64 * count for count in COUNTS],
                marker="D",
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
                s=20,
                zorder=3,
            )
            ax.set_yscale("log", base=2)
            ax.set_ylim(256, 20_480)
            ax.set_xticks(x, COUNTS)
            ax.set_title(title if row_index == 0 else "")
            if column == 0:
                ax.set_ylabel(f"{architecture.upper()}\noptimizer step")
            if row_index == 1:
                ax.set_xlabel("Training trajectories n")
        colorbar = fig.colorbar(images[-1], ax=axes[:, column], shrink=0.82, pad=0.02)
        colorbar.set_label("log10 relative L2")
    fig.suptitle(
        "Exposure × optimization surfaces (three-seed means; open diamonds = 64n)",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_training_surfaces", dpi)
    plt.close(fig)
    return names


def render_generalization_ratio_dynamics(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    fig, axes = plt.subplots(
        1, 2, figsize=(6.75, 3.0), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, architecture in zip(axes, ARCHITECTURES):
        for count in COUNTS:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture
                and int(row["trajectory_count"]) == count
                and row.get("fixed_validation_over_seen_ratio") not in {None, ""}
            ]
            steps = sorted({int(row["optimizer_step"]) for row in selected})
            means = []
            stds = []
            for step in steps:
                values = [
                    float(row["fixed_validation_over_seen_ratio"])
                    for row in selected
                    if int(row["optimizer_step"]) == step
                ]
                if len(values) != len(SEEDS):
                    raise ValueError("generalization-ratio seed matrix is incomplete")
                means.append(statistics.mean(values))
                stds.append(statistics.stdev(values))
            means_array = np.asarray(means)
            stds_array = np.asarray(stds)
            ax.plot(steps, means_array, color=COUNT_COLORS[count], label=f"n={count}")
            ax.fill_between(
                steps,
                means_array - stds_array,
                means_array + stds_array,
                color=COUNT_COLORS[count],
                alpha=0.10,
            )
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_title(architecture.upper())
        ax.set_xlabel("Optimizer step")
    axes[0].set_ylabel("Fixed validation / fixed seen one-step")
    axes[-1].legend(ncol=2, loc="upper left")
    fig.suptitle(
        "Near-one endpoint ratios can hide late optimization dynamics",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_validation_seen_dynamics", dpi)
    plt.close(fig)
    return names


def render_replay_divergence(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    metrics = (
        ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
        ("rollout_h79_relative_l2", "Selection rollout H79"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(6.75, 3.0), sharex=True, constrained_layout=True
    )
    for ax, (metric, title) in zip(axes, metrics):
        for count in COUNTS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["architecture"] == "pcno"
                    and int(row["trajectory_count"]) == count
                    and row["metric"] == metric
                ),
                key=lambda row: int(row["optimizer_step"]),
            )
            if not selected:
                raise ValueError("PCNO replay-divergence series is incomplete")
            ax.plot(
                [int(row["optimizer_step"]) for row in selected],
                [float(row["absolute_relative_difference"]) for row in selected],
                color=COUNT_COLORS[count],
                label=f"n={count}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.text(
            0.03,
            0.04,
            "PCFNO: every observation exact",
            transform=ax.transAxes,
            fontsize=7.5,
            color=COLORS["pcfno"],
        )
    axes[0].set_ylabel("PCNO replay absolute relative difference")
    axes[-1].legend(ncol=2, loc="upper right")
    fig.suptitle(
        "Seed-0 replay divergence is branch-specific but not yet causal",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_replay_divergence", dpi)
    plt.close(fig)
    return names


def render_precision_sensitivity(
    plt: Any, rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int
) -> list[str]:
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1, 2, figsize=(6.75, 2.9), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, architecture in zip(axes, ARCHITECTURES):
        by_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
        for seed in SEEDS:
            for count in COUNTS:
                matches = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed
                    and int(row["trajectory_count"]) == count
                    and row["architecture"] == architecture
                    and row["metric"] == "outside_rollout_h79_relative_l2"
                ]
                if len(matches) != 1:
                    raise ValueError("precision-ratio matrix is incomplete")
                by_seed[seed].append(float(matches[0]["fp32_over_bf16"]))
        matrix = np.asarray(list(by_seed.values()))
        for values in matrix:
            ax.plot(x, values, color=COLORS[architecture], alpha=0.22, linewidth=0.7)
        ax.plot(
            x,
            np.exp(np.mean(np.log(matrix), axis=0)),
            color=COLORS[architecture],
            marker="o",
        )
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_yscale("log", base=2)
        ax.set_title(architecture.upper())
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
    axes[0].set_ylabel("FP32 / BF16 outside H79")
    fig.suptitle(
        "Evaluator precision at identical checkpoint hashes",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_r1_precision_sensitivity", dpi)
    plt.close(fig)
    return names


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("three-seed visualization output already exists")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    integrity = validate_analysis(args.analysis_dir)
    cells = _load_csv(args.analysis_dir / "exact_exposure_cells.csv")
    ratios = _load_csv(args.analysis_dir / "architecture_ratios.csv")
    surface = _load_csv(args.analysis_dir / "training_surface_points.csv")
    replay = _load_csv(args.analysis_dir / "replay_trace_points.csv")
    precision = _load_csv(args.analysis_dir / "precision_ratios.csv")
    args.output_dir.mkdir(parents=True)
    plt = _configure_matplotlib()
    names = []
    names.extend(render_exact_exposure_scaling(plt, cells, args.output_dir, args.dpi))
    names.extend(render_architecture_ratios(plt, ratios, args.output_dir, args.dpi))
    names.extend(render_training_surfaces(plt, surface, args.output_dir, args.dpi))
    names.extend(
        render_generalization_ratio_dynamics(plt, surface, args.output_dir, args.dpi)
    )
    names.extend(render_replay_divergence(plt, replay, args.output_dir, args.dpi))
    names.extend(render_precision_sensitivity(plt, precision, args.output_dir, args.dpi))
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "analysis_manifest_sha256": integrity["analysis_manifest_sha256"],
        "checked_analysis_files": integrity["checked_analysis_files"],
        "visualizer_sha256": sha256_file(Path(__file__)),
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in names
        },
        "interpretation_limits": [
            "the outside-28 cohort differs by seed",
            "exact exposure does not fix optimizer updates, compute, or LR phase",
            "training surfaces share a 20,480-step schedule but are observational",
            "the historical and replay wrapper sources are not byte-identical",
            "precision arms evaluate identical checkpoints but do not retrain models",
            "no independent test population was accessed",
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
