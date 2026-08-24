#!/usr/bin/env python3
"""Render D094 B1-C3 exact-exposure and precision diagnostics."""

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

from scripts.time_dependent_no.analyze_pcno_bump_b1_c3 import (
    ANALYSIS_ARTIFACT_SCHEMA,
    ANALYSIS_SCHEMA,
    ARCHITECTURES,
    COUNTS,
    PRECISIONS,
    SEEDS,
)
from utility.time_dependent_no.pcno_artifacts import atomic_write_json, sha256_file

SCHEMA = "d094_b1_c3_exact_exposure_visualization_v1"
COLORS = {"pcno": "#0072B2", "pcfno": "#D55E00"}
PRECISION_STYLES = {
    "bf16": ("BF16", "-", "o"),
    "none": ("FP32", "--", "s"),
}
METRICS = (
    ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
    ("selection_rollout_h79_relative_l2", "Selection-cohort H79"),
    ("outside_rollout_h79_relative_l2", "Outside-28 H79"),
)


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
        raise ValueError("B1-C3 analysis must be a regular directory")
    analysis = _load_json(analysis_dir / "analysis.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("historical_test_population_accessed") is not False
        or analysis.get("checkpoint_reselection_performed") is not False
        or analysis.get("cell_count_per_precision") != 24
        or analysis.get("precision_arms") != list(PRECISIONS)
    ):
        raise ValueError("B1-C3 analysis contract changed")
    manifest_path = analysis_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    records = manifest.get("files")
    if manifest.get("schema") != ANALYSIS_ARTIFACT_SCHEMA or not isinstance(
        records, Mapping
    ):
        raise ValueError("B1-C3 analysis artifact contract changed")
    for name, record in records.items():
        path = analysis_dir / str(name)
        if (
            not isinstance(record, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != str(record.get("sha256"))
        ):
            raise ValueError(f"B1-C3 analysis artifact changed: {name}")
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
    values_by_count: dict[int, list[float]] = defaultdict(list)
    values_by_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
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
                raise ValueError(
                    "exact-exposure visualization cell matrix is incomplete"
                )
            value = float(matches[0][metric])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("visualized errors must be positive and finite")
            values_by_count[count].append(value)
            values_by_seed[seed].append(value)
    means = np.asarray(
        [statistics.mean(values_by_count[count]) for count in COUNTS]
    )
    stds = np.asarray(
        [statistics.stdev(values_by_count[count]) for count in COUNTS]
    )
    return means, stds, {
        seed: np.asarray(values, dtype=np.float64)
        for seed, values in values_by_seed.items()
    }


def render_exact_exposure_scaling(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    panels = (
        ("fixed_seen_one_step_relative_l2", "Fixed seen one-step"),
        ("fixed_validation_one_step_relative_l2", "Fixed validation one-step"),
        ("selection_rollout_h79_relative_l2", "Selection-cohort H79"),
        ("outside_rollout_h79_relative_l2", "Outside-28 H79"),
    )
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        2, 2, figsize=(8.4, 5.7), sharex=True, constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.flat, panels):
        for architecture in ARCHITECTURES:
            for precision in PRECISIONS:
                means, stds, raw = grouped_values(
                    rows,
                    precision=precision,
                    architecture=architecture,
                    metric=metric,
                )
                label, linestyle, marker = PRECISION_STYLES[precision]
                for seed_values in raw.values():
                    ax.plot(
                        x,
                        seed_values,
                        color=COLORS[architecture],
                        linestyle=linestyle,
                        linewidth=0.65,
                        alpha=0.16,
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=stds,
                    color=COLORS[architecture],
                    linestyle=linestyle,
                    marker=marker,
                    capsize=2,
                    label=f"{architecture.upper()} {label}",
                )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
        ax.set_ylabel("Relative L2")
    axes[0, 0].legend(ncol=2, loc="best")
    fig.suptitle(
        "Exact exposure: 64 balanced presentations per trajectory",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_exact_exposure_scaling", dpi)
    plt.close(fig)
    return names


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
                raise ValueError(
                    "architecture-ratio visualization matrix is incomplete"
                )
            values.append(float(matches[0]["pcno_over_pcfno"]))
        raw[seed] = np.asarray(values)
    geometric = np.exp(np.mean(np.log(np.stack(list(raw.values()))), axis=0))
    return geometric, raw


def render_architecture_ratios(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.2, 3.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (metric, title) in zip(axes, METRICS):
        for precision in PRECISIONS:
            geometric, raw = architecture_ratio_values(rows, metric, precision)
            label, linestyle, marker = PRECISION_STYLES[precision]
            for values in raw.values():
                ax.plot(x, values, color="#777777", linewidth=0.65, alpha=0.25)
            ax.plot(
                x,
                geometric,
                color="#009E73" if precision == "bf16" else "#CC79A7",
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_yscale("log", base=2)
        ax.set_title(title)
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
    axes[0].set_ylabel("PCNO / PCFNO error (lower favors PCNO)")
    axes[-1].legend(loc="best")
    fig.suptitle(
        "Gradient-branch association depends on deployment horizon",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_architecture_ratios", dpi)
    plt.close(fig)
    return names


def render_precision_sensitivity(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
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
                    raise ValueError(
                        "precision-ratio visualization matrix is incomplete"
                    )
                by_seed[seed].append(float(matches[0]["fp32_over_bf16"]))
        matrix = np.asarray(list(by_seed.values()))
        for values in matrix:
            ax.plot(x, values, color=COLORS[architecture], alpha=0.25, linewidth=0.8)
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
        "Evaluator-numerics sensitivity at identical checkpoints",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_precision_sensitivity", dpi)
    plt.close(fig)
    return names


def render_exact_vs_b1_c2(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, architecture in zip(axes, ARCHITECTURES):
        for role, linestyle, marker in (
            ("selected", "-", "o"),
            ("terminal", "--", "s"),
        ):
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
                        and row["reference_role"] == role
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "exact-versus-B1-C2 matrix is incomplete"
                        )
                    by_seed[seed].append(
                        float(matches[0]["exact_over_reference"])
                    )
            matrix = np.asarray(list(by_seed.values()))
            for values in matrix:
                ax.plot(
                    x,
                    values,
                    color=COLORS[architecture],
                    linestyle=linestyle,
                    alpha=0.20,
                    linewidth=0.7,
                )
            ax.plot(
                x,
                np.exp(np.mean(np.log(matrix), axis=0)),
                color=COLORS[architecture],
                linestyle=linestyle,
                marker=marker,
                label=f"vs {role}",
            )
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_yscale("log", base=2)
        ax.set_title(architecture.upper())
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
    axes[0].set_ylabel("Exact-exposure / B1-C2 outside H79")
    axes[-1].legend(loc="best")
    fig.suptitle(
        "Exact exposure versus selected and fixed-update endpoints",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_exact_vs_b1_c2", dpi)
    plt.close(fig)
    return names


def render_transition_alignment(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    panels = (
        ("permanent_pcno_lower_step", "Optimizer step"),
        (
            "permanent_pcno_lower_presentations_per_trajectory",
            "Presentations per trajectory",
        ),
    )
    x = np.arange(len(COUNTS))
    fig, axes = plt.subplots(
        1, 2, figsize=(6.7, 2.9), sharex=True, constrained_layout=True
    )
    for ax, (field, ylabel) in zip(axes, panels):
        for seed_index, seed in enumerate(SEEDS):
            values = []
            for count in COUNTS:
                matches = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed
                    and int(row["trajectory_count"]) == count
                ]
                if len(matches) != 1 or matches[0].get(field) in {None, ""}:
                    raise ValueError("transition-alignment matrix is incomplete")
                values.append(float(matches[0][field]))
            ax.plot(
                x,
                values,
                color=("#0072B2", "#D55E00")[seed_index],
                marker=("o", "s")[seed_index],
                label=f"seed {seed}",
            )
        ax.set_yscale("log", base=2)
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel("Training trajectories n")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Compute / scheduler coordinate")
    axes[1].set_title("Data-reuse coordinate")
    axes[-1].legend(loc="best")
    fig.suptitle(
        "First permanent H79 crossing: PCNO becomes lower than PCFNO",
        fontweight="bold",
    )
    names = _save(fig, output_dir, "d094_b1_c3_transition_alignment", dpi)
    plt.close(fig)
    return names


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C3 visualization output already exists")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    integrity = validate_analysis(args.analysis_dir)
    cells = _load_csv(args.analysis_dir / "exact_exposure_cells.csv")
    architecture_ratios = _load_csv(args.analysis_dir / "architecture_ratios.csv")
    precision_ratios = _load_csv(args.analysis_dir / "precision_ratios.csv")
    exact_vs_b1_c2 = _load_csv(args.analysis_dir / "exact_vs_b1_c2.csv")
    transition_alignment = _load_csv(
        args.analysis_dir / "b1_c2_transition_alignment.csv"
    )
    args.output_dir.mkdir(parents=True)
    plt = _configure_matplotlib()
    figure_names = []
    figure_names.extend(
        render_exact_exposure_scaling(plt, cells, args.output_dir, args.dpi)
    )
    figure_names.extend(
        render_architecture_ratios(
            plt, architecture_ratios, args.output_dir, args.dpi
        )
    )
    figure_names.extend(
        render_precision_sensitivity(
            plt, precision_ratios, args.output_dir, args.dpi
        )
    )
    figure_names.extend(
        render_exact_vs_b1_c2(
            plt, exact_vs_b1_c2, args.output_dir, args.dpi
        )
    )
    figure_names.extend(
        render_transition_alignment(
            plt, transition_alignment, args.output_dir, args.dpi
        )
    )
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
            for name in figure_names
        },
        "interpretation_limits": [
            "only two initialization seeds are present",
            "the outside-28 cohort differs by seed",
            "exact exposure does not fix optimizer updates, compute, or LR phase",
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
