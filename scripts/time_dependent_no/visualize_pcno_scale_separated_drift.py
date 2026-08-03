#!/usr/bin/env python3
"""Render fixed-contract figures and animations for scale-separated defects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_scale_separated_drift import (
    project_dct_bands,
    scale_separated_diagnostics,
)

SCHEMA = "pcno_scale_separated_drift_visualization_v1"
EXPECTED_RESULT_SCHEMAS = {
    "pcno_scale_separated_drift_diagnostic_v1",
    "pcno_scale_separated_drift_diagnostic_v2",
}
BANDS = ("large", "transition", "local")
ALL_FIELDS = ("total", *BANDS)
COLORS = {
    "total": "#2E3440",
    "large": "#0072B2",
    "transition": "#E69F00",
    "local": "#D55E00",
}
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
INSTANT_LIMIT = 1.0
CUMULATIVE_LIMIT = 10.0
GROWTH_LIMIT = 25.0
FPS = 4
GIF_DPI = 70


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--pathway-bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("plots", "animations", "all"), default="all"
    )
    parser.add_argument(
        "--sequence-kind",
        choices=(
            "free_cross_grid_defect",
            "teacher_exact_input_cross_grid_defect",
        ),
    )
    parser.add_argument(
        "--animation-family",
        choices=("scale", "pathway", "both"),
        default="scale",
    )
    parser.add_argument("--animation-case-ids", nargs="+")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_and_verify(results_dir: Path, bundle_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") not in EXPECTED_RESULT_SCHEMAS:
        raise ValueError("unexpected scale-diagnostic result schema")
    if summary.get("status") != "complete" or not summary.get("checks_passed"):
        raise ValueError("scale-diagnostic closure checks did not pass")
    for name, expected in summary["output_hashes"].items():
        path = results_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"result hash mismatch: {name}")
    registered_bundles = summary["input"]["bundle_sha256"]
    present = {path.name: path for path in bundle_dir.glob("*.npz")}
    if set(present) != set(registered_bundles):
        raise ValueError("visual bundle population differs from the bound analysis")
    for name, expected in registered_bundles.items():
        if sha256_file(present[name]) != expected:
            raise ValueError(f"bundle hash mismatch: {name}")
    return summary


def _verify_pathway_bundles(
    summary: Mapping[str, Any], bundle_dir: Path
) -> dict[str, Path]:
    pathway = summary.get("input", {}).get("pathway_replay")
    if not isinstance(pathway, Mapping):
        raise TypeError("results do not bind a pathway replay")
    registered = pathway.get("bundle_sha256")
    if not isinstance(registered, Mapping) or not registered:
        raise ValueError("results do not register pathway bundles")
    present = {path.name: path for path in bundle_dir.glob("*.npz")}
    if set(present) != set(registered):
        raise ValueError("pathway bundle population differs from the bound analysis")
    for name, expected in registered.items():
        if sha256_file(present[str(name)]) != str(expected):
            raise ValueError(f"pathway bundle hash mismatch: {name}")
    return present


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
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
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
        }
    )


def _save_figure(fig: Any, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _lag_one(row: Mapping[str, str]) -> float | None:
    values = json.loads(row["lag_correlations"])
    if not values:
        return None
    return _number(values[0]["correlation"])


def plot_accumulation_summary(
    aggregate: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    categories = sorted({(row["target"], row["case_id"]) for row in aggregate})
    lookup = {(row["target"], row["case_id"], row["band"]): row for row in aggregate}
    x = np.arange(len(categories), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    axis = axes[0, 0]
    for offset, metric, hatch in (
        (-0.19, "path_energy_share", "//"),
        (0.19, "endpoint_energy_share", ""),
    ):
        bottom = np.zeros(len(categories), dtype=np.float64)
        for band in BANDS:
            values = np.asarray(
                [
                    float(lookup[target, case, band][metric])
                    for target, case in categories
                ]
            )
            axis.bar(
                x + offset,
                values,
                width=0.34,
                bottom=bottom,
                color=COLORS[band],
                edgecolor="white",
                linewidth=0.35,
                hatch=hatch,
            )
            bottom += values
    axis.set_ylabel("defect energy share")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("(a) Instantaneous path vs endpoint accumulation")
    labels = [
        f"{case.replace('sv_', '')}\n{target.replace('x', '').replace('->', '→')}"
        for target, case in categories
    ]
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    legend = [Patch(facecolor=COLORS[band], label=band) for band in BANDS]
    legend.extend(
        (
            Patch(facecolor="white", edgecolor="#555555", hatch="//", label="path"),
            Patch(facecolor="white", edgecolor="#555555", label="endpoint"),
        )
    )
    axis.legend(handles=legend, ncol=5, loc="upper center")

    metric_specs = (
        ("temporal_coherence", "band coherence kappa", "(b) Net/path coherence"),
        (
            "coherent_mean_fraction",
            "coherent-mean energy fraction",
            "(c) Persistent temporal mean",
        ),
        (
            "lag1",
            "centered lag-1 correlation",
            "(d) Short-lag fluctuation persistence",
        ),
    )
    markers = ("o", "s", "^", "D", "v", "P", "X")
    case_markers = {
        case: markers[index % len(markers)]
        for index, case in enumerate(sorted({case for _, case in categories}))
    }
    targets = sorted({target for target, _ in categories})
    for axis, (metric, ylabel, title) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]), metric_specs, strict=True
    ):
        for band_index, band in enumerate(BANDS):
            for case_index, case in enumerate(sorted(case_markers)):
                values = []
                positions = []
                for target_index, target in enumerate(targets):
                    row = lookup[target, case, band]
                    value = _lag_one(row) if metric == "lag1" else _number(row[metric])
                    if value is not None:
                        positions.append(
                            target_index
                            + (band_index - 1) * 0.09
                            + (case_index - 0.5) * 0.025
                        )
                        values.append(value)
                axis.plot(
                    positions,
                    values,
                    color=COLORS[band],
                    marker=case_markers[case],
                    linestyle="-",
                    markersize=5,
                    alpha=0.9,
                )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_xticks(range(len(targets)))
        axis.set_xticklabels(
            [target.replace("x", "").replace("->", "→") for target in targets]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if metric != "lag1":
            axis.set_ylim(-0.02, 1.02)
    handles = [
        Line2D([], [], color=COLORS[band], marker="o", label=band) for band in BANDS
    ]
    handles.extend(
        Line2D([], [], color="#555555", marker=marker, linestyle="", label=case)
        for case, marker in case_markers.items()
    )
    axes[1, 1].legend(handles=handles, ncol=2, loc="best")
    fig.suptitle(
        f"D066 {sequence_label}: spatial scale and temporal persistence",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(fig, output_dir / "scale_accumulation_summary")


def _series(
    rows: list[dict[str, str]],
    *,
    case_id: str,
    target: str,
    band: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row
            for row in rows
            if row["case_id"] == case_id
            and row["target"] == target
            and row["band"] == band
        ),
        key=lambda row: int(row["step"]),
    )
    return (
        np.asarray([float(row["physical_time"]) for row in selected]),
        np.asarray([float(row[metric]) for row in selected]),
    )


def plot_scale_evolution(
    time_rows: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    cases = sorted({row["case_id"] for row in time_rows})
    targets = sorted({row["target"] for row in time_rows})
    styles = ("-", "--", "-.", ":")
    linestyles = {case: styles[index % len(styles)] for index, case in enumerate(cases)}
    panels = (
        ("instantaneous_rms", "instantaneous RMS [residual scale]"),
        ("instantaneous_energy_share", "instantaneous defect-energy share"),
        ("instantaneous_relative_to_true", "instantaneous / true-increment RMS"),
        ("cumulative_rms", "cumulative RMS [residual scale]"),
        ("cumulative_energy_share", "cumulative defect-energy share"),
        ("temporal_coherence", "prefix temporal coherence kappa"),
    )
    for target in targets:
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.1), sharex=True)
        for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
            fields = BANDS if metric.endswith("energy_share") else ALL_FIELDS
            for field in fields:
                for case in cases:
                    times, values = _series(
                        time_rows,
                        case_id=case,
                        target=target,
                        band=field,
                        metric=metric,
                    )
                    axis.plot(
                        times,
                        values,
                        color=COLORS[field],
                        linestyle=linestyles[case],
                    )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("physical time")
            if metric.endswith("energy_share") or metric == "temporal_coherence":
                axis.set_ylim(-0.02, 1.02)
        handles = [
            Line2D([], [], color=COLORS[field], label=field) for field in ALL_FIELDS
        ]
        handles.extend(
            Line2D([], [], color="#555555", linestyle=style, label=case)
            for case, style in linestyles.items()
        )
        axes[0, 2].legend(handles=handles, ncol=2, loc="best")
        fig.suptitle(
            f"D066 {sequence_label} scale evolution: {target}",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        filename = "scale_evolution_" + target.replace("->", "_to_")
        _save_figure(fig, output_dir / filename)


def plot_lag_correlations(
    aggregate: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    targets = sorted({row["target"] for row in aggregate})
    cases = sorted({row["case_id"] for row in aggregate})
    styles = ("-", "--", "-.", ":")
    linestyles = {case: styles[index % len(styles)] for index, case in enumerate(cases)}
    fig, axes = plt.subplots(1, len(targets), figsize=(9.0, 3.2), sharey=True)
    axes_array = np.atleast_1d(axes)
    for axis, target in zip(axes_array, targets, strict=True):
        for band in BANDS:
            for case in cases:
                row = next(
                    item
                    for item in aggregate
                    if item["target"] == target
                    and item["case_id"] == case
                    and item["band"] == band
                )
                lag_rows = json.loads(row["lag_correlations"])
                axis.plot(
                    [item["lag"] for item in lag_rows],
                    [item["correlation"] for item in lag_rows],
                    color=COLORS[band],
                    linestyle=linestyles[case],
                )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(target)
        axis.set_xlabel("lag [learned calls]")
    axes_array[0].set_ylabel("centered aggregate spatial lag correlation")
    handles = [Line2D([], [], color=COLORS[band], label=band) for band in BANDS]
    handles.extend(
        Line2D([], [], color="#555555", linestyle=style, label=case)
        for case, style in linestyles.items()
    )
    axes_array[-1].legend(handles=handles, ncol=2, loc="best")
    fig.suptitle(
        f"D066 {sequence_label} centered temporal lag structure",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(fig, output_dir / "scale_lag_correlations")


def plot_pod_summary(
    aggregate: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    """Summarize temporal POD concentration without pooling mesh nodes."""

    targets = sorted({row["target"] for row in aggregate})
    metric_specs = (
        (
            "uncentered_first_mode_energy_fraction",
            "energy fraction",
            "first POD mode",
        ),
        (
            "uncentered_first_three_energy_fraction",
            "energy fraction",
            "first three POD modes",
        ),
        (
            "uncentered_modes_for_95_percent",
            "number of modes",
            "modes for 95% energy",
        ),
        (
            "centered_modes_for_95_percent",
            "number of modes",
            "centered modes for 95%",
        ),
    )
    offsets = np.linspace(-0.27, 0.27, len(ALL_FIELDS))
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(13.0, 3.4))
    for axis, (metric, ylabel, title) in zip(axes, metric_specs, strict=True):
        for band, offset in zip(ALL_FIELDS, offsets, strict=True):
            medians = []
            lower = []
            upper = []
            positions = []
            for target_index, target in enumerate(targets):
                values = np.asarray(
                    [
                        float(row[metric])
                        for row in aggregate
                        if row["target"] == target and row["band"] == band
                    ],
                    dtype=np.float64,
                )
                if values.size == 0:
                    raise ValueError(f"missing POD rows: {target} {band} {metric}")
                median = float(np.median(values))
                positions.append(target_index + offset)
                medians.append(median)
                lower.append(median - float(np.quantile(values, 0.25)))
                upper.append(float(np.quantile(values, 0.75)) - median)
            axis.errorbar(
                positions,
                medians,
                yerr=np.asarray((lower, upper)),
                color=COLORS[band],
                marker="o",
                linestyle="",
                markersize=5,
                capsize=2,
                label=band,
            )
        axis.set_xticks(range(len(targets)))
        axis.set_xticklabels(
            [target.replace("x", "").replace("->", "→") for target in targets]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if "fraction" in metric:
            axis.set_ylim(0.0, 1.0)
    axes[-1].legend(ncol=2, loc="upper left")
    fig.suptitle(
        f"Temporal POD/SVD concentration: {sequence_label}\n"
        "median and IQR over six cases; residual-scaled fields",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_figure(fig, output_dir / "pod_temporal_rank_summary")


def plot_signed_growth_evolution(
    time_rows: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    """Plot per-case-normalized interaction and accumulated signed growth."""

    cases = sorted({row["case_id"] for row in time_rows})
    targets = sorted({row["target"] for row in time_rows})
    for target in targets:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
        for field in ALL_FIELDS:
            interaction_series = []
            cumulative_growth_series = []
            times = None
            for case in cases:
                case_times, interaction = _series(
                    time_rows,
                    case_id=case,
                    target=target,
                    band=field,
                    metric="signed_interaction_over_total_path_energy",
                )
                _, growth = _series(
                    time_rows,
                    case_id=case,
                    target=target,
                    band=field,
                    metric="signed_growth_over_total_path_energy",
                )
                times = case_times if times is None else times
                interaction_series.append(interaction)
                cumulative_growth_series.append(np.cumsum(growth))
            if times is None:
                raise AssertionError("signed-growth plot has no time samples")
            for axis, values in zip(
                axes,
                (interaction_series, cumulative_growth_series),
                strict=True,
            ):
                matrix = np.stack(values)
                median = np.median(matrix, axis=0)
                lower = np.quantile(matrix, 0.25, axis=0)
                upper = np.quantile(matrix, 0.75, axis=0)
                axis.plot(times, median, color=COLORS[field], label=field)
                axis.fill_between(times, lower, upper, color=COLORS[field], alpha=0.14)
        axes[0].axhline(0.0, color="#777777", linewidth=0.7)
        axes[0].set_ylabel(r"$2\langle e_n,\delta_n\rangle$ / total path energy")
        axes[0].set_title("(a) New defect interaction with the incoming state gap")
        axes[1].axhline(0.0, color="#777777", linewidth=0.7)
        axes[1].set_ylabel("cumulative signed growth / total path energy")
        axes[1].set_xlabel("physical time")
        axes[1].set_title("(b) Accumulated contribution to cross-grid gap energy")
        axes[0].legend(ncol=4, loc="best")
        fig.suptitle(
            f"D066 {sequence_label} signed growth: {target}\n"
            "median and interquartile range over cases; no node pooling",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        filename = "scale_signed_growth_" + target.replace("->", "_to_")
        _save_figure(fig, output_dir / filename)


def _component_summary_matrix(
    rows: list[dict[str, str]], *, target: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.zeros((len(COMPONENTS), len(BANDS)), dtype=np.float64)
    lower = np.zeros_like(median)
    upper = np.zeros_like(median)
    for component_index, component in enumerate(COMPONENTS):
        for band_index, band in enumerate(BANDS):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in rows
                    if row["target"] == target
                    and row["component"] == component
                    and row["band"] == band
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                raise ValueError(
                    f"missing component summary rows: {target} {component} {band}"
                )
            median[component_index, band_index] = np.median(values)
            lower[component_index, band_index] = np.quantile(values, 0.25)
            upper[component_index, band_index] = np.quantile(values, 0.75)
    return median, lower, upper


def plot_component_summary(
    rows: list[dict[str, str]], output_dir: Path, *, sequence_label: str
) -> None:
    metric_specs = (
        ("path_energy_share", "path-energy share", "YlGnBu", 0.0, 1.0),
        ("endpoint_energy_share", "endpoint-energy share", "YlGnBu", 0.0, 1.0),
        ("temporal_coherence", "temporal coherence", "YlGnBu", 0.0, 1.0),
        (
            "aggregate_signed_interaction_over_defect_energy",
            "signed interaction / defect energy",
            "RdBu_r",
            -2.0,
            2.0,
        ),
    )
    for target in sorted({row["target"] for row in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
        for axis, (metric, title, cmap, vmin, vmax) in zip(
            axes.flat, metric_specs, strict=True
        ):
            median, lower, upper = _component_summary_matrix(
                rows, target=target, metric=metric
            )
            image = axis.imshow(
                median,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            for component_index in range(len(COMPONENTS)):
                for band_index in range(len(BANDS)):
                    value = median[component_index, band_index]
                    text_color = (
                        "white"
                        if abs(value) > 0.72 * max(abs(vmin), abs(vmax))
                        else "#222222"
                    )
                    axis.text(
                        band_index,
                        component_index,
                        f"{value:.2f}\n[{lower[component_index, band_index]:.2f},"
                        f" {upper[component_index, band_index]:.2f}]",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )
            axis.set_xticks(range(len(BANDS)))
            axis.set_xticklabels(BANDS)
            axis.set_yticks(range(len(COMPONENTS)))
            axis.set_yticklabels(COMPONENTS)
            axis.set_title(title)
            axis.grid(False)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"D066 {sequence_label} band x conservative component: {target}\n"
            "median [IQR] over cases; per-case mesh aggregation precedes summary",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        filename = "scale_component_summary_" + target.replace("->", "_to_")
        _save_figure(fig, output_dir / filename)


def _median_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return tuple(float(value) for value in np.quantile(array, (0.5, 0.25, 0.75)))


def plot_pathway_attribution(
    rows: Sequence[Mapping[str, str]], output_dir: Path
) -> None:
    fields = (
        (
            "path_mesh_symmetric_attribution_share",
            "path_state_symmetric_attribution_share",
            "path energy attribution",
        ),
        (
            "endpoint_mesh_symmetric_attribution_share",
            "endpoint_state_symmetric_attribution_share",
            "endpoint energy attribution",
        ),
    )
    for target in sorted({row["target"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["target"] == target
            and row["region"] == "all"
            and row["band"] in BANDS
        ]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
        x = np.arange(len(BANDS), dtype=np.float64)
        width = 0.34
        for axis, (mesh_field, state_field, title) in zip(axes, fields, strict=True):
            for offset, field, label, color in (
                (-width / 2, mesh_field, "mesh", "#0072B2"),
                (width / 2, state_field, "state response", "#D55E00"),
            ):
                medians: list[float] = []
                lower: list[float] = []
                upper: list[float] = []
                for band in BANDS:
                    values = [
                        float(row[field]) for row in selected if row["band"] == band
                    ]
                    median, q25, q75 = _median_interval(values)
                    medians.append(median)
                    lower.append(median - q25)
                    upper.append(q75 - median)
                axis.bar(
                    x + offset, medians, width, color=color, alpha=0.85, label=label
                )
                axis.errorbar(
                    x + offset,
                    medians,
                    yerr=np.asarray((lower, upper)),
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.8,
                    capsize=2,
                )
            axis.axhline(0.0, color="#666666", linewidth=0.8)
            axis.set_xticks(x, BANDS)
            axis.set_title(title)
            axis.set_ylabel("symmetric share of total")
        axes[0].legend(ncol=2, loc="best")
        fig.suptitle(
            f"Free-defect pathway attribution: {target}\nmedian [IQR] over six cases",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        _save_figure(fig, output_dir / f"pathway_attribution_{_pair_key(target)}")


def plot_pathway_growth(
    time_rows: Sequence[Mapping[str, str]],
    aggregate_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> None:
    denominators = {
        (row["case_id"], row["target"]): float(row["path_total_energy"])
        for row in aggregate_rows
        if row["band"] == "total" and row["region"] == "all"
    }
    pathway_fields = (
        ("signed_growth_total", "total", "-"),
        ("signed_growth_mesh_attribution", "mesh", "--"),
        ("signed_growth_state_attribution", "state", ":"),
    )
    for target in sorted({row["target"] for row in time_rows}):
        selected = [
            row
            for row in time_rows
            if row["target"] == target
            and row["region"] == "all"
            and row["band"] in BANDS
        ]
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), sharex=True, sharey=True)
        for axis, band in zip(axes, BANDS, strict=True):
            band_rows = [row for row in selected if row["band"] == band]
            times = sorted({float(row["physical_time"]) for row in band_rows})
            for field, label, linestyle in pathway_fields:
                medians = []
                for time in times:
                    values = [
                        float(row[field]) / denominators[row["case_id"], row["target"]]
                        for row in band_rows
                        if float(row["physical_time"]) == time
                    ]
                    medians.append(float(np.median(values)))
                axis.plot(times, medians, label=label, linestyle=linestyle)
            axis.axhline(0.0, color="#666666", linewidth=0.8)
            axis.set_title(band)
            axis.set_xlabel("physical time")
        axes[0].set_ylabel("signed growth / total path energy")
        axes[0].legend(loc="best")
        fig.suptitle(
            f"Band-by-pathway signed growth: {target}\nmedian over six cases",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        _save_figure(fig, output_dir / f"pathway_signed_growth_{_pair_key(target)}")


def plot_region_accumulation(
    rows: Sequence[Mapping[str, str]], output_dir: Path
) -> None:
    region_labels = {
        "partition_boundary": "boundary <= 0.05",
        "partition_shock": "shock envelope",
        "partition_vortex": "vortex core",
        "partition_smooth": "smooth remainder",
    }
    colors = ("#7B8794", "#D55E00", "#009E73", "#0072B2")
    for target in sorted({row["target"] for row in rows}):
        selected = [row for row in rows if row["target"] == target]
        all_energy = {
            (row["case_id"], row["band"], row["physical_time"]): float(
                row["cumulative_total_energy"]
            )
            for row in selected
            if row["region"] == "all"
        }
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True)
        for axis, band in zip(axes, ("large", "local"), strict=True):
            band_rows = [
                row
                for row in selected
                if row["band"] == band and row["region"] in region_labels
            ]
            times = sorted({float(row["physical_time"]) for row in band_rows})
            for (region, label), color in zip(
                region_labels.items(), colors, strict=True
            ):
                medians = []
                for time in times:
                    values = []
                    for row in band_rows:
                        if (
                            row["region"] != region
                            or float(row["physical_time"]) != time
                        ):
                            continue
                        denominator = all_energy[
                            row["case_id"], row["band"], row["physical_time"]
                        ]
                        values.append(
                            float(row["cumulative_total_energy"]) / denominator
                        )
                    medians.append(float(np.median(values)))
                axis.plot(times, medians, color=color, label=label)
            axis.set_title(f"{band} cumulative defect")
            axis.set_xlabel("physical time")
        axes[0].set_ylabel("regional energy share")
        axes[1].legend(loc="best", fontsize=7)
        fig.suptitle(
            f"Where the cumulative defect lives: {target}\nmedian over six cases",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        _save_figure(fig, output_dir / f"region_accumulation_{_pair_key(target)}")


def plot_shock_profile(rows: Sequence[Mapping[str, str]], output_dir: Path) -> None:
    comparisons = (
        ("coarse_vs_reference", "coarse vs reference", "#0072B2"),
        ("restricted_fine_vs_reference", "restricted fine vs reference", "#E69F00"),
        ("coarse_vs_restricted_fine", "coarse vs restricted fine", "#D55E00"),
    )
    panels = (
        ("rms_shock_position_error", "shock-position RMS", "distance"),
        (
            "median_absolute_net_strength_log_error",
            "net-strength error",
            "absolute log ratio",
        ),
        (
            "median_absolute_total_variation_log_error",
            "profile-variation error",
            "absolute log ratio",
        ),
        (
            "median_absolute_thickness_log_error",
            "thickness error",
            "absolute log ratio",
        ),
    )
    for target in sorted({row["target"] for row in rows}):
        selected = [row for row in rows if row["target"] == target]
        fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), sharex=True)
        flat = list(axes.flat)
        for axis, (field, title, ylabel) in zip(flat[:4], panels, strict=True):
            for comparison, label, color in comparisons:
                comparison_rows = [
                    row for row in selected if row["comparison"] == comparison
                ]
                times = sorted({float(row["physical_time"]) for row in comparison_rows})
                medians = [
                    float(
                        np.median(
                            [
                                float(row[field])
                                for row in comparison_rows
                                if float(row["physical_time"]) == time
                            ]
                        )
                    )
                    for time in times
                ]
                axis.plot(times, medians, label=label, color=color)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        cross_rows = [
            row for row in selected if row["comparison"] == "coarse_vs_restricted_fine"
        ]
        times = sorted({float(row["physical_time"]) for row in cross_rows})
        for field, label, linestyle in (
            ("translation_only_energy_fraction", "translation only", "--"),
            (
                "joint_translation_dilation_amplitude_energy_fraction",
                "translation + dilation + amplitude",
                "-",
            ),
        ):
            medians = []
            for time in times:
                values = [
                    float(row[field])
                    for row in cross_rows
                    if float(row["physical_time"]) == time and row[field] != ""
                ]
                medians.append(float(np.median(values)) if values else np.nan)
            flat[4].plot(times, medians, label=label, linestyle=linestyle)
        flat[4].set_title("cross-grid shock-mode explanation")
        flat[4].set_ylabel("shock-envelope energy fraction")
        flat[4].legend(loc="best")
        flat[5].axis("off")
        flat[5].legend(
            handles=[
                Line2D([0], [0], color=color, label=label)
                for _, label, color in comparisons
            ],
            loc="center",
        )
        for axis in flat[:5]:
            axis.set_xlabel("physical time")
        fig.suptitle(
            f"Shock profile structure: {target}\nmedian over six cases; residual-scale modal fit",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        _save_figure(fig, output_dir / f"shock_profile_{_pair_key(target)}")


def _pair_key(target: str) -> str:
    return target.replace("->", "_to_")


def _resolution(target: str) -> tuple[int, int]:
    coarse = target.split("->", maxsplit=1)[0]
    nx, ny = coarse.split("x", maxsplit=1)
    return int(nx), int(ny)


def _bundle_schema(bundle: Mapping[str, np.ndarray]) -> str:
    if "schema" in bundle:
        return str(np.asarray(bundle["schema"]).item())
    metadata = json.loads(str(np.asarray(bundle["metadata_json"]).item()))
    return str(metadata["schema"])


def _defect_key(schema: str, sequence_kind: str, pair_key: str) -> str:
    if schema == "pcno_resolution_pathway_diagnostic_v2":
        if sequence_kind != "teacher_exact_input_cross_grid_defect":
            raise ValueError("D065 bundles contain only exact-input baseline defects")
        return f"baseline_defect__{pair_key}"
    if sequence_kind == "free_cross_grid_defect":
        return f"delta_free__{pair_key}"
    if sequence_kind == "teacher_exact_input_cross_grid_defect":
        return f"delta_teacher__{pair_key}"
    raise ValueError(f"unsupported sequence kind: {sequence_kind}")


def animate_bundle(
    bundle_path: Path,
    *,
    target: str,
    sequence_kind: str,
    output_dir: Path,
    domain_lengths: tuple[float, float],
) -> tuple[list[Path], list[dict[str, Any]]]:
    pair_key = _pair_key(target)
    resolution = _resolution(target)
    nx, ny = resolution
    with np.load(bundle_path, allow_pickle=False) as bundle:
        schema = _bundle_schema(bundle)
        defect = np.asarray(
            bundle[_defect_key(schema, sequence_kind, pair_key)], dtype=np.float64
        )
        truth = np.asarray(bundle[f"true_increment__{pair_key}"], dtype=np.float64)
        residual_scale = np.asarray(bundle["residual_scale"], dtype=np.float64)
        times = np.asarray(bundle["physical_times"], dtype=np.float64)
    _, _, closure, projections = scale_separated_diagnostics(
        defect,
        truth,
        resolution=resolution,
        component_scale=residual_scale,
        domain_lengths=domain_lengths,
        return_projections=True,
    )
    if projections is None:
        raise AssertionError("scale projections were not returned")
    if closure["maximum_reconstruction_abs_residual_scaled"] > 5.0e-12:
        raise ValueError("animation scale projection failed reconstruction closure")

    scaled = {
        "total": defect / residual_scale[None, None, :],
        **{
            band: value / residual_scale[None, None, :]
            for band, value in projections.items()
        },
    }
    cumulative = {name: np.cumsum(value, axis=0) for name, value in scaled.items()}
    arrays = [
        scaled["total"],
        scaled["large"],
        scaled["transition"],
        scaled["local"],
        cumulative["total"],
        cumulative["large"],
        cumulative["transition"],
        cumulative["local"],
    ]
    titles = (
        "instant total defect",
        "instant large",
        "instant transition",
        "instant local",
        "cumulative total",
        "cumulative large",
        "cumulative transition",
        "cumulative local",
    )
    limits = (INSTANT_LIMIT,) * 4 + (CUMULATIVE_LIMIT,) * 4
    field_names = (*ALL_FIELDS, *ALL_FIELDS)
    temporal_kinds = ("instantaneous",) * 4 + ("cumulative",) * 4
    case_id = bundle_path.stem
    saturation_rows: list[dict[str, Any]] = []
    for field_name, temporal_kind, array, limit in zip(
        field_names, temporal_kinds, arrays, limits, strict=True
    ):
        for component_index, component in enumerate(COMPONENTS):
            absolute = np.abs(array[:, :, component_index])
            clipped_count = int(np.count_nonzero(absolute > limit))
            saturation_rows.append(
                {
                    "case_id": case_id,
                    "target": target,
                    "sequence_kind": sequence_kind,
                    "temporal_kind": temporal_kind,
                    "field": field_name,
                    "component": component,
                    "physical_limit_residual_scale": limit,
                    "maximum_absolute_residual_scale": float(absolute.max()),
                    "sample_count": int(absolute.size),
                    "clipped_sample_count": clipped_count,
                    "clipped_fraction": float(clipped_count / absolute.size),
                }
            )
    fig, axes = plt.subplots(4, 8, figsize=(18.0, 7.8), sharex=True, sharey=True)
    images: list[Any] = []
    for component, row in enumerate(axes):
        for column, axis in enumerate(row):
            field = arrays[column][0, :, component].reshape(ny, nx)
            image = axis.imshow(
                field,
                origin="lower",
                extent=(0.0, domain_lengths[0], 0.0, domain_lengths[1]),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limits[column],
                vmax=limits[column],
                interpolation="nearest",
                animated=True,
            )
            images.append(image)
            axis.grid(False)
            axis.tick_params(labelsize=6)
            if component == 0:
                axis.set_title(titles[column], fontsize=8)
            if column == 0:
                axis.set_ylabel(COMPONENTS[component], fontsize=8)
            if component == 3:
                axis.set_xlabel("x", fontsize=7)
    title = fig.suptitle("", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "DCT-II reflective bands; instantaneous +/-1 and cumulative +/-10 "
        "frozen residual-scale units; no per-frame normalization",
        ha="center",
        fontsize=8,
    )

    def update(frame: int) -> list[Any]:
        for component in range(4):
            for column, array in enumerate(arrays):
                images[component * 8 + column].set_data(
                    array[frame, :, component].reshape(ny, nx)
                )
        title.set_text(
            f"{case_id} | {target} | {sequence_kind} | t={times[frame]:.2f} | "
            "model_all_nodes raw recurrence"
        )
        return [*images, title]

    fig.subplots_adjust(
        left=0.045, right=0.995, bottom=0.07, top=0.90, wspace=0.08, hspace=0.08
    )
    update(0)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=times.size,
        interval=1000 / FPS,
        blit=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_{pair_key}_{sequence_kind}"
    gif_path = output_dir / f"{stem}.gif"
    png_path = output_dir / f"{stem}_final.png"
    movie.save(gif_path, writer=animation.PillowWriter(fps=FPS), dpi=GIF_DPI)
    update(times.size - 1)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return [gif_path, png_path], saturation_rows


def _region_overlay(
    reference_state: np.ndarray,
    nodes: np.ndarray,
    *,
    resolution: tuple[int, int],
    gamma: float,
) -> np.ndarray:
    nx, ny = resolution
    masks, _, _ = shock_vortex_regions(
        reference_state, nodes, resolution=resolution, gamma=gamma
    )
    overlay = np.zeros((ny * nx, 4), dtype=np.float64)
    overlay[masks["partition_boundary"]] = (0.35, 0.35, 0.35, 0.12)
    overlay[masks["partition_shock"]] = (0.84, 0.37, 0.00, 0.16)
    overlay[masks["partition_vortex"]] = (0.00, 0.62, 0.45, 0.16)
    return overlay.reshape(ny, nx, 4)


def _render_pathway_movie(
    arrays: Sequence[np.ndarray],
    titles: Sequence[str],
    limits: Sequence[float],
    field_names: Sequence[str],
    temporal_kinds: Sequence[str],
    *,
    reference_states: np.ndarray,
    nodes: np.ndarray,
    times: np.ndarray,
    resolution: tuple[int, int],
    gamma: float,
    domain_lengths: tuple[float, float],
    case_id: str,
    target: str,
    family: str,
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    nx, ny = resolution
    if not (
        len(arrays)
        == len(titles)
        == len(limits)
        == len(field_names)
        == len(temporal_kinds)
        == 6
    ):
        raise ValueError("pathway movie contract requires six fields")
    saturation_rows: list[dict[str, Any]] = []
    for field_name, temporal_kind, array, limit in zip(
        field_names, temporal_kinds, arrays, limits, strict=True
    ):
        for component_index, component in enumerate(COMPONENTS):
            absolute = np.abs(array[:, :, component_index])
            clipped_count = int(np.count_nonzero(absolute > limit))
            saturation_rows.append(
                {
                    "case_id": case_id,
                    "target": target,
                    "animation_family": family,
                    "temporal_kind": temporal_kind,
                    "field": field_name,
                    "component": component,
                    "fixed_limit": limit,
                    "limit_units": (
                        "squared residual-scale units"
                        if family == "signed_growth"
                        else "residual-scale units"
                    ),
                    "maximum_absolute_scaled_value": float(absolute.max()),
                    "sample_count": int(absolute.size),
                    "clipped_sample_count": clipped_count,
                    "clipped_fraction": float(clipped_count / absolute.size),
                }
            )

    fig, axes = plt.subplots(4, 6, figsize=(14.0, 7.8), sharex=True, sharey=True)
    images: list[Any] = []
    overlays: list[Any] = []
    first_overlay = _region_overlay(
        reference_states[1], nodes, resolution=resolution, gamma=gamma
    )
    for component, row in enumerate(axes):
        for column, axis in enumerate(row):
            image = axis.imshow(
                arrays[column][0, :, component].reshape(ny, nx),
                origin="lower",
                extent=(0.0, domain_lengths[0], 0.0, domain_lengths[1]),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limits[column],
                vmax=limits[column],
                interpolation="nearest",
                animated=True,
            )
            overlay = axis.imshow(
                first_overlay,
                origin="lower",
                extent=(0.0, domain_lengths[0], 0.0, domain_lengths[1]),
                aspect="auto",
                interpolation="nearest",
                animated=True,
            )
            images.append(image)
            overlays.append(overlay)
            axis.grid(False)
            axis.tick_params(labelsize=6)
            if component == 0:
                axis.set_title(titles[column], fontsize=8)
            if column == 0:
                axis.set_ylabel(COMPONENTS[component], fontsize=8)
            if component == 3:
                axis.set_xlabel("x", fontsize=7)
    title = fig.suptitle("", fontsize=11, fontweight="bold")
    fig.legend(
        handles=(
            Patch(color=(0.35, 0.35, 0.35, 0.25), label="boundary <= 0.05"),
            Patch(color=(0.84, 0.37, 0.00, 0.25), label="shock envelope"),
            Patch(color=(0.00, 0.62, 0.45, 0.25), label="vortex core"),
        ),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.text(
        0.5,
        0.04,
        (
            "fixed +/-25 squared residual-scale units"
            if family == "signed_growth"
            else "fixed +/-1 instantaneous and +/-10 cumulative residual-scale units"
        )
        + "; reference-output region overlay; no per-frame normalization",
        ha="center",
        fontsize=8,
    )

    def update(frame: int) -> list[Any]:
        overlay_value = _region_overlay(
            reference_states[frame + 1],
            nodes,
            resolution=resolution,
            gamma=gamma,
        )
        for component in range(4):
            for column, array in enumerate(arrays):
                index = component * 6 + column
                images[index].set_data(array[frame, :, component].reshape(ny, nx))
                overlays[index].set_data(overlay_value)
        title.set_text(
            f"{case_id} | {target} | {family} | t={times[frame]:.2f} | "
            "model_all_nodes raw recurrence"
        )
        return [*images, *overlays, title]

    fig.subplots_adjust(
        left=0.05, right=0.995, bottom=0.09, top=0.90, wspace=0.08, hspace=0.08
    )
    update(0)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=times.size,
        interval=1000 / FPS,
        blit=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_{_pair_key(target)}_{family}"
    gif_path = output_dir / f"{stem}.gif"
    png_path = output_dir / f"{stem}_final.png"
    movie.save(gif_path, writer=animation.PillowWriter(fps=FPS), dpi=GIF_DPI)
    update(times.size - 1)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return [gif_path, png_path], saturation_rows


def animate_pathway_bundle(
    bundle_path: Path,
    *,
    target: str,
    output_dir: Path,
    domain_lengths: tuple[float, float],
    gamma: float,
) -> tuple[list[Path], list[dict[str, Any]]]:
    pair_key = _pair_key(target)
    resolution = _resolution(target)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        mesh = np.asarray(bundle[f"mesh_delta__{pair_key}"], dtype=np.float64)
        state = np.asarray(bundle[f"state_delta__{pair_key}"], dtype=np.float64)
        reference = np.asarray(
            bundle[f"reference_states__{pair_key}"], dtype=np.float64
        )
        nodes = np.asarray(bundle[f"nodes__{pair_key}"], dtype=np.float64)
        residual_scale = np.asarray(bundle["residual_scale"], dtype=np.float64)
        times = np.asarray(bundle["physical_times"], dtype=np.float64)
    mesh_projection, mesh_closure = project_dct_bands(
        mesh,
        resolution=resolution,
        component_scale=residual_scale,
        domain_lengths=domain_lengths,
    )
    state_projection, state_closure = project_dct_bands(
        state,
        resolution=resolution,
        component_scale=residual_scale,
        domain_lengths=domain_lengths,
    )
    if (
        max(
            mesh_closure["maximum_reconstruction_abs_residual_scaled"],
            state_closure["maximum_reconstruction_abs_residual_scaled"],
        )
        > 5.0e-12
    ):
        raise ValueError("pathway animation band projection failed closure")
    total = mesh + state
    scaled_mesh = mesh / residual_scale[None, None, :]
    scaled_state = state / residual_scale[None, None, :]
    scaled_total = total / residual_scale[None, None, :]
    cumulative_mesh = np.cumsum(scaled_mesh, axis=0)
    cumulative_state = np.cumsum(scaled_state, axis=0)
    cumulative_total = cumulative_mesh + cumulative_state
    generated, saturation = _render_pathway_movie(
        (
            scaled_total,
            scaled_mesh,
            scaled_state,
            cumulative_total,
            cumulative_mesh,
            cumulative_state,
        ),
        (
            "instant total",
            "instant mesh",
            "instant state",
            "cumulative total",
            "cumulative mesh",
            "cumulative state",
        ),
        (INSTANT_LIMIT,) * 3 + (CUMULATIVE_LIMIT,) * 3,
        ("total", "mesh", "state", "total", "mesh", "state"),
        ("instantaneous",) * 3 + ("cumulative",) * 3,
        reference_states=reference,
        nodes=nodes,
        times=times,
        resolution=resolution,
        gamma=gamma,
        domain_lengths=domain_lengths,
        case_id=bundle_path.stem,
        target=target,
        family="pathway_fields",
        output_dir=output_dir,
    )

    growth_arrays: list[np.ndarray] = []
    growth_titles: list[str] = []
    growth_names: list[str] = []
    for band in ("large", "local"):
        band_mesh = mesh_projection[band] / residual_scale[None, None, :]
        band_state = state_projection[band] / residual_scale[None, None, :]
        band_total = band_mesh + band_state
        cumulative_before = np.concatenate(
            (np.zeros_like(band_total[:1]), np.cumsum(band_total[:-1], axis=0)),
            axis=0,
        )
        cross = band_mesh * band_state
        mesh_growth = 2.0 * cumulative_before * band_mesh + np.square(band_mesh) + cross
        state_growth = (
            2.0 * cumulative_before * band_state + np.square(band_state) + cross
        )
        total_growth = mesh_growth + state_growth
        growth_arrays.extend((total_growth, mesh_growth, state_growth))
        growth_titles.extend(
            (f"{band} total growth", f"{band} mesh growth", f"{band} state growth")
        )
        growth_names.extend((f"{band}_total", f"{band}_mesh", f"{band}_state"))
    paths, rows = _render_pathway_movie(
        growth_arrays,
        growth_titles,
        (GROWTH_LIMIT,) * 6,
        growth_names,
        ("signed_growth",) * 6,
        reference_states=reference,
        nodes=nodes,
        times=times,
        resolution=resolution,
        gamma=gamma,
        domain_lengths=domain_lengths,
        case_id=bundle_path.stem,
        target=target,
        family="signed_growth",
        output_dir=output_dir,
    )
    generated.extend(paths)
    saturation.extend(rows)
    return generated, saturation


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary = _load_and_verify(args.results_dir, args.bundle_dir)
    pathway_bundles: dict[str, Path] = {}
    if args.pathway_bundle_dir is not None:
        pathway_bundles = _verify_pathway_bundles(summary, args.pathway_bundle_dir)
    if args.animation_family in {"pathway", "both"} and not pathway_bundles:
        raise ValueError(
            "pathway animations require the bound pathway bundle directory"
        )
    time_rows = _read_rows(args.results_dir / "scale_time_metrics.csv")
    aggregate = _read_rows(args.results_dir / "scale_aggregate_metrics.csv")
    component_path = args.results_dir / "scale_component_aggregate_metrics.csv"
    component_aggregate = _read_rows(component_path) if component_path.is_file() else []
    sequence_kinds = list(summary["contract"]["sequence_kinds"])
    if args.sequence_kind is None:
        if len(sequence_kinds) != 1:
            raise ValueError("select one sequence kind explicitly")
        sequence_kind = sequence_kinds[0]
    else:
        sequence_kind = args.sequence_kind
        if sequence_kind not in sequence_kinds:
            raise ValueError("selected sequence kind is absent from the bound results")
    time_rows = [row for row in time_rows if row["sequence_kind"] == sequence_kind]
    aggregate = [row for row in aggregate if row["sequence_kind"] == sequence_kind]
    component_aggregate = [
        row for row in component_aggregate if row["sequence_kind"] == sequence_kind
    ]
    if not time_rows or not aggregate:
        raise ValueError("selected sequence kind has no result rows")
    sequence_label = sequence_kind.replace("_", " ")
    args.output_dir.mkdir(parents=True)
    _style()
    generated: list[Path] = []
    if args.command in {"plots", "all"}:
        plot_accumulation_summary(
            aggregate, args.output_dir, sequence_label=sequence_label
        )
        plot_scale_evolution(time_rows, args.output_dir, sequence_label=sequence_label)
        plot_lag_correlations(aggregate, args.output_dir, sequence_label=sequence_label)
        plot_pod_summary(aggregate, args.output_dir, sequence_label=sequence_label)
        if "signed_growth_over_total_path_energy" in time_rows[0]:
            plot_signed_growth_evolution(
                time_rows, args.output_dir, sequence_label=sequence_label
            )
        if component_aggregate:
            plot_component_summary(
                component_aggregate,
                args.output_dir,
                sequence_label=sequence_label,
            )
        pathway_time_path = args.results_dir / "pathway_scale_region_time_metrics.csv"
        pathway_aggregate_path = (
            args.results_dir / "pathway_scale_region_aggregate_metrics.csv"
        )
        shock_profile_path = args.results_dir / "shock_profile_time_metrics.csv"
        if pathway_time_path.is_file() and pathway_aggregate_path.is_file():
            pathway_time = _read_rows(pathway_time_path)
            pathway_aggregate = _read_rows(pathway_aggregate_path)
            plot_pathway_attribution(pathway_aggregate, args.output_dir)
            plot_pathway_growth(pathway_time, pathway_aggregate, args.output_dir)
            plot_region_accumulation(pathway_time, args.output_dir)
        if shock_profile_path.is_file():
            plot_shock_profile(_read_rows(shock_profile_path), args.output_dir)
        generated.extend(sorted(args.output_dir.glob("*.pdf")))
        generated.extend(sorted(args.output_dir.glob("*.png")))
    saturation_rows: list[dict[str, Any]] = []
    if args.command in {"animations", "all"}:
        targets = sorted({row["target"] for row in aggregate})
        animation_dir = args.output_dir / "animations"
        selected_cases = (
            set(args.animation_case_ids)
            if args.animation_case_ids is not None
            else None
        )
        if args.animation_family in {"scale", "both"}:
            for bundle_path in sorted(args.bundle_dir.glob("*.npz")):
                if (
                    selected_cases is not None
                    and bundle_path.stem not in selected_cases
                ):
                    continue
                for target in targets:
                    animation_paths, rows = animate_bundle(
                        bundle_path,
                        target=target,
                        sequence_kind=sequence_kind,
                        output_dir=animation_dir,
                        domain_lengths=tuple(summary["contract"]["domain_lengths"]),
                    )
                    generated.extend(animation_paths)
                    saturation_rows.extend(rows)
        if args.animation_family in {"pathway", "both"}:
            for bundle_path in sorted(pathway_bundles.values()):
                if (
                    selected_cases is not None
                    and bundle_path.stem not in selected_cases
                ):
                    continue
                for target in targets:
                    animation_paths, rows = animate_pathway_bundle(
                        bundle_path,
                        target=target,
                        output_dir=animation_dir,
                        domain_lengths=tuple(summary["contract"]["domain_lengths"]),
                        gamma=float(summary["contract"].get("gamma", 1.4)),
                    )
                    generated.extend(animation_paths)
                    saturation_rows.extend(rows)
        saturation_path = args.output_dir / "visual_scale_saturation.csv"
        _write_rows(saturation_path, saturation_rows)
        generated.append(saturation_path)
    output_hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in sorted(set(generated))
    }
    visual_summary = {
        "schema": SCHEMA,
        "status": "complete",
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "diagnostic_source_hash": sha256_file(Path(__file__).resolve()),
        "boundary_policy": summary["boundary_policy"],
        "population": summary["population"],
        "sequence_kinds": [sequence_kind],
        "contract": {
            "instantaneous_limit_residual_scale": INSTANT_LIMIT,
            "cumulative_limit_residual_scale": CUMULATIVE_LIMIT,
            "signed_growth_limit_squared_residual_scale": GROWTH_LIMIT,
            "fixed_across_frames": True,
            "per_frame_normalization": False,
            "physical_extent": summary["contract"]["domain_lengths"],
            "visualization_only_subsampling": None,
            "clipping_audit": (
                "exact sample counts by case, target, band, component, and temporal kind"
                if saturation_rows
                else None
            ),
            "maximum_clipped_fraction": (
                max(float(row["clipped_fraction"]) for row in saturation_rows)
                if saturation_rows
                else None
            ),
            "fps": FPS,
            "animation_family": args.animation_family,
            "animation_case_ids": (
                sorted(args.animation_case_ids)
                if args.animation_case_ids is not None
                else None
            ),
            "region_overlay": (
                "reference output frame; fixed physical boundary <= 0.05, "
                "shock envelope <= 0.05, vortex core <= 0.18"
                if args.animation_family in {"pathway", "both"}
                else None
            ),
            "transform": summary["contract"]["transform"],
            "selected_sequence_semantics": (
                "free-rollout cross-grid increment defect; cumulative sum equals the "
                "cross-grid state gap under the verified recurrence"
                if sequence_kind == "free_cross_grid_defect"
                else summary["contract"]["teacher_cumulative_semantics"]
            ),
            "teacher_cumulative_semantics": summary["contract"][
                "teacher_cumulative_semantics"
            ],
        },
        "output_hashes": output_hashes,
    }
    (args.output_dir / "visual_summary.json").write_text(
        json.dumps(visual_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return visual_summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
