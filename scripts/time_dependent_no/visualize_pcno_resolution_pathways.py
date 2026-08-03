#!/usr/bin/env python3
"""Render fixed-scale figures for frozen PCNO resolution pathways."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.15,
        "lines.linewidth": 1.8,
    }
)
from matplotlib import animation

BRANCHES = ("spectral", "pointwise", "differential")
COMPONENTS = ("rho", "rho u", "rho v", "energy")
COLORS = {
    "spectral": "#0072B2",
    "pointwise": "#D55E00",
    "differential": "#009E73",
}
DEFAULT_PAIRS = ("125x50->250x100", "250x100->500x200")
ANIMATION_COLUMNS = (
    "true increment",
    "baseline cross-grid defect",
    "spectral gain response",
    "pointwise gain response",
    "differential gain response",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plots", "animations", "all"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _strict_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in {None, "", "None", "null"}:
        return None
    return float(value)


def _quantile_series(
    rows: Sequence[Mapping[str, str]],
    *,
    filters: Mapping[str, str],
    metric: str,
    x_key: str = "call",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        x = _number(row.get(x_key))
        value = _number(row.get(metric))
        if x is not None and value is not None and np.isfinite(value):
            grouped[x].append(value)
    if not grouped:
        raise ValueError(f"no rows for {filters} metric={metric}")
    x_values = np.asarray(sorted(grouped), dtype=np.float64)
    median = np.asarray([np.median(grouped[x]) for x in x_values])
    q25 = np.asarray([np.quantile(grouped[x], 0.25) for x in x_values])
    q75 = np.asarray([np.quantile(grouped[x], 0.75) for x in x_values])
    return x_values, median, q25, q75


def _draw_quantiles(
    axis: Any,
    rows: Sequence[Mapping[str, str]],
    *,
    filters: Mapping[str, str],
    metric: str,
    label: str,
    color: str,
) -> None:
    x, median, q25, q75 = _quantile_series(rows, filters=filters, metric=metric)
    axis.plot(x, median, color=color, linewidth=1.6, label=label)
    axis.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0.0)


def _style(axis: Any, ylabel: str) -> None:
    axis.set_xlabel("recurrent-call index (teacher-forced input)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: Any, stem: Path, *, overwrite: bool) -> None:
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        _strict_output(path, overwrite=overwrite)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_output_metrics(
    results_dir: Path, output_dir: Path, *, pair: str, overwrite: bool
) -> None:
    output_rows = _read_rows(results_dir / "output_metrics.csv")
    gain_rows = _read_rows(results_dir / "family_gain_metrics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    _draw_quantiles(
        axes[0, 0],
        output_rows,
        filters={"pair": pair},
        metric="defect_relative_to_truth_increment",
        label="baseline",
        color="#333333",
    )
    for branch in BRANCHES:
        filters = {"pair": pair, "branch": branch, "layer": "all"}
        _draw_quantiles(
            axes[0, 1],
            gain_rows,
            filters=filters,
            metric="defect_norm_elasticity",
            label=branch,
            color=COLORS[branch],
        )
        _draw_quantiles(
            axes[1, 0],
            gain_rows,
            filters=filters,
            metric="coarse_error_norm_ratio",
            label=branch,
            color=COLORS[branch],
        )
        _draw_quantiles(
            axes[1, 1],
            gain_rows,
            filters=filters,
            metric="restricted_fine_error_norm_ratio",
            label=branch,
            color=COLORS[branch],
        )
    _style(axes[0, 0], "||delta|| / ||true increment||")
    _style(axes[0, 1], "d log ||delta|| / d branch gain")
    _style(axes[1, 0], "coarse truth-error ratio at gain 0.99")
    _style(axes[1, 1], "restricted-fine truth-error ratio at gain 0.99")
    for axis in axes.flat:
        axis.axhline(0.0, color="#777777", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    axes[1, 0].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1, 1].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    figure.suptitle(f"Frozen D060 same-input pathway response: {pair}")
    _save_figure(
        figure,
        output_dir / f"output_pathway_time_{_pair_key(pair)}",
        overwrite=overwrite,
    )


def _matrix(
    rows: Sequence[Mapping[str, str]],
    *,
    filters: Mapping[str, str],
    metric: str,
    calls: Sequence[int],
    layers: Sequence[int],
) -> np.ndarray:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        call = int(row["call"])
        layer = int(row["layer"])
        value = _number(row.get(metric))
        if value is not None:
            grouped[(layer, call)].append(value)
    matrix = np.full((len(layers), len(calls)), np.nan)
    for row_index, layer in enumerate(layers):
        for column_index, call in enumerate(calls):
            values = grouped.get((layer, call), [])
            if values:
                matrix[row_index, column_index] = float(np.median(values))
    if not np.isfinite(matrix).all():
        raise ValueError(f"incomplete heatmap for {filters} metric={metric}")
    return matrix


def plot_latent_metrics(
    results_dir: Path, output_dir: Path, *, pair: str, overwrite: bool
) -> None:
    rows = _read_rows(results_dir / "latent_metrics.csv")
    calls = tuple(range(1, 31))
    layers = tuple(range(4))
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 8.0), constrained_layout=True)
    metrics = (
        ("native_symmetric_relative", "native branch commutator"),
        ("mesh_symmetric_relative", "same-hidden mesh commutator"),
        ("combined_branch_attribution", "attribution to preactivation gap"),
    )
    for row_index, branch in enumerate(BRANCHES):
        for column_index, (metric, title) in enumerate(metrics):
            matrix = _matrix(
                rows,
                filters={"pair": pair, "record_kind": "branch", "branch": branch},
                metric=metric,
                calls=calls,
                layers=layers,
            )
            if "attribution" in metric:
                limit = max(float(np.max(np.abs(matrix))), 1.0)
                cmap = "coolwarm"
                vmin, vmax = -limit, limit
            else:
                cmap = "magma"
                vmin, vmax = 0.0, max(float(np.quantile(matrix, 0.99)), 1.0e-12)
            image = axes[row_index, column_index].imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=(0.5, 30.5, -0.5, 3.5),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row_index, column_index].set_xlabel("call")
            axes[row_index, column_index].set_ylabel("PCNO block")
            axes[row_index, column_index].set_yticks(layers)
            axes[row_index, column_index].set_title(f"{branch}: {title}", fontsize=9)
            figure.colorbar(image, ax=axes[row_index, column_index], shrink=0.75)
    figure.suptitle(
        f"Latent resolution pathway (median over six cases): {pair}\n"
        "latent norms are symmetric within-layer quantities, not physical units",
        fontsize=11,
    )
    _save_figure(
        figure,
        output_dir / f"latent_pathway_heatmaps_{_pair_key(pair)}",
        overwrite=overwrite,
    )


def plot_layer_gain_metrics(
    results_dir: Path, output_dir: Path, *, pair: str, overwrite: bool
) -> None:
    rows = _read_rows(results_dir / "layer_gain_metrics.csv")
    calls = (1, 5, 15, 30)
    layers = tuple(range(4))
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), constrained_layout=True)
    for column, branch in enumerate(BRANCHES):
        elasticity = _matrix(
            rows,
            filters={"pair": pair, "branch": branch},
            metric="defect_norm_elasticity",
            calls=calls,
            layers=layers,
        )
        limit = max(float(np.max(np.abs(elasticity))), 1.0e-12)
        image = axes[0, column].imshow(
            elasticity,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"{branch}: defect elasticity", fontsize=9)
        figure.colorbar(image, ax=axes[0, column], shrink=0.75)
        worst_error = np.maximum(
            _matrix(
                rows,
                filters={"pair": pair, "branch": branch},
                metric="coarse_error_norm_ratio",
                calls=calls,
                layers=layers,
            ),
            _matrix(
                rows,
                filters={"pair": pair, "branch": branch},
                metric="restricted_fine_error_norm_ratio",
                calls=calls,
                layers=layers,
            ),
        )
        error_limit = max(float(np.max(np.abs(worst_error - 1.0))), 1.0e-12)
        image = axes[1, column].imshow(
            worst_error - 1.0,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            interpolation="nearest",
        )
        axes[1, column].set_title(
            f"{branch}: worst truth-error ratio minus 1", fontsize=9
        )
        figure.colorbar(image, ax=axes[1, column], shrink=0.75)
        for axis in axes[:, column]:
            axis.set_xticks(range(len(calls)), labels=calls)
            axis.set_yticks(range(len(layers)), labels=layers)
            axis.set_xlabel("registered call")
            axis.set_ylabel("PCNO block")
    figure.suptitle(f"Controlled 0.99 single-block gain response: {pair}")
    _save_figure(
        figure,
        output_dir / f"layer_gain_heatmaps_{_pair_key(pair)}",
        overwrite=overwrite,
    )


def _sub_005_spectral_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Recover the omitted wavelength-below-0.05 share by exact complement."""

    grouped: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    for row in rows:
        key = (
            row["case_id"],
            row["pair"],
            row["call"],
            row["branch"],
            row["layer"],
        )
        share = _number(row.get("spectral_energy_share"))
        if share is not None:
            grouped[key] += share
    return [
        {
            "case_id": case_id,
            "pair": pair,
            "call": call,
            "branch": branch,
            "layer": layer,
            "sub_005_spectral_energy_share": float(
                np.clip(1.0 - recorded_share, 0.0, 1.0)
            ),
        }
        for (case_id, pair, call, branch, layer), recorded_share in grouped.items()
    ]


def plot_response_structure(
    results_dir: Path, output_dir: Path, *, pair: str, overwrite: bool
) -> None:
    spectral_rows = _sub_005_spectral_rows(
        _read_rows(results_dir / "response_spectral_metrics.csv")
    )
    spatial_rows = _read_rows(results_dir / "response_spatial_metrics.csv")
    latent_rows = _read_rows(results_dir / "latent_metrics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    for branch in BRANCHES:
        branch_filters = {"pair": pair, "branch": branch, "layer": "all"}
        _draw_quantiles(
            axes[0, 0],
            spectral_rows,
            filters=branch_filters,
            metric="sub_005_spectral_energy_share",
            label=branch,
            color=COLORS[branch],
        )
        for axis, region in (
            (axes[0, 1], "shock_envelope_le_0.05"),
            (axes[1, 0], "boundary_le_0.05"),
        ):
            _draw_quantiles(
                axis,
                spatial_rows,
                filters={**branch_filters, "region": region},
                metric="energy_density_enrichment",
                label=branch,
                color=COLORS[branch],
            )
    input_colors = {
        "coordinates": "#56B4E9",
        "normalized_state": "#009E73",
        "node_type": "#CC79A7",
    }
    for input_group, color in input_colors.items():
        _draw_quantiles(
            axes[1, 1],
            latent_rows,
            filters={
                "pair": pair,
                "record_kind": "input_group",
                "input_group": input_group,
            },
            metric="symmetric_attribution",
            label=input_group.replace("_", " "),
            color=color,
        )
    _style(axes[0, 0], "response energy at wavelength < 0.05")
    _style(axes[0, 1], "shock-envelope energy-density enrichment")
    _style(axes[1, 0], "boundary-band energy-density enrichment")
    _style(axes[1, 1], "input-lift symmetric attribution")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylim(-0.1, 1.1)
    for axis in axes.flat:
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Physical structure of the pathway response: {pair}")
    _save_figure(
        figure,
        output_dir / f"response_structure_time_{_pair_key(pair)}",
        overwrite=overwrite,
    )


def plot_all(
    results_dir: Path, output_dir: Path, *, pairs: Sequence[str], overwrite: bool
) -> None:
    for pair in pairs:
        plot_output_metrics(results_dir, output_dir, pair=pair, overwrite=overwrite)
        plot_latent_metrics(results_dir, output_dir, pair=pair, overwrite=overwrite)
        plot_layer_gain_metrics(results_dir, output_dir, pair=pair, overwrite=overwrite)
        plot_response_structure(results_dir, output_dir, pair=pair, overwrite=overwrite)


def _pair_key(pair: str) -> str:
    return pair.replace("->", "_to_")


def _coarse_resolution(pair: str) -> tuple[int, int]:
    coarse = pair.split("->", maxsplit=1)[0]
    nx, ny = coarse.split("x", maxsplit=1)
    return int(nx), int(ny)


def animate_bundle(
    bundle_path: Path,
    output_path: Path,
    *,
    pair: str,
    limit: float,
    fps: int,
    dpi: int,
    frame_stride: int,
    overwrite: bool,
) -> None:
    _strict_output(output_path, overwrite=overwrite)
    with np.load(bundle_path, allow_pickle=False) as artifact:
        bundle = {key: np.array(artifact[key], copy=True) for key in artifact.files}
    key = _pair_key(pair)
    residual_scale = np.asarray(bundle["residual_scale"], dtype=np.float64)
    times = np.asarray(bundle["physical_times"], dtype=np.float64)
    arrays = (
        bundle[f"true_increment__{key}"],
        bundle[f"baseline_defect__{key}"],
        bundle[f"spectral_response__{key}"],
        bundle[f"pointwise_response__{key}"],
        bundle[f"differential_response__{key}"],
    )
    nx, ny = _coarse_resolution(pair)
    scaled = [
        np.asarray(value, dtype=np.float64).reshape(len(times), ny, nx, 4)
        / residual_scale[None, None, None, :]
        for value in arrays
    ]
    figure, axes = plt.subplots(4, 5, figsize=(14.0, 8.0), constrained_layout=True)
    images: list[list[Any]] = []
    for component in range(4):
        row_images = []
        for column in range(5):
            image = axes[component, column].imshow(
                scaled[column][0, :, :, component],
                origin="lower",
                extent=(0.0, 2.0, 0.0, 1.0),
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="auto",
            )
            axes[component, column].set_xticks([])
            axes[component, column].set_yticks([])
            if component == 0:
                axes[component, column].set_title(ANIMATION_COLUMNS[column], fontsize=8)
            if column == 0:
                axes[component, column].set_ylabel(COMPONENTS[component])
            row_images.append(image)
        images.append(row_images)
    title = figure.text(0.5, 0.995, "", ha="center", va="top", fontsize=10)
    scale_text = figure.text(
        0.995,
        0.5,
        f"all panels: +/-{limit:g} checkpoint residual-scale units\n"
        "gain response = (delta_0.99 - delta_1.00) / -0.01",
        rotation=90,
        va="center",
        ha="right",
        fontsize=7,
    )
    frames = list(range(0, len(times), frame_stride))
    if frames[-1] != len(times) - 1:
        frames.append(len(times) - 1)

    def update(index: int) -> list[Any]:
        for component in range(4):
            for column in range(5):
                images[component][column].set_data(
                    scaled[column][index, :, :, component]
                )
        title.set_text(f"{bundle_path.stem}  {pair}  exact-input t={times[index]:.2f}")
        return [item for row in images for item in row] + [title, scale_text]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    final_path = output_path.with_suffix(".png")
    _strict_output(final_path, overwrite=overwrite)
    update(frames[-1])
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_animations(args: argparse.Namespace) -> None:
    contract = json.loads(
        (args.results_dir / "visual_contract.json").read_text(encoding="utf-8")
    )
    limit = float(contract["common_symmetric_limit"])
    bundle_paths = sorted((args.results_dir / "bundles").glob("*.npz"))
    if args.case_ids is not None:
        selected = set(args.case_ids)
        bundle_paths = [path for path in bundle_paths if path.stem in selected]
    if not bundle_paths:
        raise ValueError("no pathway bundles selected")
    for bundle_path in bundle_paths:
        for pair in args.pairs:
            output = (
                args.output_dir
                / "animations"
                / f"{bundle_path.stem}_{_pair_key(pair)}_pathways.gif"
            )
            animate_bundle(
                bundle_path,
                output,
                pair=pair,
                limit=limit,
                fps=args.fps,
                dpi=args.dpi,
                frame_stride=args.frame_stride,
                overwrite=args.overwrite,
            )
            print(f"wrote {output}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command in {"plots", "all"}:
        plot_all(
            args.results_dir,
            args.output_dir,
            pairs=args.pairs,
            overwrite=args.overwrite,
        )
    if args.command in {"animations", "all"}:
        render_animations(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
