#!/usr/bin/env python3
"""Plot and animate residual-scale PCNO resolution diagnostics."""

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
from matplotlib import animation

COMPONENTS = ("rho", "rho u", "rho v", "E")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
ANIMATION_COLUMNS = (
    "true residual",
    "coarse prediction",
    "restricted fine",
    "coarse error",
    "fine error",
    "cross-grid defect",
    "cumulative defect",
    "signed growth",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("plots", "animations", "all"), default="all"
    )
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument(
        "--modes", nargs="+", choices=("free", "teacher"), default=("free",)
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--error-limit", type=float, default=5.0)
    parser.add_argument("--cumulative-limit", type=float, default=10.0)
    parser.add_argument("--growth-limit", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _strict_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _series(
    rows: Sequence[Mapping[str, str]],
    *,
    filters: Mapping[str, str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        time = _number(row.get("physical_time"))
        metric_value = _number(row.get(metric))
        if time is not None and metric_value is not None:
            grouped[time].append(metric_value)
    if not grouped:
        raise ValueError(f"no rows for {filters} / {metric}")
    times = np.asarray(sorted(grouped), dtype=np.float64)
    mean = np.asarray([np.mean(grouped[value]) for value in times])
    median = np.asarray([np.median(grouped[value]) for value in times])
    lower = np.asarray([np.quantile(grouped[value], 0.25) for value in times])
    upper = np.asarray([np.quantile(grouped[value], 0.75) for value in times])
    return times, mean, median, lower, upper


def _draw_series(
    axis: Any,
    rows: Sequence[Mapping[str, str]],
    *,
    filters: Mapping[str, str],
    metric: str,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    times, mean, median, lower, upper = _series(rows, filters=filters, metric=metric)
    axis.fill_between(times, lower, upper, color=color, alpha=0.14, linewidth=0)
    axis.plot(
        times, median, color=color, linestyle=linestyle, linewidth=1.8, label=label
    )
    axis.plot(times, mean, color=color, linestyle=linestyle, linewidth=0.7, alpha=0.55)


def _finish_figure(
    figure: Any,
    output_base: Path,
    *,
    overwrite: bool,
) -> None:
    for suffix in (".pdf", ".png"):
        path = output_base.with_suffix(suffix)
        _strict_output(path, overwrite=overwrite)
        figure.savefig(path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)


def _style_axis(axis: Any, ylabel: str) -> None:
    axis.set_xlabel("physical time")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)


def plot_time_metrics(results_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    rows = _read_rows(results_dir / "time_metrics.csv")
    accumulated = _read_rows(results_dir / "accumulated_change_metrics.csv")
    resolutions = ("125x50", "250x100", "500x200")
    pairs = ("125x50->250x100", "250x100->500x200")

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), constrained_layout=True)
    for index, resolution in enumerate(resolutions):
        _draw_series(
            axes[0, 0],
            rows,
            filters={
                "sequence_kind": "per_grid_free_residual_error",
                "target": resolution,
            },
            metric="instant_defect_rms",
            label=resolution,
            color=COLORS[index],
        )
        _draw_series(
            axes[0, 1],
            rows,
            filters={
                "sequence_kind": "per_grid_teacher_residual_error",
                "target": resolution,
            },
            metric="instant_defect_rms",
            label=resolution,
            color=COLORS[index],
        )
        _draw_series(
            axes[0, 2],
            rows,
            filters={
                "sequence_kind": "per_grid_free_residual_error",
                "target": resolution,
            },
            metric="instant_truth_rms",
            label=resolution,
            color=COLORS[index],
        )
    for index, pair in enumerate(pairs):
        _draw_series(
            axes[1, 0],
            rows,
            filters={"sequence_kind": "free_cross_grid_defect", "target": pair},
            metric="instant_defect_rms",
            label=f"free {pair}",
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 0],
            rows,
            filters={"sequence_kind": "teacher_cross_grid_defect", "target": pair},
            metric="instant_defect_rms",
            label=f"teacher {pair}",
            color=COLORS[index],
            linestyle="--",
        )
        _draw_series(
            axes[1, 2],
            rows,
            filters={"sequence_kind": "free_mesh_defect", "target": pair},
            metric="instant_defect_rms",
            label=f"mesh {pair}",
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 2],
            rows,
            filters={"sequence_kind": "free_state_defect", "target": pair},
            metric="instant_defect_rms",
            label=f"state {pair}",
            color=COLORS[index],
            linestyle="--",
        )
        _draw_series(
            axes[1, 1],
            rows,
            filters={"sequence_kind": "free_cross_grid_defect", "target": pair},
            metric="instant_relative",
            label=f"free {pair}",
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 1],
            rows,
            filters={"sequence_kind": "teacher_cross_grid_defect", "target": pair},
            metric="instant_relative",
            label=f"teacher {pair}",
            color=COLORS[index],
            linestyle="--",
        )
    for axis, ylabel in zip(
        axes.flat,
        (
            "free residual error [residual scale]",
            "teacher residual error [residual scale]",
            "true residual magnitude [residual scale]",
            "cross-grid defect [residual scale]",
            "cross-grid defect / true residual",
            "mesh/state defect [residual scale]",
        ),
    ):
        _style_axis(axis, ylabel)
        axis.legend(frameon=False, fontsize=7)
    _finish_figure(figure, output_dir / "residual_magnitude_time", overwrite=overwrite)

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), constrained_layout=True)
    for index, pair in enumerate(pairs):
        filters = {"sequence_kind": "free_cross_grid_defect", "target": pair}
        _draw_series(
            axes[0, 0],
            rows,
            filters=filters,
            metric="cumulative_defect_rms",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[0, 1],
            accumulated,
            filters={"trajectory_kind": "free_prediction", "target": pair},
            metric="relative_to_reference_change",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[0, 2],
            accumulated,
            filters={"trajectory_kind": "reference", "target": pair},
            metric="relative_to_reference_change",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 0],
            rows,
            filters=filters,
            metric="temporal_coherence",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 1],
            rows,
            filters=filters,
            metric="cumulative_parallel_bias",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 2],
            rows,
            filters=filters,
            metric="cumulative_defect_truth_cosine",
            label=pair,
            color=COLORS[index],
        )
    for axis, ylabel in zip(
        axes.flat,
        (
            "cumulative defect [residual scale]",
            "predicted accumulated gap / reference change",
            "reference restriction floor / reference change",
            "temporal coherence kappa",
            "cumulative parallel-bias coefficient",
            "cos(net defect, cumulative truth change)",
        ),
    ):
        _style_axis(axis, ylabel)
        axis.axhline(0.0, color="#666666", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    axes[0, 2].set_yscale("symlog", linthresh=1.0e-16)
    _finish_figure(
        figure, output_dir / "residual_accumulation_time", overwrite=overwrite
    )


def plot_structure_metrics(
    results_dir: Path, output_dir: Path, *, overwrite: bool
) -> None:
    spatial = _read_rows(results_dir / "spatial_metrics.csv")
    spectral = _read_rows(results_dir / "spectral_metrics.csv")
    pairs = ("125x50->250x100", "250x100->500x200")
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), constrained_layout=True)
    for index, pair in enumerate(pairs):
        base = {"sequence_kind": "free_cross_grid_defect", "target": pair}
        _draw_series(
            axes[0, 0],
            spatial,
            filters={
                **base,
                "diagnostic": "physical_region",
                "region": "shock_envelope_le_0.05",
            },
            metric="energy_share",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[0, 1],
            spatial,
            filters={
                **base,
                "diagnostic": "physical_region",
                "region": "boundary_le_0.05",
            },
            metric="energy_density_enrichment",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 0],
            spatial,
            filters={
                **base,
                "diagnostic": "shock_phase_projection",
                "region": "shock_envelope_le_0.05",
            },
            metric="phase_energy_fraction",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1, 1],
            spectral,
            filters={**base, "wavelength_min": "0.05", "wavelength_max": "0.125"},
            metric="spectral_energy_share",
            label=pair,
            color=COLORS[index],
        )
    for axis, ylabel in zip(
        axes.flat,
        (
            "shock-band defect energy share",
            "boundary-band energy-density enrichment",
            "shock-translation-mode energy share",
            "energy share at wavelength 0.05-0.125",
        ),
    ):
        _style_axis(axis, ylabel)
        axis.legend(frameon=False, fontsize=8)
    _finish_figure(figure, output_dir / "residual_structure_time", overwrite=overwrite)


def plot_growth_metrics(
    results_dir: Path, output_dir: Path, *, overwrite: bool
) -> None:
    rows = _read_rows(results_dir / "growth_metrics.csv")
    pairs = ("125x50->250x100", "250x100->500x200")
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.3), constrained_layout=True)
    for index, pair in enumerate(pairs):
        filters = {"sequence_kind": "free_cross_grid_defect", "target": pair}
        _draw_series(
            axes[0],
            rows,
            filters=filters,
            metric="signed_interaction",
            label=pair,
            color=COLORS[index],
        )
        _draw_series(
            axes[1],
            rows,
            filters=filters,
            metric="squared_error_growth",
            label=pair,
            color=COLORS[index],
        )
    _style_axis(axes[0], "2 <existing gap, new defect>")
    _style_axis(axes[1], "change in squared cross-grid gap")
    for axis in axes:
        axis.axhline(0.0, color="#666666", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    _finish_figure(figure, output_dir / "signed_growth_time", overwrite=overwrite)


def _pair_key(pair: str) -> str:
    return pair.replace("->", "_to_")


def _resolution(pair: str) -> tuple[int, int]:
    coarse = pair.split("->", maxsplit=1)[0]
    nx, ny = coarse.lower().split("x", maxsplit=1)
    return int(nx), int(ny)


def _animation_arrays(
    bundle: Mapping[str, np.ndarray], pair: str, mode: str
) -> list[np.ndarray]:
    key = _pair_key(pair)
    true = np.asarray(bundle[f"true_increment__{key}"], dtype=np.float64)
    coarse = np.asarray(bundle[f"coarse_increment_{mode}__{key}"], dtype=np.float64)
    fine = np.asarray(bundle[f"fine_increment_{mode}__{key}"], dtype=np.float64)
    delta = np.asarray(bundle[f"delta_{mode}__{key}"], dtype=np.float64)
    cumulative = np.asarray(bundle[f"cumulative_delta_{mode}__{key}"], dtype=np.float64)
    coarse_error_name = f"coarse_error_{mode}__{key}"
    fine_error_name = f"fine_error_{mode}__{key}"
    growth_name = f"growth_{mode}__{key}"
    coarse_error = (
        np.asarray(bundle[coarse_error_name], dtype=np.float64)
        if coarse_error_name in bundle
        else coarse - true
    )
    fine_error = (
        np.asarray(bundle[fine_error_name], dtype=np.float64)
        if fine_error_name in bundle
        else fine - true
    )
    if growth_name in bundle:
        growth = np.asarray(bundle[growth_name], dtype=np.float64)
    else:
        cumulative_before = cumulative - delta
        growth = 2.0 * cumulative_before * delta + np.square(delta)
    return [true, coarse, fine, coarse_error, fine_error, delta, cumulative, growth]


def animate_bundle(
    bundle_path: Path,
    visual_scales: Mapping[str, Any],
    output_path: Path,
    *,
    pair: str,
    mode: str,
    fps: int,
    dpi: int,
    frame_stride: int,
    error_limit: float,
    cumulative_limit: float,
    growth_limit: float,
    overwrite: bool,
) -> None:
    _strict_output(output_path, overwrite=overwrite)
    with np.load(bundle_path, allow_pickle=False) as artifact:
        bundle = {name: np.array(artifact[name], copy=True) for name in artifact.files}
    residual_scale = np.asarray(bundle["residual_scale"], dtype=np.float64)
    times = np.asarray(bundle["physical_times"], dtype=np.float64)
    arrays = _animation_arrays(bundle, pair, mode)
    nx, ny = _resolution(pair)
    scaled = []
    for column, value in enumerate(arrays):
        divisor = residual_scale if column < 7 else np.square(residual_scale)
        scaled.append(
            value.reshape(value.shape[0], ny, nx, 4) / divisor[None, None, None, :]
        )
    reference_limits = np.asarray(
        visual_scales[pair]["instantaneous"], dtype=np.float64
    )
    limits = (
        reference_limits,
        reference_limits,
        reference_limits,
        np.full(4, error_limit),
        np.full(4, error_limit),
        np.full(4, error_limit),
        np.full(4, cumulative_limit),
        np.full(4, growth_limit),
    )
    figure, axes = plt.subplots(4, 8, figsize=(20.0, 9.0), constrained_layout=True)
    images = []
    for component in range(4):
        row_images = []
        for column in range(8):
            limit = max(float(limits[column][component]), 1.0e-12)
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
                axes[component, column].set_ylabel(COMPONENTS[component], fontsize=9)
            row_images.append(image)
        images.append(row_images)
        figure.text(
            0.995,
            0.88 - component * 0.235,
            f"res +/-{reference_limits[component]:.2g}; err +/-{error_limit:g}; "
            f"cum +/-{cumulative_limit:g}; growth +/-{growth_limit:g}",
            rotation=90,
            va="center",
            ha="right",
            fontsize=6,
        )
    time_text = figure.text(0.5, 0.995, "", ha="center", va="top", fontsize=10)
    frame_indices = list(range(0, len(times), frame_stride))
    if frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)

    def update(frame_index: int) -> list[Any]:
        for component in range(4):
            for column in range(8):
                images[component][column].set_data(
                    scaled[column][frame_index, :, :, component]
                )
        time_text.set_text(
            f"{bundle_path.stem}  {pair}  {mode}  t={times[frame_index]:.2f}"
        )
        return [image for row in images for image in row] + [time_text]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    final_path = output_path.with_suffix(".png")
    _strict_output(final_path, overwrite=overwrite)
    update(frame_indices[-1])
    figure.savefig(final_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_animations(args: argparse.Namespace) -> None:
    scales = json.loads(
        (args.results_dir / "visual_scales.json").read_text(encoding="utf-8")
    )
    bundle_paths = sorted((args.results_dir / "bundles").glob("*.npz"))
    if args.case_ids is not None:
        selected = set(args.case_ids)
        bundle_paths = [path for path in bundle_paths if path.stem in selected]
    if not bundle_paths:
        raise ValueError("no diagnostic bundles selected")
    pairs = list(scales) if args.pairs is None else args.pairs
    for bundle_path in bundle_paths:
        for pair in pairs:
            if pair not in scales:
                raise ValueError(f"unknown resolution pair: {pair}")
            for mode in args.modes:
                output = (
                    args.output_dir
                    / "animations"
                    / f"{bundle_path.stem}_{_pair_key(pair)}_{mode}.gif"
                )
                animate_bundle(
                    bundle_path,
                    scales,
                    output,
                    pair=pair,
                    mode=mode,
                    fps=args.fps,
                    dpi=args.dpi,
                    frame_stride=args.frame_stride,
                    error_limit=args.error_limit,
                    cumulative_limit=args.cumulative_limit,
                    growth_limit=args.growth_limit,
                    overwrite=args.overwrite,
                )
                print(f"wrote {output}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command in {"plots", "all"}:
        plot_time_metrics(args.results_dir, args.output_dir, overwrite=args.overwrite)
        plot_structure_metrics(
            args.results_dir, args.output_dir, overwrite=args.overwrite
        )
        plot_growth_metrics(args.results_dir, args.output_dir, overwrite=args.overwrite)
    if args.command in {"animations", "all"}:
        render_animations(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
