#!/usr/bin/env python3
"""Render fixed-scale D071 correction figures and animations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    sha256_file,
    write_csv_with_paths,
)

SCHEMA = "pcno_defect_correction_visualization_v1"
RESULT_SCHEMA = "pcno_defect_correction_diagnostic_v1"
ARMS = ("baseline", "persistent_rank8", "local_dissipation", "combined")
COLORS = {
    "baseline": "#333333",
    "persistent_rank8": "#0072B2",
    "local_dissipation": "#D55E00",
    "combined": "#009E73",
}
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
INSTANT_LIMIT = 1.0
CUMULATIVE_LIMIT = 10.0
FPS = 4
GIF_DPI = 70
PANEL_FIELDS = (
    ("true_increment", "true increment", INSTANT_LIMIT, "RdBu_r"),
    ("baseline_increment", "baseline increment", INSTANT_LIMIT, "RdBu_r"),
    ("combined_increment", "corrected increment", INSTANT_LIMIT, "RdBu_r"),
    (
        "baseline_residual_error",
        "baseline residual error",
        INSTANT_LIMIT,
        "RdBu_r",
    ),
    (
        "combined_residual_error",
        "corrected residual error",
        INSTANT_LIMIT,
        "RdBu_r",
    ),
    (
        "persistent_correction",
        "persistent correction",
        INSTANT_LIMIT,
        "RdBu_r",
    ),
    ("local_correction", "local correction", INSTANT_LIMIT, "RdBu_r"),
    (
        "accumulated_correction",
        "accumulated direct correction",
        CUMULATIVE_LIMIT,
        "RdBu_r",
    ),
    (
        "accumulated_baseline_residual_error",
        "cumulative baseline residual error",
        CUMULATIVE_LIMIT,
        "RdBu_r",
    ),
    (
        "accumulated_combined_residual_error",
        "cumulative corrected residual error",
        CUMULATIVE_LIMIT,
        "RdBu_r",
    ),
    ("sensor", "local sensor", 1.0, "magma"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("plots", "animations", "all"), default="all"
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in {None, "", "None", "null"}:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _verify_results(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected D071 result schema")
    if summary.get("status") != "complete" or not summary.get(
        "scientific_interpretation_allowed"
    ):
        raise ValueError("D071 results are not complete and interpretable")
    required = {
        "metrics.csv",
        "component_metrics.csv",
        "band_metrics.csv",
        "front_metrics.csv",
        "correction_metrics.csv",
        "completion.csv",
        "teacher_completion.csv",
        "calibration.csv",
        "calibration_coefficients.npz",
    }
    if summary.get("family") == "dynamic_fv":
        required.add("reference_checks.csv")
    output_hashes = summary.get("output_hashes", {})
    for relative in required:
        if relative not in output_hashes:
            raise ValueError(f"D071 output hash is missing: {relative}")
    root = results_dir.resolve()
    for relative, expected in output_hashes.items():
        path = (results_dir / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"D071 output path escapes results directory: {relative}"
            ) from error
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"D071 output digest mismatch: {relative}")
    listed_payloads = {
        relative
        for relative in output_hashes
        if relative.startswith("visual_payloads/")
    }
    disk_payloads = {
        str(path.relative_to(results_dir)).replace("\\", "/")
        for path in (results_dir / "visual_payloads").glob("*.npz")
    }
    if listed_payloads != disk_payloads:
        raise ValueError(
            "D071 visual payload inventory differs from the hash-listed inventory"
        )
    return summary


def _series(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    **filters: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if any(str(row.get(name)) != str(value) for name, value in filters.items()):
            continue
        time = _number(row.get("physical_time"))
        result = _number(row.get(metric))
        if time is not None and result is not None:
            grouped[time].append(result)
    times = np.asarray(sorted(grouped), dtype=np.float64)
    median = np.asarray([np.median(grouped[time]) for time in times])
    lower = np.asarray([np.quantile(grouped[time], 0.25) for time in times])
    upper = np.asarray([np.quantile(grouped[time], 0.75) for time in times])
    return times, median, lower, upper


def _shared_upper_limit(
    rows: Sequence[Mapping[str, str]], metric: str, **filters: str
) -> float:
    values = []
    for row in rows:
        if any(str(row.get(name)) != str(value) for name, value in filters.items()):
            continue
        value = _number(row.get(metric))
        if value is not None:
            values.append(value)
    if not values:
        return 1.0
    maximum = max(values)
    return 1.0 if maximum <= 0.0 else 1.05 * maximum


def _draw_series(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, str]],
    metric: str,
    *,
    arm: str,
    **filters: str,
) -> None:
    times, median, lower, upper = _series(rows, metric, arm=arm, **filters)
    if times.size == 0:
        return
    axis.plot(times, median, color=COLORS[arm], linewidth=1.7, label=arm)
    axis.fill_between(times, lower, upper, color=COLORS[arm], alpha=0.14, linewidth=0)


def _finish_axis(axis: plt.Axes, *, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("physical time")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_global_metrics(
    rows: Sequence[Mapping[str, str]], output_dir: Path, resolution: str
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    specifications = (
        ("free_rollout", "all", "state_error_rms", "free state error"),
        ("free_rollout", "all", "update_error_rms", "free residual error"),
        (
            "teacher_forced",
            "all",
            "update_error_rms",
            "teacher-forced residual error",
        ),
        (
            "free_rollout",
            "smooth",
            "state_error_graph_highpass_rms",
            "smooth-region state high-pass",
        ),
    )
    for axis, (mode, region, metric, title) in zip(
        axes.flat, specifications, strict=True
    ):
        for arm in ARMS:
            _draw_series(
                axis,
                rows,
                metric,
                arm=arm,
                resolution=resolution,
                mode=mode,
                region=region,
            )
        _finish_axis(axis, title=title, ylabel="scaled RMS")
        axis.set_ylim(
            0.0,
            _shared_upper_limit(rows, metric, mode=mode, region=region),
        )
    axes[0, 0].legend(ncols=2, fontsize=8)
    figure.suptitle(f"D071 correction trajectories: {resolution}")
    return _save_figure(figure, output_dir / f"temporal_global_{resolution}")


def plot_component_metrics(
    rows: Sequence[Mapping[str, str]], output_dir: Path, resolution: str
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for axis, component in zip(axes.flat, COMPONENTS, strict=True):
        for arm in ARMS:
            _draw_series(
                axis,
                rows,
                "update_error_rms",
                arm=arm,
                resolution=resolution,
                mode="free_rollout",
                component=component,
            )
        _finish_axis(axis, title=component, ylabel="residual-scale RMS")
        axis.set_ylim(
            0.0,
            _shared_upper_limit(
                rows,
                "update_error_rms",
                mode="free_rollout",
                component=component,
            ),
        )
    axes[0, 0].legend(ncols=2, fontsize=8)
    figure.suptitle(f"D071 free-rollout component residual errors: {resolution}")
    return _save_figure(figure, output_dir / f"temporal_components_{resolution}")


def plot_region_update_metrics(
    rows: Sequence[Mapping[str, str]], output_dir: Path, resolution: str
) -> list[Path]:
    regions = ["shock", "smooth", "boundary_nodes"]
    if any(row.get("region") == "vortex" for row in rows):
        regions.insert(1, "vortex")
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for axis, region in zip(axes.flat, regions):
        for arm in ARMS:
            _draw_series(
                axis,
                rows,
                "update_error_rms",
                arm=arm,
                resolution=resolution,
                mode="free_rollout",
                region=region,
            )
        _finish_axis(axis, title=region, ylabel="residual-scale RMS")
        axis.set_ylim(
            0.0,
            _shared_upper_limit(
                rows, "update_error_rms", mode="free_rollout", region=region
            ),
        )
    for axis in axes.flat[len(regions) :]:
        axis.set_visible(False)
    axes[0, 0].legend(ncols=2, fontsize=8)
    figure.suptitle(f"D071 free-rollout regional residual errors: {resolution}")
    return _save_figure(figure, output_dir / f"temporal_regions_{resolution}")


def plot_band_metrics(
    rows: Sequence[Mapping[str, str]], output_dir: Path, resolution: str
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for axis, band in zip(axes, ("large", "transition", "local"), strict=True):
        for arm in ARMS:
            _draw_series(
                axis,
                rows,
                "rms_residual_scale",
                arm=arm,
                resolution=resolution,
                mode="free_rollout",
                field="update_error",
                band=band,
            )
        _finish_axis(axis, title=band, ylabel="residual-scale RMS")
        axis.set_ylim(
            0.0,
            _shared_upper_limit(
                rows,
                "rms_residual_scale",
                mode="free_rollout",
                field="update_error",
                band=band,
            ),
        )
    axes[0].legend(ncols=2, fontsize=8)
    figure.suptitle(f"D071 free-rollout residual-error bands: {resolution}")
    return _save_figure(figure, output_dir / f"temporal_bands_{resolution}")


def plot_front_and_correction(
    metric_rows: Sequence[Mapping[str, str]],
    front_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    resolution: str,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for arm in ARMS:
        _draw_series(
            axes[0, 0],
            metric_rows,
            "correction_rms",
            arm=arm,
            resolution=resolution,
            mode="free_rollout",
            region="all",
        )
    _finish_axis(axes[0, 0], title="applied correction", ylabel="residual-scale RMS")
    axes[0, 0].set_ylim(
        0.0,
        _shared_upper_limit(
            metric_rows,
            "correction_rms",
            mode="free_rollout",
            region="all",
        ),
    )
    front_specs = (
        ("proxy_centroid_absolute_error", "reference-mask proxy centroid"),
        ("proxy_strength_absolute_error", "reference-mask proxy strength"),
        ("proxy_thickness_absolute_error", "reference-mask proxy thickness"),
    )
    for axis, (metric, title) in zip(axes.flat[1:], front_specs, strict=True):
        for arm in ("baseline", "combined"):
            _draw_series(
                axis,
                front_rows,
                metric,
                arm=arm,
                resolution=resolution,
                mode="free_rollout",
            )
        _finish_axis(axis, title=title, ylabel="absolute proxy error")
        axis.set_ylim(0.0, _shared_upper_limit(front_rows, metric, mode="free_rollout"))
    axes[0, 0].legend(ncols=2, fontsize=8)
    axes[0, 1].legend(fontsize=8)
    figure.suptitle(f"D071 correction and descriptive shock proxies: {resolution}")
    return _save_figure(figure, output_dir / f"temporal_controls_{resolution}")


def _structured_shape(resolution: str, node_count: int) -> tuple[int, int] | None:
    try:
        nx, ny = (int(value) for value in resolution.lower().split("x"))
    except (TypeError, ValueError):
        return None
    return (nx, ny) if nx * ny == node_count else None


def _component_field(
    payload: Mapping[str, np.ndarray], name: str, component: int
) -> np.ndarray:
    values = np.asarray(payload[name], dtype=np.float64)
    if name == "sensor":
        return values
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    return values[..., component] / scale[component]


def animate_component(
    payload_path: Path,
    output_dir: Path,
    *,
    component: int,
) -> tuple[Path, list[dict[str, Any]]]:
    with np.load(payload_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    if str(payload["schema"]) != RESULT_SCHEMA:
        raise ValueError("unexpected D071 visual payload schema")
    if str(payload.get("mode", np.asarray(""))) != "free_rollout":
        raise ValueError("D071 correction animations require free-rollout payloads")
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    resolution = str(payload["resolution"])
    family = str(payload["family"])
    case_id = str(payload["case_id"])
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    shape = _structured_shape(resolution, nodes.shape[0])
    fields = {
        name: _component_field(payload, name, component)
        for name, _, _, _ in PANEL_FIELDS
    }
    frame_count = len(times)
    if any(value.shape[0] != frame_count for value in fields.values()):
        raise ValueError("D071 visual fields do not share one time axis")

    figure, axes = plt.subplots(3, 4, figsize=(16.2, 8.8), constrained_layout=True)
    artists: list[tuple[Any, tuple[int, int] | None, str]] = []
    saturation_rows = []
    for axis, (name, title, limit, cmap) in zip(axes.flat, PANEL_FIELDS):
        values = fields[name]
        first = values[0]
        vmin = 0.0 if name == "sensor" else -limit
        vmax = limit
        if shape is None:
            artist = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=first,
                s=3.0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.0,
            )
        else:
            nx, ny = shape
            artist = axis.imshow(
                first.reshape(ny, nx),
                origin="lower",
                extent=(
                    float(nodes[:, 0].min()),
                    float(nodes[:, 0].max()),
                    float(nodes[:, 1].min()),
                    float(nodes[:, 1].max()),
                ),
                interpolation="nearest",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        axis.set_title(title, fontsize=9)
        axis.set_aspect("equal")
        figure.colorbar(artist, ax=axis, shrink=0.72)
        artists.append((artist, shape, name))
        absolute = np.abs(values)
        saturation_rows.append(
            {
                "family": family,
                "case_id": case_id,
                "resolution": resolution,
                "component": COMPONENTS[component],
                "field": name,
                "fixed_limit": limit,
                "maximum_absolute_scaled_value": float(absolute.max()),
                "fraction_outside_limit": (
                    float(np.mean((values < 0.0) | (values > 1.0)))
                    if name == "sensor"
                    else float(np.mean(absolute > limit))
                ),
            }
        )
    for axis in axes.flat[len(PANEL_FIELDS) :]:
        axis.set_visible(False)

    title = figure.suptitle("")
    expected_calls = int(np.asarray(payload.get("expected_rollout_calls", frame_count)))
    rollout_complete = bool(np.asarray(payload.get("rollout_complete", False)))
    baseline_accepted = np.asarray(
        payload.get("baseline_accepted", np.ones(frame_count, dtype=bool)), dtype=bool
    )
    combined_accepted = np.asarray(
        payload.get("combined_accepted", np.ones(frame_count, dtype=bool)), dtype=bool
    )
    truncated = not rollout_complete or frame_count != expected_calls
    figure.text(
        0.5,
        0.005,
        "fixed residual scales: instantaneous/correction +/-1, cumulative +/-10; no per-frame normalization",
        ha="center",
        fontsize=8,
    )

    def update(frame: int) -> list[Any]:
        for artist, structured, name in artists:
            values = fields[name][frame]
            if structured is None:
                artist.set_array(values)
            else:
                nx, ny = structured
                artist.set_data(values.reshape(ny, nx))
        frame_failed = not baseline_accepted[frame] or not combined_accepted[frame]
        status = (
            " | INADMISSIBLE/TRUNCATED"
            if frame_failed or truncated
            else " | complete rollout"
        )
        title.set_text(
            f"{family} {case_id} {resolution} {COMPONENTS[component]} "
            f"t={times[frame]:.6g}{status}"
        )
        return [artist for artist, _, _ in artists] + [title]

    movie = animation.FuncAnimation(
        figure, update, frames=frame_count, interval=1000 / FPS, blit=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{family}_{case_id}_{resolution}_{COMPONENTS[component]}.gif"
    )
    movie.save(output_path, writer=animation.PillowWriter(fps=FPS), dpi=GIF_DPI)
    plt.close(figure)
    return output_path, saturation_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _verify_results(args.results_dir)
    args.output_dir.mkdir(parents=True)
    generated: list[Path] = []
    if args.command in {"plots", "all"}:
        metric_rows = _read_rows(args.results_dir / "metrics.csv")
        component_rows = _read_rows(args.results_dir / "component_metrics.csv")
        band_rows = _read_rows(args.results_dir / "band_metrics.csv")
        front_rows = _read_rows(args.results_dir / "front_metrics.csv")
        resolutions = sorted({row["resolution"] for row in metric_rows})
        for resolution in resolutions:
            generated.extend(
                plot_global_metrics(metric_rows, args.output_dir, resolution)
            )
            generated.extend(
                plot_component_metrics(component_rows, args.output_dir, resolution)
            )
            generated.extend(
                plot_region_update_metrics(metric_rows, args.output_dir, resolution)
            )
            generated.extend(plot_band_metrics(band_rows, args.output_dir, resolution))
            generated.extend(
                plot_front_and_correction(
                    metric_rows, front_rows, args.output_dir, resolution
                )
            )

    saturation_rows: list[dict[str, Any]] = []
    if args.command in {"animations", "all"}:
        animation_dir = args.output_dir / "animations"
        payload_paths = [
            args.results_dir / relative
            for relative in sorted(summary["output_hashes"])
            if relative.startswith("visual_payloads/")
        ]
        if not payload_paths:
            raise ValueError("D071 contains no visual payloads")
        for payload_path in payload_paths:
            for component in range(len(COMPONENTS)):
                path, rows = animate_component(
                    payload_path, animation_dir, component=component
                )
                generated.append(path)
                saturation_rows.extend(rows)
        saturation_path = args.output_dir / "visual_scale_saturation.csv"
        write_csv_with_paths(saturation_path, saturation_rows)
        generated.append(saturation_path)

    output_hashes = {
        str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(generated)
    }
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "args": jsonable_args(args),
        "family": summary["family"],
        "population": summary["population"],
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_output_hashes_verified": True,
        "aggregation": "within case-resolution before median/IQR; resolutions separate",
        "scale_contract": {
            "instantaneous_and_correction_residual_scale_limit": INSTANT_LIMIT,
            "cumulative_residual_scale_limit": CUMULATIVE_LIMIT,
            "sensor_limits": [0.0, 1.0],
            "per_frame_normalization": False,
        },
        "field_contract": {
            "residual_error": (
                "free-input predicted increment minus reference-trajectory increment"
            ),
            "accumulated_correction": (
                "sum of direct corrections, not the recurrent state gap"
            ),
        },
        "saturation_row_count": len(saturation_rows),
        "output_hashes": output_hashes,
        "source_sha256": sha256_file(Path(__file__)),
        "git": git_state(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(
        f"D071 visualization status={manifest['status']} "
        f"outputs={len(manifest['output_hashes'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
