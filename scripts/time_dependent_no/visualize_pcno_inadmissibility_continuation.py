#!/usr/bin/env python3
"""Render fixed-scale, all-frame D084 bump rollout animations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths,
    sha256_file,
)

INPUT_SCHEMA = "pcno_inadmissibility_continuation_v1"
OUTPUT_SCHEMA = "pcno_inadmissibility_visualization_v2"
KINDS = ("checkpoint_comparison", "field_intervention")
TYPE_NAMES = {0: "normal", 1: "wall", 2: "outflow", 3: "inflow"}
LONG_EDGE_FACTOR = 4.0
MIN_CIRCLE_RATIO = 0.005

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("gif", "mp4"), default="mp4")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--raster-width", type=int, default=720)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 50 or args.raster_width < 256:
        raise ValueError("fps, dpi, and raster-width must be positive and meaningful")
    return args


def _scalar(value: np.ndarray) -> Any:
    return np.asarray(value).item()


def pressure(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    state64 = np.asarray(state, dtype=np.float64)
    rho = state64[..., 0]
    momentum_sq = state64[..., 1] ** 2 + state64[..., 2] ** 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = state64[..., 3] - 0.5 * momentum_sq / rho
        return (gamma - 1.0) * internal


def internal_energy(state: np.ndarray) -> np.ndarray:
    state64 = np.asarray(state, dtype=np.float64)
    rho = state64[..., 0]
    momentum_sq = state64[..., 1] ** 2 + state64[..., 2] ** 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return state64[..., 3] - 0.5 * momentum_sq / rho


def invalid_nodes(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    state64 = np.asarray(state, dtype=np.float64)
    rho = state64[..., 0]
    internal = internal_energy(state64)
    p = (gamma - 1.0) * internal
    return (
        ~np.isfinite(state64).all(axis=-1)
        | ~np.isfinite(internal)
        | ~np.isfinite(p)
        | (rho <= 0.0)
        | (internal <= 0.0)
        | (p <= 0.0)
    )


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if _scalar(arrays["schema"]) != INPUT_SCHEMA:
        raise ValueError(f"unsupported bundle schema: {path}")
    if not bool(_scalar(arrays["all_temporal_frames_retained"])):
        raise ValueError("bundle does not retain every temporal frame")
    reference = arrays["reference_states"]
    if reference.ndim != 3 or reference.shape[0] != 80 or reference.shape[-1] != 4:
        raise ValueError("D084 bundle must contain exactly 80 four-component frames")
    required = {
        "N0_correct_deployed_states",
        "D082_correct_deployed_states",
        "D082_zero_inflow_deployed_states",
        "D082_correct_raw_proposals",
        "D082_correct_model_currents",
        "D082_zero_inflow_raw_proposals",
        "D082_zero_inflow_model_currents",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"bundle lacks animation arrays: {missing}")
    return arrays


def reference_scales(reference: np.ndarray) -> dict[str, float]:
    reference_pressure = pressure(reference)
    finite = reference_pressure[np.isfinite(reference_pressure)]
    if finite.size == 0:
        raise ValueError("reference pressure is wholly nonfinite")
    state_min = float(np.min(finite))
    state_max = float(np.max(finite))
    state_range = state_max - state_min
    if state_range <= 0.0:
        raise ValueError("reference pressure range must be positive")
    increments = np.diff(reference_pressure, axis=0)
    finite_increment = np.abs(increments[np.isfinite(increments)])
    increment_q995 = (
        0.0
        if finite_increment.size == 0
        else float(np.quantile(finite_increment, 0.995))
    )
    residual_abs_max = max(5.0 * increment_q995, 0.02 * state_range)
    error_abs_max = max(5.0 * increment_q995, 0.25 * state_range)
    return {
        "pressure_min": state_min,
        "pressure_max": state_max,
        "pressure_range": state_range,
        "reference_pressure_increment_abs_q995": increment_q995,
        "residual_abs_max": residual_abs_max,
        "error_abs_max": error_abs_max,
        "difference_abs_max": max(residual_abs_max, error_abs_max),
    }


def build_continuous_field_map(
    positions: np.ndarray, *, raster_width: int
) -> dict[str, Any]:
    """Build a presentation-only full-node piecewise-linear field map."""

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("positions must contain at least three planar nodes")
    if not np.isfinite(points).all():
        raise ValueError("positions must be finite")
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("positions must span a two-dimensional domain")

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    triangles = triangulation.triangles
    edge_pairs = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    edge_lengths = np.linalg.norm(
        points[edge_pairs[:, 0]] - points[edge_pairs[:, 1]], axis=1
    )
    nearest_spacing = np.full(points.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(nearest_spacing, edge_pairs[:, 0], edge_lengths)
    np.minimum.at(nearest_spacing, edge_pairs[:, 1], edge_lengths)
    rolled_triangles = np.roll(triangles, 1, axis=1)
    triangle_edge_lengths = np.linalg.norm(
        points[triangles] - points[rolled_triangles], axis=2
    )
    local_spacing = np.maximum(
        nearest_spacing[triangles], nearest_spacing[rolled_triangles]
    )
    long_bridge_mask = np.any(
        triangle_edge_lengths > LONG_EDGE_FACTOR * local_spacing, axis=1
    )
    flat_triangle_mask = mtri.TriAnalyzer(triangulation).get_flat_tri_mask(
        min_circle_ratio=MIN_CIRCLE_RATIO
    )
    triangle_mask = long_bridge_mask | flat_triangle_mask
    triangulation.set_mask(triangle_mask)

    raster_height = max(80, round(raster_width * y_span / x_span))
    x_coordinates = np.linspace(x_min, x_max, raster_width, dtype=np.float64)
    y_coordinates = np.linspace(y_min, y_max, raster_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    triangle_index = triangulation.get_trifinder()(grid_x, grid_y)
    valid_mask = triangle_index >= 0
    valid_flat_indices = np.flatnonzero(valid_mask.ravel())
    vertices = triangles[triangle_index[valid_mask]]
    point0 = points[vertices[:, 0]]
    point1 = points[vertices[:, 1]]
    point2 = points[vertices[:, 2]]
    queries = np.column_stack((grid_x[valid_mask], grid_y[valid_mask]))
    denominator = (point1[:, 1] - point2[:, 1]) * (
        point0[:, 0] - point2[:, 0]
    ) + (point2[:, 0] - point1[:, 0]) * (point0[:, 1] - point2[:, 1])
    weight0 = (
        (point1[:, 1] - point2[:, 1]) * (queries[:, 0] - point2[:, 0])
        + (point2[:, 0] - point1[:, 0]) * (queries[:, 1] - point2[:, 1])
    ) / denominator
    weight1 = (
        (point2[:, 1] - point0[:, 1]) * (queries[:, 0] - point2[:, 0])
        + (point0[:, 0] - point2[:, 0]) * (queries[:, 1] - point2[:, 1])
    ) / denominator
    weights = np.column_stack((weight0, weight1, 1.0 - weight0 - weight1))
    return {
        "shape": (raster_height, raster_width),
        "extent": (float(x_min), float(x_max), float(y_min), float(y_max)),
        "source_node_count": int(points.shape[0]),
        "x_coordinates": x_coordinates,
        "y_coordinates": y_coordinates,
        "valid_flat_indices": valid_flat_indices,
        "vertices": vertices,
        "weights": weights,
        "masked_triangle_count": int(np.count_nonzero(triangle_mask)),
        "triangle_count": int(triangles.shape[0]),
    }


def rasterize_field(value: np.ndarray, field_map: Mapping[str, Any]) -> np.ndarray:
    nodal_value = np.asarray(value, dtype=np.float64)
    if nodal_value.ndim != 1 or nodal_value.shape[0] != field_map["source_node_count"]:
        raise ValueError("field must provide one scalar for every source node")
    raster = np.full(np.prod(field_map["shape"]), np.nan, dtype=np.float64)
    vertices = field_map["vertices"]
    weights = field_map["weights"]
    raster[field_map["valid_flat_indices"]] = np.sum(
        nodal_value[vertices] * weights, axis=1
    )
    return raster.reshape(field_map["shape"])


def saturation_fraction(value: np.ndarray, lower: float, upper: float) -> float:
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return 0.0
    return float(np.mean((finite < lower) | (finite > upper)))


def animation_fields(
    arrays: Mapping[str, np.ndarray], kind: str
) -> tuple[list[np.ndarray], list[str], list[str], list[np.ndarray]]:
    reference_state = arrays["reference_states"]
    n0_state = arrays["N0_correct_deployed_states"]
    correct_state = arrays["D082_correct_deployed_states"]
    zero_state = arrays["D082_zero_inflow_deployed_states"]
    reference = pressure(reference_state)
    n0 = pressure(n0_state)
    correct = pressure(correct_state)
    zero = pressure(zero_state)
    if kind == "checkpoint_comparison":
        correction = np.zeros_like(reference)
        correction[1:] = correct[1:] - pressure(arrays["D082_correct_raw_proposals"])
        fields = [
            reference,
            n0,
            correct,
            n0 - reference,
            correct - reference,
            correction,
        ]
        titles = [
            "Reference",
            "N0 checkpoint",
            "D082 checkpoint",
            "N0 − reference",
            "D082 − reference",
            "D082 deployed − raw",
        ]
        scale_kinds = ["state", "state", "state", "error", "error", "residual"]
        invalid_sources = [
            reference_state,
            n0_state,
            correct_state,
            n0_state,
            correct_state,
            correct_state,
        ]
        return fields, titles, scale_kinds, invalid_sources
    if kind != "field_intervention":
        raise ValueError(f"unsupported animation kind: {kind}")
    correct_residual = np.zeros_like(reference)
    zero_residual = np.zeros_like(reference)
    correct_residual[1:] = pressure(arrays["D082_correct_raw_proposals"]) - pressure(
        arrays["D082_correct_model_currents"]
    )
    zero_residual[1:] = pressure(arrays["D082_zero_inflow_raw_proposals"]) - pressure(
        arrays["D082_zero_inflow_model_currents"]
    )
    fields = [
        reference,
        correct,
        zero,
        correct_residual,
        zero_residual,
        zero_residual - correct_residual,
    ]
    titles = [
        "Reference",
        "D082 checkpoint",
        "D082 (inflow field zeroed)",
        "D082 raw residual",
        "Zero-inflow raw residual",
        "Zero-inflow − D082 residual",
    ]
    scale_kinds = ["state", "state", "state", "residual", "residual", "residual"]
    invalid_sources = [
        reference_state,
        correct_state,
        zero_state,
        correct_state,
        zero_state,
        zero_state,
    ]
    return fields, titles, scale_kinds, invalid_sources


def frame_status(arrays: Mapping[str, np.ndarray], frame: int, kind: str) -> str:
    if kind == "checkpoint_comparison":
        variants = (
            ("N0", "N0_correct_deployed_states"),
            ("D082", "D082_correct_deployed_states"),
        )
    else:
        variants = (
            ("D082", "D082_correct_deployed_states"),
            ("inflow field zeroed", "D082_zero_inflow_deployed_states"),
        )
    counts = " · ".join(
        f"{label}: {int(np.count_nonzero(invalid_nodes(arrays[name][frame])))}"
        for label, name in variants
    )
    return f"Inadmissible nodes — {counts}    |    red × marks inadmissibility"


def render_one(
    arrays: Mapping[str, np.ndarray],
    bundle_path: Path,
    output_path: Path,
    *,
    kind: str,
    fps: int,
    dpi: int,
    raster_width: int,
) -> dict[str, Any]:
    reference = arrays["reference_states"]
    frame_count = reference.shape[0]
    if frame_count != 80:
        raise ValueError("animation must include all 80 temporal frames")
    scales = reference_scales(reference)
    fields, titles, scale_kinds, invalid_sources = animation_fields(arrays, kind)
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    field_map = build_continuous_field_map(positions, raster_width=raster_width)
    physical_times = arrays["physical_times"]
    state_norm = Normalize(scales["pressure_min"], scales["pressure_max"])
    difference_norm = TwoSlopeNorm(
        vmin=-scales["difference_abs_max"],
        vcenter=0.0,
        vmax=scales["difference_abs_max"],
    )
    norms = {
        "state": state_norm,
        "error": difference_norm,
        "residual": difference_norm,
    }
    state_cmap = plt.get_cmap("viridis").copy()
    difference_cmap = plt.get_cmap("RdBu_r").copy()
    state_cmap.set_bad("white")
    difference_cmap.set_bad("white")
    cmaps = {
        "state": state_cmap,
        "error": difference_cmap,
        "residual": difference_cmap,
    }

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 5.8))
    figure.subplots_adjust(
        left=0.025,
        right=0.90,
        bottom=0.055,
        top=0.84,
        wspace=0.025,
        hspace=0.24,
    )
    axes_flat = axes.ravel()
    images = []
    invalid_scatters = []
    unavailable_labels = []
    for axis, title, scale_kind in zip(axes_flat, titles, scale_kinds, strict=True):
        image = axis.imshow(
            rasterize_field(fields[len(images)][0], field_map),
            origin="lower",
            extent=field_map["extent"],
            cmap=cmaps[scale_kind],
            norm=norms[scale_kind],
            interpolation="bilinear",
            aspect="equal",
        )
        invalid_scatter = axis.scatter(
            [], [], s=17.0, marker="x", color="#D55E00", linewidths=0.8, zorder=5
        )
        unavailable_label = axis.text(
            0.5,
            0.5,
            "unavailable after nonfinite stop",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=9,
            color="#555555",
            visible=False,
        )
        axis.set_title(title, pad=5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        images.append(image)
        invalid_scatters.append(invalid_scatter)
        unavailable_labels.append(unavailable_label)

    state_colorbar_axis = figure.add_axes([0.92, 0.53, 0.012, 0.285])
    state_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=state_norm, cmap="viridis"),
        cax=state_colorbar_axis,
    )
    state_colorbar.set_label("pressure", labelpad=7)
    state_colorbar.ax.tick_params(labelsize=8)
    difference_colorbar_axis = figure.add_axes([0.92, 0.105, 0.012, 0.285])
    difference_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=difference_norm, cmap="RdBu_r"),
        cax=difference_colorbar_axis,
    )
    difference_colorbar.set_label("signed pressure difference", labelpad=7)
    difference_colorbar.ax.tick_params(labelsize=8)

    case = str(_scalar(arrays["trajectory"]))
    title_artist = figure.suptitle(
        "", x=0.46, y=0.965, fontsize=13, fontweight="semibold"
    )
    status_artist = figure.text(
        0.46,
        0.90,
        "",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    def update(frame: int):
        artists = []
        for image, invalid_scatter, unavailable_label, field, source in zip(
            images,
            invalid_scatters,
            unavailable_labels,
            fields,
            invalid_sources,
            strict=True,
        ):
            image.set_data(rasterize_field(field[frame], field_map))
            invalid = invalid_nodes(source[frame])
            invalid_scatter.set_offsets(
                positions[invalid] if np.any(invalid) else np.empty((0, 2))
            )
            unavailable_label.set_visible(not np.isfinite(field[frame]).any())
            artists.extend((image, invalid_scatter, unavailable_label))
        title_artist.set_text(
            f"D084 · supersonic bump case {case} · "
            f"{kind.replace('_', ' ')} · frame {frame}/79 · "
            f"t={float(physical_times[frame]):.6g}"
        )
        status_artist.set_text(frame_status(arrays, frame, kind))
        artists.extend((title_artist, status_artist))
        return artists

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=range(frame_count),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    if output_path.suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    else:
        writer = animation.PillowWriter(fps=fps)
    movie.save(output_path, writer=writer, dpi=dpi)
    plt.close(figure)

    clipping: dict[str, float] = {}
    for index, (field, scale_kind) in enumerate(zip(fields, scale_kinds, strict=True)):
        if scale_kind == "state":
            lower, upper = scales["pressure_min"], scales["pressure_max"]
        elif scale_kind == "error":
            lower, upper = -scales["error_abs_max"], scales["error_abs_max"]
        else:
            lower, upper = -scales["residual_abs_max"], scales["residual_abs_max"]
        clipping[titles[index]] = saturation_fraction(field, lower, upper)
    return {
        "trajectory": case,
        "kind": kind,
        "input": bundle_path.name,
        "input_sha256": sha256_file(bundle_path),
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "output_bytes": int(output_path.stat().st_size),
        "temporal_frames": frame_count,
        "includes_initial_state": True,
        "temporal_subsampling": False,
        "original_node_count": int(arrays["positions"].shape[0]),
        "interpolation_source_node_count": int(field_map["source_node_count"]),
        "spatial_subsampling": False,
        "presentation_raster_shape": list(field_map["shape"]),
        "continuous_field_renderer": (
            "full-node piecewise-linear triangulation rasterized without mesh edges "
            "or nodal markers"
        ),
        "masked_triangle_count": int(field_map["masked_triangle_count"]),
        "triangle_count": int(field_map["triangle_count"]),
        "mesh_edges_drawn": False,
        "nodal_markers_drawn": False,
        "boundary_marker_overlay": False,
        "inadmissible_marker_overlay": True,
        "scale_contract": {
            **scales,
            "source": "reference rollout only",
            "per_frame_autoscaling": False,
            "prediction_dependent_autoscaling": False,
        },
        "field_saturation_fraction": clipping,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for bundle_path in args.bundles:
        arrays = load_bundle(bundle_path)
        case = str(_scalar(arrays["trajectory"]))
        for kind in KINDS:
            output_path = args.output_dir / f"case_{case}_{kind}.{args.format}"
            records.append(
                render_one(
                    arrays,
                    bundle_path,
                    output_path,
                    kind=kind,
                    fps=args.fps,
                    dpi=args.dpi,
                    raster_width=args.raster_width,
                )
            )
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": {
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract": {
            "family": "bump",
            "node_type_meanings": TYPE_NAMES,
            "boundary_policy": "frozen; visualizer does not execute a model",
            "temporal_frames_per_animation": 80,
            "temporal_subsampling": False,
            "fixed_reference_only_scales": True,
            "result_dependent_autoscaling": False,
            "layout": "clean three-by-two CPGNet-style comparison",
            "mesh_edges_drawn": False,
            "nodal_markers_drawn": False,
            "boundary_marker_overlay": False,
            "spatial_subsampling": False,
            "unavailable_post_nonfinite_frames": (
                "retained as explicit blank/NaN temporal slots, never fabricated"
            ),
            "pressure_residual_definition": (
                "pressure(raw conservative proposal) minus pressure(model-current "
                "conservative input); diagnostic nonlinear primitive difference"
            ),
        },
        "animations": records,
        "claim_boundary": {
            "shown": (
                "reference, deployed checkpoint states, errors, frozen field "
                "intervention, raw residuals, projection corrections, and "
                "inadmissible-node locations"
            ),
            "not_shown": (
                "physical flux or conservation, exact DG boundary replay, or an "
                "independent causal effect of inadmissibility"
            ),
        },
    }
    atomic_write_json_with_paths(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
