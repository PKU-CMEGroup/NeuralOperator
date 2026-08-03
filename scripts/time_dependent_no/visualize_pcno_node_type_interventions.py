#!/usr/bin/env python3
"""Render provenance-bound node-type intervention figures and animations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D

INPUT_SCHEMA = "pcno_node_type_interventions_v1"
OUTPUT_SCHEMA = "pcno_node_type_visualization_v1"
FAMILY_TYPE_NAMES = {
    "bump": {0: "normal", 1: "wall", 2: "outflow", 3: "inflow"},
    "dynamic_fv": {
        0: "interior",
        1: "y-symmetry contact",
        2: "x-extrapolation contact",
        3: "contact with both",
    },
}
TYPE_COLORS = {
    0: "#999999",
    1: "#0072B2",
    2: "#D55E00",
    3: "#009E73",
}
SENSITIVITY_FIELDS = (
    ("post_lift_difference_norm", "post-lift"),
    ("block_0_pointwise_difference_norm", "block 0\npointwise"),
    ("block_0_integral_difference_norm", "block 0\nintegral"),
    ("block_0_differential_difference_norm", "block 0\ndifferential"),
    ("block_0_output_difference_norm", "block 0\noutput"),
    ("block_3_output_difference_norm", "block 3\noutput"),
    ("decoder_hidden_difference_norm", "decoder\nhidden"),
    ("normalized_residual_difference_norm", "decoded\nresidual"),
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    movie = subparsers.add_parser(
        "animation", help="render synchronized correct/all-normal state movies"
    )
    movie.add_argument("--bundles", type=Path, nargs="+", required=True)
    movie.add_argument("--completion-csv", type=Path)
    movie.add_argument("--output-stem", type=Path, required=True)
    movie.add_argument(
        "--mode",
        choices=("teacher_forced", "free_rollout"),
        required=True,
    )
    movie.add_argument("--field", choices=("density", "pressure"), required=True)
    movie.add_argument("--gamma", type=float, default=1.4)
    movie.add_argument("--units", required=True)
    movie.add_argument("--state-min", type=float, required=True)
    movie.add_argument("--state-max", type=float, required=True)
    movie.add_argument("--difference-abs-max", type=float, required=True)
    movie.add_argument("--update-abs-max", type=float, required=True)
    movie.add_argument("--linthresh", type=float, required=True)
    movie.add_argument("--boundary-distance-max", type=float, required=True)
    movie.add_argument("--scale-source", required=True)
    movie.add_argument("--fps", type=int, default=5)
    movie.add_argument("--dpi", type=int, default=100)
    movie.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    movie.add_argument("--overwrite", action="store_true")

    sensitivity = subparsers.add_parser(
        "sensitivity", help="render fixed-scale activation-difference maps"
    )
    sensitivity.add_argument("--traces", type=Path, nargs="+", required=True)
    sensitivity.add_argument("--output-stem", type=Path, required=True)
    sensitivity.add_argument("--reference-norm", type=float, required=True)
    sensitivity.add_argument("--transformed-max", type=float, required=True)
    sensitivity.add_argument("--boundary-distance-max", type=float, required=True)
    sensitivity.add_argument("--max-nodes", type=int, default=25_000)
    sensitivity.add_argument("--scale-source", required=True)
    sensitivity.add_argument("--dpi", type=int, default=180)
    sensitivity.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"expected scalar array, got shape {value.shape}")
    return value.item()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    _strict_output(path, overwrite=overwrite)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_common(values: Sequence[str], name: str) -> str:
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(f"{name} differs across inputs: {unique}")
    return unique[0]


def _resolution_sort_key(value: str) -> tuple[int, int, str]:
    try:
        nx_text, ny_text = value.lower().split("x", maxsplit=1)
        return int(nx_text), int(ny_text), value
    except (TypeError, ValueError):
        return 10**9, 10**9, value


def animation_frame_indices(frame_count: int) -> list[int]:
    """Return every comparable rollout call in temporal order."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return list(range(frame_count))


def conservative_field(
    state: np.ndarray, *, field: str, gamma: float = 1.4
) -> np.ndarray:
    """Convert [rho, rho*u, rho*v, E] to a scalar primitive field."""

    array = np.asarray(state, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("conservative state must have four components")
    density = array[..., 0]
    if field == "density":
        return density
    if field != "pressure":
        raise ValueError(f"unsupported field: {field}")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    safe_density = np.where(density != 0.0, density, np.nan)
    kinetic = 0.5 * (array[..., 1] ** 2 + array[..., 2] ** 2) / safe_density
    return (gamma - 1.0) * (array[..., 3] - kinetic)


def _validate_scale_contract(args: argparse.Namespace) -> None:
    if not args.state_min < args.state_max:
        raise ValueError("state-min must be below state-max")
    for name in (
        "difference_abs_max",
        "update_abs_max",
        "linthresh",
        "boundary_distance_max",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.linthresh >= min(args.difference_abs_max, args.update_abs_max):
        raise ValueError("linthresh must be below both symmetric scale limits")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files}
    if _scalar(result["schema"]) != INPUT_SCHEMA:
        raise ValueError(f"unexpected schema in {path}")
    return result


def _load_animation_bundles(
    paths: Sequence[Path], *, mode: str
) -> list[tuple[Path, dict[str, np.ndarray]]]:
    required_metadata = (
        "family",
        "case_id",
        "resolution",
        "visualization_only_subsampling",
        "visualization_node_indices",
        "physical_times",
        "nodes",
        "physical_node_type",
        "boundary_distance",
    )
    required_series = (
        "correct_prediction",
        "correct_residual",
        "all_normal_prediction",
        "all_normal_residual",
    )
    loaded: list[tuple[Path, dict[str, np.ndarray]]] = []
    for path in paths:
        bundle = _load_npz(path)
        missing = [name for name in required_metadata if name not in bundle] + [
            f"{mode}_{name}"
            for name in required_series
            if f"{mode}_{name}" not in bundle
        ]
        if missing:
            raise ValueError(f"{path} lacks required arrays: {missing}")
        if not bool(_scalar(bundle["visualization_only_subsampling"])):
            raise ValueError(f"{path} does not identify visualization subsampling")
        nodes = bundle["nodes"]
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"invalid node array in {path}: {nodes.shape}")
        node_count = nodes.shape[0]
        if bundle["physical_node_type"].shape != (node_count,):
            raise ValueError(f"invalid node-type array in {path}")
        if bundle["boundary_distance"].shape != (node_count,):
            raise ValueError(f"invalid boundary-distance array in {path}")
        for suffix in required_series:
            values = bundle[f"{mode}_{suffix}"]
            if values.ndim != 3 or values.shape[1:] != (node_count, 4):
                raise ValueError(
                    f"invalid {mode}_{suffix} shape in {path}: {values.shape}"
                )
        loaded.append((path, bundle))

    families = [str(_scalar(bundle["family"])) for _, bundle in loaded]
    cases = [str(_scalar(bundle["case_id"])) for _, bundle in loaded]
    _validate_common(families, "family")
    _validate_common(cases, "case_id")
    resolutions = [str(_scalar(bundle["resolution"])) for _, bundle in loaded]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError(f"duplicate animation resolutions: {resolutions}")
    return sorted(
        loaded,
        key=lambda item: _resolution_sort_key(str(_scalar(item[1]["resolution"]))),
    )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        raise ValueError("missing boolean")
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def _completion_failures(
    path: Path | None,
    bundles: Sequence[tuple[Path, dict[str, np.ndarray]]],
    *,
    mode: str,
) -> dict[str, dict[str, int | bool | None]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, int | bool | None]] = {}
    for _, bundle in bundles:
        family = str(_scalar(bundle["family"]))
        case_id = str(_scalar(bundle["case_id"]))
        resolution = str(_scalar(bundle["resolution"]))
        selected = [
            row
            for row in rows
            if row.get("family") == family
            and row.get("case_id") == case_id
            and row.get("resolution") == resolution
            and row.get("mode") == mode
            and row.get("intervention") == "all_normal"
        ]
        if not selected:
            raise ValueError(
                "completion CSV lacks all-normal rows for "
                f"{family}/{case_id}/{resolution}/{mode}"
            )
        selected.sort(key=lambda row: int(row["call"]))
        inadmissible = [
            int(row["call"])
            for row in selected
            if not _parse_bool(row.get("finite"))
            or not _parse_bool(row.get("admissible"))
        ]
        result[resolution] = {
            "rows": len(selected),
            "last_call": int(selected[-1]["call"]),
            "first_inadmissible_call": (inadmissible[0] if inadmissible else None),
            "all_recorded_calls_admissible": not inadmissible,
        }
    return result


def _field_series(
    bundle: dict[str, np.ndarray],
    *,
    mode: str,
    field: str,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct_prediction = np.asarray(
        bundle[f"{mode}_correct_prediction"], dtype=np.float64
    )
    correct_residual = np.asarray(bundle[f"{mode}_correct_residual"], dtype=np.float64)
    all_normal_prediction = np.asarray(
        bundle[f"{mode}_all_normal_prediction"], dtype=np.float64
    )
    all_normal_residual = np.asarray(
        bundle[f"{mode}_all_normal_residual"], dtype=np.float64
    )
    count = min(
        correct_prediction.shape[0],
        correct_residual.shape[0],
        all_normal_prediction.shape[0],
        all_normal_residual.shape[0],
        bundle["physical_times"].shape[0],
    )
    if count <= 0:
        raise ValueError("animation bundle has no comparable frames")
    correct_prediction = correct_prediction[:count]
    correct_residual = correct_residual[:count]
    all_normal_prediction = all_normal_prediction[:count]
    all_normal_residual = all_normal_residual[:count]
    correct_current = correct_prediction - correct_residual
    all_normal_current = all_normal_prediction - all_normal_residual
    correct_state = conservative_field(correct_prediction, field=field, gamma=gamma)
    all_normal_state = conservative_field(
        all_normal_prediction, field=field, gamma=gamma
    )
    correct_update = correct_state - conservative_field(
        correct_current, field=field, gamma=gamma
    )
    all_normal_update = all_normal_state - conservative_field(
        all_normal_current, field=field, gamma=gamma
    )
    return (
        correct_state,
        all_normal_state,
        correct_state - all_normal_state,
        correct_update,
        all_normal_update,
    )


def _clip_fraction(values: np.ndarray, lower: float, upper: float) -> float:
    array = np.asarray(values)
    finite = np.isfinite(array)
    if not np.any(finite):
        return 1.0
    selected = array[finite]
    return float(np.mean((selected < lower) | (selected > upper)))


def _configure_spatial_axis(
    axis: Any, nodes: np.ndarray, *, row: int, row_count: int
) -> None:
    x_min, y_min = np.min(nodes, axis=0)
    x_max, y_max = np.max(nodes, axis=0)
    pad_x = max(0.015 * (x_max - x_min), 1.0e-8)
    pad_y = max(0.015 * (y_max - y_min), 1.0e-8)
    axis.set_xlim(x_min - pad_x, x_max + pad_x)
    axis.set_ylim(y_min - pad_y, y_max + pad_y)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(
        labelleft=False,
        labelbottom=row == row_count - 1,
        length=2,
        pad=1,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#666666")


def _type_legend(family: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=TYPE_COLORS[index],
            markeredgewidth=1.2,
            markersize=5,
            label=f"{index}: {name}",
        )
        for index, name in FAMILY_TYPE_NAMES[family].items()
    ]


def render_animation(args: argparse.Namespace) -> dict[str, Any]:
    """Render one synchronized fixed-scale animation and its final frame."""

    _validate_scale_contract(args)
    bundles = _load_animation_bundles(args.bundles, mode=args.mode)
    family = _validate_common(
        [str(_scalar(bundle["family"])) for _, bundle in bundles],
        "family",
    )
    case_id = _validate_common(
        [str(_scalar(bundle["case_id"])) for _, bundle in bundles],
        "case_id",
    )
    if family not in FAMILY_TYPE_NAMES:
        raise ValueError(f"unsupported family: {family}")

    series = [
        _field_series(
            bundle,
            mode=args.mode,
            field=args.field,
            gamma=args.gamma,
        )
        for _, bundle in bundles
    ]
    frame_count = min(values[0].shape[0] for values in series)
    times = np.asarray(bundles[0][1]["physical_times"][:frame_count])
    for _, bundle in bundles[1:]:
        candidate = np.asarray(bundle["physical_times"][:frame_count])
        if not np.allclose(candidate, times, rtol=0.0, atol=1.0e-12):
            raise ValueError("physical times differ across synchronized bundles")
    frame_indices = animation_frame_indices(frame_count)
    completion = _completion_failures(args.completion_csv, bundles, mode=args.mode)

    movie_path = args.output_stem.with_suffix(f".{args.format}")
    png_path = args.output_stem.with_name(f"{args.output_stem.name}_final").with_suffix(
        ".png"
    )
    pdf_path = args.output_stem.with_name(f"{args.output_stem.name}_final").with_suffix(
        ".pdf"
    )
    manifest_path = args.output_stem.with_suffix(".json")
    for path in (movie_path, png_path, pdf_path, manifest_path):
        _strict_output(path, overwrite=args.overwrite)

    state_norm = Normalize(vmin=args.state_min, vmax=args.state_max)
    difference_norm = SymLogNorm(
        linthresh=args.linthresh,
        vmin=-args.difference_abs_max,
        vmax=args.difference_abs_max,
        base=10,
    )
    update_norm = SymLogNorm(
        linthresh=args.linthresh,
        vmin=-args.update_abs_max,
        vmax=args.update_abs_max,
        base=10,
    )
    boundary_norm = Normalize(vmin=0.0, vmax=args.boundary_distance_max)
    row_count = len(bundles)
    figure, axes = plt.subplots(
        row_count,
        6,
        figsize=(17.5, 2.9 * row_count + 1.25),
        squeeze=False,
        constrained_layout=True,
    )
    titles = (
        "correct types: state",
        "all-normal: state",
        "correct - all-normal",
        "correct types: physical update",
        "all-normal: physical update",
        "boundary distance + type",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="bold")

    scatters: list[list[Any]] = []
    clip_labels: list[list[Any]] = []
    boundary_scatter = None
    for row, ((_, bundle), values) in enumerate(zip(bundles, series, strict=True)):
        nodes = np.asarray(bundle["nodes"])
        node_count = nodes.shape[0]
        marker_size = max(0.18, min(9.0, 5000.0 / node_count))
        row_scatters: list[Any] = []
        row_labels: list[Any] = []
        norms = (
            state_norm,
            state_norm,
            difference_norm,
            update_norm,
            update_norm,
        )
        cmaps = ("viridis", "viridis", "RdBu_r", "RdBu_r", "RdBu_r")
        for column in range(5):
            axis = axes[row, column]
            scatter = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=values[column][0],
                s=marker_size,
                cmap=cmaps[column],
                norm=norms[column],
                linewidths=0.0,
                rasterized=True,
            )
            label = axis.text(
                0.015,
                0.025,
                "",
                transform=axis.transAxes,
                fontsize=6.5,
                color="#111111",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.74,
                    "pad": 1.2,
                },
            )
            _configure_spatial_axis(axis, nodes, row=row, row_count=row_count)
            row_scatters.append(scatter)
            row_labels.append(label)

        boundary_axis = axes[row, 5]
        distance = np.asarray(bundle["boundary_distance"], dtype=np.float64)
        boundary_scatter = boundary_axis.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=distance,
            s=marker_size,
            cmap="cividis_r",
            norm=boundary_norm,
            linewidths=0.0,
            rasterized=True,
        )
        node_type = np.asarray(bundle["physical_node_type"], dtype=np.int64)
        for type_index in sorted(FAMILY_TYPE_NAMES[family]):
            selected = node_type == type_index
            if not np.any(selected):
                continue
            boundary_axis.scatter(
                nodes[selected, 0],
                nodes[selected, 1],
                facecolors="none",
                edgecolors=TYPE_COLORS[type_index],
                s=max(marker_size * 4.0, 4.0),
                linewidths=0.45,
                rasterized=True,
            )
        resolution = str(_scalar(bundle["resolution"]))
        boundary_axis.text(
            0.015,
            0.025,
            (
                f"subsampled nodes: {node_count:,}\n"
                f"d>{args.boundary_distance_max:g}: "
                f"{100.0 * np.mean(distance > args.boundary_distance_max):.2f}%"
            ),
            transform=boundary_axis.transAxes,
            fontsize=6.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
                "pad": 1.2,
            },
        )
        if resolution in completion:
            status = completion[resolution]
            failure = status["first_inadmissible_call"]
            status_text = (
                "all recorded outputs admissible"
                if failure is None
                else f"first inadmissible all-normal call: {failure}"
            )
            boundary_axis.text(
                0.015,
                0.975,
                status_text,
                transform=boundary_axis.transAxes,
                va="top",
                fontsize=6.5,
                color="#9E2A2B" if failure is not None else "#1B5E20",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
            )
        _configure_spatial_axis(boundary_axis, nodes, row=row, row_count=row_count)
        axes[row, 0].set_ylabel(resolution)
        scatters.append(row_scatters)
        clip_labels.append(row_labels)

    if boundary_scatter is None:
        raise AssertionError("at least one animation bundle is required")
    figure.colorbar(
        scatters[0][0],
        ax=axes[:, 0:2].ravel().tolist(),
        location="bottom",
        shrink=0.62,
        pad=0.02,
        label=f"{args.field} ({args.units}); fixed state scale",
        extend="both",
    )
    figure.colorbar(
        scatters[0][2],
        ax=axes[:, 2].ravel().tolist(),
        location="bottom",
        shrink=0.82,
        pad=0.02,
        label="state difference; fixed symmetric log scale",
        extend="both",
    )
    figure.colorbar(
        scatters[0][3],
        ax=axes[:, 3:5].ravel().tolist(),
        location="bottom",
        shrink=0.62,
        pad=0.02,
        label="physical update; fixed symmetric log scale",
        extend="both",
    )
    figure.colorbar(
        boundary_scatter,
        ax=axes[:, 5].ravel().tolist(),
        location="bottom",
        shrink=0.82,
        pad=0.02,
        label=f"boundary distance, clipped at {args.boundary_distance_max:g}",
        extend="max",
    )
    axes[0, 5].legend(
        handles=_type_legend(family),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title=f"{family} node types",
        title_fontsize=7,
    )

    mode_explanation = (
        "same reference state at every call; model-facing node type is the "
        "only intervention"
        if args.mode == "teacher_forced"
        else "after call 1, differences include recurrent state divergence"
    )

    def update(frame_position: int) -> list[Any]:
        source_index = frame_indices[frame_position]
        call = source_index + 1
        updated: list[Any] = []
        failure_labels = []
        for row, values in enumerate(series):
            limits = (
                (args.state_min, args.state_max),
                (args.state_min, args.state_max),
                (-args.difference_abs_max, args.difference_abs_max),
                (-args.update_abs_max, args.update_abs_max),
                (-args.update_abs_max, args.update_abs_max),
            )
            for column in range(5):
                current = values[column][source_index]
                scatters[row][column].set_array(current)
                clipped = 100.0 * _clip_fraction(current, *limits[column])
                clip_labels[row][column].set_text(f"clipped: {clipped:.2f}%")
                updated.extend((scatters[row][column], clip_labels[row][column]))
            resolution = str(_scalar(bundles[row][1]["resolution"]))
            failure = completion.get(resolution, {}).get("first_inadmissible_call")
            if failure == call:
                failure_labels.append(resolution)
        warning = (
            " | first inadmissible all-normal output: " + ", ".join(failure_labels)
            if failure_labels
            else ""
        )
        figure.suptitle(
            f"{family} case {case_id} | {args.mode} | call {call} | "
            f"t={times[source_index]:.6g}\n"
            f"frozen-checkpoint all-normal channel-use intervention; "
            f"{mode_explanation}{warning}",
            fontsize=10,
            color="#9E2A2B" if failure_labels else "#111111",
        )
        return updated

    final_position = len(frame_indices) - 1
    update(final_position)
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / args.fps,
        blit=False,
    )
    if args.format == "mp4":
        writer: Any = animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
            ],
            metadata={
                "title": (f"{family} {case_id} node-type intervention {args.mode}")
            },
        )
    else:
        writer = animation.PillowWriter(fps=args.fps)
    movie.save(movie_path, writer=writer, dpi=args.dpi)
    plt.close(figure)

    input_records = [
        {
            "file": path.name,
            "sha256": _sha256(path),
            "resolution": str(_scalar(bundle["resolution"])),
            "saved_node_count": int(bundle["nodes"].shape[0]),
            "visualization_index_count": int(
                bundle["visualization_node_indices"].shape[0]
            ),
            "visualization_only_subsampling": bool(
                _scalar(bundle["visualization_only_subsampling"])
            ),
        }
        for path, bundle in bundles
    ]
    manifest: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "kind": "node_type_intervention_animation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": {
            "file": Path(__file__).name,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "family": family,
        "case_id": case_id,
        "mode": args.mode,
        "field": args.field,
        "units": args.units,
        "gamma": float(args.gamma),
        "intervention": "correct node types versus all-normal",
        "boundary_policy": "frozen; representation intervention only",
        "scale_contract": {
            "source": args.scale_source,
            "state": [float(args.state_min), float(args.state_max)],
            "difference": [
                -float(args.difference_abs_max),
                float(args.difference_abs_max),
            ],
            "update": [
                -float(args.update_abs_max),
                float(args.update_abs_max),
            ],
            "symmetric_linthresh": float(args.linthresh),
            "boundary_distance": [
                0.0,
                float(args.boundary_distance_max),
            ],
            "result_dependent_autoscaling": False,
        },
        "frame_contract": {
            "comparable_saved_frames": int(frame_count),
            "rendered_zero_based_indices": frame_indices,
            "rendered_calls": [index + 1 for index in frame_indices],
            "physical_times": [float(times[index]) for index in frame_indices],
            "temporal_subsampling": False,
            "all_comparable_frames_rendered": True,
            "fps": int(args.fps),
        },
        "completion_and_admissibility": completion,
        "claim_boundary": {
            "teacher_forced": (
                "same-state frozen-checkpoint node-type channel intervention"
            ),
            "free_rollout": (
                "after call 1, state differences include recurrent divergence"
            ),
            "not_claimed": (
                "not a training ablation, boundary-condition improvement, or "
                "causal decomposition of PCNO branches"
            ),
        },
        "inputs": input_records,
        "completion_csv": (
            None
            if args.completion_csv is None
            else {
                "file": args.completion_csv.name,
                "sha256": _sha256(args.completion_csv),
            }
        ),
        "outputs": {
            movie_path.name: _sha256(movie_path),
            png_path.name: _sha256(png_path),
            pdf_path.name: _sha256(pdf_path),
        },
    }
    _write_json(manifest_path, manifest, overwrite=args.overwrite)
    return manifest


def _sensitivity_indices(
    node_count: int, resolution: str, maximum_nodes: int
) -> tuple[np.ndarray, str]:
    if maximum_nodes <= 0:
        raise ValueError("max-nodes must be positive")
    if node_count <= maximum_nodes:
        return np.arange(node_count, dtype=np.int64), "all_nodes"
    try:
        nx, ny, _ = _resolution_sort_key(resolution)
        if nx * ny != node_count:
            raise ValueError
        stride = int(np.ceil(np.sqrt(node_count / maximum_nodes)))
        indices = np.arange(node_count, dtype=np.int64).reshape(ny, nx)
        return (
            indices[::stride, ::stride].reshape(-1),
            f"regular_grid_stride_{stride}",
        )
    except ValueError:
        stride = int(np.ceil(node_count / maximum_nodes))
        return (
            np.arange(0, node_count, stride, dtype=np.int64),
            f"native_graph_index_stride_{stride}",
        )


def _load_sensitivity_traces(
    paths: Sequence[Path],
) -> list[tuple[Path, dict[str, np.ndarray]]]:
    metadata = (
        "family",
        "case_id",
        "resolution",
        "mode",
        "call",
        "physical_time",
        "intervention",
        "nodes",
        "physical_node_type",
        "boundary_distance",
    )
    loaded = []
    for path in paths:
        trace = _load_npz(path)
        missing = [
            name
            for name in (*metadata, *(name for name, _ in SENSITIVITY_FIELDS))
            if name not in trace
        ]
        if missing:
            raise ValueError(f"{path} lacks sensitivity arrays: {missing}")
        node_count = trace["nodes"].shape[0]
        if trace["nodes"].shape != (node_count, 2):
            raise ValueError(f"invalid nodes in {path}")
        if trace["physical_node_type"].shape != (node_count,):
            raise ValueError(f"invalid node types in {path}")
        if trace["boundary_distance"].shape != (node_count,):
            raise ValueError(f"invalid boundary distance in {path}")
        for field, _ in SENSITIVITY_FIELDS:
            values = np.asarray(trace[field])
            if values.shape != (node_count,):
                raise ValueError(f"invalid {field} in {path}: {values.shape}")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{field} must contain finite norms in {path}")
        loaded.append((path, trace))

    family = _validate_common(
        [str(_scalar(trace["family"])) for _, trace in loaded],
        "family",
    )
    if family not in FAMILY_TYPE_NAMES:
        raise ValueError(f"unsupported family: {family}")
    _validate_common(
        [str(_scalar(trace["case_id"])) for _, trace in loaded],
        "case_id",
    )
    _validate_common(
        [str(_scalar(trace["mode"])) for _, trace in loaded],
        "mode",
    )
    intervention = _validate_common(
        [str(_scalar(trace["intervention"])) for _, trace in loaded],
        "intervention",
    )
    if intervention != "all_normal":
        raise ValueError(
            "sensitivity maps are preregistered for the all-normal intervention"
        )
    keys = [
        (
            str(_scalar(trace["resolution"])),
            int(_scalar(trace["call"])),
        )
        for _, trace in loaded
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate sensitivity trace rows: {keys}")
    return sorted(
        loaded,
        key=lambda item: (
            _resolution_sort_key(str(_scalar(item[1]["resolution"]))),
            int(_scalar(item[1]["call"])),
        ),
    )


def render_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    """Render post-lift, branch, block, and decoder difference-norm maps."""

    if (
        args.reference_norm <= 0.0
        or args.transformed_max <= 0.0
        or args.boundary_distance_max <= 0.0
    ):
        raise ValueError("sensitivity scales must be positive")
    traces = _load_sensitivity_traces(args.traces)
    family = str(_scalar(traces[0][1]["family"]))
    case_id = str(_scalar(traces[0][1]["case_id"]))
    mode = str(_scalar(traces[0][1]["mode"]))
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    manifest_path = args.output_stem.with_suffix(".json")
    for path in (png_path, pdf_path, manifest_path):
        _strict_output(path, overwrite=args.overwrite)

    row_count = len(traces)
    column_count = len(SENSITIVITY_FIELDS) + 1
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(20.0, 2.65 * row_count + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    sensitivity_norm = Normalize(vmin=0.0, vmax=args.transformed_max)
    boundary_norm = Normalize(vmin=0.0, vmax=args.boundary_distance_max)
    sensitivity_scatter = None
    boundary_scatter = None
    input_records = []
    for row, (path, trace) in enumerate(traces):
        resolution = str(_scalar(trace["resolution"]))
        original_count = trace["nodes"].shape[0]
        indices, selection = _sensitivity_indices(
            original_count, resolution, args.max_nodes
        )
        nodes = np.asarray(trace["nodes"])[indices]
        marker_size = max(0.18, min(9.0, 5000.0 / nodes.shape[0]))
        field_clipping = {}
        for column, (field, title) in enumerate(SENSITIVITY_FIELDS):
            values = np.asarray(trace[field], dtype=np.float64)[indices]
            transformed = np.log10(1.0 + values / args.reference_norm)
            clipped = float(np.mean(transformed > args.transformed_max))
            field_clipping[field] = clipped
            axis = axes[row, column]
            sensitivity_scatter = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=transformed,
                s=marker_size,
                cmap="magma",
                norm=sensitivity_norm,
                linewidths=0.0,
                rasterized=True,
            )
            axis.text(
                0.015,
                0.025,
                f"clipped: {100.0 * clipped:.2f}%",
                transform=axis.transAxes,
                fontsize=6.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.74,
                    "pad": 1.2,
                },
            )
            if row == 0:
                axis.set_title(title, fontweight="bold")
            _configure_spatial_axis(axis, nodes, row=row, row_count=row_count)

        boundary_axis = axes[row, -1]
        distance = np.asarray(trace["boundary_distance"], dtype=np.float64)[indices]
        boundary_scatter = boundary_axis.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=distance,
            s=marker_size,
            cmap="cividis_r",
            norm=boundary_norm,
            linewidths=0.0,
            rasterized=True,
        )
        node_type = np.asarray(trace["physical_node_type"], dtype=np.int64)[indices]
        for type_index in sorted(FAMILY_TYPE_NAMES[family]):
            selected = node_type == type_index
            if not np.any(selected):
                continue
            boundary_axis.scatter(
                nodes[selected, 0],
                nodes[selected, 1],
                facecolors="none",
                edgecolors=TYPE_COLORS[type_index],
                s=max(marker_size * 4.0, 4.0),
                linewidths=0.45,
                rasterized=True,
            )
        if row == 0:
            boundary_axis.set_title("boundary distance + type", fontweight="bold")
        _configure_spatial_axis(boundary_axis, nodes, row=row, row_count=row_count)
        call = int(_scalar(trace["call"]))
        physical_time = float(_scalar(trace["physical_time"]))
        axes[row, 0].set_ylabel(f"{resolution}\ncall {call}, t={physical_time:.4g}")
        input_records.append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "resolution": resolution,
                "call": call,
                "physical_time": physical_time,
                "original_node_count": int(original_count),
                "plotted_node_count": int(indices.size),
                "selection": selection,
                "field_clipping_fraction": field_clipping,
            }
        )

    if sensitivity_scatter is None or boundary_scatter is None:
        raise AssertionError("at least one sensitivity trace is required")
    figure.colorbar(
        sensitivity_scatter,
        ax=axes[:, :-1].ravel().tolist(),
        location="bottom",
        shrink=0.52,
        pad=0.02,
        label=(
            "log10(1 + activation-difference norm / "
            f"{args.reference_norm:g}); fixed diagnostic scale"
        ),
        extend="max",
    )
    figure.colorbar(
        boundary_scatter,
        ax=axes[:, -1].ravel().tolist(),
        location="bottom",
        shrink=0.82,
        pad=0.02,
        label=f"boundary distance, clipped at {args.boundary_distance_max:g}",
        extend="max",
    )
    axes[0, -1].legend(
        handles=_type_legend(family),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title=f"{family} node types",
        title_fontsize=7,
    )
    figure.suptitle(
        f"{family} case {case_id} | {mode} | correct minus all-normal\n"
        "hidden-state and branch sensitivity norms are diagnostics, "
        "not causal branch shares",
        fontsize=10,
    )
    figure.savefig(png_path, dpi=args.dpi)
    figure.savefig(pdf_path)
    plt.close(figure)

    manifest: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "kind": "node_type_activation_sensitivity_map",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": {
            "file": Path(__file__).name,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "family": family,
        "case_id": case_id,
        "mode": mode,
        "intervention": "correct node types versus all-normal",
        "fields": [field for field, _ in SENSITIVITY_FIELDS],
        "scale_contract": {
            "source": args.scale_source,
            "transform": (
                "log10(1 + per-node Euclidean difference norm / reference_norm)"
            ),
            "reference_norm": float(args.reference_norm),
            "transformed_range": [0.0, float(args.transformed_max)],
            "boundary_distance": [
                0.0,
                float(args.boundary_distance_max),
            ],
            "result_dependent_autoscaling": False,
        },
        "boundary_policy": "frozen; representation intervention only",
        "claim_boundary": {
            "activation_maps": (
                "observed hidden and branch difference norms under an "
                "all-normal intervention"
            ),
            "not_claimed": (
                "not causal branch attribution, not a JVP proof, and not a "
                "training ablation"
            ),
        },
        "inputs": input_records,
        "outputs": {
            png_path.name: _sha256(png_path),
            pdf_path.name: _sha256(pdf_path),
        },
    }
    _write_json(manifest_path, manifest, overwrite=args.overwrite)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "animation":
        render_animation(args)
    else:
        render_sensitivity(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
