#!/usr/bin/env python3
"""Render all-frame D072 rollout comparisons and activation sensitivity maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LogNorm, Normalize

INPUT_SCHEMA = "pcno_boundary_field_evaluation_v1"
OUTPUT_SCHEMA = "pcno_boundary_field_visualization_v1"
KINDS = ("comparison", "intervention")
MODES = ("teacher_forced", "free_rollout")
FIELDS = ("density", "pressure")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    movie = subparsers.add_parser("animation")
    movie.add_argument("--bundle", type=Path, required=True)
    movie.add_argument("--reference-scales", type=Path, required=True)
    movie.add_argument("--output-stem", type=Path, required=True)
    movie.add_argument("--kind", choices=KINDS, required=True)
    movie.add_argument("--mode", choices=MODES, required=True)
    movie.add_argument("--field", choices=FIELDS, required=True)
    movie.add_argument("--gamma", type=float, default=1.4)
    movie.add_argument("--units", required=True)
    movie.add_argument("--fps", type=int, default=6)
    movie.add_argument("--dpi", type=int, default=100)
    movie.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    movie.add_argument("--overwrite", action="store_true")

    sensitivity = subparsers.add_parser("sensitivity")
    sensitivity.add_argument("--trace", type=Path, required=True)
    sensitivity.add_argument("--output-stem", type=Path, required=True)
    sensitivity.add_argument("--max-nodes", type=int, default=25_000)
    sensitivity.add_argument("--dpi", type=int, default=180)
    sensitivity.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files}
    if str(np.asarray(result["schema"]).item()) != INPUT_SCHEMA:
        raise ValueError(f"unexpected input schema in {path}")
    return result


def _strict_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    _strict_output(path, overwrite=overwrite)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def animation_frame_indices(frame_count: int) -> list[int]:
    """Return every stored reference frame, including the initial condition."""

    if frame_count < 2:
        raise ValueError("an animation requires an initial and predicted frame")
    return list(range(frame_count))


def conservative_field(
    state: np.ndarray, *, field: str, gamma: float = 1.4
) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    if value.shape[-1] != 4:
        raise ValueError("conservative states must have four components")
    density = value[..., 0]
    if field == "density":
        return density
    if field != "pressure":
        raise ValueError(f"unsupported field: {field}")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kinetic = 0.5 * (value[..., 1] ** 2 + value[..., 2] ** 2) / density
        return (gamma - 1.0) * (value[..., 3] - kinetic)


def _grid_shape(resolution: str, node_count: int) -> tuple[int, int] | None:
    if resolution == "native_graph":
        return None
    try:
        nx_text, ny_text = resolution.lower().split("x", maxsplit=1)
        nx, ny = int(nx_text), int(ny_text)
    except ValueError as exc:
        raise ValueError(f"invalid resolution label: {resolution}") from exc
    return (nx, ny) if nx * ny == node_count else None


def _semantic_overlay(fields: np.ndarray) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError("semantic collars must have two or three channels")
    colors = np.asarray(
        [
            [0.0, 0.45, 0.70],
            [0.85, 0.33, 0.10],
            [0.0, 0.62, 0.45],
        ]
    )[: values.shape[1]]
    weighted = values @ colors
    amplitude = np.maximum(values.max(axis=-1, keepdims=True), 1.0e-12)
    mixed = weighted / np.maximum(values.sum(axis=-1, keepdims=True), 1.0e-12)
    background = np.ones_like(mixed)
    return background * (1.0 - amplitude) + mixed * amplitude


def _panel_data(
    bundle: dict[str, np.ndarray],
    *,
    kind: str,
    mode: str,
    field: str,
    gamma: float,
) -> tuple[list[str], list[str], list[np.ndarray]]:
    truth = conservative_field(bundle["reference_states"], field=field, gamma=gamma)

    def model(name: str) -> np.ndarray:
        key = f"{mode}_{name}"
        if key not in bundle:
            raise ValueError(f"bundle lacks {key}")
        return conservative_field(bundle[key], field=field, gamma=gamma)

    n0 = model("N0_correct")
    g1 = model("G1_correct")
    s1 = model("S1_correct")
    if kind == "comparison":
        return (
            [
                "Reference",
                "N0: no boundary field",
                "G1: union collar",
                "S1: semantic collars",
                "N0 minus reference",
                "G1 minus reference",
                "S1 minus reference",
                "Semantic collar overlay",
            ],
            ["state", "state", "state", "state", "error", "error", "error", "overlay"],
            [truth, n0, g1, s1, n0 - truth, g1 - truth, s1 - truth],
        )
    s1_zero = model("S1_zero_all")
    return (
        [
            "Reference",
            "S1: correct collars",
            "S1: all collars zero",
            "Correct minus zero",
            "Correct minus reference",
            "Zero minus reference",
            "Correct update",
            "Semantic collar overlay",
        ],
        [
            "state",
            "state",
            "state",
            "difference",
            "error",
            "error",
            "difference",
            "overlay",
        ],
        [
            truth,
            s1,
            s1_zero,
            s1 - s1_zero,
            s1 - truth,
            s1_zero - truth,
            np.concatenate((np.zeros_like(s1[:1]), np.diff(s1, axis=0)), axis=0),
        ],
    )


def _draw_scalar(
    axis: plt.Axes,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    grid: tuple[int, int] | None,
    cmap: str,
    norm: Normalize,
) -> Any:
    if grid is not None:
        nx, ny = grid
        x = nodes[:, 0].reshape(ny, nx)
        y = nodes[:, 1].reshape(ny, nx)
        image = axis.pcolormesh(
            x,
            y,
            values.reshape(ny, nx),
            shading="nearest",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        axis.set_xlim(float(x.min()), float(x.max()))
        axis.set_ylim(float(y.min()), float(y.max()))
        return image
    return axis.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=values,
        s=max(1.0, min(9.0, 20_000.0 / nodes.shape[0])),
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        rasterized=True,
    )


def _draw_overlay(
    axis: plt.Axes,
    nodes: np.ndarray,
    rgb: np.ndarray,
    *,
    grid: tuple[int, int] | None,
) -> Any:
    if grid is not None:
        nx, ny = grid
        extent = (
            float(nodes[:, 0].min()),
            float(nodes[:, 0].max()),
            float(nodes[:, 1].min()),
            float(nodes[:, 1].max()),
        )
        return axis.imshow(
            rgb.reshape(ny, nx, 3),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="auto",
        )
    return axis.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=rgb,
        s=max(1.0, min(9.0, 20_000.0 / nodes.shape[0])),
        linewidths=0.0,
        rasterized=True,
    )


def render_animation(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_npz(args.bundle)
    scales = json.loads(args.reference_scales.read_text(encoding="utf-8"))
    field_scale = scales["fields"][args.field]
    if not bool(scales.get("outcome_independent")):
        raise ValueError("animation scales are not marked outcome-independent")
    nodes = np.asarray(bundle["nodes"], dtype=np.float64)
    fields = np.asarray(bundle["boundary_features"], dtype=np.float64)
    frame_count = int(np.asarray(bundle["requested_frame_count"]).item())
    truth = bundle["reference_states"]
    if truth.shape[0] != frame_count:
        raise ValueError("reference frame count differs from the animation contract")
    if not bool(np.asarray(bundle["all_reference_frames_included"]).item()):
        raise ValueError("bundle does not attest that all reference frames are present")
    frames = animation_frame_indices(frame_count)
    titles, kinds, arrays = _panel_data(
        bundle,
        kind=args.kind,
        mode=args.mode,
        field=args.field,
        gamma=args.gamma,
    )
    if any(array.shape != (frame_count, nodes.shape[0]) for array in arrays):
        raise ValueError("animation state arrays do not match frames and nodes")
    resolution = str(np.asarray(bundle["resolution"]).item())
    grid = _grid_shape(resolution, nodes.shape[0])
    overlay = _semantic_overlay(fields)
    state_norm = Normalize(
        vmin=float(field_scale["minimum"]), vmax=float(field_scale["maximum"])
    )
    error_norm = Normalize(
        vmin=-float(field_scale["error_abs_max"]),
        vmax=float(field_scale["error_abs_max"]),
    )
    difference_norm = Normalize(
        vmin=-float(field_scale["difference_abs_max"]),
        vmax=float(field_scale["difference_abs_max"]),
    )
    output_path = args.output_stem.with_suffix(f".{args.format}")
    final_png = args.output_stem.with_name(args.output_stem.name + "_final.png")
    final_pdf = args.output_stem.with_name(args.output_stem.name + "_final.pdf")
    manifest_path = args.output_stem.with_name(args.output_stem.name + "_manifest.json")
    for path in (output_path, final_png, final_pdf, manifest_path):
        _strict_output(path, overwrite=args.overwrite)

    figure, axes = plt.subplots(2, 4, figsize=(12.4, 5.7), constrained_layout=True)
    axes_flat = list(axes.flat)

    def draw(frame: int) -> list[Any]:
        artists: list[Any] = []
        for index, (axis, title, panel_kind) in enumerate(
            zip(axes_flat, titles, kinds, strict=True)
        ):
            axis.clear()
            axis.set_title(title)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            if panel_kind == "overlay":
                artist = _draw_overlay(axis, nodes, overlay, grid=grid)
            else:
                value = arrays[index][frame]
                valid = np.isfinite(value)
                if not np.any(valid):
                    axis.text(
                        0.5,
                        0.5,
                        "rollout stopped\n(no fabricated state)",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                    artist = axis.scatter([], [])
                else:
                    norm = (
                        state_norm
                        if panel_kind == "state"
                        else error_norm if panel_kind == "error" else difference_norm
                    )
                    artist = _draw_scalar(
                        axis,
                        nodes[valid],
                        value[valid],
                        grid=grid if np.all(valid) else None,
                        cmap="viridis" if panel_kind == "state" else "coolwarm",
                        norm=norm,
                    )
            artists.append(artist)
        time_value = float(bundle["physical_times"][frame])
        figure.suptitle(
            f"{np.asarray(bundle['family']).item()} | {np.asarray(bundle['case_id']).item()} | "
            f"{args.mode.replace('_', ' ')} | {args.field} | frame {frame}/{frame_count - 1} | "
            f"t={time_value:.6g} {args.units}",
            fontsize=11,
        )
        return artists

    draw(frames[-1])
    figure.savefig(final_png, dpi=max(args.dpi, 160))
    figure.savefig(final_pdf)
    movie = animation.FuncAnimation(
        figure,
        draw,
        frames=frames,
        interval=1000.0 / args.fps,
        blit=False,
        repeat=False,
    )
    if args.format == "mp4":
        writer: Any = animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=3500,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        writer = animation.PillowWriter(fps=args.fps)
    movie.save(output_path, writer=writer, dpi=args.dpi)
    plt.close(figure)

    manifest = {
        "schema": OUTPUT_SCHEMA,
        "kind": "all_frame_rollout_animation",
        "comparison_kind": args.kind,
        "mode": args.mode,
        "field": args.field,
        "source_bundle": str(args.bundle),
        "source_bundle_sha256": _sha256(args.bundle),
        "reference_scales": str(args.reference_scales),
        "reference_scales_sha256": _sha256(args.reference_scales),
        "scale_source": field_scale["scale_source"],
        "outcome_independent_scales": True,
        "frame_indices": frames,
        "frame_count": len(frames),
        "expected_frame_count": frame_count,
        "all_frames_included": frames == list(range(frame_count)),
        "post_failure_policy": str(np.asarray(bundle["post_failure_values"]).item()),
        "outputs": {
            output_path.name: _sha256(output_path),
            final_png.name: _sha256(final_png),
            final_pdf.name: _sha256(final_pdf),
        },
    }
    _write_json(manifest_path, manifest, overwrite=args.overwrite)
    return manifest


SENSITIVITY_FIELDS = (
    ("analytical_boundary_lift_norm", "Analytical $W_B B$"),
    ("post_lift_difference_norm", "Post-lift difference"),
    ("block_0_pointwise_difference_norm", "Block 0 pointwise"),
    ("block_0_integral_difference_norm", "Block 0 integral"),
    ("block_0_differential_difference_norm", "Block 0 differential"),
    ("block_0_output_difference_norm", "Block 0 output"),
    ("block_3_pointwise_difference_norm", "Block 3 pointwise"),
    ("block_3_integral_difference_norm", "Block 3 integral"),
    ("block_3_differential_difference_norm", "Block 3 differential"),
    ("block_3_output_difference_norm", "Block 3 output"),
    ("decoder_hidden_difference_norm", "Decoder hidden"),
    ("normalized_residual_difference_norm", "Decoded residual"),
)


def render_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    trace = _load_npz(args.trace)
    missing = [name for name, _ in SENSITIVITY_FIELDS if name not in trace]
    if missing:
        raise ValueError(f"trace lacks sensitivity arrays: {missing}")
    nodes = np.asarray(trace["nodes"], dtype=np.float64)
    if nodes.shape[0] > args.max_nodes:
        stride = math.ceil(nodes.shape[0] / args.max_nodes)
        indices = np.arange(0, nodes.shape[0], stride, dtype=np.int64)
    else:
        indices = np.arange(nodes.shape[0], dtype=np.int64)
    values = [
        np.asarray(trace[name], dtype=np.float64)[indices]
        for name, _ in SENSITIVITY_FIELDS
    ]
    positive = np.concatenate([value[value > 0.0] for value in values])
    maximum = max(float(max(value.max() for value in values)), 1.0e-30)
    minimum = max(
        float(positive.min()) if positive.size else maximum * 1.0e-6,
        maximum * 1.0e-6,
    )
    norm = LogNorm(vmin=minimum, vmax=maximum)
    output_png = args.output_stem.with_suffix(".png")
    output_pdf = args.output_stem.with_suffix(".pdf")
    manifest_path = args.output_stem.with_name(args.output_stem.name + "_manifest.json")
    for path in (output_png, output_pdf, manifest_path):
        _strict_output(path, overwrite=args.overwrite)
    figure, axes = plt.subplots(3, 4, figsize=(12.4, 8.0), constrained_layout=True)
    last_artist = None
    for axis, (_, title), value in zip(
        axes.flat, SENSITIVITY_FIELDS, values, strict=True
    ):
        last_artist = axis.scatter(
            nodes[indices, 0],
            nodes[indices, 1],
            c=np.maximum(value, minimum),
            s=max(1.0, min(9.0, 20_000.0 / indices.size)),
            cmap="magma",
            norm=norm,
            linewidths=0.0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if last_artist is None:
        raise AssertionError("no sensitivity panel was rendered")
    figure.colorbar(
        last_artist, ax=list(axes.flat), shrink=0.72, label="latent-vector L2 norm"
    )
    figure.suptitle(
        f"{np.asarray(trace['family']).item()} | {np.asarray(trace['case_id']).item()} | "
        f"{np.asarray(trace['mode']).item()} | call {int(np.asarray(trace['call']).item())} | "
        "correct collars minus all-zero intervention",
        fontsize=11,
    )
    figure.savefig(output_png, dpi=args.dpi)
    figure.savefig(output_pdf)
    plt.close(figure)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "kind": "boundary_field_sensitivity_map",
        "source_trace": str(args.trace),
        "source_trace_sha256": _sha256(args.trace),
        "node_count": int(indices.size),
        "scale": {
            "kind": "shared_log_norm_across_all_panels",
            "minimum": minimum,
            "maximum": maximum,
            "source": "exact extrema of this frozen diagnostic trace",
        },
        "causal_language": "diagnostic propagation norms, not causal branch shares",
        "outputs": {
            output_png.name: _sha256(output_png),
            output_pdf.name: _sha256(output_pdf),
        },
    }
    _write_json(manifest_path, manifest, overwrite=args.overwrite)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "animation":
        render_animation(args)
    else:
        render_sensitivity(args)


if __name__ == "__main__":
    main()
