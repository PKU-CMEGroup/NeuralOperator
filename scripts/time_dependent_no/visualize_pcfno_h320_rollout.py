#!/usr/bin/env python3
"""Render fixed-scale, reference-free PCFNO H320 rollout diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_KEYS,
    ANIMATION_SCHEMA,
    CHECKPOINT_SHA256,
    PRODUCTION_STEPS,
    SCHEMA,
    TRUTH_STEPS,
    WORKING_ID,
    build_final_hash_manifest,
    conservative_to_primitive_numpy,
    internal_energy_numpy,
    load_npz,
    sha256_file,
    verify_final_hash_manifest,
    write_json,
)
from scripts.time_dependent_no.visualize_pcno_inadmissibility_continuation import (
    build_continuous_field_map,
    rasterize_field,
    saturation_fraction,
)

VISUAL_SCHEMA = "w26_l1_pcfno_h320_visualization_v1"
COLORS = {
    "187": "#0072B2",
    "54": "#D55E00",
    "227": "#009E73",
    "233": "#CC79A7",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--raster-width", type=int, default=360)
    parser.add_argument("--ffmpeg-path", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 50 or args.raster_width < 256:
        raise ValueError("fps, dpi, and raster-width must be positive and meaningful")
    return args


def _scalar(value: np.ndarray) -> Any:
    return np.asarray(value).item()


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    arrays = load_npz(path)
    required = {
        "schema",
        "working_id",
        "trajectory",
        "checkpoint_sha256",
        "requested_horizon",
        "recorded_call_count",
        "termination_call",
        "physical_times",
        "positions",
        "node_type",
        "deployed_states_conservative",
        "state_scale",
        "presentation_scales_json",
        "event_admissible",
        "event_bounded",
        "event_finite",
        "common_amplitude_ratio",
        "common_scaled_rms_ratio",
        "minimum_internal_energy",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"animation bundle lacks required arrays: {missing}")
    if str(_scalar(arrays["schema"])) != ANIMATION_SCHEMA:
        raise ValueError(f"unsupported animation bundle schema: {path}")
    if str(_scalar(arrays["working_id"])) != WORKING_ID:
        raise ValueError(f"animation bundle working ID changed: {path}")
    if str(_scalar(arrays["checkpoint_sha256"])) != CHECKPOINT_SHA256:
        raise ValueError(f"animation bundle checkpoint changed: {path}")
    states = np.asarray(arrays["deployed_states_conservative"])
    recorded = int(_scalar(arrays["recorded_call_count"]))
    if states.ndim != 3 or states.shape[-1] != 4 or states.shape[0] != recorded + 1:
        raise ValueError("animation bundle must retain H0 and every recorded call")
    if np.asarray(arrays["positions"]).shape != (states.shape[1], 2):
        raise ValueError("animation positions do not match the retained node axis")
    for name in (
        "physical_times",
        "event_admissible",
        "event_bounded",
        "event_finite",
        "common_amplitude_ratio",
        "common_scaled_rms_ratio",
        "minimum_internal_energy",
    ):
        if np.asarray(arrays[name]).shape != (recorded + 1,):
            raise ValueError(f"animation array {name} does not match the frame count")
    return arrays


def invalid_categories(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    primitive = conservative_to_primitive_numpy(value, gamma=gamma)
    internal = internal_energy_numpy(value)
    nonfinite = (
        ~np.isfinite(value).all(axis=-1)
        | ~np.isfinite(primitive).all(axis=-1)
        | ~np.isfinite(internal)
    )
    density = ~nonfinite & (primitive[..., 0] <= 0.0)
    energy = ~nonfinite & ~density & ((internal <= 0.0) | (primitive[..., 3] <= 0.0))
    category = np.zeros(value.shape[:-1], dtype=np.int8)
    category[density] = 1
    category[energy] = 2
    category[nonfinite] = 3
    return category


def animation_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    states = np.asarray(arrays["deployed_states_conservative"], dtype=np.float64)
    state_scale = np.asarray(arrays["state_scale"], dtype=np.float64)
    if state_scale.shape != (4,) or not np.isfinite(state_scale).all():
        raise ValueError("bundle state scale must contain four finite components")
    primitive = conservative_to_primitive_numpy(states, gamma=1.4)
    internal = internal_energy_numpy(states)
    pressure_increment = np.zeros_like(primitive[..., 3])
    pressure_increment[1:] = np.diff(primitive[..., 3], axis=0)
    scaled_increment = np.zeros_like(primitive[..., 3])
    scaled_increment[1:] = np.linalg.norm(
        np.diff(states, axis=0) / state_scale[None, None, :], axis=-1
    )
    return {
        "density": primitive[..., 0],
        "pressure": primitive[..., 3],
        "internal_energy": internal,
        "pressure_increment": pressure_increment,
        "scaled_increment": scaled_increment,
        "invalid_category": invalid_categories(states),
    }


def _first_false(values: np.ndarray) -> int | None:
    failed = np.flatnonzero(~np.asarray(values, dtype=bool))
    return None if failed.size == 0 else int(failed[0])


def _metric_text(value: float) -> str:
    return "unavailable" if not math.isfinite(float(value)) else f"{float(value):.3g}"


def render_animation(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    fps: int,
    dpi: int,
    raster_width: int,
) -> dict[str, Any]:
    key = str(_scalar(arrays["trajectory"]))
    states = np.asarray(arrays["deployed_states_conservative"], dtype=np.float64)
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    times = np.asarray(arrays["physical_times"], dtype=np.float64)
    admissible = np.asarray(arrays["event_admissible"], dtype=bool)
    bounded = np.asarray(arrays["event_bounded"], dtype=bool)
    finite = np.asarray(arrays["event_finite"], dtype=bool)
    amplitude = np.asarray(arrays["common_amplitude_ratio"], dtype=np.float64)
    rms = np.asarray(arrays["common_scaled_rms_ratio"], dtype=np.float64)
    minimum_internal = np.asarray(arrays["minimum_internal_energy"], dtype=np.float64)
    scales = json.loads(str(_scalar(arrays["presentation_scales_json"])))
    fields = animation_fields(arrays)
    field_map = build_continuous_field_map(positions, raster_width=raster_width)
    extent = field_map["extent"]

    panels = (
        ("Density", "density", scales["density_min"], scales["density_max"], "viridis"),
        (
            "Pressure",
            "pressure",
            scales["pressure_min"],
            scales["pressure_max"],
            "viridis",
        ),
        (
            "Internal energy",
            "internal_energy",
            scales["internal_energy_min"],
            scales["internal_energy_max"],
            "viridis",
        ),
        (
            "Pressure increment",
            "pressure_increment",
            -scales["pressure_increment_abs_max"],
            scales["pressure_increment_abs_max"],
            "coolwarm",
        ),
        (
            "Checkpoint-scaled state increment",
            "scaled_increment",
            0.0,
            scales["scaled_increment_max"],
            "magma",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.2), constrained_layout=True)
    images = []
    for axis, (title, _, lower, upper, cmap) in zip(axes.flat[:5], panels, strict=True):
        if lower < 0.0 < upper and cmap == "coolwarm":
            norm = TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
            image = axis.imshow(
                np.full(field_map["shape"], np.nan),
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                np.full(field_map["shape"], np.nan),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.72)
        images.append(image)

    invalid_cmap = ListedColormap(("#F4F4F4", "#E69F00", "#D55E00", "#000000"))
    invalid_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), invalid_cmap.N)
    invalid_scatter = axes.flat[5].scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.zeros(positions.shape[0]),
        s=1.4,
        linewidths=0.0,
        cmap=invalid_cmap,
        norm=invalid_norm,
        rasterized=True,
    )
    axes.flat[5].set_title("Admissibility category")
    axes.flat[5].set_aspect("equal")
    axes.flat[5].set_xticks([])
    axes.flat[5].set_yticks([])
    category_bar = fig.colorbar(
        invalid_scatter, ax=axes.flat[5], shrink=0.72, ticks=(0, 1, 2, 3)
    )
    category_bar.ax.set_yticklabels(("valid", "density", "energy/p", "nonfinite"))
    status = fig.text(0.5, 0.008, "", ha="center", va="bottom", fontsize=8.5)
    first_inadmissible = _first_false(admissible)

    def update(frame: int) -> list[Any]:
        artists: list[Any] = []
        for image, (_, name, _, _, _) in zip(images, panels, strict=True):
            image.set_data(
                np.ma.masked_invalid(rasterize_field(fields[name][frame], field_map))
            )
            artists.append(image)
        categories = fields["invalid_category"][frame]
        invalid_scatter.set_array(categories.astype(np.float64, copy=False))
        artists.append(invalid_scatter)
        if frame <= TRUTH_STEPS:
            phase = "bitwise-checked H79 prefix"
        elif first_inadmissible is not None and frame >= first_inadmissible:
            phase = "reference-free finite-invalid recurrence"
        else:
            phase = "reference-free recurrence"
        status.set_text(
            f"case {key} | call {frame}/{int(_scalar(arrays['requested_horizon']))} | "
            f"t={times[frame]:.3f} | {phase}\n"
            f"events A/B/F={int(admissible[frame])}/{int(bounded[frame])}/"
            f"{int(finite[frame])} | amplitude={_metric_text(amplitude[frame])} | "
            f"RMS={_metric_text(rms[frame])} | min internal energy="
            f"{_metric_text(minimum_internal[frame])}"
        )
        fig.suptitle("PCFNO autonomous rollout (no reference after call 79)")
        artists.append(status)
        return artists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=4500,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    elif output_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError("animation output must end in .mp4 or .gif")
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=states.shape[0],
        interval=1000 / fps,
        blit=False,
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)

    saturation = {}
    for _, name, lower, upper, _ in panels:
        saturation[name] = saturation_fraction(fields[name], lower, upper)
    return {
        "trajectory": key,
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "rendered_frame_count": int(states.shape[0]),
        "all_retained_frames_rendered": True,
        "first_inadmissible_frame": first_inadmissible,
        "terminal_call": int(states.shape[0] - 1),
        "saturation_fraction": saturation,
        "raster": {
            "width": int(field_map["shape"][1]),
            "height": int(field_map["shape"][0]),
            "source_node_count": int(field_map["source_node_count"]),
            "masked_triangle_count": int(field_map["masked_triangle_count"]),
        },
        "encoder": (
            {
                "writer": "matplotlib.FFMpegWriter",
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "bitrate_kbps": 4500,
            }
            if output_path.suffix.lower() == ".mp4"
            else {"writer": "matplotlib.PillowWriter"}
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nullable_float(value: str) -> float:
    return float("nan") if value == "" else float(value)


def render_survival(input_dir: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(input_dir / "survival.csv")
    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    labels = {
        "admissible": "Admissible",
        "bounded": "Within H79 envelope",
        "finite": "Finite",
    }
    colors = {"admissible": "#D55E00", "bounded": "#0072B2", "finite": "#009E73"}
    for event in ("admissible", "bounded", "finite"):
        selected = [row for row in rows if row["event"] == event]
        calls = [int(row["call"]) for row in selected]
        survival = [float(row["kaplan_meier_survival"]) for row in selected]
        axis.step(
            calls, survival, where="post", label=labels[event], color=colors[event]
        )
    axis.axvline(
        TRUTH_STEPS,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="last truth-aligned call",
    )
    axis.set(
        xlabel="Autonomous model call",
        ylabel="Accepted-prefix survival",
        xlim=(0, PRODUCTION_STEPS),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    axis.set_title("PCFNO H320 descriptive event survival (30 open-validation cases)")
    outputs = [
        output_dir / "pcfno_h320_event_survival.pdf",
        output_dir / "pcfno_h320_event_survival.png",
    ]
    for path in outputs:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def render_selected_curves(input_dir: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(input_dir / "call_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for key in ANIMATION_KEYS:
        selected = [row for row in rows if row["trajectory"] == key]
        calls = np.asarray([int(row["call"]) for row in selected])
        minimum_internal = np.asarray(
            [_nullable_float(row["deployed_min_internal_energy"]) for row in selected]
        )
        amplitude = np.asarray(
            [_nullable_float(row["common_amplitude_ratio"]) for row in selected]
        )
        rms = np.asarray(
            [_nullable_float(row["common_scaled_rms_ratio"]) for row in selected]
        )
        color = COLORS[key]
        axes[0].plot(calls, minimum_internal, color=color, label=f"case {key}")
        axes[1].plot(calls, amplitude, color=color, label=f"case {key} amplitude")
        axes[1].plot(calls, rms, color=color, linestyle="--", label=f"case {key} RMS")
    for axis in axes:
        axis.axvline(TRUTH_STEPS, color="#666666", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.22)
        axis.set_xlabel("Autonomous model call")
        axis.set_xlim(0, PRODUCTION_STEPS)
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_yscale("symlog", linthresh=1.0e-3)
    axes[0].set_ylabel("Minimum internal energy")
    axes[0].set_title("Admissibility precursor")
    axes[1].axhline(1.0, color="#222222", linewidth=0.8)
    axes[1].axhline(10.0, color="#888888", linewidth=0.8, linestyle="--")
    axes[1].axhline(100.0, color="#888888", linewidth=0.8, linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Ratio to fixed H1-H79 envelope")
    axes[1].set_title("Reference-envelope departure")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False)
    axes[1].legend(ncol=2, frameon=False, fontsize=7)
    fig.suptitle(
        "Selected PCFNO H320 trajectories; dashed vertical line ends truth support"
    )
    outputs = [
        output_dir / "pcfno_h320_selected_curves.pdf",
        output_dir / "pcfno_h320_selected_curves.png",
    ]
    for path in outputs:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    input_summary = json.loads(
        (args.input_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        input_summary.get("schema") != SCHEMA
        or input_summary.get("status") != "completed"
    ):
        raise ValueError("input is not a completed registered H320 result")
    if (
        input_summary.get("mode") != "production"
        or int(input_summary["requested_horizon"]) != PRODUCTION_STEPS
    ):
        raise ValueError("visualization requires the registered production H320 result")
    manifest_path = args.input_dir / "final_hash_manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_verification = verify_final_hash_manifest(args.input_dir, input_manifest)
    if args.ffmpeg_path is not None:
        if not args.ffmpeg_path.is_file():
            raise FileNotFoundError(args.ffmpeg_path)
        matplotlib.rcParams["animation.ffmpeg_path"] = str(args.ffmpeg_path.resolve())

    bundles = {}
    for key in ANIMATION_KEYS:
        bundle = load_bundle(args.input_dir / f"trajectory_{key}.npz")
        if str(_scalar(bundle["trajectory"])) != key:
            raise ValueError(f"trajectory bundle identity changed for {key}")
        if int(_scalar(bundle["requested_horizon"])) != PRODUCTION_STEPS:
            raise ValueError(f"trajectory {key} was not requested through H320")
        bundles[key] = bundle

    args.output_dir.mkdir(parents=True)
    extension = f".{args.format}"
    animation_records = {}
    for key in ANIMATION_KEYS:
        output_path = args.output_dir / f"pcfno_h320_trajectory_{key}{extension}"
        animation_records[key] = render_animation(
            bundles[key],
            output_path,
            fps=args.fps,
            dpi=args.dpi,
            raster_width=args.raster_width,
        )
        print(json.dumps(animation_records[key], sort_keys=True), flush=True)

    figure_paths = [
        *render_survival(args.input_dir, args.output_dir),
        *render_selected_curves(args.input_dir, args.output_dir),
    ]
    figure_records = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in figure_paths
    ]
    summary = {
        "schema": VISUAL_SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "input_result_manifest": {
            "path": "final_hash_manifest.json",
            "sha256": sha256_file(manifest_path),
            **input_verification,
        },
        "format": args.format,
        "fps": args.fps,
        "dpi": args.dpi,
        "raster_width": args.raster_width,
        "encoder_executable": (
            None if args.ffmpeg_path is None else args.ffmpeg_path.name
        ),
        "animations": animation_records,
        "figures": figure_records,
        "all_retained_frames_rendered": all(
            record["all_retained_frames_rendered"]
            for record in animation_records.values()
        ),
        "reference_panels_rendered": False,
        "claim_boundary": (
            "H80-H320 panels are descriptive PCFNO recurrence only; no reference "
            "accuracy, physical-validity, conservation, or asymptotic-stability claim"
        ),
    }
    write_json(args.output_dir / "visualization_summary.json", summary)
    artifact_names = [
        *[record["path"] for record in animation_records.values()],
        *[record["path"] for record in figure_records],
        "visualization_summary.json",
    ]
    manifest = build_final_hash_manifest(args.output_dir, artifact_names)
    write_json(args.output_dir / "final_hash_manifest.json", manifest)
    verify_final_hash_manifest(args.output_dir, manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
