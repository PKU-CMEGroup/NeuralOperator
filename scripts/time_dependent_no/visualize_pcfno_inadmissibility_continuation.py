#!/usr/bin/env python3
"""Render PCFNO invalid-continuation animations and pre-failure diagnostics."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcfno_inadmissibility_continuation import (
    NUM_STEPS,
    SCHEMA,
    TRAJECTORY_KEYS,
    conservative_fields,
    failure_location,
    invalid_node_mask,
    sha256_file,
    write_json,
)

VISUAL_SCHEMA = "w26_l2_pcfno_inadmissibility_visualization_v1"
COLORS = {"54": "#0072B2", "227": "#D55E00", "233": "#009E73"}

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
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 50:
        raise ValueError("fps and dpi must be positive and meaningful")
    return args


def _scalar(value: np.ndarray) -> Any:
    return np.asarray(value).item()


def load_case(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if str(_scalar(arrays["schema"])) != SCHEMA:
        raise ValueError(f"unsupported continuation bundle: {path}")
    required = {
        "reference_states_conservative",
        "deployed_states_conservative",
        "positions",
        "node_type",
        "physical_times",
        "failure_call",
        "last_retained_call",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"bundle lacks required arrays: {missing}")
    reference = arrays["reference_states_conservative"]
    prediction = arrays["deployed_states_conservative"]
    if reference.shape != prediction.shape or reference.shape[0] != NUM_STEPS + 1:
        raise ValueError(
            "bundle must retain aligned reference/prediction slots through H79"
        )
    return arrays


def animation_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    reference = np.asarray(arrays["reference_states_conservative"], dtype=np.float64)
    prediction = np.asarray(arrays["deployed_states_conservative"], dtype=np.float64)
    reference_pressure = conservative_fields(reference)["pressure"]
    prediction_pressure = conservative_fields(prediction)["pressure"]
    true_residual = np.diff(reference_pressure, axis=0)
    predicted_residual = np.diff(prediction_pressure, axis=0)
    return {
        "reference_pressure": reference_pressure,
        "prediction_pressure": prediction_pressure,
        "pressure_error": prediction_pressure - reference_pressure,
        "true_pressure_residual": true_residual,
        "predicted_pressure_residual": predicted_residual,
        "pressure_residual_error": predicted_residual - true_residual,
    }


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("visualized field has no finite value")
    return float(np.quantile(finite, quantile))


def field_scales(fields: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    reference = fields["reference_pressure"]
    pressure_min = _finite_quantile(reference, 0.005)
    pressure_max = _finite_quantile(reference, 0.995)
    if pressure_max <= pressure_min:
        padding = max(abs(pressure_min), 1.0) * 1.0e-6
        pressure_min -= padding
        pressure_max += padding
    pressure_range = max(pressure_max - pressure_min, np.finfo(float).eps)

    def symmetric(name: str, floor: float) -> tuple[float, float]:
        limit = max(_finite_quantile(np.abs(fields[name]), 0.995), floor)
        return (-limit, limit)

    return {
        "pressure": (pressure_min, pressure_max),
        "state_error": symmetric("pressure_error", 0.02 * pressure_range),
        "residual": symmetric("true_pressure_residual", 0.01 * pressure_range),
        "residual_error": symmetric("pressure_residual_error", 0.02 * pressure_range),
    }


def _saturation_fraction(value: np.ndarray, limits: tuple[float, float]) -> float:
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return 0.0
    return float(np.mean((finite < limits[0]) | (finite > limits[1])))


def render_animation(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    key = str(_scalar(arrays["trajectory_key"]))
    failure_call = int(_scalar(arrays["failure_call"]))
    last_call = int(_scalar(arrays["last_retained_call"]))
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    times = np.asarray(arrays["physical_times"], dtype=np.float64)
    prediction = np.asarray(arrays["deployed_states_conservative"], dtype=np.float64)
    fields = animation_fields(arrays)
    scales = field_scales(fields)
    panels = (
        ("reference pressure", "reference_pressure", "pressure"),
        ("PCFNO pressure", "prediction_pressure", "pressure"),
        ("pressure state error", "pressure_error", "state_error"),
        ("true pressure residual", "true_pressure_residual", "residual"),
        ("PCFNO pressure residual", "predicted_pressure_residual", "residual"),
        ("pressure residual error", "pressure_residual_error", "residual_error"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), constrained_layout=True)
    scatters = []
    invalid_markers = []
    for axis, (title, _, scale_name) in zip(axes.flat, panels, strict=True):
        vmin, vmax = scales[scale_name]
        cmap = "viridis" if scale_name == "pressure" else "coolwarm"
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=np.zeros(positions.shape[0]),
            s=0.55,
            linewidths=0.0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        invalid = axis.scatter(
            [], [], s=7.0, facecolors="none", edgecolors="#CC0000", linewidths=0.5
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        scatters.append(scatter)
        invalid_markers.append(invalid)
    fig.colorbar(
        scatters[0], ax=[axes[0, 0], axes[0, 1]], shrink=0.72, label="pressure"
    )
    fig.colorbar(scatters[2], ax=axes[0, 2], shrink=0.72, label="pressure error")
    fig.colorbar(
        scatters[3],
        ax=[axes[1, 0], axes[1, 1]],
        shrink=0.72,
        label="pressure residual",
    )
    fig.colorbar(scatters[5], ax=axes[1, 2], shrink=0.72, label="residual error")
    status = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)

    def panel_value(name: str, call: int) -> np.ndarray:
        value = fields[name]
        if "residual" in name:
            return value[call - 1]
        return value[call]

    def update(frame: int) -> list[Any]:
        call = frame + 1
        invalid = invalid_node_mask(prediction[call])
        invalid_positions = positions[invalid]
        artists: list[Any] = []
        for scatter, marker, (_, name, _) in zip(
            scatters, invalid_markers, panels, strict=True
        ):
            scatter.set_array(np.ma.masked_invalid(panel_value(name, call)))
            if (
                name
                in {
                    "prediction_pressure",
                    "pressure_error",
                    "predicted_pressure_residual",
                    "pressure_residual_error",
                }
                and invalid_positions.size
            ):
                marker.set_offsets(invalid_positions)
            else:
                marker.set_offsets(np.empty((0, 2)))
            artists.extend((scatter, marker))
        phase = (
            "last admissible input"
            if call == failure_call - 1
            else "first invalid deployed proposal"
            if call == failure_call
            else "invalid-state continuation"
            if call > failure_call
            else "strict valid prefix"
        )
        status.set_text(
            f"trajectory {key} | call {call}/{NUM_STEPS} | t={times[call]:.3f} | "
            f"{phase} | invalid nodes={int(np.count_nonzero(invalid))}"
        )
        fig.suptitle("PCFNO continuation after finite Euler inadmissibility")
        artists.append(status)
        return artists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif output_path.suffix.lower() == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]
        )
    else:
        raise ValueError("animation output must end in .gif or .mp4")
    movie = animation.FuncAnimation(
        fig, update, frames=last_call, interval=1000 / fps, blit=False
    )
    movie.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)
    saturation = {
        "pressure": max(
            _saturation_fraction(fields[name], scales["pressure"])
            for name in ("reference_pressure", "prediction_pressure")
        ),
        "state_error": _saturation_fraction(
            fields["pressure_error"], scales["state_error"]
        ),
        "residual": max(
            _saturation_fraction(fields[name], scales["residual"])
            for name in ("true_pressure_residual", "predicted_pressure_residual")
        ),
        "residual_error": _saturation_fraction(
            fields["pressure_residual_error"], scales["residual_error"]
        ),
    }
    return {
        "trajectory": key,
        "frames": last_call,
        "failure_call": failure_call,
        "fps": fps,
        "scales": {name: list(values) for name, values in scales.items()},
        "maximum_saturation_fraction": saturation,
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "rendering": "all graph nodes, no temporal subsampling, fixed scales over time",
    }


def _save_figure(fig: Any, base: Path) -> list[Path]:
    outputs = []
    for suffix in (".pdf", ".png"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _global_prefailure_scales(
    cases: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, tuple[float, float]]:
    reference_pressure = []
    state_error = []
    true_residual = []
    predicted_residual = []
    residual_error = []
    failed_internal = []
    for arrays in cases.values():
        call = int(_scalar(arrays["failure_call"]))
        fields = animation_fields(arrays)
        reference_pressure.append(fields["reference_pressure"][call - 1])
        state_error.append(fields["pressure_error"][call - 1])
        true_residual.append(fields["true_pressure_residual"][call - 1])
        predicted_residual.append(fields["predicted_pressure_residual"][call - 1])
        residual_error.append(fields["pressure_residual_error"][call - 1])
        failed_internal.append(
            conservative_fields(arrays["deployed_states_conservative"][call])[
                "internal_energy"
            ]
        )
    pressure_flat = np.concatenate(reference_pressure)
    pmin = _finite_quantile(pressure_flat, 0.005)
    pmax = _finite_quantile(pressure_flat, 0.995)
    if pmax <= pmin:
        padding = max(abs(pmin), 1.0) * 1.0e-6
        pmin -= padding
        pmax += padding
    prange = max(pmax - pmin, np.finfo(float).eps)

    def symmetric(values: Sequence[np.ndarray], floor: float) -> tuple[float, float]:
        limit = max(_finite_quantile(np.abs(np.concatenate(values)), 0.995), floor)
        return (-limit, limit)

    internal_flat = np.concatenate(failed_internal)
    internal_min = min(_finite_quantile(internal_flat, 0.001), -1.0e-6)
    internal_max = max(_finite_quantile(internal_flat, 0.995), 1.0e-6)
    return {
        "pressure": (pmin, pmax),
        "state_error": symmetric(state_error, 0.02 * prange),
        "residual": symmetric(true_residual + predicted_residual, 0.01 * prange),
        "residual_error": symmetric(residual_error, 0.02 * prange),
        "internal_energy": (internal_min, internal_max),
    }


def _field_axis(
    axis: Any,
    positions: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    limits: tuple[float, float],
    cmap: str,
    invalid_position: np.ndarray | None = None,
) -> Any:
    scatter = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.asarray(values, dtype=np.float64),
        s=0.55,
        linewidths=0.0,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        rasterized=True,
    )
    if invalid_position is not None:
        axis.plot(
            invalid_position[0],
            invalid_position[1],
            marker="X",
            markersize=6,
            markerfacecolor="#F0E442",
            markeredgecolor="black",
            linestyle="none",
        )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    return scatter


def render_prefailure_figures(
    cases: Mapping[str, Mapping[str, np.ndarray]], output_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    scales = _global_prefailure_scales(cases)
    state_fig, state_axes = plt.subplots(
        len(TRAJECTORY_KEYS), 4, figsize=(14.0, 7.2), constrained_layout=True
    )
    residual_fig, residual_axes = plt.subplots(
        len(TRAJECTORY_KEYS), 3, figsize=(11.2, 7.2), constrained_layout=True
    )
    state_handles: list[Any] = []
    residual_handles: list[Any] = []
    for row, key in enumerate(TRAJECTORY_KEYS):
        arrays = cases[key]
        call = int(_scalar(arrays["failure_call"]))
        node = int(
            failure_location(
                arrays["deployed_states_conservative"][call],
                arrays["node_type"],
                gamma=1.4,
            )["node"]
        )
        positions = np.asarray(arrays["positions"], dtype=np.float64)
        fields = animation_fields(arrays)
        invalid_position = positions[node]
        reference_before = fields["reference_pressure"][call - 1]
        prediction_before = fields["prediction_pressure"][call - 1]
        state_error = fields["pressure_error"][call - 1]
        failed_internal = conservative_fields(
            arrays["deployed_states_conservative"][call]
        )["internal_energy"]
        state_values = (
            (reference_before, "reference pressure", scales["pressure"], "viridis"),
            (prediction_before, "PCFNO pressure", scales["pressure"], "viridis"),
            (state_error, "pressure error", scales["state_error"], "coolwarm"),
            (
                failed_internal,
                "failed-proposal internal energy",
                scales["internal_energy"],
                "coolwarm",
            ),
        )
        for column, (values, title, limits, cmap) in enumerate(state_values):
            handle = _field_axis(
                state_axes[row, column],
                positions,
                values,
                title=(title if row == 0 else ""),
                limits=limits,
                cmap=cmap,
                invalid_position=invalid_position if column == 3 else None,
            )
            if column == 0:
                state_axes[row, column].set_ylabel(
                    f"trajectory {key}\ncall {call - 1} → {call}"
                )
            if row == 0:
                state_handles.append(handle)
        residual_values = (
            (
                fields["true_pressure_residual"][call - 1],
                "true pressure residual",
                scales["residual"],
            ),
            (
                fields["predicted_pressure_residual"][call - 1],
                "PCFNO pressure residual",
                scales["residual"],
            ),
            (
                fields["pressure_residual_error"][call - 1],
                "pressure residual error",
                scales["residual_error"],
            ),
        )
        for column, (values, title, limits) in enumerate(residual_values):
            handle = _field_axis(
                residual_axes[row, column],
                positions,
                values,
                title=(title if row == 0 else ""),
                limits=limits,
                cmap="coolwarm",
                invalid_position=invalid_position if column > 0 else None,
            )
            if column == 0:
                residual_axes[row, column].set_ylabel(
                    f"trajectory {key}\nfailure call {call}"
                )
            if row == 0:
                residual_handles.append(handle)
    for handle, column, label in zip(
        state_handles,
        range(4),
        ("pressure", "pressure", "pressure error", "internal-energy density"),
        strict=True,
    ):
        state_fig.colorbar(handle, ax=state_axes[:, column], shrink=0.62, label=label)
    for handle, column, label in zip(
        residual_handles,
        range(3),
        ("pressure increment", "pressure increment", "increment error"),
        strict=True,
    ):
        residual_fig.colorbar(
            handle, ax=residual_axes[:, column], shrink=0.62, label=label
        )
    state_fig.suptitle("State immediately before each first invalid PCFNO proposal")
    residual_fig.suptitle("Residual that generates each first invalid PCFNO proposal")
    outputs = _save_figure(state_fig, output_dir / "fig_prefailure_state")
    outputs.extend(_save_figure(residual_fig, output_dir / "fig_prefailure_residual"))
    return outputs, {name: list(values) for name, values in scales.items()}


def _optional_float(value: str) -> float | None:
    if value in ("", "None", "null"):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def render_error_time(call_metrics_path: Path, output_dir: Path) -> list[Path]:
    with call_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.4), sharex=True, constrained_layout=True
    )
    for key in TRAJECTORY_KEYS:
        selected = [row for row in rows if row["trajectory"] == key]
        calls = np.asarray([int(row["call"]) for row in selected])
        times = np.asarray([float(row["physical_time"]) for row in selected])
        errors = np.asarray(
            [
                _optional_float(row["deployed_all_proxy_scaled_relative_l2"])
                for row in selected
            ],
            dtype=float,
        )
        minimum_internal = np.asarray(
            [_optional_float(row["deployed_min_internal_energy"]) for row in selected],
            dtype=float,
        )
        failure_call = int(
            next(
                row["call"]
                for row in selected
                if row["is_first_inadmissible_call"] == "True"
            )
        )
        axes[0].plot(times, errors, color=COLORS[key], label=f"trajectory {key}")
        axes[1].plot(
            times, minimum_internal, color=COLORS[key], label=f"trajectory {key}"
        )
        failure_time = times[np.flatnonzero(calls == failure_call)[0]]
        axes[0].scatter(
            [failure_time],
            [errors[calls == failure_call][0]],
            marker="X",
            s=45,
            color=COLORS[key],
            zorder=4,
        )
        axes[1].scatter(
            [failure_time],
            [minimum_internal[calls == failure_call][0]],
            marker="X",
            s=45,
            color=COLORS[key],
            zorder=4,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("proxy state-scaled relative L2")
    axes[0].set_title("PCFNO error before and after the first invalid proposal")
    axes[0].legend(ncol=3, frameon=False)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("minimum internal-energy density")
    axes[1].set_xlabel("physical time")
    axes[1].text(
        0.01,
        0.03,
        "X: first invalid deployed proposal",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
    )
    return _save_figure(fig, output_dir / "fig_rollout_error_vs_time")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.input_dir / "manifest.json"
    summary_path = args.input_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("input continuation lacks summary/manifest")
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("schema") != SCHEMA
        or input_manifest.get("status") != "completed"
    ):
        raise ValueError("input continuation manifest is not completed")
    cases = {
        key: load_case(args.input_dir / "trajectories" / f"trajectory_{key}.npz")
        for key in TRAJECTORY_KEYS
    }
    args.output_dir.mkdir(parents=True)
    animations = []
    for key in TRAJECTORY_KEYS:
        output = args.output_dir / f"pcfno_trajectory_{key}_continuation.{args.format}"
        animations.append(
            render_animation(cases[key], output, fps=args.fps, dpi=args.dpi)
        )
    static_outputs, static_scales = render_prefailure_figures(cases, args.output_dir)
    static_outputs.extend(
        render_error_time(args.input_dir / "call_metrics.csv", args.output_dir)
    )
    outputs = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    payload = {
        "schema": VISUAL_SCHEMA,
        "status": "completed",
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_summary_sha256": sha256_file(summary_path),
        "animations": animations,
        "static_scales": static_scales,
        "static_outputs": [path.name for path in static_outputs],
        "output_count": len(outputs),
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": (
            "visualization of a finite-invalid diagnostic recurrence; color saturation "
            "is quantified and does not imply disappearance of error"
        ),
    }
    write_json(args.output_dir / "manifest.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
