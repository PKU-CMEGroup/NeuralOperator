#!/usr/bin/env python3
"""Analyze and render the registered matched PCFNO/PCNO H320 comparison."""

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
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_KEYS,
    EVENT_NAMES,
    PRODUCTION_STEPS,
    TRUTH_STEPS,
    build_final_hash_manifest,
    conservative_to_primitive_numpy,
    internal_energy_numpy,
    load_npz,
    sha256_file,
    verify_final_hash_manifest,
    write_csv,
    write_json,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_SCHEMA as PCFNO_ANIMATION_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    CHECKPOINT_SHA256 as PCFNO_CHECKPOINT_SHA256,
)
from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    WORKING_ID as PCFNO_WORKING_ID,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    ANIMATION_SCHEMA as PCNO_ANIMATION_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    CHECKPOINT_SHA256 as PCNO_CHECKPOINT_SHA256,
)
from scripts.time_dependent_no.evaluate_pcno_h320_rollout import (
    WORKING_ID as PCNO_WORKING_ID,
)
from scripts.time_dependent_no.visualize_pcfno_h320_rollout import (
    invalid_categories,
)
from scripts.time_dependent_no.visualize_pcno_inadmissibility_continuation import (
    build_continuous_field_map,
    rasterize_field,
    saturation_fraction,
)

SCHEMA = "w26_l1_pcfno_pcno_h320_comparison_v1"
WORKING_ID = "W26-L1-PCFNO-PCNO-H320-C1-S20260718"
PCFNO_RESULT_MANIFEST_SHA256 = (
    "69c1b017e18cce9371f1c8b6f71763b021a36e9c75453a7242ae1b2df707c8a4"
)
PCFNO_CALIBRATION_SHA256 = (
    "568c489bf237eca73c040fe2f5149f1caeb5974272414cc188f2630ca40e99b2"
)
DT = 0.025
MODEL_ORDER = ("pcno", "pcfno")
MODEL_LABELS = {"pcno": "PCNO", "pcfno": "PCFNO"}
MODEL_COLORS = {"pcno": "#0072B2", "pcfno": "#D55E00"}
EVENT_LABELS = {
    "admissible": r"$T_{admissible}$",
    "bounded": r"$T_{bounded}$",
    "finite": r"$T_{finite}$",
}
EVENT_LINESTYLES = {"admissible": "-", "bounded": "--", "finite": ":"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcfno-dir", type=Path, required=True)
    parser.add_argument("--pcno-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--raster-width", type=int, default=180)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 50 or args.raster_width < 32:
        raise ValueError("fps, dpi, and raster width must be positive and meaningful")
    return args


def _scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected a scalar array")
    return array.item()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_result_root(root: Path, *, model: str) -> dict[str, Any]:
    if model not in MODEL_ORDER:
        raise ValueError(f"unknown model: {model}")
    manifest_path = root / "final_hash_manifest.json"
    manifest = _read_json(manifest_path)
    verification = verify_final_hash_manifest(root, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    if model == "pcfno" and manifest_sha256 != PCFNO_RESULT_MANIFEST_SHA256:
        raise ValueError("PCFNO input is not the registered closed H320 result")
    summary = _read_json(root / "summary.json")
    contract = _read_json(root / "run_contract.json")
    expected_working_id = PCNO_WORKING_ID if model == "pcno" else PCFNO_WORKING_ID
    expected_checkpoint = (
        PCNO_CHECKPOINT_SHA256 if model == "pcno" else PCFNO_CHECKPOINT_SHA256
    )
    prefix_contract_ok = (
        summary.get("all_prefixes_source_bound") is True
        if model == "pcno"
        else summary.get("all_prefixes_bitwise_equal") is True
    )
    checks = {
        "completed": summary.get("status") == "completed",
        "production": summary.get("mode") == "production",
        "horizon": int(summary.get("requested_horizon", -1)) == PRODUCTION_STEPS,
        "population": int(summary.get("trajectory_count", -1)) == 30,
        "prefixes": prefix_contract_ok,
        "working_id": summary.get("working_id") == expected_working_id,
        "contract_working_id": contract.get("working_id") == expected_working_id,
        "checkpoint": contract.get("checkpoint", {}).get("sha256")
        == expected_checkpoint,
        "accuracy_disabled": summary.get("accuracy_evaluated") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{model} result contract failed: {failed}")
    cases = summary.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != {
        str(value) for value in contract["full_registered_population"]
    }:
        raise ValueError(f"{model} summary case population changed")
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "manifest_sha256": manifest_sha256,
        "manifest_verification": verification,
        "calibration_sha256": sha256_file(root / "calibration_contract.json"),
    }


def verify_pair(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pcno = results["pcno"]
    pcfno = results["pcfno"]
    pcno_calibration = (pcno["root"] / "calibration_contract.json").read_bytes()
    pcfno_calibration = (pcfno["root"] / "calibration_contract.json").read_bytes()
    checks = {
        "pcfno_calibration_bound": pcfno["calibration_sha256"]
        == PCFNO_CALIBRATION_SHA256,
        "calibration_byte_identity": pcno_calibration == pcfno_calibration,
        "thresholds": pcno["contract"]["thresholds"] == pcfno["contract"]["thresholds"],
        "population": pcno["contract"]["full_registered_population"]
        == pcfno["contract"]["full_registered_population"],
        "animation_keys": pcno["contract"]["animation_keys"]
        == pcfno["contract"]["animation_keys"]
        == list(ANIMATION_KEYS),
        "precision": pcno["contract"]["precision"]
        == pcfno["contract"]["precision"]
        == "fp32_batch_one",
        "data": pcno["contract"]["data"] == pcfno["contract"]["data"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"paired result contract failed: {failed}")
    return checks


def horizon_records(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        cases = results[model]["summary"]["cases"]
        for trajectory in sorted(cases, key=int):
            for event_name in EVENT_NAMES:
                event = cases[trajectory]["events"][event_name]
                accepted = int(event["accepted_prefix_calls"])
                records.append(
                    {
                        "model": model,
                        "trajectory": trajectory,
                        "event": event_name,
                        "accepted_prefix_calls": accepted,
                        "physical_time": accepted * DT,
                        "first_failure_call": event["first_failure_call"],
                        "right_censored": bool(event["right_censored"]),
                        "recovered_after_failure": bool(
                            event["recovered_after_failure"]
                        ),
                    }
                )
    return records


def summarize_horizons(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for event_name in EVENT_NAMES:
            rows = [
                row
                for row in records
                if row["model"] == model and row["event"] == event_name
            ]
            if len(rows) != 30:
                raise ValueError(f"{model}/{event_name} does not contain 30 cases")
            values = np.asarray(
                [row["accepted_prefix_calls"] for row in rows], dtype=np.float64
            )
            quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
            output.append(
                {
                    "model": model,
                    "event": event_name,
                    "failed_count": sum(not row["right_censored"] for row in rows),
                    "censored_count": sum(row["right_censored"] for row in rows),
                    "minimum_calls": float(quantiles[0]),
                    "q1_calls": float(quantiles[1]),
                    "median_calls": float(quantiles[2]),
                    "q3_calls": float(quantiles[3]),
                    "maximum_calls": float(quantiles[4]),
                    "minimum_physical_time": float(quantiles[0] * DT),
                    "q1_physical_time": float(quantiles[1] * DT),
                    "median_physical_time": float(quantiles[2] * DT),
                    "q3_physical_time": float(quantiles[3] * DT),
                    "maximum_physical_time": float(quantiles[4] * DT),
                }
            )
    return output


def paired_horizons(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["trajectory"]), str(row["event"])): row
        for row in records
    }
    rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    trajectories = sorted({str(row["trajectory"]) for row in records}, key=int)
    for event_name in EVENT_NAMES:
        event_counts = {"pcno_later": 0, "tie": 0, "pcfno_later": 0}
        for trajectory in trajectories:
            pcno = lookup[("pcno", trajectory, event_name)]
            pcfno = lookup[("pcfno", trajectory, event_name)]
            pcno_value = int(pcno["accepted_prefix_calls"])
            pcfno_value = int(pcfno["accepted_prefix_calls"])
            if pcno_value > pcfno_value:
                winner = "pcno_later"
            elif pcno_value < pcfno_value:
                winner = "pcfno_later"
            else:
                winner = "tie"
            event_counts[winner] += 1
            rows.append(
                {
                    "trajectory": trajectory,
                    "event": event_name,
                    "pcno_calls": pcno_value,
                    "pcno_right_censored": bool(pcno["right_censored"]),
                    "pcfno_calls": pcfno_value,
                    "pcfno_right_censored": bool(pcfno["right_censored"]),
                    "pcno_minus_pcfno_calls": pcno_value - pcfno_value,
                    "descriptive_winner": winner,
                }
            )
        counts[event_name] = event_counts
    return rows, counts


def write_markdown_table(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "| model | horizon | failed / censored | min / Q1 / median / Q3 / max calls | median physical time |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = " / ".join(
            f"{float(row[name]):g}"
            for name in (
                "minimum_calls",
                "q1_calls",
                "median_calls",
                "q3_calls",
                "maximum_calls",
            )
        )
        lines.append(
            f"| {MODEL_LABELS[str(row['model'])]} | $T_{{\\rm {row['event']}}}$ | "
            f"{row['failed_count']} / {row['censored_count']} | {values} | "
            f"{float(row['median_physical_time']):g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def plot_stability_events(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    configure_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75), constrained_layout=True)
    offsets = {"pcno": -0.14, "pcfno": 0.14}
    for event_index, event_name in enumerate(EVENT_NAMES):
        for model in MODEL_ORDER:
            rows = [
                row
                for row in records
                if row["model"] == model and row["event"] == event_name
            ]
            failed = sorted(
                int(row["accepted_prefix_calls"])
                for row in rows
                if not row["right_censored"]
            )
            censored = sorted(
                int(row["accepted_prefix_calls"])
                for row in rows
                if row["right_censored"]
            )
            center = event_index + offsets[model]
            if failed:
                jitter = np.linspace(-0.075, 0.075, len(failed))
                axes[0].scatter(
                    failed,
                    center + jitter,
                    s=18,
                    color=MODEL_COLORS[model],
                    alpha=0.72,
                    linewidths=0,
                    zorder=3,
                )
            if censored:
                jitter = np.linspace(-0.075, 0.075, len(censored))
                axes[0].scatter(
                    censored,
                    center + jitter,
                    s=25,
                    marker=">",
                    facecolors="none",
                    edgecolors=MODEL_COLORS[model],
                    linewidths=0.9,
                    alpha=0.85,
                    zorder=3,
                )
            values = failed + censored
            axes[0].scatter(
                [float(np.median(values))],
                [center],
                marker="|",
                s=165,
                color="#111111",
                linewidths=1.5,
                zorder=4,
            )
    axes[0].axvline(TRUTH_STEPS, color="#777777", linewidth=0.8, linestyle=":")
    axes[0].axvline(PRODUCTION_STEPS, color="#AAAAAA", linewidth=0.7, linestyle=":")
    axes[0].set_xlim(0, 330)
    axes[0].set_xticks([0, 79, 100, 160, 240, 320])
    axes[0].set_xlabel("Accepted recurrent calls")
    axes[0].set_yticks(np.arange(len(EVENT_NAMES)))
    axes[0].set_yticklabels([EVENT_LABELS[name] for name in EVENT_NAMES])
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Three reference-free stability horizons", loc="left")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].text(
        0.015,
        0.98,
        "PCNO",
        transform=axes[0].transAxes,
        color=MODEL_COLORS["pcno"],
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    axes[0].text(
        0.015,
        0.90,
        "PCFNO",
        transform=axes[0].transAxes,
        color=MODEL_COLORS["pcfno"],
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    axes[0].text(
        0.015,
        0.80,
        "circle: observed failure;  >: right-censored at H320;  |: median",
        transform=axes[0].transAxes,
        color="#555555",
        fontsize=6.8,
        va="top",
    )

    calls = np.arange(PRODUCTION_STEPS + 1)
    for model in MODEL_ORDER:
        for event_name in EVENT_NAMES:
            rows = [
                row
                for row in records
                if row["model"] == model and row["event"] == event_name
            ]
            accepted = np.asarray(
                [int(row["accepted_prefix_calls"]) for row in rows], dtype=np.int64
            )
            survival = np.asarray([(accepted >= call).mean() for call in calls])
            axes[1].step(
                calls,
                survival,
                where="post",
                color=MODEL_COLORS[model],
                linestyle=EVENT_LINESTYLES[event_name],
                linewidth=1.5,
                label=f"{MODEL_LABELS[model]} {event_name}",
            )
    axes[1].axvline(TRUTH_STEPS, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].set_xlim(0, PRODUCTION_STEPS)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xticks([0, 79, 100, 160, 240, 320])
    axes[1].set_xlabel("Recurrent call")
    axes[1].set_ylabel("Accepted-prefix survival")
    axes[1].set_title("(b) Event survival through H320", loc="left")
    axes[1].grid(alpha=0.18)
    axes[1].legend(loc="upper right", frameon=False, fontsize=6.8, ncol=2)
    axes[1].text(
        0.99,
        0.03,
        "matching truth ends at H79",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=6.8,
    )

    prefix = output_dir / "pcfno_pcno_h320_stability_events"
    outputs = [prefix.with_suffix(".png"), prefix.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def frame_schedule(horizon: int, event_calls: Sequence[int | None]) -> list[int]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    calls = set(range(0, min(horizon, 80) + 1, 4))
    if horizon > 80:
        calls.update(range(80, min(horizon, 140) + 1, 2))
    if horizon > 140:
        calls.update(range(140, horizon + 1, 4))
    calls.update(
        value
        for value in (79, 100, 120, 160, 200, 240, 280, horizon)
        if value <= horizon
    )
    calls.update(
        int(value)
        for value in event_calls
        if value is not None and 0 <= int(value) <= horizon
    )
    return sorted(calls)


def load_bundle(root: Path, *, model: str, trajectory: str) -> dict[str, np.ndarray]:
    arrays = load_npz(root / f"trajectory_{trajectory}.npz")
    expected_schema = (
        PCNO_ANIMATION_SCHEMA if model == "pcno" else PCFNO_ANIMATION_SCHEMA
    )
    expected_working_id = PCNO_WORKING_ID if model == "pcno" else PCFNO_WORKING_ID
    expected_checkpoint = (
        PCNO_CHECKPOINT_SHA256 if model == "pcno" else PCFNO_CHECKPOINT_SHA256
    )
    if str(_scalar(arrays["schema"])) != expected_schema:
        raise ValueError(f"{model} bundle schema changed")
    if str(_scalar(arrays["working_id"])) != expected_working_id:
        raise ValueError(f"{model} bundle working ID changed")
    if str(_scalar(arrays["checkpoint_sha256"])) != expected_checkpoint:
        raise ValueError(f"{model} bundle checkpoint changed")
    if str(_scalar(arrays["trajectory"])) != trajectory:
        raise ValueError(f"{model} bundle trajectory changed")
    if int(_scalar(arrays["requested_horizon"])) != PRODUCTION_STEPS:
        raise ValueError(f"{model} bundle horizon changed")
    recorded = int(_scalar(arrays["recorded_call_count"]))
    states = np.asarray(arrays["deployed_states_conservative"])
    if states.shape != (recorded + 1, states.shape[1], 4):
        raise ValueError(f"{model} state bundle length changed")
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
            raise ValueError(f"{model} bundle field {name} has the wrong length")
    return arrays


def verify_bundle_pair(bundles: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    for name in ("positions", "node_type", "state_scale"):
        if not np.array_equal(bundles["pcno"][name], bundles["pcfno"][name]):
            raise ValueError(f"paired animation bundle field changed: {name}")
    if str(_scalar(bundles["pcno"]["presentation_scales_json"])) != str(
        _scalar(bundles["pcfno"]["presentation_scales_json"])
    ):
        raise ValueError("paired animation bundles use different presentation scales")
    if not np.array_equal(
        bundles["pcno"]["deployed_states_conservative"][0],
        bundles["pcfno"]["deployed_states_conservative"][0],
    ):
        raise ValueError("paired animation bundles do not share frame zero")


def animation_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    states = np.asarray(arrays["deployed_states_conservative"], dtype=np.float64)
    state_scale = np.asarray(arrays["state_scale"], dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        primitive = conservative_to_primitive_numpy(states, gamma=1.4)
        internal = internal_energy_numpy(states)
        scaled_increment = np.zeros(states.shape[:-1], dtype=np.float64)
        scaled_increment[1:] = np.linalg.norm(
            np.diff(states, axis=0) / state_scale[None, None, :], axis=-1
        )
    return {
        "pressure": primitive[..., 3],
        "internal_energy": internal,
        "scaled_increment": scaled_increment,
        "invalid_category": invalid_categories(states),
    }


def _metric_text(value: float) -> str:
    return "unavailable" if not math.isfinite(float(value)) else f"{float(value):.3g}"


def _event_horizon_text(case: Mapping[str, Any]) -> str:
    values = []
    for name, short in (("admissible", "A"), ("bounded", "B"), ("finite", "F")):
        event = case["events"][name]
        suffix = ">" if event["right_censored"] else ""
        values.append(f"{short}:{suffix}{event['accepted_prefix_calls']}")
    return " ".join(values)


def render_comparison_gif(
    bundles: Mapping[str, Mapping[str, np.ndarray]],
    cases: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    *,
    fps: int,
    dpi: int,
    raster_width: int,
    frame_calls: Sequence[int] | None = None,
) -> dict[str, Any]:
    verify_bundle_pair(bundles)
    trajectory = str(_scalar(bundles["pcno"]["trajectory"]))
    positions = np.asarray(bundles["pcno"]["positions"], dtype=np.float64)
    scales = json.loads(str(_scalar(bundles["pcno"]["presentation_scales_json"])))
    fields = {model: animation_fields(bundles[model]) for model in MODEL_ORDER}
    field_map = build_continuous_field_map(positions, raster_width=raster_width)
    extent = field_map["extent"]
    event_calls = [
        cases[model]["events"][event]["first_failure_call"]
        for model in MODEL_ORDER
        for event in EVENT_NAMES
    ]
    if frame_calls is None:
        rendered_calls = frame_schedule(PRODUCTION_STEPS, event_calls)
    else:
        rendered_calls = [int(value) for value in frame_calls]
        if (
            not rendered_calls
            or rendered_calls != sorted(set(rendered_calls))
            or rendered_calls[0] < 0
            or rendered_calls[-1] > PRODUCTION_STEPS
        ):
            raise ValueError(
                "explicit frame calls must be sorted, unique, and in H0-H320"
            )

    configure_report_style()
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.0), constrained_layout=True)
    panels = (
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
            "Checkpoint-scaled increment",
            "scaled_increment",
            0.0,
            scales["scaled_increment_max"],
            "magma",
        ),
    )
    continuous_images: dict[str, list[Any]] = {model: [] for model in MODEL_ORDER}
    invalid_scatters: dict[str, Any] = {}
    termination_texts: dict[str, list[Any]] = {model: [] for model in MODEL_ORDER}
    status_texts: dict[str, Any] = {}
    invalid_cmap = ListedColormap(("#F4F4F4", "#E69F00", "#D55E00", "#000000"))
    invalid_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), invalid_cmap.N)

    for row_index, model in enumerate(MODEL_ORDER):
        for column, (title, _, lower, upper, cmap) in enumerate(panels):
            axis = axes[row_index, column]
            image = axis.imshow(
                np.full(field_map["shape"], np.nan),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            axis.set_title(f"{MODEL_LABELS[model]} {title}")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, shrink=0.68)
            continuous_images[model].append(image)
            termination_texts[model].append(
                axis.text(
                    0.5,
                    0.5,
                    "",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#B2182B",
                    bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#B2182B"},
                    visible=False,
                )
            )
        axis = axes[row_index, 3]
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=np.zeros(positions.shape[0]),
            s=1.4,
            linewidths=0.0,
            cmap=invalid_cmap,
            norm=invalid_norm,
            rasterized=True,
        )
        axis.set_title(f"{MODEL_LABELS[model]} admissibility")
        axis.set_aspect("equal")
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks([])
        axis.set_yticks([])
        bar = fig.colorbar(scatter, ax=axis, shrink=0.68, ticks=(0, 1, 2, 3))
        bar.ax.set_yticklabels(("valid", "density", "energy/p", "nonfinite"))
        invalid_scatters[model] = scatter
        termination_texts[model].append(
            axis.text(
                0.5,
                0.5,
                "",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#B2182B",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#B2182B"},
                visible=False,
            )
        )
        status_texts[model] = axes[row_index, 0].text(
            0.01,
            0.01,
            "",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
        )

    def update(call: int) -> list[Any]:
        artists: list[Any] = []
        for model in MODEL_ORDER:
            arrays = bundles[model]
            recorded = int(_scalar(arrays["recorded_call_count"]))
            terminated = call > recorded
            for message in termination_texts[model]:
                message.set_visible(terminated)
                if terminated:
                    message.set_text(f"recurrence stopped\nnonfinite at H{recorded}")
                artists.append(message)
            if terminated:
                for image in continuous_images[model]:
                    image.set_data(np.ma.masked_all(field_map["shape"]))
                    artists.append(image)
                invalid_scatters[model].set_offsets(np.empty((0, 2)))
                invalid_scatters[model].set_array(np.empty((0,)))
                status_texts[model].set_text(
                    f"{_event_horizon_text(cases[model])}\nterminated at H{recorded}"
                )
                artists.extend((invalid_scatters[model], status_texts[model]))
                continue

            invalid_scatters[model].set_offsets(positions)
            for image, (_, name, _, _, _) in zip(
                continuous_images[model], panels, strict=True
            ):
                image.set_data(
                    np.ma.masked_invalid(
                        rasterize_field(fields[model][name][call], field_map)
                    )
                )
                artists.append(image)
            invalid_scatters[model].set_array(
                fields[model]["invalid_category"][call].astype(np.float64, copy=False)
            )
            amplitude = float(arrays["common_amplitude_ratio"][call])
            rms = float(arrays["common_scaled_rms_ratio"][call])
            minimum_internal = float(arrays["minimum_internal_energy"][call])
            status_texts[model].set_text(
                f"{_event_horizon_text(cases[model])}\n"
                f"amp={_metric_text(amplitude)}  RMS={_metric_text(rms)}  "
                f"min IE={_metric_text(minimum_internal)}"
            )
            artists.extend((invalid_scatters[model], status_texts[model]))

        phase = (
            "bitwise-checked prefix"
            if call <= TRUTH_STEPS
            else "reference-free recurrence"
        )
        fig.suptitle(
            f"case {trajectory} | H{call}/{PRODUCTION_STEPS} | "
            f"t={call * DT:.3f} | {phase} | shared fixed H79-reference scales",
            fontsize=10,
        )
        return artists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=rendered_calls,
        interval=1000 / fps,
        blit=False,
    )
    movie.save(
        output_path,
        writer=animation.PillowWriter(fps=fps),
        dpi=dpi,
        savefig_kwargs={"facecolor": "white", "transparent": False},
    )
    plt.close(fig)

    saturation: dict[str, Any] = {}
    for model in MODEL_ORDER:
        saturation[model] = {
            name: saturation_fraction(fields[model][name], lower, upper)
            for _, name, lower, upper, _ in panels
        }
    return {
        "trajectory": trajectory,
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "rendered_frame_count": len(rendered_calls),
        "rendered_calls": rendered_calls,
        "fps": fps,
        "pcno_recorded_calls": int(_scalar(bundles["pcno"]["recorded_call_count"])),
        "pcfno_recorded_calls": int(_scalar(bundles["pcfno"]["recorded_call_count"])),
        "saturation_fraction": saturation,
        "encoder": {"writer": "matplotlib.PillowWriter", "format": "gif"},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = {
        "pcno": verify_result_root(args.pcno_dir, model="pcno"),
        "pcfno": verify_result_root(args.pcfno_dir, model="pcfno"),
    }
    pair_checks = verify_pair(results)
    records = horizon_records(results)
    summaries = summarize_horizons(records)
    paired_rows, paired_counts = paired_horizons(records)

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "horizon_summary.csv", summaries)
    write_csv(args.output_dir / "paired_horizons.csv", paired_rows)
    write_markdown_table(args.output_dir / "horizon_summary.md", summaries)
    static_outputs = plot_stability_events(records, args.output_dir)

    animations = []
    for trajectory in ANIMATION_KEYS:
        bundles = {
            model: load_bundle(
                results[model]["root"], model=model, trajectory=trajectory
            )
            for model in MODEL_ORDER
        }
        cases = {
            model: results[model]["summary"]["cases"][trajectory]
            for model in MODEL_ORDER
        }
        animations.append(
            render_comparison_gif(
                bundles,
                cases,
                args.output_dir
                / f"pcfno_pcno_h320_trajectory_{trajectory}_comparison.gif",
                fps=args.fps,
                dpi=args.dpi,
                raster_width=args.raster_width,
            )
        )

    summary = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "requested_horizon": PRODUCTION_STEPS,
        "truth_supported_through_call": TRUTH_STEPS,
        "dt": DT,
        "input_manifests": {
            model: {
                "sha256": results[model]["manifest_sha256"],
                **results[model]["manifest_verification"],
            }
            for model in MODEL_ORDER
        },
        "pair_checks": pair_checks,
        "horizon_summaries": summaries,
        "paired_counts": paired_counts,
        "animations": animations,
        "static_outputs": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in static_outputs
        ],
        "claim_boundary": (
            "H80-H320 is reference-free recurrence; visual shock arrest and scale "
            "growth are descriptive, not accuracy, physical-validity, conservation, "
            "asymptotic-stability, or seed-general gradient effects"
        ),
    }
    write_json(args.output_dir / "comparison_summary.json", summary)
    artifact_names = [
        "horizon_summary.csv",
        "paired_horizons.csv",
        "horizon_summary.md",
        "pcfno_pcno_h320_stability_events.png",
        "pcfno_pcno_h320_stability_events.pdf",
        *[
            f"pcfno_pcno_h320_trajectory_{trajectory}_comparison.gif"
            for trajectory in ANIMATION_KEYS
        ],
        "comparison_summary.json",
    ]
    manifest = build_final_hash_manifest(args.output_dir, artifact_names)
    write_json(args.output_dir / "final_hash_manifest.json", manifest)
    verification = verify_final_hash_manifest(args.output_dir, manifest)
    summary["final_hash_manifest"] = {
        "sha256": sha256_file(args.output_dir / "final_hash_manifest.json"),
        **verification,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
