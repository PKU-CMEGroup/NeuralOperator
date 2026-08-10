#!/usr/bin/env python3
"""Render hash-verified D076--D078 response-controller diagnostics."""

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

SCHEMA = "pcno_response_gain_controller_visualization_v1"
SUPPORTED_RESULTS = {
    (
        "pcno_response_gain_controller_diagnostic_v1",
        "d076_response_probe",
    ),
    (
        "pcno_strength_grouped_response_controller_diagnostic_v1",
        "d077_strength_grouped_response_probe",
    ),
    (
        "pcno_deterministic_response_controller_diagnostic_v1",
        "d078_deterministic_strength_grouped_response_probe",
    ),
}
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
REQUIRED_TABLES = {
    "calibration_controller_selections.csv",
    "evaluation_case_summary.csv",
    "evaluation_comparisons.csv",
    "evaluation_gain_selections.csv",
    "evaluation_probe_replay.csv",
    "sequence_time_metrics.csv",
    "visual_payload_inventory.csv",
}
PREDICTION_PANEL_FIELDS = (
    ("true_increment", "true increment r", "residual"),
    ("uncorrected_increment", "baseline prediction r+b", "residual"),
    ("corrected_increment", "selected prediction r+b+C", "residual"),
    ("base_defect", "same-input baseline defect b", "defect"),
    ("correction", "selected correction C", "defect"),
    ("corrected_defect", "selected residual defect", "defect"),
    ("cumulative_error", "selected cumulative defect", "cumulative"),
    ("signed_growth", "signed local error growth", "growth"),
)
RESPONSE_PANEL_FIELDS = (
    ("true_increment", "true increment r", "residual"),
    ("zero_defect", "zero-arm defect", "defect"),
    ("static_defect", "static gain-0.125 defect", "defect"),
    ("selected_defect", "selected defect", "defect"),
    ("response_increment", "selected-minus-zero increment", "defect"),
    ("cumulative_error", "selected cumulative defect", "cumulative"),
    ("response_state", "selected-minus-zero state", "cumulative"),
    ("signed_growth", "signed local error growth", "growth"),
)
COLORS = {
    "baseline": "#333333",
    "static": "#E69F00",
    "selected": "#0072B2",
    "endpoint": "#0072B2",
    "residual": "#009E73",
    "control": "#D55E00",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("plots", "animations", "all"), default="all"
    )
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--panel-set",
        choices=("response", "prediction"),
        default="response",
        help="Use the registered D076 response panels or prediction-focused panels.",
    )
    parser.add_argument(
        "--allow-failed-contract-diagnostics",
        action="store_true",
        help=(
            "Allow rendering a failed-contract run only as explicitly watermarked "
            "diagnostic evidence. This never enables scientific interpretation."
        ),
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 30 or args.frame_stride < 1:
        raise ValueError("fps, dpi, and frame-stride must be positive")
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected scalar payload field")
    return array.item()


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"result path escapes results directory: {relative}") from error
    return path


def _verify_results(
    results_dir: Path, *, allow_failed_contract: bool
) -> tuple[dict[str, Any], list[Path], str | None]:
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = (summary.get("schema"), summary.get("experiment_contract"))
    if identity not in SUPPORTED_RESULTS:
        raise ValueError(f"unsupported response-controller result: {identity}")

    status = str(summary.get("status"))
    passed = bool(summary.get("contract_checks_passed"))
    interpretable = bool(summary.get("scientific_interpretation_allowed"))
    watermark = None
    if status == "smoke_complete" and passed and not interpretable:
        watermark = "H2 SMOKE - NON-SCIENTIFIC"
    elif status == "complete" and passed and interpretable:
        pass
    elif (
        status == "failed_contract"
        and not passed
        and not interpretable
        and allow_failed_contract
    ):
        watermark = "FAILED CONTRACT - DIAGNOSTIC ONLY"
    else:
        raise ValueError(
            "response-controller result is not eligible for the requested rendering"
        )

    output_hashes = summary.get("output_hashes")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise ValueError("summary contains no output hashes")
    missing = sorted(REQUIRED_TABLES - set(output_hashes))
    if missing:
        raise ValueError(f"output hashes omit required tables: {missing}")
    for relative, expected in output_hashes.items():
        path = _contained_path(results_dir, str(relative))
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"output digest mismatch: {relative}")

    selections = {
        row["case_id"]: row
        for row in _read_rows(results_dir / "evaluation_gain_selections.csv")
    }
    inventory = _read_rows(results_dir / "visual_payload_inventory.csv")
    inventory_paths = {row["relative_path"] for row in inventory}
    hash_paths = {
        str(relative)
        for relative in output_hashes
        if str(relative).startswith("visual_payloads/")
    }
    disk_paths = {
        path.relative_to(results_dir).as_posix()
        for path in (results_dir / "visual_payloads").glob("*.npz")
    }
    if inventory_paths != hash_paths or inventory_paths != disk_paths:
        raise ValueError("visual payload inventories disagree")
    for row in inventory:
        selection = selections.get(row["case_id"])
        relative = row["relative_path"]
        if (
            selection is None
            or row.get("sha256") != output_hashes.get(relative)
            or row["selected_candidate"] != selection["selected_candidate"]
            or not np.isclose(
                float(row["selected_gain"]), float(selection["selected_gain"])
            )
        ):
            raise ValueError(f"visual selection binding failed: {relative}")
    return (
        summary,
        [_contained_path(results_dir, value) for value in sorted(hash_paths)],
        watermark,
    )


def _load_payload(path: Path, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    if str(_scalar(payload["schema"])) != str(summary["schema"]):
        raise ValueError("visual payload schema differs from summary")
    if str(_scalar(payload["family"])) != str(summary["family"]):
        raise ValueError("visual payload family differs from summary")
    if tuple(payload["component_names"].tolist()) != COMPONENTS:
        raise ValueError("visual payload component inventory changed")
    calls = int(_scalar(payload["expected_calls"]))
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    weights = np.asarray(payload["weights"], dtype=np.float64)
    node_type = np.asarray(payload["node_type"], dtype=np.int64)
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    if (
        calls < 1
        or not bool(_scalar(payload["baseline_complete"]))
        or not bool(_scalar(payload["selected_complete"]))
        or nodes.ndim != 2
        or nodes.shape[1] != 2
        or weights.shape != (nodes.shape[0],)
        or node_type.shape != (nodes.shape[0],)
        or times.shape != (calls,)
        or scale.shape != (len(COMPONENTS),)
        or np.any(weights <= 0.0)
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(scale))
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError("invalid visual payload geometry, clock, weights, or scales")
    if summary["family"] == "dynamic_fv" and not set(np.unique(node_type)) <= {
        0,
        1,
        2,
        3,
    }:
        raise ValueError("dynamic payload node types violate family-local meanings")

    fields = (
        "true_increment",
        "baseline_defect",
        "corrected_input_base_defect",
        "correction",
        "corrected_defect",
        "cumulative_error",
        "signed_growth_density_scaled",
        "signed_growth_contribution",
        "static_defect",
        "static_cumulative_error",
        "selected_minus_zero_state",
        "selected_minus_static_state",
    )
    expected_shape = (calls, nodes.shape[0], len(COMPONENTS))
    for name in fields:
        value = np.asarray(payload[name])
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid visual field: {name}")
    defect = np.asarray(payload["corrected_defect"], dtype=np.float64)
    cumulative = np.asarray(payload["cumulative_error"], dtype=np.float64)
    if not np.allclose(cumulative, np.cumsum(defect, axis=0), rtol=0.0, atol=2e-7):
        raise ValueError("stored cumulative-defect replay failed")
    previous = np.concatenate((np.zeros_like(cumulative[:1]), cumulative[:-1]))
    growth = (2.0 * previous * defect + np.square(defect)) / np.square(
        scale[None, None, :]
    )
    if not np.allclose(
        payload["signed_growth_density_scaled"], growth, rtol=2e-5, atol=2e-6
    ):
        raise ValueError("stored signed-growth replay failed")
    return payload


def _derived_fields(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)[None, None, :]
    truth = np.asarray(payload["true_increment"], dtype=np.float64)
    base = np.asarray(payload["corrected_input_base_defect"], dtype=np.float64)
    correction = np.asarray(payload["correction"], dtype=np.float64)
    corrected = np.asarray(payload["corrected_defect"], dtype=np.float64)
    response_state = np.asarray(
        payload["selected_minus_zero_state"], dtype=np.float64
    )
    previous_response = np.concatenate(
        (np.zeros_like(response_state[:1]), response_state[:-1]), axis=0
    )
    return {
        "true_increment": truth / scale,
        "uncorrected_increment": (truth + base) / scale,
        "corrected_increment": (truth + corrected) / scale,
        "base_defect": base / scale,
        "correction": correction / scale,
        "corrected_defect": corrected / scale,
        "zero_defect": np.asarray(payload["baseline_defect"], dtype=np.float64)
        / scale,
        "static_defect": np.asarray(payload["static_defect"], dtype=np.float64)
        / scale,
        "selected_defect": corrected / scale,
        "response_increment": (response_state - previous_response) / scale,
        "response_state": response_state / scale,
        "cumulative_error": np.asarray(
            payload["cumulative_error"], dtype=np.float64
        )
        / scale,
        "signed_growth": np.asarray(
            payload["signed_growth_density_scaled"], dtype=np.float64
        ),
    }


def _watermark(figure: plt.Figure, text: str | None) -> None:
    if text:
        figure.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=28,
            color="#8B0000",
            alpha=0.12,
            rotation=24,
            weight="bold",
        )


def _save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, watermark: str | None
) -> list[Path]:
    _watermark(figure, watermark)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_calibration_response(
    results_dir: Path, output_dir: Path, watermark: str | None
) -> list[Path]:
    rows = _read_rows(results_dir / "calibration_controller_selections.csv")
    by_position: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_position[row["case_id"].rsplit("_", 1)[1]].append(row)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for position, values in sorted(by_position.items()):
        ordered = sorted(values, key=lambda row: row["query_group_id"])
        strength = np.asarray(
            [int(row["query_group_id"].split("e")[-1]) for row in ordered]
        )
        axes[0].plot(
            strength,
            [float(row["selected_gain"]) for row in ordered],
            marker="o",
            label=position,
        )
        axes[1].plot(
            strength,
            [float(row["actual_endpoint_state_ratio"]) for row in ordered],
            marker="o",
            label=position,
        )
        axes[2].plot(
            strength,
            [float(row["actual_residual_rms_ratio"]) for row in ordered],
            marker="o",
            label=position,
        )
    axes[0].set_ylabel("selected gain")
    axes[1].set_ylabel("selected / zero endpoint error")
    axes[2].set_ylabel("selected / zero residual RMS")
    axes[1].axhline(0.95, color="#777777", linestyle="--", linewidth=0.8)
    axes[2].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    for axis, title in zip(
        axes,
        ("group-cross-fit gain", "endpoint response", "residual response"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("physical-strength group index")
        axis.grid(True, alpha=0.35)
        axis.legend(fontsize=8)
    return _save_figure(figure, output_dir, "calibration_strength_response", watermark)


def plot_evaluation_summary(
    results_dir: Path, output_dir: Path, watermark: str | None
) -> list[Path]:
    selections = sorted(
        _read_rows(results_dir / "evaluation_gain_selections.csv"),
        key=lambda row: row["case_id"],
    )
    comparisons = {
        (row["case_id"], row["comparison"]): row
        for row in _read_rows(results_dir / "evaluation_comparisons.csv")
    }
    labels = [row["case_id"].replace("sv_", "") for row in selections]
    x = np.arange(len(labels), dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), constrained_layout=True)
    gain = [float(row["selected_gain"]) for row in selections]
    axes[0, 0].bar(x, gain, color=COLORS["selected"])
    axes[0, 0].set_ylabel("selected gain")
    axes[0, 0].set_title("target-free H5 choice")

    for comparison, label, color, marker in (
        ("selected_vs_zero", "selected / zero", COLORS["selected"], "o"),
        ("static_vs_zero", "static / zero", COLORS["static"], "s"),
    ):
        axes[0, 1].plot(
            x,
            [
                float(comparisons[(row["case_id"], comparison)]["endpoint_state_ratio"])
                for row in selections
            ],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            x,
            [
                float(comparisons[(row["case_id"], comparison)]["residual_rms_ratio"])
                for row in selections
            ],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            x,
            [
                float(comparisons[(row["case_id"], comparison)]["maximum_control_ratio"])
                for row in selections
            ],
            marker=marker,
            color=color,
            label=label,
        )
    axes[0, 1].set_title("H30 endpoint error")
    axes[1, 0].set_title("H30 aggregate residual RMS")
    axes[1, 1].set_title("worst registered control")
    axes[0, 1].set_ylabel("ratio to zero")
    axes[1, 0].set_ylabel("ratio to zero")
    axes[1, 1].set_ylabel("ratio to zero")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.grid(True, alpha=0.35)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend(fontsize=8)
    return _save_figure(figure, output_dir, "evaluation_controller_summary", watermark)


def plot_replay_failure(
    results_dir: Path, output_dir: Path, watermark: str | None
) -> list[Path]:
    rows = sorted(
        _read_rows(results_dir / "evaluation_probe_replay.csv"),
        key=lambda row: row["case_id"],
    )
    labels = [row["case_id"].replace("sv_", "") for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    colors = [
        "#999999" if float(row["selected_gain"]) == 0.0 else COLORS["selected"]
        for row in rows
    ]
    absolute = [
        float(row["maximum_absolute"]) / float(row["absolute_limit"]) for row in rows
    ]
    relative = [
        float(row["relative_l2"]) / float(row["relative_limit"]) for row in rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].bar(x, absolute, color=colors)
    axes[1].bar(x, relative, color=colors)
    for axis, title in zip(
        axes,
        ("maximum absolute / frozen limit", "relative L2 / frozen limit"),
        strict=True,
    ):
        axis.axhline(1.0, color="#8B0000", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel("threshold multiple")
        axis.grid(True, axis="y", alpha=0.35)
    axes[1].text(
        0.98,
        0.96,
        "gray = zero correction",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    return _save_figure(figure, output_dir, "prefix_replay_contract", watermark)


def _case_series(
    rows: Sequence[Mapping[str, str]], arm: str, field: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if (
            row["arm"] == arm
            and row["defect_kind"] == "corrected_defect_on_own_recurrent_inputs"
        ):
            grouped[row["case_id"]].append((int(row["step"]), float(row[field])))
    return {
        case: (
            np.asarray([item[0] for item in sorted(values)]),
            np.asarray([item[1] for item in sorted(values)]),
        )
        for case, values in grouped.items()
    }


def _draw_case_first(
    axis: plt.Axes,
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    label: str,
    color: str,
) -> None:
    by_step: dict[int, list[float]] = defaultdict(list)
    for steps, values in series.values():
        axis.plot(steps, values, color=color, alpha=0.2, linewidth=0.8)
        for step, value in zip(steps, values, strict=True):
            by_step[int(step)].append(float(value))
    steps = np.asarray(sorted(by_step), dtype=np.float64)
    median = np.asarray([np.median(by_step[int(step)]) for step in steps])
    lower = np.asarray([np.quantile(by_step[int(step)], 0.25) for step in steps])
    upper = np.asarray([np.quantile(by_step[int(step)], 0.75) for step in steps])
    axis.fill_between(steps, lower, upper, color=color, alpha=0.12, linewidth=0.0)
    axis.plot(steps, median, color=color, linewidth=1.8, label=label)


def plot_temporal_structure(
    results_dir: Path, output_dir: Path, watermark: str | None
) -> list[Path]:
    rows = _read_rows(results_dir / "sequence_time_metrics.csv")
    fields = (
        ("instant_relative", "instant defect / truth increment"),
        ("cumulative_relative", "cumulative defect / truth change"),
        ("temporal_coherence", "temporal coherence"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat[:3], fields, strict=True):
        for arm, label, color in (
            ("baseline", "zero", COLORS["baseline"]),
            ("selected", "selected", COLORS["selected"]),
        ):
            _draw_case_first(
                axis,
                _case_series(rows, arm, field),
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("rollout step")
        axis.grid(True, alpha=0.35)
        axis.legend(fontsize=8)

    baseline = _case_series(rows, "baseline", "cumulative_defect_rms")
    selected = _case_series(rows, "selected", "cumulative_defect_rms")
    ratio_series = {}
    for case in sorted(set(baseline) & set(selected)):
        if not np.array_equal(baseline[case][0], selected[case][0]):
            raise ValueError(f"temporal step inventory differs for {case}")
        ratio_series[case] = (
            baseline[case][0],
            selected[case][1] / np.maximum(baseline[case][1], 1e-30),
        )
    _draw_case_first(
        axes[1, 1],
        ratio_series,
        label="selected / zero",
        color=COLORS["selected"],
    )
    axes[1, 1].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("cumulative defect RMS ratio")
    axes[1, 1].set_xlabel("rollout step")
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend(fontsize=8)
    return _save_figure(figure, output_dir, "temporal_defect_structure", watermark)


def plot_subspace_response(
    results_dir: Path, output_dir: Path, watermark: str | None
) -> list[Path]:
    rows = [row for row in _case_metric_rows(results_dir) if row["selected_gain"] > 0]
    labels = [row["case_id"].replace("sv_", "") for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    for offset, field, label, color in (
        (-width, "selected_parallel_energy_ratio", "rank-8 parallel", "#D55E00"),
        (0.0, "selected_orthogonal_energy_ratio", "orthogonal", "#777777"),
        (width, "selected_total_energy_ratio", "total", "#0072B2"),
    ):
        axis.bar(
            x + offset,
            [float(row[field]) for row in rows],
            width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#222222", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("selected / same-input baseline energy")
    axis.set_title("where the persistent rank-8 correction acts")
    axis.grid(True, axis="y", alpha=0.35)
    axis.legend(fontsize=8)
    return _save_figure(figure, output_dir, "subspace_energy_response", watermark)


def _structured_shape(resolution: str, count: int) -> tuple[int, int] | None:
    try:
        nx, ny = (int(value) for value in resolution.lower().split("x"))
    except (TypeError, ValueError):
        return None
    return (nx, ny) if nx * ny == count else None


def _structured_extent(
    nodes: np.ndarray, shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    nx, ny = shape
    grid = nodes.reshape(ny, nx, 2)
    dx = float(np.median(np.diff(grid[0, :, 0]))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(grid[:, 0, 1]))) if ny > 1 else 1.0
    return (
        float(grid[0, 0, 0] - 0.5 * dx),
        float(grid[0, -1, 0] + 0.5 * dx),
        float(grid[0, 0, 1] - 0.5 * dy),
        float(grid[-1, 0, 1] + 0.5 * dy),
    )


def _frame_indices(count: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, count, stride))
    if values[-1] != count - 1:
        values.append(count - 1)
    return tuple(values)


def _population_limits(
    payload_paths: Sequence[Path],
    summary: Mapping[str, Any],
    case_ids: set[str] | None,
    panel_fields: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, np.ndarray], list[Path]]:
    limits = {
        name: np.zeros(len(COMPONENTS), dtype=np.float64)
        for name in ("residual", "defect", "cumulative", "growth")
    }
    selected_paths = []
    observed = set()
    for path in payload_paths:
        payload = _load_payload(path, summary)
        case_id = str(_scalar(payload["case_id"]))
        observed.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue
        selected_paths.append(path)
        fields = _derived_fields(payload)
        for name, _, group in panel_fields:
            limits[group] = np.maximum(
                limits[group], np.max(np.abs(fields[name]), axis=(0, 1))
            )
    if case_ids is not None and case_ids - observed:
        raise ValueError(f"requested visualization cases are absent: {case_ids - observed}")
    if not selected_paths:
        raise ValueError("no visual payloads selected")
    for values in limits.values():
        values[values == 0.0] = 1.0
    return limits, selected_paths


def animate_payload_component(
    payload_path: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    limits: Mapping[str, np.ndarray],
    *,
    component: int,
    fps: int,
    dpi: int,
    frame_stride: int,
    watermark: str | None,
    panel_fields: Sequence[tuple[str, str, str]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = _load_payload(payload_path, summary)
    fields = _derived_fields(payload)
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    family = str(_scalar(payload["family"]))
    case_id = str(_scalar(payload["case_id"]))
    resolution = str(_scalar(payload["resolution"]))
    candidate = str(_scalar(payload["selected_candidate"]))
    gain = float(_scalar(payload["selected_gain"]))
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    shape = _structured_shape(resolution, nodes.shape[0])
    extent = None if shape is None else _structured_extent(nodes, shape)
    frames = _frame_indices(len(times), frame_stride)

    figure, axes = plt.subplots(2, 4, figsize=(16.6, 7.4), constrained_layout=True)
    artists = []
    saturation_rows = []
    for axis, (name, title, group) in zip(axes.flat, panel_fields, strict=True):
        values = fields[name][..., component]
        limit = float(limits[group][component])
        if shape is None:
            artist = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=values[frames[0]],
                s=3.0,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                linewidths=0.0,
            )
        else:
            nx, ny = shape
            artist = axis.imshow(
                values[frames[0]].reshape(ny, nx),
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        axis.set_title(title, fontsize=9)
        axis.set_aspect("equal")
        figure.colorbar(artist, ax=axis, shrink=0.72, label=f"{group} scale")
        artists.append((artist, shape, name))
        saturation_rows.append(
            {
                "family": family,
                "case_id": case_id,
                "resolution": resolution,
                "component": COMPONENTS[component],
                "field": name,
                "scale_group": group,
                "fixed_limit": limit,
                "maximum_absolute_scaled_value": float(np.max(np.abs(values))),
                "fraction_outside_limit": float(np.mean(np.abs(values) > limit)),
            }
        )
    title = figure.suptitle("")
    figure.text(
        0.5,
        0.005,
        "frozen component residual scales; exact population maxima; no clipping "
        "or per-frame normalization",
        ha="center",
        fontsize=8,
    )
    _watermark(figure, watermark)

    def update(frame: int) -> list[Any]:
        for artist, structured, name in artists:
            values = fields[name][frame, :, component]
            if structured is None:
                artist.set_array(values)
            else:
                nx, ny = structured
                artist.set_data(values.reshape(ny, nx))
        title.set_text(
            f"{family} {case_id} {resolution} {COMPONENTS[component]} "
            f"t={times[frame]:.6g} selected={candidate} gain={gain:g}"
        )
        return [artist for artist, _, _ in artists] + [title]

    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family}_{case_id}_{resolution}_{COMPONENTS[component]}"
    gif_path = output_dir / f"{stem}.gif"
    final_path = output_dir / f"{stem}_final.png"
    movie.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    update(frames[-1])
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [gif_path, final_path], saturation_rows


def _case_metric_rows(results_dir: Path) -> list[dict[str, Any]]:
    selections = sorted(
        _read_rows(results_dir / "evaluation_gain_selections.csv"),
        key=lambda row: row["case_id"],
    )
    comparisons = {
        (row["case_id"], row["comparison"]): row
        for row in _read_rows(results_dir / "evaluation_comparisons.csv")
    }
    replay = {
        row["case_id"]: row
        for row in _read_rows(results_dir / "evaluation_probe_replay.csv")
    }
    sequences: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(results_dir / "sequence_time_metrics.csv"):
        if row["defect_kind"] == "corrected_defect_on_own_recurrent_inputs":
            sequences[(row["case_id"], row["arm"])].append(row)
    final_sequence = {
        key: max(rows, key=lambda row: int(row["step"]))
        for key, rows in sequences.items()
    }
    projection_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    projection_path = results_dir / "projection_metrics.csv"
    if projection_path.is_file():
        for row in _read_rows(projection_path):
            if row["arm"] == "selected":
                projection_by_case[row["case_id"]].append(row)

    result = []
    for selection in selections:
        case_id = selection["case_id"]
        selected = comparisons[(case_id, "selected_vs_zero")]
        static = comparisons[(case_id, "static_vs_zero")]
        baseline_sequence = final_sequence[(case_id, "baseline")]
        selected_sequence = final_sequence[(case_id, "selected")]
        replay_row = replay[case_id]
        projected = projection_by_case.get(case_id, [])
        before_parallel = sum(float(row["before_parallel_energy"]) for row in projected)
        before_orthogonal = sum(
            float(row["before_orthogonal_energy"]) for row in projected
        )
        before_total = sum(float(row["before_total_energy"]) for row in projected)
        predicted = selection["predicted_endpoint_ratio"]
        result.append(
            {
                "case_id": case_id,
                "selected_candidate": selection["selected_candidate"],
                "selected_gain": float(selection["selected_gain"]),
                "selection_reason": selection["selection_reason"],
                "predicted_endpoint_ratio": None
                if predicted == ""
                else float(predicted),
                "selected_zero_endpoint_ratio": float(
                    selected["endpoint_state_ratio"]
                ),
                "selected_zero_residual_ratio": float(selected["residual_rms_ratio"]),
                "selected_zero_maximum_control_ratio": float(
                    selected["maximum_control_ratio"]
                ),
                "static_zero_endpoint_ratio": float(static["endpoint_state_ratio"]),
                "static_zero_residual_ratio": float(static["residual_rms_ratio"]),
                "static_zero_maximum_control_ratio": float(
                    static["maximum_control_ratio"]
                ),
                "prefix_absolute": float(replay_row["maximum_absolute"]),
                "prefix_relative_l2": float(replay_row["relative_l2"]),
                "prefix_absolute_threshold_multiple": float(
                    replay_row["maximum_absolute"]
                )
                / float(replay_row["absolute_limit"]),
                "prefix_relative_threshold_multiple": float(replay_row["relative_l2"])
                / float(replay_row["relative_limit"]),
                "final_cumulative_relative_zero": float(
                    baseline_sequence["cumulative_relative"]
                ),
                "final_cumulative_relative_selected": float(
                    selected_sequence["cumulative_relative"]
                ),
                "final_cumulative_defect_rms_selected_zero_ratio": float(
                    selected_sequence["cumulative_defect_rms"]
                )
                / max(float(baseline_sequence["cumulative_defect_rms"]), 1e-30),
                "final_temporal_coherence_zero": float(
                    baseline_sequence["temporal_coherence"]
                ),
                "final_temporal_coherence_selected": float(
                    selected_sequence["temporal_coherence"]
                ),
                "final_parallel_bias_zero": float(
                    baseline_sequence["cumulative_parallel_bias"]
                ),
                "final_parallel_bias_selected": float(
                    selected_sequence["cumulative_parallel_bias"]
                ),
                "selected_parallel_energy_ratio": None
                if before_parallel == 0.0
                else sum(float(row["after_parallel_energy"]) for row in projected)
                / before_parallel,
                "selected_orthogonal_energy_ratio": None
                if before_orthogonal == 0.0
                else sum(float(row["after_orthogonal_energy"]) for row in projected)
                / before_orthogonal,
                "selected_total_energy_ratio": None
                if before_total == 0.0
                else sum(float(row["after_total_energy"]) for row in projected)
                / before_total,
            }
        )
    return result


def _analysis_summary(
    results_dir: Path,
    summary: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in _read_rows(results_dir / "evaluation_comparisons.csv")
        if row["comparison"] == "selected_vs_zero"
    ]
    replay = _read_rows(results_dir / "evaluation_probe_replay.csv")
    return {
        "input_summary_sha256": sha256_file(results_dir / "summary.json"),
        "input_status": summary["status"],
        "scientific_interpretation_allowed": False
        if summary["status"] == "failed_contract"
        else bool(summary["scientific_interpretation_allowed"]),
        "case_count": len(selected),
        "nonzero_case_count": sum(
            float(row["selected_gain"]) > 0
            for row in _read_rows(results_dir / "evaluation_gain_selections.csv")
        ),
        "selected_vs_zero": {
            "median_endpoint_state_ratio": float(
                np.median([float(row["endpoint_state_ratio"]) for row in selected])
            ),
            "median_residual_rms_ratio": float(
                np.median([float(row["residual_rms_ratio"]) for row in selected])
            ),
            "maximum_control_ratio": float(
                np.max([float(row["maximum_control_ratio"]) for row in selected])
            ),
        },
        "prefix_replay": {
            "all_passed": all(row["passed"].lower() == "true" for row in replay),
            "maximum_absolute": float(
                np.max([float(row["maximum_absolute"]) for row in replay])
            ),
            "maximum_relative_l2": float(
                np.max([float(row["relative_l2"]) for row in replay])
            ),
            "absolute_threshold_multiples": [
                float(row["maximum_absolute"]) / float(row["absolute_limit"])
                for row in replay
            ],
            "relative_threshold_multiples": [
                float(row["relative_l2"]) / float(row["relative_limit"])
                for row in replay
            ],
        },
        "nonzero_selected_cases": {
            "case_ids": [
                row["case_id"] for row in case_rows if row["selected_gain"] > 0
            ],
            "median_endpoint_state_ratio": float(
                np.median(
                    [
                        row["selected_zero_endpoint_ratio"]
                        for row in case_rows
                        if row["selected_gain"] > 0
                    ]
                )
            ),
            "median_residual_rms_ratio": float(
                np.median(
                    [
                        row["selected_zero_residual_ratio"]
                        for row in case_rows
                        if row["selected_gain"] > 0
                    ]
                )
            ),
            "median_final_cumulative_defect_rms_ratio": float(
                np.median(
                    [
                        row["final_cumulative_defect_rms_selected_zero_ratio"]
                        for row in case_rows
                        if row["selected_gain"] > 0
                    ]
                )
            ),
            "parallel_energy_ratios": [
                row["selected_parallel_energy_ratio"]
                for row in case_rows
                if row["selected_gain"] > 0
            ],
            "orthogonal_energy_ratios": [
                row["selected_orthogonal_energy_ratio"]
                for row in case_rows
                if row["selected_gain"] > 0
            ],
        },
        "aggregation": (
            "all temporal curves and population summaries are formed from "
            "per-case scalars; nodewise values are never pooled across cases"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary, payload_paths, watermark = _verify_results(
        args.results_dir,
        allow_failed_contract=args.allow_failed_contract_diagnostics,
    )
    args.output_dir.mkdir(parents=True)
    generated: list[Path] = []
    if args.command in {"plots", "all"}:
        figure_dir = args.output_dir / "figures"
        generated.extend(plot_calibration_response(args.results_dir, figure_dir, watermark))
        generated.extend(plot_evaluation_summary(args.results_dir, figure_dir, watermark))
        generated.extend(plot_replay_failure(args.results_dir, figure_dir, watermark))
        generated.extend(plot_temporal_structure(args.results_dir, figure_dir, watermark))
        generated.extend(plot_subspace_response(args.results_dir, figure_dir, watermark))

    limits = None
    selected_payload_paths: list[Path] = []
    saturation_rows: list[dict[str, Any]] = []
    if args.command in {"animations", "all"}:
        panel_fields = (
            RESPONSE_PANEL_FIELDS
            if args.panel_set == "response"
            else PREDICTION_PANEL_FIELDS
        )
        limits, selected_payload_paths = _population_limits(
            payload_paths,
            summary,
            None if args.case_ids is None else set(args.case_ids),
            panel_fields,
        )
        animation_dir = args.output_dir / "animations"
        for payload_path in selected_payload_paths:
            for component in range(len(COMPONENTS)):
                paths, rows = animate_payload_component(
                    payload_path,
                    animation_dir,
                    summary,
                    limits,
                    component=component,
                    fps=args.fps,
                    dpi=args.dpi,
                    frame_stride=args.frame_stride,
                    watermark=watermark,
                    panel_fields=panel_fields,
                )
                generated.extend(paths)
                saturation_rows.extend(rows)
        saturation_path = args.output_dir / "visual_scale_saturation.csv"
        write_csv_with_paths(saturation_path, saturation_rows)
        generated.append(saturation_path)

    case_rows = _case_metric_rows(args.results_dir)
    case_metrics_path = args.output_dir / "case_metrics.csv"
    write_csv_with_paths(case_metrics_path, case_rows)
    generated.append(case_metrics_path)
    analysis = _analysis_summary(args.results_dir, summary, case_rows)
    analysis_path = args.output_dir / "analysis_summary.json"
    write_json(analysis_path, analysis)
    generated.append(analysis_path)
    output_hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in sorted(generated)
    }
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "args": jsonable_args(args),
        "family": summary["family"],
        "input_status": summary["status"],
        "input_contract_checks_passed": summary["contract_checks_passed"],
        "scientific_interpretation_allowed": analysis[
            "scientific_interpretation_allowed"
        ],
        "watermark": watermark,
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_output_hashes_verified": True,
        "aggregation": analysis["aggregation"],
        "scale_contract": {
            "residual_and_defect": "field divided by frozen D_r component scale",
            "cumulative": "state gap divided by frozen D_r component scale",
            "growth": "(2 e delta + delta^2) divided by D_r^2",
            "population_exact_max_by_group_and_component": None
            if limits is None
            else {key: value.tolist() for key, value in limits.items()},
            "percentile_clipping": False,
            "per_frame_normalization": False,
        },
        "panel_set": args.panel_set,
        "animation_payloads": [
            path.relative_to(args.results_dir.resolve()).as_posix()
            for path in selected_payload_paths
        ],
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
        f"response-controller visualization status={manifest['status']} "
        f"outputs={len(manifest['output_hashes'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
