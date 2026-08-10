#!/usr/bin/env python3
"""Render verified D074 native-correction metrics and fixed-scale fields."""

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

SCHEMA = "pcno_native_residual_correction_visualization_v1"
RESULT_SCHEMA = "pcno_native_residual_correction_diagnostic_v1"
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
COLORS = {
    "baseline": "#333333",
    "selected": "#0072B2",
    "same_input_base": "#D55E00",
    "rho": "#0072B2",
    "rho_u": "#D55E00",
    "rho_v": "#009E73",
    "energy": "#CC79A7",
}
SEQUENCE_SPECS = (
    (
        "baseline",
        "corrected_defect_on_own_recurrent_inputs",
        "baseline defect",
        COLORS["baseline"],
        "-",
    ),
    (
        "selected",
        "corrected_defect_on_own_recurrent_inputs",
        "corrected defect",
        COLORS["selected"],
        "-",
    ),
    (
        "selected",
        "base_defect_on_same_corrected_inputs",
        "same-input base defect",
        COLORS["same_input_base"],
        "--",
    ),
)
PANEL_FIELDS = (
    ("true_increment", "true increment r", "residual"),
    ("uncorrected_increment", "uncorrected increment r+b", "residual"),
    ("corrected_increment", "corrected increment r+b+C", "residual"),
    ("base_defect", "same-input defect b", "defect"),
    ("correction", "correction C", "defect"),
    ("corrected_defect", "corrected defect delta", "defect"),
    ("cumulative_error", "cumulative state gap e", "cumulative"),
    ("signed_growth", "signed local growth density", "growth"),
)
REQUIRED_TABLES = {
    "case_contracts.csv",
    "evaluation_call_metrics.csv",
    "evaluation_ratios.csv",
    "lag_correlations.csv",
    "pod_summaries.csv",
    "selector_rows.csv",
    "selector_summary.csv",
    "sequence_time_metrics.csv",
    "signed_component_budgets.csv",
    "visual_payload_inventory.csv",
}
D075_REQUIRED_TABLES = {"correction_integral_audit.csv"}
CORRECTION_POLICIES = {
    "raw",
    "energy_integral_neutral",
    "all_integrals_neutral",
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
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 30 or args.frame_stride < 1:
        raise ValueError("fps, dpi, and frame-stride must be positive")
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in {None, "", "None", "null"}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


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
        raise ValueError(
            f"D074 output path escapes results directory: {relative}"
        ) from error
    return path


def _verify_results(results_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected D074 result schema")
    status = summary.get("status")
    smoke = status == "smoke_complete"
    if status not in {"complete", "smoke_complete"} or not summary.get(
        "contract_checks_passed"
    ):
        raise ValueError("D074 results did not complete the core contract")
    if smoke and summary.get("scientific_interpretation_allowed"):
        raise ValueError("D074 smoke cannot allow scientific interpretation")
    if not smoke and not summary.get("scientific_interpretation_allowed"):
        raise ValueError("complete D074 results are not interpretable")

    required = set(REQUIRED_TABLES)
    experiment_contract = str(summary.get("experiment_contract", "d074_raw"))
    if experiment_contract not in {"d074_raw", "d075_integral_neutral"}:
        raise ValueError(
            f"unsupported native-correction experiment contract: {experiment_contract}"
        )
    if experiment_contract == "d075_integral_neutral":
        required.update(D075_REQUIRED_TABLES)
    selected = summary.get("selector", {}).get("selected", {})
    if int(selected.get("rank", 0)) > 0:
        required.update({"projection_metrics.csv", "projection_component_metrics.csv"})
    output_hashes = summary.get("output_hashes")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise ValueError("D074 summary contains no output hashes")
    missing = sorted(required - set(output_hashes))
    if missing:
        raise ValueError(f"D074 output hashes omit required tables: {missing}")
    for relative, expected in output_hashes.items():
        path = _contained_path(results_dir, str(relative))
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"D074 output digest mismatch: {relative}")

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
        raise ValueError("D074 visual payload inventories disagree")
    for row in inventory:
        relative = row["relative_path"]
        if row.get("sha256") != output_hashes.get(relative):
            raise ValueError(f"D074 visual inventory digest mismatch: {relative}")
    return summary, [
        _contained_path(results_dir, value) for value in sorted(hash_paths)
    ]


def _load_payload(path: Path, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    if _scalar(payload["schema"]) != RESULT_SCHEMA:
        raise ValueError("unexpected D074 visual payload schema")
    family = str(_scalar(payload["family"]))
    if family != summary.get("family"):
        raise ValueError("D074 payload family differs from summary")
    selected = str(_scalar(payload["selected_candidate"]))
    frozen = summary.get("selector", {}).get("selected", {})
    if selected != frozen.get("key"):
        raise ValueError("D074 payload candidate differs from frozen selector")
    selected_rank = int(_scalar(payload["selected_rank"]))
    if selected_rank != int(frozen.get("rank", -1)):
        raise ValueError("D074 payload rank differs from frozen selector")
    selected_gain = float(_scalar(payload["selected_gain"]))
    if not np.isclose(selected_gain, float(frozen.get("gain", np.nan))):
        raise ValueError("D074 payload gain differs from frozen selector")
    selected_policy = (
        str(_scalar(payload["selected_correction_policy"]))
        if "selected_correction_policy" in payload
        else "raw"
    )
    frozen_policy = str(frozen.get("correction_policy", "raw"))
    if selected_policy not in CORRECTION_POLICIES:
        raise ValueError("D075 payload correction policy is unsupported")
    if selected_policy != frozen_policy:
        raise ValueError("D075 payload correction policy differs from frozen selector")
    if tuple(payload["component_names"].tolist()) != COMPONENTS:
        raise ValueError("D074 payload component inventory changed")
    calls = int(_scalar(payload["expected_calls"]))
    if (
        calls < 1
        or not bool(_scalar(payload["baseline_complete"]))
        or not bool(_scalar(payload["selected_complete"]))
    ):
        raise ValueError("D074 payload is incomplete")
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    weights = np.asarray(payload["weights"], dtype=np.float64)
    node_type = np.asarray(payload["node_type"], dtype=np.int64)
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    if (
        nodes.ndim != 2
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
        raise ValueError("invalid D074 payload geometry, weights, clock, or scales")
    if family == "dynamic_fv" and not set(np.unique(node_type)) <= {0, 1, 2, 3}:
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
    )
    expected_shape = (calls, nodes.shape[0], len(COMPONENTS))
    for name in fields:
        value = np.asarray(payload[name])
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid D074 visual field: {name}")

    defect = np.asarray(payload["corrected_defect"], dtype=np.float64)
    cumulative = np.asarray(payload["cumulative_error"], dtype=np.float64)
    if not np.allclose(cumulative, np.cumsum(defect, axis=0), rtol=0.0, atol=2.0e-7):
        raise ValueError("stored D074 cumulative-error replay failed")
    previous = np.concatenate((np.zeros_like(cumulative[:1]), cumulative[:-1]), axis=0)
    growth = (2.0 * previous * defect + np.square(defect)) / np.square(
        scale[None, None, :]
    )
    stored_growth = np.asarray(
        payload["signed_growth_density_scaled"], dtype=np.float64
    )
    if not np.allclose(stored_growth, growth, rtol=2.0e-5, atol=2.0e-6):
        raise ValueError("stored D074 signed-growth density replay failed")
    contribution = growth * weights[None, :, None] / float(weights.sum())
    stored_contribution = np.asarray(
        payload["signed_growth_contribution"], dtype=np.float64
    )
    if not np.allclose(stored_contribution, contribution, rtol=2.0e-5, atol=2.0e-8):
        raise ValueError("stored D074 signed-growth contribution replay failed")
    return payload


def _derived_fields(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)[None, None, :]
    truth = np.asarray(payload["true_increment"], dtype=np.float64)
    base = np.asarray(payload["corrected_input_base_defect"], dtype=np.float64)
    correction = np.asarray(payload["correction"], dtype=np.float64)
    corrected = np.asarray(payload["corrected_defect"], dtype=np.float64)
    return {
        "true_increment": truth / scale,
        "uncorrected_increment": (truth + base) / scale,
        "corrected_increment": (truth + corrected) / scale,
        "base_defect": base / scale,
        "correction": correction / scale,
        "corrected_defect": corrected / scale,
        "cumulative_error": np.asarray(payload["cumulative_error"], dtype=np.float64)
        / scale,
        "signed_growth": np.asarray(
            payload["signed_growth_density_scaled"], dtype=np.float64
        ),
    }


def _frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    if frame_count < 1 or stride < 1:
        raise ValueError("frame count and stride must be positive")
    values = list(range(0, frame_count, stride))
    if values[-1] != frame_count - 1:
        values.append(frame_count - 1)
    return tuple(values)


def _case_aggregated_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    filters: Mapping[str, str],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    by_case: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if any(str(row.get(key)) != str(value) for key, value in filters.items()):
            continue
        x = _number(str(row.get(x_key, "")))
        y = _number(str(row.get(y_key, "")))
        if x is not None and y is not None:
            by_case[str(row["case_id"])].append((x, y))
    case_series = {}
    grouped: dict[float, list[float]] = defaultdict(list)
    for case_id, values in by_case.items():
        ordered = sorted(values)
        x = np.asarray([value[0] for value in ordered], dtype=np.float64)
        y = np.asarray([value[1] for value in ordered], dtype=np.float64)
        case_series[case_id] = (x, y)
        for time, result in ordered:
            grouped[time].append(result)
    times = np.asarray(sorted(grouped), dtype=np.float64)
    median = np.asarray([np.median(grouped[time]) for time in times])
    lower = np.asarray([np.quantile(grouped[time], 0.25) for time in times])
    upper = np.asarray([np.quantile(grouped[time], 0.75) for time in times])
    return case_series, times, median, lower, upper


def _draw_case_aggregate(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    label: str,
    color: str,
    linestyle: str = "-",
    filters: Mapping[str, str] | None = None,
) -> None:
    case_series, times, median, lower, upper = _case_aggregated_series(
        rows, x_key=x_key, y_key=y_key, filters=filters or {}
    )
    for x, y in case_series.values():
        axis.plot(x, y, color=color, linestyle=linestyle, alpha=0.22, linewidth=0.8)
    if times.size:
        axis.fill_between(times, lower, upper, color=color, alpha=0.12, linewidth=0)
        axis.plot(
            times,
            median,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )


def _style_axis(axis: plt.Axes, *, zero_line: bool = False) -> None:
    if zero_line:
        axis.axhline(0.0, color="#888888", linewidth=0.7)
    axis.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axis.set_xlabel("physical time")


def _watermark(figure: plt.Figure, smoke: bool) -> None:
    if smoke:
        figure.text(
            0.5,
            0.5,
            "H2 SMOKE — NON-SCIENTIFIC",
            ha="center",
            va="center",
            fontsize=28,
            color="#8B0000",
            alpha=0.11,
            rotation=24,
            weight="bold",
        )


def _save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, *, smoke: bool
) -> list[Path]:
    _watermark(figure, smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _time_joined_sequence_rows(results_dir: Path) -> list[dict[str, Any]]:
    contracts = _read_rows(results_dir / "case_contracts.csv")
    time_steps = {
        (row["family"], row["case_id"], row["resolution"]): float(row["time_step"])
        for row in contracts
    }
    result = []
    for raw in _read_rows(results_dir / "sequence_time_metrics.csv"):
        row: dict[str, Any] = dict(raw)
        key = (row["family"], row["case_id"], row["resolution"])
        if key not in time_steps:
            raise ValueError(f"sequence row lacks a case clock: {key}")
        row["physical_time"] = int(row["step"]) * time_steps[key]
        result.append(row)
    return result


def plot_trajectory_metrics(
    results_dir: Path, output_dir: Path, *, selected_zero: bool, smoke: bool
) -> list[Path]:
    call_rows: list[dict[str, Any]] = _read_rows(
        results_dir / "evaluation_call_metrics.csv"
    )
    sequence_rows = _time_joined_sequence_rows(results_dir)
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    _draw_case_aggregate(
        axes[0, 0],
        call_rows,
        x_key="physical_time",
        y_key="state_error_rms",
        label="baseline",
        color=COLORS["baseline"],
        filters={"arm": "baseline"},
    )
    _draw_case_aggregate(
        axes[0, 0],
        call_rows,
        x_key="physical_time",
        y_key="state_error_rms",
        label="selected",
        color=COLORS["selected"],
        filters={"arm": "selected"},
    )
    axes[0, 0].set_title("state-scale rollout error")
    if selected_zero:
        axes[0, 0].text(
            0.03,
            0.95,
            "selected zero: curves overlap",
            transform=axes[0, 0].transAxes,
            va="top",
            fontsize=8,
        )
    metrics = (
        ("instant_defect_rms", "instantaneous defect RMS"),
        ("instant_relative", "defect / true increment"),
        ("cumulative_defect_rms", "cumulative defect RMS"),
        ("cumulative_relative", "cumulative / truth change"),
        ("temporal_coherence", "temporal coherence"),
    )
    for axis, (metric, title) in zip(axes.flat[1:], metrics, strict=True):
        for arm, kind, label, color, linestyle in SEQUENCE_SPECS:
            _draw_case_aggregate(
                axis,
                sequence_rows,
                x_key="physical_time",
                y_key=metric,
                label=label,
                color=color,
                linestyle=linestyle,
                filters={"arm": arm, "defect_kind": kind},
            )
        axis.set_title(title)
    for axis in axes.flat:
        _style_axis(axis)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    return _save_figure(figure, output_dir, "trajectory_metrics", smoke=smoke)


def plot_signed_growth_alignment(
    results_dir: Path, output_dir: Path, *, selected_zero: bool, smoke: bool
) -> list[Path]:
    rows: list[dict[str, Any]] = _read_rows(results_dir / "evaluation_call_metrics.csv")
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0, 0], "signed_interaction", "interaction 2<e,delta>"),
        (axes[0, 1], "defect_energy", "innovation ||delta||^2"),
        (axes[0, 2], "squared_error_growth", "net squared-error growth"),
    ):
        for arm in ("baseline", "selected"):
            _draw_case_aggregate(
                axis,
                rows,
                x_key="physical_time",
                y_key=metric,
                label=arm,
                color=COLORS[arm],
                filters={"arm": arm},
            )
        axis.set_title(title)
        _style_axis(axis, zero_line=True)
    if selected_zero:
        axes[1, 0].text(
            0.5,
            0.5,
            "N/A: C=0\ncosine undefined",
            transform=axes[1, 0].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        _draw_case_aggregate(
            axes[1, 0],
            rows,
            x_key="physical_time",
            y_key="correction_base_cosine",
            label="cos(C,b)",
            color=COLORS["selected"],
            filters={"arm": "selected"},
        )
    axes[1, 0].set_title("same-input correction alignment")
    for metric, label, color, linestyle in (
        ("twice_correction_base_inner", "2<C,b>", "#D55E00", "-"),
        ("correction_energy", "||C||^2", "#009E73", "--"),
    ):
        _draw_case_aggregate(
            axes[1, 1],
            rows,
            x_key="physical_time",
            y_key=metric,
            label=label,
            color=color,
            linestyle=linestyle,
            filters={"arm": "selected"},
        )
    axes[1, 1].set_title("same-input correction terms")
    _draw_case_aggregate(
        axes[1, 2],
        rows,
        x_key="physical_time",
        y_key="corrected_minus_base_energy",
        label="||b+C||² - ||b||²",
        color="#CC79A7",
        filters={"arm": "selected"},
    )
    axes[1, 2].set_title("same-input defect-energy change")
    for axis in axes[1]:
        _style_axis(axis, zero_line=True)
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    return _save_figure(figure, output_dir, "signed_growth_alignment", smoke=smoke)


def _summed_component_rows(
    rows: Sequence[Mapping[str, str]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {metric: 0.0 for metric in metrics}
    )
    for row in rows:
        key = (
            row["family"],
            row["case_id"],
            row["resolution"],
            row["arm"],
            row["physical_time"],
        )
        for metric in metrics:
            value = _number(row.get(metric))
            if value is None:
                raise ValueError(f"component budget is missing {metric}")
            grouped[key][metric] += value
    result = []
    for key, values in sorted(grouped.items()):
        family, case_id, resolution, arm, physical_time = key
        result.append(
            {
                "family": family,
                "case_id": case_id,
                "resolution": resolution,
                "arm": arm,
                "physical_time": physical_time,
                **values,
            }
        )
    return result


def plot_subspace_component_budgets(
    results_dir: Path, output_dir: Path, *, smoke: bool
) -> list[Path]:
    projection: list[dict[str, Any]] = _read_rows(
        results_dir / "projection_metrics.csv"
    )
    components: list[dict[str, str]] = _read_rows(
        results_dir / "projection_component_metrics.csv"
    )
    budgets: list[dict[str, Any]] = _read_rows(
        results_dir / "signed_component_budgets.csv"
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    for metric, label, color, linestyle in (
        ("before_parallel_energy", "parallel before", "#D55E00", "-"),
        ("after_parallel_energy", "parallel after", "#0072B2", "--"),
    ):
        _draw_case_aggregate(
            axes[0, 0],
            projection,
            x_key="physical_time",
            y_key=metric,
            label=label,
            color=color,
            linestyle=linestyle,
            filters={"arm": "selected"},
        )
    axes[0, 0].set_title("selected-subspace energy")
    for metric, label, color, linestyle in (
        ("before_orthogonal_energy", "type-0 orthogonal before", "#009E73", "-"),
        ("after_orthogonal_energy", "type-0 orthogonal after", "#009E73", "--"),
        (
            "before_excluded_non_type0_energy",
            "excluded support before",
            "#CC79A7",
            "-",
        ),
        (
            "after_excluded_non_type0_energy",
            "excluded support after",
            "#CC79A7",
            "--",
        ),
    ):
        _draw_case_aggregate(
            axes[0, 1],
            projection,
            x_key="physical_time",
            y_key=metric,
            label=label,
            color=color,
            linestyle=linestyle,
            filters={"arm": "selected"},
        )
    axes[0, 1].set_title("untargeted energy partitions")
    for component in COMPONENTS:
        _draw_case_aggregate(
            axes[0, 2],
            components,
            x_key="physical_time",
            y_key="correction_total_energy",
            label=component,
            color=COLORS[component],
            filters={"arm": "selected", "component": component},
        )
    axes[0, 2].set_title("correction energy by component")

    summed = _summed_component_rows(
        components,
        (
            "correction_constant_mode_energy",
            "correction_nonconstant_modes_energy",
            "twice_constant_nonconstant_inner",
        ),
    )
    for metric, label, color, linestyle in (
        ("correction_constant_mode_energy", "constant", "#0072B2", "-"),
        ("correction_nonconstant_modes_energy", "nonconstant", "#D55E00", "--"),
        ("twice_constant_nonconstant_inner", "cross term", "#009E73", ":"),
    ):
        _draw_case_aggregate(
            axes[1, 0],
            summed,
            x_key="physical_time",
            y_key=metric,
            label=label,
            color=color,
            linestyle=linestyle,
            filters={"arm": "selected"},
        )
    axes[1, 0].set_title("constant/nonconstant correction energy")
    for axis, field, title in (
        (axes[1, 1], "correction", "signed correction integral"),
        (axes[1, 2], "cumulative_correction", "signed cumulative correction"),
    ):
        for component in COMPONENTS:
            _draw_case_aggregate(
                axis,
                budgets,
                x_key="physical_time",
                y_key="scaled_signed_quantity",
                label=component,
                color=COLORS[component],
                filters={
                    "arm": "selected",
                    "field": field,
                    "component": component,
                },
            )
        axis.set_title(title)
    for axis in axes.flat:
        _style_axis(axis, zero_line=axis in axes[1])
        axis.legend(fontsize=7)
    return _save_figure(figure, output_dir, "subspace_component_budgets", smoke=smoke)


def _selector_display_medians(
    results_dir: Path, selector: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[float, float] | None]:
    """Return raw per-case medians even when a candidate is ineligible."""

    values: dict[str, list[tuple[float, float] | None]] = {}
    for row in _read_rows(results_dir / "selector_rows.csv"):
        endpoint = _number(row.get("endpoint_state_ratio"))
        residual = _number(row.get("residual_rms_ratio"))
        value = (
            (endpoint, residual)
            if row.get("complete") == "True"
            and endpoint is not None
            and residual is not None
            else None
        )
        values.setdefault(row["candidate"], []).append(value)
    medians: dict[str, tuple[float, float] | None] = {}
    for row in selector:
        candidate = row["candidate"]
        candidate_values = values.get(candidate, [])
        expected_count = int(row.get("case_count", len(candidate_values)))
        if (
            len(candidate_values) != expected_count
            or not candidate_values
            or any(value is None for value in candidate_values)
        ):
            medians[candidate] = None
            continue
        array = np.asarray(candidate_values, dtype=np.float64)
        medians[candidate] = tuple(np.median(array, axis=0))
        official = (
            _number(row.get("median_endpoint_state_ratio")),
            _number(row.get("median_residual_rms_ratio")),
        )
        if all(value is not None for value in official) and not np.allclose(
            medians[candidate], official, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"selector display median disagrees for {candidate}")
    return medians


def plot_selection_no_harm(
    results_dir: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    smoke: bool,
) -> list[Path]:
    selector = _read_rows(results_dir / "selector_summary.csv")
    ratios = _read_rows(results_dir / "evaluation_ratios.csv")
    if not ratios:
        raise ValueError("D074 evaluation ratios are empty")
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.8), constrained_layout=True)
    selected_key = summary["selector"]["selected"]["key"]
    display_medians = _selector_display_medians(results_dir, selector)
    omitted = []
    for row in selector:
        display = display_medians[row["candidate"]]
        if display is None:
            omitted.append(row["candidate"])
            continue
        x, y = display
        selected = row["candidate"] == selected_key
        eligible = row.get("eligible") == "True"
        axes[0].scatter(
            [x],
            [y],
            s=70 if selected else 38,
            marker="*" if selected else "o",
            color="#0072B2" if eligible else "#999999",
            edgecolor="#222222",
            linewidth=0.5,
        )
        axes[0].annotate(
            row["candidate"],
            (x, y),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[0].axvline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].scatter([], [], color="#0072B2", label="eligible selector score")
    axes[0].scatter(
        [], [], color="#999999", label="ineligible: raw display median only"
    )
    axes[0].set_xlabel("raw median endpoint-state ratio")
    axes[0].set_ylabel("raw median residual-RMS ratio")
    axes[0].set_title("calibration raw medians")
    axes[0].legend(fontsize=7)
    if omitted:
        axes[0].text(
            0.02,
            0.02,
            f"omitted incomplete candidates: {len(omitted)}",
            transform=axes[0].transAxes,
            fontsize=7,
        )
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)

    case_ids = [row["case_id"] for row in ratios]
    x = np.arange(len(case_ids), dtype=np.float64)
    width = 0.36
    state_values = np.asarray([float(row["final_state_ratio"]) for row in ratios])
    residual_values = np.asarray([float(row["residual_rms_ratio"]) for row in ratios])
    axes[1].bar(
        x - width / 2, state_values, width, label="endpoint state", color="#0072B2"
    )
    axes[1].bar(
        x + width / 2, residual_values, width, label="residual RMS", color="#D55E00"
    )
    axes[1].axhline(1.0, color="#333333", linewidth=0.9)
    axes[1].axhline(1.05, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, case_ids, rotation=35, ha="right")
    axes[1].set_ylabel("selected / baseline")
    axes[1].set_title("evaluation primary ratios")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.65)

    control_maps = [json.loads(row["control_ratios_json"]) for row in ratios]
    control_keys = sorted(control_maps[0])
    if any(sorted(values) != control_keys for values in control_maps):
        raise ValueError("D074 cases do not share one control inventory")
    matrix = np.asarray(
        [[float(values[key]) for key in control_keys] for values in control_maps]
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("D074 control heatmap contains nonfinite ratios")
    limit = max(1.05, float(matrix.max()))
    artist = axes[2].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=limit,
    )
    axes[2].set_yticks(np.arange(len(case_ids)), case_ids)
    short_keys = [
        key.replace("physical_volume_integral_state_scale_", "integral_")
        .replace("component_residual_rms__", "residual_")
        .replace("endpoint_state__", "endpoint_")
        for key in control_keys
    ]
    axes[2].set_xticks(
        np.arange(len(control_keys)), short_keys, rotation=70, ha="right", fontsize=6
    )
    axes[2].set_title("primary no-harm controls")
    figure.colorbar(artist, ax=axes[2], label="selected / baseline")
    if matrix.size <= 120:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axes[2].text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=(
                        "white"
                        if matrix[row_index, column_index] > 0.72 * limit
                        else "black"
                    ),
                )
    return _save_figure(figure, output_dir, "selection_no_harm", smoke=smoke)


def _sequence_label(arm: str, defect_kind: str) -> str:
    if arm == "baseline":
        return "baseline delta"
    if defect_kind == "base_defect_on_same_corrected_inputs":
        return "selected-input b"
    return "selected delta"


def _minimum_case_calls(results_dir: Path) -> int:
    calls = [
        int(row["calls"]) for row in _read_rows(results_dir / "case_contracts.csv")
    ]
    if not calls or min(calls) < 1:
        raise ValueError("D074 case contracts contain no positive call count")
    return min(calls)


def plot_temporal_rank_structure(
    results_dir: Path, output_dir: Path, *, smoke: bool
) -> list[Path]:
    calls = _minimum_case_calls(results_dir)
    if calls < 3:
        figure, axis = plt.subplots(figsize=(8.4, 3.4), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.56,
            f"N/A: only {calls} rollout calls are available.",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
        )
        axis.text(
            0.5,
            0.39,
            "Lag correlation and centered POD are algebraically underresolved at H2; "
            "interpret them only after H30.",
            ha="center",
            va="center",
            fontsize=10,
        )
        return _save_figure(figure, output_dir, "temporal_rank_structure", smoke=smoke)
    lag: list[dict[str, Any]] = _read_rows(results_dir / "lag_correlations.csv")
    pod = _read_rows(results_dir / "pod_summaries.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for arm, kind, label, color, linestyle in SEQUENCE_SPECS:
        _draw_case_aggregate(
            axes[0],
            lag,
            x_key="lag",
            y_key="correlation",
            label=label,
            color=color,
            linestyle=linestyle,
            filters={"arm": arm, "defect_kind": kind},
        )
    axes[0].set_title("temporal lag correlation")
    axes[0].set_xlabel("lag")
    axes[0].set_ylabel("correlation")
    axes[0].axhline(0.0, color="#888888", linewidth=0.7)
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axes[0].legend(fontsize=7)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pod:
        groups[
            (_sequence_label(row["arm"], row["defect_kind"]), row["centering"])
        ].append(row)
    labels = [f"{name}\n{centering}" for name, centering in sorted(groups)]
    first = [
        np.median([float(row["first_mode_energy_fraction"]) for row in groups[key]])
        for key in sorted(groups)
    ]
    first_three = [
        np.median([float(row["first_three_energy_fraction"]) for row in groups[key]])
        for key in sorted(groups)
    ]
    modes = [
        np.median([float(row["modes_for_95_percent"]) for row in groups[key]])
        for key in sorted(groups)
    ]
    x = np.arange(len(labels), dtype=np.float64)
    axes[1].bar(x - 0.18, first, 0.36, label="first mode", color="#0072B2")
    axes[1].bar(x + 0.18, first_three, 0.36, label="first three", color="#D55E00")
    axes[1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("energy fraction")
    axes[1].set_title("POD concentration")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axes[2].bar(x, modes, color="#009E73")
    axes[2].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[2].set_ylabel("modes")
    axes[2].set_title("modes for 95% energy")
    axes[2].grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.65)
    return _save_figure(figure, output_dir, "temporal_rank_structure", smoke=smoke)


def plot_subspace_not_applicable(output_dir: Path, *, smoke: bool) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        "ZERO selected: no nonzero correction subspace was applied.",
        ha="center",
        va="center",
        fontsize=12,
    )
    return _save_figure(
        figure, output_dir, "subspace_component_budgets_not_applicable", smoke=smoke
    )


def _population_limits(
    payload_paths: Sequence[Path],
    summary: Mapping[str, Any],
    case_ids: set[str] | None,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    limits = {
        name: np.zeros(len(COMPONENTS), dtype=np.float64)
        for name in ("residual", "defect", "cumulative", "growth")
    }
    selected_paths = []
    observed_cases = set()
    for path in payload_paths:
        payload = _load_payload(path, summary)
        case_id = str(_scalar(payload["case_id"]))
        observed_cases.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue
        selected_paths.append(path)
        fields = _derived_fields(payload)
        for name, _, group in PANEL_FIELDS:
            maximum = np.max(np.abs(fields[name]), axis=(0, 1))
            limits[group] = np.maximum(limits[group], maximum)
    if case_ids is not None:
        unknown = sorted(case_ids - observed_cases)
        if unknown:
            raise ValueError(
                f"requested D074 visualization cases are absent: {unknown}"
            )
    if not selected_paths:
        raise ValueError("no D074 visual payloads selected")
    for value in limits.values():
        value[value == 0.0] = 1.0
    return limits, selected_paths


def _structured_shape(resolution: str, node_count: int) -> tuple[int, int] | None:
    try:
        nx, ny = (int(value) for value in resolution.lower().split("x"))
    except (TypeError, ValueError):
        return None
    return (nx, ny) if nx * ny == node_count else None


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
    smoke: bool,
) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = _load_payload(payload_path, summary)
    fields = _derived_fields(payload)
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    family = str(_scalar(payload["family"]))
    case_id = str(_scalar(payload["case_id"]))
    resolution = str(_scalar(payload["resolution"]))
    candidate = str(_scalar(payload["selected_candidate"]))
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    shape = _structured_shape(resolution, nodes.shape[0])
    extent = None if shape is None else _structured_extent(nodes, shape)
    frame_indices = _frame_indices(len(times), frame_stride)

    figure, axes = plt.subplots(2, 4, figsize=(16.6, 7.4), constrained_layout=True)
    artists: list[tuple[Any, tuple[int, int] | None, str]] = []
    saturation_rows = []
    for axis, (name, title, group) in zip(axes.flat, PANEL_FIELDS, strict=True):
        values = fields[name][..., component]
        limit = float(limits[group][component])
        if shape is None:
            artist = axis.scatter(
                nodes[:, 0],
                nodes[:, 1],
                c=values[frame_indices[0]],
                s=3.0,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                linewidths=0.0,
            )
        else:
            nx, ny = shape
            artist = axis.imshow(
                values[frame_indices[0]].reshape(ny, nx),
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
        absolute = np.abs(values)
        saturation_rows.append(
            {
                "family": family,
                "case_id": case_id,
                "resolution": resolution,
                "component": COMPONENTS[component],
                "field": name,
                "scale_group": group,
                "fixed_limit": limit,
                "maximum_absolute_scaled_value": float(absolute.max()),
                "fraction_outside_limit": float(np.mean(absolute > limit)),
            }
        )
    title = figure.suptitle("")
    footer = (
        "population-fixed residual scales; defect panels use a separate shared scale; "
        "no percentile clipping or per-frame normalization"
    )
    figure.text(0.5, 0.005, footer, ha="center", fontsize=8)
    _watermark(figure, smoke)

    def update(frame: int) -> list[Any]:
        for artist, structured, name in artists:
            values = fields[name][frame, :, component]
            if structured is None:
                artist.set_array(values)
            else:
                nx, ny = structured
                artist.set_data(values.reshape(ny, nx))
        suffix = " | H2 SMOKE — NON-SCIENTIFIC" if smoke else ""
        title.set_text(
            f"{family} {case_id} {resolution} {COMPONENTS[component]} "
            f"t={times[frame]:.6g} selected={candidate}{suffix}"
        )
        return [artist for artist, _, _ in artists] + [title]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        blit=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family}_{case_id}_{resolution}_{COMPONENTS[component]}"
    gif_path = output_dir / f"{stem}.gif"
    final_path = output_dir / f"{stem}_final.png"
    movie.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    update(frame_indices[-1])
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [gif_path, final_path], saturation_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary, payload_paths = _verify_results(args.results_dir)
    smoke = summary["status"] == "smoke_complete"
    selected_rank = int(summary["selector"]["selected"]["rank"])
    selected_zero = selected_rank == 0
    args.output_dir.mkdir(parents=True)
    generated: list[Path] = []
    if args.command in {"plots", "all"}:
        figure_dir = args.output_dir / "figures"
        generated.extend(
            plot_trajectory_metrics(
                args.results_dir,
                figure_dir,
                selected_zero=selected_zero,
                smoke=smoke,
            )
        )
        generated.extend(
            plot_signed_growth_alignment(
                args.results_dir,
                figure_dir,
                selected_zero=selected_zero,
                smoke=smoke,
            )
        )
        if selected_rank > 0:
            generated.extend(
                plot_subspace_component_budgets(
                    args.results_dir, figure_dir, smoke=smoke
                )
            )
        else:
            generated.extend(plot_subspace_not_applicable(figure_dir, smoke=smoke))
        generated.extend(
            plot_selection_no_harm(args.results_dir, figure_dir, summary, smoke=smoke)
        )
        generated.extend(
            plot_temporal_rank_structure(args.results_dir, figure_dir, smoke=smoke)
        )

    limits = None
    selected_payload_paths: list[Path] = []
    saturation_rows: list[dict[str, Any]] = []
    if args.command in {"animations", "all"}:
        limits, selected_payload_paths = _population_limits(
            payload_paths,
            summary,
            None if args.case_ids is None else set(args.case_ids),
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
                    smoke=smoke,
                )
                generated.extend(paths)
                saturation_rows.extend(rows)
        saturation_path = args.output_dir / "visual_scale_saturation.csv"
        write_csv_with_paths(saturation_path, saturation_rows)
        generated.append(saturation_path)

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
        "scientific_interpretation_allowed": summary[
            "scientific_interpretation_allowed"
        ],
        "watermark": "H2 SMOKE — NON-SCIENTIFIC" if smoke else None,
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_output_hashes_verified": True,
        "aggregation": (
            "nodewise fields are reduced within case; plot medians/IQR are then "
            "formed across case-level scalars; resolutions are never pooled"
        ),
        "scale_contract": {
            "units": {
                "residual": "field divided by frozen D_r component scale",
                "defect": "field divided by frozen D_r component scale",
                "cumulative": "state gap divided by frozen D_r component scale",
                "growth": "(2 e delta + delta^2) divided by D_r^2",
            },
            "population_exact_max_by_group_and_component": (
                None
                if limits is None
                else {key: value.tolist() for key, value in limits.items()}
            ),
            "residual_panels_share_scale": True,
            "defect_panels_share_scale": True,
            "percentile_clipping": False,
            "per_frame_normalization": False,
        },
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
        f"D074 visualization status={manifest['status']} "
        f"outputs={len(manifest['output_hashes'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
