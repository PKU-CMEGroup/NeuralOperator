#!/usr/bin/env python3
"""Render fixed-contract D073-A radius, metric, and same-hidden defect views."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_fine_grained_pathways import (
    CALLS,
    CASE_IDS,
)
from scripts.time_dependent_no.analyze_pcno_physical_radius_geometry import (
    MODES,
    VISUAL_CASE_IDS,
)
from scripts.time_dependent_no.analyze_pcno_physical_radius_geometry import (
    SCHEMA as RESULT_SCHEMA,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_resolution_transfer import parse_resolution

SCHEMA = "pcno_physical_radius_geometry_visualization_v2"
RESULT_MANIFEST_SCHEMA = "pcno_physical_radius_geometry_manifest_v2"
REQUIRED_RESULT_FILES = (
    "geometry_inventory.csv",
    "operator_checks.csv",
    "arm_metrics.csv",
    "component_metrics.csv",
    "case_aggregates.csv",
    "gate_summary.csv",
    "closure_checks.csv",
    "completion.csv",
    "reference_checks.csv",
    "replay_metrics.csv",
)
PAIRS = ("125x50->250x100", "250x100->500x200")
ABSOLUTE_RMS_LIMIT = 2.0
RATIO_LIMIT = 2.0
PHYSICAL_SCALE_MULTIPLIER = 1.0
COMPONENT_LABELS = ("rho", "rho_u", "rho_v", "E")
ARM_COLORS = {"A1": "#D55E00", "A2": "#0072B2"}
ARM_COLORS["A0"] = "#555555"
DENOMINATOR_MINIMUM = 1.0e-8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=70)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 20:
        raise ValueError("fps and dpi must be positive visualization values")
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str | None) -> float | None:
    if value in {None, "", "None", "null"}:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _expected_payloads() -> tuple[str, ...]:
    return tuple(
        f"visual_payloads/{case_id}__{pair.replace('->', '_to_')}__{mode}.npz"
        for case_id, pair, mode in product(VISUAL_CASE_IDS, PAIRS, MODES)
    )


def _expected_rendered_outputs() -> set[str]:
    names = {"radius_support_inventory.png", "radius_support_inventory.pdf"}
    for pair, mode in product(PAIRS, MODES):
        safe_pair = pair.replace("->", "_to_")
        for stem in (
            f"a2_vs_a1_band_evolution_{safe_pair}_{mode}",
            f"a2_vs_a1_case_signs_{safe_pair}_{mode}",
        ):
            names.add(f"{stem}.png")
            names.add(f"{stem}.pdf")
    for relative in _expected_payloads():
        stem = Path(relative).stem
        names.add(f"{stem}__full.gif")
        names.add(f"{stem}__scale_split.gif")
    names.add("visual_scale_saturation_by_call.csv")
    return names


def _verify_results(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    manifest_path = results_dir / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != RESULT_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("scientific_interpretation_allowed") is not True
    ):
        raise ValueError("D073 results are not a complete interpretable full run")
    if manifest.get("schema") != RESULT_MANIFEST_SCHEMA:
        raise ValueError("unexpected D073 result manifest schema")
    expected = set(REQUIRED_RESULT_FILES) | set(_expected_payloads())
    output_hashes = summary.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != expected:
        raise ValueError("D073 result output inventory mismatch")
    manifest_hashes = manifest.get("output_hashes")
    if not isinstance(manifest_hashes, dict) or set(manifest_hashes) != expected | {
        "summary.json"
    }:
        raise ValueError("D073 result manifest inventory mismatch")
    summary_sha = sha256_file(summary_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("result_schema") != RESULT_SCHEMA
        or manifest.get("science_result") is not True
        or manifest.get("summary_sha256") != summary_sha
        or manifest.get("output_count") != len(manifest_hashes)
        or manifest_hashes.get("summary.json") != summary_sha
    ):
        raise ValueError("D073 summary digest differs from its manifest")
    contract = summary.get("contract", {})
    if (
        contract.get("diagnostic_calls") != list(CALLS)
        or contract.get("modes") != list(MODES)
        or contract.get("visual_case_ids") != list(VISUAL_CASE_IDS)
        or summary.get("population", {}).get("case_ids") != list(CASE_IDS)
    ):
        raise ValueError("D073 summary contract fields differ from the visual contract")
    for relative, expected_sha in output_hashes.items():
        path = results_dir / relative
        if (
            not path.is_file()
            or sha256_file(path) != expected_sha
            or manifest_hashes.get(relative) != expected_sha
        ):
            raise ValueError(f"D073 result digest mismatch: {relative}")
    disk_files = {
        path.relative_to(results_dir).as_posix()
        for path in results_dir.rglob("*")
        if path.is_file()
    }
    if disk_files != expected | {"summary.json", "manifest.json"}:
        raise ValueError("D073 result disk inventory mismatch")
    return summary


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    paths = (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return list(paths)


def plot_radius_support_inventory(
    geometry_rows: Sequence[Mapping[str, str]],
    operator_rows: Sequence[Mapping[str, str]],
    output_stem: Path,
) -> list[Path]:
    resolutions = ["125x50", "250x100", "500x200"]
    geometry = {row["resolution"]: row for row in geometry_rows}
    operators = {(row["resolution"], row["arm"]): row for row in operator_rows}
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)
    x = np.arange(len(resolutions))
    local_radii = [
        float(geometry[value]["local_two_hop_radius"]) for value in resolutions
    ]
    fixed_radii = [
        float(geometry[value]["fixed_training_radius"]) for value in resolutions
    ]
    axes[0, 0].plot(x, local_radii, marker="o", label="A1 local radius")
    axes[0, 0].plot(x, fixed_radii, marker="s", label="A2 fixed radius")
    axes[0, 0].set_ylabel("physical graph radius")
    axes[0, 0].legend(fontsize=8)
    for arm, offset in (("A1", -0.12), ("A2", 0.12)):
        axes[0, 1].bar(
            x + offset,
            [
                float(operators[(value, arm)]["neighbor_count_median"])
                for value in resolutions
            ],
            width=0.24,
            label=arm,
        )
        axes[1, 0].bar(
            x + offset,
            [
                float(operators[(value, arm)]["support_weight_median"])
                for value in resolutions
            ],
            width=0.24,
            label=arm,
        )
    axes[0, 1].set_ylabel("median neighbors")
    axes[1, 0].set_ylabel("median support weight")
    changed = [
        float(operators[(value, "A1")]["changed_row_fraction"]) for value in resolutions
    ]
    axes[1, 1].bar(x, changed, color="#009E73")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("A1/A2 changed-row fraction")
    for arm, offset in (("A1", -0.12), ("A2", 0.12)):
        boundary_ratio = [
            float(operators[(value, arm)]["non_type0_mean_neighbor_count"])
            / float(operators[(value, arm)]["type0_mean_neighbor_count"])
            for value in resolutions
        ]
        type0_boundary_source_share = [
            sum(
                float(
                    operators[(value, arm)][
                        f"target_type_0_source_type_{source_type}_coefficient_share"
                    ]
                )
                for source_type in (1, 2, 3)
            )
            for value in resolutions
        ]
        axes[0, 2].bar(x + offset, boundary_ratio, width=0.24, label=arm)
        axes[1, 2].bar(x + offset, type0_boundary_source_share, width=0.24, label=arm)
    axes[0, 2].axhline(1.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[0, 2].set_ylabel("boundary/type-0 neighbor ratio")
    axes[1, 2].set_ylabel("type-0 rows: boundary-source weight share")
    for axis in axes.flat:
        axis.set_xticks(x, resolutions)
        axis.grid(True, axis="y", color="#d0d0d0", linewidth=0.5)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    axes[1, 2].legend(fontsize=8)
    figure.suptitle("D073 geometry-only radius and support inventory")
    return _save_figure(figure, output_stem)


def _stratum_values(
    rows: Sequence[Mapping[str, str]], *, pair: str, mode: str
) -> dict[tuple[str, str, int, str], float]:
    values = {}
    for row in rows:
        if (
            row["pair"] == pair
            and row["mode"] == mode
            and row["pathway_level"] == "decoded_residual"
            and row["arm"] in {"A0", "A1", "A2"}
        ):
            values[(row["case_id"], row["band"], int(row["call"]), row["arm"])] = float(
                row["defect_scaled_rms"]
            )
    return values


def _finite_column_summary(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    median = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    q25 = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    q75 = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        if finite.size:
            median[column] = np.median(finite)
            q25[column], q75[column] = np.quantile(finite, (0.25, 0.75))
    return median, q25, q75


def plot_band_evolution(
    rows: Sequence[Mapping[str, str]],
    output_stem: Path,
    *,
    pair: str,
    mode: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    values = _stratum_values(rows, pair=pair, mode=mode)
    bands = ("total", "large", "transition", "local")
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 7.0), constrained_layout=True)
    saturation = []
    for column, band in enumerate(bands):
        for arm in ("A0", "A1", "A2"):
            case_matrix = np.asarray(
                [
                    [values[(case_id, band, call, arm)] for call in CALLS]
                    for case_id in CASE_IDS
                ]
            )
            for case_values in case_matrix:
                axes[0, column].plot(
                    CALLS, case_values, color=ARM_COLORS[arm], alpha=0.22, linewidth=0.8
                )
            median = np.median(case_matrix, axis=0)
            q25, q75 = np.quantile(case_matrix, (0.25, 0.75), axis=0)
            axes[0, column].plot(
                CALLS,
                median,
                color=ARM_COLORS[arm],
                marker="o",
                linewidth=2.0,
                label=arm,
            )
            axes[0, column].fill_between(
                CALLS, q25, q75, color=ARM_COLORS[arm], alpha=0.12
            )
            saturation.append(
                {
                    "figure": output_stem.name,
                    "panel": f"absolute_{band}_{arm}",
                    "fixed_limit": ABSOLUTE_RMS_LIMIT,
                    "fraction_outside_limit": float(
                        np.mean(case_matrix > ABSOLUTE_RMS_LIMIT)
                    ),
                    "maximum": float(np.max(case_matrix)),
                }
            )
        for denominator_arm, label, color in (
            ("A1", "A2/A1", "#000000"),
            ("A0", "A2/A0", "#009E73"),
        ):
            numerator = np.asarray(
                [
                    [values[(case_id, band, call, "A2")] for call in CALLS]
                    for case_id in CASE_IDS
                ]
            )
            denominator = np.asarray(
                [
                    [values[(case_id, band, call, denominator_arm)] for call in CALLS]
                    for case_id in CASE_IDS
                ]
            )
            valid = denominator > DENOMINATOR_MINIMUM
            ratio_matrix = np.full_like(numerator, np.nan)
            ratio_matrix[valid] = numerator[valid] / denominator[valid]
            for case_values in ratio_matrix:
                axes[1, column].plot(
                    CALLS, case_values, color=color, alpha=0.22, linewidth=0.8
                )
            median, q25, q75 = _finite_column_summary(ratio_matrix)
            axes[1, column].plot(
                CALLS,
                median,
                color=color,
                marker="o",
                linewidth=2.0,
                label=label,
            )
            axes[1, column].fill_between(CALLS, q25, q75, color=color, alpha=0.12)
            finite = ratio_matrix[np.isfinite(ratio_matrix)]
            saturation.append(
                {
                    "figure": output_stem.name,
                    "panel": f"ratio_{band}_{label.replace('/', '_')}",
                    "fixed_limit": RATIO_LIMIT,
                    "denominator_minimum": DENOMINATOR_MINIMUM,
                    "invalid_denominator_count": int(np.size(valid) - np.sum(valid)),
                    "fraction_outside_limit": (
                        None
                        if finite.size == 0
                        else float(np.mean(finite > RATIO_LIMIT))
                    ),
                    "maximum": None if finite.size == 0 else float(np.max(finite)),
                }
            )
        axes[1, column].axhline(1.0, color="#D55E00", linestyle=":", linewidth=1.0)
        axes[0, column].set_title(f"absolute {band}")
        axes[1, column].set_title(f"radius/native ratios {band}")
        axes[0, column].set_ylim(0.0, ABSOLUTE_RMS_LIMIT)
        axes[1, column].set_ylim(0.0, RATIO_LIMIT)
        axes[0, column].set_ylabel("residual-scale RMS")
        axes[1, column].set_ylabel("defect ratio")
        for axis in axes[:, column]:
            axis.set_xlabel("model call")
            axis.grid(True, color="#d0d0d0", linewidth=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle(f"D073 A2 versus A1: {pair}, {mode}")
    return _save_figure(figure, output_stem), saturation


def plot_case_signs(
    rows: Sequence[Mapping[str, str]],
    output_stem: Path,
    *,
    pair: str,
    mode: str,
) -> list[Path]:
    selected = {
        (row["case_id"], row["arm"]): float(row["call_rms_aggregate"])
        for row in rows
        if row["pair"] == pair
        and row["mode"] == mode
        and row["pathway_level"] == "decoded_residual"
        and row["band"] == "large"
        and row["arm"] in {"A1", "A2"}
    }
    denominator = np.asarray(
        [selected[(case_id, "A1")] for case_id in CASE_IDS], dtype=np.float64
    )
    numerator = np.asarray(
        [selected[(case_id, "A2")] for case_id in CASE_IDS], dtype=np.float64
    )
    valid = denominator > DENOMINATOR_MINIMUM
    reduction = np.full(len(CASE_IDS), np.nan, dtype=np.float64)
    reduction[valid] = 1.0 - numerator[valid] / denominator[valid]
    figure, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    colors = np.where(
        ~valid, "#999999", np.where(reduction >= 0.0, "#0072B2", "#D55E00")
    )
    axis.bar(np.arange(len(CASE_IDS)), np.where(valid, reduction, 0.0), color=colors)
    for case_index in np.flatnonzero(~valid):
        axis.annotate(
            "invalid\nA1 floor",
            (case_index, 0.0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.axhline(0.2, color="#009E73", linewidth=0.8, linestyle=":")
    axis.set_xticks(np.arange(len(CASE_IDS)), CASE_IDS, rotation=30, ha="right")
    axis.set_ylabel("1 - D(A2)/D(A1), large band")
    axis.set_title(f"D073 case signs: {pair}, {mode}")
    axis.grid(True, axis="y", color="#d0d0d0", linewidth=0.5)
    return _save_figure(figure, output_stem)


def _load_payload(
    path: Path,
    *,
    expected_residual_scale: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]) for name in loaded.files}
    if str(payload.get("schema", "")) != "pcno_physical_radius_visual_payload_v1":
        raise ValueError(f"unexpected D073 visual payload schema: {path}")
    case_id = str(payload.get("case_id", ""))
    pair = str(payload.get("pair", ""))
    mode = str(payload.get("mode", ""))
    if case_id not in VISUAL_CASE_IDS or pair not in PAIRS or mode not in MODES:
        raise ValueError("D073 payload metadata lies outside the visual contract")
    pieces = path.stem.split("__")
    if len(pieces) == 3:
        filename_case, filename_pair, filename_mode = pieces
        if (
            filename_case != case_id
            or filename_pair.replace("_to_", "->") != pair
            or filename_mode != mode
        ):
            raise ValueError("D073 payload filename and intrinsic metadata differ")
    calls = np.asarray(payload.get("call"), dtype=np.int64)
    if tuple(calls.tolist()) != CALLS:
        raise ValueError(
            f"D073 visual payload does not contain calls 1,5,15,30: {path}"
        )
    resolution = parse_resolution(str(payload["resolution"]))
    if resolution != parse_resolution(pair.split("->", maxsplit=1)[0]):
        raise ValueError("D073 payload resolution is not the coarse pair member")
    node_count = resolution[0] * resolution[1]
    required_fields = {
        f"{arm}_{band}"
        for arm in ("A0", "A1", "A2")
        for band in ("total", "large", "local")
    } | {f"A2_minus_A1_{band}" for band in ("total", "large", "local")}
    missing = required_fields - set(payload)
    if missing:
        raise ValueError(f"D073 visual payload is missing fields: {sorted(missing)}")
    expected_inventory = required_fields | {
        "schema",
        "case_id",
        "pair",
        "mode",
        "resolution",
        "nodes",
        "weights",
        "node_type",
        "residual_scale",
        "call",
        "physical_time",
    }
    if set(payload) != expected_inventory:
        raise ValueError("D073 visual payload field inventory mismatch")
    for name in required_fields:
        value = np.asarray(payload[name])
        if value.shape != (len(CALLS), node_count, 4) or not np.isfinite(value).all():
            raise ValueError(f"invalid D073 visual field {name}: {value.shape}")
    for band in ("total", "large", "local"):
        if not np.allclose(
            payload[f"A2_minus_A1_{band}"],
            payload[f"A2_{band}"] - payload[f"A1_{band}"],
            rtol=2.0e-6,
            atol=2.0e-7,
        ):
            raise ValueError(f"D073 A2-minus-A1 payload closure failed: {band}")
    scale = np.asarray(payload.get("residual_scale"), dtype=np.float64)
    if scale.shape != (4,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("D073 visual payload has an invalid residual scale")
    if expected_residual_scale is not None and not np.array_equal(
        scale, np.asarray(expected_residual_scale, dtype=np.float64)
    ):
        raise ValueError("D073 payload and summary residual scales differ")
    nodes = np.asarray(payload.get("nodes"), dtype=np.float64)
    weights = np.asarray(payload.get("weights"), dtype=np.float64)
    node_type = np.asarray(payload.get("node_type"), dtype=np.int64)
    times = np.asarray(payload.get("physical_time"), dtype=np.float64)
    if nodes.shape != (node_count, 2) or not np.isfinite(nodes).all():
        raise ValueError("D073 payload nodes are invalid")
    if (
        weights.shape != (node_count,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("D073 payload weights are invalid")
    if node_type.shape != (node_count,) or np.any((node_type < 0) | (node_type > 3)):
        raise ValueError("D073 dynamic payload node types are invalid")
    if (
        times.shape != (len(CALLS),)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("D073 payload physical times are invalid")
    nx, ny = resolution
    x = nodes[:, 0].reshape(ny, nx)
    y = nodes[:, 1].reshape(ny, nx)
    if (
        np.any(np.diff(x, axis=1) <= 0.0)
        or np.any(np.diff(y, axis=0) <= 0.0)
        or not np.allclose(x, x[:1])
        or not np.allclose(y, y[:, :1])
    ):
        raise ValueError("D073 payload nodes are not in structured FV order")
    return payload


def _physical_extent(
    payload: Mapping[str, np.ndarray],
) -> tuple[float, float, float, float]:
    nx, ny = parse_resolution(str(payload["resolution"]))
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    x = nodes[:, 0].reshape(ny, nx)
    y = nodes[:, 1].reshape(ny, nx)
    dx = float(np.median(np.diff(x[0])))
    dy = float(np.median(np.diff(y[:, 0])))
    return (
        float(x[0, 0] - 0.5 * dx),
        float(x[0, -1] + 0.5 * dx),
        float(y[0, 0] - 0.5 * dy),
        float(y[-1, 0] + 0.5 * dy),
    )


def _movie_columns(scale_split: bool) -> tuple[tuple[str, str], ...]:
    arms = (
        ("A0", "A0"),
        ("A1", "A1"),
        ("A2", "A2"),
        ("A2-A1", "A2_minus_A1"),
    )
    bands = ("large", "local") if scale_split else ("total",)
    return tuple(
        (f"{label} {band}", f"{key}_{band}") for band in bands for label, key in arms
    )


def _animation_saturation(
    payload: Mapping[str, np.ndarray],
    *,
    movie: str,
    columns: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    calls = np.asarray(payload["call"], dtype=np.int64)
    rows = []
    for title, field in columns:
        values = np.asarray(payload[field], dtype=np.float64)
        for frame, call in enumerate(calls):
            for component, component_name in enumerate(COMPONENT_LABELS):
                limit = PHYSICAL_SCALE_MULTIPLIER * scale[component]
                selected = np.abs(values[frame, :, component])
                rows.append(
                    {
                        "case_id": str(payload["case_id"]),
                        "pair": str(payload["pair"]),
                        "mode": str(payload["mode"]),
                        "movie": movie,
                        "panel": title,
                        "field": field,
                        "call": int(call),
                        "component": component_name,
                        "fixed_physical_limit": limit,
                        "maximum_absolute_value": float(np.max(selected)),
                        "maximum_to_limit": float(np.max(selected) / limit),
                        "fraction_clipped": float(np.mean(selected > limit)),
                    }
                )
    return rows


def _render_movie(
    payload: Mapping[str, np.ndarray],
    output: Path,
    *,
    scale_split: bool,
    fps: int,
    dpi: int,
) -> list[dict[str, Any]]:
    nx, ny = parse_resolution(str(payload["resolution"]))
    columns = _movie_columns(scale_split)
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    extent = _physical_extent(payload)
    figure, axes = plt.subplots(
        4,
        len(columns),
        figsize=(2.2 * len(columns), 7.8),
        constrained_layout=True,
        squeeze=False,
    )
    images = []
    for component in range(4):
        limit = PHYSICAL_SCALE_MULTIPLIER * scale[component]
        for column, (title, field) in enumerate(columns):
            image = axes[component, column].imshow(
                np.asarray(payload[field])[0, :, component].reshape(ny, nx),
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                extent=extent,
                aspect="equal",
            )
            images.append((image, field, component))
            axes[component, column].set_xticks([])
            axes[component, column].set_yticks([])
            axes[component, column].set_aspect("equal", adjustable="box")
            if component == 0:
                axes[component, column].set_title(title, fontsize=8)
            if column == 0:
                axes[component, column].set_ylabel(COMPONENT_LABELS[component])
        figure.colorbar(
            images[-1][0],
            ax=axes[component, :].tolist(),
            shrink=0.65,
            pad=0.01,
        )
    title = figure.suptitle("")
    figure.text(
        0.5,
        0.005,
        "Decoded increment commutator: d_a = F_c^a(R h_f) - R F_f^a(h_f); "
        "not a truth-residual error or modified rollout.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    calls = np.asarray(payload["call"], dtype=np.int64)
    times = np.asarray(payload["physical_time"], dtype=np.float64)

    def update(frame: int) -> list[Any]:
        artists: list[Any] = []
        for image, field, component in images:
            image.set_data(
                np.asarray(payload[field])[frame, :, component].reshape(ny, nx)
            )
            artists.append(image)
        title.set_text(
            f"D073 same-hidden {payload['case_id']} {payload['pair']} {payload['mode']} "
            f"call={calls[frame]} t={times[frame]:.6g}"
        )
        artists.append(title)
        return artists

    animation = FuncAnimation(
        figure, update, frames=len(calls), interval=1000 / fps, blit=False
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    movie = "scale_split" if scale_split else "full"
    return _animation_saturation(payload, movie=movie, columns=columns)


def animate_payload(
    payload_path: Path,
    full_output: Path,
    scale_split_output: Path,
    *,
    fps: int,
    dpi: int,
    expected_residual_scale: Sequence[float] | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = _load_payload(
        payload_path, expected_residual_scale=expected_residual_scale
    )
    saturation = _render_movie(
        payload, full_output, scale_split=False, fps=fps, dpi=dpi
    )
    saturation.extend(
        _render_movie(
            payload,
            scale_split_output,
            scale_split=True,
            fps=fps,
            dpi=dpi,
        )
    )
    return [full_output, scale_split_output], saturation


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _verify_results(args.results_dir)
    args.output_dir.mkdir(parents=True)
    geometry_rows = _read_rows(args.results_dir / "geometry_inventory.csv")
    operator_rows = _read_rows(args.results_dir / "operator_checks.csv")
    arm_rows = _read_rows(args.results_dir / "arm_metrics.csv")
    aggregate_rows = _read_rows(args.results_dir / "case_aggregates.csv")
    generated: list[Path] = []
    saturation_rows: list[dict[str, Any]] = []
    generated.extend(
        plot_radius_support_inventory(
            geometry_rows,
            operator_rows,
            args.output_dir / "radius_support_inventory",
        )
    )
    for pair in PAIRS:
        safe_pair = pair.replace("->", "_to_")
        for mode in MODES:
            paths, saturation = plot_band_evolution(
                arm_rows,
                args.output_dir / f"a2_vs_a1_band_evolution_{safe_pair}_{mode}",
                pair=pair,
                mode=mode,
            )
            generated.extend(paths)
            saturation_rows.extend(saturation)
            generated.extend(
                plot_case_signs(
                    aggregate_rows,
                    args.output_dir / f"a2_vs_a1_case_signs_{safe_pair}_{mode}",
                    pair=pair,
                    mode=mode,
                )
            )
    for relative in _expected_payloads():
        payload_path = args.results_dir / relative
        stem = Path(relative).stem
        paths, saturation = animate_payload(
            payload_path,
            args.output_dir / f"{stem}__full.gif",
            args.output_dir / f"{stem}__scale_split.gif",
            fps=args.fps,
            dpi=args.dpi,
            expected_residual_scale=summary.get("residual_scale"),
        )
        generated.extend(paths)
        saturation_rows.extend(saturation)
    saturation_path = args.output_dir / "visual_scale_saturation_by_call.csv"
    write_csv_with_paths(saturation_path, saturation_rows)
    generated.append(saturation_path)
    observed_outputs = {
        path.relative_to(args.output_dir).as_posix() for path in generated
    }
    expected_outputs = _expected_rendered_outputs()
    if observed_outputs != expected_outputs:
        raise RuntimeError(
            "D073 visual output inventory changed: "
            f"missing={sorted(expected_outputs - observed_outputs)} "
            f"extra={sorted(observed_outputs - expected_outputs)}"
        )
    output_hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in sorted(generated)
    }
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "args": jsonable_args(args),
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_manifest_sha256": sha256_file(args.results_dir / "manifest.json"),
        "input_output_hashes_verified": True,
        "claim_boundary": (
            "four-call same-hidden representation views; not modified rollouts, "
            "temporal cancellation, or cumulative-drift evidence"
        ),
        "field_definition": (
            "A0/A1/A2 panels are decoded increment commutators "
            "d_a=F_c^a(R h_f)-R F_f^a(h_f); A2-minus-A1 is their signed "
            "response, not an individual prediction or truth error"
        ),
        "aggregation": (
            "case traces retained; median and IQR after per-case metrics; pairs "
            "and input modes remain separate"
        ),
        "fixed_scale_contract": {
            "absolute_residual_scaled_rms": [0.0, ABSOLUTE_RMS_LIMIT],
            "a2_a1_ratio": [0.0, RATIO_LIMIT],
            "spatial_physical_limit": (
                "plus/minus one frozen checkpoint residual scale per component"
            ),
            "per_frame_normalization": False,
        },
        "visual_case_ids": list(VISUAL_CASE_IDS),
        "calls": list(CALLS),
        "output_hashes": output_hashes,
        "output_count": len(output_hashes),
        "source_sha256": sha256_file(Path(__file__)),
        "git": git_state(),
        "input_decision": summary["decision"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(
        f"D073 visualization status={manifest['status']} "
        f"outputs={len(manifest['output_hashes'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
