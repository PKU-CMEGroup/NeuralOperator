#!/usr/bin/env python3
"""Render fixed-contract figures for the D070 pathway isolation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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

SCHEMA = "pcno_fine_grained_pathway_visualization_v2"
RESULT_SCHEMA = "pcno_fine_grained_pathway_diagnostic_v2"
REQUIRED_RESULT_FILES = (
    "arm_metrics.csv",
    "pathway_terms.csv",
    "closure_metrics.csv",
    "direct_output_metrics.csv",
    "commutator_metrics.csv",
    "recurrence_metrics.csv",
    "trace_replay_metrics.csv",
    "completion.csv",
    "replay_metrics.csv",
    "reference_checks.csv",
)
REDUCTION_LIMIT = 1.0
LOG_TERM_LIMIT = 2.0
BANDS = ("total", "large", "transition", "local")
PAIR_COLORS = ("#0072B2", "#D55E00")
MODE_STYLES = {"teacher_forced": "-", "free_rollout": "--"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        raise ValueError("unexpected D070 result schema")
    if summary.get("status") != "complete" or not summary.get(
        "scientific_interpretation_allowed"
    ):
        raise ValueError("D070 results are not complete and interpretable")
    output_hashes = summary.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(
        REQUIRED_RESULT_FILES
    ):
        raise ValueError("D070 output inventory mismatch")
    for relative, expected in output_hashes.items():
        path = results_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"D070 output digest mismatch: {relative}")
    return summary


def _arm_label(row: Mapping[str, str]) -> str:
    return f"{row['arm']}@L{row['layer']}"


def _term_label(row: Mapping[str, str]) -> str:
    return f"{row['pathway']}:{row['term']}@L{row['layer']}"


def _matrix(
    rows: Sequence[Mapping[str, str]],
    *,
    row_label: Callable[[Mapping[str, str]], str],
    value_field: str,
) -> tuple[list[str], list[int], np.ndarray]:
    labels = sorted({row_label(row) for row in rows})
    calls = sorted({int(row["call"]) for row in rows})
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(value_field))
        if value is not None:
            grouped[(row_label(row), int(row["call"]))].append(value)
    matrix = np.full((len(labels), len(calls)), np.nan, dtype=np.float64)
    for row_index, label in enumerate(labels):
        for column_index, call in enumerate(calls):
            values = grouped.get((label, call), ())
            if values:
                matrix[row_index, column_index] = float(np.median(values))
    return labels, calls, matrix


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_heatmap(
    rows: Sequence[Mapping[str, str]],
    output_stem: Path,
    *,
    row_label: Callable[[Mapping[str, str]], str],
    value_field: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> tuple[list[Path], dict[str, Any]]:
    labels, calls, matrix = _matrix(rows, row_label=row_label, value_field=value_field)
    if not labels or not calls:
        raise ValueError(f"no rows available for {title}")
    height = max(4.0, 0.27 * len(labels) + 1.7)
    figure, axis = plt.subplots(figsize=(8.2, height), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(calls)), labels=[str(call) for call in calls])
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    axis.set_xlabel("model call")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    finite = matrix[np.isfinite(matrix)]
    saturation = {
        "figure": output_stem.name,
        "value_field": value_field,
        "fixed_vmin": vmin,
        "fixed_vmax": vmax,
        "minimum": None if finite.size == 0 else float(finite.min()),
        "maximum": None if finite.size == 0 else float(finite.max()),
        "fraction_outside_limits": (
            None
            if finite.size == 0
            else float(np.mean((finite < vmin) | (finite > vmax)))
        ),
    }
    return _save_figure(figure, output_stem), saturation


def _candidate(summary: Mapping[str, Any]) -> tuple[str, str, str]:
    mechanism = summary.get("mechanism_selection") or {}
    mechanism_decision = str(mechanism.get("decision", ""))
    if mechanism_decision == "selected":
        selected = mechanism["selected"]
        return str(selected["arm"]), str(selected["layer"]), "selected"
    candidates = (
        mechanism.get("qualified", ())
        if mechanism_decision == "multiple_supported"
        else mechanism.get("candidates", ())
    )
    scored = []
    for candidate in candidates:
        values = [
            pair.get("median_large_reduction")
            for pair in candidate.get("pairs", ())
            if pair.get("median_large_reduction") is not None
        ]
        if values:
            scored.append((float(np.mean(values)), candidate))
    if not scored:
        raise ValueError("D070 mechanism summary contains no plottable candidate")
    candidate = max(scored, key=lambda item: item[0])[1]
    suffix = (
        "multiple_supported"
        if mechanism_decision == "multiple_supported"
        else "unresolved"
    )
    return str(candidate["arm"]), str(candidate["layer"]), f"best_descriptive_{suffix}"


def plot_candidate_bands(
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    summary: Mapping[str, Any],
) -> tuple[list[Path], dict[str, str]]:
    arm, layer, decision = _candidate(summary)
    pairs = sorted({row["pair"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for axis, band in zip(axes.flat, BANDS, strict=True):
        for pair_index, pair in enumerate(pairs):
            for mode, style in MODE_STYLES.items():
                selected = [
                    row
                    for row in rows
                    if row["arm"] == arm
                    and row["layer"] == layer
                    and row["pair"] == pair
                    and row["mode"] == mode
                    and row["band"] == band
                ]
                grouped: dict[int, list[float]] = defaultdict(list)
                for row in selected:
                    value = _number(row.get("defect_norm_ratio"))
                    if value is not None:
                        grouped[int(row["call"])].append(value)
                calls = sorted(grouped)
                if calls:
                    medians = [float(np.median(grouped[call])) for call in calls]
                    axis.plot(
                        calls,
                        medians,
                        color=PAIR_COLORS[pair_index],
                        linestyle=style,
                        marker="o",
                        label=f"{pair} {mode}",
                    )
        axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_title(band)
        axis.set_xlabel("model call")
        axis.set_ylabel("candidate / baseline defect RMS")
        axis.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(f"D070 {decision}: {arm}@L{layer}")
    paths = _save_figure(figure, output_dir / "candidate_band_evolution")
    return paths, {"arm": arm, "layer": layer, "decision": decision}


def plot_candidate_absolute_trajectories(
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    summary: Mapping[str, Any],
) -> list[Path]:
    arm, layer, decision = _candidate(summary)
    selected = [
        row
        for row in rows
        if row.get("record_kind") == "same_hidden_single_layer"
        and row["arm"] == arm
        and row["layer"] == layer
        and row["band"] == "large"
    ]
    pairs = sorted({row["pair"] for row in selected})
    modes = [
        mode for mode in MODE_STYLES if any(row["mode"] == mode for row in selected)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    fields = (
        ("baseline_defect_rms", "baseline", "#333333"),
        ("arm_defect_rms", "intervention", "#0072B2"),
        ("response_rms", "signed-response magnitude", "#D55E00"),
    )
    maximum = max(
        (
            value
            for row in selected
            for field, _, _ in fields
            if (value := _number(row.get(field))) is not None
        ),
        default=1.0,
    )
    for axis, (pair, mode) in zip(
        axes.flat, ((pair, mode) for pair in pairs for mode in modes)
    ):
        pair_rows = [
            row for row in selected if row["pair"] == pair and row["mode"] == mode
        ]
        for field, label, color in fields:
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in pair_rows:
                value = _number(row.get(field))
                if value is not None:
                    grouped[int(row["call"])].append(value)
            calls = sorted(grouped)
            if calls:
                axis.plot(
                    calls,
                    [float(np.median(grouped[call])) for call in calls],
                    marker="o",
                    linewidth=1.7,
                    color=color,
                    label=label,
                )
        axis.set_title(f"{pair}, {mode}")
        axis.set_xlabel("model call")
        axis.set_ylabel("absolute residual-scale RMS")
        axis.set_ylim(0.0, 1.05 * maximum if maximum > 0.0 else 1.0)
        axis.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.65)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"D070 {decision}: absolute large-band defect, {arm}@L{layer}")
    return _save_figure(figure, output_dir / "candidate_absolute_large_band")


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _verify_results(args.results_dir)
    args.output_dir.mkdir(parents=True)
    arm_rows = _read_rows(args.results_dir / "arm_metrics.csv")
    term_rows = _read_rows(args.results_dir / "pathway_terms.csv")
    generated: list[Path] = []
    saturation_rows: list[dict[str, Any]] = []
    pairs = sorted({row["pair"] for row in arm_rows})
    modes = sorted({row["mode"] for row in arm_rows})
    for pair in pairs:
        safe_pair = pair.replace("->", "_to_")
        for mode in modes:
            selected_arms = [
                row
                for row in arm_rows
                if row["pair"] == pair
                and row["mode"] == mode
                and row["band"] == "large"
            ]
            paths, saturation = plot_heatmap(
                selected_arms,
                args.output_dir / f"large_reduction_{safe_pair}_{mode}",
                row_label=_arm_label,
                value_field="defect_reduction",
                title=f"large-band reduction: {pair}, {mode}",
                cmap="RdBu",
                vmin=-REDUCTION_LIMIT,
                vmax=REDUCTION_LIMIT,
            )
            generated.extend(paths)
            saturation_rows.append(saturation)
            selected_terms = [
                row for row in term_rows if row["pair"] == pair and row["mode"] == mode
            ]
            transformed = []
            for row in selected_terms:
                copy = dict(row)
                value = _number(row.get("term_to_mesh_ratio"))
                copy["log10_term_to_mesh_ratio"] = (
                    None
                    if value is None
                    else np.log10(max(value, np.finfo(float).tiny))
                )
                transformed.append(copy)
            paths, saturation = plot_heatmap(
                transformed,
                args.output_dir / f"pathway_terms_{safe_pair}_{mode}",
                row_label=_term_label,
                value_field="log10_term_to_mesh_ratio",
                title=f"log10 pathway-term / mesh-gap RMS: {pair}, {mode}",
                cmap="viridis",
                vmin=-LOG_TERM_LIMIT,
                vmax=LOG_TERM_LIMIT,
            )
            generated.extend(paths)
            saturation_rows.append(saturation)
    candidate_paths, candidate = plot_candidate_bands(
        arm_rows, args.output_dir, summary
    )
    generated.extend(candidate_paths)
    generated.extend(
        plot_candidate_absolute_trajectories(arm_rows, args.output_dir, summary)
    )
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
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_output_hashes_verified": True,
        "aggregation": "within case before median; resolution pairs and modes separate",
        "fixed_color_contract": {
            "defect_reduction": [-REDUCTION_LIMIT, REDUCTION_LIMIT],
            "log10_term_to_mesh_ratio": [-LOG_TERM_LIMIT, LOG_TERM_LIMIT],
        },
        "candidate_displayed": candidate,
        "output_hashes": output_hashes,
        "source_sha256": sha256_file(Path(__file__)),
        "git": git_state(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(
        f"D070 visualization status={manifest['status']} "
        f"outputs={len(manifest['output_hashes'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
