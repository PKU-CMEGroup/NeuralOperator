#!/usr/bin/env python3
"""Plot matched-training and frozen-intervention D072 summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMA = "pcno_boundary_field_result_figure_v1"
ARMS = ("N0", "G1", "S1")
COLORS = {"N0": "#777777", "G1": "#D55E00", "S1": "#0072B2"}

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
    parser.add_argument("--training-analysis", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--outcomes-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)


def _float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _aggregate_lookup(training: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    result = {}
    for row in training["aggregate"]:
        if row["mean"] is not None:
            result[(str(row["arm"]), str(row["metric"]))] = float(row["mean"])
    return result


def render(args: argparse.Namespace) -> dict[str, Any]:
    training = _read_json(args.training_analysis)
    evaluation = _read_json(args.evaluation_summary)
    outcomes = _read_csv(args.outcomes_csv)
    family = str(evaluation["family"])
    primary_horizon = int(evaluation["contract"]["endpoints"][-1])
    metric = f"H{primary_horizon}_relative_l2"
    records = training["records"]
    lookup = _aggregate_lookup(training)

    output_png = args.output_stem.with_suffix(".png")
    output_pdf = args.output_stem.with_suffix(".pdf")
    output_json = args.output_stem.with_name(args.output_stem.name + "_manifest.json")
    for path in (output_png, output_pdf, output_json):
        _strict_output(path, overwrite=args.overwrite)

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), constrained_layout=True)

    axis = axes[0, 0]
    seeds = sorted({int(row["seed"]) for row in records})
    x = np.arange(len(seeds), dtype=np.float64)
    for arm in ARMS:
        values = [
            float(
                next(
                    row[metric]
                    for row in records
                    if int(row["seed"]) == seed and row["arm"] == arm
                )
            )
            for seed in seeds
        ]
        axis.plot(
            x,
            np.asarray(values) * 1.0e3,
            marker="o",
            linewidth=1.5,
            color=COLORS[arm],
            label=arm,
        )
    axis.set_xticks(x, [str(seed)[-2:] for seed in seeds])
    axis.set_xlabel("training seed (last two digits)")
    axis.set_ylabel(f"H{primary_horizon} relative L2 ($\times 10^{{-3}}$)")
    axis.set_title("a  Matched-seed rollout error")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(ncol=3)

    axis = axes[0, 1]
    ratio_metrics = (
        (metric, f"H{primary_horizon}"),
        ("selected_one_step_relative_l2", "one-step\nall"),
        ("selected_one_step_boundary_relative_l2", "one-step\nboundary"),
        ("selected_one_step_normal_relative_l2", "one-step\ninterior"),
    )
    width = 0.23
    positions = np.arange(len(ratio_metrics), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS):
        values = [
            lookup[(arm, name)] / lookup[("N0", name)] for name, _ in ratio_metrics
        ]
        axis.bar(
            positions + (arm_index - 1) * width,
            values,
            width=width,
            color=COLORS[arm],
            label=arm,
        )
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xticks(positions, [label for _, label in ratio_metrics])
    axis.set_ylabel("seed-mean ratio to N0")
    axis.set_title("b  Rollout gain is absent one step ahead")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)

    axis = axes[1, 0]
    endpoint_rows = [
        row
        for row in evaluation["aggregate"]["endpoint_summary"]
        if int(row["call"]) == primary_horizon and row["arm"] == "S1"
    ]
    order = ["correct", "zero_all"] + [
        f"zero_{name}" for name in evaluation["contract"]["semantic_names"]
    ]
    endpoint_by_name = {str(row["intervention"]): row for row in endpoint_rows}
    missing = [name for name in order if name not in endpoint_by_name]
    if missing:
        raise ValueError(f"evaluation lacks S1 endpoint interventions: {missing}")
    values = [float(endpoint_by_name[name]["mean_relative_l2"]) for name in order]
    labels = [name.replace("zero_", "-").replace("_", " ") for name in order]
    bars = axis.bar(
        np.arange(len(order)),
        np.asarray(values) * 1.0e3,
        color=[COLORS["S1"]] + ["#CC79A7"] * (len(order) - 1),
    )
    bars[1].set_color("#B2182B")
    axis.set_xticks(np.arange(len(order)), labels, rotation=22, ha="right")
    axis.set_ylabel(f"H{primary_horizon} relative L2 ($\times 10^{{-3}}$)")
    axis.set_title("c  Frozen S1 field interventions")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)

    axis = axes[1, 1]
    region_order = [
        "boundary_distance_le_0.05",
        "shock",
        "vortex",
        "smooth",
    ]
    if family == "bump":
        region_order = ["boundary_distance_le_0.05", "shock", "smooth"]
    selected_rows = [
        row
        for row in outcomes
        if row["mode"] == "free_rollout"
        and int(row["call"]) == primary_horizon
        and row["arm"] == "S1"
        and row["intervention"] == "zero_all"
    ]
    region_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in selected_rows:
        value = _float(row.get("state_gap_to_arm_correct_rms"))
        if value is not None:
            region_values[row["region"]].append(value)
    missing_regions = [name for name in region_order if not region_values[name]]
    if missing_regions:
        raise ValueError(f"evaluation lacks intervention regions: {missing_regions}")
    means = [float(np.mean(region_values[name])) for name in region_order]
    stds = [
        (
            0.0
            if len(region_values[name]) < 2
            else float(np.std(region_values[name], ddof=1))
        )
        for name in region_order
    ]
    labels = [
        "near boundary" if name.startswith("boundary_distance") else name
        for name in region_order
    ]
    axis.bar(
        np.arange(len(region_order)),
        means,
        yerr=stds,
        capsize=2.5,
        color="#56B4E9",
    )
    axis.set_xticks(np.arange(len(region_order)), labels, rotation=18, ha="right")
    axis.set_ylabel("S1 zero-field state gap (scaled RMS)")
    axis.set_title("d  Where the frozen intervention propagates")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)

    figure.suptitle(
        f"D072 {family}: matched training versus frozen boundary-field intervention",
        fontsize=11,
    )
    figure.savefig(output_png)
    figure.savefig(output_pdf)
    plt.close(figure)

    manifest = {
        "schema": SCHEMA,
        "family": family,
        "primary_horizon": primary_horizon,
        "inputs": {
            str(args.training_analysis): _sha256(args.training_analysis),
            str(args.evaluation_summary): _sha256(args.evaluation_summary),
            str(args.outcomes_csv): _sha256(args.outcomes_csv),
        },
        "panels": {
            "a": "paired seed-level selected-checkpoint rollout errors",
            "b": "seed-mean metric ratios; N0 equals one",
            "c": "one selected seed, frozen S1 interventions on the full open cohort",
            "d": "case mean plus sample standard deviation of within-checkpoint state gaps",
        },
        "causal_language": {
            "training": "matched association under one training recipe",
            "intervention": "within-checkpoint causal effect on the open population",
        },
        "outputs": {
            output_png.name: _sha256(output_png),
            output_pdf.name: _sha256(output_pdf),
        },
    }
    output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    render(parse_args(argv))


if __name__ == "__main__":
    main()
