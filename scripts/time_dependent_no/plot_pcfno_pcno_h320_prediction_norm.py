#!/usr/bin/env python3
"""Plot the post-hoc prediction-size diagnostic for the matched H320 rollouts."""

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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    PRODUCTION_STEPS,
    TRAJECTORY_KEYS,
    TRUTH_STEPS,
    build_final_hash_manifest,
    sha256_file,
    verify_final_hash_manifest,
    write_csv,
    write_json,
)
from scripts.time_dependent_no.visualize_pcfno_pcno_h320_comparison import (
    ANIMATION_KEYS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    configure_report_style,
    verify_pair,
    verify_result_root,
)

SCHEMA = "w26_l1_pcfno_pcno_h320_prediction_norm_v1"
WORKING_ID = "W26-L1-PCFNO-PCNO-H320-NORM-P1-POSTHOC"
PCNO_RESULT_MANIFEST_SHA256 = (
    "6db0133d432cfee1b29029a7c6d677ee2ca590d109bf4b171bbc69b4be6e94ef"
)
SNAPSHOT_CALLS = (80, 100, 120, 160, 200, 240, 280, 320)
FIXED_LINESTYLES = {"187": "-", "54": "--", "227": "-.", "233": ":"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcno-dir", type=Path, required=True)
    parser.add_argument("--pcfno-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    return args


def _read_rows(path: Path, model: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trajectory",
            "call",
            "physical_time",
            "event_finite",
            "common_scaled_rms_ratio",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"call-metric schema changed: {path}")
        rows = []
        for source in reader:
            call = int(source["call"])
            if call <= TRUTH_STEPS:
                continue
            ratio_text = source["common_scaled_rms_ratio"]
            ratio = None if ratio_text == "" else float(ratio_text)
            rows.append(
                {
                    "model": model,
                    "trajectory": source["trajectory"],
                    "call": call,
                    "physical_time": float(source["physical_time"]),
                    "event_finite": source["event_finite"] == "True",
                    "prediction_scaled_l2_ratio": ratio,
                }
            )
    observed = {row["trajectory"] for row in rows}
    if observed != set(TRAJECTORY_KEYS):
        raise ValueError(f"{model} call metrics changed population")
    return rows


def _case_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in rows if row["trajectory"] == key),
        key=lambda row: int(row["call"]),
    )


def _first_nonfinite_calls(summary: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        key: summary["cases"][key]["events"]["finite"]["first_failure_call"]
        for key in TRAJECTORY_KEYS
    }


def _summarize_model(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    finite_ratios = [
        float(row["prediction_scaled_l2_ratio"])
        for row in rows
        if row["prediction_scaled_l2_ratio"] is not None
    ]
    first_above_ten: dict[str, int | None] = {}
    for key in TRAJECTORY_KEYS:
        first_above_ten[key] = next(
            (
                int(row["call"])
                for row in _case_rows(rows, key)
                if row["prediction_scaled_l2_ratio"] is not None
                and float(row["prediction_scaled_l2_ratio"]) > 10.0
            ),
            None,
        )
    snapshots = {}
    for call in SNAPSHOT_CALLS:
        values = [
            float(row["prediction_scaled_l2_ratio"])
            for row in rows
            if int(row["call"]) == call
            and row["prediction_scaled_l2_ratio"] is not None
        ]
        snapshots[str(call)] = {
            "finite_value_count": len(values),
            "finite_only_median": None if not values else float(np.median(values)),
            "finite_only_q1": None if not values else float(np.quantile(values, 0.25)),
            "finite_only_q3": None if not values else float(np.quantile(values, 0.75)),
        }
    nonfinite = _first_nonfinite_calls(summary)
    return {
        "trajectory_count": len(TRAJECTORY_KEYS),
        "maximum_finite_prediction_scaled_l2_ratio": max(finite_ratios),
        "first_ratio_above_10_count": sum(
            value is not None for value in first_above_ten.values()
        ),
        "first_ratio_above_10_calls": first_above_ten,
        "native_nonfinite_count": sum(
            value is not None for value in nonfinite.values()
        ),
        "native_nonfinite_calls": nonfinite,
        "snapshots": snapshots,
    }


def plot_growth(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    configure_report_style()
    finite_values = [
        float(row["prediction_scaled_l2_ratio"])
        for rows in rows_by_model.values()
        for row in rows
        if row["prediction_scaled_l2_ratio"] is not None
    ]
    y_max = 10.0 ** math.ceil(math.log10(max(finite_values) * 1.5))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75), sharex=True, sharey=True)
    for panel, model in enumerate(MODEL_ORDER):
        axis = axes[panel]
        rows = rows_by_model[model]
        for key in TRAJECTORY_KEYS:
            selected = _case_rows(rows, key)
            calls = np.asarray([int(row["call"]) for row in selected])
            ratios = np.asarray(
                [
                    np.nan
                    if row["prediction_scaled_l2_ratio"] is None
                    else float(row["prediction_scaled_l2_ratio"])
                    for row in selected
                ],
                dtype=np.float64,
            )
            highlighted = key in FIXED_LINESTYLES
            axis.plot(
                calls,
                ratios,
                color=MODEL_COLORS[model],
                linewidth=1.25 if highlighted else 0.55,
                linestyle=FIXED_LINESTYLES.get(key, "-"),
                alpha=0.9 if highlighted else 0.20,
                zorder=3 if highlighted else 1,
            )
        nonfinite_calls = _first_nonfinite_calls(summaries[model])
        failed_calls = [call for call in nonfinite_calls.values() if call is not None]
        if failed_calls:
            axis.scatter(
                failed_calls,
                [y_max / 1.8] * len(failed_calls),
                marker="x",
                color="#111111",
                s=16,
                linewidths=0.8,
                zorder=5,
            )
        axis.axhline(1.0, color="#666666", linewidth=0.8, linestyle=":")
        axis.axhline(10.0, color="#B22222", linewidth=0.8, linestyle="--")
        axis.axvline(TRUTH_STEPS, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(79, PRODUCTION_STEPS)
        axis.set_ylim(0.5, y_max)
        axis.set_xticks([79, 100, 160, 240, 320])
        axis.grid(axis="y", which="major", alpha=0.17)
        axis.set_xlabel("Recurrent call")
        axis.set_title(f"({chr(97 + panel)}) {MODEL_LABELS[model]}", loc="left")
        axis.text(
            0.98,
            0.96,
            f"native nonfinite: {len(failed_calls)}/30",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="#444444",
        )
    axes[0].set_ylabel(r"Prediction-size ratio $R_2(h)$")
    fig.suptitle("Reference-free growth of the scaled prediction $L^2$ size", y=0.98)
    handles = [
        Line2D([0], [0], color="#555555", linewidth=0.7, alpha=0.45, label="one case"),
        *[
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.25,
                linestyle=FIXED_LINESTYLES[key],
                label=f"GIF case {key}",
            )
            for key in ANIMATION_KEYS
        ],
        Line2D(
            [0],
            [0],
            color="#666666",
            linewidth=0.8,
            linestyle=":",
            label=r"H1--H79 maximum: $R_2=1$",
        ),
        Line2D(
            [0],
            [0],
            color="#B22222",
            linewidth=0.8,
            linestyle="--",
            label=r"registered RMS bound: $R_2=10$",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#111111",
            linestyle="none",
            markersize=4,
            label="first native nonfinite",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=6.6,
    )
    fig.text(
        0.5,
        0.115,
        "H80--H320 has no reference: growth diagnoses scale departure, not rollout error or shock position.",
        ha="center",
        fontsize=6.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 0.94))
    outputs = [
        output_dir / "pcfno_pcno_h320_prediction_l2_growth.png",
        output_dir / "pcfno_pcno_h320_prediction_l2_growth.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    plt.close(fig)
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = {
        "pcno": verify_result_root(args.pcno_dir, model="pcno"),
        "pcfno": verify_result_root(args.pcfno_dir, model="pcfno"),
    }
    if results["pcno"]["manifest_sha256"] != PCNO_RESULT_MANIFEST_SHA256:
        raise ValueError("full-PCNO input is not the closed H320 result")
    pair_checks = verify_pair(results)
    rows_by_model = {
        model: _read_rows(results[model]["root"] / "call_metrics.csv", model)
        for model in MODEL_ORDER
    }
    args.output_dir.mkdir(parents=True)
    curves = [row for model in MODEL_ORDER for row in rows_by_model[model]]
    write_csv(args.output_dir / "prediction_norm_curves.csv", curves)
    outputs = plot_growth(
        rows_by_model,
        {model: results[model]["summary"] for model in MODEL_ORDER},
        args.output_dir,
    )
    summary = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "registration_status": "post_hoc_owner_requested_2026-08-14",
        "truth_supported_through_call": TRUTH_STEPS,
        "analyzed_calls": [TRUTH_STEPS + 1, PRODUCTION_STEPS],
        "metric": {
            "name": "prediction_scaled_l2_ratio",
            "formula": (
                "sqrt(sum_i w_i ||qhat_i / D||_2^2 / sum_i w_i) divided by "
                "the per-case maximum of the same reference quantity on H1-H79"
            ),
            "q_order": ["density", "velocity_x", "velocity_y", "pressure"],
            "weights": "physical node weights",
            "component_scale": "shared reference-only component RMS D",
            "future_truth_used": False,
            "interpretation": (
                "prediction-size and blow-up diagnostic only; not error, phase, "
                "shock-position, admissibility, or conservation"
            ),
        },
        "input_manifests": {
            model: results[model]["manifest_sha256"] for model in MODEL_ORDER
        },
        "pair_checks": pair_checks,
        "source_sha256": sha256_file(Path(__file__)),
        "models": {
            model: _summarize_model(rows_by_model[model], results[model]["summary"])
            for model in MODEL_ORDER
        },
        "outputs": [path.name for path in outputs],
    }
    write_json(args.output_dir / "prediction_norm_summary.json", summary)
    artifact_names = [
        "prediction_norm_curves.csv",
        "prediction_norm_summary.json",
        *[path.name for path in outputs],
    ]
    manifest = build_final_hash_manifest(args.output_dir, artifact_names)
    write_json(args.output_dir / "final_hash_manifest.json", manifest)
    verification = verify_final_hash_manifest(args.output_dir, manifest)
    return {
        "summary": summary,
        "final_hash_manifest_sha256": sha256_file(
            args.output_dir / "final_hash_manifest.json"
        ),
        "verification": verification,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
