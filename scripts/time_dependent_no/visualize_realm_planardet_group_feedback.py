"""Visualize the frozen PlanarDet field-group recurrence diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_realm_planardet_group_feedback import (
    ALL_METRIC_GROUPS,
    ARM_ORDER,
    MATERIAL_EFFECT_RATIO,
    NEAR_NULL_RATIO,
    RESULT_SCHEMA,
)
from utility.time_dependent_no.realm_benchmark import canonical_json_sha256

PMAX_RESULT_SCHEMA = "w26_l4_planardet_pmax_projection_evaluation_v2"
VISUALIZATION_SCHEMA = "w26_l4_planardet_group_feedback_visualization_v1"

COLORS = {
    "baseline": "#6B7280",
    "chem": "#009E73",
    "T": "#D55E00",
    "rho": "#0072B2",
    "u": "#CC79A7",
    "p": "#E69F00",
}
MARKERS = {"chem": "o", "T": "s", "rho": "^", "u": "D"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pmax-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_result(result: Mapping[str, Any]) -> None:
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    if (
        result.get("schema") != RESULT_SCHEMA
        or canonical_json_sha256(unsigned) != result.get("canonical_payload_sha256")
        or result.get("diagnostic_interpretation_allowed") is not True
        or result.get("evaluator_closure", {}).get("all_gates_pass") is not True
        or result.get("test_object_opened") is not False
        or set(result.get("arms", {})) != set(ARM_ORDER)
    ):
        raise ValueError("group-feedback result identity or closure differs")


def _validate_pmax_result(result: Mapping[str, Any]) -> None:
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    if (
        result.get("schema") != PMAX_RESULT_SCHEMA
        or canonical_json_sha256(unsigned) != result.get("canonical_payload_sha256")
        or result.get("projection_diagnostic_interpretation_allowed") is not True
        or result.get("evaluator_closure", {}).get("all_gates_pass") is not True
        or result.get("test_object_opened") is not False
    ):
        raise ValueError("pMax reference result identity or closure differs")


def _group_means(summary: Mapping[str, Any]) -> dict[str, float]:
    grouped = summary.get("npe_group_by_call")
    if not isinstance(grouped, Mapping) or set(grouped) != set(ALL_METRIC_GROUPS):
        raise ValueError("summary lacks the frozen grouped errors")
    means: dict[str, float] = {}
    for group in ALL_METRIC_GROUPS:
        values = np.asarray(grouped[group], dtype=np.float64)
        if values.shape != (49,) or not np.isfinite(values).all():
            raise ValueError("grouped error histories must be finite H49 vectors")
        means[group] = float(values.mean())
    return means


def build_plot_data(
    result: Mapping[str, Any],
    *,
    pmax_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_result(result)
    baseline = result["baseline_free_summary"]
    baseline_groups = _group_means(baseline)
    baseline_total = np.asarray(baseline["npe_total_by_call"], dtype=np.float64)
    baseline_correlation = np.asarray(
        baseline["decoded_correlation_by_call"], dtype=np.float64
    )
    if (
        baseline_total.shape != (49,)
        or baseline_correlation.shape != (49,)
        or not np.isfinite(baseline_total).all()
        or not np.isfinite(baseline_correlation).all()
    ):
        raise ValueError("baseline histories must be finite H49 vectors")

    matrix = np.empty((len(ARM_ORDER), len(ALL_METRIC_GROUPS)), dtype=np.float64)
    primary_ratios: dict[str, float] = {}
    primary_ratio_by_call: dict[str, np.ndarray] = {}
    total_by_call = {"baseline": baseline_total}
    correlation_by_call = {"baseline": baseline_correlation}
    for row, arm in enumerate(ARM_ORDER):
        arm_payload = result["arms"][arm]
        summary = arm_payload["raw_summary"]
        means = _group_means(summary)
        for column, group in enumerate(ALL_METRIC_GROUPS):
            matrix[row, column] = means[group] / baseline_groups[group]
        comparison = arm_payload["comparison_to_unintervened_free_baseline"]
        primary = comparison["primary_untouched_non_pMax_groups"]
        primary_ratios[arm] = float(primary["ratio"])
        primary_groups = tuple(primary["groups"])
        baseline_per_call = np.sum(
            [baseline["npe_group_by_call"][group] for group in primary_groups], axis=0
        )
        feedback_per_call = np.sum(
            [summary["npe_group_by_call"][group] for group in primary_groups], axis=0
        )
        if not np.all(baseline_per_call > 0.0):
            raise ValueError("primary baseline error must be positive at every call")
        primary_ratio_by_call[arm] = feedback_per_call / baseline_per_call
        total_by_call[arm] = np.asarray(summary["npe_total_by_call"], dtype=np.float64)
        correlation_by_call[arm] = np.asarray(
            summary["decoded_correlation_by_call"], dtype=np.float64
        )

    pmax_ratio: float | None = None
    if pmax_result is not None:
        _validate_pmax_result(pmax_result)
        pmax_ratio = float(
            pmax_result["comparison_to_unprojected_baseline"]["free_recurrence"][
                "non_pMax_grouped_npe_mean"
            ]["ratio"]
        )
        if not math.isfinite(pmax_ratio) or pmax_ratio <= 0.0:
            raise ValueError("pMax reference ratio must be finite and positive")

    return {
        "baseline_group_means": baseline_groups,
        "group_ratio_matrix": matrix,
        "primary_ratios": primary_ratios,
        "pmax_primary_ratio": pmax_ratio,
        "primary_ratio_by_call": primary_ratio_by_call,
        "total_by_call": total_by_call,
        "correlation_by_call": correlation_by_call,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
            "lines.markersize": 4,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def _plot_primary_effects(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    labels = list(ARM_ORDER)
    values = [float(data["primary_ratios"][group]) for group in labels]
    pmax_ratio = data["pmax_primary_ratio"]
    if pmax_ratio is not None:
        labels.append("pMax (P0b)")
        values.append(float(pmax_ratio))
    colors = [
        COLORS["p"] if label.startswith("pMax") else COLORS[label] for label in labels
    ]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    ax.axvspan(
        1.0 - NEAR_NULL_RATIO,
        1.0 + NEAR_NULL_RATIO,
        color="#D1D5DB",
        alpha=0.45,
        label="near-null band",
    )
    ax.axvline(1.0 - MATERIAL_EFFECT_RATIO, color="#009E73", ls="--", lw=1)
    ax.axvline(1.0, color="#374151", lw=1)
    ax.axvline(1.0 + MATERIAL_EFFECT_RATIO, color="#D55E00", ls="--", lw=1)
    bars = ax.barh(y, values, color=colors, height=0.58, edgecolor="white")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(1.22, max(values) + 0.08))
    ax.set_xlabel("Untouched-group error ratio (feedback / baseline)")
    ax.set_title("Causal recurrence effect by corrected field group")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}×",
            va="center",
            fontsize=8,
        )
    ax.text(
        0.01,
        -0.21,
        "Lower is better. Each arm is scored only on groups untouched by that intervention.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#4B5563",
    )
    return _save(fig, output_dir, "primary_effect_ratios")


def _plot_group_matrix(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    matrix = np.asarray(data["group_ratio_matrix"], dtype=np.float64)
    log_matrix = np.log2(matrix)
    bound = max(2.0, float(np.abs(log_matrix).max()))
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    image = ax.imshow(
        log_matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(ALL_METRIC_GROUPS)), ALL_METRIC_GROUPS)
    ax.set_yticks(np.arange(len(ARM_ORDER)), ARM_ORDER)
    ax.set_xlabel("Raw proposal group scored")
    ax.set_ylabel("Truth-feedback group")
    ax.set_title("Cross-group recurrence response")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if abs(log_matrix[row, column]) > 1.25 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}×",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)
    colorbar.set_label("log$_2$(feedback / baseline)")
    return _save(fig, output_dir, "cross_group_error_ratio_matrix")


def _plot_error_accumulation(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    calls = np.arange(1, 50)
    fig, axes = plt.subplots(2, 1, figsize=(6.75, 5.0), sharex=True)
    top, bottom = axes
    top.plot(
        calls,
        data["total_by_call"]["baseline"],
        color=COLORS["baseline"],
        label="unintervened baseline",
        zorder=4,
    )
    for arm in ARM_ORDER:
        top.plot(
            calls,
            data["total_by_call"][arm],
            color=COLORS[arm],
            label=f"{arm} feedback",
            marker=MARKERS[arm],
            markevery=8,
        )
    top.set_ylabel("All-group NPE per call")
    top.set_title("Raw-proposal error accumulation")
    top.legend(ncol=3, loc="upper left")
    top.set_ylim(bottom=0.0)

    bottom.axhspan(
        1.0 - NEAR_NULL_RATIO,
        1.0 + NEAR_NULL_RATIO,
        color="#D1D5DB",
        alpha=0.45,
    )
    bottom.axhline(1.0 - MATERIAL_EFFECT_RATIO, color="#009E73", ls="--", lw=1)
    bottom.axhline(1.0, color="#374151", lw=1)
    bottom.axhline(1.0 + MATERIAL_EFFECT_RATIO, color="#D55E00", ls="--", lw=1)
    for arm in ARM_ORDER:
        bottom.plot(
            calls,
            data["primary_ratio_by_call"][arm],
            color=COLORS[arm],
            label=arm,
            marker=MARKERS[arm],
            markevery=8,
        )
    bottom.set_xlabel("Autoregressive call")
    bottom.set_ylabel("Untouched non-pMax ratio")
    bottom.set_title("Causal effect over the rollout")
    bottom.set_xlim(1, 49)
    bottom.set_ylim(0.25, 1.35)
    bottom.legend(ncol=4, loc="lower left")
    fig.tight_layout(h_pad=1.3)
    return _save(fig, output_dir, "error_accumulation")


def _write_csvs(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    primary_path = output_dir / "primary_effect_ratios.csv"
    with primary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feedback_group", "primary_untouched_non_pMax_ratio"])
        for arm in ARM_ORDER:
            writer.writerow([arm, data["primary_ratios"][arm]])
        if data["pmax_primary_ratio"] is not None:
            writer.writerow(["pMax_P0b", data["pmax_primary_ratio"]])

    matrix_path = output_dir / "cross_group_error_ratio_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feedback_group", *ALL_METRIC_GROUPS])
        for arm, row in zip(ARM_ORDER, data["group_ratio_matrix"], strict=True):
            writer.writerow([arm, *row.tolist()])

    per_call_path = output_dir / "error_accumulation.csv"
    with per_call_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["call", "baseline_all_group_npe"]
        for arm in ARM_ORDER:
            header.extend([f"{arm}_all_group_npe", f"{arm}_primary_ratio"])
        writer.writerow(header)
        for index in range(49):
            row: list[Any] = [index + 1, data["total_by_call"]["baseline"][index]]
            for arm in ARM_ORDER:
                row.extend(
                    [
                        data["total_by_call"][arm][index],
                        data["primary_ratio_by_call"][arm][index],
                    ]
                )
            writer.writerow(row)
    return [primary_path, matrix_path, per_call_path]


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    result = _load_json(args.result)
    pmax_result = _load_json(args.pmax_result) if args.pmax_result else None
    data = build_plot_data(result, pmax_result=pmax_result)
    args.output_dir.mkdir(parents=True)
    _style()
    outputs = []
    outputs.extend(_plot_primary_effects(data, args.output_dir))
    outputs.extend(_plot_group_matrix(data, args.output_dir))
    outputs.extend(_plot_error_accumulation(data, args.output_dir))
    outputs.extend(_write_csvs(data, args.output_dir))
    manifest: dict[str, Any] = {
        "schema": VISUALIZATION_SCHEMA,
        "inputs": {
            "result_sha256": _sha256(args.result),
            "pmax_result_sha256": (
                _sha256(args.pmax_result) if args.pmax_result else None
            ),
        },
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        },
        "test_object_opened": False,
    }
    manifest["canonical_payload_sha256"] = canonical_json_sha256(manifest)
    manifest_path = args.output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_visualization(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
