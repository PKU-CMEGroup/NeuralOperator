#!/usr/bin/env python3
"""Visualize the completed D094 B1-C5-A repeatability matrix.

The figures keep three questions separate: aggregate H79 ordering, the
horizon-dependent selected-to-terminal crossover, and local numerical
sensitivity in per-case rankings and physical-event detection.  FP32 is a
precision control, not a ground-truth evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_repeatability import (
    ARTIFACT_SCHEMA as ANALYSIS_ARTIFACT_SCHEMA,
)
from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_repeatability import (
    EXPECTED_EXECUTIONS,
)
from scripts.time_dependent_no.analyze_pcno_bump_b1_c5_repeatability import (
    SCHEMA as ANALYSIS_SCHEMA,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    SCHEMA as EVALUATOR_SCHEMA,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    sha256_file,
)

SCHEMA = "d094_b1_c5_repeatability_visualization_v1"
TRAJECTORY_COUNTS = (128, 256)
PRIMARY_ROLES = ("selected", "terminal")
HORIZONS = (20, 40, 60, 79)
BF16_EXECUTIONS = EXPECTED_EXECUTIONS[:3]
FP32_EXECUTION = EXPECTED_EXECUTIONS[3]

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
GRAY = "#7A7A7A"
ROLE_COLORS = {"selected": BLUE, "terminal": VERMILLION}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV must contain at least one row: {path}")
    return rows


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _validate_record(path: Path, record: Mapping[str, Any]) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != str(record["sha256"])
    ):
        raise ValueError(f"bound artifact changed: {path}")


def _validate_artifact_manifest(
    directory: Path, expected_schema: str
) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != expected_schema
        or manifest.get("historical_test_population_accessed") is not False
    ):
        raise ValueError(f"invalid artifact manifest: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError(f"artifact manifest lacks files: {manifest_path}")
    for relative, record in files.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"invalid file record: {relative}")
        _validate_record(directory / str(relative), record)
    return manifest


def _cell_map(summary: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    cells: dict[tuple[int, str], Mapping[str, Any]] = {}
    for cell in summary["cells"]:
        key = int(cell["trajectory_count"]), str(cell["checkpoint_role"])
        if key in cells:
            raise ValueError(f"duplicate evaluator cell: {key}")
        cells[key] = cell
    expected = {(count, role) for count in TRAJECTORY_COUNTS for role in PRIMARY_ROLES}
    if not expected <= set(cells):
        raise ValueError("evaluator summary lacks a primary cell")
    return cells


def load_matrix(
    matrix_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Load only hash-bound summaries and analysis tables."""

    if matrix_dir.is_symlink() or not matrix_dir.is_dir():
        raise ValueError("matrix directory must be a regular directory")
    analysis_dir = matrix_dir / "analysis"
    _validate_artifact_manifest(analysis_dir, ANALYSIS_ARTIFACT_SCHEMA)
    analysis = _load_json(analysis_dir / "summary.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("historical_test_population_accessed") is not False
        or analysis.get("checkpoint_reselection_on_outside_cases") is not False
    ):
        raise ValueError(
            "B1-C5-A analysis is incomplete or violates its access contract"
        )

    records = analysis.get("input_summaries")
    if not isinstance(records, list) or len(records) != len(EXPECTED_EXECUTIONS):
        raise ValueError("B1-C5-A input-summary binding is incomplete")
    by_execution = {str(record["execution"]): record for record in records}
    if tuple(by_execution) != EXPECTED_EXECUTIONS:
        raise ValueError("B1-C5-A execution order changed")

    summaries: dict[str, dict[str, Any]] = {}
    for execution in EXPECTED_EXECUTIONS:
        execution_dir = matrix_dir / execution
        path = execution_dir / str(by_execution[execution]["path_name"])
        _validate_record(path, by_execution[execution])
        evaluator_manifest = _load_json(execution_dir / "artifact_manifest.json")
        if evaluator_manifest.get("historical_test_population_accessed") is not False:
            raise ValueError(f"historical access was not false: {execution}")
        summary_record = evaluator_manifest.get("files", {}).get("summary.json")
        if not isinstance(summary_record, Mapping):
            raise TypeError(
                f"evaluator manifest does not bind summary.json: {execution}"
            )
        _validate_record(path, summary_record)

        summary = _load_json(path)
        expected_amp = "bf16" if execution in BF16_EXECUTIONS else "none"
        if (
            summary.get("schema") != EVALUATOR_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("historical_test_population_accessed") is not False
            or summary.get("checkpoint_reselection_on_outside_cases") is not False
            or summary.get("evaluation_numerics", {}).get("amp") != expected_amp
        ):
            raise ValueError(f"invalid evaluator summary: {execution}")
        for key, cell in _cell_map(summary).items():
            if key[1] not in PRIMARY_ROLES:
                continue
            rollout = cell["outside_selection_rollout"]
            if (
                float(rollout["completion_rate"]) != 1.0
                or int(rollout["hard_failure_count"]) != 0
                or int(rollout["num_trajectories"]) != 28
            ):
                raise ValueError(
                    f"primary rollout did not complete cleanly: {execution} {key}"
                )
        summaries[execution] = summary

    return (
        analysis,
        summaries,
        _read_csv(analysis_dir / "cell_repeatability.csv"),
        _read_csv(analysis_dir / "per_case_selected_ordering.csv"),
        _read_csv(analysis_dir / "physical_event_repeatability.csv"),
    )


def endpoint_profiles(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int, str], np.ndarray]:
    profiles: dict[tuple[str, int, str], np.ndarray] = {}
    for execution in EXPECTED_EXECUTIONS:
        cells = _cell_map(summaries[execution])
        for count in TRAJECTORY_COUNTS:
            for role in PRIMARY_ROLES:
                endpoints = cells[count, role]["outside_selection_rollout"][
                    "mean_endpoint_relative_l2"
                ]
                values = np.asarray([float(endpoints[str(step)]) for step in HORIZONS])
                if values.shape != (len(HORIZONS),) or not np.all(np.isfinite(values)):
                    raise ValueError("nonfinite or incomplete endpoint profile")
                profiles[execution, count, role] = values
    return profiles


def horizon_crossovers(
    profiles: Mapping[tuple[str, int, str], np.ndarray],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for count in TRAJECTORY_COUNTS:
        selected = np.mean(
            [profiles[execution, count, "selected"] for execution in BF16_EXECUTIONS],
            axis=0,
        )
        terminal = np.mean(
            [profiles[execution, count, "terminal"] for execution in BF16_EXECUTIONS],
            axis=0,
        )
        delta = terminal - selected
        result[str(count)] = {
            "h20_terminal_minus_selected": float(delta[0]),
            "h20_percent_of_selected": float(100.0 * delta[0] / selected[0]),
            "h79_terminal_minus_selected": float(delta[-1]),
            "h79_percent_of_selected": float(100.0 * delta[-1] / selected[-1]),
        }
    return result


def selected_case_points(
    summaries: Mapping[str, Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    execution_values: dict[tuple[str, int], dict[str, float]] = {}
    for execution in EXPECTED_EXECUTIONS:
        cells = _cell_map(summaries[execution])
        for count in TRAJECTORY_COUNTS:
            trajectories = cells[count, "selected"]["outside_selection_rollout"][
                "trajectories"
            ]
            execution_values[execution, count] = {
                str(row["trajectory"]): float(row["final_relative_l2"])
                for row in trajectories
            }

    points: list[dict[str, Any]] = []
    for row in case_rows:
        trajectory = str(row["trajectory"])
        values = {
            count: np.asarray(
                [
                    execution_values[execution, count][trajectory]
                    for execution in EXPECTED_EXECUTIONS
                ]
            )
            for count in TRAJECTORY_COUNTS
        }
        for index, execution in enumerate(EXPECTED_EXECUTIONS):
            recorded = float(row[f"{execution}_n256_minus_n128_h79"])
            actual = float(values[256][index] - values[128][index])
            if not math.isclose(recorded, actual, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"per-case delta binding changed: {trajectory}")
        points.append(
            {
                "trajectory": trajectory,
                "n128_mean": float(np.mean(values[128])),
                "n256_mean": float(np.mean(values[256])),
                "stable_winner": _is_true(row["stable_winner"]),
                "winner": str(row["winner"]),
            }
        )
    return points


def physical_event_counts(
    event_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, str], dict[str, int]]:
    counts = {
        (count, role): {
            "stable_no_violation": 0,
            "stable_violation": 0,
            "precision_sensitive": 0,
        }
        for count in TRAJECTORY_COUNTS
        for role in PRIMARY_ROLES
    }
    call_fields = tuple(f"{execution}_call" for execution in EXPECTED_EXECUTIONS)
    for row in event_rows:
        key = int(row["trajectory_count"]), str(row["checkpoint_role"])
        if key not in counts:
            raise ValueError(f"unexpected physical-event cell: {key}")
        calls = [str(row[field]).strip() for field in call_fields]
        if not _is_true(row["event_stable"]):
            category = "precision_sensitive"
        elif any(calls):
            category = "stable_violation"
        else:
            category = "stable_no_violation"
        counts[key][category] += 1
    if any(sum(value.values()) != 28 for value in counts.values()):
        raise ValueError("physical-event table does not contain 28 cases per cell")
    return counts


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "grid.alpha": 0.16,
            "lines.linewidth": 1.7,
        }
    )
    return plt


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[dict[str, Any]]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return outputs


def plot_horizon_profiles(
    profiles: Mapping[tuple[str, int, str], np.ndarray],
    crossovers: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.35), sharex=True, sharey=True)
    for axis, count in zip(axes, TRAJECTORY_COUNTS, strict=True):
        for role in PRIMARY_ROLES:
            bf16 = np.stack(
                [profiles[execution, count, role] for execution in BF16_EXECUTIONS]
            )
            mean = np.mean(bf16, axis=0)
            axis.fill_between(
                HORIZONS,
                np.min(bf16, axis=0),
                np.max(bf16, axis=0),
                color=ROLE_COLORS[role],
                alpha=0.14,
                linewidth=0,
            )
            axis.plot(
                HORIZONS,
                mean,
                color=ROLE_COLORS[role],
                marker="o" if role == "selected" else "s",
                label=f"{role} BF16 mean/range",
            )
            axis.plot(
                HORIZONS,
                profiles[FP32_EXECUTION, count, role],
                color=ROLE_COLORS[role],
                linestyle="none",
                marker="x",
                markersize=5.5,
                markeredgewidth=1.2,
                label=f"{role} FP32 control",
            )
        values = crossovers[str(count)]
        axis.text(
            0.035,
            0.965,
            (
                f"terminal - selected\n"
                f"H20: {values['h20_percent_of_selected']:+.1f}%\n"
                f"H79: {values['h79_percent_of_selected']:+.1f}%"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
        )
        axis.set_title(f"PCNO, n={count}")
        axis.set_xlabel("Autoregressive call")
        axis.set_xticks(HORIZONS)
    axes[0].set_ylabel("Outside-cohort relative L2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "D094 B1-C5-A: later training helps early rollout but hurts the tail",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    outputs = _save_figure(fig, output_dir, "b1_c5_horizon_crossover")
    plt.close(fig)
    return outputs


def plot_repeatability_summary(
    cell_rows: Sequence[Mapping[str, str]],
    case_points: Sequence[Mapping[str, Any]],
    event_counts: Mapping[tuple[int, str], Mapping[str, int]],
    analysis: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    ax_h79, ax_floor, ax_cases, ax_events = axes.ravel()

    labels = []
    cell_by_key = {}
    for row in cell_rows:
        key = int(row["trajectory_count"]), str(row["checkpoint_role"])
        cell_by_key[key] = row
    keys = [(128, "selected"), (128, "terminal"), (256, "selected"), (256, "terminal")]
    execution_offsets = dict(
        zip(EXPECTED_EXECUTIONS, (-0.12, -0.04, 0.04, 0.12), strict=True)
    )
    for x, key in enumerate(keys):
        row = cell_by_key[key]
        labels.append(f"n={key[0]}\n{key[1]}")
        values = {
            execution: (
                float(row[f"{execution}_h79"])
                if execution in BF16_EXECUTIONS
                else float(row["fp32_h79"])
            )
            for execution in EXPECTED_EXECUTIONS
        }
        for execution, value in values.items():
            ax_h79.scatter(
                x + execution_offsets[execution],
                value,
                color=BLUE if execution in BF16_EXECUTIONS else VERMILLION,
                marker="o" if execution in BF16_EXECUTIONS else "X",
                s=28,
                zorder=4,
            )
    for count, start in ((128, 0), (256, 2)):
        for execution in EXPECTED_EXECUTIONS:
            first = cell_by_key[count, "selected"]
            second = cell_by_key[count, "terminal"]
            first_value = (
                float(first[f"{execution}_h79"])
                if execution in BF16_EXECUTIONS
                else float(first["fp32_h79"])
            )
            second_value = (
                float(second[f"{execution}_h79"])
                if execution in BF16_EXECUTIONS
                else float(second["fp32_h79"])
            )
            offset = execution_offsets[execution]
            ax_h79.plot(
                [start + offset, start + 1 + offset],
                [first_value, second_value],
                color="#BBBBBB",
                linewidth=0.7,
                zorder=1,
            )
    ax_h79.scatter([], [], color=BLUE, marker="o", label="BF16 process")
    ax_h79.scatter([], [], color=VERMILLION, marker="X", label="FP32 control")
    ax_h79.set_xticks(range(len(labels)), labels)
    ax_h79.set_ylabel("H79 relative L2")
    ax_h79.set_title("A  Aggregate ordering and checkpoint direction")
    ax_h79.legend(loc="upper left")
    ax_h79.text(
        0.98,
        0.04,
        "magnified y-axis",
        transform=ax_h79.transAxes,
        ha="right",
        fontsize=7,
    )

    gap = float(analysis["selected_count_ordering"]["smallest_cross_count_h79_gap"])
    fractions = [
        100.0 * float(cell_by_key[key]["all_execution_h79_range"]) / gap for key in keys
    ]
    ax_floor.bar(
        range(len(keys)),
        fractions,
        color=[BLUE, BLUE, VERMILLION, VERMILLION],
        alpha=0.82,
    )
    ax_floor.axhline(
        25.0,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="preregistered ceiling",
    )
    ax_floor.set_xticks(range(len(labels)), labels)
    ax_floor.set_ylabel("Numerical range / smallest n-gap (%)")
    ax_floor.set_title("B  Aggregate effect exceeds evaluator spread")
    ax_floor.legend(loc="upper right")
    for index, value in enumerate(fractions):
        ax_floor.text(index, value + 0.8, f"{value:.1f}%", ha="center", fontsize=7.5)
    ax_floor.set_ylim(0, max(28.5, max(fractions) + 4.0))

    categories = {
        "n128": (BLUE, "o", "stable n=128 win"),
        "n256": (VERMILLION, "s", "stable n=256 win"),
        "unstable": (GRAY, "D", "precision-sensitive winner"),
    }
    for category, (color, marker, label) in categories.items():
        rows = [
            point
            for point in case_points
            if (point["winner"] if point["stable_winner"] else "unstable") == category
        ]
        ax_cases.scatter(
            [point["n128_mean"] for point in rows],
            [point["n256_mean"] for point in rows],
            color=color,
            marker=marker,
            s=28,
            alpha=0.88,
            label=label,
        )
    all_values = [
        float(point[field])
        for point in case_points
        for field in ("n128_mean", "n256_mean")
    ]
    lower, upper = min(all_values) * 0.88, max(all_values) * 1.12
    ax_cases.plot(
        [lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=0.9
    )
    ax_cases.set_xscale("log")
    ax_cases.set_yscale("log")
    ax_cases.set_xlim(lower, upper)
    ax_cases.set_ylim(lower, upper)
    ax_cases.set_xlabel("n=128 selected H79 (four-run mean)")
    ax_cases.set_ylabel("n=256 selected H79 (four-run mean)")
    ax_cases.set_title("C  Aggregate ordering hides case heterogeneity")
    ax_cases.legend(loc="upper left")
    for point in case_points:
        if point["trajectory"] in {"25", "83", "152"}:
            ax_cases.annotate(
                point["trajectory"],
                (point["n128_mean"], point["n256_mean"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    event_labels = [f"n={count}\n{role}" for count, role in keys]
    bottoms = np.zeros(len(keys), dtype=np.float64)
    for category, color, label in (
        ("stable_no_violation", GREEN, "stable no violation"),
        ("stable_violation", PURPLE, "stable violation"),
        ("precision_sensitive", YELLOW, "precision-sensitive event"),
    ):
        values = np.asarray([event_counts[key][category] for key in keys])
        ax_events.bar(
            range(len(keys)), values, bottom=bottoms, color=color, label=label
        )
        bottoms += values
    ax_events.set_xticks(range(len(keys)), event_labels)
    ax_events.set_ylim(0, 31)
    ax_events.set_ylabel("Outside cases (of 28)")
    ax_events.set_title("D  Admissibility events are locally precision-sensitive")
    ax_events.legend(loc="lower left")
    ax_events.text(
        0.98,
        0.05,
        "448/448 rollouts completed\n0 hard failures",
        transform=ax_events.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )

    fig.suptitle(
        "D094 B1-C5-A: robust aggregate signal, fragile local events", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outputs = _save_figure(fig, output_dir, "b1_c5_repeatability_summary")
    plt.close(fig)
    return outputs


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis, summaries, cell_rows, case_rows, event_rows = load_matrix(args.matrix_dir)
    profiles = endpoint_profiles(summaries)
    crossovers = horizon_crossovers(profiles)
    case_points = selected_case_points(summaries, case_rows)
    events = physical_event_counts(event_rows)

    figures = []
    figures.extend(plot_horizon_profiles(profiles, crossovers, args.output_dir))
    figures.extend(
        plot_repeatability_summary(
            cell_rows, case_points, events, analysis, args.output_dir
        )
    )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "figures": figures,
        "source_bindings": {
            "analysis_summary_sha256": sha256_file(
                args.matrix_dir / "analysis" / "summary.json"
            ),
            "analysis_artifact_manifest_sha256": sha256_file(
                args.matrix_dir / "analysis" / "artifact_manifest.json"
            ),
            "retrieval_manifest_sha256": (
                sha256_file(args.matrix_dir / "retrieval.sha256")
                if (args.matrix_dir / "retrieval.sha256").is_file()
                else None
            ),
            "evaluator_summaries": {
                execution: sha256_file(args.matrix_dir / execution / "summary.json")
                for execution in EXPECTED_EXECUTIONS
            },
            "renderer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "derived_findings": {
            "horizon_crossovers_bf16_mean": crossovers,
            "selected_count_ordering": analysis["selected_count_ordering"],
            "selected_to_terminal_direction": analysis[
                "selected_to_terminal_direction"
            ],
            "per_case_selected_ordering": analysis["per_case_selected_ordering"],
            "maximum_disagreement_case": analysis["maximum_disagreement_case"],
            "physical_event_repeatability": analysis["physical_event_repeatability"],
            "physical_event_categories": {
                f"n{count}_{role}": dict(events[count, role])
                for count in TRAJECTORY_COUNTS
                for role in PRIMARY_ROLES
            },
            "primary_rollouts_completed": 448,
            "primary_hard_failures": 0,
        },
        "figure_semantics": {
            "bf16_band": "minimum-to-maximum range over three fresh processes",
            "fp32_marker": "precision control; not ground truth",
            "per_case_point": "mean over three BF16 processes and one FP32 control",
            "physical_event_row": "presence, cause, and first-call agreement across four executions",
        },
        "claims_not_supported": [
            "multi-seed training claim",
            "architecture comparison",
            "capacity bottleneck",
            "optimizer or representation convergence",
            "FP32 as ground-truth evaluation",
            "physical conservation",
            "historical test-population claim",
        ],
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "figures": len(figures)}, sort_keys=True))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
