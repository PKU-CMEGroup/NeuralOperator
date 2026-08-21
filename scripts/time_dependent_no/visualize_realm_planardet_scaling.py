"""Compare D092/D093 PlanarDet training dynamics and open-validation rollouts.

The visualizer consumes manifest-closed evaluation arrays.  It renders common-
scale architecture/data-exposure comparisons, including truth-fed one-step
residual animations that never use accumulated free-rollout predictions and a
frozen-initial ``U(t) = U(0)`` baseline on the same normalized-state metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_realm_planardet_scaling import (
    EVALUATION_SOURCE_PATHS,
)
from scripts.time_dependent_no.visualize_realm_planardet_pcno import (
    _decode_visualization_bundle,
    _oriented,
)
from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    grouped_normalized_prediction_error,
)
from utility.time_dependent_no.realm_planardet import (
    PLANARDET_FIELDS,
    PLANARDET_GROUPS,
    PLANARDET_VAL_GROUPS,
    load_planardet_metadata,
    sha256_file,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    VALIDATION_HORIZON,
    load_normalized_trajectories,
    load_normalizer_bundle,
)

PAPER_FFNO_VALIDATION_SUM = 12.577
PAPER_FFNO_TEST_SUM = 9.5377
PAPER_FFNO_TEST_CORRELATION = 0.99144
ONE_CALL_STEPS = 490
PRESENTATIONS_PER_OPTIMIZER_STEP = 7
TWO_CALL_WINDOW_COUNT = 48
VISUALIZER_PATH = "scripts/time_dependent_no/visualize_realm_planardet_scaling.py"
BASE_VISUALIZER_PATH = "scripts/time_dependent_no/visualize_realm_planardet_pcno.py"
VISUALIZATION_SOURCE_PATHS = tuple(
    sorted({*EVALUATION_SOURCE_PATHS, VISUALIZER_PATH, BASE_VISUALIZER_PATH})
)

ARCHITECTURES = ("pcno", "pcfno", "ffno")
ARCHITECTURE_LABELS = {"pcno": "PCNO", "pcfno": "PCFNO", "ffno": "FFNO"}
ARCHITECTURE_COLORS = {
    "pcno": "#0072B2",
    "pcfno": "#E69F00",
    "ffno": "#009E73",
}
EXPOSURE_STYLES = {
    7: {"linestyle": "-", "marker": "o", "label": "7 trajectories"},
    3: {"linestyle": "--", "marker": "s", "label": "3 trajectories"},
}

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
        "grid.linestyle": "-",
        "lines.linewidth": 1.7,
    }
)


@dataclass(frozen=True)
class ModelRecord:
    key: str
    architecture: str
    exposure: int
    training_dir: Path
    evaluation_dir: Path
    training_complete: bool

    @property
    def label(self) -> str:
        return f"{ARCHITECTURE_LABELS[self.architecture]}-{self.exposure}"


def _sampler_accounting(exposure: int, completed_step: int) -> dict[str, float | int]:
    """Describe D093's shared-window accumulation without inventing exposure."""

    if exposure not in (3, 7):
        raise ValueError("D093 exposure must be three or seven trajectories")
    if completed_step < 0:
        raise ValueError("completed step must be nonnegative")
    two_call_steps = max(0, completed_step - ONE_CALL_STEPS)
    return {
        "optimizer_steps": completed_step,
        "raw_micro_presentations": (
            completed_step * PRESENTATIONS_PER_OPTIMIZER_STEP
        ),
        "active_condition_window_terms_per_step": exposure,
        "duplicate_condition_window_terms_per_step": (
            PRESENTATIONS_PER_OPTIMIZER_STEP - exposure
        ),
        "scheduled_window_positions_per_step": 1,
        "two_call_window_cycles_per_active_trajectory": (
            two_call_steps / TWO_CALL_WINDOW_COUNT
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d092-training-dir", type=Path, required=True)
    parser.add_argument("--d092-evaluation-dir", type=Path, required=True)
    parser.add_argument("--d093-training-root", type=Path, required=True)
    parser.add_argument("--d093-evaluation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--animation-fields",
        nargs="+",
        default=("H2O", "T", "pMax"),
    )
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--animation-dpi", type=int, default=90)
    parser.add_argument("--skip-animations", action="store_true")
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path.name}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV output requires at least one row")
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_pair(fig: plt.Figure, base: Path) -> tuple[Path, Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def _validate_fields(values: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(values)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("animation fields must be nonempty and unique")
    unknown = set(fields) - set(PLANARDET_FIELDS)
    if unknown:
        raise ValueError(f"unknown PlanarDet fields: {sorted(unknown)}")
    return fields


def _validate_evaluation(directory: Path) -> Mapping[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("evaluation directory must be regular")
    final = _load_json(directory / "final_hash_manifest.json")
    files = final.get("files")
    if (
        final.get("self_hash_excluded") is not True
        or final.get("test_object_opened") is not False
        or not isinstance(files, Mapping)
    ):
        raise ValueError("evaluation final manifest differs")
    for name, expected in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"evaluation artifact hash differs: {name}")
    result = _load_json(directory / "result.json")
    digest = result.get("canonical_payload_sha256")
    unsigned = {
        key: value for key, value in result.items() if key != "canonical_payload_sha256"
    }
    if (
        digest != canonical_json_sha256(unsigned)
        or result.get("test_object_opened") is not False
        or result.get("evaluator_closure", {}).get("all_gates_pass") is not True
    ):
        raise ValueError("evaluation result identity or closure differs")
    return result


def _model_records(args: argparse.Namespace) -> tuple[ModelRecord, ...]:
    d093 = {
        ("pcno", 3): "pcno_n3_seed0_v3",
        ("pcfno", 7): "pcfno_n7_seed0_v3",
        ("pcfno", 3): "pcfno_n3_seed0_v3",
        ("ffno", 7): "ffno_n7_seed0_v3",
        ("ffno", 3): "ffno_n3_seed0_v3",
    }
    records = [
        ModelRecord(
            key="pcno_n7",
            architecture="pcno",
            exposure=7,
            training_dir=args.d092_training_dir,
            evaluation_dir=args.d092_evaluation_dir,
            training_complete=True,
        )
    ]
    for (architecture, exposure), name in d093.items():
        records.append(
            ModelRecord(
                key=name,
                architecture=architecture,
                exposure=exposure,
                training_dir=args.d093_training_root / name,
                evaluation_dir=args.d093_evaluation_root / name,
                training_complete=architecture == "pcno" or (
                    architecture == "ffno" and exposure == 7
                ),
            )
        )
    return tuple(
        sorted(records, key=lambda item: (item.exposure != 7, ARCHITECTURES.index(item.architecture)))
    )


def _history_rows(record: ModelRecord) -> list[Mapping[str, Any]]:
    history = _load_json(record.training_dir / "history.json")
    rows = history.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{record.label} history lacks rows")
    steps = [row.get("completed_step") for row in rows]
    if steps[0] != 1 or steps != sorted(set(steps)):
        raise ValueError(f"{record.label} history step sequence differs")
    return rows


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if row.get("checkpoint_eligible") is True]
    if not eligible:
        raise ValueError("history has no eligible checkpoint")
    return min(eligible, key=lambda row: float(row["validation"]["realm_npe_mean"]))


def _training_table(
    records: Sequence[ModelRecord],
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for row in histories[record.key]:
            accounting = _sampler_accounting(record.exposure, int(row["completed_step"]))
            rows.append(
                {
                    "model": record.label,
                    "architecture": record.architecture,
                    "train_trajectories": record.exposure,
                    "training_complete": record.training_complete,
                    "optimizer_step": int(row["completed_step"]),
                    "training_calls": int(row["calls"]),
                    "interval_train_grouped_mse": float(
                        row["interval_mean_train_grouped_loss"]
                    ),
                    "truth_input_validation_npe": float(
                        row["validation"]["realm_npe_mean"]
                    ),
                    "learning_rate": float(row["learning_rate_used"]),
                    **accounting,
                }
            )
    return rows


def _render_loss_curves(
    records: Sequence[ModelRecord],
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), sharex="col")
    for column, architecture in enumerate(ARCHITECTURES):
        for exposure in (7, 3):
            record = next(
                item
                for item in records
                if item.architecture == architecture and item.exposure == exposure
            )
            rows = histories[record.key]
            steps = np.asarray([row["completed_step"] for row in rows])
            train = np.asarray([row["interval_mean_train_grouped_loss"] for row in rows])
            validation = np.asarray(
                [row["validation"]["realm_npe_mean"] for row in rows]
            )
            style = EXPOSURE_STYLES[exposure]
            color = "#0072B2" if exposure == 7 else "#D55E00"
            axes[0, column].semilogy(
                steps,
                train,
                color=color,
                linestyle=style["linestyle"],
                label=style["label"],
            )
            axes[1, column].semilogy(
                steps,
                validation,
                color=color,
                linestyle=style["linestyle"],
                label=style["label"],
            )
            best = _best_row(rows)
            axes[1, column].scatter(
                [best["completed_step"]],
                [best["validation"]["realm_npe_mean"]],
                color=color,
                marker=style["marker"],
                s=28,
                zorder=5,
            )
            if not record.training_complete:
                axes[0, column].scatter(
                    [steps[-1]], [train[-1]], color="#000000", marker="x", s=28
                )
                axes[1, column].scatter(
                    [steps[-1]], [validation[-1]], color="#000000", marker="x", s=28
                )
        for row in range(2):
            axes[row, column].axvline(
                ONE_CALL_STEPS,
                color="#777777",
                linestyle=":",
                linewidth=1.1,
            )
        axes[0, column].set_title(ARCHITECTURE_LABELS[architecture])
        axes[0, column].set_ylabel("Interval training objective")
        axes[1, column].set_ylabel("Truth-input validation NPE")
        axes[1, column].set_xlabel("Optimizer step")
        axes[0, column].legend(loc="best")
    fig.suptitle(
        "PlanarDet training dynamics: matched schedule, three vs seven conditions",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        "Dotted line: switch after step 490 from one-call to detached two-call training. "
        "All seven accumulated micro-presentations share one temporal window; the "
        "three-condition arm duplicates four condition-window terms before each update. "
        "Black x: last successfully serialized validation before an interrupted run.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    return _save_pair(fig, output_dir / "training_validation_curves")


def _summary_rows(
    records: Sequence[ModelRecord],
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        history = histories[record.key]
        best = _best_row(history)
        last = history[-1]
        best_accounting = _sampler_accounting(
            record.exposure, int(best["completed_step"])
        )
        last_accounting = _sampler_accounting(
            record.exposure, int(last["completed_step"])
        )
        result = results[record.key]
        teacher = result["views"]["ordered_teacher_forced"]["summary"]
        free = result["views"]["free_recurrence"]["summary"]
        free_sum = float(free["realm_npe_sum_source"])
        teacher_sum = float(teacher["realm_npe_sum_source"])
        rows.append(
            {
                "model": record.label,
                "architecture": record.architecture,
                "train_trajectories": record.exposure,
                "training_complete": record.training_complete,
                "best_step": int(best["completed_step"]),
                "best_interval_train_objective": float(
                    best["interval_mean_train_grouped_loss"]
                ),
                "best_truth_input_validation_npe": float(
                    best["validation"]["realm_npe_mean"]
                ),
                "last_successful_step": int(last["completed_step"]),
                "last_interval_train_objective": float(
                    last["interval_mean_train_grouped_loss"]
                ),
                "last_truth_input_validation_npe": float(
                    last["validation"]["realm_npe_mean"]
                ),
                "last_over_best_validation": float(
                    last["validation"]["realm_npe_mean"]
                    / best["validation"]["realm_npe_mean"]
                ),
                "teacher_h49_npe_sum": teacher_sum,
                "free_h49_npe_sum": free_sum,
                "free_over_teacher_sum": free_sum / teacher_sum,
                "free_over_paper_ffno_validation": free_sum
                / PAPER_FFNO_VALIDATION_SUM,
                "teacher_admissible_calls": int(teacher["admissible_call_count"]),
                "free_admissible_calls": int(free["admissible_call_count"]),
                "teacher_bounded_calls": int(teacher["bounded_call_count"]),
                "free_bounded_calls": int(free["bounded_call_count"]),
                "teacher_decoded_finite": bool(teacher["all_decoded_finite"]),
                "free_decoded_finite": bool(free["all_decoded_finite"]),
                "best_raw_micro_presentations": int(
                    best_accounting["raw_micro_presentations"]
                ),
                "best_two_call_window_cycles_per_active_trajectory": float(
                    best_accounting["two_call_window_cycles_per_active_trajectory"]
                ),
                "last_raw_micro_presentations": int(
                    last_accounting["raw_micro_presentations"]
                ),
                "last_two_call_window_cycles_per_active_trajectory": float(
                    last_accounting["two_call_window_cycles_per_active_trajectory"]
                ),
                "duplicate_condition_window_terms_per_step": int(
                    best_accounting["duplicate_condition_window_terms_per_step"]
                ),
            }
        )
    return rows


def _render_best_teacher(
    summary_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(ARCHITECTURES))
    width = 0.34
    for offset, exposure in ((-width / 2, 7), (width / 2, 3)):
        values = [
            next(
                float(row["best_truth_input_validation_npe"])
                for row in summary_rows
                if row["architecture"] == architecture
                and row["train_trajectories"] == exposure
            )
            for architecture in ARCHITECTURES
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color="#0072B2" if exposure == 7 else "#D55E00",
            label=f"{exposure} trajectories",
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
    ax.set_yscale("log")
    ax.set_xticks(x, [ARCHITECTURE_LABELS[value] for value in ARCHITECTURES])
    ax.set_ylabel("Best truth-input validation NPE")
    ax.set_title("Three vs seven conditions under shared-window accumulation")
    ax.legend()
    fig.tight_layout()
    return _save_pair(fig, output_dir / "best_teacher_validation")


def _rollout_curve_rows(
    records: Sequence[ModelRecord], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        teacher = results[record.key]["views"]["ordered_teacher_forced"]["summary"][
            "npe_total_by_call"
        ]
        free = results[record.key]["views"]["free_recurrence"]["summary"][
            "npe_total_by_call"
        ]
        teacher_cumulative = 0.0
        free_cumulative = 0.0
        for index, (teacher_value, free_value) in enumerate(
            zip(teacher, free, strict=True)
        ):
            teacher_cumulative += float(teacher_value)
            free_cumulative += float(free_value)
            rows.append(
                {
                    "model": record.label,
                    "architecture": record.architecture,
                    "train_trajectories": record.exposure,
                    "call": index + 1,
                    "teacher_npe": float(teacher_value),
                    "free_npe": float(free_value),
                    "teacher_cumulative_npe": teacher_cumulative,
                    "free_cumulative_npe": free_cumulative,
                }
            )
    return rows


def _frozen_initial_persistence_rows(
    normalized_sequence: torch.Tensor,
    physical_times_s: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Score ``U_hat(t_k) = U(t_0)`` without refreshing the input from truth."""

    if (
        not torch.is_floating_point(normalized_sequence)
        or normalized_sequence.ndim != 4
        or normalized_sequence.shape[0] < 2
        or normalized_sequence.shape[1] != PLANARDET_GROUPS.total_channels
    ):
        raise ValueError("normalized sequence must have float [time>=2, 13, y, x]")
    if not bool(torch.isfinite(normalized_sequence).all()):
        raise ValueError("normalized persistence truth must be finite")
    times = np.asarray(physical_times_s, dtype=np.float64)
    if (
        times.ndim != 1
        or times.shape[0] != normalized_sequence.shape[0]
        or not bool(np.isfinite(times).all())
        or not bool(np.all(np.diff(times) > 0.0))
    ):
        raise ValueError("physical times must be finite, increasing, and match time")

    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    with torch.no_grad():
        initial = normalized_sequence[0:1].unsqueeze(0)
        for call in range(1, normalized_sequence.shape[0]):
            truth = normalized_sequence[call : call + 1].unsqueeze(0)
            metric = grouped_normalized_prediction_error(
                initial,
                truth,
                groups=PLANARDET_GROUPS,
            )
            total = float(metric.total_per_call[0, 0].item())
            cumulative += total
            row: dict[str, Any] = {
                "call": call,
                "physical_time_s": float(times[call]),
                "elapsed_time_s": float(times[call] - times[0]),
                "frozen_u0_npe": total,
                "frozen_u0_cumulative_npe": cumulative,
            }
            for name, values in metric.grouped_per_call.items():
                row[f"{name}_npe"] = float(values[0, 0].item())
            rows.append(row)
    return rows


def _render_rollout_with_frozen_u0(
    records: Sequence[ModelRecord],
    results: Mapping[str, Mapping[str, Any]],
    persistence_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    if not persistence_rows:
        raise ValueError("frozen-initial persistence rows must be nonempty")
    times = np.asarray(
        [row["physical_time_s"] for row in persistence_rows], dtype=np.float64
    )
    baseline = np.asarray(
        [row["frozen_u0_npe"] for row in persistence_rows], dtype=np.float64
    )
    cumulative_baseline = np.cumsum(baseline)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex="col")
    for row_index, exposure in enumerate((7, 3)):
        for record in [item for item in records if item.exposure == exposure]:
            free = np.asarray(
                results[record.key]["views"]["free_recurrence"]["summary"][
                    "npe_total_by_call"
                ],
                dtype=np.float64,
            )
            if free.shape != baseline.shape:
                raise ValueError(
                    f"{record.label} rollout horizon differs from persistence"
                )
            color = ARCHITECTURE_COLORS[record.architecture]
            label = ARCHITECTURE_LABELS[record.architecture]
            axes[row_index, 0].semilogy(times, free, color=color, label=label)
            axes[row_index, 1].semilogy(
                times, np.cumsum(free), color=color, label=label
            )
        axes[row_index, 0].semilogy(
            times,
            baseline,
            color="#202020",
            linestyle="-.",
            linewidth=1.9,
            label=r"Frozen $U_0$",
        )
        axes[row_index, 1].semilogy(
            times,
            cumulative_baseline,
            color="#202020",
            linestyle="-.",
            linewidth=1.9,
            label=r"Frozen $U_0$",
        )
        axes[row_index, 0].set_ylabel(f"{exposure} trajectories\nPer-call NPE")
        axes[row_index, 1].set_ylabel("Cumulative NPE")
        axes[row_index, 0].legend(loc="best")
        axes[row_index, 1].legend(loc="best")
    axes[0, 0].set_title(r"Free rollout vs. frozen $U_0$")
    axes[0, 1].set_title("Cumulative error over the H49 horizon")
    axes[1, 0].set_xlabel("Physical time [s]")
    axes[1, 1].set_xlabel("Physical time [s]")
    fig.tight_layout()
    return _save_pair(fig, output_dir / "rollout_error_with_frozen_u0")


def _render_rollout_curves(
    records: Sequence[ModelRecord],
    results: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex="col")
    for row_index, exposure in enumerate((7, 3)):
        for record in [item for item in records if item.exposure == exposure]:
            teacher = np.asarray(
                results[record.key]["views"]["ordered_teacher_forced"]["summary"][
                    "npe_total_by_call"
                ],
                dtype=np.float64,
            )
            free = np.asarray(
                results[record.key]["views"]["free_recurrence"]["summary"][
                    "npe_total_by_call"
                ],
                dtype=np.float64,
            )
            calls = np.arange(1, len(free) + 1)
            color = ARCHITECTURE_COLORS[record.architecture]
            axes[row_index, 0].semilogy(
                calls, free, color=color, label=ARCHITECTURE_LABELS[record.architecture]
            )
            axes[row_index, 0].semilogy(
                calls, teacher, color=color, linestyle=":", alpha=0.72
            )
            axes[row_index, 1].semilogy(
                calls,
                np.cumsum(free),
                color=color,
                label=ARCHITECTURE_LABELS[record.architecture],
            )
            axes[row_index, 1].semilogy(
                calls, np.cumsum(teacher), color=color, linestyle=":", alpha=0.72
            )
        axes[row_index, 1].axhline(
            PAPER_FFNO_VALIDATION_SUM,
            color="#CC79A7",
            linestyle="--",
            linewidth=1.2,
            label="Paper FFNO val total" if row_index == 0 else None,
        )
        axes[row_index, 0].set_ylabel(f"{exposure} trajectories\nPer-call NPE")
        axes[row_index, 1].set_ylabel("Cumulative NPE")
        axes[row_index, 0].legend(loc="best")
        axes[row_index, 1].legend(loc="best")
    axes[0, 0].set_title("Free rollout (solid) vs truth input (dotted)")
    axes[0, 1].set_title("Released-code horizon aggregation")
    axes[1, 0].set_xlabel("Autoregressive call")
    axes[1, 1].set_xlabel("Autoregressive call")
    fig.tight_layout()
    return _save_pair(fig, output_dir / "rollout_error_accumulation")


def _render_paper_comparison(
    summary_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            int(row["train_trajectories"]) != 7,
            ARCHITECTURES.index(str(row["architecture"])),
        ),
    )
    labels = [str(row["model"]) for row in ordered]
    values = [float(row["free_h49_npe_sum"]) for row in ordered]
    colors = [ARCHITECTURE_COLORS[str(row["architecture"])] for row in ordered]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(
        PAPER_FFNO_VALIDATION_SUM,
        color="#CC79A7",
        linestyle="--",
        label=f"REALM paper FFNO validation = {PAPER_FFNO_VALIDATION_SUM:g}",
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Free H49 released-code NPE sum")
    ax.set_title("Open-validation numerical comparison (contracts differ)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_pair(fig, output_dir / "paper_validation_comparison")


def _finite_quantile(values: Sequence[np.ndarray], quantile: float) -> float:
    parts = [np.abs(value[np.isfinite(value)]).reshape(-1) for value in values]
    parts = [part for part in parts if part.size]
    if not parts:
        raise RuntimeError("visual scale has no finite values")
    joined = np.concatenate(parts)
    result = float(np.quantile(joined, quantile))
    if not math.isfinite(result) or result <= 0.0:
        result = max(float(np.max(joined)), 1.0e-12)
    return result


def _render_state_animation(
    field: str,
    records: Sequence[ModelRecord],
    visuals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    results: Mapping[str, Mapping[str, Any]],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_path: Path,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    ordered = list(records)
    truth = _oriented(visuals[ordered[0].key][field]["truth"], x, y)
    predictions = {
        record.key: _oriented(visuals[record.key][field]["free"], x, y)
        for record in ordered
    }
    state_min = float(np.nanmin(truth))
    state_max = float(np.nanmax(truth))
    if state_max <= state_min:
        state_max = state_min + max(abs(state_min), 1.0) * 1.0e-6
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), constrained_layout=True)
    image_axes = list(axes.flat[:7])
    titles = ["Reference", *[record.label for record in ordered]]
    images = []
    for axis, title in zip(image_axes, titles, strict=True):
        image = axis.imshow(
            np.zeros_like(truth[0]),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=state_min,
            vmax=state_max,
        )
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        images.append(image)
    fig.colorbar(images[0], ax=image_axes, shrink=0.72, label=f"{field} (released units)")
    curve_axis = axes.flat[7]
    calls = np.arange(1, VALIDATION_HORIZON + 1)
    curve_values: dict[str, np.ndarray] = {}
    curve_lines = {}
    for record in ordered:
        values = np.asarray(
            results[record.key]["views"]["free_recurrence"]["summary"][
                "npe_total_by_call"
            ],
            dtype=np.float64,
        )
        curve_values[record.key] = values
        curve_axis.semilogy(
            calls,
            values,
            color=ARCHITECTURE_COLORS[record.architecture],
            linestyle=EXPOSURE_STYLES[record.exposure]["linestyle"],
            alpha=0.22,
        )
        (line,) = curve_axis.semilogy(
            [],
            [],
            color=ARCHITECTURE_COLORS[record.architecture],
            linestyle=EXPOSURE_STYLES[record.exposure]["linestyle"],
            label=record.label,
        )
        curve_lines[record.key] = line
    cursor = curve_axis.axvline(1, color="#333333", linestyle=":")
    curve_axis.set(
        xlim=(1, VALIDATION_HORIZON),
        xlabel="Autoregressive call",
        ylabel="Free NPE",
    )
    curve_axis.legend(ncol=2, fontsize=6.5)
    status = fig.suptitle("")

    def update(frame: int) -> list[Any]:
        images[0].set_data(np.ma.masked_invalid(truth[frame]))
        artists: list[Any] = [images[0]]
        for image, record in zip(images[1:], ordered, strict=True):
            image.set_data(np.ma.masked_invalid(predictions[record.key][frame]))
            artists.append(image)
            if frame:
                curve_lines[record.key].set_data(
                    calls[:frame], curve_values[record.key][:frame]
                )
            else:
                curve_lines[record.key].set_data([], [])
            artists.append(curve_lines[record.key])
        cursor.set_xdata([max(frame, 1), max(frame, 1)])
        status.set_text(
            f"Open validation | free rollout | {field} | call {frame}/{VALIDATION_HORIZON} | "
            f"t={times[frame]:.9g} s | common truth-range scale"
        )
        artists.extend((cursor, status))
        return artists

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=VALIDATION_HORIZON + 1,
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
    saturation = {
        record.label: float(
            np.mean(
                (predictions[record.key] < state_min)
                | (predictions[record.key] > state_max)
            )
        )
        for record in ordered
    }
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "field": field,
        "view": "free_recurrence",
        "frames": VALIDATION_HORIZON + 1,
        "state_min": state_min,
        "state_max": state_max,
        "scale_source": "full-grid released validation truth range",
        "prediction_saturation_fraction": saturation,
    }


def _residual_npe(truth: np.ndarray, error: np.ndarray) -> np.ndarray:
    result = []
    for truth_frame, error_frame in zip(truth, error, strict=True):
        finite = np.isfinite(truth_frame) & np.isfinite(error_frame)
        denominator = float(np.square(truth_frame[finite]).sum(dtype=np.float64))
        numerator = float(np.square(error_frame[finite]).sum(dtype=np.float64))
        result.append(
            numerator / denominator if finite.any() and denominator > 0.0 else math.nan
        )
    return np.asarray(result)


def _truth_input_residuals(
    truth: np.ndarray, teacher_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if truth.shape != teacher_prediction.shape or truth.shape[0] < 2:
        raise ValueError("truth and teacher prediction trajectories must match")
    true_residual = truth[1:] - truth[:-1]
    predicted_residual = teacher_prediction[1:] - truth[:-1]
    prediction_error = predicted_residual - true_residual
    return true_residual, predicted_residual, prediction_error


def _render_residual_animation(
    field: str,
    records: Sequence[ModelRecord],
    visuals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_path: Path,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    ordered = list(records)
    truth = _oriented(visuals[ordered[0].key][field]["truth"], x, y)
    predicted: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    true_residual: np.ndarray | None = None
    for record in ordered:
        record_truth, predicted[record.key], errors[record.key] = (
            _truth_input_residuals(
                truth,
                _oriented(visuals[record.key][field]["teacher"], x, y),
            )
        )
        if true_residual is None:
            true_residual = record_truth
        elif not np.array_equal(record_truth, true_residual, equal_nan=True):
            raise ValueError("models do not share the same validation truth")
    if true_residual is None:
        raise ValueError("residual animation requires at least one model")
    residual_limit = _finite_quantile(
        [true_residual, *predicted.values()], 0.995
    )
    error_limit = _finite_quantile(list(errors.values()), 0.995)
    residual_npe = {
        record.key: _residual_npe(true_residual, errors[record.key])
        for record in ordered
    }
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    fig = plt.figure(figsize=(17.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 7, height_ratios=(1.0, 1.0, 0.72))
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(7)]
    error_axes = [fig.add_subplot(grid[1, index]) for index in range(1, 7)]
    curve_axis = fig.add_subplot(grid[2, :])
    true_image = top_axes[0].imshow(
        np.zeros_like(true_residual[0]),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    top_axes[0].set_title("True residual")
    predicted_images = []
    error_images = []
    for index, record in enumerate(ordered, start=1):
        predicted_image = top_axes[index].imshow(
            np.zeros_like(true_residual[0]),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-residual_limit,
            vmax=residual_limit,
        )
        top_axes[index].set_title(f"{record.label} predicted")
        error_image = error_axes[index - 1].imshow(
            np.zeros_like(true_residual[0]),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )
        error_axes[index - 1].set_title(f"{record.label} error")
        predicted_images.append(predicted_image)
        error_images.append(error_image)
    text_axis = fig.add_subplot(grid[1, 0])
    text_axis.axis("off")
    text_axis.text(
        0.5,
        0.5,
        "Truth-fed one-step view\n\n"
        "Predicted residual:\nÛ(t+1 | U(t)) - U(t)\n\n"
        "Error:\nÛ(t+1 | U(t)) - U(t+1)",
        ha="center",
        va="center",
        fontsize=8,
    )
    for axis in (*top_axes, *error_axes):
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.colorbar(
        true_image,
        ax=top_axes,
        shrink=0.7,
        label=f"{field} one-step residual (released units)",
    )
    fig.colorbar(
        error_images[0],
        ax=error_axes,
        shrink=0.7,
        label=f"{field} one-step prediction error",
    )
    calls = np.arange(1, VALIDATION_HORIZON + 1)
    curve_lines = {}
    for record in ordered:
        curve_axis.semilogy(
            calls,
            residual_npe[record.key],
            color=ARCHITECTURE_COLORS[record.architecture],
            linestyle=EXPOSURE_STYLES[record.exposure]["linestyle"],
            alpha=0.22,
        )
        (line,) = curve_axis.semilogy(
            [],
            [],
            color=ARCHITECTURE_COLORS[record.architecture],
            linestyle=EXPOSURE_STYLES[record.exposure]["linestyle"],
            label=record.label,
        )
        curve_lines[record.key] = line
    cursor = curve_axis.axvline(1, color="#333333", linestyle=":")
    curve_axis.set(
        xlim=(1, VALIDATION_HORIZON),
        xlabel="Truth-input pair / call",
        ylabel="One-step residual NPE",
    )
    curve_axis.legend(ncol=6, loc="upper center")
    status = fig.suptitle("")

    def update(frame: int) -> list[Any]:
        true_image.set_data(np.ma.masked_invalid(true_residual[frame]))
        artists: list[Any] = [true_image]
        for predicted_image, error_image, record in zip(
            predicted_images, error_images, ordered, strict=True
        ):
            predicted_image.set_data(np.ma.masked_invalid(predicted[record.key][frame]))
            error_image.set_data(np.ma.masked_invalid(errors[record.key][frame]))
            curve_lines[record.key].set_data(
                calls[: frame + 1], residual_npe[record.key][: frame + 1]
            )
            artists.extend((predicted_image, error_image, curve_lines[record.key]))
        cursor.set_xdata([frame + 1, frame + 1])
        status.set_text(
            f"Open validation | {field} | truth-input pair {frame + 1}/{VALIDATION_HORIZON} | "
            f"t={times[frame + 1]:.9g} s | no accumulated prediction is used"
        )
        artists.extend((cursor, status))
        return artists

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=VALIDATION_HORIZON,
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
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": output_path.stat().st_size,
        "field": field,
        "view": "ordered_teacher_forced_one_step_residual",
        "conditioning": "released truth U(t) restored before every prediction",
        "true_residual_definition": "U(t+1)-U(t)",
        "predicted_residual_definition": "U_hat(t+1|U(t))-U(t)",
        "prediction_error_definition": "predicted_residual-true_residual",
        "accumulated_rollout_error_visualized": False,
        "frames": VALIDATION_HORIZON,
        "residual_abs_q99_5_common_scale": residual_limit,
        "prediction_error_abs_q99_5_common_scale": error_limit,
        "scale_source": "pooled finite values across all six models on visualization grid",
    }


def _render_final_snapshot(
    field: str,
    records: Sequence[ModelRecord],
    visuals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    ordered = list(records)
    truth = _oriented(visuals[ordered[0].key][field]["truth"], x, y)
    predictions = {
        record.key: _oriented(visuals[record.key][field]["free"], x, y)
        for record in ordered
    }
    state_min = float(np.nanmin(truth))
    state_max = float(np.nanmax(truth))
    extent = (
        float(np.min(x)) * 1.0e3,
        float(np.max(x)) * 1.0e3,
        float(np.min(y)) * 1.0e3,
        float(np.max(y)) * 1.0e3,
    )
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.0), constrained_layout=True)
    titles = ["Reference", *[record.label for record in ordered], ""]
    values = [truth[-1], *[predictions[record.key][-1] for record in ordered], None]
    image = None
    for axis, title, value in zip(axes.flat, titles, values, strict=True):
        if value is None:
            axis.axis("off")
            continue
        image = axis.imshow(
            np.ma.masked_invalid(value),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=state_min,
            vmax=state_max,
        )
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, label=f"{field} (released units)")
    fig.suptitle(f"Open-validation free rollout at call {VALIDATION_HORIZON}")
    return _save_pair(fig, output_dir / f"rollout_final_{field}")


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    fields = _validate_fields(args.animation_fields)
    if args.spatial_stride < 1 or args.fps < 1 or args.animation_dpi < 40:
        raise ValueError("visualization stride/fps/dpi arguments are invalid")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("visualization output directory must be absent")

    records = _model_records(args)
    histories = {record.key: _history_rows(record) for record in records}
    results = {record.key: _validate_evaluation(record.evaluation_dir) for record in records}
    for record in records:
        result = results[record.key]
        if record.key != "pcno_n7" and (
            result.get("architecture") != record.architecture
            or result.get("train_trajectory_count") != record.exposure
            or result.get("training_attempt", {}).get("complete")
            is not record.training_complete
        ):
            raise ValueError(f"{record.label} evaluation identity differs")
    args.output_dir.mkdir(parents=True)

    training_rows = _training_table(records, histories)
    summary_rows = _summary_rows(records, histories, results)
    rollout_rows = _rollout_curve_rows(records, results)
    training_csv = args.output_dir / "training_validation_curves.csv"
    summary_csv = args.output_dir / "architecture_exposure_summary.csv"
    rollout_csv = args.output_dir / "rollout_error_accumulation.csv"
    _write_csv(training_csv, training_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(rollout_csv, rollout_rows)

    figure_paths: list[Path] = []
    figure_paths.extend(_render_loss_curves(records, histories, args.output_dir))
    figure_paths.extend(_render_best_teacher(summary_rows, args.output_dir))
    figure_paths.extend(_render_rollout_curves(records, results, args.output_dir))
    figure_paths.extend(_render_paper_comparison(summary_rows, args.output_dir))

    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    expected_normalizer_sha256 = _load_json(
        next(record for record in records if record.key != "pcno_n7").training_dir
        / "input_manifest.json"
    )["normalizer_arrays_sha256"]
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=expected_normalizer_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    normalizer = bundle.normalizer(1, dtype=torch.float32)
    validation_normalized, validation_native_batched = load_normalized_trajectories(
        args.data_root,
        split="val",
        groups=PLANARDET_VAL_GROUPS,
        normalizer=normalizer,
        retain_native=not args.skip_animations,
    )
    times = np.asarray(metadata.times)
    if validation_normalized.shape[1] != VALIDATION_HORIZON + 1:
        raise ValueError("validation trajectory differs from the H49 contract")
    persistence_rows = _frozen_initial_persistence_rows(
        validation_normalized[0], times[: VALIDATION_HORIZON + 1]
    )
    persistence_csv = args.output_dir / "frozen_initial_persistence.csv"
    _write_csv(persistence_csv, persistence_rows)
    figure_paths.extend(
        _render_rollout_with_frozen_u0(
            records, results, persistence_rows, args.output_dir
        )
    )
    persistence_group_means = {
        name: float(np.mean([row[f"{name}_npe"] for row in persistence_rows]))
        for name in PLANARDET_GROUPS.slices()
    }
    persistence_mean = float(
        np.mean([row["frozen_u0_npe"] for row in persistence_rows])
    )
    persistence_sum = float(persistence_rows[-1]["frozen_u0_cumulative_npe"])
    del validation_normalized

    visuals: dict[str, Mapping[str, Mapping[str, np.ndarray]]] = {}
    if not args.skip_animations:
        if validation_native_batched is None:
            raise RuntimeError("validation native truth was not retained")
        truth_native = validation_native_batched[0].numpy()
        for record in records:
            teacher = np.load(
                record.evaluation_dir / "teacher_prediction_normalized.npy",
                allow_pickle=False,
            )
            free = np.load(
                record.evaluation_dir / "free_prediction_normalized.npy",
                allow_pickle=False,
            )
            decoded, _teacher_events, _free_events, _scales = (
                _decode_visualization_bundle(
                    teacher,
                    free,
                    truth_native,
                    bundle=bundle,
                    fields=fields,
                    spatial_stride=args.spatial_stride,
                )
            )
            visuals[record.key] = decoded

    x = np.asarray(metadata.x)[:: args.spatial_stride]
    y = np.asarray(metadata.y)[:: args.spatial_stride]
    animation_records: list[dict[str, Any]] = []
    for field in (() if args.skip_animations else fields):
        state_path = args.output_dir / f"rollout_state_{field}.gif"
        residual_path = args.output_dir / f"one_step_residual_{field}.gif"
        animation_records.append(
            _render_state_animation(
                field,
                records,
                visuals,
                results,
                times,
                x,
                y,
                output_path=state_path,
                fps=args.fps,
                dpi=args.animation_dpi,
            )
        )
        animation_records.append(
            _render_residual_animation(
                field,
                records,
                visuals,
                times,
                x,
                y,
                output_path=residual_path,
                fps=args.fps,
                dpi=args.animation_dpi,
            )
        )
        figure_paths.extend(
            _render_final_snapshot(field, records, visuals, x, y, args.output_dir)
        )

    summary_payload: dict[str, Any] = {
        "schema": "w26_l4_d093_architecture_exposure_visualization_v3",
        "models": summary_rows,
        "sampler_interpretation": {
            "micro_presentations_per_optimizer_step": (
                PRESENTATIONS_PER_OPTIMIZER_STEP
            ),
            "one_shared_temporal_window_per_optimizer_step": True,
            "n7_condition_window_terms_per_step": 7,
            "n7_duplicate_condition_window_terms_per_step": 0,
            "n3_condition_window_terms_per_step": 3,
            "n3_duplicate_condition_window_terms_per_step": 4,
            "two_call_window_count": TWO_CALL_WINDOW_COUNT,
            "raw_micro_presentations_are_distinct_exposure": False,
            "interpretation": (
                "the n=3 arm reweights three condition-window gradients inside one "
                "accumulated update; it does not make 7/3 successive updated-model "
                "passes per active trajectory"
            ),
        },
        "frozen_initial_persistence": {
            "definition": "U_hat(t_k) = U(t_0) for every one of 49 target times",
            "space": "Box-Cox-transformed and train-z-score-normalized state",
            "metric": "sum of five channel-group MSE values at each target time",
            "realm_npe_mean": persistence_mean,
            "realm_npe_sum_source": persistence_sum,
            "group_mse_mean": persistence_group_means,
            "per_call_table": persistence_csv.name,
            "distinct_from_truth_input_persistence": (
                "the prediction is never refreshed from U(t_{k-1})"
            ),
        },
        "paper_reference": {
            "scenario": "PlanarDet",
            "architecture": "FFNO",
            "validation_error": PAPER_FFNO_VALIDATION_SUM,
            "test_error": PAPER_FFNO_TEST_SUM,
            "test_correlation": PAPER_FFNO_TEST_CORRELATION,
            "direct_comparison_allowed": False,
            "reason": (
                "our FFNO uses residual parameterization/common D093 training and "
                "a coordinate normalization that is not paper-faithful"
            ),
        },
        "normalizer_arrays_sha256": expected_normalizer_sha256,
        "validation_group": PLANARDET_VAL_GROUPS[0],
        "spatial_stride_visualization_only": args.spatial_stride,
        "full_grid_metrics": True,
        "animation_fields": [] if args.skip_animations else list(fields),
        "animations_skipped": bool(args.skip_animations),
        "animations": animation_records,
        "figures": [
            {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in figure_paths
        ],
        "tables": [
            {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in (training_csv, summary_csv, rollout_csv, persistence_csv)
        ],
        "test_object_opened": False,
        "anti_claims": [
            "one seed and one open validation trajectory do not establish architecture superiority",
            "training objective and truth-input validation NPE differ after the one-call phase",
            "interrupted runs retain useful checkpoints but are not completed training runs",
            "paper validation values are numerical references under a different FFNO contract",
            "raw micro-presentation counts are not distinct trajectory-window exposure",
        ],
    }
    summary_payload["canonical_payload_sha256"] = canonical_json_sha256(summary_payload)
    summary_path = args.output_dir / "visualization_summary.json"
    _write_json(summary_path, summary_payload)
    retained = [
        *figure_paths,
        *[args.output_dir / record["path"] for record in animation_records],
        training_csv,
        summary_csv,
        rollout_csv,
        persistence_csv,
        summary_path,
    ]
    final_manifest = {
        "schema": "w26_l4_d093_architecture_exposure_visualization_final_manifest_v3",
        "files": {
            path.name: sha256_file(path) for path in sorted(retained, key=lambda item: item.name)
        },
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_visualization(args)
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "model_count": len(summary["models"]),
                "animation_count": len(summary["animations"]),
                "test_object_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
