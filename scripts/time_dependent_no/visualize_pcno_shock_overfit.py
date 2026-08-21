#!/usr/bin/env python3
"""Generate registered W26-L2-P0 overfit convergence and profile figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    FRONT_BAND_WIDTH,
    PULSE_WIDTH,
)

SCHEMA = "w26_l2_p0_full_pcno_overfit_v1"
VISUAL_SCHEMA = "w26_l2_p0_full_pcno_visuals_v1"
FULL_WIDTH = (6.75, 2.8)
PROFILE_SIZE = (6.75, 5.0)
COLORS = {
    "full": "#0072B2",
    "held": "#E69F00",
    "error": "#D55E00",
    "target": "#222222",
    "range": "#56B4E9",
    "gate": "#009E73",
    "band": "#F0E442",
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
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(
        pdf,
        metadata={"Creator": "W26-L2 PCNO visualizer", "CreationDate": None},
    )
    fig.savefig(
        png,
        dpi=300,
        metadata={"Software": "W26-L2 PCNO visualizer"},
    )
    plt.close(fig)
    return [pdf, png]


def _convergence_figure(history: list[dict[str, Any]]) -> plt.Figure:
    steps = np.asarray([row["step"] for row in history], dtype=np.int64)
    train_mean = np.asarray(
        [row["train"]["mean_relative_increment_l2"] for row in history]
    )
    held_mean = np.asarray(
        [row["held"]["mean_relative_increment_l2"] for row in history]
    )
    train_maximum = np.asarray(
        [row["train"]["maximum_relative_increment_l2"] for row in history]
    )
    held_maximum = np.asarray(
        [row["held"]["maximum_relative_increment_l2"] for row in history]
    )
    positive_floor = np.finfo(np.float64).tiny
    fig, axes = plt.subplots(1, 2, figsize=FULL_WIDTH, constrained_layout=True)
    for axis, train, held, title in (
        (axes[0], train_mean, held_mean, "Mean case error"),
        (axes[1], train_maximum, held_maximum, "Worst case error"),
    ):
        axis.semilogy(
            steps,
            np.maximum(train, positive_floor),
            color=COLORS["full"],
            marker="o",
            markevery=max(1, len(steps) // 8),
            markersize=3.5,
            label="Train phases",
        )
        axis.semilogy(
            steps,
            np.maximum(held, positive_floor),
            color=COLORS["held"],
            linestyle="--",
            marker="s",
            markevery=max(1, len(steps) // 8),
            markersize=3.2,
            label="Held phases",
        )
        axis.axhline(
            1.0e-3,
            color=COLORS["gate"],
            linewidth=1.1,
            linestyle=":",
            label="Train L2 guard",
        )
        axis.set_title(title)
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Relative increment L2")
    axes[0].legend(loc="best")
    fig.suptitle("Full PCNO fixed-grid overfit", fontsize=11, fontweight="bold")
    return fig


def _choose_profile_indices(payload: dict[str, np.ndarray]) -> list[int]:
    selected: list[int] = []
    specifications = (
        ("train", "step", 0.5),
        ("held", "step", 0.375),
        ("train", "pulse", 0.5),
        ("held", "pulse", 0.375),
    )
    for split, family, desired_phase in specifications:
        candidates = np.flatnonzero(
            (payload["splits"] == split) & (payload["families"] == family)
        )
        if not candidates.size:
            raise ValueError(f"no {split} {family} profile is available")
        scores = np.abs(
            payload["phases"][candidates] - desired_phase
        ) + 1.0e-3 * np.abs(payload["anchor_indices"][candidates] - 1)
        selected.append(int(candidates[int(np.argmin(scores))]))
    return selected


def _fronts(family: str, position: float) -> tuple[float, ...]:
    if family == "step":
        return (position + DISPLACEMENT,)
    if family == "pulse":
        return (
            position + DISPLACEMENT,
            position + DISPLACEMENT + PULSE_WIDTH,
        )
    return ()


def _profiles_figure(
    payload: dict[str, np.ndarray], indices: list[int]
) -> tuple[plt.Figure, dict[str, float]]:
    predicted_next = payload["current"] + payload["predicted_increment"]
    target_next = payload["target_next"]
    selected_prediction = predicted_next[indices]
    selected_target = target_next[indices]
    state_min = float(min(np.min(selected_prediction), np.min(selected_target)))
    state_max = float(max(np.max(selected_prediction), np.max(selected_target)))
    state_margin = max(0.04, 0.05 * (state_max - state_min))
    state_limits = (state_min - state_margin, state_max + state_margin)
    selected_errors = (
        payload["predicted_increment"][indices] - payload["target_increment"][indices]
    )
    error_limit = max(
        1.0e-6,
        float(np.max(np.abs(np.mean(selected_errors, axis=1)))),
    )
    nx = selected_prediction.shape[-1]
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    fig, axes = plt.subplots(2, 2, figsize=PROFILE_SIZE, constrained_layout=True)
    secondary_axes: list[plt.Axes] = []
    for axis, index in zip(axes.reshape(-1), indices, strict=True):
        target_profile = np.mean(target_next[index], axis=0)
        prediction_profile = np.mean(predicted_next[index], axis=0)
        prediction_min = np.min(predicted_next[index], axis=0)
        prediction_max = np.max(predicted_next[index], axis=0)
        error_profile = np.mean(
            payload["predicted_increment"][index] - payload["target_increment"][index],
            axis=0,
        )
        axis.fill_between(
            x,
            prediction_min,
            prediction_max,
            color=COLORS["range"],
            alpha=0.2,
            linewidth=0,
            label="Pred. y-range",
        )
        axis.plot(
            x,
            target_profile,
            color=COLORS["target"],
            linestyle="--",
            label="Exact next",
        )
        axis.plot(x, prediction_profile, color=COLORS["full"], label="PCNO next")
        for front in _fronts(
            str(payload["families"][index]), float(payload["positions"][index])
        ):
            axis.axvspan(
                front - FRONT_BAND_WIDTH,
                front + FRONT_BAND_WIDTH,
                color=COLORS["band"],
                alpha=0.12,
                linewidth=0,
            )
        error_axis = axis.twinx()
        error_axis.plot(
            x,
            error_profile,
            color=COLORS["error"],
            linestyle=":",
            linewidth=1.2,
            label="Increment error",
        )
        error_axis.axhline(0.0, color="#777777", linewidth=0.6)
        error_axis.set_ylim(-1.05 * error_limit, 1.05 * error_limit)
        error_axis.spines["top"].set_visible(False)
        error_axis.grid(False)
        secondary_axes.append(error_axis)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(*state_limits)
        axis.set_title(
            f"{str(payload['splits'][index]).title()} "
            f"{payload['families'][index]!s}, phase={payload['phases'][index]:g}"
        )
        axis.set_xlabel("Physical x")
        axis.set_ylabel("Next-state cell average")
    secondary_axes[1].set_ylabel("Increment error")
    secondary_axes[3].set_ylabel("Increment error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    error_handles, error_labels = secondary_axes[0].get_legend_handles_labels()
    fig.legend(
        handles + error_handles,
        labels + error_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        "Fixed-grid translated-front profiles (common scales)",
        fontsize=11,
        fontweight="bold",
    )
    return fig, {
        "state_min": state_limits[0],
        "state_max": state_limits[1],
        "symmetric_increment_error_limit": 1.05 * error_limit,
    }


def generate_visualizations(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    required = {
        "history": run_dir / "history.json",
        "case_metrics": run_dir / "case_metrics.json",
        "predictions": run_dir / "predictions.npz",
        "run_contract": run_dir / "run_contract.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing visualization input {name}: {path}")
    contract = json.loads(required["run_contract"].read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ValueError("run contract schema does not match the W26-L2 visualizer")
    history = json.loads(required["history"].read_text(encoding="utf-8"))
    with np.load(required["predictions"], allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    _style()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    output_paths.extend(
        _save_figure(
            _convergence_figure(history),
            figures_dir / "fig_overfit_convergence",
        )
    )
    selected = _choose_profile_indices(payload)
    profile_figure, profile_scales = _profiles_figure(payload, selected)
    output_paths.extend(
        _save_figure(profile_figure, figures_dir / "fig_overfit_profiles")
    )
    manifest = {
        "schema": VISUAL_SCHEMA,
        "run_schema": SCHEMA,
        "working_run_id": contract["working_run_id"],
        "input_hashes": {name: _sha256(path) for name, path in required.items()},
        "output_hashes": {
            path.relative_to(run_dir).as_posix(): _sha256(path) for path in output_paths
        },
        "plotted_case_ids": [str(payload["case_ids"][index]) for index in selected],
        "profile_scales": profile_scales,
        "style": {
            "palette": "Okabe-Ito",
            "pdf_vector": True,
            "png_dpi": 300,
            "common_profile_scales": True,
            "front_band_width": FRONT_BAND_WIDTH,
        },
    }
    visual_manifest = figures_dir / "visual_manifest.json"
    _write_json(visual_manifest, manifest)
    return {
        "manifest": visual_manifest.relative_to(run_dir).as_posix(),
        "files": [path.relative_to(run_dir).as_posix() for path in output_paths],
        "plotted_case_ids": manifest["plotted_case_ids"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_visualizations(args.run_dir)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
