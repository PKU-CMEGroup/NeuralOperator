#!/usr/bin/env python3
"""Render checkpoint-free W26-L2-P2-W0 figures from JSON/NPZ outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("full_native", "full_gradient_scale_0p1", "no_gradient")
COLORS = {
    "full_native": "#0072B2",
    "full_gradient_scale_0p1": "#009E73",
    "no_gradient": "#D55E00",
}
LABELS = {
    "full_native": "Full",
    "full_gradient_scale_0p1": "Full, grad x0.1",
    "no_gradient": "No gradient",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf = figure_dir / f"{stem}.pdf"
    png = figure_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"Creator": "W26-L2-P2-W0", "CreationDate": None})
    fig.savefig(png, dpi=300, metadata={"Software": "W26-L2-P2-W0"})
    plt.close(fig)
    return [pdf, png]


def _phase_key(phase: float) -> str:
    return str(float(phase)).replace(".", "p")


def _row(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    family: str,
    phase: float,
    call: int,
    path: str,
) -> Mapping[str, Any]:
    selected = [
        item
        for item in rows
        if item["arm"] == arm
        and item["seed"] == seed
        and item["family"] == family
        and item["phase"] == phase
        and item["call"] == call
        and item["path"] == path
    ]
    if len(selected) != 1:
        raise ValueError("expected exactly one metric row for figure selection")
    return selected[0]


def _shade_masks(
    ax: plt.Axes,
    x: np.ndarray,
    active: np.ndarray,
    wake: np.ndarray,
) -> None:
    ax.fill_between(
        x,
        0,
        1,
        where=wake,
        color="#CC79A7",
        alpha=0.10,
        transform=ax.get_xaxis_transform(),
        linewidth=0,
    )
    ax.fill_between(
        x,
        0,
        1,
        where=active,
        color="#F0E442",
        alpha=0.13,
        transform=ax.get_xaxis_transform(),
        linewidth=0,
    )


def _residual_profiles(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
) -> list[Path]:
    x = arrays["x_centers"]
    phase = 0.875
    call = 2
    seed = 1701
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=True)
    for column, family in enumerate(("step", "pulse")):
        case_prefix = f"case__{family}__p{_phase_key(phase)}__call{call}"
        true_residual = np.mean(arrays[f"{case_prefix}__true_residual"], axis=0)
        active = arrays[f"{case_prefix}__mask_active"].astype(bool)
        wake = arrays[f"{case_prefix}__mask_wake"].astype(bool)
        representative = _row(
            rows,
            arm="full_native",
            seed=seed,
            family=family,
            phase=phase,
            call=call,
            path="recurrent",
        )
        fronts = representative["increment_front_positions"]
        for row_ax in axes[:, column]:
            _shade_masks(row_ax, x, active, wake)
            for index, position in enumerate(fronts):
                row_ax.axvline(
                    position,
                    color="0.35",
                    linestyle="--" if index % 2 == 0 else ":",
                    linewidth=0.8,
                    alpha=0.75,
                )
        axes[0, column].plot(x, true_residual, color="black", linewidth=1.5, label="True residual")
        for arm in ARMS:
            prefix = f"{arm}__s{seed}__{family}__p{_phase_key(phase)}__call{call}"
            axes[0, column].plot(
                x,
                np.mean(arrays[f"{prefix}__recurrent_residual"], axis=0),
                color=COLORS[arm],
                linewidth=1.0,
                label=LABELS[arm],
            )
            axes[1, column].plot(
                x,
                np.mean(arrays[f"{prefix}__recurrent_error"], axis=0),
                color=COLORS[arm],
                linewidth=1.0,
                label=LABELS[arm],
            )
        full_prefix = f"full_native__s{seed}__{family}__p{_phase_key(phase)}__call{call}"
        axes[2, column].plot(
            x,
            np.mean(arrays[f"{full_prefix}__teacher_error"], axis=0),
            color="#56B4E9",
            linewidth=1.1,
            label="Fresh error",
        )
        axes[2, column].plot(
            x,
            np.mean(arrays[f"{full_prefix}__propagated_contribution"], axis=0),
            color="#CC79A7",
            linewidth=1.1,
            label="Propagated contribution",
        )
        axes[2, column].plot(
            x,
            np.mean(arrays[f"{full_prefix}__recurrent_error"], axis=0),
            color="black",
            linewidth=0.9,
            linestyle="--",
            label="Total recurrent error",
        )
        axes[0, column].set_title("Translated step" if family == "step" else "Translated pulse")
        axes[2, column].set_xlabel("x")
        axes[1, column].set_yscale("symlog", linthresh=1.0e-3, linscale=0.7)
    axes[0, 0].set_ylabel("Residual")
    axes[1, 0].set_ylabel("Residual error")
    axes[2, 0].set_ylabel("Error component")
    axes[0, 0].legend(ncol=2, fontsize=7, loc="best")
    axes[2, 0].legend(ncol=1, fontsize=7, loc="best")
    fig.text(0.5, 0.995, "Yellow: active exact residual; magenta: passed-front wake", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, figure_dir, "fig_residual_truth_prediction_error")


def _spacetime_error(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
) -> list[Path]:
    x = arrays["x_centers"]
    phase = 0.875
    seed = 1701
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex=True, sharey=True)
    matrices = []
    for family in ("step", "pulse"):
        for arm in ARMS:
            matrices.append(
                np.stack(
                    [
                        np.mean(
                            arrays[
                                f"{arm}__s{seed}__{family}__p{_phase_key(phase)}__call{call}__recurrent_error"
                            ],
                            axis=0,
                        )
                        for call in range(3)
                    ]
                )
            )
    limit = max(float(np.max(np.abs(value))) for value in matrices)
    index = 0
    image = None
    for row_index, family in enumerate(("step", "pulse")):
        for column, arm in enumerate(ARMS):
            matrix = matrices[index]
            index += 1
            image = axes[row_index, column].imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=(x[0], x[-1], -0.5, 2.5),
                cmap="RdBu_r",
                norm=SymLogNorm(
                    linthresh=1.0e-3,
                    linscale=0.7,
                    vmin=-limit,
                    vmax=limit,
                    base=10,
                ),
                interpolation="nearest",
            )
            for front_index in range(4 if family == "pulse" else 2):
                positions = []
                for call in range(3):
                    record = _row(
                        rows,
                        arm=arm,
                        seed=seed,
                        family=family,
                        phase=phase,
                        call=call,
                        path="recurrent",
                    )
                    fronts = record["increment_front_positions"]
                    positions.append(fronts[front_index])
                axes[row_index, column].plot(
                    positions,
                    np.arange(3),
                    color="black",
                    linewidth=0.7,
                    linestyle="--" if front_index % 2 == 0 else ":",
                )
            axes[row_index, column].set_yticks((0, 1, 2))
            axes[row_index, column].set_xlabel("x")
            if row_index == 0:
                axes[row_index, column].set_title(LABELS[arm])
        axes[row_index, 0].set_ylabel(("Step" if family == "step" else "Pulse") + " call")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Recurrent residual error", shrink=0.82)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.12, top=0.91, wspace=0.16, hspace=0.24)
    return _save(fig, figure_dir, "fig_spacetime_residual_error")


def _region_curves(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    metrics = (("normalized_l1", "Wake normalized |error|"), ("oscillatory_mass", "Wake oscillatory mass"))
    for column, family in enumerate(("step", "pulse")):
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            for arm in ARMS:
                means = []
                stds = []
                for call in (1, 2):
                    values = [
                        row["metrics"]["regions"]["wake"][metric]
                        for row in rows
                        if row["arm"] == arm
                        and row["family"] == family
                        and row["path"] == "recurrent"
                        and row["call"] == call
                        and row["metrics"]["regions"]["wake"]["valid"]
                    ]
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values, ddof=1)))
                    ax.scatter(
                        np.full(len(values), call) + np.linspace(-0.025, 0.025, len(values)),
                        values,
                        color=COLORS[arm],
                        s=7,
                        alpha=0.25,
                        linewidths=0,
                    )
                ax.errorbar(
                    (1, 2),
                    means,
                    yerr=stds,
                    color=COLORS[arm],
                    marker="o",
                    markersize=3,
                    capsize=2,
                    linewidth=1.1,
                    label=LABELS[arm],
                )
            ax.set_xticks((1, 2))
            ax.set_xlabel("Call")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title("Translated step" if family == "step" else "Translated pulse")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, figure_dir, "fig_wake_region_metrics")


def _smooth_controls(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    x = np.arange(len(ARMS), dtype=float)
    width = 0.18
    for ax, family in zip(axes, ("smooth_tanh", "smooth_sine"), strict=True):
        seed_values = {}
        for arm_index, arm in enumerate(ARMS):
            per_seed = []
            for seed in (1701, 1702, 1703):
                values = [
                    row["metrics"]["relative_increment_l2"]
                    for row in rows
                    if row["arm"] == arm
                    and row["seed"] == seed
                    and row["family"] == family
                    and row["path"] == "teacher"
                ]
                per_seed.append(float(np.mean(values)))
            seed_values[arm] = per_seed
            ax.bar(
                arm_index,
                np.mean(per_seed),
                yerr=np.std(per_seed, ddof=1),
                width=0.62,
                color=COLORS[arm],
                alpha=0.72,
                capsize=2,
            )
            ax.scatter(
                arm_index + np.linspace(-width, width, len(per_seed)),
                per_seed,
                color="black",
                s=10,
                zorder=3,
            )
        ax.set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=18, ha="right")
        ax.set_ylabel("Teacher-forced relative residual L2")
        ax.set_yscale("log")
        ax.set_title("Steep tanh" if family == "smooth_tanh" else "Smooth sine")
    fig.tight_layout()
    return _save(fig, figure_dir, "fig_smooth_control_comparison")


def _decomposition(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    x = np.arange(len(ARMS), dtype=float)
    for ax, family in zip(axes, ("step", "pulse"), strict=True):
        for arm_index, arm in enumerate(ARMS):
            shares = [
                row["decomposition"]["wake"]["propagated_component_energy_share"]
                for row in rows
                if row["arm"] == arm
                and row["family"] == family
                and row["path"] == "recurrent"
                and row["call"] == 2
                and row["decomposition"]["wake"]["valid"]
            ]
            mean_share = float(np.mean(shares))
            ax.bar(
                arm_index,
                1.0 - mean_share,
                width=0.62,
                color="#56B4E9",
                alpha=0.82,
                label="Fresh component" if arm_index == 0 else None,
            )
            ax.bar(
                arm_index,
                mean_share,
                bottom=1.0 - mean_share,
                width=0.62,
                color="#CC79A7",
                alpha=0.82,
                label="Propagated component" if arm_index == 0 else None,
            )
            ax.scatter(
                arm_index + np.linspace(-0.18, 0.18, len(shares)),
                shares,
                color="black",
                s=8,
                alpha=0.45,
                zorder=3,
            )
        ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Translated step" if family == "step" else "Translated pulse")
        ax.set_ylabel("Component-energy share at call 2")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return _save(fig, figure_dir, "fig_fresh_propagated_decomposition")


def _phase_summary(
    rows: Sequence[Mapping[str, Any]], figure_dir: Path
) -> list[Path]:
    """Separate trained and held phase instead of averaging their error scales."""

    trained_arms = ("full_native", "no_gradient")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    metrics = (
        ("relative_increment_l2", "Call-2 recurrent relative L2"),
        ("wake_oscillatory_mass", "Call-2 wake oscillatory mass"),
    )
    for column, family in enumerate(("step", "pulse")):
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            for arm in trained_arms:
                means = []
                stds = []
                for phase in (0.0, 0.875):
                    selected = [
                        row
                        for row in rows
                        if row["arm"] == arm
                        and row["family"] == family
                        and row["phase"] == phase
                        and row["path"] == "recurrent"
                        and row["call"] == 2
                    ]
                    if metric == "relative_increment_l2":
                        values = [row["metrics"][metric] for row in selected]
                    else:
                        values = [
                            row["metrics"]["regions"]["wake"]["oscillatory_mass"]
                            for row in selected
                        ]
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values, ddof=1)))
                    ax.scatter(
                        np.full(len(values), phase)
                        + np.linspace(-0.012, 0.012, len(values)),
                        values,
                        color=COLORS[arm],
                        s=9,
                        alpha=0.35,
                        linewidths=0,
                    )
                ax.errorbar(
                    (0.0, 0.875),
                    means,
                    yerr=stds,
                    color=COLORS[arm],
                    marker="o",
                    markersize=3,
                    capsize=2,
                    linewidth=1.1,
                    label=LABELS[arm],
                )
            ax.set_yscale("log")
            ax.set_xticks((0.0, 0.875), ("trained 0", "held 0.875"))
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title("Translated step" if family == "step" else "Translated pulse")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, figure_dir, "fig_trained_vs_held_phase")


def render_figures(
    output_dir: Path, *, figure_dir: Path | None = None
) -> list[Path]:
    output = Path(output_dir).resolve()
    rows = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _style()
    target_figure_dir = (
        Path(figure_dir).resolve() if figure_dir is not None else output / "figures"
    )
    paths: list[Path] = []
    paths.extend(_residual_profiles(arrays, rows, target_figure_dir))
    paths.extend(_spacetime_error(arrays, rows, target_figure_dir))
    paths.extend(_region_curves(rows, target_figure_dir))
    paths.extend(_smooth_controls(rows, target_figure_dir))
    paths.extend(_decomposition(rows, target_figure_dir))
    paths.extend(_phase_summary(rows, target_figure_dir))
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = render_figures(args.output_dir, figure_dir=args.figure_dir)
    print(json.dumps([str(path) for path in paths], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
