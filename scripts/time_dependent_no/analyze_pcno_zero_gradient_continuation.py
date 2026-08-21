#!/usr/bin/env python3
"""Verify, compare, and visualize the W26-L2-P2-C0 continuation matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_gradient_ablation import (
    _aggregate_metrics,
    _paired_control_failures,
    _primary_rows,
)
from scripts.time_dependent_no.continue_pcno_zero_gradient import (
    ARMS,
    BRANCH_MASKS,
    SEEDS,
    branch_mask_name,
)
from scripts.time_dependent_no.continue_pcno_zero_gradient import (
    SCHEMA as RUN_SCHEMA,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    sha256_file,
    write_json,
)
from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    FRONT_BAND_WIDTH,
    PULSE_WIDTH,
    SMOOTH_FILTER_PASS_WAVENUMBER,
    SMOOTH_FILTER_STOP_WAVENUMBER,
    build_structured_cell_grid,
    physical_cosine_spectrum,
)

SCHEMA = "w26_l2_p2_c0_zero_gradient_continuation_analysis_v1"
DEFAULT_MATRIX = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2_c0"
DEFAULT_OUTPUT = DEFAULT_MATRIX / "analysis"
ANALYSIS_SOURCE_PATHS = (
    "scripts/time_dependent_no/analyze_pcno_zero_gradient_continuation.py",
    "scripts/time_dependent_no/analyze_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
)
COLORS = {
    "continue_no_gradient": "#D55E00",
    "activate_zero_gradient": "#0072B2",
}
LABELS = {
    "continue_no_gradient": "Continue no gradient",
    "activate_zero_gradient": "Activate zero-output gradient",
}
LINESTYLES = {"continue_no_gradient": "--", "activate_zero_gradient": "-"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in environment.items() if key != "pid"}


def _verify_run(run_dir: Path, arm: str, seed: int, *, smoke: bool) -> dict[str, Any]:
    required = (
        "manifest.json",
        "run_contract.json",
        "summary.json",
        "history.json",
        "case_metrics.json",
        "nested_case_metrics.json",
        "predictions.npz",
        "branch_cube_metrics.json",
        "branch_cube_predictions.npz",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name} is missing {missing}")
    manifest = _read_json(run_dir / "manifest.json")
    contract = _read_json(run_dir / "run_contract.json")
    summary = _read_json(run_dir / "summary.json")
    if any(
        value.get("schema") != RUN_SCHEMA for value in (manifest, contract, summary)
    ):
        raise ValueError(f"{run_dir.name} has the wrong schema")
    if any(value.get("status") != "completed" for value in (manifest, summary)):
        raise ValueError(f"{run_dir.name} is incomplete")
    if bool(summary["science_result"]) == smoke:
        raise ValueError(f"{run_dir.name} scientific/smoke status differs")
    if summary["arm"] != arm or int(summary["seed"]) != seed:
        raise ValueError(f"{run_dir.name} arm or seed differs")
    if manifest["config_digest"] != contract["config_digest"]:
        raise ValueError(f"{run_dir.name} config digest does not close")
    if manifest["population_sha256"] != contract["population"]["sha256"]:
        raise ValueError(f"{run_dir.name} population digest does not close")
    if manifest["parent_checkpoint_sha256"] != contract["parent"]["checkpoint_sha256"]:
        raise ValueError(f"{run_dir.name} parent checkpoint does not close")
    if manifest["source_sha256"] != contract["source_sha256"]:
        raise ValueError(f"{run_dir.name} source hashes do not close")
    if manifest["output_count"] != len(manifest["output_hashes"]):
        raise ValueError(f"{run_dir.name} output count does not close")
    for relative, expected in manifest["output_hashes"].items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{run_dir.name} output hash mismatch: {relative}")
    if float(summary["initialization"]["parent_output_max_abs"]) > float(
        summary["initialization"]["output_closure_tolerance"]
    ):
        raise ValueError(f"{run_dir.name} failed parent-output closure")
    if float(summary["branch_replay_max_abs"]) > 1.0e-5:
        raise ValueError(f"{run_dir.name} failed all-on branch replay closure")
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "contract": contract,
        "summary": summary,
        "history": _read_json(run_dir / "history.json"),
        "native_rows": _read_json(run_dir / "case_metrics.json"),
        "nested_rows": _read_json(run_dir / "nested_case_metrics.json"),
        "predictions": _load_npz(run_dir / "predictions.npz"),
        "branch_metrics": _read_json(run_dir / "branch_cube_metrics.json"),
        "branch_predictions": _load_npz(run_dir / "branch_cube_predictions.npz"),
    }


def _shared_gradient_pairing(
    runs: Mapping[tuple[str, int], Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, dict[str, Any]]:
    pairing = {}
    for seed in seeds:
        control = runs[("continue_no_gradient", seed)]["contract"]["initialization"][
            "first_backward"
        ]
        active = runs[("activate_zero_gradient", seed)]["contract"]["initialization"][
            "first_backward"
        ]
        control_l2 = float(control["shared_gradient_l2"])
        active_l2 = float(active["shared_gradient_l2"])
        pairing[str(seed)] = {
            "control_sha256": control["shared_gradient_sha256"],
            "active_sha256": active["shared_gradient_sha256"],
            "exact_hash_match": (
                control["shared_gradient_sha256"] == active["shared_gradient_sha256"]
            ),
            "control_l2": control_l2,
            "active_l2": active_l2,
            "relative_l2_norm_difference": float(
                abs(control_l2 - active_l2)
                / max(control_l2, active_l2, np.finfo(float).tiny)
            ),
            "posthoc_scalar_norm_close": math.isclose(
                control_l2, active_l2, rel_tol=1.0e-8, abs_tol=1.0e-12
            ),
        }
    return pairing


def _load_matrix(
    matrix_dir: Path,
    *,
    smoke: bool,
    allow_nonbitwise_shared_gradient: bool = False,
) -> dict[tuple[str, int], dict[str, Any]]:
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = {
        (arm, seed): _verify_run(matrix_dir / f"{arm}_s{seed}", arm, seed, smoke=smoke)
        for seed in seeds
        for arm in ARMS
    }
    source_sets = {
        json.dumps(run["contract"]["source_sha256"], sort_keys=True)
        for run in runs.values()
    }
    provenance_sets = {
        json.dumps(run["contract"]["provenance_sha256"], sort_keys=True)
        for run in runs.values()
    }
    population_sets = {run["manifest"]["population_sha256"] for run in runs.values()}
    environment_sets = {
        json.dumps(_stable_environment(run["contract"]["environment"]), sort_keys=True)
        for run in runs.values()
    }
    if len(source_sets) != 1 or len(provenance_sets) != 1 or len(population_sets) != 1:
        raise ValueError(
            "continuation matrix does not share source/provenance/population"
        )
    if len(environment_sets) != 1:
        raise ValueError("continuation matrix does not share one execution environment")
    for seed in seeds:
        control = runs[("continue_no_gradient", seed)]
        active = runs[("activate_zero_gradient", seed)]
        for key in (
            "checkpoint_sha256",
            "manifest_sha256",
            "contract_sha256",
            "summary_sha256",
            "config_digest",
            "population_sha256",
        ):
            if control["contract"]["parent"][key] != active["contract"]["parent"][key]:
                raise ValueError(f"seed {seed} does not share the exact parent {key}")
        control_initial = control["contract"]["initialization"]
        active_initial = active["contract"]["initialization"]
        if (
            control_initial["shared_non_gradient_sha256"]
            != active_initial["shared_non_gradient_sha256"]
        ):
            raise ValueError(f"seed {seed} shared parent parameters differ")
        if (
            control_initial["first_backward"]["shared_gradient_sha256"]
            != active_initial["first_backward"]["shared_gradient_sha256"]
            and not allow_nonbitwise_shared_gradient
        ):
            raise ValueError(f"seed {seed} step-zero shared gradients differ")
        if any(row["gw2_nonzero"] != 0.0 for row in active_initial["gradient_state"]):
            raise ValueError(
                f"seed {seed} active branch did not start at exact zero output"
            )
        if not any(
            (row["gw2_gradient_l2"] or 0.0) > 0.0
            for row in active_initial["first_backward"]["branches"]
        ):
            raise ValueError(f"seed {seed} active branch cannot leave zero output")
    return runs


def _aggregate_selected(
    rows: list[dict[str, Any]], family: str | None = None
) -> dict[str, float]:
    selected = (
        rows if family is None else [row for row in rows if row["family"] == family]
    )
    return _aggregate_metrics(selected)


def _comparison_rows(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        for arm in ARMS:
            run = runs[(arm, seed)]
            primary_rows = _primary_rows(run)
            smooth_rows = [
                row
                for row in run["native_rows"]
                if row["split"] == "held"
                and row["family"] in {"smooth_tanh", "smooth_sine"}
            ]
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "parent_primary_value": float(
                        run["summary"]["parent"]["primary_value"]
                    ),
                    "primary_value": float(run["summary"]["primary_value"]),
                    "final_optimizer_loss": float(run["history"][-1]["optimizer_loss"]),
                    "primary_population": _aggregate_selected(primary_rows),
                    "primary_step": _aggregate_selected(primary_rows, "step"),
                    "primary_pulse": _aggregate_selected(primary_rows, "pulse"),
                    "held_smooth": _aggregate_selected(smooth_rows),
                    "elapsed_training_seconds": float(
                        run["summary"]["elapsed_training_seconds_this_invocation"]
                    ),
                }
            )
    return rows


def _paired_decision(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    primary_improvements = []
    train_improvements = []
    control_parent_improvements = []
    active_parent_improvements = []
    failures = {}
    for seed in seeds:
        control = runs[("continue_no_gradient", seed)]
        active = runs[("activate_zero_gradient", seed)]
        parent_primary = float(control["summary"]["parent"]["primary_value"])
        control_primary = float(control["summary"]["primary_value"])
        active_primary = float(active["summary"]["primary_value"])
        control_parent_improvements.append(
            (parent_primary - control_primary)
            / max(parent_primary, np.finfo(float).tiny)
        )
        active_parent_improvements.append(
            (parent_primary - active_primary)
            / max(parent_primary, np.finfo(float).tiny)
        )
        primary_improvements.append(
            (control_primary - active_primary)
            / max(control_primary, np.finfo(float).tiny)
        )
        control_train = float(control["history"][-1]["optimizer_loss"])
        active_train = float(active["history"][-1]["optimizer_loss"])
        train_improvements.append(
            (control_train - active_train) / max(control_train, np.finfo(float).tiny)
        )
        failures[str(seed)] = _paired_control_failures(control, active)
    primary_wins = sum(value > 0.0 for value in primary_improvements)
    train_wins = sum(value > 0.0 for value in train_improvements)
    primary_median = float(statistics.median(primary_improvements))
    train_median = float(statistics.median(train_improvements))
    primary_gate = primary_wins >= 2 and primary_median >= 0.05
    no_harm = all(not value for value in failures.values())
    learned = all(
        sum(
            row["gw2_l2"]
            for row in runs[("activate_zero_gradient", seed)]["summary"][
                "final_gradient_state"
            ]
        )
        > 1.0e-6
        for seed in seeds
    )
    if primary_gate and train_wins >= 2:
        interpretation = (
            "optimization_path_rescue"
            if no_harm
            else "optimization_path_rescue_with_structure_regressions"
        )
    elif train_wins >= 2 and primary_median <= 0.0:
        interpretation = "gradient_capacity_refits_train_but_harms_held_phase"
    elif learned and not primary_gate:
        interpretation = "gradient_branch_learns_without_material_held_benefit"
    else:
        interpretation = "inconclusive_or_activation_failure"
    return {
        "primary_relative_improvements": {
            str(seed): float(value)
            for seed, value in zip(seeds, primary_improvements, strict=True)
        },
        "training_objective_relative_improvements": {
            str(seed): float(value)
            for seed, value in zip(seeds, train_improvements, strict=True)
        },
        "continued_control_vs_parent_relative_improvements": {
            str(seed): float(value)
            for seed, value in zip(seeds, control_parent_improvements, strict=True)
        },
        "activated_vs_parent_relative_improvements": {
            str(seed): float(value)
            for seed, value in zip(seeds, active_parent_improvements, strict=True)
        },
        "median_continued_control_vs_parent_relative_improvement": float(
            statistics.median(control_parent_improvements)
        ),
        "median_activated_vs_parent_relative_improvement": float(
            statistics.median(active_parent_improvements)
        ),
        "primary_win_count": primary_wins,
        "training_objective_win_count": train_wins,
        "median_primary_relative_improvement": primary_median,
        "median_training_objective_relative_improvement": train_median,
        "primary_gate_pass": primary_gate,
        "control_failure_count": sum(len(value) for value in failures.values()),
        "control_failures": failures,
        "strict_rescue_gate_pass": primary_gate and train_wins >= 2 and no_harm,
        "active_gradient_learned": learned,
        "interpretation": interpretation,
    }


def _feature_positions(family: str, position: float) -> tuple[float, ...]:
    if family in {"step", "smooth_tanh"}:
        return (position, position + DISPLACEMENT)
    if family == "pulse":
        return (
            position,
            position + DISPLACEMENT,
            position + PULSE_WIDTH,
            position + DISPLACEMENT + PULSE_WIDTH,
        )
    return ()


def _branch_attribution(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    all_on = branch_mask_name((1, 1, 1))
    off_masks = {
        "spectral": branch_mask_name((0, 1, 1)),
        "pointwise": branch_mask_name((1, 0, 1)),
        "differential": branch_mask_name((1, 1, 0)),
    }
    result: dict[str, Any] = {"mask_rows": [], "acute_full_minus_branch_off": []}
    for seed in seeds:
        run = runs[("activate_zero_gradient", seed)]
        native = run["branch_metrics"]["populations"]["native"]
        mask_aggregates: dict[str, dict[str, float]] = {}
        for mask in BRANCH_MASKS:
            name = branch_mask_name(mask)
            rows = native[name]["case_metrics"]
            selected = [
                row
                for row in rows
                if row["split"] == "held"
                and row["family"] in {"step", "pulse"}
                and math.isclose(float(row["phase"]), 0.875)
            ]
            if not selected:
                selected = [
                    row
                    for row in rows
                    if row["split"] == "held" and row["family"] in {"step", "pulse"}
                ]
            aggregate = _aggregate_selected(selected)
            mask_aggregates[name] = aggregate
            result["mask_rows"].append(
                {
                    "seed": seed,
                    "mask": name,
                    "enabled": list(mask),
                    "primary_value": float(native[name]["primary_value"]),
                    "primary_metrics": aggregate,
                    "step_oscillatory_mass": _aggregate_selected(selected, "step")[
                        "oscillatory_mass_outside_front_band"
                    ],
                    "pulse_overshoot": _aggregate_selected(selected, "pulse")[
                        "normalized_overshoot"
                    ],
                    "pulse_undershoot": _aggregate_selected(selected, "pulse")[
                        "normalized_undershoot"
                    ],
                }
            )
        payload = run["predictions"]
        ny, nx = payload["current"].shape[1:]
        grid = build_structured_cell_grid((nx, ny))
        volume = grid.hx * grid.hy
        branch_payload = run["branch_predictions"]
        full = branch_payload[f"native__{all_on}"]
        target = payload["target_increment"]
        selected_indices = [
            index
            for index, family_value in enumerate(payload["families"])
            if str(family_value) in {"step", "pulse"}
            and str(payload["splits"][index]) == "held"
            and math.isclose(float(payload["phases"][index]), 0.875)
        ]
        if not selected_indices:
            selected_indices = [
                index
                for index, family_value in enumerate(payload["families"])
                if str(family_value) in {"step", "pulse"}
                and str(payload["splits"][index]) == "held"
            ]
        for branch, off_name in off_masks.items():
            off = branch_payload[f"native__{off_name}"]
            contributions = full - off
            cosines = []
            off_energies = []
            full_energies = []
            for index in selected_indices:
                family = str(payload["families"][index])
                features = _feature_positions(
                    family, float(payload["positions"][index])
                )
                outside_x = np.ones(grid.nx, dtype=bool)
                for position in features:
                    outside_x &= np.abs(grid.x_centers - position) > FRONT_BAND_WIDTH
                outside = np.repeat(outside_x[None, :], grid.ny, axis=0)
                off_error = off[index] - target[index]
                full_error = full[index] - target[index]
                contribution = contributions[index]
                a = off_error[outside].astype(np.float64)
                b = contribution[outside].astype(np.float64)
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                cosines.append(
                    float(np.dot(a, b) / denominator) if denominator else 0.0
                )
                off_energies.append(float(np.sum(np.square(a)) * volume))
                full_energies.append(
                    float(np.sum(np.square(full_error[outside])) * volume)
                )
            full_oscillatory_mass = mask_aggregates[all_on][
                "oscillatory_mass_outside_front_band"
            ]
            off_oscillatory_mass = mask_aggregates[off_name][
                "oscillatory_mass_outside_front_band"
            ]
            result["acute_full_minus_branch_off"].append(
                {
                    "seed": seed,
                    "branch": branch,
                    "mean_outside_error_contribution_cosine": float(np.mean(cosines)),
                    "mean_outside_error_energy_off": float(np.mean(off_energies)),
                    "mean_outside_error_energy_full": float(np.mean(full_energies)),
                    "mean_outside_error_energy_ratio_full_to_off": float(
                        np.mean(full_energies)
                        / max(np.mean(off_energies), np.finfo(float).tiny)
                    ),
                    "oscillatory_mass_full": full_oscillatory_mass,
                    "oscillatory_mass_off": off_oscillatory_mass,
                    "oscillatory_mass_ratio_full_to_off": float(
                        full_oscillatory_mass
                        / max(off_oscillatory_mass, np.finfo(float).tiny)
                    ),
                    "cancellation_supported": bool(
                        np.mean(cosines) < 0.0
                        and np.mean(full_energies) < np.mean(off_energies)
                        and full_oscillatory_mass < off_oscillatory_mass
                    ),
                }
            )
    return result


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
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf, metadata={"Creator": "W26-L2 P2-C0 analyzer", "CreationDate": None}
    )
    fig.savefig(png, dpi=300, metadata={"Software": "W26-L2 P2-C0 analyzer"})
    plt.close(fig)
    return [pdf, png]


def _paired_figure(rows: list[dict[str, Any]], seeds: Sequence[int]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "primary_value", "Held phase-0.875 primary"),
        (axes[1], "final_optimizer_loss", "Final training objective"),
    ):
        for seed in seeds:
            values = [
                next(
                    row[key]
                    for row in rows
                    if row["arm"] == arm and row["seed"] == seed
                )
                for arm in ARMS
            ]
            axis.plot((0, 1), values, color="#888888", alpha=0.6, linewidth=1.0)
        for index, arm in enumerate(ARMS):
            values = [row[key] for row in rows if row["arm"] == arm]
            axis.scatter(
                np.full(len(values), index),
                values,
                color=COLORS[arm],
                edgecolor="white",
                s=30,
            )
            axis.scatter(
                index, statistics.median(values), marker="D", color="#222222", s=18
            )
        axis.set_yscale("log")
        axis.set_xticks((0, 1), ("Keep no-grad", "Activate grad"))
        axis.set_title(title)
    fig.suptitle("Matched continuation from the same no-gradient checkpoint")
    return fig


def _convergence_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    for arm in ARMS:
        histories = [runs[(arm, seed)]["history"] for seed in seeds]
        steps = np.asarray([row["step"] for row in histories[0]])
        for axis, accessor in (
            (axes[0], lambda row: row["optimizer_loss"]),
            (axes[1], lambda row: row["held"]["mean_relative_increment_l2"]),
        ):
            values = np.asarray(
                [[accessor(row) for row in history] for history in histories]
            )
            axis.plot(
                steps,
                np.mean(values, axis=0),
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                label=LABELS[arm],
            )
            axis.fill_between(
                steps,
                np.min(values, axis=0),
                np.max(values, axis=0),
                color=COLORS[arm],
                alpha=0.15,
                linewidth=0,
            )
    axes[0].set_title("Training objective")
    axes[1].set_title("Held-population relative L2")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Continuation update")
    axes[0].set_ylabel("Error")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Continuation dynamics (mean and seed range)")
    return fig


def _gradient_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)
    histories = [runs[("activate_zero_gradient", seed)]["history"] for seed in seeds]
    steps = np.asarray([row["step"] for row in histories[0]])
    gw2 = np.asarray(
        [
            [
                [layer["gw2_l2"] for layer in row["gradient_parameters"]]
                for row in history
            ]
            for history in histories
        ]
    )
    gw1 = np.asarray(
        [
            [[layer["gw1"] for layer in row["gradient_parameters"]] for row in history]
            for history in histories
        ]
    )
    for layer in range(gw2.shape[2]):
        axes[0].plot(steps, np.mean(gw2[:, :, layer], axis=0), label=f"L{layer}")
        axes[1].plot(steps, np.mean(gw1[:, :, layer], axis=0), label=f"L{layer}")
    axes[0].set_ylabel("gw2 Frobenius norm")
    axes[1].set_ylabel("gw1 scalar")
    for axis in axes:
        axis.set_xlabel("Continuation update")
        axis.legend(frameon=False, ncol=2, fontsize=7)
    axes[0].set_title("Zero-output weights become active")
    axes[1].set_title("Learned differential scaling")
    return fig


def _structure_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    specifications = (
        ("Relative L2", lambda row: row["primary_population"]["relative_increment_l2"]),
        (
            "Step osc. mass",
            lambda row: row["primary_step"]["oscillatory_mass_outside_front_band"],
        ),
        ("Pulse overshoot", lambda row: row["primary_pulse"]["normalized_overshoot"]),
        ("Pulse undershoot", lambda row: row["primary_pulse"]["normalized_undershoot"]),
        (
            "TV excess",
            lambda row: row["primary_population"]["positive_total_variation_excess"],
        ),
        (
            "TV deficit",
            lambda row: row["primary_population"]["total_variation_deficit"],
        ),
        ("Smooth error", lambda row: row["primary_population"]["smooth_region_error"]),
        ("Smooth-control L2", lambda row: row["held_smooth"]["relative_increment_l2"]),
    )
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 5.0), constrained_layout=True)
    for axis, (title, accessor) in zip(axes.ravel(), specifications, strict=True):
        for index, arm in enumerate(ARMS):
            values = [accessor(row) for row in rows if row["arm"] == arm]
            axis.bar(
                index,
                np.mean(values),
                yerr=np.std(values, ddof=1) if len(values) > 1 else 0.0,
                color=COLORS[arm],
                capsize=2,
            )
        axis.set_xticks((0, 1), ("Keep", "Activate"), rotation=20)
        axis.set_title(title)
        if all(accessor(row) > 0.0 for row in rows):
            axis.set_yscale("log")
    fig.suptitle("Held phase-0.875 structure diagnostics")
    return fig


def _representative_indices(payload: dict[str, np.ndarray]) -> list[int]:
    result = []
    for family in ("step", "pulse"):
        candidates = [
            index
            for index, value in enumerate(payload["families"])
            if str(value) == family
            and str(payload["splits"][index]) == "held"
            and math.isclose(float(payload["phases"][index]), 0.875)
        ]
        if not candidates:
            candidates = [
                index
                for index, value in enumerate(payload["families"])
                if str(value) == family and str(payload["splits"][index]) == "held"
            ]
        if not candidates:
            raise ValueError(f"missing representative {family} case")
        result.append(candidates[len(candidates) // 2])
    return result


def _profiles_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seed: int
) -> plt.Figure:
    reference = runs[(ARMS[0], seed)]["predictions"]
    selected = _representative_indices(reference)
    nx = reference["current"].shape[2]
    x = (np.arange(nx) + 0.5) / nx
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0), constrained_layout=True)
    for row_index, case_index in enumerate(selected):
        target_next = np.mean(reference["target_next"][case_index], axis=0)
        axes[row_index, 0].plot(
            x, target_next, color="#222222", linewidth=1.4, label="Exact"
        )
        for arm in ARMS:
            payload = runs[(arm, seed)]["predictions"]
            predicted_next = np.mean(
                payload["current"][case_index]
                + payload["predicted_increment"][case_index],
                axis=0,
            )
            error = np.mean(
                payload["predicted_increment"][case_index]
                - payload["target_increment"][case_index],
                axis=0,
            )
            axes[row_index, 0].plot(
                x,
                predicted_next,
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                label=LABELS[arm],
            )
            axes[row_index, 1].plot(
                x,
                error,
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                label=LABELS[arm],
            )
        family = str(reference["families"][case_index]).title()
        axes[row_index, 0].set_ylabel(f"{family} next state")
        axes[row_index, 1].set_ylabel(f"{family} increment error")
        for axis in axes[row_index]:
            for position in _feature_positions(
                str(reference["families"][case_index]),
                float(reference["positions"][case_index]),
            ):
                axis.axvspan(
                    position - FRONT_BAND_WIDTH,
                    position + FRONT_BAND_WIDTH,
                    color="#999999",
                    alpha=0.12,
                    linewidth=0,
                )
            axis.set_xlim(0.15, 0.9)
    axes[0, 0].set_title("Target and prediction")
    axes[0, 1].set_title("Signed error")
    for axis in axes[-1]:
        axis.set_xlabel("x")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"Held shock-phase profiles with fixed physical front bands, seed {seed}"
    )
    return fig


def _branch_figure(attribution: dict[str, Any], seeds: Sequence[int]) -> plt.Figure:
    names = [branch_mask_name(mask) for mask in BRANCH_MASKS]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "primary_value", "Held primary"),
        (axes[1], "step_oscillatory_mass", "Step outside-band oscillatory mass"),
    ):
        means = []
        deviations = []
        for name in names:
            values = [
                row[key] for row in attribution["mask_rows"] if row["mask"] == name
            ]
            means.append(np.mean(values))
            deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        axes_values = np.arange(len(names))
        axis.bar(axes_values, means, yerr=deviations, color="#5B8DB8", capsize=2)
        axis.set_xticks(axes_values, names, rotation=45, ha="right")
        if max(means) > 0.0:
            axis.set_yscale("log")
        axis.set_title(title)
    fig.suptitle(
        f"Frozen 2^3 branch cube after continuation ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})"
    )
    return fig


def _cancellation_figure(attribution: dict[str, Any]) -> plt.Figure:
    branches = ("spectral", "pointwise", "differential")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), constrained_layout=True)
    for index, branch in enumerate(branches):
        rows = [
            row
            for row in attribution["acute_full_minus_branch_off"]
            if row["branch"] == branch
        ]
        cosines = [row["mean_outside_error_contribution_cosine"] for row in rows]
        ratios = [row["mean_outside_error_energy_ratio_full_to_off"] for row in rows]
        axes[0].bar(
            index,
            np.mean(cosines),
            yerr=np.std(cosines, ddof=1) if len(cosines) > 1 else 0.0,
            color="#7A5195",
            capsize=2,
        )
        axes[1].bar(
            index,
            np.mean(ratios),
            yerr=np.std(ratios, ddof=1) if len(ratios) > 1 else 0.0,
            color="#EF5675",
            capsize=2,
        )
    for axis in axes:
        axis.set_xticks(
            range(3), ("Spectral", "Pointwise", "Differential"), rotation=20
        )
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].axhline(1.0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("cos(off error, decoded addition)")
    axes[1].set_ylabel("full / branch-off outside error energy")
    axes[0].set_title("Negative alignment supports cancellation")
    axes[1].set_title("Below one means branch reduces error")
    return fig


def _resolution_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    resolutions = (32, 64, 128)
    fig, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for arm in ARMS:
        means = []
        deviations = []
        for nx in resolutions:
            per_seed = []
            for seed in seeds:
                run = runs[(arm, seed)]
                rows = (
                    [
                        row
                        for row in run["native_rows"]
                        if row["split"] == "held" and row["family"] in {"step", "pulse"}
                    ]
                    if nx == 64
                    else [
                        row
                        for row in run["nested_rows"]
                        if row["resolution"] == [nx, nx // 2]
                        and row["family"] in {"step", "pulse"}
                    ]
                )
                per_seed.append(
                    float(
                        np.mean(
                            [row["metrics"]["relative_increment_l2"] for row in rows]
                        )
                    )
                )
            means.append(np.mean(per_seed))
            deviations.append(np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0)
        axis.errorbar(
            resolutions,
            means,
            yerr=deviations,
            marker="o",
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            label=LABELS[arm],
            capsize=2,
        )
    axis.set_yscale("log")
    axis.set_xticks(resolutions)
    axis.set_xlabel("nx (common physical cases)")
    axis.set_ylabel("Mean discontinuous relative increment L2")
    axis.set_title("Resolution transfer after continuation")
    axis.legend(frameon=False, fontsize=7)
    return fig


def _spectrum_figure(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    sample = runs[(ARMS[0], seeds[0])]["predictions"]
    ny, nx = sample["current"].shape[1:]
    grid = build_structured_cell_grid((nx, ny))
    q, _ = physical_cosine_spectrum(np.zeros(grid.array_shape), grid)
    edges = np.arange(0.0, math.ceil(float(np.max(q))) + 2.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axis = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    for arm in ARMS:
        spectra = []
        for seed in seeds:
            payload = runs[(arm, seed)]["predictions"]
            curves = []
            for index in _representative_indices(payload):
                error = (
                    payload["predicted_increment"][index]
                    - payload["target_increment"][index]
                )
                physical_q, energy = physical_cosine_spectrum(error, grid)
                curves.append(np.histogram(physical_q, bins=edges, weights=energy)[0])
            spectra.append(np.mean(curves, axis=0))
        values = np.asarray(spectra)
        mean = np.mean(values, axis=0)
        floor = max(float(np.max(mean)) * 1.0e-12, np.finfo(float).tiny)
        axis.plot(
            centers,
            np.maximum(mean, floor),
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            label=LABELS[arm],
        )
        axis.fill_between(
            centers,
            np.maximum(np.min(values, axis=0), floor),
            np.maximum(np.max(values, axis=0), floor),
            color=COLORS[arm],
            alpha=0.15,
            linewidth=0,
        )
    axis.axvline(
        SMOOTH_FILTER_PASS_WAVENUMBER, color="#666666", linestyle="--", linewidth=0.8
    )
    axis.axvline(
        SMOOTH_FILTER_STOP_WAVENUMBER, color="#666666", linestyle=":", linewidth=0.8
    )
    axis.set_yscale("log")
    axis.set_xlabel("Physical cycles per unit length")
    axis.set_ylabel("Radial error energy")
    axis.set_title("Physical error spectrum; energy is not by itself ripple")
    axis.legend(frameon=False, fontsize=7)
    return fig


def _representative_seed(
    runs: dict[tuple[str, int], dict[str, Any]], seeds: Sequence[int]
) -> int:
    values = {
        seed: float(runs[("activate_zero_gradient", seed)]["summary"]["primary_value"])
        for seed in seeds
    }
    median = statistics.median(values.values())
    return min(seeds, key=lambda seed: (abs(values[seed] - median), seed))


def run_analysis(
    matrix_dir: Path,
    output_dir: Path,
    *,
    smoke: bool,
    allow_nonbitwise_shared_gradient: bool = False,
) -> dict[str, Any]:
    matrix_dir = matrix_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = _load_matrix(
        matrix_dir,
        smoke=smoke,
        allow_nonbitwise_shared_gradient=allow_nonbitwise_shared_gradient,
    )
    gradient_pairing = _shared_gradient_pairing(runs, seeds)
    registered_gradient_closure_pass = all(
        row["exact_hash_match"] for row in gradient_pairing.values()
    )
    rows = _comparison_rows(runs, seeds)
    decision = _paired_decision(runs, seeds)
    decision["preclosure_descriptive_interpretation"] = decision["interpretation"]
    decision["registered_step_zero_shared_gradient_hash_gate_pass"] = (
        registered_gradient_closure_pass
    )
    decision["descriptive_only_due_to_closure"] = not registered_gradient_closure_pass
    if not registered_gradient_closure_pass:
        decision["strict_rescue_gate_pass"] = False
        decision["interpretation"] = (
            "activation_failure_inconclusive_nonbitwise_step_zero_shared_gradient"
        )
    attribution = _branch_attribution(runs, seeds)
    representative_seed = _representative_seed(runs, seeds)
    closures = {
        "all_passed": registered_gradient_closure_pass,
        "descriptive_after_nonbitwise_gradient_closure": (
            allow_nonbitwise_shared_gradient and not registered_gradient_closure_pass
        ),
        "input_manifest_sha256": {
            f"{arm}_s{seed}": sha256_file(run["run_dir"] / "manifest.json")
            for (arm, seed), run in runs.items()
        },
        "population_sha256": next(iter(runs.values()))["manifest"]["population_sha256"],
        "source_sha256": next(iter(runs.values()))["contract"]["source_sha256"],
        "provenance_sha256": next(iter(runs.values()))["contract"]["provenance_sha256"],
        "paired_parent_checkpoints": {
            str(seed): runs[(ARMS[0], seed)]["contract"]["parent"]["checkpoint_sha256"]
            for seed in seeds
        },
        "step_zero_shared_gradient_sha256": {
            str(seed): runs[(ARMS[0], seed)]["contract"]["initialization"][
                "first_backward"
            ]["shared_gradient_sha256"]
            for seed in seeds
        },
        "step_zero_shared_gradient_pairing": gradient_pairing,
    }
    write_json(output_dir / "comparison_rows.json", rows)
    write_json(output_dir / "paired_decision.json", decision)
    write_json(output_dir / "branch_attribution.json", attribution)
    write_json(output_dir / "pairing_closures.json", closures)
    _style()
    figures = []
    figures += _save(
        _paired_figure(rows, seeds), output_dir, "fig_paired_primary_and_train"
    )
    figures += _save(
        _convergence_figure(runs, seeds), output_dir, "fig_continuation_convergence"
    )
    figures += _save(
        _gradient_figure(runs, seeds), output_dir, "fig_gradient_activation"
    )
    figures += _save(_structure_figure(rows), output_dir, "fig_structure_metrics")
    figures += _save(
        _profiles_figure(runs, representative_seed), output_dir, "fig_phase_profiles"
    )
    figures += _save(
        _branch_figure(attribution, seeds), output_dir, "fig_branch_factorial"
    )
    figures += _save(
        _cancellation_figure(attribution), output_dir, "fig_branch_cancellation"
    )
    if not smoke:
        figures += _save(
            _resolution_figure(runs, seeds), output_dir, "fig_resolution_transfer"
        )
    figures += _save(_spectrum_figure(runs, seeds), output_dir, "fig_physical_spectrum")
    visual = {
        "schema": SCHEMA,
        "files": [path.name for path in figures],
        "sha256": {path.name: sha256_file(path) for path in figures},
    }
    write_json(output_dir / "visual_manifest.json", visual)
    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "science_result": not smoke,
        "run_count": len(runs),
        "seeds": list(seeds),
        "representative_seed": representative_seed,
        "pairing_closures": closures,
        "paired_decision": decision,
        "comparison_rows": rows,
        "branch_attribution": attribution,
        "visualizations": visual,
        "claim_boundary": (
            "Frozen masks are acute, non-additive counterfactuals. Negative decoded "
            "alignment plus lower outside-band error and registered oscillatory mass "
            "supports cancellation but does not identify a unique additive cause or "
            "strict spectral Gibbs phenomenon. A failed exact step-zero shared-gradient "
            "hash gate makes all paired effects descriptive and the registered causal "
            "decision inconclusive."
        ),
        "external_result_to_claim_review": "not_run_private_scope_not_authorized",
    }
    write_json(output_dir / "summary.json", summary)
    outputs = sorted(
        path for path in output_dir.iterdir() if path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "science_result": not smoke,
        "analysis_source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in ANALYSIS_SOURCE_PATHS
        },
        "input_manifest_sha256": closures["input_manifest_sha256"],
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--descriptive-after-nonbitwise-gradient-closure",
        action="store_true",
        help=(
            "emit descriptive outputs while retaining a failed registered exact-hash "
            "closure and an inconclusive causal verdict"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        args.matrix_dir,
        args.output_dir,
        smoke=args.smoke,
        allow_nonbitwise_shared_gradient=(
            args.descriptive_after_nonbitwise_gradient_closure
        ),
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
