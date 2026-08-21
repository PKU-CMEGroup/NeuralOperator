#!/usr/bin/env python3
"""Run the W26-L2-P3 frozen, zero-inclusive intermediate-filter screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_gradient_ablation import (
    _paired_control_failures,
    _stable_environment,
    _thresholds,
    _verify_run,
)
from scripts.time_dependent_no.analyze_pcno_p2_frozen_branch_cube import (
    P2_POPULATION_SHA256,
    P2_TRAINER_SHA256,
    _registered_populations,
    _source_hashes,
    _verify_registered_inputs,
)
from scripts.time_dependent_no.analyze_pcno_shock_pathways import (
    _mask_aggregates,
    _metric_scalars,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
    build_batch,
    build_model,
    case_metric_rows,
    expand_fourier_tensors,
    model_state_sha256,
    sha256_file,
    write_json,
    write_npz,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import SEEDS
from utility.time_dependent_no.pcno_intermediate_filter_screen import (
    NATIVE_REPLAY,
    REGISTERED_INTERVENTIONS,
    prepare_fixed_physical_operator,
    replay_pcno_intervention,
)
from utility.time_dependent_no.pcno_shock_representation import (
    StructuredCellGrid,
    physical_cosine_spectrum,
)

SCHEMA = "w26_l2_p3_frozen_intermediate_filter_v1"
WORKING_ID = "W26-L2-P3-FS"
SELECTED_LAYER = 0
ARMS = ("F0",) + REGISTERED_INTERVENTIONS
DEFAULT_MATRIX = (
    ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2_gradient_ablation" / "runs"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p3_intermediate_filter"
)
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_resolution_pathways.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "utility/time_dependent_no/pcno_intermediate_filter_screen.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "scripts/time_dependent_no/analyze_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_p2_frozen_branch_cube.py",
    "scripts/time_dependent_no/analyze_pcno_intermediate_filter_screen.py",
)
PROVENANCE_PATHS = ("docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",)
LABELS = {
    "F0": "Native bypass",
    "R_grad_fixed_physical": "Fixed-radius grad",
    "S_grad": "Smooth raw grad",
    "H_grad": "Hard raw grad",
    "S_differential_output": "Smooth D output",
    "S_pointwise_output": "Smooth P output",
    "S_preactivation": "Smooth preactivation",
    "S_postactivation": "Smooth postactivation",
}
COLORS = dict(zip(ARMS, plt.get_cmap("tab10").colors, strict=False))
PRIMARY_METRIC = "relative_increment_l2"
STRUCTURE_METRICS = (
    "normalized_overshoot",
    "normalized_undershoot",
    "oscillatory_mass_outside_front_band",
    "smooth_region_error",
    "positive_total_variation_excess",
    "total_variation_deficit",
    "maximum_front_position_error",
    "maximum_front_strength_error",
    "maximum_front_thickness_excess",
    "increment_integral_error",
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_full_runs(
    matrix_dir: Path, *, smoke: bool
) -> dict[tuple[str, int], dict[str, Any]]:
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = {
        ("full", seed): _verify_run(
            matrix_dir / f"full_s{seed}", "full", seed, smoke=smoke
        )
        for seed in seeds
    }
    source_sets = {
        json.dumps(run["manifest"]["source_sha256"], sort_keys=True)
        for run in runs.values()
    }
    provenance_sets = {
        json.dumps(run["contract"]["provenance_sha256"], sort_keys=True)
        for run in runs.values()
    }
    population_sets = {run["contract"]["population"]["sha256"] for run in runs.values()}
    environment_sets = {
        json.dumps(_stable_environment(run["contract"]["environment"]), sort_keys=True)
        for run in runs.values()
    }
    normalized_configs = set()
    for run in runs.values():
        config = dict(run["contract"]["config"])
        for key in ("seed", "variant", "working_run_id"):
            config.pop(key)
        normalized_configs.add(json.dumps(config, sort_keys=True))
    closures = {
        "source": source_sets,
        "provenance": provenance_sets,
        "population": population_sets,
        "environment": environment_sets,
        "config": normalized_configs,
    }
    failures = [name for name, values in closures.items() if len(values) != 1]
    if failures:
        raise ValueError(f"full checkpoints differ in {failures}")
    return runs


def _case_rows(
    records: list[RegisteredCase],
    predictions: np.ndarray,
    grid: StructuredCellGrid,
    *,
    arm: str,
) -> list[dict[str, Any]]:
    rows = case_metric_rows(records, predictions, grid)
    for row in rows:
        row["resolution"] = list(grid.resolution)
        row["arm"] = arm
        row["scalars"] = _metric_scalars(row["metrics"])
    return rows


@torch.inference_mode()
def _predict_population(
    model: torch.nn.Module,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], float]:
    predictions = {arm: [] for arm in ARMS}
    maximum_replay_error = 0.0
    model.eval()
    fixed_operator = prepare_fixed_physical_operator(
        grid,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        model_input, _, aux = build_batch(selected, grid, device)
        fourier_tensors = expand_fourier_tensors(model, aux, len(selected))
        direct = model(model_input, aux, fourier_tensors=fourier_tensors)
        replay = replay_pcno_intervention(
            model,
            model_input,
            aux,
            grid,
            intervention=NATIVE_REPLAY,
            selected_layer=SELECTED_LAYER,
            fourier_tensors=fourier_tensors,
        )
        maximum_replay_error = max(
            maximum_replay_error,
            float(torch.max(torch.abs(replay - direct)).detach().cpu()),
        )
        predictions["F0"].append(direct[..., 0].detach().cpu().numpy())
        for arm in REGISTERED_INTERVENTIONS:
            output = replay_pcno_intervention(
                model,
                model_input,
                aux,
                grid,
                intervention=arm,
                selected_layer=SELECTED_LAYER,
                fixed_operator=(
                    fixed_operator if arm == "R_grad_fixed_physical" else None
                ),
                fourier_tensors=fourier_tensors,
            )
            predictions[arm].append(output[..., 0].detach().cpu().numpy())
    return (
        {
            arm: np.concatenate(parts, axis=0).reshape(len(records), grid.ny, grid.nx)
            for arm, parts in predictions.items()
        },
        maximum_replay_error,
    )


def _is_primary(row: Mapping[str, Any]) -> bool:
    return (
        tuple(row["resolution"]) == (64, 32)
        and row["split"] == "held"
        and row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.875)
    )


def _is_train_anchor(row: Mapping[str, Any]) -> bool:
    return (
        tuple(row["resolution"]) == (64, 32)
        and row["split"] == "train"
        and row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.0)
    )


def _row_map(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (tuple(row["resolution"]), row["case_id"])
        if key in result:
            raise ValueError(f"duplicate metric row {key}")
        result[key] = row
    return result


def _front_valid(row: Mapping[str, Any]) -> bool:
    return all(front["front_gate_valid"] for front in row["metrics"]["fronts"])


def _absolute_row_pass(row: dict[str, Any]) -> bool:
    scalars = row["scalars"]
    return _front_valid(row) and all(
        scalars[metric] <= limit for metric, limit in _thresholds(row).items()
    )


def _relative_row_pass(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    limits = _thresholds(baseline)
    return _front_valid(candidate) and all(
        candidate["scalars"][metric] <= max(limit, 1.05 * baseline["scalars"][metric])
        for metric, limit in limits.items()
    )


def _mean(rows: Sequence[dict[str, Any]], metric: str) -> float:
    if not rows:
        raise ValueError("cannot average an empty metric population")
    return float(np.mean([row["scalars"][metric] for row in rows]))


def _nonnegative_ratio(candidate: float, baseline: float) -> float:
    tolerance = 32.0 * np.finfo(float).eps
    if baseline <= tolerance:
        return 1.0 if candidate <= tolerance else float("inf")
    return float(candidate / baseline)


def _seed_decision(
    baseline_run: Mapping[str, Any], candidate_run: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_rows = baseline_run["native_rows"] + baseline_run["nested_rows"]
    candidate_rows = candidate_run["native_rows"] + candidate_run["nested_rows"]
    baseline_map = _row_map(baseline_rows)
    candidate_map = _row_map(candidate_rows)
    if baseline_map.keys() != candidate_map.keys():
        raise ValueError("candidate and native bypass populations differ")

    primary_keys = [key for key, row in baseline_map.items() if _is_primary(row)]
    train_keys = [key for key, row in baseline_map.items() if _is_train_anchor(row)]
    if len(primary_keys) != 6 or len(train_keys) != 6:
        raise ValueError("registered 6+6 native decision population is incomplete")
    primary_baseline = [baseline_map[key] for key in primary_keys]
    primary_candidate = [candidate_map[key] for key in primary_keys]
    baseline_mean = _mean(primary_baseline, PRIMARY_METRIC)
    candidate_mean = _mean(primary_candidate, PRIMARY_METRIC)
    relative_reduction = (baseline_mean - candidate_mean) / max(
        baseline_mean, np.finfo(float).tiny
    )
    held_success = [
        (
            baseline_map[key]["scalars"][PRIMARY_METRIC]
            - candidate_map[key]["scalars"][PRIMARY_METRIC]
        )
        / max(baseline_map[key]["scalars"][PRIMARY_METRIC], np.finfo(float).tiny)
        >= 0.20
        and _absolute_row_pass(candidate_map[key])
        for key in primary_keys
    ]
    train_success = [
        _relative_row_pass(baseline_map[key], candidate_map[key]) for key in train_keys
    ]
    paired_failures = _paired_control_failures(
        dict(baseline_run),
        dict(candidate_run),
    )

    nested_baseline = [
        row
        for row in baseline_run["nested_rows"]
        if row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.875)
    ]
    nested_candidate = [
        row
        for row in candidate_run["nested_rows"]
        if row["family"] in {"step", "pulse"}
        and math.isclose(float(row["phase"]), 0.875)
    ]
    nested_reduction = None
    if nested_baseline:
        nested_base_mean = _mean(nested_baseline, PRIMARY_METRIC)
        nested_candidate_mean = _mean(nested_candidate, PRIMARY_METRIC)
        nested_reduction = (nested_base_mean - nested_candidate_mean) / max(
            nested_base_mean, np.finfo(float).tiny
        )
    return {
        "primary_baseline": baseline_mean,
        "primary_candidate": candidate_mean,
        "primary_relative_reduction": float(relative_reduction),
        "held_success_count": int(sum(held_success)),
        "train_success_count": int(sum(train_success)),
        "twelve_case_success_count": int(sum(held_success) + sum(train_success)),
        "registered_seed_gate_pass": bool(
            sum(held_success) >= 5 and sum(held_success) + sum(train_success) >= 10
        ),
        "primary_absolute_gate_pass": bool(
            all(_absolute_row_pass(row) for row in primary_candidate)
        ),
        "paired_control_failure_count": len(paired_failures),
        "paired_control_failures": paired_failures,
        "nested_phase_0p875_relative_reduction": (
            None if nested_reduction is None else float(nested_reduction)
        ),
    }


def build_decision(
    runs: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    strict_replay_pass: bool,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for arm in REGISTERED_INTERVENTIONS:
        seed_rows = {
            str(seed): _seed_decision(runs[f"s{seed}"]["F0"], runs[f"s{seed}"][arm])
            for seed in seeds
        }
        seed_gate_count = sum(
            row["registered_seed_gate_pass"] for row in seed_rows.values()
        )
        absolute_gate_count = sum(
            row["primary_absolute_gate_pass"] for row in seed_rows.values()
        )
        control_failure_count = sum(
            row["paired_control_failure_count"] for row in seed_rows.values()
        )
        nested_gate_count = sum(
            row["nested_phase_0p875_relative_reduction"] is not None
            and row["nested_phase_0p875_relative_reduction"] >= 0.20
            for row in seed_rows.values()
        )
        base_pass = bool(
            strict_replay_pass
            and seed_gate_count >= 2
            and absolute_gate_count >= 2
            and control_failure_count == 0
        )
        screen_pass = base_pass and (
            arm != "R_grad_fixed_physical" or nested_gate_count >= 2
        )
        reductions = [row["primary_relative_reduction"] for row in seed_rows.values()]
        decisions[arm] = {
            "seed_results": seed_rows,
            "seed_gate_count": int(seed_gate_count),
            "primary_absolute_gate_count": int(absolute_gate_count),
            "paired_control_failure_count": int(control_failure_count),
            "nested_reduction_gate_count": int(nested_gate_count),
            "median_primary_relative_reduction": float(np.median(reductions)),
            "screen_pass": bool(screen_pass and arm != "H_grad"),
            "negative_control_only": arm == "H_grad",
        }

    eligible = []
    if decisions["R_grad_fixed_physical"]["screen_pass"]:
        eligible.append("R_grad_fixed_physical")
    smooth_passes = [
        arm
        for arm in REGISTERED_INTERVENTIONS
        if arm not in {"R_grad_fixed_physical", "H_grad"}
        and decisions[arm]["screen_pass"]
    ]
    smooth_passes.sort(
        key=lambda arm: (
            -decisions[arm]["seed_gate_count"],
            -decisions[arm]["median_primary_relative_reduction"],
            decisions[arm]["paired_control_failure_count"],
            REGISTERED_INTERVENTIONS.index(arm),
        )
    )
    eligible.extend(smooth_passes[:1])
    return {
        "selected_layer": SELECTED_LAYER,
        "primary_population": "native held phase 0.875 step/pulse, six cases per seed",
        "twelve_case_rule": (
            "six held cases require >=20% L2 reduction and every absolute guard; "
            "six train-phase-0 cases require paired <=1.05 no-harm; pass is >=5/6 "
            "held and >=10/12 total in >=2/3 seeds"
        ),
        "global_no_harm_rule": (
            "zero paired case-level failures over every native and nested control"
        ),
        "strict_replay_pass": bool(strict_replay_pass),
        "arms": decisions,
        "eligible_for_later_training": eligible,
        "eligible_count": len(eligible),
        "eligibility_cap_respected": len(eligible) <= 2,
        "training_authorized": False,
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
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"Creator": WORKING_ID, "CreationDate": None})
    fig.savefig(png, dpi=300, metadata={"Software": WORKING_ID})
    plt.close(fig)
    return [pdf, png]


def _primary_figure(decision: Mapping[str, Any], seeds: Sequence[int]) -> plt.Figure:
    arms = list(REGISTERED_INTERVENTIONS)
    values = np.asarray(
        [
            [
                decision["arms"][arm]["seed_results"][str(seed)]["primary_candidate"]
                / decision["arms"][arm]["seed_results"][str(seed)]["primary_baseline"]
                for seed in seeds
            ]
            for arm in arms
        ]
    )
    fig, axis = plt.subplots(figsize=(8.6, 3.5), constrained_layout=True)
    positions = np.arange(len(arms))
    axis.bar(
        positions,
        values.mean(axis=1),
        yerr=values.std(axis=1, ddof=1) if len(seeds) > 1 else None,
        color=[COLORS[arm] for arm in arms],
        alpha=0.82,
        capsize=2,
    )
    for index in range(len(arms)):
        axis.scatter(
            np.full(len(seeds), index), values[index], color="#222222", s=10, zorder=3
        )
    axis.axhline(1.0, color="#333333", linewidth=0.8)
    axis.axhline(0.8, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xticks(positions, [LABELS[arm] for arm in arms], rotation=30, ha="right")
    axis.set_ylabel("Candidate / native held-phase L2")
    axis.set_title("Frozen block-0 interventions; seed dots and mean +/- SD")
    return fig


def _structure_figure(
    runs: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]
) -> plt.Figure:
    arms = list(REGISTERED_INTERVENTIONS)
    ratios = np.zeros((len(arms), len(STRUCTURE_METRICS)), dtype=np.float64)
    for i, arm in enumerate(arms):
        for j, metric in enumerate(STRUCTURE_METRICS):
            paired = []
            for seed in seeds:
                baseline = [
                    row
                    for row in runs[f"s{seed}"]["F0"]["native_rows"]
                    if _is_primary(row)
                ]
                candidate = [
                    row
                    for row in runs[f"s{seed}"][arm]["native_rows"]
                    if _is_primary(row)
                ]
                paired.append(
                    _nonnegative_ratio(
                        _mean(candidate, metric),
                        _mean(baseline, metric),
                    )
                )
            ratios[i, j] = float(np.median(paired))
    shown = np.clip(np.log2(np.maximum(ratios, 2.0**-4)), -4.0, 4.0)
    fig, axis = plt.subplots(figsize=(10.2, 4.0), constrained_layout=True)
    image = axis.imshow(shown, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    axis.set_yticks(range(len(arms)), [LABELS[arm] for arm in arms])
    axis.set_xticks(
        range(len(STRUCTURE_METRICS)),
        [name.replace("_", "\n") for name in STRUCTURE_METRICS],
        rotation=35,
        ha="right",
    )
    axis.grid(False)
    axis.set_title("Median paired structure ratio (color is clipped log2 ratio)")
    fig.colorbar(image, ax=axis, label="log2(candidate/native)")
    return fig


def _representative_indices(records: Sequence[RegisteredCase]) -> dict[str, int]:
    result = {}
    for family in ("step", "pulse"):
        matches = [
            index
            for index, record in enumerate(records)
            if record.split == "held"
            and record.family == family
            and math.isclose(float(record.phase), 0.875)
            and int(record.anchor_index) == 1
        ]
        if len(matches) != 1:
            raise ValueError(f"representative {family} case is not unique")
        result[family] = matches[0]
    return result


def _profile_figure(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
    seed: int,
) -> plt.Figure:
    indices = _representative_indices(records)
    fig, axes = plt.subplots(
        2, 2, figsize=(9.4, 5.6), sharex=True, constrained_layout=True
    )
    for row_index, family in enumerate(("step", "pulse")):
        index = indices[family]
        target = records[index].case.increment.mean(axis=0)
        axes[row_index, 0].plot(
            grid.x_centers, target, color="#111111", linewidth=1.4, label="Target"
        )
        axes[row_index, 1].axhline(0.0, color="#111111", linewidth=0.7)
        for arm in ARMS:
            prediction = arrays[f"s{seed}__native__{arm}"][index].mean(axis=0)
            axes[row_index, 0].plot(
                grid.x_centers,
                prediction,
                color=COLORS[arm],
                linewidth=0.9,
                label=LABELS[arm],
            )
            axes[row_index, 1].plot(
                grid.x_centers,
                prediction - target,
                color=COLORS[arm],
                linewidth=0.9,
                label=LABELS[arm],
            )
        axes[row_index, 0].set_ylabel(f"{family.title()} increment")
        axes[row_index, 1].set_ylabel(f"{family.title()} error")
    axes[0, 0].set_title("Target and frozen predictions")
    axes[0, 1].set_title("Prediction minus exact increment")
    axes[1, 0].set_xlabel("Physical x")
    axes[1, 1].set_xlabel("Physical x")
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=3)
    return fig


def _error_map_figure(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
    seed: int,
) -> plt.Figure:
    indices = _representative_indices(records)
    errors = {
        (arm, family): arrays[f"s{seed}__native__{arm}"][indices[family]]
        - records[indices[family]].case.increment
        for arm in ARMS
        for family in ("step", "pulse")
    }
    scale = max(float(np.max(np.abs(value))) for value in errors.values())
    fig, axes = plt.subplots(
        len(ARMS),
        2,
        figsize=(7.2, 12.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, arm in enumerate(ARMS):
        for column, family in enumerate(("step", "pulse")):
            image = axes[row, column].imshow(
                errors[(arm, family)],
                origin="lower",
                extent=(0.0, grid.lengths[0], 0.0, grid.lengths[1]),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
            )
            if column == 0:
                axes[row, column].set_ylabel(LABELS[arm])
            if row == 0:
                axes[row, column].set_title(f"{family.title()} error")
            axes[row, column].grid(False)
    if image is not None:
        fig.colorbar(image, ax=axes, label="Predicted minus exact increment")
    return fig


def _spectrum_figure(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
    seed: int,
) -> plt.Figure:
    indices = _representative_indices(records)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for axis, family in zip(axes, ("step", "pulse"), strict=True):
        index = indices[family]
        spectra = {}
        maximum_q = 0.0
        for arm in ARMS:
            error = (
                arrays[f"s{seed}__native__{arm}"][index] - records[index].case.increment
            )
            q, energy = physical_cosine_spectrum(error, grid)
            spectra[arm] = (q, energy)
            maximum_q = max(maximum_q, float(np.max(q)))
        bins = np.linspace(0.0, maximum_q, 33)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for arm in ARMS:
            q, energy = spectra[arm]
            binned, _ = np.histogram(q, bins=bins, weights=energy)
            axis.plot(
                centers, binned, color=COLORS[arm], linewidth=0.9, label=LABELS[arm]
            )
        axis.set_yscale("log")
        axis.set_xlabel("Physical wavenumber (cycles / unit length)")
        axis.set_title(f"{family.title()} error spectrum")
    axes[0].set_ylabel("Physical DCT error energy per radial bin")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    return fig


def run_analysis(
    matrix_dir: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    smoke: bool,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if not smoke and device.type != "cuda":
        raise RuntimeError("the production P3 frozen-screen contract requires CUDA")
    matrix_dir = matrix_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "running"},
    )

    all_runs = _load_full_runs(matrix_dir, smoke=smoke)
    identities = _verify_registered_inputs(all_runs, smoke=smoke)
    seeds = (SEEDS[0],) if smoke else SEEDS
    contract = {
        "schema": SCHEMA,
        "working_run_id": WORKING_ID + ("-SMOKE" if smoke else ""),
        "status": "running",
        "science_result_eligible": not smoke,
        "optimization_performed": False,
        "selected_layer": SELECTED_LAYER,
        "zero_arm": "native output computed once and reused; no transform or replay",
        "arms": list(ARMS),
        "filter": {
            "transform": "orthonormal DCT-II on [ny,nx] cell centers",
            "physical_wavenumber_units": "cycles_per_unit_length",
            "smooth_pass": 8.0,
            "smooth_stop": 16.0,
            "hard_cutoff": 8.0,
            "fixed_physical_radius": 1.0 / 32.0,
        },
        "checkpoint_population_sha256": P2_POPULATION_SHA256,
        "checkpoint_trainer_sha256": P2_TRAINER_SHA256,
        "inputs": identities,
        "source_sha256": _source_hashes(SOURCE_PATHS),
        "provenance_sha256": _source_hashes(PROVENANCE_PATHS),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
    }
    write_json(output_dir / "run_contract.json", contract)
    started = time.perf_counter()
    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    closures: dict[str, dict[str, float]] = {}
    figure_records = None
    figure_grid = None

    for seed in seeds:
        source_run = all_runs[("full", seed)]
        config = source_run["contract"]["config"]
        checkpoint = torch.load(
            source_run["run_dir"] / "checkpoint.pt",
            map_location=device,
            weights_only=False,
        )
        if checkpoint["schema"] != source_run["manifest"]["schema"]:
            raise ValueError(f"full_s{seed} checkpoint schema differs")
        if checkpoint["config_digest"] != source_run["manifest"]["config_digest"]:
            raise ValueError(f"full_s{seed} checkpoint config digest differs")
        model = build_model(config, device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        state_hash = model_state_sha256(model)
        if state_hash != source_run["summary"]["final_model_state_sha256"]:
            raise ValueError(f"full_s{seed} model-state hash differs")
        model.eval()
        run_name = f"s{seed}"
        results[run_name] = {}
        closures[run_name] = {}
        for population_name, grid, records in _registered_populations(
            config, smoke=smoke
        ):
            if population_name == "native" and figure_records is None:
                figure_records, figure_grid = records, grid
            population_predictions, replay_error = _predict_population(
                model,
                records,
                grid,
                device,
                batch_size=batch_size,
            )
            closures[run_name][population_name] = replay_error
            if replay_error > 1.0e-4:
                raise ValueError(
                    f"{run_name}/{population_name} native replay exceeds 1e-4: {replay_error}"
                )
            for arm, prediction in population_predictions.items():
                arrays[f"{run_name}__{population_name}__{arm}"] = prediction
                rows = _case_rows(records, prediction, grid, arm=arm)
                arm_run = results[run_name].setdefault(
                    arm,
                    {"native_rows": [], "nested_rows": [], "populations": {}},
                )
                destination = (
                    "native_rows" if population_name == "native" else "nested_rows"
                )
                arm_run[destination].extend(rows)
                arm_run["populations"][population_name] = {
                    "case_count": len(rows),
                    "aggregates": _mask_aggregates(rows),
                }
            print(
                json.dumps(
                    {
                        "stage": "population_complete",
                        "seed": seed,
                        "population": population_name,
                        "native_replay_max_abs": replay_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if model_state_sha256(model) != state_hash:
            raise RuntimeError(f"full_s{seed} frozen screen mutated the model")
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    maximum_replay = max(value for row in closures.values() for value in row.values())
    strict_replay_pass = maximum_replay <= 1.0e-5
    decision = build_decision(results, seeds, strict_replay_pass=strict_replay_pass)
    write_json(output_dir / "case_metrics.json", results)
    write_json(output_dir / "replay_closures.json", closures)
    write_json(output_dir / "decision_summary.json", decision)
    write_npz(output_dir / "predictions.npz", **arrays)

    if figure_records is None or figure_grid is None:
        raise RuntimeError("native figure population was not constructed")
    _style()
    figures = []
    figures += _save(_primary_figure(decision, seeds), output_dir, "fig_primary_ratios")
    figures += _save(
        _structure_figure(results, seeds), output_dir, "fig_structure_tradeoffs"
    )
    figures += _save(
        _profile_figure(arrays, figure_records, figure_grid, seeds[0]),
        output_dir,
        "fig_representative_profiles",
    )
    figures += _save(
        _error_map_figure(arrays, figure_records, figure_grid, seeds[0]),
        output_dir,
        "fig_representative_error_maps",
    )
    figures += _save(
        _spectrum_figure(arrays, figure_records, figure_grid, seeds[0]),
        output_dir,
        "fig_physical_error_spectra",
    )
    visual_manifest = {
        "schema": SCHEMA,
        "files": [path.name for path in figures],
        "sha256": {path.name: sha256_file(path) for path in figures},
    }
    write_json(output_dir / "visual_manifest.json", visual_manifest)

    summary = {
        "schema": SCHEMA,
        "working_run_id": contract["working_run_id"],
        "status": "completed",
        "science_result": bool(not smoke and strict_replay_pass),
        "optimization_performed": False,
        "checkpoint_count": len(seeds),
        "arm_count_including_zero": len(ARMS),
        "maximum_native_replay_max_abs": maximum_replay,
        "strict_1e_5_replay_pass": strict_replay_pass,
        "descriptive_1e_4_compatibility_pass": maximum_replay <= 1.0e-4,
        "zero_bypass_recomputed": False,
        "decision": decision,
        "visualizations": visual_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "claim_boundary": (
            "Frozen one-step synthetic checkpoint interventions test causal headroom at "
            "one preselected PCNO block. They are not training results, recurrent "
            "results, strict Gibbs identification, a limiter, or proof of operator "
            "convergence. Lower spectral energy without the registered front, TV, "
            "integral, and smooth controls is not a pass."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "completed"},
    )
    outputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": contract["working_run_id"],
        "status": "completed",
        "science_result": summary["science_result"],
        "source_sha256": contract["source_sha256"],
        "provenance_sha256": contract["provenance_sha256"],
        "input_manifest_sha256": {
            name: value["manifest_sha256"] for name, value in identities.items()
        },
        "input_checkpoint_sha256": {
            name: value["checkpoint_sha256"] for name, value in identities.items()
        },
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        args.matrix_dir,
        args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
