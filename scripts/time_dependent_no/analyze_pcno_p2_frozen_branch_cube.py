#!/usr/bin/env python3
"""Run the inference-only W26-L2-P2-F branch cube on all nine P2 models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from itertools import product
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

from scripts.time_dependent_no.analyze_pcno_gradient_ablation import _load_matrix
from scripts.time_dependent_no.analyze_pcno_shock_pathways import (
    _factorial_effects,
    _mask_aggregates,
    _metric_scalars,
    build_common_physical_cases,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
    build_batch,
    build_model,
    build_registered_cases,
    case_metric_rows,
    model_state_sha256,
    sha256_file,
    write_json,
    write_npz,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    SEEDS,
    VARIANTS,
    _predict_batched,
    _primary_metric,
    apply_variant,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import trace_pcno_branches
from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    FRONT_BAND_WIDTH,
    PULSE_WIDTH,
    StructuredCellGrid,
    build_structured_cell_grid,
    physical_cosine_spectrum,
)

SCHEMA = "w26_l2_p2_frozen_branch_cube_v1"
WORKING_ID = "W26-L2-P2-F"
BRANCHES = ("spectral", "pointwise", "differential")
BRANCH_MASKS = tuple(product((0, 1), repeat=3))
DEFAULT_MATRIX = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2_frozen_cube"
P2_POPULATION_SHA256 = (
    "0104edcdd1078b3b175647e880739081afd49217f0bc7efed773b39ed422c358"
)
P2_TRAINER_SHA256 = (
    "8c638b69987e68c5c7ecd4a1d3705de1f006e5824bd510d99fcfdfcab37a43c9"
)
P2_REGISTRY: Mapping[str, Mapping[str, str]] = {
    "full_s1701": {
        "manifest": "adfbd224d36fb63df249e69b8a672c8e79f7cefa8eef78491ecde02d83dd92ba",
        "checkpoint": "05c2b6d0fd185b0285fa663bb3625fa2d05cb3137b06f79a05f21402d930f71e",
    },
    "full_s1702": {
        "manifest": "f21899b8ef8741af14de91dbd7f4cc9b23ab32e6767a98e56dc5785fe705cf3b",
        "checkpoint": "c788a45a3b3a469da5255163b8bef8492aa2775515663a2fbf1080e238921d2f",
    },
    "full_s1703": {
        "manifest": "cbf5721e11835a2cce3e9ee7fc5f80b8ee5cfd39dcbc897d754259d6d8f1ecf2",
        "checkpoint": "5bee79de1e08d150b94e9bf4abfb87e52b3ec24d04ccc4d4ae15f4cd9c735eaf",
    },
    "local_replacement_s1701": {
        "manifest": "07fb4a87ad6ad1db81fb9c3bacd9259737c5dfc8b39a865c8eaf97a5a93c817f",
        "checkpoint": "574f2c67479e9528c0cd90afbfd459c4d279f1ae78ac5a9bfb964761eea89df8",
    },
    "local_replacement_s1702": {
        "manifest": "1d1fab1b687824fa3f9df13113f4f97536f77ffc47a88e4362a846fbe248fbaf",
        "checkpoint": "e00632c9bbe86b7465a9907d05a8d8bd3c465fd06759ebfe495111c3eac83ec4",
    },
    "local_replacement_s1703": {
        "manifest": "c9ddfa47e75d6db1f96fa2f22e5385442bb5a9cf60fa86239e2a137970d26cfa",
        "checkpoint": "746a5b7eb9a762f002204a2b4fc227e74f06843caba9533645480dd6c0148bb5",
    },
    "no_gradient_s1701": {
        "manifest": "6b763ff580502b3978b2c16051be796ea70d497e52bc93e876bb4617b00f55a7",
        "checkpoint": "a383bbddb4c09914922010d76d7db634925c7819b5293de0eca100e635bd5f13",
    },
    "no_gradient_s1702": {
        "manifest": "3cebb8f6ae1be44f1c0ffbbdb03db02552422c6028634c402a39c1fb615dd81d",
        "checkpoint": "4f5f738be3628d1ec3149fb5ac7e46f7ba557a811d8e0cd56c10e61b314b50bd",
    },
    "no_gradient_s1703": {
        "manifest": "73f1d20920793a08a50d8e0de4abdcc2195261306b8ff6a4b6cf4cd7afebef93",
        "checkpoint": "b83f78692f8b6fa2f76b30f53cdf920b5bffeeade48102e1fc6f76b8ca6363bc",
    },
}
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "scripts/time_dependent_no/train_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_p2_frozen_branch_cube.py",
)
PROVENANCE_PATHS = (
    "docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",
)
COLORS = {
    "full": "#0072B2",
    "no_gradient": "#D55E00",
    "local_replacement": "#009E73",
}
LABELS = {
    "full": "Full PCNO",
    "no_gradient": "No gradient",
    "local_replacement": "Matched local",
}


def branch_mask_name(mask: Sequence[int]) -> str:
    if len(mask) != 3 or any(value not in (0, 1) for value in mask):
        raise ValueError("branch mask must contain three binary values")
    return "".join(str(int(value)) for value in mask)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _third_slot_label(variant: str) -> str:
    return {
        "full": "learned_differential",
        "no_gradient": "functional_exact_zero",
        "local_replacement": "parameter_matched_two_hop_local",
    }[variant]


def _feature_positions(record: RegisteredCase) -> tuple[float, ...]:
    if record.family in {"step", "smooth_tanh"}:
        return (record.position, record.position + DISPLACEMENT)
    if record.family == "pulse":
        return (
            record.position,
            record.position + DISPLACEMENT,
            record.position + PULSE_WIDTH,
            record.position + DISPLACEMENT + PULSE_WIDTH,
        )
    return ()


def _registered_populations(
    config: Mapping[str, Any], *, smoke: bool
) -> list[tuple[str, StructuredCellGrid, list[RegisteredCase]]]:
    native_grid = build_structured_cell_grid(config["resolution"])
    native_records = build_registered_cases(native_grid, config, split="train")
    native_records += build_registered_cases(native_grid, config, split="held")
    populations = [("native", native_grid, native_records)]
    if smoke:
        return populations
    for resolution in config["evaluation_resolutions"]:
        if tuple(resolution) == tuple(config["resolution"]):
            continue
        grid = build_structured_cell_grid(resolution)
        populations.append(
            (
                f"{grid.nx}x{grid.ny}",
                grid,
                build_common_physical_cases(grid, config, split="held"),
            )
        )
    return populations


@torch.inference_mode()
def _predict_with_mask(
    model: torch.nn.Module,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    mask: Sequence[int],
    batch_size: int,
) -> np.ndarray:
    gains = {
        (layer, branch): float(enabled)
        for layer in range(len(model.ws))
        for branch, enabled in zip(BRANCHES, mask, strict=True)
    }
    predictions = []
    model.eval()
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        model_input, _, aux = build_batch(selected, grid, device)
        output, _ = trace_pcno_branches(
            model,
            model_input,
            aux,
            branch_gains=gains,
            collect_summaries=False,
        )
        predictions.append(
            output[..., 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(len(selected), grid.ny, grid.nx)
        )
    return np.concatenate(predictions, axis=0)


def _case_rows(
    records: list[RegisteredCase],
    predictions: np.ndarray,
    grid: StructuredCellGrid,
    *,
    mask: str,
) -> list[dict[str, Any]]:
    rows = case_metric_rows(records, predictions, grid)
    for row in rows:
        row["resolution"] = list(grid.resolution)
        row["mask"] = mask
        row["scalars"] = _metric_scalars(row["metrics"])
    return rows


def _selected_indices(records: Sequence[RegisteredCase]) -> list[int]:
    selected = [
        index
        for index, record in enumerate(records)
        if record.split == "held"
        and record.family in {"step", "pulse"}
        and math.isclose(float(record.phase), 0.875)
    ]
    if selected:
        return selected
    return [
        index
        for index, record in enumerate(records)
        if record.split == "held" and record.family in {"step", "pulse"}
    ]


def _outside_front_mask(record: RegisteredCase, grid: StructuredCellGrid) -> np.ndarray:
    outside_x = np.ones(grid.nx, dtype=bool)
    for position in _feature_positions(record):
        outside_x &= np.abs(grid.x_centers - position) > FRONT_BAND_WIDTH
    return np.repeat(outside_x[None, :], grid.ny, axis=0)


def _mean_metric(
    aggregates: Mapping[str, Any], group: str, metric: str
) -> float:
    if (
        group == "held_discontinuous_phase_0p875"
        and aggregates[group]["case_count"] == 0
    ):
        group = "held_discontinuous"
    value = aggregates[group]["mean"][metric]
    if value is None:
        raise ValueError(f"missing aggregate {group}/{metric}")
    return float(value)


def _acute_deletion_rows(
    *,
    variant: str,
    seed: int,
    population_name: str,
    grid: StructuredCellGrid,
    records: list[RegisteredCase],
    predictions: Mapping[str, np.ndarray],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    full = predictions["111"]
    target = np.stack([record.case.increment for record in records])
    selected = _selected_indices(records)
    rows = []
    off_masks = {"spectral": "011", "pointwise": "101", "differential": "110"}
    for branch, off_name in off_masks.items():
        off = predictions[off_name]
        contribution = full - off
        cosines = []
        off_energies = []
        full_energies = []
        contribution_energies = []
        for index in selected:
            outside = _outside_front_mask(records[index], grid)
            off_error = (off[index] - target[index])[outside].astype(np.float64)
            full_error = (full[index] - target[index])[outside].astype(np.float64)
            decoded_addition = contribution[index][outside].astype(np.float64)
            denominator = float(
                np.linalg.norm(off_error) * np.linalg.norm(decoded_addition)
            )
            if denominator > 0.0:
                cosines.append(float(np.dot(off_error, decoded_addition) / denominator))
            volume = grid.hx * grid.hy
            off_energies.append(float(np.sum(np.square(off_error)) * volume))
            full_energies.append(float(np.sum(np.square(full_error)) * volume))
            contribution_energies.append(
                float(np.sum(np.square(decoded_addition)) * volume)
            )
        full_l2 = _mean_metric(
            aggregates["111"],
            "held_discontinuous_phase_0p875",
            "relative_increment_l2",
        )
        off_l2 = _mean_metric(
            aggregates[off_name],
            "held_discontinuous_phase_0p875",
            "relative_increment_l2",
        )
        full_ripple = _mean_metric(
            aggregates["111"],
            "held_discontinuous_phase_0p875",
            "oscillatory_mass_outside_front_band",
        )
        off_ripple = _mean_metric(
            aggregates[off_name],
            "held_discontinuous_phase_0p875",
            "oscillatory_mass_outside_front_band",
        )
        energy_ratio = float(
            np.mean(full_energies)
            / max(float(np.mean(off_energies)), np.finfo(float).tiny)
        )
        ripple_ratio = float(full_ripple / max(off_ripple, np.finfo(float).tiny))
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "population": population_name,
                "branch_slot": branch,
                "branch_semantics": (
                    _third_slot_label(variant)
                    if branch == "differential"
                    else branch
                ),
                "off_mask": off_name,
                "selected_case_count": len(selected),
                "mean_outside_error_contribution_cosine": (
                    float(np.mean(cosines)) if cosines else None
                ),
                "mean_outside_decoded_contribution_energy": float(
                    np.mean(contribution_energies)
                ),
                "mean_outside_error_energy_off": float(np.mean(off_energies)),
                "mean_outside_error_energy_full": float(np.mean(full_energies)),
                "outside_error_energy_ratio_full_to_off": energy_ratio,
                "relative_increment_l2_full": full_l2,
                "relative_increment_l2_off": off_l2,
                "relative_increment_l2_ratio_full_to_off": float(
                    full_l2 / max(off_l2, np.finfo(float).tiny)
                ),
                "oscillatory_mass_full": full_ripple,
                "oscillatory_mass_off": off_ripple,
                "oscillatory_mass_ratio_full_to_off": ripple_ratio,
                "cancellation_supported": bool(
                    cosines
                    and float(np.mean(cosines)) < 0.0
                    and energy_ratio < 1.0
                    and ripple_ratio < 1.0
                ),
            }
        )
    return rows


def _verify_registered_inputs(
    runs: Mapping[tuple[str, int], Mapping[str, Any]], *, smoke: bool
) -> dict[str, Any]:
    identities = {}
    for (variant, seed), run in runs.items():
        name = f"{variant}_s{seed}"
        manifest_sha256 = sha256_file(run["run_dir"] / "manifest.json")
        checkpoint_sha256 = run["manifest"]["output_hashes"]["checkpoint.pt"]
        if not smoke:
            expected = P2_REGISTRY[name]
            if manifest_sha256 != expected["manifest"]:
                raise ValueError(f"{name} manifest differs from the P2-F registry")
            if checkpoint_sha256 != expected["checkpoint"]:
                raise ValueError(f"{name} checkpoint differs from the P2-F registry")
        identities[name] = {
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "model_state_sha256": run["summary"]["final_model_state_sha256"],
        }
    sample = next(iter(runs.values()))
    if not smoke:
        if sample["manifest"]["population_sha256"] != P2_POPULATION_SHA256:
            raise ValueError("P2 population differs from the P2-F registry")
        trainer_hash = sample["manifest"]["source_sha256"].get(
            "scripts/time_dependent_no/train_pcno_gradient_ablation.py"
        )
        if trainer_hash != P2_TRAINER_SHA256:
            raise ValueError("P2 trainer differs from the P2-F registry")
    return identities


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
    fig.savefig(pdf, metadata={"Creator": "W26-L2-P2-F", "CreationDate": None})
    fig.savefig(png, dpi=300, metadata={"Software": "W26-L2-P2-F"})
    plt.close(fig)
    return [pdf, png]


def _aggregate_value(
    cube: Mapping[str, Any],
    variant: str,
    seed: int,
    population: str,
    mask: str,
    metric: str,
) -> float:
    return _mean_metric(
        cube[f"{variant}_s{seed}"]["populations"][population][mask]["aggregates"],
        "held_discontinuous_phase_0p875",
        metric,
    )


def _factorial_figure(cube: Mapping[str, Any], seeds: Sequence[int]) -> plt.Figure:
    masks = [branch_mask_name(mask) for mask in BRANCH_MASKS]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.4), constrained_layout=True)
    metrics = (
        ("relative_increment_l2", "Held phase-0.875 relative increment L2"),
        (
            "oscillatory_mass_outside_front_band",
            "Held phase-0.875 outside-band oscillatory mass",
        ),
    )
    for row, variant in enumerate(VARIANTS):
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            values = np.asarray(
                [
                    [
                        _aggregate_value(cube, variant, seed, "native", mask, metric)
                        for seed in seeds
                    ]
                    for mask in masks
                ]
            )
            axis.bar(
                np.arange(len(masks)),
                np.mean(values, axis=1),
                yerr=np.std(values, axis=1, ddof=1) if len(seeds) > 1 else None,
                color=COLORS[variant],
                alpha=0.82,
                capsize=2,
            )
            if bool(np.all(np.mean(values, axis=1) > 0.0)):
                axis.set_yscale("log")
            axis.set_xticks(np.arange(len(masks)), masks, rotation=45, ha="right")
            axis.set_title(f"{LABELS[variant]}: {title}")
    fig.suptitle("Frozen branch cube; bits are spectral, pointwise, third path")
    return fig


def _deletion_figure(cube: Mapping[str, Any], seeds: Sequence[int]) -> plt.Figure:
    branches = ("spectral", "pointwise", "differential")
    off_masks = {"spectral": "011", "pointwise": "101", "differential": "110"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    width = 0.24
    for variant_index, variant in enumerate(VARIANTS):
        l2_ratios = []
        ripple_ratios = []
        l2_std = []
        ripple_std = []
        for branch in branches:
            off_name = off_masks[branch]
            per_seed_l2 = []
            per_seed_ripple = []
            for seed in seeds:
                full_l2 = _aggregate_value(
                    cube, variant, seed, "native", "111", "relative_increment_l2"
                )
                off_l2 = _aggregate_value(
                    cube, variant, seed, "native", off_name, "relative_increment_l2"
                )
                full_ripple = _aggregate_value(
                    cube,
                    variant,
                    seed,
                    "native",
                    "111",
                    "oscillatory_mass_outside_front_band",
                )
                off_ripple = _aggregate_value(
                    cube,
                    variant,
                    seed,
                    "native",
                    off_name,
                    "oscillatory_mass_outside_front_band",
                )
                per_seed_l2.append(off_l2 / max(full_l2, np.finfo(float).tiny))
                per_seed_ripple.append(
                    off_ripple / max(full_ripple, np.finfo(float).tiny)
                )
            l2_ratios.append(np.mean(per_seed_l2))
            ripple_ratios.append(np.mean(per_seed_ripple))
            l2_std.append(np.std(per_seed_l2, ddof=1) if len(seeds) > 1 else 0.0)
            ripple_std.append(
                np.std(per_seed_ripple, ddof=1) if len(seeds) > 1 else 0.0
            )
        positions = np.arange(3) + (variant_index - 1) * width
        axes[0].bar(
            positions,
            l2_ratios,
            width,
            yerr=l2_std,
            color=COLORS[variant],
            label=LABELS[variant],
            capsize=2,
        )
        axes[1].bar(
            positions,
            ripple_ratios,
            width,
            yerr=ripple_std,
            color=COLORS[variant],
            label=LABELS[variant],
            capsize=2,
        )
    for axis, title in zip(
        axes,
        ("Branch-off / all-on L2", "Branch-off / all-on oscillatory mass"),
        strict=True,
    ):
        axis.axhline(1.0, color="#333333", linewidth=0.8)
        axis.set_xticks(range(3), ("Spectral", "Pointwise", "Third path"))
        axis.set_yscale("log")
        axis.set_title(title)
    axes[0].legend(frameon=False, fontsize=7)
    return fig


def _cancellation_figure(rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    branches = ("spectral", "pointwise", "differential")
    width = 0.24
    for variant_index, variant in enumerate(VARIANTS):
        cosines = []
        ratios = []
        cosine_std = []
        ratio_std = []
        for branch in branches:
            selected = [
                row
                for row in rows
                if row["variant"] == variant
                and row["population"] == "native"
                and row["branch_slot"] == branch
            ]
            finite_cosines = [
                float(row["mean_outside_error_contribution_cosine"])
                for row in selected
                if row["mean_outside_error_contribution_cosine"] is not None
            ]
            values = [float(row["outside_error_energy_ratio_full_to_off"]) for row in selected]
            cosines.append(float(np.mean(finite_cosines)) if finite_cosines else 0.0)
            ratios.append(float(np.mean(values)))
            cosine_std.append(
                float(np.std(finite_cosines, ddof=1)) if len(finite_cosines) > 1 else 0.0
            )
            ratio_std.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        positions = np.arange(3) + (variant_index - 1) * width
        axes[0].bar(
            positions,
            cosines,
            width,
            yerr=cosine_std,
            color=COLORS[variant],
            label=LABELS[variant],
            capsize=2,
        )
        axes[1].bar(
            positions,
            ratios,
            width,
            yerr=ratio_std,
            color=COLORS[variant],
            label=LABELS[variant],
            capsize=2,
        )
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].axhline(1.0, color="#333333", linewidth=0.8)
    for axis in axes:
        axis.set_xticks(range(3), ("Spectral", "Pointwise", "Third path"))
    axes[0].set_title("cos(branch-off error, decoded addition)")
    axes[1].set_title("All-on / branch-off outside error energy")
    axes[0].legend(frameon=False, fontsize=7)
    return fig


def _representative_case_index(records: Sequence[RegisteredCase]) -> int:
    for index, record in enumerate(records):
        if (
            record.split == "held"
            and record.family == "step"
            and record.anchor_index == 1
            and math.isclose(float(record.phase), 0.875)
        ):
            return index
    for index, record in enumerate(records):
        if record.split == "held" and record.family == "step":
            return index
    raise ValueError("a representative held step was not found")


def _profile_figure(
    cube: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
) -> plt.Figure:
    seed = 1702 if "full_s1702__native__111" in arrays else SEEDS[0]
    index = _representative_case_index(records)
    target = records[index].case.increment.mean(axis=0)
    masks = ("111", "011", "101", "110")
    titles = ("All on", "Spectral off", "Pointwise off", "Third path off")
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.0), sharex=True, constrained_layout=True)
    for row, variant in enumerate(VARIANTS):
        for column, (mask, title) in enumerate(zip(masks, titles, strict=True)):
            axis = axes[row, column]
            prediction = arrays[f"{variant}_s{seed}__native__{mask}"][index].mean(axis=0)
            axis.plot(grid.x_centers, target, color="#222222", linewidth=1.2, label="Target")
            axis.plot(
                grid.x_centers,
                prediction,
                color=COLORS[variant],
                linewidth=1.0,
                label="Prediction",
            )
            for position in _feature_positions(records[index]):
                axis.axvspan(
                    position - FRONT_BAND_WIDTH,
                    position + FRONT_BAND_WIDTH,
                    color="#BBBBBB",
                    alpha=0.18,
                    linewidth=0,
                )
            axis.set_xlim(0.2, 0.8)
            axis.set_title(f"{LABELS[variant]}: {title}")
    axes[0, 0].legend(frameon=False, fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("x")
    fig.suptitle("Fixed held subcell phase, seed 1702; shaded bands are physical front bands")
    return fig


def _error_map_figure(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
) -> plt.Figure:
    seed = 1702 if "full_s1702__native__111" in arrays else SEEDS[0]
    index = _representative_case_index(records)
    target = records[index].case.increment
    masks = ("111", "011", "101", "110")
    titles = ("All on", "Spectral off", "Pointwise off", "Third path off")
    errors = [
        arrays[f"{variant}_s{seed}__native__{mask}"][index] - target
        for variant in VARIANTS
        for mask in masks
    ]
    limit = max(float(np.max(np.abs(error))) for error in errors)
    fig, axes = plt.subplots(3, 4, figsize=(10.8, 6.6), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for row, variant in enumerate(VARIANTS):
        for column, (mask, title) in enumerate(zip(masks, titles, strict=True)):
            error = arrays[f"{variant}_s{seed}__native__{mask}"][index] - target
            image = axes[row, column].imshow(
                error,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 0.5),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set_title(f"{LABELS[variant]}: {title}")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Signed increment error", shrink=0.8)
    fig.suptitle("Held phase-0.875 error maps with one common color scale")
    return fig


def _resolution_figure(
    cube: Mapping[str, Any], seeds: Sequence[int]
) -> plt.Figure:
    sample = cube[f"{VARIANTS[0]}_s{seeds[0]}"]["populations"]
    populations = sorted(
        sample,
        key=lambda name: 64 if name == "native" else int(name.split("x")[0]),
    )
    nx_values = [64 if name == "native" else int(name.split("x")[0]) for name in populations]
    masks = ("111", "011", "101", "110")
    labels = ("All on", "S off", "P off", "Third off")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    for axis, variant in zip(axes, VARIANTS, strict=True):
        for mask, label in zip(masks, labels, strict=True):
            values = []
            deviations = []
            for population in populations:
                per_seed = [
                    _aggregate_value(
                        cube,
                        variant,
                        seed,
                        population,
                        mask,
                        "relative_increment_l2",
                    )
                    for seed in seeds
                ]
                values.append(float(np.mean(per_seed)))
                deviations.append(
                    float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0
                )
            axis.errorbar(nx_values, values, yerr=deviations, marker="o", label=label, capsize=2)
        axis.set_yscale("log")
        axis.set_xticks(nx_values)
        axis.set_title(LABELS[variant])
        axis.set_xlabel("nx (fixed physical cases)")
    axes[0].set_ylabel("Held discontinuous relative increment L2")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Acute branch dependence across the registered nested grids")
    return fig


def _spectrum_figure(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[RegisteredCase],
    grid: StructuredCellGrid,
    seeds: Sequence[int],
) -> plt.Figure:
    selected = _selected_indices(records)
    q, _ = physical_cosine_spectrum(np.zeros(grid.array_shape), grid)
    edges = np.arange(0.0, math.ceil(float(np.max(q))) + 2.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    masks = ("111", "011", "101", "110")
    labels = ("All on", "S off", "P off", "Third off")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    target = np.stack([record.case.increment for record in records])
    for axis, variant in zip(axes, VARIANTS, strict=True):
        for mask, label in zip(masks, labels, strict=True):
            curves = []
            for seed in seeds:
                prediction = arrays[f"{variant}_s{seed}__native__{mask}"]
                for index in selected:
                    physical_q, energy = physical_cosine_spectrum(
                        prediction[index] - target[index], grid
                    )
                    curves.append(np.histogram(physical_q, bins=edges, weights=energy)[0])
            mean = np.mean(curves, axis=0)
            floor = max(float(np.max(mean)) * 1.0e-12, np.finfo(float).tiny)
            axis.plot(centers, np.maximum(mean, floor), label=label)
        axis.set_yscale("log")
        axis.set_title(LABELS[variant])
        axis.set_xlabel("Physical cycles per unit length")
    axes[0].set_ylabel("Physical error energy")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Held phase-0.875 physical spectra; energy alone is not ripple")
    return fig


def _decision_summary(
    cube: Mapping[str, Any], deletion_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    off_masks = {"spectral": "011", "pointwise": "101", "differential": "110"}
    variants = {}
    for variant in VARIANTS:
        branch_rows = {}
        for branch, off_mask in off_masks.items():
            l2_ratios = []
            ripple_ratios = []
            cancellation = []
            for seed in seeds:
                full_l2 = _aggregate_value(
                    cube, variant, seed, "native", "111", "relative_increment_l2"
                )
                off_l2 = _aggregate_value(
                    cube, variant, seed, "native", off_mask, "relative_increment_l2"
                )
                full_ripple = _aggregate_value(
                    cube,
                    variant,
                    seed,
                    "native",
                    "111",
                    "oscillatory_mass_outside_front_band",
                )
                off_ripple = _aggregate_value(
                    cube,
                    variant,
                    seed,
                    "native",
                    off_mask,
                    "oscillatory_mass_outside_front_band",
                )
                l2_ratios.append(off_l2 / max(full_l2, np.finfo(float).tiny))
                ripple_ratios.append(
                    off_ripple / max(full_ripple, np.finfo(float).tiny)
                )
                deletion = next(
                    row
                    for row in deletion_rows
                    if row["variant"] == variant
                    and row["seed"] == seed
                    and row["population"] == "native"
                    and row["branch_slot"] == branch
                )
                cancellation.append(bool(deletion["cancellation_supported"]))
            branch_rows[branch] = {
                "semantics": _third_slot_label(variant) if branch == "differential" else branch,
                "branch_off_to_all_on_l2_ratios": l2_ratios,
                "median_branch_off_to_all_on_l2_ratio": float(statistics.median(l2_ratios)),
                "branch_off_to_all_on_oscillatory_mass_ratios": ripple_ratios,
                "median_branch_off_to_all_on_oscillatory_mass_ratio": float(
                    statistics.median(ripple_ratios)
                ),
                "cancellation_supported_seed_count": sum(cancellation),
            }
        variants[variant] = branch_rows
    return {
        "variants": variants,
        "interpretation_rules": {
            "essential_acute_capacity": "branch-off L2 ratio materially above one across seeds",
            "decoded_cancellation": (
                "negative branch-off-error/addition cosine plus lower all-on outside "
                "error energy and lower registered outside-band oscillatory mass"
            ),
            "ripple_boundary": (
                "a shock-local bump or high-frequency energy alone is not Gibbs or ripple"
            ),
            "causal_boundary": (
                "frozen masks are acute non-additive counterfactuals and do not equal "
                "independently trained architectures or recurrent causality"
            ),
        },
    }


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
        raise RuntimeError("the production P2-F contract requires CUDA")
    matrix_dir = matrix_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "running"},
    )
    seeds = (SEEDS[0],) if smoke else SEEDS
    runs = _load_matrix(matrix_dir, smoke=smoke)
    identities = _verify_registered_inputs(runs, smoke=smoke)
    source_hashes = _source_hashes(SOURCE_PATHS)
    provenance_hashes = _source_hashes(PROVENANCE_PATHS)
    run_contract = {
        "schema": SCHEMA,
        "working_run_id": WORKING_ID + ("-SMOKE" if smoke else ""),
        "status": "running",
        "science_result_eligible": not smoke,
        "optimization_performed": False,
        "checkpoint_count": len(runs),
        "variants": list(VARIANTS),
        "seeds": list(seeds),
        "branch_bit_order": list(BRANCHES),
        "branch_masks": [branch_mask_name(mask) for mask in BRANCH_MASKS],
        "third_slot_semantics": {
            variant: _third_slot_label(variant) for variant in VARIANTS
        },
        "population": {
            "native": "registered P2 train plus held one-step cases",
            "nested": "registered common-physical held one-step cases",
            "recurrent": False,
            "sealed_population": False,
            "population_sha256": next(iter(runs.values()))["manifest"][
                "population_sha256"
            ],
        },
        "closures": {
            "all_manifest_outputs_sha256_verified": True,
            "loaded_model_state_matches_summary": True,
            "registered_all_on_trace_to_direct_max_abs_tolerance": 1.0e-5,
            "descriptive_compatibility_max_abs_tolerance": 1.0e-4,
            "mask_111_uses_native_direct_bypass": True,
            "functional_no_gradient_third_bit_max_abs_tolerance": 0.0,
        },
        "inputs": identities,
        "source_sha256": source_hashes,
        "provenance_sha256": provenance_hashes,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    write_json(output_dir / "run_contract.json", run_contract)

    started = time.perf_counter()
    cube: dict[str, Any] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    deletion_rows: list[dict[str, Any]] = []
    replay_closures: dict[str, dict[str, float]] = {}
    no_gradient_invariance: dict[str, dict[str, float]] = {}
    native_records_for_figures: list[RegisteredCase] | None = None
    native_grid_for_figures: StructuredCellGrid | None = None

    for seed in seeds:
        for variant in VARIANTS:
            run_name = f"{variant}_s{seed}"
            run = runs[(variant, seed)]
            config = run["contract"]["config"]
            checkpoint = torch.load(
                run["run_dir"] / "checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
            if checkpoint["schema"] != run["manifest"]["schema"]:
                raise ValueError(f"{run_name} checkpoint schema differs")
            if checkpoint["config_digest"] != run["manifest"]["config_digest"]:
                raise ValueError(f"{run_name} checkpoint config digest differs")
            model = apply_variant(build_model(config, device), variant)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            loaded_state = model_state_sha256(model)
            if loaded_state != run["summary"]["final_model_state_sha256"]:
                raise ValueError(f"{run_name} loaded model state does not close")
            model.eval()
            populations = _registered_populations(config, smoke=smoke)
            cube[run_name] = {
                "variant": variant,
                "seed": seed,
                "model_state_sha256": loaded_state,
                "third_slot_semantics": _third_slot_label(variant),
                "populations": {},
            }
            replay_closures[run_name] = {}
            no_gradient_invariance[run_name] = {}
            for population_name, grid, records in populations:
                if population_name == "native" and native_records_for_figures is None:
                    native_records_for_figures = records
                    native_grid_for_figures = grid
                direct = _predict_batched(
                    model, records, grid, device, batch_size=batch_size
                )
                predictions: dict[str, np.ndarray] = {}
                trace_predictions: dict[str, np.ndarray] = {}
                population_metrics = {}
                for mask in BRANCH_MASKS:
                    mask_name = branch_mask_name(mask)
                    traced = _predict_with_mask(
                        model,
                        records,
                        grid,
                        device,
                        mask=mask,
                        batch_size=batch_size,
                    )
                    trace_predictions[mask_name] = traced
                    prediction = direct if mask_name == "111" else traced
                    predictions[mask_name] = prediction
                    prediction_arrays[
                        f"{run_name}__{population_name}__{mask_name}"
                    ] = prediction
                    rows = _case_rows(
                        records, prediction, grid, mask=mask_name
                    )
                    population_metrics[mask_name] = {
                        "mask": list(mask),
                        "primary_value": _primary_metric(rows),
                        "aggregates": _mask_aggregates(rows),
                        "case_metrics": rows,
                    }
                replay = float(np.max(np.abs(trace_predictions["111"] - direct)))
                replay_relative_l2 = float(
                    np.linalg.norm(
                        trace_predictions["111"].astype(np.float64)
                        - direct.astype(np.float64)
                    )
                    / max(
                        float(np.linalg.norm(direct.astype(np.float64))),
                        np.finfo(float).tiny,
                    )
                )
                replay_closures[run_name][population_name] = replay
                print(
                    json.dumps(
                        {
                            "stage": "population_replay",
                            "run": run_name,
                            "population": population_name,
                            "all_on_replay_max_abs": replay,
                            "all_on_replay_relative_l2": replay_relative_l2,
                            "direct_max_abs": float(np.max(np.abs(direct))),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if replay > 1.0e-4:
                    raise ValueError(
                        f"{run_name}/{population_name} all-on replay exceeds the "
                        "descriptive compatibility tolerance: "
                        f"max_abs={replay}, relative_l2={replay_relative_l2}"
                    )
                population_metrics["factorial_effects"] = _factorial_effects(
                    {
                        mask: population_metrics[mask]["aggregates"]
                        for mask in (branch_mask_name(value) for value in BRANCH_MASKS)
                    }
                )
                cube[run_name]["populations"][population_name] = population_metrics
                deletion_rows.extend(
                    _acute_deletion_rows(
                        variant=variant,
                        seed=seed,
                        population_name=population_name,
                        grid=grid,
                        records=records,
                        predictions=predictions,
                        aggregates={
                            mask: population_metrics[mask]["aggregates"]
                            for mask in (
                                branch_mask_name(value) for value in BRANCH_MASKS
                            )
                        },
                    )
                )
                if variant == "no_gradient":
                    maximum = 0.0
                    for left in ("000", "010", "100", "110"):
                        right = left[:2] + "1"
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    np.abs(
                                        trace_predictions[left]
                                        - trace_predictions[right]
                                    )
                                )
                            ),
                        )
                    no_gradient_invariance[run_name][population_name] = maximum
                    if maximum != 0.0:
                        raise ValueError(
                            f"{run_name}/{population_name} functional zero is not exact"
                        )
                print(
                    json.dumps(
                        {
                            "stage": "population_complete",
                            "run": run_name,
                            "population": population_name,
                            "all_on_replay_max_abs": replay,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if model_state_sha256(model) != loaded_state:
                raise RuntimeError(f"{run_name} frozen analysis mutated the model")
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if native_records_for_figures is None or native_grid_for_figures is None:
        raise RuntimeError("native population was not constructed")
    decision = _decision_summary(cube, deletion_rows, seeds)
    write_json(output_dir / "branch_cube_metrics.json", cube)
    write_json(output_dir / "acute_deletion_attribution.json", deletion_rows)
    write_json(output_dir / "decision_summary.json", decision)
    write_json(output_dir / "replay_closures.json", replay_closures)
    write_json(
        output_dir / "functional_no_gradient_invariance.json",
        no_gradient_invariance,
    )
    write_npz(output_dir / "branch_cube_predictions.npz", **prediction_arrays)

    _style()
    figures = []
    figures += _save(
        _factorial_figure(cube, seeds), output_dir, "fig_branch_factorial"
    )
    figures += _save(
        _deletion_figure(cube, seeds), output_dir, "fig_branch_deletion_ratios"
    )
    figures += _save(
        _cancellation_figure(deletion_rows), output_dir, "fig_branch_cancellation"
    )
    figures += _save(
        _profile_figure(
            cube,
            prediction_arrays,
            native_records_for_figures,
            native_grid_for_figures,
        ),
        output_dir,
        "fig_held_phase_profiles",
    )
    figures += _save(
        _error_map_figure(
            prediction_arrays, native_records_for_figures, native_grid_for_figures
        ),
        output_dir,
        "fig_held_phase_error_maps",
    )
    if not smoke:
        figures += _save(
            _resolution_figure(cube, seeds),
            output_dir,
            "fig_resolution_branch_dependence",
        )
    figures += _save(
        _spectrum_figure(
            prediction_arrays,
            native_records_for_figures,
            native_grid_for_figures,
            seeds,
        ),
        output_dir,
        "fig_physical_spectrum",
    )
    visual_manifest = {
        "schema": SCHEMA,
        "files": [path.name for path in figures],
        "sha256": {path.name: sha256_file(path) for path in figures},
    }
    write_json(output_dir / "visual_manifest.json", visual_manifest)

    elapsed = time.perf_counter() - started
    maximum_replay = max(
        value for run in replay_closures.values() for value in run.values()
    )
    maximum_no_gradient_invariance = max(
        value for run in no_gradient_invariance.values() for value in run.values()
    )
    summary = {
        "schema": SCHEMA,
        "working_run_id": run_contract["working_run_id"],
        "status": "completed",
        "science_result": not smoke,
        "optimization_performed": False,
        "checkpoint_count": len(runs),
        "population_count_per_checkpoint": len(
            next(iter(cube.values()))["populations"]
        ),
        "branch_mask_count": len(BRANCH_MASKS),
        "closures": {
            "maximum_all_on_replay_max_abs": maximum_replay,
            "registered_1e_5_replay_gate_pass": maximum_replay <= 1.0e-5,
            "descriptive_1e_4_compatibility_pass": maximum_replay <= 1.0e-4,
            "mask_111_native_direct_bypass_max_abs": 0.0,
            "maximum_functional_no_gradient_third_bit_max_abs": (
                maximum_no_gradient_invariance
            ),
            "functional_no_gradient_exact_zero_pass": (
                maximum_no_gradient_invariance == 0.0
            ),
            "all_passed": (
                maximum_replay <= 1.0e-5
                and maximum_no_gradient_invariance == 0.0
            ),
        },
        "strict_registered_result": maximum_replay <= 1.0e-5,
        "descriptive_compatibility_result": maximum_replay <= 1.0e-4,
        "decision": decision,
        "visualizations": visual_manifest,
        "elapsed_seconds": elapsed,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "claim_boundary": (
            "P2-F is frozen one-step attribution on synthetic registered populations. "
            "Its third slot is a differential branch only for full PCNO, exact zero "
            "for no-gradient PCNO, and a matched local replacement for that arm. "
            "Mask 111 uses the native direct bypass while independent all-on replay "
            "is reported. A failed registered 1e-5 replay gate makes the cube "
            "descriptive even when the maintained 1e-4 compatibility floor passes. "
            "Masks are acute non-additive counterfactuals; no recurrent, Euler, "
            "strict Gibbs, or general operator claim follows."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "completed"},
    )
    output_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": run_contract["working_run_id"],
        "status": "completed",
        "science_result": not smoke,
        "input_manifest_sha256": {
            name: value["manifest_sha256"] for name, value in identities.items()
        },
        "input_checkpoint_sha256": {
            name: value["checkpoint_sha256"] for name, value in identities.items()
        },
        "source_sha256": source_hashes,
        "output_hashes": {path.name: sha256_file(path) for path in output_paths},
        "output_count": len(output_paths),
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
