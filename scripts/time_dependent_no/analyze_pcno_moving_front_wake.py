#!/usr/bin/env python3
"""Evaluate frozen P2 PCNO checkpoints on a three-call moving-front wake test."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_gradient_ablation import _load_matrix
from scripts.time_dependent_no.analyze_pcno_p2_frozen_branch_cube import (
    P2_POPULATION_SHA256,
    P2_TRAINER_SHA256,
    _verify_registered_inputs,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    build_model,
    model_state_sha256,
    sha256_file,
    write_json,
    write_npz,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    SEEDS,
    apply_variant,
)
from utility.time_dependent_no.pcno_shock_representation import (
    DISPLACEMENT,
    FRONT_BAND_WIDTH,
    PULSE_WIDTH,
    StructuredCellGrid,
    SyntheticFrontCase,
    _oscillatory_lobe_mass,
    build_structured_cell_grid,
    make_translated_front_case,
    pcno_aux_tensors,
)

SCHEMA = "w26_l2_p2_moving_front_wake_v1"
WORKING_ID = "W26-L2-P2-W0"
DEFAULT_MATRIX = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2_wake"
FAMILIES = ("step", "pulse", "smooth_tanh", "smooth_sine")
PHASES = (0.0, 0.875)
CALLS = 3
START_ANCHOR = 1.0 / 4.0
CUDA_CALL0_MAX_ABS_TOLERANCE = 1.0e-5
ARMS: Mapping[str, tuple[str, float]] = {
    "full_native": ("full", 1.0),
    "full_gradient_scale_0p1": ("full", 0.1),
    "no_gradient": ("no_gradient", 1.0),
}
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/train_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_p2_frozen_branch_cube.py",
    "scripts/time_dependent_no/analyze_pcno_moving_front_wake.py",
    "scripts/time_dependent_no/visualize_pcno_moving_front_wake.py",
)
PROVENANCE_PATH = "docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _field_input(
    state: np.ndarray,
    grid: StructuredCellGrid,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != grid.array_shape or not np.isfinite(values).all():
        raise ValueError("state must be finite and match the structured grid")
    features = np.concatenate(
        (grid.nodes, grid.geometry.node_rhos, values.reshape(-1, 1)), axis=1
    )
    return torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


def scaled_auxiliary_tensors(
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    gradient_weight_scale: float,
) -> tuple[torch.Tensor, ...]:
    """Build PCNO auxiliary tensors with only the LS gradient weights scaled."""

    scale = float(gradient_weight_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("gradient_weight_scale must be finite and positive")
    aux = pcno_aux_tensors(grid, dtype=torch.float32, device=device)
    return (*aux[:-1], aux[-1] * scale)


def moving_cases(
    grid: StructuredCellGrid,
    family: str,
    phase: float,
    *,
    calls: int = CALLS,
) -> tuple[SyntheticFrontCase, ...]:
    if family not in FAMILIES:
        raise ValueError(f"family must lie in {FAMILIES}")
    if calls < 1:
        raise ValueError("calls must be positive")
    start = START_ANCHOR + float(phase) * grid.hx
    return tuple(
        make_translated_front_case(
            grid,
            family,
            position=start + call * DISPLACEMENT,
        )
        for call in range(calls)
    )


def spatial_masks(
    case: SyntheticFrontCase,
    grid: StructuredCellGrid,
    *,
    initial_position: float,
) -> dict[str, np.ndarray]:
    """Return active, historical-wake, and ahead masks in x."""

    x = grid.x_centers
    if not case.is_discontinuous:
        zeros = np.zeros(grid.nx, dtype=bool)
        return {"active": zeros, "wake": zeros, "ahead": np.ones(grid.nx, bool)}

    support = np.any(np.abs(case.increment) > 1.0e-14, axis=0)
    front_band = np.zeros(grid.nx, dtype=bool)
    for front in case.increment_fronts:
        front_band |= np.abs(x - front.position) <= FRONT_BAND_WIDTH
    active = support | front_band

    traversed = np.zeros(grid.nx, dtype=bool)
    distance = max(0.0, case.position - float(initial_position))
    initial_edges = [float(initial_position)]
    if case.family == "pulse":
        initial_edges.append(float(initial_position) + PULSE_WIDTH)
    for initial_edge in initial_edges:
        traversed |= (x >= initial_edge) & (x <= initial_edge + distance)
    wake = traversed & ~active & ~support
    ahead = ~(active | wake)
    return {"active": active, "wake": wake, "ahead": ahead}


def _physical_tv(field: np.ndarray, grid: StructuredCellGrid) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(
        np.sum(np.abs(np.diff(values, axis=1))) * grid.hy
        + np.sum(np.abs(np.diff(values, axis=0))) * grid.hx
    )


def _region_metrics(
    error: np.ndarray,
    true_residual: np.ndarray,
    mask_x: np.ndarray,
    grid: StructuredCellGrid,
    *,
    normalization_l1: float,
) -> dict[str, Any]:
    valid = bool(np.any(mask_x))
    if not valid:
        return {
            "valid": False,
            "cell_count": 0,
            "physical_l1": None,
            "normalized_l1": None,
            "physical_l2": None,
            "normalized_l2": None,
            "signed_bias": None,
            "max_abs": None,
            "oscillatory_mass": None,
            "lobe_count": None,
            "coverage_above_1e_3": None,
            "true_residual_max_abs": None,
        }
    area = grid.hx * grid.hy
    selected_error = np.asarray(error, dtype=np.float64)[:, mask_x]
    selected_true = np.asarray(true_residual, dtype=np.float64)[:, mask_x]
    physical_l1 = float(np.sum(np.abs(selected_error)) * area)
    physical_l2 = float(np.sqrt(np.sum(np.square(selected_error)) * area))
    true_l2 = float(np.sqrt(np.sum(np.square(true_residual)) * area))
    oscillatory, lobes, signed_bias = _oscillatory_lobe_mass(
        np.asarray(error, dtype=np.float64),
        np.asarray(mask_x, dtype=bool),
        cell_area=area,
        normalization=normalization_l1,
        tolerance=1.0e-3,
    )
    return {
        "valid": True,
        "cell_count": int(selected_error.size),
        "physical_l1": physical_l1,
        "normalized_l1": physical_l1 / max(normalization_l1, np.finfo(float).tiny),
        "physical_l2": physical_l2,
        "normalized_l2": physical_l2 / max(true_l2, np.finfo(float).tiny),
        "signed_bias": signed_bias,
        "max_abs": float(np.max(np.abs(selected_error))),
        "oscillatory_mass": oscillatory,
        "lobe_count": int(lobes),
        "coverage_above_1e_3": float(np.mean(np.abs(selected_error) > 1.0e-3)),
        "true_residual_max_abs": float(np.max(np.abs(selected_true))),
    }


def error_metrics(
    predicted_residual: np.ndarray,
    predicted_next: np.ndarray,
    case: SyntheticFrontCase,
    grid: StructuredCellGrid,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    predicted = np.asarray(predicted_residual, dtype=np.float64)
    next_state = np.asarray(predicted_next, dtype=np.float64)
    if predicted.shape != grid.array_shape or next_state.shape != grid.array_shape:
        raise ValueError("predictions must match the structured grid")
    area = grid.hx * grid.hy
    residual_error = predicted - case.increment
    next_error = next_state - case.target
    true_l1 = float(np.sum(np.abs(case.increment)) * area)
    true_l2 = float(np.sqrt(np.sum(np.square(case.increment)) * area))
    next_l2 = float(np.sqrt(np.sum(np.square(case.target)) * area))
    predicted_tv = _physical_tv(predicted, grid)
    target_tv = _physical_tv(case.increment, grid)
    return {
        "relative_increment_l2": float(
            np.sqrt(np.sum(np.square(residual_error)) * area)
            / max(true_l2, np.finfo(float).tiny)
        ),
        "normalized_increment_l1": float(
            np.sum(np.abs(residual_error)) * area
            / max(true_l1, np.finfo(float).tiny)
        ),
        "relative_next_state_l2": float(
            np.sqrt(np.sum(np.square(next_error)) * area)
            / max(next_l2, np.finfo(float).tiny)
        ),
        "residual_error_max_abs": float(np.max(np.abs(residual_error))),
        "next_state_error_max_abs": float(np.max(np.abs(next_error))),
        "positive_tv_excess": max(0.0, predicted_tv - target_tv),
        "tv_deficit": max(0.0, target_tv - predicted_tv),
        "regions": {
            name: _region_metrics(
                residual_error,
                case.increment,
                mask,
                grid,
                normalization_l1=true_l1,
            )
            for name, mask in masks.items()
        },
    }


@torch.inference_mode()
def evaluate_sequence(
    model: torch.nn.Module,
    cases: Sequence[SyntheticFrontCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    gradient_weight_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate teacher-forced and recurrent paths and retain exact decomposition."""

    aux = scaled_auxiliary_tensors(
        grid, device, gradient_weight_scale=gradient_weight_scale
    )
    fourier = model.prepare_fourier_tensors(aux[1], aux[2])
    model.eval()
    recurrent_state = np.asarray(cases[0].current, dtype=np.float64).copy()
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for call, case in enumerate(cases):
        teacher = (
            model(_field_input(case.current, grid, device), aux, fourier_tensors=fourier)
            .detach()
            .cpu()
            .numpy()[0, :, 0]
            .reshape(grid.array_shape)
            .astype(np.float64)
        )
        recurrent = (
            model(_field_input(recurrent_state, grid, device), aux, fourier_tensors=fourier)
            .detach()
            .cpu()
            .numpy()[0, :, 0]
            .reshape(grid.array_shape)
            .astype(np.float64)
        )
        teacher_error = teacher - case.increment
        recurrent_error = recurrent - case.increment
        propagated = recurrent - teacher
        closure = recurrent_error - teacher_error - propagated
        recurrent_next = recurrent_state + recurrent
        teacher_next = case.current + teacher
        records.append(
            {
                "call": call,
                "teacher_recurrent_call0_max_abs": (
                    float(np.max(np.abs(teacher - recurrent))) if call == 0 else None
                ),
                "decomposition_max_abs": float(np.max(np.abs(closure))),
            }
        )
        arrays.update(
            {
                f"call{call}__teacher_residual": teacher,
                f"call{call}__recurrent_residual": recurrent,
                f"call{call}__teacher_error": teacher_error,
                f"call{call}__recurrent_error": recurrent_error,
                f"call{call}__propagated_contribution": propagated,
                f"call{call}__recurrent_state": recurrent_state.copy(),
                f"call{call}__teacher_next": teacher_next,
                f"call{call}__recurrent_next": recurrent_next,
                f"call{call}__recurrent_next_error": recurrent_next - case.target,
            }
        )
        recurrent_state = recurrent_next
    return records, arrays


def _component_metrics(
    fresh: np.ndarray,
    propagated: np.ndarray,
    mask_x: np.ndarray,
    grid: StructuredCellGrid,
) -> dict[str, Any]:
    if not np.any(mask_x):
        return {
            "valid": False,
            "fresh_energy": None,
            "propagated_energy": None,
            "cross_inner_product": None,
            "propagated_component_energy_share": None,
        }
    area = grid.hx * grid.hy
    f = fresh[:, mask_x]
    p = propagated[:, mask_x]
    fresh_energy = float(np.sum(np.square(f)) * area)
    propagated_energy = float(np.sum(np.square(p)) * area)
    denominator = fresh_energy + propagated_energy
    return {
        "valid": True,
        "fresh_energy": fresh_energy,
        "propagated_energy": propagated_energy,
        "cross_inner_product": float(np.sum(f * p) * area),
        "propagated_component_energy_share": (
            propagated_energy / max(denominator, np.finfo(float).tiny)
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(float(value) for value in values))


def _aggregate_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    discontinuous = [row for row in rows if row["family"] in {"step", "pulse"}]
    persistent: dict[str, Any] = {}
    regenerated: dict[str, Any] = {}
    for arm in ARMS:
        persistent[arm] = {}
        regenerated[arm] = {}
        for family in ("step", "pulse"):
            persistent[arm][family] = {}
            regenerated[arm][family] = {}
            for phase in PHASES:
                call1 = {
                    row["seed"]: row
                    for row in discontinuous
                    if row["arm"] == arm
                    and row["family"] == family
                    and row["phase"] == phase
                    and row["path"] == "recurrent"
                    and row["call"] == 1
                }
                call2 = {
                    row["seed"]: row
                    for row in discontinuous
                    if row["arm"] == arm
                    and row["family"] == family
                    and row["phase"] == phase
                    and row["path"] == "recurrent"
                    and row["call"] == 2
                }
                teacher2 = {
                    row["seed"]: row
                    for row in discontinuous
                    if row["arm"] == arm
                    and row["family"] == family
                    and row["phase"] == phase
                    and row["path"] == "teacher"
                    and row["call"] == 2
                }
                seed_flags = {}
                regeneration_flags = {}
                for seed in sorted(call2):
                    wake1 = call1[seed]["metrics"]["regions"]["wake"]
                    wake2 = call2[seed]["metrics"]["regions"]["wake"]
                    teacher_wake = teacher2[seed]["metrics"]["regions"]["wake"]
                    flag = bool(
                        wake1["valid"]
                        and wake2["valid"]
                        and wake2["oscillatory_mass"] >= 1.0e-3
                        and wake2["coverage_above_1e_3"] >= 0.5
                        and wake2["normalized_l1"] >= 0.5 * wake1["normalized_l1"]
                    )
                    seed_flags[str(seed)] = flag
                    component = call2[seed]["decomposition"]["wake"]
                    regeneration_flags[str(seed)] = bool(
                        flag
                        and teacher_wake["valid"]
                        and wake2["oscillatory_mass"]
                        >= 1.2 * teacher_wake["oscillatory_mass"]
                        and wake2["oscillatory_mass"]
                        - teacher_wake["oscillatory_mass"]
                        >= 1.0e-4
                        and component["propagated_component_energy_share"] > 0.5
                    )
                persistent[arm][family][str(phase)] = {
                    "seed_flags": seed_flags,
                    "passing_seed_count": sum(seed_flags.values()),
                    "phase_pass": sum(seed_flags.values()) >= 2,
                }
                regenerated[arm][family][str(phase)] = {
                    "seed_flags": regeneration_flags,
                    "passing_seed_count": sum(regeneration_flags.values()),
                    "phase_pass": sum(regeneration_flags.values()) >= 2,
                }
            persistent[arm][family]["both_phases_pass"] = all(
                persistent[arm][family][str(phase)]["phase_pass"] for phase in PHASES
            )
            regenerated[arm][family]["both_phases_pass"] = bool(
                persistent[arm][family]["both_phases_pass"]
                and all(
                    regenerated[arm][family][str(phase)]["phase_pass"]
                    for phase in PHASES
                )
            )

    sine_rows = [
        row
        for row in rows
        if row["family"] == "smooth_sine" and row["path"] == "teacher"
    ]
    smooth_pairs = {}
    for seed in SEEDS:
        full = _mean(
            row["metrics"]["relative_increment_l2"]
            for row in sine_rows
            if row["arm"] == "full_native" and row["seed"] == seed
        )
        no_gradient = _mean(
            row["metrics"]["relative_increment_l2"]
            for row in sine_rows
            if row["arm"] == "no_gradient" and row["seed"] == seed
        )
        full_wake = _mean(
            row["metrics"]["regions"]["wake"]["oscillatory_mass"]
            for row in discontinuous
            if row["arm"] == "full_native"
            and row["seed"] == seed
            and row["path"] == "recurrent"
            and row["call"] == 2
        )
        no_gradient_wake = _mean(
            row["metrics"]["regions"]["wake"]["oscillatory_mass"]
            for row in discontinuous
            if row["arm"] == "no_gradient"
            and row["seed"] == seed
            and row["path"] == "recurrent"
            and row["call"] == 2
        )
        smooth_pairs[str(seed)] = {
            "full": full,
            "no_gradient": no_gradient,
            "full_to_no_gradient_ratio": full / max(no_gradient, np.finfo(float).tiny),
            "five_percent_full_win": full <= 0.95 * no_gradient,
            "full_discontinuous_wake": full_wake,
            "no_gradient_discontinuous_wake": no_gradient_wake,
            "full_smooth_win_without_larger_wake": bool(
                full <= 0.95 * no_gradient and full_wake <= no_gradient_wake
            ),
        }

    scale_pairs = {}
    for seed in SEEDS:
        values: dict[str, dict[str, float]] = {}
        for arm in ("full_native", "full_gradient_scale_0p1"):
            values[arm] = {
                "discontinuous_relative_l2": _mean(
                    row["metrics"]["relative_increment_l2"]
                    for row in discontinuous
                    if row["arm"] == arm
                    and row["seed"] == seed
                    and row["path"] == "recurrent"
                    and row["call"] == 2
                ),
                "discontinuous_wake_oscillatory_mass": _mean(
                    row["metrics"]["regions"]["wake"]["oscillatory_mass"]
                    for row in discontinuous
                    if row["arm"] == arm
                    and row["seed"] == seed
                    and row["path"] == "recurrent"
                    and row["call"] == 2
                ),
                "smooth_sine_relative_l2": _mean(
                    row["metrics"]["relative_increment_l2"]
                    for row in sine_rows
                    if row["arm"] == arm and row["seed"] == seed
                ),
            }
        native = values["full_native"]
        scaled = values["full_gradient_scale_0p1"]
        scale_pairs[str(seed)] = {
            "native": native,
            "scaled": scaled,
            "scaled_to_native_discontinuous_l2_ratio": (
                scaled["discontinuous_relative_l2"]
                / max(native["discontinuous_relative_l2"], np.finfo(float).tiny)
            ),
            "scaled_to_native_wake_mass_ratio": (
                scaled["discontinuous_wake_oscillatory_mass"]
                / max(
                    native["discontinuous_wake_oscillatory_mass"],
                    np.finfo(float).tiny,
                )
            ),
            "scaled_to_native_sine_l2_ratio": (
                scaled["smooth_sine_relative_l2"]
                / max(native["smooth_sine_relative_l2"], np.finfo(float).tiny)
            ),
            "helpful_without_sine_harm": bool(
                scaled["discontinuous_relative_l2"]
                < native["discontinuous_relative_l2"]
                and scaled["discontinuous_wake_oscillatory_mass"]
                < native["discontinuous_wake_oscillatory_mass"]
                and scaled["smooth_sine_relative_l2"]
                <= 1.05 * native["smooth_sine_relative_l2"]
            ),
        }
    return {
        "path_wide_persistent_wake": persistent,
        "recurrently_regenerated": regenerated,
        "smooth_sine_full_vs_no_gradient": {
            "paired": smooth_pairs,
            "five_percent_full_win_seed_count": sum(
                value["five_percent_full_win"] for value in smooth_pairs.values()
            ),
            "registered_useful_seed_count": sum(
                value["full_smooth_win_without_larger_wake"]
                for value in smooth_pairs.values()
            ),
        },
        "full_gradient_scale_0p1": {
            "paired": scale_pairs,
            "helpful_without_sine_harm_seed_count": sum(
                value["helpful_without_sine_harm"]
                for value in scale_pairs.values()
            ),
            "registered_helpful": sum(
                value["helpful_without_sine_harm"]
                for value in scale_pairs.values()
            )
            >= 2,
        },
    }


def run_analysis(
    matrix_dir: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("production W26-L2-P2-W0 requires CUDA")
    matrix_dir = matrix_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "running"},
    )

    runs = _load_matrix(matrix_dir, smoke=False)
    identities = _verify_registered_inputs(runs, smoke=False)
    sample = runs[("full", SEEDS[0])]
    config = sample["contract"]["config"]
    grid = build_structured_cell_grid(config["resolution"])
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    contract = {
        "schema": SCHEMA,
        "working_run_id": WORKING_ID,
        "status": "running",
        "science_result_eligible": True,
        "optimization_performed": False,
        "resolution": list(grid.resolution),
        "families": list(FAMILIES),
        "phases": list(PHASES),
        "calls": CALLS,
        "displacement": DISPLACEMENT,
        "pulse_width": PULSE_WIDTH,
        "front_band_half_width": FRONT_BAND_WIDTH,
        "closures": {
            "teacher_recurrent_call0_max_abs_tolerance": (
                CUDA_CALL0_MAX_ABS_TOLERANCE
            ),
            "fresh_plus_propagated_max_abs_tolerance": 1.0e-12,
            "true_residual_in_wake_max_abs_tolerance": 1.0e-12,
        },
        "arms": {
            arm: {"checkpoint_variant": variant, "gradient_weight_scale": scale}
            for arm, (variant, scale) in ARMS.items()
        },
        "checkpoint_identities": {
            name: value
            for name, value in identities.items()
            if name.startswith("full_") or name.startswith("no_gradient_")
        },
        "population_sha256": P2_POPULATION_SHA256,
        "trainer_sha256": P2_TRAINER_SHA256,
        "source_sha256": source_hashes,
        "preregistration_sha256": sha256_file(ROOT / PROVENANCE_PATH),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device),
        },
    }
    write_json(output_dir / "run_contract.json", contract)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "x_centers": grid.x_centers,
        "y_centers": grid.y_centers,
    }
    loaded_states: dict[str, str] = {}
    started = time.perf_counter()
    for seed in SEEDS:
        loaded_models: dict[str, torch.nn.Module] = {}
        for variant in ("full", "no_gradient"):
            run = runs[(variant, seed)]
            checkpoint = torch.load(
                run["run_dir"] / "checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
            model = apply_variant(build_model(config, device), variant)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            state_hash = model_state_sha256(model)
            if state_hash != run["summary"]["final_model_state_sha256"]:
                raise ValueError(f"{variant}_s{seed} loaded state does not close")
            model.eval()
            loaded_models[variant] = model
            loaded_states[f"{variant}_s{seed}"] = state_hash

        for family in FAMILIES:
            for phase in PHASES:
                cases = moving_cases(grid, family, phase)
                initial_position = cases[0].position
                case_prefix = f"case__{family}__p{str(phase).replace('.', 'p')}"
                for call, case in enumerate(cases):
                    masks = spatial_masks(
                        case, grid, initial_position=initial_position
                    )
                    arrays[f"{case_prefix}__call{call}__current"] = case.current
                    arrays[f"{case_prefix}__call{call}__target"] = case.target
                    arrays[f"{case_prefix}__call{call}__true_residual"] = case.increment
                    for region, mask in masks.items():
                        arrays[f"{case_prefix}__call{call}__mask_{region}"] = mask

                for arm, (variant, scale) in ARMS.items():
                    sequence_records, sequence_arrays = evaluate_sequence(
                        loaded_models[variant],
                        cases,
                        grid,
                        device,
                        gradient_weight_scale=scale,
                    )
                    arm_prefix = f"{arm}__s{seed}__{family}__p{str(phase).replace('.', 'p')}"
                    for key, value in sequence_arrays.items():
                        arrays[f"{arm_prefix}__{key}"] = value
                    for call, case in enumerate(cases):
                        masks = spatial_masks(
                            case, grid, initial_position=initial_position
                        )
                        teacher = sequence_arrays[f"call{call}__teacher_residual"]
                        recurrent = sequence_arrays[f"call{call}__recurrent_residual"]
                        recurrent_state = sequence_arrays[f"call{call}__recurrent_state"]
                        fresh_error = sequence_arrays[f"call{call}__teacher_error"]
                        propagated = sequence_arrays[
                            f"call{call}__propagated_contribution"
                        ]
                        decomposition = {
                            region: _component_metrics(
                                fresh_error, propagated, mask, grid
                            )
                            for region, mask in masks.items()
                        }
                        common = {
                            "arm": arm,
                            "checkpoint_variant": variant,
                            "gradient_weight_scale": scale,
                            "seed": seed,
                            "family": family,
                            "phase": phase,
                            "call": call,
                            "position": case.position,
                            "increment_front_positions": [
                                front.position for front in case.increment_fronts
                            ],
                            "target_front_positions": [
                                front.position for front in case.target_fronts
                            ],
                            "decomposition": decomposition,
                            "decomposition_max_abs": sequence_records[call][
                                "decomposition_max_abs"
                            ],
                            "teacher_recurrent_call0_max_abs": sequence_records[call][
                                "teacher_recurrent_call0_max_abs"
                            ],
                        }
                        rows.append(
                            {
                                **common,
                                "path": "teacher",
                                "metrics": error_metrics(
                                    teacher,
                                    case.current + teacher,
                                    case,
                                    grid,
                                    masks,
                                ),
                            }
                        )
                        rows.append(
                            {
                                **common,
                                "path": "recurrent",
                                "metrics": error_metrics(
                                    recurrent,
                                    recurrent_state + recurrent,
                                    case,
                                    grid,
                                    masks,
                                ),
                            }
                        )
    elapsed = time.perf_counter() - started

    maximum_call0 = max(
        row["teacher_recurrent_call0_max_abs"]
        for row in rows
        if row["call"] == 0 and row["teacher_recurrent_call0_max_abs"] is not None
    )
    maximum_decomposition = max(row["decomposition_max_abs"] for row in rows)
    maximum_true_wake = max(
        row["metrics"]["regions"]["wake"]["true_residual_max_abs"]
        for row in rows
        if row["metrics"]["regions"]["wake"]["valid"]
    )
    closures = {
        "teacher_recurrent_call0_max_abs": maximum_call0,
        "teacher_recurrent_call0_pass": (
            maximum_call0 <= CUDA_CALL0_MAX_ABS_TOLERANCE
        ),
        "fresh_plus_propagated_max_abs": maximum_decomposition,
        "fresh_plus_propagated_pass": maximum_decomposition <= 1.0e-12,
        "true_residual_in_wake_max_abs": maximum_true_wake,
        "true_residual_in_wake_pass": maximum_true_wake <= 1.0e-12,
    }
    if not all(value for key, value in closures.items() if key.endswith("_pass")):
        raise ValueError(f"scientific closure failed: {closures}")

    write_json(output_dir / "metrics.json", rows)
    write_npz(output_dir / "arrays.npz", **arrays)
    decisions = _aggregate_decisions(rows)
    summary = {
        "schema": SCHEMA,
        "working_run_id": WORKING_ID,
        "status": "complete",
        "science_result_eligible": True,
        "elapsed_seconds": elapsed,
        "loaded_model_state_sha256": loaded_states,
        "closures": closures,
        "decisions": decisions,
        "claim_boundary": (
            "Gibbs is analogical; exact old/new residual fronts, isolated bumps, "
            "and high-frequency energy are not classified as ripple wakes."
        ),
    }
    write_json(output_dir / "summary.json", summary)

    from scripts.time_dependent_no.visualize_pcno_moving_front_wake import (
        render_figures,
    )

    figures = render_figures(output_dir)
    contract["status"] = "complete"
    contract["elapsed_seconds"] = elapsed
    write_json(output_dir / "run_contract.json", contract)
    output_hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "run_state.json"}:
            output_hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    manifest = {
        "schema": SCHEMA,
        "working_run_id": WORKING_ID,
        "status": "complete",
        "science_result_eligible": True,
        "optimization_performed": False,
        "output_hashes": output_hashes,
        "figure_files": [path.relative_to(output_dir).as_posix() for path in figures],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "run_state.json",
        {"schema": SCHEMA, "working_run_id": WORKING_ID, "status": "complete"},
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(args.matrix_dir, args.output_dir, device_name=args.device)
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
