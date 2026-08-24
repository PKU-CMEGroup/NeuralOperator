#!/usr/bin/env python3
"""Train and evaluate one W26-L2 fixed-representation PCNO arm."""

from __future__ import annotations

import argparse
import json
import math
import random
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

from scripts.time_dependent_no.analyze_pcno_moving_front_wake import (
    CALLS,
    FAMILIES,
    PHASES,
    _component_metrics,
    error_metrics,
    evaluate_sequence,
    moving_cases,
    spatial_masks,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
    build_batch,
    build_model,
    build_registered_cases,
    case_metric_rows,
    evaluate_split,
    expand_fourier_tensors,
    mapping_sha256,
    model_state_sha256,
    population_sha256,
    relative_squared_errors,
    sha256_file,
    write_checkpoint,
    write_json,
    write_npz,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    registered_config as p0_registered_config,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    _environment,
    _git_state,
    _resolve_device,
    _source_hashes,
)
from utility.time_dependent_no.pcno_shock_representation import (
    StructuredCellGrid,
    build_structured_cell_grid,
    physical_spectrum_summary,
    shock_representation_metrics,
)
from utility.time_dependent_no.pcno_structured_representation import (
    REPRESENTATIONS,
    StructuredCosinePCNO,
)

SCHEMA = "w26_l2_representation_routing_v1"
WORKING_PREFIX = "W26-L2-REP"
STAGES = ("smoke", "screen", "confirm")
CONFIRM_SEEDS = (1701, 1702, 1703)
DEFAULT_ROOT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_representation"
SOURCE_PATHS = (
    "pcno/__init__.py",
    "pcno/geo_utility.py",
    "pcno/pcno.py",
    "utility/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/__init__.py",
    "utility/time_dependent_no/cpg_mesh_contract.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_fv_geometry.py",
    "utility/time_dependent_no/pcno_resolution_pathways.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "utility/time_dependent_no/pcno_structured_representation.py",
    "utility/time_dependent_no/shock_vortex_coarse_cfd.py",
    "utility/time_dependent_no/shock_vortex_fv.py",
    "scripts/time_dependent_no/analyze_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_p2_frozen_branch_cube.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/train_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_moving_front_wake.py",
    "scripts/time_dependent_no/train_pcno_representation_routing.py",
)
PROVENANCE_PATHS = (
    "docs/time_dependent_no/W26_L2_REPRESENTATION_ROUTING_PREREGISTRATION.md",
)


def _representation_token(representation: str) -> str:
    return {
        "native": "NATIVE",
        "dct_pre_half_smooth": "DCT-PRE-HALF",
        "dct_coupled_half_smooth": "DCT-COUPLED-HALF",
    }[representation]


def registered_config(
    representation: str,
    seed: int,
    stage: str,
    *,
    device_type: str,
) -> dict[str, Any]:
    """Return the frozen M0/M1/M2 contract for one representation arm."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must lie in {REPRESENTATIONS}")
    if stage not in STAGES:
        raise ValueError(f"stage must lie in {STAGES}")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if stage == "screen" and seed_value != 1701:
        raise ValueError("the production screen is frozen to seed 1701")
    if stage == "confirm" and seed_value not in CONFIRM_SEEDS:
        raise ValueError(f"confirmation seed must lie in {CONFIRM_SEEDS}")

    smoke = stage == "smoke"
    config = p0_registered_config(smoke=smoke, device_type=device_type)
    steps = {"smoke": 2, "screen": 5_000, "confirm": 20_000}[stage]
    config.update(
        {
            "schema": SCHEMA,
            "working_run_id": (
                f"{WORKING_PREFIX}-{stage.upper()}-"
                f"{_representation_token(representation)}-S{seed_value}"
            ),
            "science_result_eligible": not smoke,
            "claim_eligible": stage == "confirm",
            "evidence_level": {
                "smoke": "non_scientific_engineering_smoke",
                "screen": "exploratory_single_seed_routing",
                "confirm": "bounded_paired_seed_confirmation",
            }[stage],
            "stage": stage,
            "representation": representation,
            "seed": seed_value,
            "paired_confirm_seeds": list(CONFIRM_SEEDS),
            "steps": steps,
            "evaluation_interval": 1 if smoke else 100,
            "checkpoint_interval": 1 if smoke else 500,
            "backbone": "full_pcno",
            "full_pcno_only": True,
            "calls": CALLS,
            "moving_families": list(FAMILIES),
            "moving_phases": list(PHASES),
            "ground_truth": "analytic_cell_average_synthetic_only",
            "loss_space": "physical_residual",
            "recurrence_space": "physical_state",
            "primary_metric": (
                "held_phase_0p875_step_pulse_recurrent_call2_"
                "mean_relative_increment_l2"
            ),
            "primary_direction": "lower_is_better",
            "smooth_control": (
                "held_phase_0p875_smooth_sine_recurrent_call2_"
                "relative_next_state_l2"
            ),
            "selection_rule": "final_registered_update",
        }
    )
    return config


def _checkpoint_payload(
    *,
    model: StructuredCosinePCNO,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_step: int,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
    config_digest: str,
    population_digest: str,
    backbone_initialization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": dict(config),
        "config_digest": config_digest,
        "population_sha256": population_digest,
        "backbone_initialization_sha256": backbone_initialization_sha256,
        "completed_step": int(completed_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_step_outputs(
    model: StructuredCosinePCNO,
    train_records: list[RegisteredCase],
    held_records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_input, train_target, train_aux = build_batch(train_records, grid, device)
    held_input, held_target, held_aux = build_batch(held_records, grid, device)
    train_fourier = expand_fourier_tensors(model, train_aux, len(train_records))
    held_fourier = expand_fourier_tensors(model, held_aux, len(held_records))
    train_prediction, train_scores = evaluate_split(
        model,
        train_input,
        train_target,
        train_aux,
        train_fourier,
        train_records,
        grid,
    )
    held_prediction, held_scores = evaluate_split(
        model,
        held_input,
        held_target,
        held_aux,
        held_fourier,
        held_records,
        grid,
    )
    all_records = train_records + held_records
    rows = case_metric_rows(train_records, train_prediction, grid)
    rows.extend(case_metric_rows(held_records, held_prediction, grid))
    arrays = {
        "case_ids": np.asarray([record.case_id for record in all_records]),
        "splits": np.asarray([record.split for record in all_records]),
        "families": np.asarray([record.family for record in all_records]),
        "anchor_indices": np.asarray(
            [record.anchor_index for record in all_records], dtype=np.int64
        ),
        "phases": np.asarray([record.phase for record in all_records]),
        "positions": np.asarray([record.position for record in all_records]),
        "current": np.stack([record.case.current for record in all_records]),
        "target_next": np.stack([record.case.target for record in all_records]),
        "target_increment": np.stack(
            [record.case.increment for record in all_records]
        ),
        "predicted_increment": np.concatenate(
            (train_prediction, held_prediction), axis=0
        ),
    }
    return {"train": train_scores, "held": held_scores, "rows": rows}, arrays


def _recurrent_outputs(
    model: StructuredCosinePCNO,
    grid: StructuredCellGrid,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "x_centers": grid.x_centers,
        "y_centers": grid.y_centers,
    }
    call_records: list[dict[str, Any]] = []
    for family in FAMILIES:
        for phase in PHASES:
            cases = moving_cases(grid, family, phase, calls=CALLS)
            initial_position = cases[0].position
            phase_token = str(float(phase)).replace(".", "p")
            case_prefix = f"case__{family}__p{phase_token}"
            for call, case in enumerate(cases):
                masks = spatial_masks(case, grid, initial_position=initial_position)
                arrays[f"{case_prefix}__call{call}__current"] = case.current
                arrays[f"{case_prefix}__call{call}__target"] = case.target
                arrays[f"{case_prefix}__call{call}__true_residual"] = case.increment
                for region, mask in masks.items():
                    arrays[f"{case_prefix}__call{call}__mask_{region}"] = mask

            sequence_records, sequence_arrays = evaluate_sequence(
                model,
                cases,
                grid,
                device,
                gradient_weight_scale=1.0,
            )
            for key, value in sequence_arrays.items():
                arrays[f"{case_prefix}__{key}"] = value
            for call, case in enumerate(cases):
                masks = spatial_masks(case, grid, initial_position=initial_position)
                teacher = sequence_arrays[f"call{call}__teacher_residual"]
                recurrent = sequence_arrays[f"call{call}__recurrent_residual"]
                recurrent_state = sequence_arrays[f"call{call}__recurrent_state"]
                fresh_error = sequence_arrays[f"call{call}__teacher_error"]
                propagated = sequence_arrays[
                    f"call{call}__propagated_contribution"
                ]
                component_masks = {**masks, "all": np.ones(grid.nx, dtype=bool)}
                decomposition = {
                    region: _component_metrics(
                        fresh_error, propagated, mask, grid
                    )
                    for region, mask in component_masks.items()
                }
                common = {
                    "family": family,
                    "phase": float(phase),
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
                call_records.append(common)
                for path, residual, next_state in (
                    ("teacher", teacher, case.current + teacher),
                    ("recurrent", recurrent, recurrent_state + recurrent),
                ):
                    metrics = error_metrics(
                        residual,
                        next_state,
                        case,
                        grid,
                        masks,
                    )
                    metrics["physical_next_state_structure"] = (
                        shock_representation_metrics(
                            next_state - case.current,
                            case,
                            grid,
                        )
                    )
                    metrics["residual_error_physical_spectrum"] = (
                        physical_spectrum_summary(residual - case.increment, grid)
                    )
                    rows.append({**common, "path": path, "metrics": metrics})

    call0_values = [
        row["teacher_recurrent_call0_max_abs"]
        for row in call_records
        if row["call"] == 0
        and row["teacher_recurrent_call0_max_abs"] is not None
    ]
    wake_truth_values = [
        row["metrics"]["regions"]["wake"]["true_residual_max_abs"]
        for row in rows
        if row["metrics"]["regions"]["wake"]["valid"]
    ]
    closures = {
        "teacher_recurrent_call0_max_abs": max(call0_values, default=0.0),
        "teacher_recurrent_call0_tolerance": 1.0e-5,
        "fresh_plus_propagated_max_abs": max(
            (row["decomposition_max_abs"] for row in call_records), default=0.0
        ),
        "fresh_plus_propagated_tolerance": 1.0e-12,
        "true_residual_in_wake_max_abs": max(wake_truth_values, default=0.0),
        "true_residual_in_wake_tolerance": 1.0e-12,
    }
    closures["teacher_recurrent_call0_pass"] = bool(
        closures["teacher_recurrent_call0_max_abs"]
        <= closures["teacher_recurrent_call0_tolerance"]
    )
    closures["fresh_plus_propagated_pass"] = bool(
        closures["fresh_plus_propagated_max_abs"]
        <= closures["fresh_plus_propagated_tolerance"]
    )
    closures["true_residual_in_wake_pass"] = bool(
        closures["true_residual_in_wake_max_abs"]
        <= closures["true_residual_in_wake_tolerance"]
    )
    if not all(
        value for key, value in closures.items() if key.endswith("_pass")
    ):
        raise ValueError(f"recurrent scientific closure failed: {closures}")
    return rows, arrays, closures


def _mean_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: set[str],
    phase: float,
    call: int,
    path: str,
    metric: str,
) -> float:
    selected = [
        float(row["metrics"][metric])
        for row in rows
        if row["family"] in families
        and abs(float(row["phase"]) - float(phase)) < 1.0e-12
        and int(row["call"]) == int(call)
        and row["path"] == path
    ]
    if not selected:
        raise ValueError("registered recurrent metric selection is empty")
    return float(np.mean(selected))


def _resume_checkpoint(
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    path = output_dir / "checkpoint_latest.pt"
    if not path.is_file():
        path = output_dir / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError("no checkpoint exists for resume")
    return torch.load(path, map_location=device, weights_only=False)


def run_experiment(
    output_dir: Path,
    *,
    representation: str,
    seed: int,
    stage: str,
    device_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if stage != "smoke" and device.type != "cuda":
        raise RuntimeError("production screen and confirmation require CUDA")
    config = registered_config(
        representation,
        seed,
        stage,
        device_type=device.type,
    )
    config_digest = mapping_sha256(config)
    output_dir = output_dir.resolve()
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
        completed_state = output_dir / "run_state.json"
        if (output_dir / "manifest.json").is_file():
            raise ValueError("completed output cannot be resumed")
        if completed_state.is_file():
            saved_state = json.loads(completed_state.read_text(encoding="utf-8"))
            if saved_state.get("status") == "completed":
                raise ValueError("completed output cannot be resumed")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_value = int(config["seed"])
    _seed_everything(seed_value)
    grid = build_structured_cell_grid(config["resolution"])
    train_records = build_registered_cases(grid, config, split="train")
    held_records = build_registered_cases(grid, config, split="held")
    all_records = train_records + held_records
    population_digest = population_sha256(all_records)
    source_hashes = _source_hashes(SOURCE_PATHS)
    provenance_hashes = _source_hashes(PROVENANCE_PATHS)

    backbone = build_model(config, device)
    backbone_initialization_sha256 = model_state_sha256(backbone)
    model = StructuredCosinePCNO(backbone, grid, representation)
    wrapper_initial_state_sha256 = model_state_sha256(model)
    run_contract = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result_eligible": bool(config["science_result_eligible"]),
        "claim_eligible": bool(config["claim_eligible"]),
        "config": config,
        "config_digest": config_digest,
        "population": {
            "kind": "analytic_cell_average_synthetic",
            "sha256": population_digest,
            "train_case_ids": [record.case_id for record in train_records],
            "held_case_ids": [record.case_id for record in held_records],
            "moving_families": list(FAMILIES),
            "moving_phases": list(PHASES),
            "moving_calls": CALLS,
        },
        "representation": {
            "name": representation,
            "scalar_state_input_only": representation != "native",
            "physical_residual_decode": (
                representation == "dct_coupled_half_smooth"
            ),
            "coordinates_quadrature_geometry_fourier_unchanged": True,
            "learned_parameters_added": 0,
        },
        "backbone_initialization_sha256": backbone_initialization_sha256,
        "wrapper_initial_state_sha256": wrapper_initial_state_sha256,
        "source_sha256": source_hashes,
        "provenance_sha256": provenance_hashes,
        "git": _git_state(),
        "environment": _environment(device),
    }
    contract_path = output_dir / "run_contract.json"
    if resume:
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        checks = {
            "config_digest": (saved["config_digest"], config_digest),
            "population_sha256": (
                saved["population"]["sha256"],
                population_digest,
            ),
            "source_sha256": (saved["source_sha256"], source_hashes),
            "provenance_sha256": (
                saved["provenance_sha256"],
                provenance_hashes,
            ),
            "backbone_initialization_sha256": (
                saved["backbone_initialization_sha256"],
                backbone_initialization_sha256,
            ),
        }
        mismatches = [name for name, (left, right) in checks.items() if left != right]
        if mismatches:
            raise ValueError(f"resume identity differs: {mismatches}")
    else:
        write_json(contract_path, run_contract)
        write_json(
            output_dir / "run_state.json",
            {
                "schema": SCHEMA,
                "working_run_id": config["working_run_id"],
                "status": "running",
                "completed_step": 0,
                "steps": config["steps"],
            },
        )

    train_input, train_target, train_aux = build_batch(train_records, grid, device)
    held_input, held_target, held_aux = build_batch(held_records, grid, device)
    train_fourier = expand_fourier_tensors(model, train_aux, len(train_records))
    held_fourier = expand_fourier_tensors(model, held_aux, len(held_records))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["steps"]),
        eta_min=float(config["minimum_learning_rate"]),
    )
    history: list[dict[str, Any]] = []
    start_step = 1
    checkpoint_latest = output_dir / "checkpoint_latest.pt"
    if resume:
        checkpoint = _resume_checkpoint(output_dir, device)
        if checkpoint["config_digest"] != config_digest:
            raise ValueError("resume checkpoint config digest differs")
        if checkpoint["population_sha256"] != population_digest:
            raise ValueError("resume checkpoint population digest differs")
        if (
            checkpoint["backbone_initialization_sha256"]
            != backbone_initialization_sha256
        ):
            raise ValueError("resume checkpoint backbone initialization differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint["history"])
        start_step = int(checkpoint["completed_step"]) + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        write_json(
            output_dir / "run_state.json",
            {
                "schema": SCHEMA,
                "working_run_id": config["working_run_id"],
                "status": "running",
                "completed_step": int(checkpoint["completed_step"]),
                "steps": config["steps"],
            },
        )

    def append_evaluation(step: int, optimizer_loss: float | None) -> None:
        _, train_scores = evaluate_split(
            model,
            train_input,
            train_target,
            train_aux,
            train_fourier,
            train_records,
            grid,
        )
        _, held_scores = evaluate_split(
            model,
            held_input,
            held_target,
            held_aux,
            held_fourier,
            held_records,
            grid,
        )
        history.append(
            {
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_loss": optimizer_loss,
                "train": train_scores,
                "held": held_scores,
            }
        )
        write_json(output_dir / "history.json", history)

    if not history:
        append_evaluation(0, None)
    started = time.perf_counter()
    last_loss: float | None = None
    for step in range(start_step, int(config["steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            train_input,
            train_aux,
            fourier_tensors=train_fourier,
        )
        loss = torch.mean(
            relative_squared_errors(prediction, train_target, train_aux[2])
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite physical loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(value).all()) for value in gradients
        ):
            raise FloatingPointError(f"nonfinite or missing gradients at step {step}")
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % int(config["evaluation_interval"]) == 0:
            append_evaluation(step, last_loss)
            print(
                json.dumps(
                    {
                        "working_run_id": config["working_run_id"],
                        "step": step,
                        "steps": config["steps"],
                        "optimizer_loss": last_loss,
                        "train_mean_relative_l2": history[-1]["train"][
                            "mean_relative_increment_l2"
                        ],
                        "held_mean_relative_l2": history[-1]["held"][
                            "mean_relative_increment_l2"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % int(config["checkpoint_interval"]) == 0:
            write_checkpoint(
                checkpoint_latest,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_step=step,
                    history=history,
                    config=config,
                    config_digest=config_digest,
                    population_digest=population_digest,
                    backbone_initialization_sha256=(
                        backbone_initialization_sha256
                    ),
                ),
            )
            write_json(
                output_dir / "run_state.json",
                {
                    "schema": SCHEMA,
                    "working_run_id": config["working_run_id"],
                    "status": "running",
                    "completed_step": step,
                    "steps": config["steps"],
                },
            )

    elapsed = time.perf_counter() - started
    if not history or history[-1]["step"] != int(config["steps"]):
        append_evaluation(int(config["steps"]), last_loss)
    one_step, one_step_arrays = _one_step_outputs(
        model, train_records, held_records, grid, device
    )
    recurrent_rows, recurrent_arrays, closures = _recurrent_outputs(
        model, grid, device
    )
    primary_value = _mean_metric(
        recurrent_rows,
        families={"step", "pulse"},
        phase=0.875,
        call=2,
        path="recurrent",
        metric="relative_increment_l2",
    )
    teacher_primary_value = _mean_metric(
        recurrent_rows,
        families={"step", "pulse"},
        phase=0.875,
        call=2,
        path="teacher",
        metric="relative_increment_l2",
    )
    smooth_control_value = _mean_metric(
        recurrent_rows,
        families={"smooth_sine"},
        phase=0.875,
        call=2,
        path="recurrent",
        metric="relative_next_state_l2",
    )

    final_checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=int(config["steps"]),
        history=history,
        config=config,
        config_digest=config_digest,
        population_digest=population_digest,
        backbone_initialization_sha256=backbone_initialization_sha256,
    )
    write_checkpoint(output_dir / "checkpoint.pt", final_checkpoint)
    if checkpoint_latest.is_file():
        checkpoint_latest.unlink()
    write_json(output_dir / "one_step_metrics.json", one_step)
    write_npz(output_dir / "one_step_predictions.npz", **one_step_arrays)
    write_json(output_dir / "recurrent_metrics.json", recurrent_rows)
    write_npz(output_dir / "recurrent_arrays.npz", **recurrent_arrays)
    summary = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "status": "completed",
        "science_result": bool(config["science_result_eligible"]),
        "claim_eligible": bool(config["claim_eligible"]),
        "evidence_level": config["evidence_level"],
        "stage": stage,
        "representation": representation,
        "seed": seed_value,
        "completed_steps": config["steps"],
        "selection_rule": config["selection_rule"],
        "primary_metric": config["primary_metric"],
        "primary_value": primary_value,
        "teacher_primary_value": teacher_primary_value,
        "smooth_control": config["smooth_control"],
        "smooth_control_value": smooth_control_value,
        "one_step": {"train": one_step["train"], "held": one_step["held"]},
        "recurrent_row_count": len(recurrent_rows),
        "closures": closures,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_initialization_sha256": backbone_initialization_sha256,
        "wrapper_initial_state_sha256": wrapper_initial_state_sha256,
        "final_model_state_sha256": model_state_sha256(model),
        "elapsed_training_seconds_this_invocation": elapsed,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "claim_boundary": (
            "A run is one arm of an empirical representation study; it does not "
            "identify an optimizer or spectral-bias mechanism."
        ),
    }
    if not math.isfinite(primary_value) or not math.isfinite(smooth_control_value):
        raise FloatingPointError("nonfinite registered summary metric")
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {
            "schema": SCHEMA,
            "working_run_id": config["working_run_id"],
            "status": "completed",
            "completed_step": config["steps"],
            "steps": config["steps"],
            "primary_value": primary_value,
        },
    )
    outputs = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "status": "completed",
        "science_result": bool(config["science_result_eligible"]),
        "claim_eligible": bool(config["claim_eligible"]),
        "config_digest": config_digest,
        "population_sha256": population_digest,
        "source_sha256": source_hashes,
        "provenance_sha256": provenance_hashes,
        "output_hashes": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in outputs
        },
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (
            DEFAULT_ROOT
            / args.stage
            / f"{args.representation}_s{int(args.seed)}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment(
        args.output,
        representation=str(args.representation),
        seed=int(args.seed),
        stage=str(args.stage),
        device_name=str(args.device),
        resume=bool(args.resume),
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
