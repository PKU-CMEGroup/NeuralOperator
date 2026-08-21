#!/usr/bin/env python3
"""Fit the registered W26-L2-P0 full PCNO synthetic moving-front map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.pcno_shock_representation import (
    ANCHORS,
    CASE_FAMILIES,
    DOMAIN_LENGTHS,
    HELD_PHASES,
    NATIVE_RESOLUTION,
    TRAIN_PHASES,
    StructuredCellGrid,
    SyntheticFrontCase,
    build_structured_cell_grid,
    make_translated_front_case,
    pcno_aux_tensors,
    pcno_case_input,
    shock_representation_metrics,
)

SCHEMA = "w26_l2_p0_full_pcno_overfit_v1"
WORKING_RUN_ID = "W26-L2-P0-FULL-S1701"
DEFAULT_OUTPUT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p0_full_s1701"
SOURCE_PATHS = (
    "pcno/__init__.py",
    "pcno/geo_utility.py",
    "pcno/pcno.py",
    "utility/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/__init__.py",
    "utility/time_dependent_no/pcno_fv_geometry.py",
    "utility/time_dependent_no/pcno_resolution_pathways.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "utility/time_dependent_no/shock_vortex_coarse_cfd.py",
    "utility/time_dependent_no/shock_vortex_fv.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/visualize_pcno_shock_overfit.py",
)
PROVENANCE_PATHS = ("docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",)


@dataclass(frozen=True)
class RegisteredCase:
    case_id: str
    split: str
    family: str
    anchor_index: int
    phase: float
    position: float
    case: SyntheticFrontCase


def _phase_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def registered_config(*, smoke: bool, device_type: str) -> dict[str, Any]:
    """Return the immutable scientific contract or a non-scientific smoke."""

    if smoke:
        return {
            "schema": SCHEMA,
            "working_run_id": f"{WORKING_RUN_ID}-SMOKE",
            "science_result_eligible": False,
            "variant": "full_pcno",
            "seed": 1701,
            "resolution": [16, 8],
            "anchors": [ANCHORS[1]],
            "train_phases": [0.0, 0.5],
            "held_phases": [0.375],
            "families": list(CASE_FAMILIES),
            "width": 4,
            "block_count": 1,
            "decoder_width": 4,
            "k_max": 1,
            "steps": 2,
            "learning_rate": 1.0e-3,
            "minimum_learning_rate": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "evaluation_interval": 1,
            "checkpoint_interval": 1,
            "device_type": device_type,
            "precision": "float32",
            "selection_rule": "final_registered_update",
            "target_normalization": "identity",
            "objective": "mean_case_squared_relative_increment_l2",
        }
    return {
        "schema": SCHEMA,
        "working_run_id": WORKING_RUN_ID,
        "science_result_eligible": True,
        "variant": "full_pcno",
        "seed": 1701,
        "resolution": list(NATIVE_RESOLUTION),
        "anchors": list(ANCHORS),
        "train_phases": list(TRAIN_PHASES),
        "held_phases": list(HELD_PHASES),
        "families": list(CASE_FAMILIES),
        "width": 128,
        "block_count": 4,
        "decoder_width": 128,
        "k_max": 8,
        "steps": 20_000,
        "learning_rate": 1.0e-3,
        "minimum_learning_rate": 1.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "evaluation_interval": 100,
        "checkpoint_interval": 500,
        "device_type": device_type,
        "precision": "float32",
        "selection_rule": "final_registered_update",
        "target_normalization": "identity",
        "objective": "mean_case_squared_relative_increment_l2",
    }


def build_registered_cases(
    grid: StructuredCellGrid,
    config: dict[str, Any],
    *,
    split: str,
) -> list[RegisteredCase]:
    if split not in {"train", "held"}:
        raise ValueError("split must be train or held")
    phases = config[f"{split}_phases"]
    records: list[RegisteredCase] = []
    for family in config["families"]:
        for anchor_index, anchor in enumerate(config["anchors"]):
            for phase in phases:
                position = float(anchor) + float(phase) * grid.hx
                records.append(
                    RegisteredCase(
                        case_id=(
                            f"{split}_{family}_a{anchor_index}_phase_"
                            f"{_phase_token(float(phase))}"
                        ),
                        split=split,
                        family=str(family),
                        anchor_index=anchor_index,
                        phase=float(phase),
                        position=position,
                        case=make_translated_front_case(
                            grid,
                            str(family),
                            position=position,
                        ),
                    )
                )
    return records


def population_sha256(records: list[RegisteredCase]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.case_id.encode("utf-8"))
        for value in (record.case.current, record.case.target, record.case.increment):
            array = np.ascontiguousarray(value, dtype=np.float64)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def build_model(config: dict[str, Any], device: torch.device) -> PCNO:
    modes = torch.as_tensor(
        compute_Fourier_modes(
            2,
            [int(config["k_max"]), int(config["k_max"])],
            list(DOMAIN_LENGTHS),
        ),
        dtype=torch.float32,
        device=device,
    )
    width = int(config["width"])
    model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[width] * (int(config["block_count"]) + 1),
        fc_dim=int(config["decoder_width"]),
        in_dim=4,
        out_dim=1,
        act="gelu",
    )
    return model.to(device)


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_batch(
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    model_input = torch.cat(
        [pcno_case_input(record.case, grid) for record in records], dim=0
    ).to(device)
    target = torch.as_tensor(
        np.stack([record.case.increment.reshape(-1, 1) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    single_aux = pcno_aux_tensors(grid, dtype=torch.float32, device=device)
    batch_size = len(records)
    aux = tuple(value.expand(batch_size, *value.shape[1:]) for value in single_aux)
    return model_input, target, aux


def expand_fourier_tensors(
    model: PCNO,
    aux: tuple[torch.Tensor, ...],
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    single = model.prepare_fourier_tensors(aux[1][:1], aux[2][:1])
    return tuple(value.expand(batch_size, *value.shape[1:]) for value in single)


def relative_squared_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.sum(node_weights * torch.square(prediction - target), dim=(1, 2))
    denominator = torch.sum(node_weights * torch.square(target), dim=(1, 2))
    return numerator / torch.clamp_min(denominator, torch.finfo(target.dtype).tiny)


@torch.inference_mode()
def evaluate_split(
    model: PCNO,
    model_input: torch.Tensor,
    target: torch.Tensor,
    aux: tuple[torch.Tensor, ...],
    fourier_tensors: tuple[torch.Tensor, ...],
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    prediction = model(model_input, aux, fourier_tensors=fourier_tensors)
    relative_l2 = torch.sqrt(relative_squared_errors(prediction, target, aux[2]))
    values = relative_l2.detach().cpu().numpy()
    family_rows: dict[str, dict[str, float]] = {}
    for family in CASE_FAMILIES:
        selected = [
            index for index, record in enumerate(records) if record.family == family
        ]
        if selected:
            family_values = values[selected]
            family_rows[family] = {
                "mean_relative_increment_l2": float(np.mean(family_values)),
                "maximum_relative_increment_l2": float(np.max(family_values)),
            }
    prediction_array = (
        prediction[..., 0]
        .detach()
        .cpu()
        .numpy()
        .reshape(len(records), grid.ny, grid.nx)
    )
    return prediction_array, {
        "objective": float(torch.mean(torch.square(relative_l2))),
        "mean_relative_increment_l2": float(np.mean(values)),
        "maximum_relative_increment_l2": float(np.max(values)),
        "families": family_rows,
    }


def case_metric_rows(
    records: list[RegisteredCase],
    predictions: np.ndarray,
    grid: StructuredCellGrid,
) -> list[dict[str, Any]]:
    if predictions.shape != (len(records), grid.ny, grid.nx):
        raise ValueError("predictions do not match the registered case grid")
    rows: list[dict[str, Any]] = []
    for record, prediction in zip(records, predictions, strict=True):
        rows.append(
            {
                "case_id": record.case_id,
                "split": record.split,
                "family": record.family,
                "anchor_index": record.anchor_index,
                "phase": record.phase,
                "position": record.position,
                "metrics": shock_representation_metrics(
                    prediction,
                    record.case,
                    grid,
                ),
            }
        )
    return rows


def _case_guard(
    row: dict[str, Any],
    *,
    grid_spacing: float,
    held: bool,
) -> dict[str, Any]:
    metrics = row["metrics"]
    thresholds = {
        "relative_increment_l2": 1.0e-3 if not held else 5.0e-3,
        "normalized_overshoot": 0.01 if not held else 0.02,
        "normalized_undershoot": 0.01 if not held else 0.02,
        "oscillatory_mass_outside_front_band": 1.0e-3,
        "smooth_region_error": 1.0e-3 if not held else 5.0e-3,
        "positive_total_variation_excess": 0.02 if not held else 0.05,
        "total_variation_deficit": 0.02 if not held else 0.05,
        "increment_integral_error": 0.005 if not held else 0.01,
        "next_state_integral_error": 0.005 if not held else 0.01,
    }
    failures = [
        name
        for name, threshold in thresholds.items()
        if float(metrics[name]) > threshold
    ]
    front_thresholds = {
        "position_error": grid_spacing / (2.0 if held else 4.0),
        "strength_error": 0.05 if held else 0.02,
        "thickness_excess": grid_spacing if held else grid_spacing / 2.0,
    }
    for front_index, front in enumerate(metrics["fronts"]):
        if not bool(front["front_gate_valid"]):
            failures.append(f"front_{front_index}:crossing_invalid")
        for name, threshold in front_thresholds.items():
            value = front[name]
            if value is None or (
                name == "thickness_excess" and max(0.0, value) > threshold
            ):
                if value is None or name == "thickness_excess":
                    failures.append(f"front_{front_index}:{name}")
            elif float(value) > threshold:
                failures.append(f"front_{front_index}:{name}")
    return {
        "case_id": row["case_id"],
        "passed": not failures,
        "failures": sorted(set(failures)),
    }


def capacity_summary(
    rows: list[dict[str, Any]],
    *,
    grid_spacing: float,
) -> dict[str, Any]:
    train_rows = [
        row
        for row in rows
        if row["split"] == "train" and row["family"] in {"step", "pulse"}
    ]
    held_rows = [
        row
        for row in rows
        if row["split"] == "held" and row["family"] in {"step", "pulse"}
    ]
    train_decisions = [
        _case_guard(row, grid_spacing=grid_spacing, held=False) for row in train_rows
    ]
    held_decisions = [
        _case_guard(row, grid_spacing=grid_spacing, held=True) for row in held_rows
    ]
    held_by_family: dict[str, Any] = {}
    for family in ("step", "pulse"):
        selected = [
            decision
            for decision, row in zip(held_decisions, held_rows, strict=True)
            if row["family"] == family
        ]
        held_by_family[family] = {
            "passed_case_count": sum(bool(item["passed"]) for item in selected),
            "case_count": len(selected),
            "ten_of_twelve_gate": len(selected) == 12
            and sum(bool(item["passed"]) for item in selected) >= 10,
        }
    fixed_grid_pass = bool(train_decisions) and all(
        bool(item["passed"]) for item in train_decisions
    )
    return {
        "fixed_grid_capacity_status": (
            "passed" if fixed_grid_pass else "not_demonstrated_single_seed"
        ),
        "fixed_grid_capacity_pass": fixed_grid_pass,
        "train_discontinuous_case_count": len(train_decisions),
        "train_discontinuous_pass_count": sum(
            bool(item["passed"]) for item in train_decisions
        ),
        "train_case_decisions": train_decisions,
        "held_phase_descriptive": {
            "claim_allowed": False,
            "reason": "native held-phase rows precede the registered resolution stage",
            "by_family": held_by_family,
            "case_decisions": held_decisions,
        },
    }


def _source_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "available": True,
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status_short": run("status", "--short"),
            "unavailable_reason": None,
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "head": None,
            "branch": None,
            "status_short": None,
            "unavailable_reason": type(error).__name__,
        }


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "pid": os.getpid(),
    }


def _checkpoint_payload(
    *,
    model: PCNO,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_step: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    config_digest: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": config,
        "config_digest": config_digest,
        "completed_step": completed_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run_experiment(
    output_dir: Path,
    *,
    smoke: bool,
    device_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if not smoke and device.type != "cuda":
        raise RuntimeError("the production P0 contract requires a CUDA device")
    config = registered_config(smoke=smoke, device_type=device.type)
    config_digest = mapping_sha256(config)
    output_dir = output_dir.resolve()
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    grid = build_structured_cell_grid(config["resolution"])
    train_records = build_registered_cases(grid, config, split="train")
    held_records = build_registered_cases(grid, config, split="held")
    all_records = train_records + held_records
    population_digest = population_sha256(all_records)
    source_hashes = _source_hashes(SOURCE_PATHS)
    provenance_hashes = _source_hashes(PROVENANCE_PATHS)
    run_contract = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": config,
        "config_digest": config_digest,
        "population": {
            "kind": "analytic_cell_average_synthetic",
            "sha256": population_digest,
            "train_case_ids": [record.case_id for record in train_records],
            "held_case_ids": [record.case_id for record in held_records],
        },
        "source_sha256": source_hashes,
        "provenance_sha256": provenance_hashes,
        "git": _git_state(),
        "environment": _environment(device),
        "gradient_ablation_decision": {
            "executed": False,
            "decision": "defer_to_registered_p2_three_arm_three_seed_study",
            "reason": (
                "full versus functional no-gradient alone confounds pathway "
                "removal with active capacity"
            ),
        },
    }
    contract_path = output_dir / "run_contract.json"
    if resume:
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved["config_digest"] != config_digest:
            raise ValueError("resume config digest differs from the saved contract")
        if saved["population"]["sha256"] != population_digest:
            raise ValueError("resume synthetic population digest differs")
        if saved["source_sha256"] != source_hashes:
            raise ValueError("resume executable source hashes differ")
    else:
        write_json(contract_path, run_contract)
    write_json(
        output_dir / "run_state.json",
        {
            "schema": SCHEMA,
            "status": "running",
            "completed_step": 0,
            "steps": config["steps"],
        },
    )

    model = build_model(config, device)
    initial_model_sha256 = model_state_sha256(model)
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
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if resume:
        resume_checkpoint_path = (
            checkpoint_path
            if checkpoint_path.is_file()
            else output_dir / "checkpoint.pt"
        )
        checkpoint = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint["config_digest"] != config_digest:
            raise ValueError("resume checkpoint config digest differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint["history"])
        start_step = int(checkpoint["completed_step"]) + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

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
        prediction = model(train_input, train_aux, fourier_tensors=train_fourier)
        loss = torch.mean(
            relative_squared_errors(prediction, train_target, train_aux[2])
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradients_finite = bool(
            gradients
            and torch.stack([torch.isfinite(value).all() for value in gradients]).all()
        )
        if not gradients_finite:
            raise FloatingPointError(f"nonfinite or missing gradients at step {step}")
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % int(config["evaluation_interval"]) == 0:
            append_evaluation(step, last_loss)
            print(
                json.dumps(
                    {
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
                checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_step=step,
                    history=history,
                    config=config,
                    config_digest=config_digest,
                ),
            )
            write_json(
                output_dir / "run_state.json",
                {
                    "schema": SCHEMA,
                    "status": "running",
                    "completed_step": step,
                    "steps": config["steps"],
                },
            )

    elapsed = time.perf_counter() - started
    if not history or history[-1]["step"] != int(config["steps"]):
        append_evaluation(int(config["steps"]), last_loss)
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
    predictions = np.concatenate((train_prediction, held_prediction), axis=0)
    metric_rows = case_metric_rows(train_records, train_prediction, grid)
    metric_rows.extend(case_metric_rows(held_records, held_prediction, grid))
    capacity = capacity_summary(metric_rows, grid_spacing=grid.hx)
    final_checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=int(config["steps"]),
        history=history,
        config=config,
        config_digest=config_digest,
    )
    write_checkpoint(output_dir / "checkpoint.pt", final_checkpoint)
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
    write_json(output_dir / "case_metrics.json", metric_rows)
    write_npz(
        output_dir / "predictions.npz",
        case_ids=np.asarray([record.case_id for record in all_records]),
        splits=np.asarray([record.split for record in all_records]),
        families=np.asarray([record.family for record in all_records]),
        anchor_indices=np.asarray([record.anchor_index for record in all_records]),
        phases=np.asarray([record.phase for record in all_records]),
        positions=np.asarray([record.position for record in all_records]),
        current=np.stack([record.case.current for record in all_records]),
        target_next=np.stack([record.case.target for record in all_records]),
        target_increment=np.stack([record.case.increment for record in all_records]),
        predicted_increment=predictions,
    )
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result": bool(config["science_result_eligible"]),
        "status": "completed",
        "variant": "full_pcno",
        "seed": config["seed"],
        "completed_steps": config["steps"],
        "selection_rule": config["selection_rule"],
        "train": train_scores,
        "held": held_scores,
        "capacity": capacity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_state_sha256": initial_model_sha256,
        "final_model_state_sha256": model_state_sha256(model),
        "elapsed_training_seconds_this_invocation": elapsed,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "gradient_ablation": run_contract["gradient_ablation_decision"],
    }
    write_json(output_dir / "summary.json", summary)

    from scripts.time_dependent_no.visualize_pcno_shock_overfit import (
        generate_visualizations,
    )

    visualization = generate_visualizations(output_dir)
    summary["visualizations"] = visualization
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {
            "schema": SCHEMA,
            "status": "completed",
            "completed_step": config["steps"],
            "steps": config["steps"],
            "fixed_grid_capacity_status": capacity["fixed_grid_capacity_status"],
        },
    )
    output_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result": bool(config["science_result_eligible"]),
        "status": "completed",
        "config_digest": config_digest,
        "population_sha256": population_digest,
        "source_sha256": source_hashes,
        "output_hashes": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in output_paths
        },
        "output_count": len(output_paths),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny non-scientific CPU-compatible contract smoke",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact saved contract from checkpoint_latest.pt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_experiment(
            args.output_dir,
            smoke=bool(args.smoke),
            device_name=str(args.device),
            resume=bool(args.resume),
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "run_state.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
