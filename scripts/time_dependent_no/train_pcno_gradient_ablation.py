#!/usr/bin/env python3
"""Train one arm of the matched W26-L2-P2 PCNO gradient ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import PCNO, graph_neighbor_average
from scripts.time_dependent_no.analyze_pcno_shock_pathways import (
    build_common_physical_cases,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
    _case_guard,
    build_batch,
    build_model,
    build_registered_cases,
    capacity_summary,
    case_metric_rows,
    evaluate_split,
    expand_fourier_tensors,
    mapping_sha256,
    model_state_sha256,
    population_sha256,
    registered_config as p0_registered_config,
    relative_squared_errors,
    sha256_file,
    write_checkpoint,
    write_json,
    write_npz,
)
from utility.time_dependent_no.pcno_shock_representation import (
    CASE_FAMILIES,
    REGISTERED_RESOLUTIONS,
    StructuredCellGrid,
    build_structured_cell_grid,
)

SCHEMA = "w26_l2_p2_matched_gradient_ablation_v1"
WORKING_PREFIX = "W26-L2-P2"
VARIANTS = ("full", "no_gradient", "local_replacement")
SEEDS = (1701, 1702, 1703)
DEFAULT_ROOT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2"
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/cpg_mesh_contract.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "scripts/time_dependent_no/train_pcno_gradient_ablation.py",
)
PROVENANCE_PATHS = (
    "docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",
)


class FunctionalNoGradient(nn.Module):
    """Keep the initialized gradient module frozen while returning exact zero."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        for parameter in self.original.parameters():
            parameter.requires_grad_(False)
        self.out_channels = int(self.original.gw2.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        neighbor_degree: torch.Tensor | None = None,
        flat_edge_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del directed_edges, edge_gradient_weights, neighbor_degree, flat_edge_indices
        return x.new_zeros((x.shape[0], self.out_channels, x.shape[2]))


class ParameterMatchedLocalReplacement(nn.Module):
    """Two-hop local capacity control with exactly ``2*C^2+1`` parameters."""

    def __init__(self, channels: int, *, initial_scale: float = 0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(initial_scale)))
        self.v1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.v2 = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        neighbor_degree: torch.Tensor | None = None,
        flat_edge_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del edge_gradient_weights
        local = graph_neighbor_average(
            self.v1(x),
            directed_edges,
            iterations=2,
            neighbor_degree=neighbor_degree,
            flat_edge_indices=flat_edge_indices,
        )
        return self.v2(F.softsign(self.alpha * local))


def _variant_token(variant: str) -> str:
    return {
        "full": "FULL",
        "no_gradient": "NO-GRAD",
        "local_replacement": "LOCAL",
    }[variant]


def registered_config(
    variant: str,
    seed: int,
    *,
    smoke: bool,
    device_type: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must lie in {VARIANTS}")
    if seed not in SEEDS and not smoke:
        raise ValueError(f"production seed must lie in {SEEDS}")
    config = p0_registered_config(smoke=smoke, device_type=device_type)
    config.update(
        {
            "schema": SCHEMA,
            "working_run_id": (
                f"{WORKING_PREFIX}-{_variant_token(variant)}-S{seed}"
                + ("-SMOKE" if smoke else "")
            ),
            "science_result_eligible": not smoke,
            "variant": variant,
            "seed": int(seed),
            "paired_seeds": list(SEEDS),
            "primary_metric": (
                "native_held_phase_0p875_discontinuous_mean_relative_increment_l2"
            ),
            "primary_direction": "lower_is_better",
            "paired_win_minimum_seeds": 2,
            "paired_median_improvement_minimum": 0.05,
            "control_ratio_maximum": 1.05,
            "evaluation_resolutions": (
                [list(config["resolution"])]
                if smoke
                else [list(value) for value in REGISTERED_RESOLUTIONS]
            ),
            "evaluation_batch_size": 2 if smoke else 8,
        }
    )
    return config


def apply_variant(model: PCNO, variant: str) -> PCNO:
    """Apply one registered arm after common PCNO initialization."""

    if variant == "full":
        return model
    if variant == "no_gradient":
        model.gws = nn.ModuleList(
            [FunctionalNoGradient(module) for module in model.gws]
        )
        return model
    if variant == "local_replacement":
        channels = [int(module.gw2.out_channels) for module in model.gws]
        reference = next(model.parameters())
        model.gws = nn.ModuleList(
            [
                ParameterMatchedLocalReplacement(value).to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for value in channels
            ]
        )
        return model
    raise ValueError(f"variant must lie in {VARIANTS}")


def parameter_summary(model: PCNO) -> dict[str, int]:
    gradient_total = sum(parameter.numel() for module in model.gws for parameter in module.parameters())
    gradient_trainable = sum(
        parameter.numel()
        for module in model.gws
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "gradient_or_replacement_total": gradient_total,
        "gradient_or_replacement_trainable": gradient_trainable,
    }


def shared_state_sha256(model: PCNO) -> str:
    """Hash initialized parameters shared by all arms, excluding ``gws``."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("gws."):
            continue
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
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
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "pid": os.getpid(),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_payload(
    *,
    model: PCNO,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_step: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    config_digest: str,
    shared_initialization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": config,
        "config_digest": config_digest,
        "shared_initialization_sha256": shared_initialization_sha256,
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


@torch.inference_mode()
def _predict_batched(
    model: PCNO,
    records: list[RegisteredCase],
    grid: StructuredCellGrid,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        model_input, _, aux = build_batch(selected, grid, device)
        fourier = expand_fourier_tensors(model, aux, len(selected))
        output = model(model_input, aux, fourier_tensors=fourier)
        predictions.append(
            output[..., 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(len(selected), grid.ny, grid.nx)
        )
    return np.concatenate(predictions, axis=0)


def _scores(
    records: list[RegisteredCase], predictions: np.ndarray, grid: StructuredCellGrid
) -> dict[str, Any]:
    values = []
    for record, prediction in zip(records, predictions, strict=True):
        numerator = float(np.sum(np.square(prediction - record.case.increment)))
        denominator = float(np.sum(np.square(record.case.increment)))
        values.append(math.sqrt(numerator / max(denominator, np.finfo(float).tiny)))
    values_array = np.asarray(values)
    families = {}
    for family in CASE_FAMILIES:
        selected = [
            values[index]
            for index, record in enumerate(records)
            if record.family == family
        ]
        if selected:
            families[family] = {
                "mean_relative_increment_l2": float(np.mean(selected)),
                "maximum_relative_increment_l2": float(np.max(selected)),
            }
    return {
        "mean_relative_increment_l2": float(np.mean(values_array)),
        "maximum_relative_increment_l2": float(np.max(values_array)),
        "families": families,
    }


def _guard_summary(
    rows: list[dict[str, Any]], grid: StructuredCellGrid
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("step", "pulse"):
        selected = [row for row in rows if row["family"] == family]
        decisions = [
            _case_guard(row, grid_spacing=grid.hx, held=True) for row in selected
        ]
        result[family] = {
            "passed_case_count": sum(item["passed"] for item in decisions),
            "case_count": len(decisions),
            "ten_of_twelve_gate": len(decisions) == 12
            and sum(item["passed"] for item in decisions) >= 10,
            "case_decisions": decisions,
        }
    return result


def _primary_metric(rows: list[dict[str, Any]]) -> float:
    selected = [
        row["metrics"]["relative_increment_l2"]
        for row in rows
        if row["split"] == "held"
        and row["family"] in {"step", "pulse"}
        and abs(float(row["phase"]) - 0.875) < 1.0e-12
    ]
    if not selected:
        selected = [
            row["metrics"]["relative_increment_l2"]
            for row in rows
            if row["split"] == "held" and row["family"] in {"step", "pulse"}
        ]
    return float(np.mean(selected))


def run_experiment(
    output_dir: Path,
    *,
    variant: str,
    seed: int,
    smoke: bool,
    device_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    if not smoke and device.type != "cuda":
        raise RuntimeError("the production P2 contract requires CUDA")
    config = registered_config(
        variant, seed, smoke=smoke, device_type=device.type
    )
    config_digest = mapping_sha256(config)
    output_dir = output_dir.resolve()
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    grid = build_structured_cell_grid(config["resolution"])
    train_records = build_registered_cases(grid, config, split="train")
    held_records = build_registered_cases(grid, config, split="held")
    all_records = train_records + held_records
    population_digest = population_sha256(all_records)
    model = apply_variant(build_model(config, device), variant)
    initial_model_sha256 = model_state_sha256(model)
    shared_initialization = shared_state_sha256(model)
    parameters = parameter_summary(model)
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
        "variant_semantics": {
            "full": "maintained PCNO differential branch",
            "no_gradient": (
                "initialized differential parameters retained frozen and excluded "
                "from optimization; forward response is exact zero"
            ),
            "local_replacement": (
                "V2(Softsign(alpha*A_h^2*V1(x))); bias-free C-to-C maps; "
                "2*C^2+1 active parameters per block"
            ),
        }[variant],
        "parameter_summary": parameters,
        "shared_initialization_sha256": shared_initialization,
        "source_sha256": _source_hashes(SOURCE_PATHS),
        "provenance_sha256": _source_hashes(PROVENANCE_PATHS),
        "git": _git_state(),
        "environment": _environment(device),
    }
    contract_path = output_dir / "run_contract.json"
    if resume:
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved["config_digest"] != config_digest:
            raise ValueError("resume config digest differs")
        if saved["population"]["sha256"] != population_digest:
            raise ValueError("resume population digest differs")
        if saved["source_sha256"] != run_contract["source_sha256"]:
            raise ValueError("resume executable source hashes differ")
        if saved["shared_initialization_sha256"] != shared_initialization:
            raise ValueError("resume shared initialization differs")
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

    train_input, train_target, train_aux = build_batch(train_records, grid, device)
    held_input, held_target, held_aux = build_batch(held_records, grid, device)
    train_fourier = expand_fourier_tensors(model, train_aux, len(train_records))
    held_fourier = expand_fourier_tensors(model, held_aux, len(held_records))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
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
        resume_path = (
            checkpoint_latest
            if checkpoint_latest.is_file()
            else output_dir / "checkpoint.pt"
        )
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad for parameter in trainable if parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
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
                    shared_initialization_sha256=shared_initialization,
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
    native_predictions = np.concatenate((train_prediction, held_prediction), axis=0)
    native_rows = case_metric_rows(train_records, train_prediction, grid)
    native_rows.extend(case_metric_rows(held_records, held_prediction, grid))
    capacity = capacity_summary(native_rows, grid_spacing=grid.hx)
    primary = _primary_metric(native_rows)

    nested_rows: list[dict[str, Any]] = []
    nested_arrays: dict[str, np.ndarray] = {}
    nested_metadata: dict[str, Any] = {}
    nested_guards: dict[str, Any] = {}
    nested_scores: dict[str, Any] = {}
    for resolution in config["evaluation_resolutions"]:
        if tuple(resolution) == tuple(config["resolution"]):
            continue
        nested_grid = build_structured_cell_grid(resolution)
        records = build_common_physical_cases(nested_grid, config, split="held")
        predictions = _predict_batched(
            model,
            records,
            nested_grid,
            device,
            batch_size=int(config["evaluation_batch_size"]),
        )
        rows = case_metric_rows(records, predictions, nested_grid)
        key = f"{nested_grid.nx}x{nested_grid.ny}"
        for row in rows:
            row["resolution"] = list(nested_grid.resolution)
        nested_rows.extend(rows)
        nested_arrays[f"predicted_increment_{key}"] = predictions
        nested_arrays[f"target_increment_{key}"] = np.stack(
            [record.case.increment for record in records]
        )
        nested_metadata[key] = {
            "case_ids": [record.case_id for record in records],
            "families": [record.family for record in records],
            "phases": [record.phase for record in records],
            "positions": [record.position for record in records],
        }
        nested_guards[key] = _guard_summary(
            [row for row in rows if row["family"] in {"step", "pulse"}],
            nested_grid,
        )
        nested_scores[key] = _scores(records, predictions, nested_grid)

    final_checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=int(config["steps"]),
        history=history,
        config=config,
        config_digest=config_digest,
        shared_initialization_sha256=shared_initialization,
    )
    write_checkpoint(output_dir / "checkpoint.pt", final_checkpoint)
    if checkpoint_latest.is_file():
        checkpoint_latest.unlink()
    write_json(output_dir / "case_metrics.json", native_rows)
    write_json(output_dir / "nested_case_metrics.json", nested_rows)
    write_json(output_dir / "nested_metadata.json", nested_metadata)
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
        predicted_increment=native_predictions,
    )
    write_npz(output_dir / "nested_predictions.npz", **nested_arrays)
    summary = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result": bool(config["science_result_eligible"]),
        "status": "completed",
        "variant": variant,
        "seed": seed,
        "completed_steps": config["steps"],
        "selection_rule": config["selection_rule"],
        "primary_metric": config["primary_metric"],
        "primary_value": primary,
        "train": train_scores,
        "held": held_scores,
        "capacity": capacity,
        "nested_scores": nested_scores,
        "nested_guards": nested_guards,
        "parameter_summary": parameters,
        "shared_initialization_sha256": shared_initialization,
        "initial_model_state_sha256": initial_model_sha256,
        "final_model_state_sha256": model_state_sha256(model),
        "elapsed_training_seconds_this_invocation": elapsed,
        "maximum_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_state.json",
        {
            "schema": SCHEMA,
            "status": "completed",
            "completed_step": config["steps"],
            "steps": config["steps"],
            "primary_value": primary,
        },
    )
    outputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result": bool(config["science_result_eligible"]),
        "status": "completed",
        "config_digest": config_digest,
        "population_sha256": population_digest,
        "source_sha256": run_contract["source_sha256"],
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_ROOT / f"{args.variant}_s{args.seed}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment(
        args.output_dir,
        variant=args.variant,
        seed=args.seed,
        smoke=args.smoke,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
