#!/usr/bin/env python3
"""Continue a P2 no-gradient checkpoint with or without zero-output gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcno.pcno import PCNO
from scripts.time_dependent_no.analyze_pcno_shock_pathways import (
    build_common_physical_cases,
)
from scripts.time_dependent_no.fit_pcno_shock_representation import (
    RegisteredCase,
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
    relative_squared_errors,
    sha256_file,
    write_checkpoint,
    write_json,
    write_npz,
)
from scripts.time_dependent_no.train_pcno_gradient_ablation import (
    FunctionalNoGradient,
    _environment,
    _git_state,
    _guard_summary,
    _predict_batched,
    _primary_metric,
    _scores,
    apply_variant,
    parameter_summary,
    shared_state_sha256,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import trace_pcno_branches
from utility.time_dependent_no.pcno_shock_representation import (
    REGISTERED_RESOLUTIONS,
    StructuredCellGrid,
    build_structured_cell_grid,
)

SCHEMA = "w26_l2_p2_c0_zero_gradient_continuation_v1"
WORKING_PREFIX = "W26-L2-P2-C0"
ARMS = ("continue_no_gradient", "activate_zero_gradient")
SEEDS = (1701, 1702, 1703)
BRANCHES = ("spectral", "pointwise", "differential")
BRANCH_MASKS = tuple(product((0, 1), repeat=3))
DEFAULT_ROOT = ROOT / "artifacts" / "time_dependent_no" / "w26_l2_p2_c0"
PARENT_POPULATION_SHA256 = (
    "0104edcdd1078b3b175647e880739081afd49217f0bc7efed773b39ed422c358"
)
PARENT_TRAINER_SHA256 = (
    "8c638b69987e68c5c7ecd4a1d3705de1f006e5824bd510d99fcfdfcab37a43c9"
)
PARENT_REGISTRY: Mapping[int, Mapping[str, str]] = {
    1701: {
        "checkpoint_sha256": "a383bbddb4c09914922010d76d7db634925c7819b5293de0eca100e635bd5f13",
        "config_digest": "7700bdc75e13c8a2a34b259fa35f8cf44d47a91ba314e476caa8c28833755f94",
        "model_state_sha256": "73dcd9e50997401f47ceb503886ae6dd755620da71a1d4bf3bb7af9906066825",
    },
    1702: {
        "checkpoint_sha256": "4f5f738be3628d1ec3149fb5ac7e46f7ba557a811d8e0cd56c10e61b314b50bd",
        "config_digest": "1cc48d084100f8896a40aca14514f69812562321deac19463e87b675b4e9e084",
        "model_state_sha256": "d7420231c3a9a10cac89040e84615d9b243b0db3044b6c3e8ca05a91652d0321",
    },
    1703: {
        "checkpoint_sha256": "b83f78692f8b6fa2f76b30f53cdf920b5bffeeade48102e1fc6f76b8ca6363bc",
        "config_digest": "db4bf7ac225f878a74c7b8e732466f84877856e3d7fe184a39bba4a05e012b57",
        "model_state_sha256": "7015a788c4a9f4a98c8a422277eaae2d2cfebe85bd60dde68c55367f3c8ef9c8",
    },
}
SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_shock_representation.py",
    "scripts/time_dependent_no/fit_pcno_shock_representation.py",
    "scripts/time_dependent_no/analyze_pcno_shock_pathways.py",
    "scripts/time_dependent_no/train_pcno_gradient_ablation.py",
    "scripts/time_dependent_no/continue_pcno_zero_gradient.py",
)
PROVENANCE_PATHS = ("docs/time_dependent_no/W26_L2_SHOCK_PATHWAY_PREREGISTRATION.md",)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def _arm_token(arm: str) -> str:
    return {
        "continue_no_gradient": "KEEP-NO-GRAD",
        "activate_zero_gradient": "ACTIVATE-ZERO-GRAD",
    }[arm]


def branch_mask_name(mask: Sequence[int]) -> str:
    if len(mask) != len(BRANCHES) or any(value not in (0, 1) for value in mask):
        raise ValueError("branch mask must contain three binary values")
    return "".join(
        f"{name[0].upper()}{value}" for name, value in zip(BRANCHES, mask, strict=True)
    )


def registered_config(
    parent_config: Mapping[str, Any],
    arm: str,
    seed: int,
    *,
    smoke: bool,
    device_type: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must lie in {ARMS}")
    if seed not in SEEDS and not smoke:
        raise ValueError(f"production seed must lie in {SEEDS}")
    config = json.loads(json.dumps(parent_config))
    config.update(
        {
            "schema": SCHEMA,
            "working_run_id": (
                f"{WORKING_PREFIX}-{_arm_token(arm)}-S{seed}"
                + ("-SMOKE" if smoke else "")
            ),
            "science_result_eligible": not smoke,
            "continuation_arm": arm,
            "seed": int(seed),
            "paired_seeds": list(SEEDS),
            "steps": 2 if smoke else 10_000,
            "evaluation_interval": 1 if smoke else 100,
            "checkpoint_interval": 1 if smoke else 500,
            "selection_rule": "final_continuation_step_only",
            "optimizer_restart": "fresh_matched_adamw_without_parent_moments",
            "zero_gradient_initialization": (
                "retain_parent_gw1_and_zero_only_bias_free_gw2_weight"
            ),
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
            "device_type": device_type,
            "branch_cube_order": list(BRANCHES),
            "branch_cube_masks": [branch_mask_name(mask) for mask in BRANCH_MASKS],
        }
    )
    return config


def unwrap_no_gradient_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the functional no-gradient wrapper keys back to a full PCNO."""

    result: dict[str, torch.Tensor] = {}
    wrapped = 0
    for name, value in state.items():
        if name.startswith("gws."):
            marker = ".original."
            if marker not in name:
                raise ValueError(f"unexpected no-gradient branch key: {name}")
            name = name.replace(marker, ".", 1)
            wrapped += 1
        if name in result:
            raise ValueError(f"duplicate transformed state key: {name}")
        result[name] = value
    if wrapped == 0:
        raise ValueError("parent state has no functional no-gradient wrapper keys")
    return result


def zero_gradient_outputs(model: PCNO) -> list[dict[str, float]]:
    """Zero only gw2, retaining gw1 so the branch can leave the zero-output state."""

    rows = []
    with torch.no_grad():
        for layer, module in enumerate(model.gws):
            gw1 = float(module.gw1.detach().cpu())
            module.gw2.weight.zero_()
            rows.append(
                {
                    "layer": layer,
                    "gw1": gw1,
                    "gw2_nonzero": float(torch.count_nonzero(module.gw2.weight)),
                }
            )
    return rows


def _tensor_digest(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _gradient_state(model: PCNO) -> list[dict[str, Any]]:
    rows = []
    for layer, module in enumerate(model.gws):
        original = (
            module.original if isinstance(module, FunctionalNoGradient) else module
        )
        rows.append(
            {
                "layer": layer,
                "functional_no_gradient": isinstance(module, FunctionalNoGradient),
                "gw1": float(original.gw1.detach().cpu()),
                "gw2_l2": float(
                    torch.linalg.vector_norm(original.gw2.weight.detach()).cpu()
                ),
                "gw2_max_abs": float(original.gw2.weight.detach().abs().max().cpu()),
                "trainable": any(
                    parameter.requires_grad for parameter in module.parameters()
                ),
            }
        )
    return rows


def _backward_state(model: PCNO) -> dict[str, Any]:
    branch_rows = []
    shared = []
    for name, parameter in model.named_parameters():
        if name.startswith("gws."):
            continue
        if parameter.grad is not None:
            shared.append((name, parameter.grad))
    for layer, module in enumerate(model.gws):
        original = (
            module.original if isinstance(module, FunctionalNoGradient) else module
        )
        gw1_grad = original.gw1.grad
        gw2_grad = original.gw2.weight.grad
        branch_rows.append(
            {
                "layer": layer,
                "gw1_gradient_l2": (
                    None
                    if gw1_grad is None
                    else float(torch.linalg.vector_norm(gw1_grad.detach()).cpu())
                ),
                "gw2_gradient_l2": (
                    None
                    if gw2_grad is None
                    else float(torch.linalg.vector_norm(gw2_grad.detach()).cpu())
                ),
                "finite": bool(
                    (gw1_grad is None or torch.isfinite(gw1_grad).all())
                    and (gw2_grad is None or torch.isfinite(gw2_grad).all())
                ),
            }
        )
    shared_l2 = math.sqrt(
        sum(
            float(torch.sum(tensor.detach().double().square()).cpu())
            for _, tensor in shared
        )
    )
    return {
        "branches": branch_rows,
        "shared_gradient_l2": shared_l2,
        "shared_gradient_sha256": _tensor_digest(shared),
    }


def _verify_parent(
    parent_dir: Path,
    *,
    seed: int,
    enforce_registry: bool,
) -> dict[str, Any]:
    required = ("checkpoint.pt", "manifest.json", "run_contract.json", "summary.json")
    missing = [name for name in required if not (parent_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"parent directory is missing {missing}")
    checkpoint_path = parent_dir / "checkpoint.pt"
    manifest = _read_json(parent_dir / "manifest.json")
    contract = _read_json(parent_dir / "run_contract.json")
    summary = _read_json(parent_dir / "summary.json")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if manifest["output_hashes"].get("checkpoint.pt") != checkpoint_sha256:
        raise ValueError("parent checkpoint does not close to its manifest")
    if contract["config_digest"] != manifest["config_digest"]:
        raise ValueError("parent config digest does not close")
    if contract["population"]["sha256"] != manifest["population_sha256"]:
        raise ValueError("parent population digest does not close")
    if summary.get("variant") != "no_gradient" or int(summary.get("seed")) != seed:
        raise ValueError("parent is not the requested P2 no-gradient seed")
    if summary.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("parent run is incomplete")
    if enforce_registry:
        expected = PARENT_REGISTRY[seed]
        observed = {
            "checkpoint_sha256": checkpoint_sha256,
            "config_digest": manifest["config_digest"],
            "model_state_sha256": summary["final_model_state_sha256"],
        }
        if observed != expected:
            raise ValueError("parent no-gradient checkpoint differs from the registry")
        if manifest["population_sha256"] != PARENT_POPULATION_SHA256:
            raise ValueError(
                "parent population differs from the registered P2 population"
            )
        trainer = contract["source_sha256"].get(
            "scripts/time_dependent_no/train_pcno_gradient_ablation.py"
        )
        if trainer != PARENT_TRAINER_SHA256:
            raise ValueError(
                "parent trainer source differs from the registered P2 source"
            )
    return {
        "path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": sha256_file(parent_dir / "manifest.json"),
        "contract_sha256": sha256_file(parent_dir / "run_contract.json"),
        "summary_sha256": sha256_file(parent_dir / "summary.json"),
        "manifest": manifest,
        "contract": contract,
        "summary": summary,
    }


@torch.inference_mode()
def _predict_with_mask(
    model: PCNO,
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
    model.eval()
    predictions = []
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


def _branch_cube(
    model: PCNO,
    populations: Sequence[tuple[str, StructuredCellGrid, list[RegisteredCase]]],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {"branch_order": list(BRANCHES), "populations": {}}
    arrays: dict[str, np.ndarray] = {}
    for population_name, grid, records in populations:
        population_result = {}
        direct = _predict_batched(model, records, grid, device, batch_size=batch_size)
        for mask in BRANCH_MASKS:
            name = branch_mask_name(mask)
            prediction = _predict_with_mask(
                model,
                records,
                grid,
                device,
                mask=mask,
                batch_size=batch_size,
            )
            rows = case_metric_rows(records, prediction, grid)
            population_result[name] = {
                "mask": list(mask),
                "scores": _scores(records, prediction, grid),
                "primary_value": _primary_metric(rows),
                "case_metrics": rows,
            }
            arrays[f"{population_name}__{name}"] = prediction
            if mask == (1, 1, 1):
                population_result[name]["direct_replay_max_abs"] = float(
                    np.max(np.abs(prediction - direct))
                )
        metrics["populations"][population_name] = population_result
    return metrics, arrays


def _checkpoint_payload(
    *,
    model: PCNO,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_step: int,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
    config_digest: str,
    parent: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": dict(config),
        "config_digest": config_digest,
        "parent": dict(parent),
        "initialization": dict(initialization),
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


def run_experiment(
    output_dir: Path,
    parent_dir: Path,
    *,
    arm: str,
    seed: int,
    smoke: bool,
    device_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must lie in {ARMS}")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not smoke and device.type != "cuda":
        raise RuntimeError("the production P2-C0 contract requires CUDA")
    parent_dir = parent_dir.resolve()
    parent = _verify_parent(parent_dir, seed=seed, enforce_registry=not smoke)
    parent_config = parent["contract"]["config"]
    config = registered_config(
        parent_config, arm, seed, smoke=smoke, device_type=device.type
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

    checkpoint = torch.load(parent["path"], map_location=device, weights_only=False)
    if checkpoint["config_digest"] != parent["manifest"]["config_digest"]:
        raise ValueError("parent checkpoint config digest differs")
    parent_model = apply_variant(build_model(parent_config, device), "no_gradient")
    parent_model.load_state_dict(checkpoint["model_state"], strict=True)
    if (
        model_state_sha256(parent_model)
        != parent["summary"]["final_model_state_sha256"]
    ):
        raise ValueError("loaded parent model does not close to the parent summary")

    if arm == "continue_no_gradient":
        model = parent_model
        initialization_rows = _gradient_state(model)
    else:
        model = build_model(parent_config, device)
        model.load_state_dict(
            unwrap_no_gradient_state(checkpoint["model_state"]), strict=True
        )
        initialization_rows = zero_gradient_outputs(model)
    shared_initialization = shared_state_sha256(model)
    parent_shared_initialization = shared_state_sha256(parent_model)
    if shared_initialization != parent_shared_initialization:
        raise ValueError("continuation changed a non-gradient parent parameter")

    grid = build_structured_cell_grid(config["resolution"])
    train_records = build_registered_cases(grid, config, split="train")
    held_records = build_registered_cases(grid, config, split="held")
    all_records = train_records + held_records
    population_digest = population_sha256(all_records)
    if population_digest != parent["manifest"]["population_sha256"]:
        raise ValueError("continuation population differs from the parent population")
    train_input, train_target, train_aux = build_batch(train_records, grid, device)
    held_input, held_target, held_aux = build_batch(held_records, grid, device)
    train_fourier = expand_fourier_tensors(model, train_aux, len(train_records))
    held_fourier = expand_fourier_tensors(model, held_aux, len(held_records))

    parent_prediction = _predict_batched(
        parent_model,
        all_records,
        grid,
        device,
        batch_size=int(config["evaluation_batch_size"]),
    )
    initial_prediction = _predict_batched(
        model,
        all_records,
        grid,
        device,
        batch_size=int(config["evaluation_batch_size"]),
    )
    output_closure = float(np.max(np.abs(initial_prediction - parent_prediction)))
    closure_tolerance = 1.0e-5 if device.type == "cuda" else 0.0
    if output_closure > closure_tolerance:
        raise ValueError("zero-output continuation does not close to its parent")

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
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

    model.train()
    optimizer.zero_grad(set_to_none=True)
    diagnostic_prediction = model(train_input, train_aux, fourier_tensors=train_fourier)
    diagnostic_loss = torch.mean(
        relative_squared_errors(diagnostic_prediction, train_target, train_aux[2])
    )
    diagnostic_loss.backward()
    first_backward = _backward_state(model)
    if not all(row["finite"] for row in first_backward["branches"]):
        raise FloatingPointError(
            "nonfinite differential gradient in step-zero diagnostic"
        )
    if arm == "activate_zero_gradient":
        gw2_norms = [row["gw2_gradient_l2"] for row in first_backward["branches"]]
        if not any(value is not None and value > 0.0 for value in gw2_norms):
            raise RuntimeError(
                "zero-output differential branch has no material gw2 gradient"
            )
        if any(
            row["gw1_gradient_l2"] is None or row["gw1_gradient_l2"] != 0.0
            for row in first_backward["branches"]
        ):
            raise RuntimeError(
                "gw1 must have exact zero gradient while gw2 is exactly zero"
            )
    optimizer.zero_grad(set_to_none=True)

    initialization = {
        "parent_model_state_sha256": parent["summary"]["final_model_state_sha256"],
        "shared_non_gradient_sha256": shared_initialization,
        "parent_shared_non_gradient_sha256": parent_shared_initialization,
        "initial_model_state_sha256": model_state_sha256(model),
        "parent_output_max_abs": output_closure,
        "output_closure_tolerance": closure_tolerance,
        "gradient_state": initialization_rows,
        "first_backward": first_backward,
        "fresh_optimizer_state_entry_count": len(optimizer.state),
    }
    parent_reference = {
        "working_run_id": parent["summary"]["working_run_id"],
        "primary_value": parent["summary"]["primary_value"],
        "train_mean_relative_increment_l2": parent["summary"]["train"][
            "mean_relative_increment_l2"
        ],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "manifest_sha256": parent["manifest_sha256"],
        "contract_sha256": parent["contract_sha256"],
        "summary_sha256": parent["summary_sha256"],
        "config_digest": parent["manifest"]["config_digest"],
        "population_sha256": parent["manifest"]["population_sha256"],
    }
    run_contract = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "config": config,
        "config_digest": config_digest,
        "population": {
            "kind": "same_ordered_analytic_cell_average_population_as_parent",
            "sha256": population_digest,
            "train_case_ids": [record.case_id for record in train_records],
            "held_case_ids": [record.case_id for record in held_records],
        },
        "parent": parent_reference,
        "initialization": initialization,
        "parameter_summary": parameter_summary(model),
        "source_sha256": _source_hashes(SOURCE_PATHS),
        "provenance_sha256": _source_hashes(PROVENANCE_PATHS),
        "git": _git_state(),
        "environment": _environment(device),
    }
    contract_path = output_dir / "run_contract.json"
    if resume:
        saved = _read_json(contract_path)
        for key in ("config_digest", "population", "parent", "source_sha256"):
            if saved[key] != run_contract[key]:
                raise ValueError(f"resume {key} differs")
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

    history: list[dict[str, Any]] = []
    start_step = 1
    checkpoint_latest = output_dir / "checkpoint_latest.pt"
    if resume:
        resume_path = (
            checkpoint_latest
            if checkpoint_latest.is_file()
            else output_dir / "checkpoint.pt"
        )
        saved_checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if saved_checkpoint["config_digest"] != config_digest:
            raise ValueError("resume checkpoint config digest differs")
        model.load_state_dict(saved_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(saved_checkpoint["optimizer_state"])
        scheduler.load_state_dict(saved_checkpoint["scheduler_state"])
        history = list(saved_checkpoint["history"])
        start_step = int(saved_checkpoint["completed_step"]) + 1
        torch.set_rng_state(saved_checkpoint["torch_rng_state"])
        if device.type == "cuda" and saved_checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(saved_checkpoint["cuda_rng_state"])

    def append_evaluation(
        step: int, optimizer_loss: float | None, backward: Mapping[str, Any] | None
    ) -> None:
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
            model, held_input, held_target, held_aux, held_fourier, held_records, grid
        )
        history.append(
            {
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_loss": optimizer_loss,
                "train": train_scores,
                "held": held_scores,
                "gradient_parameters": _gradient_state(model),
                "backward": None if backward is None else dict(backward),
            }
        )
        write_json(output_dir / "history.json", history)

    if not history:
        append_evaluation(0, float(diagnostic_loss.detach()), first_backward)
    started = time.perf_counter()
    last_loss: float | None = None
    last_backward: dict[str, Any] | None = None
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
        last_backward = _backward_state(model)
        gradients = [
            parameter.grad for parameter in trainable if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(value).all()) for value in gradients
        ):
            raise FloatingPointError(f"nonfinite or missing gradients at step {step}")
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % int(config["evaluation_interval"]) == 0:
            append_evaluation(step, last_loss, last_backward)
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
                        "gradient_gw2_l2": [
                            row["gw2_l2"] for row in history[-1]["gradient_parameters"]
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
                    parent=parent_reference,
                    initialization=initialization,
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
        append_evaluation(int(config["steps"]), last_loss, last_backward)
    train_prediction, train_scores = evaluate_split(
        model, train_input, train_target, train_aux, train_fourier, train_records, grid
    )
    held_prediction, held_scores = evaluate_split(
        model, held_input, held_target, held_aux, held_fourier, held_records, grid
    )
    native_prediction = np.concatenate((train_prediction, held_prediction), axis=0)
    native_rows = case_metric_rows(train_records, train_prediction, grid)
    native_rows.extend(case_metric_rows(held_records, held_prediction, grid))

    nested_rows: list[dict[str, Any]] = []
    nested_arrays: dict[str, np.ndarray] = {}
    nested_metadata: dict[str, Any] = {}
    nested_guards: dict[str, Any] = {}
    nested_scores: dict[str, Any] = {}
    cube_populations: list[tuple[str, StructuredCellGrid, list[RegisteredCase]]] = [
        ("native", grid, all_records)
    ]
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
        nested_guards[key] = _guard_summary(rows, nested_grid)
        nested_scores[key] = _scores(records, predictions, nested_grid)
        cube_populations.append((key, nested_grid, records))

    branch_metrics, branch_arrays = _branch_cube(
        model,
        cube_populations,
        device,
        batch_size=int(config["evaluation_batch_size"]),
    )
    replay_max = max(
        float(population[branch_mask_name((1, 1, 1))]["direct_replay_max_abs"])
        for population in branch_metrics["populations"].values()
    )
    if replay_max > 1.0e-5:
        raise ValueError("all-on branch replay does not close to direct PCNO output")

    final_checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_step=int(config["steps"]),
        history=history,
        config=config,
        config_digest=config_digest,
        parent=parent_reference,
        initialization=initialization,
    )
    write_checkpoint(output_dir / "checkpoint.pt", final_checkpoint)
    if checkpoint_latest.is_file():
        checkpoint_latest.unlink()
    write_json(output_dir / "case_metrics.json", native_rows)
    write_json(output_dir / "nested_case_metrics.json", nested_rows)
    write_json(output_dir / "nested_metadata.json", nested_metadata)
    write_json(output_dir / "branch_cube_metrics.json", branch_metrics)
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
        parent_predicted_increment=parent_prediction,
        initial_predicted_increment=initial_prediction,
        predicted_increment=native_prediction,
    )
    write_npz(output_dir / "nested_predictions.npz", **nested_arrays)
    write_npz(output_dir / "branch_cube_predictions.npz", **branch_arrays)
    capacity = capacity_summary(native_rows, grid_spacing=grid.hx)
    primary = _primary_metric(native_rows)
    summary = {
        "schema": SCHEMA,
        "working_run_id": config["working_run_id"],
        "science_result": bool(config["science_result_eligible"]),
        "status": "completed",
        "arm": arm,
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
        "parameter_summary": parameter_summary(model),
        "parent": parent_reference,
        "initialization": initialization,
        "final_gradient_state": _gradient_state(model),
        "final_model_state_sha256": model_state_sha256(model),
        "branch_replay_max_abs": replay_max,
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
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "source_sha256": run_contract["source_sha256"],
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "output_count": len(outputs),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--parent-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_ROOT / f"{args.arm}_s{args.seed}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment(
        args.output_dir,
        args.parent_dir,
        arm=args.arm,
        seed=args.seed,
        smoke=args.smoke,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
