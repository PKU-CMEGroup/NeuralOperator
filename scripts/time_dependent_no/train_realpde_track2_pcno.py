#!/usr/bin/env python3
"""Train a fixed-geometry PCNO forecast checkpoint for RealPDE Track 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_torch_save,
    git_state,
    runtime_environment,
    sha256_file,
    write_json,
)
from utility.time_dependent_no.pcno_runtime import (
    autocast_context,
    select_device,
    synchronize,
)
from utility.time_dependent_no.realpde_track2 import (
    TRACK2_INPUT_STEPS,
    RealPDETrack2PCNO,
    Track2FileMetadata,
    Track2Geometry,
    Track2Normalization,
    Track2WindowDataset,
    discover_track2_files,
    estimate_track2_residual_scale,
    fit_track2_frame_normalization,
    load_track2_geometry_from_simulation,
    load_track2_geometry_from_simulations,
    parameter_count,
    split_track2_files,
    track2_error_tensors,
    track2_forecast_score,
    track2_training_loss,
)

MODEL_PAYLOAD_SCHEMA = "realpde_track2_pcno_model_v2"
TRAINING_PAYLOAD_SCHEMA = "realpde_track2_pcno_training_v2"
STAGES = ("sim_pretrain", "real_finetune")
TRANSFER_SCALING_KEYS = {
    "input_mean",
    "target_mean",
    "input_std",
    "target_std",
    "residual_scale",
}
SOURCE_FILES = (
    "scripts/time_dependent_no/train_realpde_track2_pcno.py",
    "utility/time_dependent_no/realpde_track2.py",
    "utility/time_dependent_no/pcno_boundary_fields.py",
    "pcno/pcno.py",
    "pcno/geo_utility.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-simulation",
        type=Path,
        default=None,
        help="Simulation HDF5 used for released-grid coordinate provenance.",
    )
    parser.add_argument(
        "--normalization-path",
        type=Path,
        default=None,
        help="Required official real-data statistics for real_finetune.",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Model-only simulation checkpoint used to initialize real fine-tuning.",
    )
    checkpoint.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Training checkpoint from an interrupted run of the same stage.",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument(
        "--maximum-files",
        type=int,
        default=None,
        help="Qualification-only cap applied before splitting.",
    )
    parser.add_argument("--collar-width", type=float, default=0.01)
    parser.add_argument("--modes", type=int, nargs=2, default=(20, 10))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=(96, 96, 96, 96, 96),
    )
    parser.add_argument("--fc-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.02)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--residual-scale-windows", type=int, default=4096)
    parser.add_argument("--relative-l2-weight", type=float, default=0.05)
    parser.add_argument("--tke-weight", type=float, default=0.05)
    parser.add_argument("--mvpe-weight", type=float, default=0.05)
    parser.add_argument("--boundary-weight", type=float, default=0.10)
    parser.add_argument("--free-rollout-loss-weight", type=float, default=0.25)
    parser.add_argument("--free-rollout-loss-every", type=int, default=4)
    parser.add_argument("--evaluation-every", type=int, default=500)
    parser.add_argument("--evaluation-windows", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--rollout-trajectories", type=int, default=8)
    parser.add_argument("--rollout-blocks", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp",
        choices=("auto", "none", "bf16", "fp16"),
        default="auto",
    )
    args = parser.parse_args(argv)
    return validate_args(args)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    positive_names = (
        "window_stride",
        "batch_size",
        "gradient_accumulation_steps",
        "fc_dim",
        "residual_scale_windows",
        "evaluation_every",
        "evaluation_windows",
        "evaluation_batch_size",
        "rollout_trajectories",
        "rollout_blocks",
        "free_rollout_loss_every",
        "checkpoint_every",
    )
    for name in positive_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.maximum_files is not None and args.maximum_files < 2:
        raise ValueError("--maximum-files must be at least two")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must lie strictly between 0 and 0.5")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must lie in [0, 1)")
    if not 0.0 < args.minimum_learning_rate_ratio <= 1.0:
        raise ValueError("--minimum-learning-rate-ratio must lie in (0, 1]")
    if args.gradient_clip <= 0.0 or args.weight_decay < 0.0:
        raise ValueError(
            "gradient clipping must be positive and weight decay nonnegative"
        )
    if args.collar_width <= 0.0:
        raise ValueError("--collar-width must be positive")
    if any(value < 1 for value in args.modes):
        raise ValueError("--modes must contain positive integers")
    if len(args.layers) < 2 or any(value < 1 for value in args.layers):
        raise ValueError("--layers must contain at least two positive widths")
    for name in (
        "relative_l2_weight",
        "tke_weight",
        "mvpe_weight",
        "boundary_weight",
        "free_rollout_loss_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")

    if args.max_updates is None:
        args.max_updates = 16000 if args.stage == "sim_pretrain" else 8000
    if args.learning_rate is None:
        args.learning_rate = 1.0e-3 if args.stage == "sim_pretrain" else 2.0e-4
    if args.max_updates < 1 or args.learning_rate <= 0.0:
        raise ValueError("maximum updates and learning rate must be positive")

    if args.stage == "real_finetune":
        if args.normalization_path is None:
            raise ValueError("real_finetune requires --normalization-path")
        if args.geometry_simulation is None:
            raise ValueError("real_finetune requires --geometry-simulation")
        if args.init_checkpoint is None and args.resume_checkpoint is None:
            raise ValueError(
                "real_finetune requires --init-checkpoint or --resume-checkpoint"
            )
    return args


def resolve_amp(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type != "cuda":
            return "none"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")
    return requested


def digest_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} does not contain a checkpoint mapping")
    return dict(value)


def metadata_manifest(
    metadata: Sequence[Track2FileMetadata],
) -> dict[str, Any]:
    files = []
    for item in metadata:
        files.append(
            {
                **item.to_mapping(),
                "bytes": int(item.path.stat().st_size),
                "sha256": sha256_file(item.path),
            }
        )
    return {
        "files": files,
        "digest": digest_mapping({"files": files}),
    }


def select_files(
    metadata: Sequence[Track2FileMetadata],
    maximum_files: int | None,
) -> list[Track2FileMetadata]:
    selected = list(metadata)
    if maximum_files is None or maximum_files >= len(selected):
        return selected
    grouped: dict[int, list[Track2FileMetadata]] = {}
    for item in selected:
        grouped.setdefault(item.angle_of_attack, []).append(item)
    result: list[Track2FileMetadata] = []
    angles = sorted(grouped)
    offsets = {angle: 0 for angle in angles}
    while len(result) < maximum_files:
        progressed = False
        for angle in angles:
            offset = offsets[angle]
            if offset < len(grouped[angle]) and len(result) < maximum_files:
                result.append(grouped[angle][offset])
                offsets[angle] += 1
                progressed = True
        if not progressed:
            break
    return result


def source_manifest() -> dict[str, Any]:
    files = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    return {"files": files, "digest": digest_mapping(files)}


def stage_normalization(
    args: argparse.Namespace,
    train_metadata: Sequence[Track2FileMetadata],
) -> Track2Normalization:
    if args.normalization_path is not None:
        return Track2Normalization.from_file(args.normalization_path)
    if args.stage != "sim_pretrain":
        raise ValueError("real data must use the released normalization file")
    return fit_track2_frame_normalization(train_metadata)


def schedule_learning_rate(
    update: int,
    *,
    max_updates: int,
    base_learning_rate: float,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    warmup_updates = round(max_updates * warmup_fraction)
    if warmup_updates > 0 and update <= warmup_updates:
        progress = update / warmup_updates
        multiplier = 0.1 + 0.9 * progress
    else:
        denominator = max(max_updates - warmup_updates, 1)
        progress = min(max((update - warmup_updates) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = minimum_ratio + (1.0 - minimum_ratio) * cosine
    return float(base_learning_rate * multiplier)


def validation_indices(length: int, count: int) -> np.ndarray:
    return np.linspace(
        0,
        length - 1,
        num=min(length, count),
        dtype=np.int64,
    )


def collate_dataset(
    dataset: Track2WindowDataset,
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[1] for sample in samples]),
    )


def rollout_window_references(
    dataset: Track2WindowDataset,
    *,
    blocks: int,
    stride: int,
) -> list[tuple[int, int]]:
    required_frames = TRACK2_INPUT_STEPS * (blocks + 1)
    result = []
    for trajectory_index, values in enumerate(dataset.trajectories):
        maximum = values.shape[0] - required_frames
        result.extend(
            (trajectory_index, start) for start in range(0, maximum + 1, stride)
        )
    if not result:
        raise ValueError("training data contains no attached rollout windows")
    return result


def collate_rollout_dataset(
    dataset: Track2WindowDataset,
    references: Sequence[tuple[int, int]],
    indices: Sequence[int],
    *,
    blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = TRACK2_INPUT_STEPS * (blocks + 1)
    values = torch.from_numpy(
        np.stack(
            [
                dataset.trajectories[trajectory_index][start : start + frame_count]
                for trajectory_index, start in (
                    references[int(index)] for index in indices
                )
            ]
        )
    )
    normalization = dataset.normalization
    input_mean = torch.as_tensor(normalization.input_mean, dtype=values.dtype)
    target_mean = torch.as_tensor(normalization.target_mean, dtype=values.dtype)
    input_std = torch.as_tensor(normalization.input_std, dtype=values.dtype)
    target_std = torch.as_tensor(normalization.target_std, dtype=values.dtype)
    return (
        (values[:, :TRACK2_INPUT_STEPS] - input_mean) / input_std,
        (values[:, TRACK2_INPUT_STEPS:] - target_mean) / target_std,
    )


def batch_slices(indices: np.ndarray, batch_size: int):
    for first in range(0, len(indices), batch_size):
        yield indices[first : first + batch_size]


@torch.inference_mode()
def evaluate(
    model: RealPDETrack2PCNO,
    dataset: Track2WindowDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    model.eval()
    totals = {"normalized_mse": 0.0, "rel_l2": 0.0, "tke": 0.0, "mvpe": 0.0}
    examples = 0
    elapsed = 0.0
    for index_batch in batch_slices(indices, batch_size):
        input_norm, target_norm = collate_dataset(dataset, index_batch)
        input_norm = input_norm.to(device, non_blocking=True)
        target_norm = target_norm.to(device, non_blocking=True)
        synchronize(device)
        started = perf_counter()
        with autocast_context(device, amp):
            prediction = model(input_norm)
        synchronize(device)
        elapsed += perf_counter() - started
        squared = torch.square(prediction.float() - target_norm.float())
        errors = track2_error_tensors(
            prediction.float(),
            target_norm.float(),
            model,
        )
        current = int(input_norm.shape[0])
        totals["normalized_mse"] += float(squared.mean()) * current
        for name in ("rel_l2", "tke", "mvpe"):
            totals[name] += float(errors[name].sum())
        examples += current
    result = {name: value / examples for name, value in totals.items()}
    result["accuracy_score"] = track2_forecast_score(result)
    result["model_call_seconds"] = elapsed
    result["seconds_per_window"] = elapsed / examples
    result["windows"] = examples
    return result


@torch.inference_mode()
def evaluate_persistence(
    model: RealPDETrack2PCNO,
    dataset: Track2WindowDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    totals = {"normalized_mse": 0.0, "rel_l2": 0.0, "tke": 0.0, "mvpe": 0.0}
    examples = 0
    for index_batch in batch_slices(indices, batch_size):
        input_norm, target_norm = collate_dataset(dataset, index_batch)
        input_norm = input_norm.to(device)
        target_norm = target_norm.to(device)
        last_raw = model.input_to_raw(input_norm[:, -1, ..., :2])
        persistence = model.raw_to_target(last_raw)[:, None].expand_as(target_norm)
        errors = track2_error_tensors(persistence, target_norm, model)
        current = int(input_norm.shape[0])
        totals["normalized_mse"] += (
            float(torch.square(persistence - target_norm).mean()) * current
        )
        for name in ("rel_l2", "tke", "mvpe"):
            totals[name] += float(errors[name].sum())
        examples += current
    result = {name: value / examples for name, value in totals.items()}
    result["accuracy_score"] = track2_forecast_score(result)
    result["windows"] = examples
    return result


@torch.inference_mode()
def evaluate_free_rollout(
    model: RealPDETrack2PCNO,
    dataset: Track2WindowDataset,
    *,
    trajectory_count: int,
    rollout_blocks: int,
    device: torch.device,
    amp: str,
    persistence: bool = False,
) -> dict[str, float]:
    chosen = np.linspace(
        0,
        len(dataset.trajectories) - 1,
        num=min(trajectory_count, len(dataset.trajectories)),
        dtype=np.int64,
    )
    available_blocks = min(
        (dataset.trajectories[int(index)].shape[0] - TRACK2_INPUT_STEPS)
        // TRACK2_INPUT_STEPS
        for index in chosen
    )
    blocks = min(rollout_blocks, available_blocks)
    if blocks < 1:
        raise ValueError("validation trajectories contain no free-rollout block")
    initial_raw = np.stack(
        [dataset.trajectories[int(index)][:TRACK2_INPUT_STEPS] for index in chosen]
    )
    target_raw = np.stack(
        [
            dataset.trajectories[int(index)][
                TRACK2_INPUT_STEPS : TRACK2_INPUT_STEPS * (blocks + 1)
            ]
            for index in chosen
        ]
    )
    current_raw = torch.from_numpy(initial_raw).to(device)
    target_raw_tensor = torch.from_numpy(target_raw).to(device)
    prediction_blocks = []
    elapsed = 0.0
    if persistence:
        prediction_raw = current_raw[:, -1:, ...].expand(
            -1,
            blocks * TRACK2_INPUT_STEPS,
            -1,
            -1,
            -1,
        )
    else:
        for _ in range(blocks):
            current_norm = model.raw_to_input(current_raw)
            synchronize(device)
            started = perf_counter()
            with autocast_context(device, amp):
                prediction_norm = model(current_norm)
            synchronize(device)
            elapsed += perf_counter() - started
            prediction_raw_block = model.target_to_raw(prediction_norm.float())
            prediction_blocks.append(prediction_raw_block)
            current_raw = prediction_raw_block
        prediction_raw = torch.cat(prediction_blocks, dim=1)
    prediction_norm = model.raw_to_target(prediction_raw)
    target_norm = model.raw_to_target(target_raw_tensor)
    errors = track2_error_tensors(prediction_norm, target_norm, model)
    result = {
        "normalized_mse": float(torch.square(prediction_norm - target_norm).mean()),
        **{name: float(value.mean()) for name, value in errors.items()},
        "trajectories": len(chosen),
        "blocks": int(blocks),
        "frames": int(blocks * TRACK2_INPUT_STEPS),
        "model_call_seconds": elapsed,
    }
    result["accuracy_score"] = track2_forecast_score(result)
    return result


def cpu_model_state(
    model: RealPDETrack2PCNO,
    *,
    parameter_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    parameter_names = {name for name, _ in model.named_parameters()}
    state = {}
    for name, value in model.state_dict().items():
        copied = value.detach().cpu()
        if (
            parameter_dtype is not None
            and name in parameter_names
            and copied.is_floating_point()
        ):
            copied = copied.to(parameter_dtype)
        state[name] = copied
    return state


def model_payload(
    model: RealPDETrack2PCNO,
    geometry: Track2Geometry,
    *,
    stage: str,
    update: int,
    run_signature: str,
    direct_validation: Mapping[str, float],
    rollout_validation: Mapping[str, float],
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": MODEL_PAYLOAD_SCHEMA,
        "checkpoint_role": "frozen_forecast_model",
        "stage": stage,
        "update": int(update),
        "model_config": model.model_config(),
        "geometry": geometry.to_mapping(),
        "normalization": model.normalization().to_mapping(),
        "residual_scale": model.residual_scale[0, :, 0, 0].detach().cpu().numpy(),
        "model_state": cpu_model_state(model, parameter_dtype=torch.float16),
        "parameter_storage": "float16",
        "target_contract": {
            "learned_fields": ["velocity_u", "velocity_v"],
            "pressure_interface_channel": "literal_zero",
            "conservative_variables_available": False,
            "reason": "released real trajectories expose velocity but not density",
        },
        "boundary_contract": {
            "hard_enforcement": (
                "input_persistent_zero_within_simulation_derived_airfoil_support"
            ),
            "support": "training_simulation_mask_union_plus_physical_collar",
            "outer_rectangle": "encoded_semantically_not_hard_projected",
            "piv_zero_region": "causal_missingness_encoding_not_physical_wall",
        },
        "run_signature": run_signature,
        "direct_validation": dict(direct_validation),
        "free_rollout_validation": dict(rollout_validation),
        "parent": dict(parent) if parent is not None else None,
    }


def save_model_payload(payload: Mapping[str, Any], path: Path) -> None:
    atomic_torch_save(payload, path)
    maximum_bytes = 256 * 1024 * 1024
    if path.stat().st_size >= maximum_bytes:
        raise RuntimeError(
            f"model artifact is {path.stat().st_size} bytes and exceeds 256 MiB"
        )


def recursive_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return value


def training_payload(
    model: RealPDETrack2PCNO,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    update: int,
    best_score: float,
    best_update: int,
    run_signature: str,
    batch_generator: torch.Generator,
    last_direct_validation: Mapping[str, float],
    last_rollout_validation: Mapping[str, float],
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": TRAINING_PAYLOAD_SCHEMA,
        "checkpoint_role": "resumable_training_state",
        "update": int(update),
        "model_state": cpu_model_state(model, parameter_dtype=None),
        "optimizer_state": recursive_to_cpu(optimizer.state_dict()),
        "scaler_state": scaler.state_dict(),
        "best_score": float(best_score),
        "best_update": int(best_update),
        "run_signature": run_signature,
        "batch_generator_state": batch_generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "last_direct_validation": dict(last_direct_validation),
        "last_rollout_validation": dict(last_rollout_validation),
        "parent": dict(parent) if parent is not None else None,
    }


def restore_rng_state(
    payload: Mapping[str, Any],
    batch_generator: torch.Generator,
) -> None:
    batch_generator.set_state(payload["batch_generator_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def assert_matching_geometry(
    source: Mapping[str, Any],
    target: Track2Geometry,
) -> None:
    previous = Track2Geometry.from_mapping(source)
    comparisons = (
        np.array_equal(previous.solid_mask, target.solid_mask),
        np.allclose(previous.nodes, target.nodes, rtol=0.0, atol=1.0e-6),
        np.allclose(
            previous.static_features,
            target.static_features,
            rtol=0.0,
            atol=1.0e-6,
        ),
    )
    if not all(comparisons):
        raise ValueError("initialization checkpoint and current geometry differ")


def initialize_from_model_payload(
    model: RealPDETrack2PCNO,
    geometry: Track2Geometry,
    path: Path,
) -> dict[str, Any]:
    payload = load_mapping(path)
    if payload.get("schema") != MODEL_PAYLOAD_SCHEMA:
        raise ValueError("initialization checkpoint is not a Track 2 model payload")
    config = payload["model_config"]
    expected = model.model_config()
    for key in ("n_modes", "layers", "fc_dim"):
        if config[key] != expected[key]:
            raise ValueError(f"initialization model config differs for {key}")
    assert_matching_geometry(payload["geometry"], geometry)
    filtered = {
        name: value
        for name, value in payload["model_state"].items()
        if name not in TRANSFER_SCALING_KEYS
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    if set(incompatible.missing_keys) != TRANSFER_SCALING_KEYS:
        raise ValueError(
            f"unexpected missing transfer keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected transfer keys: {incompatible.unexpected_keys}")
    return {
        "checkpoint_name": path.name,
        "sha256": sha256_file(path),
        "source_stage": payload.get("stage"),
        "source_update": payload.get("update"),
        "transfer_policy": "all_model_state_except_domain_normalization_and_residual_scale",
    }


def write_metric(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def prepare_output_directory(
    output_dir: Path,
    *,
    resume: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"{output_dir} is nonempty; use a new directory or --resume-checkpoint"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and not (output_dir / "best_model.pt").is_file():
        raise FileNotFoundError(
            "resume requires the original output directory with best_model.pt"
        )


def stable_run_contract(
    args: argparse.Namespace,
    *,
    source: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    train_metadata: Sequence[Track2FileMetadata],
    validation_metadata: Sequence[Track2FileMetadata],
    geometry_source: Path,
    geometry: Track2Geometry,
    normalization: Track2Normalization,
    residual_scale: np.ndarray,
    model: RealPDETrack2PCNO,
) -> dict[str, Any]:
    return {
        "stage": args.stage,
        "source_manifest": dict(source),
        "data_manifest_digest": data_manifest["digest"],
        "train_files": [item.path.name for item in train_metadata],
        "validation_files": [item.path.name for item in validation_metadata],
        "geometry_source": {
            "name": geometry_source.name,
            "sha256": sha256_file(geometry_source),
        },
        "geometry_contract": geometry.contract,
        "normalization": normalization.to_mapping(),
        "residual_scale": residual_scale.tolist(),
        "model_config": model.model_config(),
        "optimization": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "window_stride": args.window_stride,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_updates": args.max_updates,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
            "warmup_fraction": args.warmup_fraction,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "relative_l2_weight": args.relative_l2_weight,
            "tke_weight": args.tke_weight,
            "mvpe_weight": args.mvpe_weight,
            "boundary_weight": args.boundary_weight,
            "free_rollout_loss_weight": args.free_rollout_loss_weight,
            "free_rollout_loss_every": args.free_rollout_loss_every,
            "free_rollout_training_blocks": 2,
            "free_rollout_recurrence_gradient": "attached",
        },
        "validation": {
            "fraction": args.validation_fraction,
            "direct_windows": args.evaluation_windows,
            "rollout_trajectories": args.rollout_trajectories,
            "rollout_blocks": args.rollout_blocks,
            "selection": "maximum_frozen_free_rollout_accuracy_score",
        },
        "target_contract": {
            "prediction": "direct_20_to_20_velocity",
            "recurrence": "target_normalized_output_to_raw_to_input_normalized",
            "pressure": "zero_interface_channel",
            "state_class": "velocity_not_conservative_due_to_missing_density",
            "generated_state_exposure": "attached_differentiable_second_call",
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_output_directory(
        args.output_dir,
        resume=args.resume_checkpoint is not None,
    )
    device = select_device(args.device)
    amp = resolve_amp(args.amp, device)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metadata = select_files(
        discover_track2_files(args.data_dir),
        args.maximum_files,
    )
    train_metadata, validation_metadata = split_track2_files(
        metadata,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    if args.stage == "sim_pretrain":
        geometry_source = train_metadata[0].path
        geometry = load_track2_geometry_from_simulations(
            [item.path for item in train_metadata],
            collar_width=args.collar_width,
        )
    else:
        geometry_source = args.geometry_simulation
        if not geometry_source.is_file():
            raise FileNotFoundError(geometry_source)
        geometry_payload_path = (
            args.init_checkpoint
            if args.init_checkpoint is not None
            else args.output_dir / "best_model.pt"
        )
        geometry_payload = load_mapping(geometry_payload_path)
        geometry = Track2Geometry.from_mapping(geometry_payload["geometry"])
        coordinate_probe = load_track2_geometry_from_simulation(
            geometry_source,
            collar_width=args.collar_width,
        )
        if not np.allclose(
            geometry.nodes,
            coordinate_probe.nodes,
            rtol=0.0,
            atol=3.0e-5,
        ):
            raise ValueError("geometry simulation coordinates differ from checkpoint")
        if not np.isclose(
            geometry.contract["boundary_collar_width"],
            args.collar_width,
        ):
            raise ValueError("boundary collar width differs from checkpoint")
    normalization = stage_normalization(args, train_metadata)
    train_dataset = Track2WindowDataset(
        train_metadata,
        normalization,
        stride=args.window_stride,
        expected_geometry=geometry,
        validate_solid=args.stage == "sim_pretrain",
    )
    validation_dataset = Track2WindowDataset(
        validation_metadata,
        normalization,
        stride=args.window_stride,
        expected_geometry=geometry,
        validate_solid=args.stage == "sim_pretrain",
    )
    residual_scale = estimate_track2_residual_scale(
        train_dataset,
        maximum_windows=args.residual_scale_windows,
    )
    model = RealPDETrack2PCNO(
        normalization=normalization,
        geometry=geometry,
        residual_scale=residual_scale,
        n_modes=args.modes,
        layers=args.layers,
        fc_dim=args.fc_dim,
        zero_initialize=(
            args.init_checkpoint is None and args.resume_checkpoint is None
        ),
    ).to(device)

    parent = None
    if args.init_checkpoint is not None:
        parent = initialize_from_model_payload(
            model,
            geometry,
            args.init_checkpoint,
        )

    source = source_manifest()
    data = metadata_manifest(metadata)
    contract = stable_run_contract(
        args,
        source=source,
        data_manifest=data,
        train_metadata=train_metadata,
        validation_metadata=validation_metadata,
        geometry_source=geometry_source,
        geometry=geometry,
        normalization=normalization,
        residual_scale=residual_scale,
        model=model,
    )
    run_signature = digest_mapping(contract)
    if args.resume_checkpoint is not None:
        saved_best = load_mapping(args.output_dir / "best_model.pt")
        if saved_best.get("run_signature") != run_signature:
            raise ValueError("saved best model and current run contracts differ")
    run_record = {
        **contract,
        "schema": "realpde_track2_pcno_run_v1",
        "run_signature": run_signature,
        "git": git_state(),
        "runtime": runtime_environment(device),
        "resolved_amp": amp,
        "parameter_count": parameter_count(model),
        "parameter_bytes_float32": parameter_count(model) * 4,
        "data_manifest": data,
    }
    write_json(args.output_dir / "run_contract.json", run_record)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp == "fp16",
    )
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(args.seed + 1)
    start_update = 0
    best_score = -math.inf
    best_update = -1
    last_direct: dict[str, float] = {}
    last_rollout: dict[str, float] = {}

    if args.resume_checkpoint is not None:
        resume_payload = load_mapping(args.resume_checkpoint)
        if resume_payload.get("schema") != TRAINING_PAYLOAD_SCHEMA:
            raise ValueError("resume checkpoint has the wrong schema")
        if resume_payload.get("run_signature") != run_signature:
            raise ValueError("resume checkpoint and current run contracts differ")
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload["scaler_state"])
        restore_rng_state(resume_payload, batch_generator)
        start_update = int(resume_payload["update"])
        best_score = float(resume_payload["best_score"])
        best_update = int(resume_payload["best_update"])
        last_direct = dict(resume_payload["last_direct_validation"])
        last_rollout = dict(resume_payload["last_rollout_validation"])
        saved_parent = resume_payload.get("parent")
        parent = dict(saved_parent) if saved_parent is not None else None
        if start_update >= args.max_updates:
            raise ValueError("resume checkpoint already reached --max-updates")

    direct_indices = validation_indices(
        len(validation_dataset),
        args.evaluation_windows,
    )
    rollout_train_windows = (
        rollout_window_references(
            train_dataset,
            blocks=2,
            stride=args.window_stride,
        )
        if args.free_rollout_loss_weight > 0.0
        else []
    )
    persistence_direct = evaluate_persistence(
        model,
        validation_dataset,
        direct_indices,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    persistence_rollout = evaluate_free_rollout(
        model,
        validation_dataset,
        trajectory_count=args.rollout_trajectories,
        rollout_blocks=args.rollout_blocks,
        device=device,
        amp=amp,
        persistence=True,
    )
    write_json(
        args.output_dir / "baselines.json",
        {
            "persistence_direct": persistence_direct,
            "persistence_free_rollout": persistence_rollout,
        },
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    if start_update == 0:
        last_direct = evaluate(
            model,
            validation_dataset,
            direct_indices,
            batch_size=args.evaluation_batch_size,
            device=device,
            amp=amp,
        )
        last_rollout = evaluate_free_rollout(
            model,
            validation_dataset,
            trajectory_count=args.rollout_trajectories,
            rollout_blocks=args.rollout_blocks,
            device=device,
            amp=amp,
        )
        best_score = float(last_rollout["accuracy_score"])
        best_update = 0
        initial_payload = model_payload(
            model,
            geometry,
            stage=args.stage,
            update=0,
            run_signature=run_signature,
            direct_validation=last_direct,
            rollout_validation=last_rollout,
            parent=parent,
        )
        save_model_payload(initial_payload, args.output_dir / "best_model.pt")
        write_metric(
            metrics_path,
            {
                "update": 0,
                "direct_validation": last_direct,
                "free_rollout_validation": last_rollout,
                "selected": True,
            },
        )

    rolling = {
        "loss": 0.0,
        "normalized_mse": 0.0,
        "rel_l2": 0.0,
        "tke": 0.0,
        "mvpe": 0.0,
        "attached_second_call_loss": 0.0,
        "updates": 0,
    }
    train_started = perf_counter()
    model.train()
    for update in range(start_update + 1, args.max_updates + 1):
        learning_rate = schedule_learning_rate(
            update,
            max_updates=args.max_updates,
            base_learning_rate=args.learning_rate,
            warmup_fraction=args.warmup_fraction,
            minimum_ratio=args.minimum_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        update_totals = {
            "loss": 0.0,
            "normalized_mse": 0.0,
            "rel_l2": 0.0,
            "tke": 0.0,
            "mvpe": 0.0,
            "attached_second_call_loss": 0.0,
        }
        attached_rollout = (
            args.free_rollout_loss_weight > 0.0
            and update % args.free_rollout_loss_every == 0
        )
        for _ in range(args.gradient_accumulation_steps):
            if attached_rollout:
                indices = torch.randint(
                    len(rollout_train_windows),
                    (args.batch_size,),
                    generator=batch_generator,
                ).tolist()
                input_norm, target_norm = collate_rollout_dataset(
                    train_dataset,
                    rollout_train_windows,
                    indices,
                    blocks=2,
                )
            else:
                indices = torch.randint(
                    len(train_dataset),
                    (args.batch_size,),
                    generator=batch_generator,
                ).tolist()
                input_norm, target_norm = collate_dataset(
                    train_dataset,
                    indices,
                )
            input_norm = input_norm.to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            with autocast_context(device, amp):
                prediction = model(input_norm)
                direct_target = target_norm[:, :TRACK2_INPUT_STEPS]
                direct_loss, terms = track2_training_loss(
                    model,
                    prediction,
                    direct_target,
                    relative_l2_weight=args.relative_l2_weight,
                    tke_weight=args.tke_weight,
                    mvpe_weight=args.mvpe_weight,
                    boundary_weight=args.boundary_weight,
                )
                if attached_rollout:
                    generated_input = model.raw_to_input(
                        model.target_to_raw(prediction.float())
                    )
                    second_prediction = model(generated_input)
                    second_target = target_norm[
                        :,
                        TRACK2_INPUT_STEPS : 2 * TRACK2_INPUT_STEPS,
                    ]
                    second_loss, _ = track2_training_loss(
                        model,
                        second_prediction,
                        second_target,
                        relative_l2_weight=args.relative_l2_weight,
                        tke_weight=args.tke_weight,
                        mvpe_weight=args.mvpe_weight,
                        boundary_weight=args.boundary_weight,
                    )
                    loss = direct_loss + (args.free_rollout_loss_weight * second_loss)
                else:
                    second_loss = torch.zeros_like(direct_loss)
                    loss = direct_loss
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            update_totals["loss"] = update_totals["loss"] + loss.detach()
            for name in ("normalized_mse", "rel_l2", "tke", "mvpe"):
                update_totals[name] = update_totals[name] + terms[name].mean().detach()
            update_totals["attached_second_call_loss"] = (
                update_totals["attached_second_call_loss"] + second_loss.detach()
            )

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        for name, value in update_totals.items():
            rolling[name] += value / args.gradient_accumulation_steps
        rolling["updates"] += 1

        evaluated = update % args.evaluation_every == 0 or update == args.max_updates
        selected = False
        if evaluated:
            last_direct = evaluate(
                model,
                validation_dataset,
                direct_indices,
                batch_size=args.evaluation_batch_size,
                device=device,
                amp=amp,
            )
            last_rollout = evaluate_free_rollout(
                model,
                validation_dataset,
                trajectory_count=args.rollout_trajectories,
                rollout_blocks=args.rollout_blocks,
                device=device,
                amp=amp,
            )
            selection_score = float(last_rollout["accuracy_score"])
            if selection_score > best_score:
                best_score = selection_score
                best_update = update
                selected = True
                save_model_payload(
                    model_payload(
                        model,
                        geometry,
                        stage=args.stage,
                        update=update,
                        run_signature=run_signature,
                        direct_validation=last_direct,
                        rollout_validation=last_rollout,
                        parent=parent,
                    ),
                    args.output_dir / "best_model.pt",
                )
            divisor = max(int(rolling["updates"]), 1)
            write_metric(
                metrics_path,
                {
                    "update": update,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "train": {
                        name: float(value / divisor)
                        for name, value in rolling.items()
                        if name != "updates"
                    },
                    "direct_validation": last_direct,
                    "free_rollout_validation": last_rollout,
                    "selected": selected,
                },
            )
            rolling = {
                "loss": 0.0,
                "normalized_mse": 0.0,
                "rel_l2": 0.0,
                "tke": 0.0,
                "mvpe": 0.0,
                "attached_second_call_loss": 0.0,
                "updates": 0,
            }
            model.train()

        if update % args.checkpoint_every == 0 or update == args.max_updates:
            atomic_torch_save(
                training_payload(
                    model,
                    optimizer,
                    scaler,
                    update=update,
                    best_score=best_score,
                    best_update=best_update,
                    run_signature=run_signature,
                    batch_generator=batch_generator,
                    last_direct_validation=last_direct,
                    last_rollout_validation=last_rollout,
                    parent=parent,
                ),
                args.output_dir / "last_training.pt",
            )

    final_payload = model_payload(
        model,
        geometry,
        stage=args.stage,
        update=args.max_updates,
        run_signature=run_signature,
        direct_validation=last_direct,
        rollout_validation=last_rollout,
        parent=parent,
    )
    save_model_payload(final_payload, args.output_dir / "final_model.pt")
    summary = {
        "schema": "realpde_track2_pcno_summary_v1",
        "stage": args.stage,
        "updates": args.max_updates,
        "best_update": best_update,
        "best_free_rollout_accuracy_score": best_score,
        "last_direct_validation": last_direct,
        "last_free_rollout_validation": last_rollout,
        "persistence_direct": persistence_direct,
        "persistence_free_rollout": persistence_rollout,
        "parameter_count": parameter_count(model),
        "training_seconds": perf_counter() - train_started,
        "artifacts": {
            "best_model": "best_model.pt",
            "best_model_sha256": sha256_file(args.output_dir / "best_model.pt"),
            "best_model_bytes": (args.output_dir / "best_model.pt").stat().st_size,
            "final_model": "final_model.pt",
            "final_model_sha256": sha256_file(args.output_dir / "final_model.pt"),
            "last_training": "last_training.pt",
            "run_contract": "run_contract.json",
            "metrics": "metrics.jsonl",
            "baselines": "baselines.json",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
