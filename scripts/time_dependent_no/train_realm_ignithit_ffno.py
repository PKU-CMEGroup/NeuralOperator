"""Train the frozen D088 IgnitHIT direct FFNO reconstruction.

This entry point is intentionally narrow. It accepts only the exact open
IgnitHIT manifest, P1b normalizer, output directory, and an optional exact
``last.pt`` resume checkpoint. Scientific hyperparameters are constants frozen
in the D088 preregistration; this is not a general REALM training CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_OPEN_MANIFEST_SHA256,
    MagnitudeEnvelope,
    RealmNormalizer,
    canonical_json_sha256,
    decoded_admissibility,
    decoded_boundedness,
    decoded_spatial_pearson,
    grouped_normalized_prediction_error,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    grouped_next_state_mse,
    normalize_realm_coordinates,
    parameter_count_within_reported_tolerance,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    BOUNDEDNESS_EXPANSION_FACTOR,
    BOUNDEDNESS_QUANTILE,
    BOX_COX_LAMBDA,
    PRIMARY_BOX_COX_EPSILON,
    SCALE_STABILIZER,
    STD_CORRECTION,
    TRAIN_GROUPS,
    TRAJECTORY_DTYPE,
    TRAJECTORY_SHAPE,
    VAL_GROUPS,
    load_ignithit_metadata,
    load_ignithit_trajectory,
    sha256_file,
    trajectory_relative_path,
    validate_local_open_tree,
)

RUN_ID = "d088_realm_ignithit_p1c_direct_seed0_5000_20260812a"
SEED = 0
DEVICE = "cuda:0"
DTYPE = torch.float32
TOTAL_STEPS = 5_000
SCHEDULER_TOTAL_STEPS = 5_001
VALIDATION_INTERVAL = 50
MICROBATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 26
ACCUMULATION_STEPS = 26
MAX_LR = 1.0e-3
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
ONECYCLE_PCT_START = 0.3
ONECYCLE_BASE_MOMENTUM = 0.85
ONECYCLE_MAX_MOMENTUM = 0.95
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 10_000.0
VALIDATION_HORIZON = 29
NORMALIZER_ARRAYS_SHA256 = (
    "368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20"
)
NORMALIZER_ARRAY_KEYS = frozenset(
    {
        "primary_mean",
        "primary_std",
        "primary_scale",
        "source_sensitivity_mean",
        "source_sensitivity_std",
        "source_sensitivity_scale",
        "raw_train_mean",
        "raw_train_std",
        "train_max_abs",
    }
)
BEST_CHECKPOINT_SCHEMA = "d088_ignithit_ffno_best_v1"
LAST_CHECKPOINT_SCHEMA = "d088_ignithit_ffno_last_v1"


@dataclass(frozen=True)
class StepSample:
    case_indices: tuple[int, ...]
    frame_start: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume only from OUTPUT_DIR/last.pt under the exact frozen contract",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="operational planned stop at a registered validation step",
    )
    return parser


def legal_adjacent_frame_pairs(
    frame_count: int = TRAJECTORY_SHAPE[0],
) -> tuple[tuple[int, int], ...]:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if frame_count < 2:
        raise ValueError("at least two frames are required")
    return tuple((start, start + 1) for start in range(frame_count - 1))


def draw_step_sample(
    generator: random.Random,
    *,
    case_count: int = EFFECTIVE_BATCH_SIZE,
    frame_count: int = TRAJECTORY_SHAPE[0],
) -> StepSample:
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count <= 0
    ):
        raise ValueError("case_count must be a positive integer")
    pairs = legal_adjacent_frame_pairs(frame_count)
    case_indices = list(range(case_count))
    generator.shuffle(case_indices)
    frame_start = generator.randrange(len(pairs))
    return StepSample(tuple(case_indices), frame_start)


def validation_steps(
    *,
    total_steps: int = TOTAL_STEPS,
    interval: int = VALIDATION_INTERVAL,
) -> tuple[int, ...]:
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps <= 0
    ):
        raise ValueError("total_steps must be a positive integer")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("interval must be a positive integer")
    result = {1, total_steps}
    result.update(range(interval, total_steps + 1, interval))
    return tuple(sorted(result))


def scaled_microbatch_loss(
    loss: torch.Tensor, *, effective_batch_size: int
) -> torch.Tensor:
    if loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError("microbatch loss must be a floating scalar")
    if (
        isinstance(effective_batch_size, bool)
        or not isinstance(effective_batch_size, int)
        or effective_batch_size <= 0
    ):
        raise ValueError("effective_batch_size must be a positive integer")
    return loss / effective_batch_size


def is_strict_improvement(score: float, best_score: float) -> bool:
    if not math.isfinite(score):
        raise ValueError("selection score must be finite")
    if math.isnan(best_score):
        raise ValueError("best selection score cannot be NaN")
    return score < best_score


def direct_rollout(
    model: nn.Module,
    initial_state: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    calls: int,
) -> torch.Tensor:
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        raise ValueError("calls must be a positive integer")
    current = initial_state
    predictions: list[torch.Tensor] = []
    for _ in range(calls):
        proposal = model(current, coordinates)
        if proposal.shape != current.shape:
            raise ValueError("direct proposal shape differs from the recurrent state")
        if not bool(torch.isfinite(proposal).all()):
            raise RuntimeError("validation returned a nonfinite normalized proposal")
        predictions.append(proposal)
        current = proposal
    return torch.stack(predictions, dim=1)


def summarize_validation_predictions(
    prediction_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    prediction_decoded: torch.Tensor,
    truth_decoded: torch.Tensor,
    *,
    case_keys: Sequence[str],
    train_max_abs: torch.Tensor,
) -> dict[str, Any]:
    if len(case_keys) != prediction_normalized.shape[0] or len(set(case_keys)) != len(
        case_keys
    ):
        raise ValueError("validation case keys must be unique and match the case axis")
    if prediction_normalized.shape != truth_normalized.shape:
        raise ValueError("normalized prediction and truth shapes differ")
    if prediction_decoded.shape != truth_decoded.shape:
        raise ValueError("decoded prediction and truth shapes differ")
    if prediction_decoded.shape != prediction_normalized.shape:
        raise ValueError("normalized and decoded validation shapes differ")
    if prediction_normalized.ndim != 5:
        raise ValueError("validation tensors require [case, call, channel, y, x]")
    if not bool(torch.isfinite(prediction_decoded).all()):
        raise RuntimeError("validation returned a nonfinite decoded proposal")

    errors = grouped_normalized_prediction_error(
        prediction_normalized,
        truth_normalized,
    )
    correlation = decoded_spatial_pearson(prediction_decoded, truth_decoded)
    envelope = MagnitudeEnvelope(
        max_abs=train_max_abs.to(
            device=prediction_decoded.device,
            dtype=prediction_decoded.dtype,
        ),
        quantile=BOUNDEDNESS_QUANTILE,
        channel_axis=0,
    )
    status_counts: Counter[str] = Counter()
    for case in correlation.statuses:
        for call in case:
            status_counts.update(call)

    per_case: list[dict[str, Any]] = []
    all_admissible = True
    all_bounded = True
    maximum_ratio: float | None = 0.0
    for case_index, case_key in enumerate(case_keys):
        admissible_calls = 0
        bounded_calls = 0
        for call_index in range(prediction_decoded.shape[1]):
            state = prediction_decoded[case_index, call_index]
            admissibility = decoded_admissibility(state, channel_axis=0)
            boundedness = decoded_boundedness(
                state,
                envelope,
                expansion_factor=BOUNDEDNESS_EXPANSION_FACTOR,
                channel_axis=0,
            )
            admissible_calls += int(admissibility.admissible)
            bounded_calls += int(boundedness.bounded)
            all_admissible = all_admissible and admissibility.admissible
            all_bounded = all_bounded and boundedness.bounded
            if all(math.isfinite(value) for value in boundedness.max_ratio_by_channel):
                if maximum_ratio is not None:
                    maximum_ratio = max(
                        maximum_ratio,
                        *boundedness.max_ratio_by_channel,
                    )
            else:
                maximum_ratio = None
        correlation_value = float(correlation.per_case_mean[case_index].item())
        per_case.append(
            {
                "case_key": case_key,
                "realm_npe_mean": float(errors.per_case_mean[case_index].item()),
                "realm_npe_sum_source": float(errors.per_case_sum[case_index].item()),
                "decoded_correlation": correlation_value
                if math.isfinite(correlation_value)
                else None,
                "admissible_call_count": admissible_calls,
                "bounded_call_count": bounded_calls,
            }
        )

    grouped_curves = {
        name: values.mean(dim=0).detach().cpu().tolist()
        for name, values in sorted(errors.grouped_per_call.items())
    }
    population_correlation = correlation.population_case_first_mean
    return {
        "selection_metric": "realm_npe_mean",
        "realm_npe_mean": errors.realm_npe_mean,
        "realm_npe_sum_source": errors.realm_npe_sum_source,
        "decoded_correlation_case_first": population_correlation
        if math.isfinite(population_correlation)
        else None,
        "npe_total_case_first_by_call": errors.total_per_call.mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "npe_group_case_first_by_call": grouped_curves,
        "correlation_status_counts": dict(sorted(status_counts.items())),
        "all_normalized_finite": bool(torch.isfinite(prediction_normalized).all()),
        "all_decoded_finite": True,
        "all_released_state_admissible": all_admissible,
        "all_bounded_10x_train_max": all_bounded,
        "max_boundedness_ratio": maximum_ratio,
        "case_count": len(case_keys),
        "call_count": prediction_normalized.shape[1],
        "per_case": per_case,
    }


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    scheduler_total_steps: int = SCHEDULER_TOTAL_STEPS,
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.OneCycleLR]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MAX_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=scheduler_total_steps,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=ONECYCLE_BASE_MOMENTUM,
        max_momentum=ONECYCLE_MAX_MOMENTUM,
        div_factor=ONECYCLE_DIV_FACTOR,
        final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
        three_phase=False,
    )
    return optimizer, scheduler


def _recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_recursive_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_recursive_to_cpu(item) for item in value]
    return value


def structured_state_sha256(value: Any) -> str:
    """Hash a nested checkpoint payload without relying on pickle bytes."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            descriptor = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
            digest.update(b"tensor")
            digest.update(len(descriptor).to_bytes(8, byteorder="big"))
            digest.update(descriptor)
            digest.update(len(raw).to_bytes(8, byteorder="big"))
            digest.update(raw)
            return
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("structured state mapping keys must be strings")
            digest.update(b"mapping")
            digest.update(len(item).to_bytes(8, byteorder="big"))
            for key in sorted(item):
                encoded_key = key.encode("utf-8")
                digest.update(len(encoded_key).to_bytes(8, byteorder="big"))
                digest.update(encoded_key)
                update(item[key])
            return
        if isinstance(item, (tuple, list)):
            digest.update(b"tuple" if isinstance(item, tuple) else b"list")
            digest.update(len(item).to_bytes(8, byteorder="big"))
            for child in item:
                update(child)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            encoded = json.dumps(
                {"type": type(item).__name__, "value": item},
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(b"scalar")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)
            return
        raise TypeError(f"unsupported structured state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _rng_payload(
    order_generator: random.Random,
    *,
    include_cuda: bool,
) -> dict[str, Any]:
    return {
        "order_rng_state": order_generator.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def build_best_checkpoint(
    model: nn.Module,
    *,
    run_signature: str,
    completed_step: int,
    validation: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    model_state = _recursive_to_cpu(model.state_dict())
    normalizer_state_cpu = _recursive_to_cpu(normalizer_state)
    return {
        "schema": BEST_CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "run_signature": run_signature,
        "completed_step": completed_step,
        "selection_metric": "realm_npe_mean",
        "selection_value": float(validation["realm_npe_mean"]),
        "validation": dict(validation),
        "model_config": asdict(RealmFFNOConfig()),
        "model_state": model_state,
        "model_state_sha256": structured_state_sha256(model_state),
        "normalizer_state": normalizer_state_cpu,
        "normalizer_state_sha256": structured_state_sha256(normalizer_state_cpu),
        "provenance": dict(provenance),
        "resume_supported": False,
    }


def build_last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: random.Random,
    *,
    run_signature: str,
    completed_step: int,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str | None,
    history: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    model_state = _recursive_to_cpu(model.state_dict())
    payload = {
        "schema": LAST_CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "run_signature": run_signature,
        "completed_step": completed_step,
        "best_step": best_step,
        "best_score": best_score,
        "best_model_state_sha256": best_model_state_sha256,
        "history": [dict(row) for row in history],
        "model_state": model_state,
        "model_state_sha256": structured_state_sha256(model_state),
        "optimizer_state": _recursive_to_cpu(optimizer.state_dict()),
        "scheduler_state": _recursive_to_cpu(scheduler.state_dict()),
        "provenance": dict(provenance),
        "resume_supported": True,
    }
    include_cuda = any(parameter.is_cuda for parameter in model.parameters())
    payload.update(
        _recursive_to_cpu(_rng_payload(order_generator, include_cuda=include_cuda))
    )
    return payload


def restore_last_checkpoint(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: random.Random,
    *,
    expected_run_signature: str,
) -> tuple[int, int, float, str | None, list[dict[str, Any]]]:
    if checkpoint.get("schema") != LAST_CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint has the wrong schema")
    if checkpoint.get("run_signature") != expected_run_signature:
        raise ValueError("resume checkpoint and current run contracts differ")
    if not bool(checkpoint.get("resume_supported")):
        raise ValueError("checkpoint does not support resume")
    checkpoint_provenance = checkpoint.get("provenance")
    if (
        not isinstance(checkpoint_provenance, Mapping)
        or canonical_json_sha256(checkpoint_provenance) != expected_run_signature
    ):
        raise ValueError("resume checkpoint provenance is inconsistent")
    checkpoint_model_state = checkpoint.get("model_state")
    if not isinstance(checkpoint_model_state, Mapping) or structured_state_sha256(
        checkpoint_model_state
    ) != checkpoint.get("model_state_sha256"):
        raise ValueError("resume checkpoint model-state digest differs")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    order_generator.setstate(checkpoint["order_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_state = checkpoint.get("cuda_rng_state")
    model_uses_cuda = any(parameter.is_cuda for parameter in model.parameters())
    if model_uses_cuda:
        if cuda_state is None:
            raise ValueError("CUDA resume checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_state)
    completed_step = int(checkpoint["completed_step"])
    best_step = int(checkpoint["best_step"])
    best_score = float(checkpoint["best_score"])
    best_model_state_sha256 = checkpoint.get("best_model_state_sha256")
    if (
        best_step <= 0
        or best_step > completed_step
        or not math.isfinite(best_score)
        or not isinstance(best_model_state_sha256, str)
        or len(best_model_state_sha256) != 64
    ):
        raise ValueError("resume checkpoint best-model identity is invalid")
    history = [dict(row) for row in checkpoint["history"]]
    return completed_step, best_step, best_score, best_model_state_sha256, history


def _best_checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    run_signature: str,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str | None,
    normalizer_state_sha256: str,
    provenance: Mapping[str, str],
) -> bool:
    model_state = checkpoint.get("model_state")
    normalizer_state = checkpoint.get("normalizer_state")
    if not isinstance(model_state, Mapping) or not isinstance(
        normalizer_state, Mapping
    ):
        return False
    try:
        actual_model_digest = structured_state_sha256(model_state)
        actual_normalizer_digest = structured_state_sha256(normalizer_state)
        checkpoint_step = int(checkpoint.get("completed_step", -1))
        checkpoint_score = float(checkpoint.get("selection_value", math.inf))
    except (TypeError, ValueError):
        return False
    return (
        checkpoint.get("schema") == BEST_CHECKPOINT_SCHEMA
        and checkpoint.get("run_signature") == run_signature
        and canonical_json_sha256(provenance) == run_signature
        and checkpoint.get("provenance") == provenance
        and checkpoint_step == best_step
        and checkpoint_score == best_score
        and actual_model_digest == best_model_state_sha256
        and checkpoint.get("model_state_sha256") == actual_model_digest
        and actual_normalizer_digest == normalizer_state_sha256
        and checkpoint.get("normalizer_state_sha256") == actual_normalizer_digest
    )


def _recoverable_current_best_validation(
    history: Sequence[Mapping[str, Any]],
    *,
    completed_step: int,
    best_step: int,
    best_score: float,
    current_model_state_sha256: str,
    best_model_state_sha256: str | None,
) -> Mapping[str, Any] | None:
    if best_step != completed_step or not history:
        return None
    last_row = history[-1]
    validation = last_row.get("validation")
    if not isinstance(validation, Mapping):
        return None
    try:
        row_step = int(last_row.get("completed_step", -1))
        row_score = float(validation.get("realm_npe_mean", math.inf))
    except (TypeError, ValueError):
        return None
    if (
        row_step != completed_step
        or row_score != best_score
        or current_model_state_sha256 != best_model_state_sha256
    ):
        return None
    return validation


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("JSON root must be an object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _torch_save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _prepare_output_directory(
    output_dir: Path,
    data_root: Path,
    resume_checkpoint: Path | None,
) -> None:
    if _is_within(output_dir, data_root):
        raise ValueError("output directory must be outside the exact data root")
    if resume_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("new output directory must be absent or empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    expected = output_dir / "last.pt"
    if resume_checkpoint.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("resume is permitted only from OUTPUT_DIR/last.pt")
    if not output_dir.is_dir() or not expected.is_file():
        raise FileNotFoundError("resume output directory or last.pt is missing")


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def frozen_training_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_reconstruction_v1",
        "run_id": RUN_ID,
        "claim_kind": "declared_released_source_reconstruction",
        "paper_faithful": False,
        "seed": SEED,
        "model": asdict(RealmFFNOConfig()),
        "parameter_target": 8_936_500,
        "parameter_relative_tolerance": 0.005,
        "target_parameterization": "direct_next_normalized_state",
        "exposure": "one_call_all_29_adjacent_pairs",
        "ordered_train_groups": list(TRAIN_GROUPS),
        "ordered_validation_groups": list(VAL_GROUPS),
        "total_steps": TOTAL_STEPS,
        "presentations_per_step": EFFECTIVE_BATCH_SIZE,
        "total_presentations": TOTAL_STEPS * EFFECTIVE_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer": {
            "name": "Adam",
            "max_lr_argument": MAX_LR,
            "betas": list(ADAM_BETAS),
            "epsilon": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
        },
        "scheduler": {
            "name": "OneCycleLR",
            "total_steps": SCHEDULER_TOTAL_STEPS,
            "max_lr": MAX_LR,
            "pct_start": ONECYCLE_PCT_START,
            "anneal_strategy": "cos",
            "cycle_momentum": True,
            "base_momentum": ONECYCLE_BASE_MOMENTUM,
            "max_momentum": ONECYCLE_MAX_MOMENTUM,
            "div_factor": ONECYCLE_DIV_FACTOR,
            "final_div_factor": ONECYCLE_FINAL_DIV_FACTOR,
            "three_phase": False,
            "step_order": "after_optimizer_step",
        },
        "validation_steps": list(validation_steps()),
        "validation_start_frame": 0,
        "validation_horizon": VALIDATION_HORIZON,
        "selection_metric": "realm_npe_mean",
        "selection_tie_policy": "strict_improvement_keeps_earliest",
        "precision": {
            "dtype": "float32",
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
        },
        "checkpoint_schemas": {
            "best": BEST_CHECKPOINT_SCHEMA,
            "last": LAST_CHECKPOINT_SCHEMA,
        },
        "test_object_available": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        Path(__file__).resolve(),
    )
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_executed_source_v1",
        "files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _runtime_manifest(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": SEED,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _normalizers_from_arrays(
    path: Path,
) -> tuple[RealmNormalizer, RealmNormalizer, torch.Tensor, dict[str, Any]]:
    if sha256_file(path) != NORMALIZER_ARRAYS_SHA256:
        raise ValueError("normalizer arrays SHA-256 does not match frozen P1b")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != NORMALIZER_ARRAY_KEYS:
            raise ValueError("normalizer array keys do not match the P1b contract")
        mean = np.asarray(arrays["primary_mean"])
        scale = np.asarray(arrays["primary_scale"])
        train_max_abs = np.asarray(arrays["train_max_abs"])
    for name, value in (
        ("mean", mean),
        ("scale", scale),
        ("train_max_abs", train_max_abs),
    ):
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError(f"normalizer {name} must be a finite 12-channel vector")
    if not (scale > 0.0).all() or not (train_max_abs >= 0.0).all():
        raise ValueError("normalizer scale/envelope domain is invalid")

    def make(channel_axis: int) -> RealmNormalizer:
        return RealmNormalizer(
            mean=torch.as_tensor(mean, dtype=DTYPE),
            scale=torch.as_tensor(scale, dtype=DTYPE),
            transformed_channels=tuple(range(8)),
            channel_axis=channel_axis,
            box_cox_lambda=BOX_COX_LAMBDA,
            box_cox_epsilon=PRIMARY_BOX_COX_EPSILON,
            std_correction=STD_CORRECTION,
            scale_stabilizer=SCALE_STABILIZER,
        )

    state = {
        "mean": torch.as_tensor(mean, dtype=DTYPE),
        "scale": torch.as_tensor(scale, dtype=DTYPE),
        "transformed_channels": tuple(range(8)),
        "box_cox_lambda": BOX_COX_LAMBDA,
        "box_cox_epsilon": PRIMARY_BOX_COX_EPSILON,
        "std_correction": STD_CORRECTION,
        "scale_stabilizer": SCALE_STABILIZER,
        "source_sha256": NORMALIZER_ARRAYS_SHA256,
    }
    return make(1), make(2), torch.as_tensor(train_max_abs, dtype=DTYPE), state


def _load_normalized_trajectories(
    data_root: Path,
    split: str,
    groups: Sequence[str],
    normalizer: RealmNormalizer,
    *,
    retain_native: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    normalized = torch.empty((len(groups), *TRAJECTORY_SHAPE), dtype=DTYPE)
    native_out = (
        torch.empty((len(groups), *TRAJECTORY_SHAPE), dtype=DTYPE)
        if retain_native
        else None
    )
    for index, group in enumerate(groups):
        relative = trajectory_relative_path(split, group)
        native_array = load_ignithit_trajectory(
            data_root.joinpath(*PurePosixPath(relative).parts)
        )
        if native_array.dtype != TRAJECTORY_DTYPE:
            raise ValueError("trajectory dtype differs from the frozen contract")
        native = torch.from_numpy(native_array)
        normalized[index].copy_(normalizer.encode(native))
        if native_out is not None:
            native_out[index].copy_(native)
    return normalized, native_out


def _all_finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _run_validation(
    model: nn.Module,
    validation_normalized: torch.Tensor,
    validation_native: torch.Tensor,
    coordinates: torch.Tensor,
    trajectory_normalizer: RealmNormalizer,
    train_max_abs: torch.Tensor,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        prediction = direct_rollout(
            model,
            validation_normalized[:, 0],
            coordinates,
            calls=VALIDATION_HORIZON,
        )
        decoded = trajectory_normalizer.decode(
            prediction,
            inverse_domain_policy="nan",
        )
        summary = summarize_validation_predictions(
            prediction,
            validation_normalized[:, 1 : VALIDATION_HORIZON + 1],
            decoded,
            validation_native[:, 1 : VALIDATION_HORIZON + 1],
            case_keys=VAL_GROUPS,
            train_max_abs=train_max_abs,
        )
    model.train(was_training)
    return summary


def _validate_stop_after_step(value: int | None) -> int:
    if value is None:
        return TOTAL_STEPS
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("stop-after-step must be an integer")
    if value not in validation_steps() or value >= TOTAL_STEPS:
        raise ValueError(
            "stop-after-step must be a registered pre-final validation step"
        )
    return value


def _write_or_validate_json(
    path: Path, payload: Mapping[str, Any], *, resume: bool
) -> None:
    if resume:
        if _load_json(path) != payload:
            raise ValueError(f"resume manifest differs: {path.name}")
    else:
        _write_json_atomic(path, payload)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    stop_after_step = _validate_stop_after_step(args.stop_after_step)
    _prepare_output_directory(args.output_dir, args.data_root, args.resume_checkpoint)
    resume = args.resume_checkpoint is not None

    manifest_payload = _load_json(args.manifest)
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    manifest_summary = validate_ignithit_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_ignithit_metadata(args.data_root / "data" / "data.npz")
    if (
        tuple(metadata.train_groups) != TRAIN_GROUPS
        or tuple(metadata.val_groups) != VAL_GROUPS
    ):
        raise ValueError(
            "metadata split order differs from the frozen training contract"
        )
    if len(set(VAL_GROUPS)) != len(VAL_GROUPS) or len(VAL_GROUPS) != 5:
        raise RuntimeError("validation population must contain five unique keys")
    if len(TRAIN_GROUPS) != EFFECTIVE_BATCH_SIZE:
        raise RuntimeError("effective batch must equal the 26 registered train cases")

    state_normalizer, trajectory_normalizer, train_max_abs, normalizer_state = (
        _normalizers_from_arrays(args.normalizer_arrays)
    )
    _configure_determinism()
    device = torch.device(DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the frozen baseline requires exactly one visible CUDA device"
        )
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    config = frozen_training_contract()
    source_manifest = _source_manifest()
    runtime_manifest = _runtime_manifest(device)
    input_manifest: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_inputs_v1",
        "repository": repository,
        "revision": revision,
        "open_manifest_sha256": manifest_summary["manifest_sha256"],
        "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        "open_entry_count": len(inventory),
        "open_total_bytes": sum(entry.size for entry in entries),
        "normalizer_arrays_sha256": NORMALIZER_ARRAYS_SHA256,
        "ordered_train_groups": list(TRAIN_GROUPS),
        "ordered_validation_groups": list(VAL_GROUPS),
        "test_object_opened": False,
    }
    input_manifest["canonical_payload_sha256"] = canonical_json_sha256(input_manifest)
    if input_manifest["open_manifest_sha256"] != IGNITHIT_OPEN_MANIFEST_SHA256:
        raise RuntimeError("validated open manifest differs from the frozen identity")
    provenance = {
        "config_digest": config["canonical_payload_sha256"],
        "input_digest": input_manifest["canonical_payload_sha256"],
        "source_digest": source_manifest["canonical_payload_sha256"],
        "runtime_digest": runtime_manifest["canonical_payload_sha256"],
    }
    run_signature = canonical_json_sha256(provenance)

    for name, payload in (
        ("config.json", config),
        ("input_manifest.json", input_manifest),
        ("source_manifest.json", source_manifest),
        ("runtime_manifest.json", runtime_manifest),
    ):
        _write_or_validate_json(args.output_dir / name, payload, resume=resume)

    train_normalized, _ = _load_normalized_trajectories(
        args.data_root,
        "train",
        TRAIN_GROUPS,
        state_normalizer,
        retain_native=False,
    )
    validation_normalized, validation_native = _load_normalized_trajectories(
        args.data_root,
        "val",
        VAL_GROUPS,
        state_normalizer,
        retain_native=True,
    )
    if validation_native is None:
        raise RuntimeError("validation native truth was not retained")
    validation_normalized_device = validation_normalized.to(device)
    validation_native_device = validation_native.to(device)

    coordinates = normalize_realm_coordinates(
        torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=DTYPE)
    ).to(device)
    model = RealmFFNO2d().to(device=device, dtype=DTYPE)
    parameter_count = trainable_parameter_count(model)
    if not parameter_count_within_reported_tolerance(parameter_count):
        raise RuntimeError("FFNO-M parameter count is outside the frozen tolerance")
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    order_generator = random.Random(SEED)
    completed_step = 0
    best_step = 0
    best_score = math.inf
    best_model_state_sha256: str | None = None
    history: list[dict[str, Any]] = []
    if resume:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("resume checkpoint root must be a mapping")
        (
            completed_step,
            best_step,
            best_score,
            best_model_state_sha256,
            history,
        ) = restore_last_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            order_generator,
            expected_run_signature=run_signature,
        )
        best_path = args.output_dir / "best.pt"
        best_checkpoint: Mapping[str, Any] | None = None
        if best_path.is_file():
            loaded_best = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(loaded_best, Mapping):
                raise TypeError("best checkpoint root must be a mapping")
            best_checkpoint = loaded_best
        normalizer_state_sha256 = structured_state_sha256(normalizer_state)
        best_matches = best_checkpoint is not None and _best_checkpoint_matches(
            best_checkpoint,
            run_signature=run_signature,
            best_step=best_step,
            best_score=best_score,
            best_model_state_sha256=best_model_state_sha256,
            normalizer_state_sha256=normalizer_state_sha256,
            provenance=provenance,
        )
        if not best_matches:
            recoverable_validation = _recoverable_current_best_validation(
                history,
                completed_step=completed_step,
                best_step=best_step,
                best_score=best_score,
                current_model_state_sha256=structured_state_sha256(model.state_dict()),
                best_model_state_sha256=best_model_state_sha256,
            )
            if recoverable_validation is None:
                raise ValueError("best and last checkpoint identities differ")
            recovered_best = build_best_checkpoint(
                model,
                run_signature=run_signature,
                completed_step=completed_step,
                validation=recoverable_validation,
                normalizer_state=normalizer_state,
                provenance=provenance,
            )
            _torch_save_atomic(best_path, recovered_best)
        _write_json_atomic(
            args.output_dir / "history.json",
            {"schema": "d088_ignithit_ffno_direct_history_v1", "rows": history},
        )
        if completed_step >= stop_after_step:
            raise ValueError("resume checkpoint already reached the requested stop")

    started_at = time.monotonic()
    registered_validation_steps = set(validation_steps())
    for step in range(completed_step + 1, stop_after_step + 1):
        sample = draw_step_sample(order_generator)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        group_sums: Counter[str] = Counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for case_index in sample.case_indices:
            input_state = (
                train_normalized[case_index, sample.frame_start].unsqueeze(0).to(device)
            )
            target_state = (
                train_normalized[case_index, sample.frame_start + 1]
                .unsqueeze(0)
                .to(device)
            )
            prediction = model(input_state, coordinates)
            loss, by_group = grouped_next_state_mse(prediction, target_state)
            if not bool(torch.isfinite(prediction).all()) or not bool(
                torch.isfinite(loss)
            ):
                raise RuntimeError("training returned a nonfinite proposal or loss")
            loss_sum += float(loss.detach().item())
            for name, value in by_group.items():
                group_sums[name] += float(value.detach().item())
            scaled_microbatch_loss(
                loss,
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
            ).backward()
        if not _all_finite_gradients(model):
            raise RuntimeError("training returned a missing or nonfinite gradient")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("optimizer step produced a nonfinite parameter")
        scheduler.step()
        completed_step = step

        if step not in registered_validation_steps:
            continue
        validation_started = time.monotonic()
        validation = _run_validation(
            model,
            validation_normalized_device,
            validation_native_device,
            coordinates,
            trajectory_normalizer,
            train_max_abs,
        )
        validation_seconds = time.monotonic() - validation_started
        score = float(validation["realm_npe_mean"])
        if not math.isfinite(score):
            raise RuntimeError("selection metric is nonfinite")
        row = {
            "completed_step": step,
            "frame_start": sample.frame_start,
            "case_order": list(sample.case_indices),
            "learning_rate_used": learning_rate_used,
            "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "mean_train_grouped_loss": loss_sum / EFFECTIVE_BATCH_SIZE,
            "mean_train_loss_by_group": {
                name: value / EFFECTIVE_BATCH_SIZE
                for name, value in sorted(group_sums.items())
            },
            "validation_seconds": validation_seconds,
            "validation": validation,
        }
        history.append(row)
        improved = is_strict_improvement(score, best_score)
        if improved:
            best_score = score
            best_step = step
            best_model_state_sha256 = structured_state_sha256(model.state_dict())
        last_payload = build_last_checkpoint(
            model,
            optimizer,
            scheduler,
            order_generator,
            run_signature=run_signature,
            completed_step=step,
            best_step=best_step,
            best_score=best_score,
            best_model_state_sha256=best_model_state_sha256,
            history=history,
            provenance=provenance,
        )
        _torch_save_atomic(args.output_dir / "last.pt", last_payload)
        if improved:
            best_payload = build_best_checkpoint(
                model,
                run_signature=run_signature,
                completed_step=step,
                validation=validation,
                normalizer_state=normalizer_state,
                provenance=provenance,
            )
            _torch_save_atomic(args.output_dir / "best.pt", best_payload)
        _write_json_atomic(
            args.output_dir / "history.json",
            {"schema": "d088_ignithit_ffno_direct_history_v1", "rows": history},
        )

    complete = completed_step == TOTAL_STEPS
    status: dict[str, Any] = {
        "schema": "d088_ignithit_ffno_direct_status_v1",
        "run_id": RUN_ID,
        "run_signature": run_signature,
        "complete": complete,
        "completed_step": completed_step,
        "target_steps": TOTAL_STEPS,
        "best_step": best_step,
        "best_realm_npe_mean": best_score,
        "elapsed_seconds_this_process": time.monotonic() - started_at,
        "peak_cuda_allocated_bytes_this_process": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_cuda_reserved_bytes_this_process": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "checkpoint_paths": {"best": "best.pt", "last": "last.pt"},
        "anti_claims": [
            "this is a declared released-source reconstruction, not paper-faithful history",
            "validation is a model-selection population, not untouched test evidence",
            "released-state admissibility is not complete composition conservation",
            "a partial run is not a baseline result",
        ],
    }
    _write_json_atomic(args.output_dir / "status.json", status)
    if complete:
        hashes = {
            name: sha256_file(args.output_dir / name)
            for name in (
                "config.json",
                "input_manifest.json",
                "source_manifest.json",
                "runtime_manifest.json",
                "history.json",
                "best.pt",
                "last.pt",
                "status.json",
            )
        }
        final_manifest = {
            "schema": "d088_ignithit_ffno_direct_final_hash_manifest_v1",
            "files": hashes,
            "self_hash_excluded": True,
        }
        _write_json_atomic(args.output_dir / "final_hash_manifest.json", final_manifest)
        status["final_hash_manifest_sha256"] = sha256_file(
            args.output_dir / "final_hash_manifest.json"
        )
        _write_json_atomic(args.output_dir / "summary.json", status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        status = run_training(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
