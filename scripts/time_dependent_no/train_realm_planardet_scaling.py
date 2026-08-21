"""Preflight and train the controlled REALM PlanarDet scaling study.

The executable has two fail-closed modes. ``preflight`` measures one exact
full-grid two-call optimizer step for a frozen architecture. ``train`` requires
a preregistration that binds that preflight, the open release, the audited
all-train normalizer, and every transitive executable source.  Neither mode
accepts a test path or discovers a released test object.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.train_realm_planardet_pcno import (
    AUDIT_MANIFEST_SCHEMA,
    _all_finite_gradients,
    _configure_determinism,
    _load_json,
    _prepare_output_directory,
    _run_teacher_validation,
    _torch_save_atomic,
    _write_json_atomic,
    _write_or_validate_json,
)
from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    parse_manifest_payload,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    normalize_realm_coordinates,
)
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
)
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_COORDINATE_ORDER,
    DOMAIN_LENGTHS_XY,
    PLANARDET_OPEN_MANIFEST_SHA256,
    PLANARDET_TRAIN_GROUPS,
    PLANARDET_VAL_GROUPS,
    PlanarDetMetadata,
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    build_source_manifest,
    recursive_to_cpu,
    structured_state_sha256,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    ADAM_BETAS,
    ADAM_EPS,
    CHANNELS,
    EFFECTIVE_BATCH_SIZE,
    FC_DIM,
    MAX_LR,
    MODE_COUNTS_XY,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    WEIGHT_DECAY,
    PlanarDetNormalizerBundle,
    competence_gate,
    grouped_planardet_mse,
    load_normalized_trajectories,
    load_normalizer_bundle,
    scaled_microbatch_loss,
    training_prediction,
    validate_frozen_controls,
    validation_control_summary,
)
from utility.time_dependent_no.realm_planardet_scaling import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    FFNO_WIDTH,
    PCNO_WIDTH,
    PREFLIGHT_SCHEMA,
    PRESENTATIONS_PER_STEP,
    SCALING_SOURCE_PATHS,
    PlanarDetScalingContract,
    scheduled_scaling_window,
    validation_steps,
)

DEVICE_NAME = "cuda:0"
DTYPE = torch.float32
MIN_RESERVED_HEADROOM_FRACTION = 0.18
BEST_CHECKPOINT_SCHEMA = "w26_l4_planardet_arch_data_scaling_best_v1"
LAST_CHECKPOINT_SCHEMA = "w26_l4_planardet_arch_data_scaling_last_v1"


def _add_release_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--data-audit-final-manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_release_arguments(preflight)
    preflight.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    _add_release_arguments(train)
    train.add_argument("--preregistration", type=Path, required=True)
    train.add_argument("--preflight-result", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume only from OUTPUT_DIR/last.pt under the identical contract",
    )
    train.add_argument(
        "--stop-after-step",
        type=int,
        help="operational stop at a registered pre-final validation step",
    )
    return parser


def _source_manifest() -> dict[str, Any]:
    return build_source_manifest(REPO_ROOT, entrypoints=SCALING_SOURCE_PATHS)


def _runtime_manifest(device: torch.device, *, seed: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_arch_data_scaling_cuda_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_signed_payload(payload: Mapping[str, Any], *, name: str) -> None:
    digest = payload.get("canonical_payload_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    if digest != canonical_json_sha256(unsigned):
        raise ValueError(f"{name} canonical payload digest differs")


def _validate_release_and_normalizer(
    *,
    manifest_path: Path,
    data_root: Path,
    normalizer_arrays_path: Path,
    data_audit_final_manifest_path: Path,
    expected_open_manifest_payload_sha256: str | None = None,
    expected_normalizer_arrays_sha256: str | None = None,
    expected_data_audit_final_manifest_sha256: str | None = None,
) -> tuple[
    Mapping[str, Any],
    str,
    str,
    list[Any],
    PlanarDetMetadata,
    PlanarDetNormalizerBundle,
]:
    manifest_payload = _load_json(manifest_path)
    manifest_payload_sha256 = canonical_json_sha256(manifest_payload)
    if (
        expected_open_manifest_payload_sha256 is not None
        and manifest_payload_sha256 != expected_open_manifest_payload_sha256
    ):
        raise ValueError("open manifest payload SHA-256 differs")
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(data_root, entries)
    metadata = load_planardet_metadata(data_root / "data" / "data.npz")
    if (
        metadata.train_groups != PLANARDET_TRAIN_GROUPS
        or metadata.val_groups != PLANARDET_VAL_GROUPS
    ):
        raise ValueError("released split order differs from scaling contract")

    audit_sha256 = sha256_file(data_audit_final_manifest_path)
    normalizer_sha256 = sha256_file(normalizer_arrays_path)
    if (
        expected_data_audit_final_manifest_sha256 is not None
        and audit_sha256 != expected_data_audit_final_manifest_sha256
    ):
        raise ValueError("closed data-audit manifest SHA-256 differs")
    if (
        expected_normalizer_arrays_sha256 is not None
        and normalizer_sha256 != expected_normalizer_arrays_sha256
    ):
        raise ValueError("normalizer arrays SHA-256 differs")
    audit = _load_json(data_audit_final_manifest_path)
    files = audit.get("files")
    if (
        audit.get("schema") != AUDIT_MANIFEST_SCHEMA
        or audit.get("open_manifest_sha256") != PLANARDET_OPEN_MANIFEST_SHA256
        or audit.get("self_hash_excluded") is not True
        or not isinstance(files, Mapping)
        or files.get("normalizer_arrays.npz") != normalizer_sha256
    ):
        raise ValueError("closed data audit does not bind release and normalizer")
    bundle = load_normalizer_bundle(
        normalizer_arrays_path,
        expected_sha256=normalizer_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    return (
        manifest_payload,
        repository,
        revision,
        inventory,
        metadata,
        bundle,
    )


def _build_model(
    architecture: str,
    metadata: PlanarDetMetadata,
) -> nn.Module:
    if architecture in {"pcno", "pcfno"}:
        geometry = build_realm_regular_grid_geometry(
            metadata.canonical_coords_yx,
            domain_lengths_xy=DOMAIN_LENGTHS_XY,
            released_coordinate_order=CANONICAL_COORDINATE_ORDER,
        )
        return RealmRegularGridPCNO(
            config=RealmPCNOConfig(
                channels=CHANNELS,
                mode_counts_xy=MODE_COUNTS_XY,
                layers=(PCNO_WIDTH,) * 5,
                fc_dim=FC_DIM,
                zero_initialize_head=True,
                use_gradient=architecture == "pcno",
            ),
            geometry=geometry,
        )
    if architecture == "ffno":
        model = RealmFFNO2d(
            RealmFFNOConfig(
                state_channels=CHANNELS,
                coordinate_channels=2,
                output_channels=CHANNELS,
                width=FFNO_WIDTH,
                layers=4,
                modes_y=32,
                modes_x=32,
                feedforward_factor=4,
                feedforward_layers=2,
                layer_norm=True,
                head_width=128,
            )
        )
        final = model.output_projection[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("FFNO output head contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        return model
    raise ValueError("architecture must be pcno, pcfno, or ffno")


def _coordinates_for_architecture(
    architecture: str,
    bundle: PlanarDetNormalizerBundle,
    *,
    device: torch.device,
) -> torch.Tensor:
    coordinates = bundle.canonical_coordinates_yx.unsqueeze(0).to(
        device=device, dtype=DTYPE
    )
    if architecture == "ffno":
        coordinates = normalize_realm_coordinates(coordinates)
    return coordinates


def _build_optimizer_and_scheduler(
    model: nn.Module,
    contract: PlanarDetScalingContract,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    optimizer = torch.optim.AdamW(
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
        total_steps=contract.total_steps,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=ONECYCLE_DIV_FACTOR,
        final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
        three_phase=False,
    )
    return optimizer, scheduler


def _validate_device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("scaling study requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("scaling study requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    return device


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("fresh preflight output must be absent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (
        manifest_payload,
        repository,
        revision,
        inventory,
        metadata,
        bundle,
    ) = _validate_release_and_normalizer(
        manifest_path=args.manifest,
        data_root=args.data_root,
        normalizer_arrays_path=args.normalizer_arrays,
        data_audit_final_manifest_path=args.data_audit_final_manifest,
    )
    source_manifest = _source_manifest()
    _configure_determinism(0)
    device = _validate_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    status = "failed"
    finite_step = False
    error: str | None = None
    parameter_count: int | None = None
    gradient_norm: float | None = None
    try:
        model = _build_model(args.architecture, metadata).to(device=device, dtype=DTYPE)
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if parameter_count != EXPECTED_PARAMETER_COUNTS[args.architecture]:
            raise RuntimeError("preflight parameter count differs")
        coordinates = _coordinates_for_architecture(
            args.architecture, bundle, device=device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=MAX_LR,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
            foreach=False,
            fused=False,
        )
        current = torch.zeros(
            (1, CHANNELS, *metadata.canonical_coords_yx.shape[1:]),
            device=device,
            dtype=DTYPE,
        )
        target = torch.full_like(current, 0.01)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(PRESENTATIONS_PER_STEP):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = training_prediction(model, current, coordinates, calls=2)
                loss, _ = grouped_planardet_mse(prediction, target)
            scaled_microbatch_loss(
                loss, effective_batch_size=PRESENTATIONS_PER_STEP
            ).backward()
        if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError("preflight proposal or loss is nonfinite")
        if not _all_finite_gradients(model):
            raise RuntimeError("preflight gradient is missing or nonfinite")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), math.inf, error_if_nonfinite=True
            )
            .detach()
            .item()
        )
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("preflight optimizer step is nonfinite")
        finite_step = True
    except torch.cuda.OutOfMemoryError as exc:
        error = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    except (RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_headroom = 1.0 - peak_reserved / total_memory
    passes_memory = reserved_headroom >= MIN_RESERVED_HEADROOM_FRACTION
    if finite_step and passes_memory:
        status = "pass"
    elif finite_step:
        status = "insufficient_headroom"
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "architecture": args.architecture,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[args.architecture],
        "finite_two_call_optimizer_step": finite_step,
        "passes_memory_gate": passes_memory,
        "minimum_reserved_headroom_fraction": MIN_RESERVED_HEADROOM_FRACTION,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
        "reserved_headroom_fraction": reserved_headroom,
        "gradient_l2_norm": gradient_norm,
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        "source_manifest": source_manifest,
        "inputs": {
            "repository": repository,
            "revision": revision,
            "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
            "open_entry_count": len(inventory),
            "normalizer_arrays_sha256": sha256_file(args.normalizer_arrays),
            "data_audit_final_manifest_sha256": sha256_file(
                args.data_audit_final_manifest
            ),
        },
        "trajectory_arrays_opened": False,
        "test_object_opened": False,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    _write_json_atomic(args.output, payload)
    return payload


def _validate_preflight(
    path: Path,
    *,
    contract: PlanarDetScalingContract,
    source_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if sha256_file(path) != contract.preflight_result_sha256:
        raise ValueError("full-grid preflight SHA-256 differs")
    payload = _load_json(path)
    _validate_signed_payload(payload, name="full-grid preflight")
    inputs = payload.get("inputs")
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("architecture") != contract.architecture
        or payload.get("parameter_count") != contract.expected_parameter_count
        or payload.get("expected_parameter_count") != contract.expected_parameter_count
        or payload.get("finite_two_call_optimizer_step") is not True
        or payload.get("passes_memory_gate") is not True
        or payload.get("source_manifest") != source_manifest
        or not isinstance(inputs, Mapping)
        or inputs.get("open_manifest_payload_sha256")
        != contract.open_manifest_payload_sha256
        or inputs.get("normalizer_arrays_sha256") != contract.normalizer_arrays_sha256
        or inputs.get("data_audit_final_manifest_sha256")
        != contract.data_audit_final_manifest_sha256
        or payload.get("trajectory_arrays_opened") is not False
        or payload.get("test_object_opened") is not False
    ):
        raise ValueError("full-grid preflight does not authorize this run")
    return payload


def _validate_stop_after_step(
    value: int | None, contract: PlanarDetScalingContract
) -> int:
    if value is None:
        return contract.total_steps
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("stop-after-step must be an integer")
    if value not in validation_steps(contract) or value >= contract.total_steps:
        raise ValueError("stop-after-step must be a registered pre-final validation")
    return value


def _rng_payload(*, include_cuda: bool) -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def _best_checkpoint(
    model: nn.Module,
    *,
    contract: PlanarDetScalingContract,
    config: Mapping[str, Any],
    run_signature: str,
    completed_step: int,
    validation: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    model_state = recursive_to_cpu(model.state_dict())
    normalizer_state_cpu = recursive_to_cpu(normalizer_state)
    return {
        "schema": BEST_CHECKPOINT_SCHEMA,
        "run_id": contract.run_id,
        "run_signature": run_signature,
        "architecture": contract.architecture,
        "completed_step": completed_step,
        "selection_metric": "teacher_forced_realm_npe_mean",
        "selection_value": float(validation["realm_npe_mean"]),
        "validation": dict(validation),
        "competence_gate": competence_gate(validation, contract),
        "model_config": dict(config["model"]),
        "model_state": model_state,
        "model_state_sha256": structured_state_sha256(model_state),
        "normalizer_state": normalizer_state_cpu,
        "normalizer_state_sha256": structured_state_sha256(normalizer_state_cpu),
        "provenance": dict(provenance),
        "resume_supported": False,
        "test_object_opened": False,
    }


def _last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    contract: PlanarDetScalingContract,
    run_signature: str,
    completed_step: int,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str | None,
    history: Sequence[Mapping[str, Any]],
    elapsed_seconds_total: float,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    model_state = recursive_to_cpu(model.state_dict())
    payload: dict[str, Any] = {
        "schema": LAST_CHECKPOINT_SCHEMA,
        "run_id": contract.run_id,
        "run_signature": run_signature,
        "architecture": contract.architecture,
        "completed_step": completed_step,
        "best_step": best_step,
        "best_score": best_score,
        "best_model_state_sha256": best_model_state_sha256,
        "history": [dict(row) for row in history],
        "elapsed_seconds_total": elapsed_seconds_total,
        "model_state": model_state,
        "model_state_sha256": structured_state_sha256(model_state),
        "optimizer_state": recursive_to_cpu(optimizer.state_dict()),
        "scheduler_state": recursive_to_cpu(scheduler.state_dict()),
        "provenance": dict(provenance),
        "resume_supported": True,
        "test_object_opened": False,
    }
    payload.update(
        recursive_to_cpu(
            _rng_payload(
                include_cuda=any(parameter.is_cuda for parameter in model.parameters())
            )
        )
    )
    return payload


def _restore_last_checkpoint(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    contract: PlanarDetScalingContract,
    expected_run_signature: str,
) -> tuple[int, int, float, str | None, list[dict[str, Any]], float]:
    if (
        checkpoint.get("schema") != LAST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != contract.run_id
        or checkpoint.get("run_signature") != expected_run_signature
        or checkpoint.get("architecture") != contract.architecture
        or checkpoint.get("resume_supported") is not True
        or checkpoint.get("test_object_opened") is not False
    ):
        raise ValueError("resume checkpoint identity differs")
    provenance = checkpoint.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or canonical_json_sha256(provenance) != expected_run_signature
    ):
        raise ValueError("resume checkpoint provenance differs")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or structured_state_sha256(
        model_state
    ) != checkpoint.get("model_state_sha256"):
        raise ValueError("resume model-state digest differs")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_state = checkpoint.get("cuda_rng_state")
    if not isinstance(cuda_state, Sequence) or len(cuda_state) != 1:
        raise ValueError("resume CUDA RNG state differs")
    torch.cuda.set_rng_state_all(list(cuda_state))

    completed = int(checkpoint["completed_step"])
    best_step = int(checkpoint["best_step"])
    best_score = float(checkpoint["best_score"])
    best_hash = checkpoint.get("best_model_state_sha256")
    history = checkpoint.get("history")
    elapsed = float(checkpoint.get("elapsed_seconds_total", math.nan))
    if (
        completed not in validation_steps(contract)
        or best_step < 0
        or best_step > completed
        or not isinstance(history, list)
        or (history and history[-1].get("completed_step") != completed)
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or (best_step == 0) != (best_hash is None)
        or (
            best_step > 0
            and (not math.isfinite(best_score) or not isinstance(best_hash, str))
        )
    ):
        raise ValueError("resume progress state differs")
    return completed, best_step, best_score, best_hash, list(history), elapsed


def _validate_retained_best(
    path: Path,
    *,
    contract: PlanarDetScalingContract,
    run_signature: str,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str,
    provenance: Mapping[str, str],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("retained best checkpoint is missing")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("retained best checkpoint root must be a mapping")
    model_state = checkpoint.get("model_state")
    if (
        checkpoint.get("schema") != BEST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != contract.run_id
        or checkpoint.get("run_signature") != run_signature
        or checkpoint.get("architecture") != contract.architecture
        or checkpoint.get("completed_step") != best_step
        or checkpoint.get("selection_value") != best_score
        or checkpoint.get("model_state_sha256") != best_model_state_sha256
        or not isinstance(model_state, Mapping)
        or structured_state_sha256(model_state) != best_model_state_sha256
        or checkpoint.get("provenance") != provenance
        or checkpoint.get("resume_supported") is not False
        or checkpoint.get("test_object_opened") is not False
    ):
        raise ValueError("retained best checkpoint identity differs")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = _load_json(args.preregistration)
    contract = PlanarDetScalingContract.from_payload(preregistration)
    stop_after_step = _validate_stop_after_step(args.stop_after_step, contract)
    (
        manifest_payload,
        repository,
        revision,
        inventory,
        metadata,
        bundle,
    ) = _validate_release_and_normalizer(
        manifest_path=args.manifest,
        data_root=args.data_root,
        normalizer_arrays_path=args.normalizer_arrays,
        data_audit_final_manifest_path=args.data_audit_final_manifest,
        expected_open_manifest_payload_sha256=(contract.open_manifest_payload_sha256),
        expected_normalizer_arrays_sha256=contract.normalizer_arrays_sha256,
        expected_data_audit_final_manifest_sha256=(
            contract.data_audit_final_manifest_sha256
        ),
    )
    source_manifest = _source_manifest()
    _validate_preflight(
        args.preflight_result,
        contract=contract,
        source_manifest=source_manifest,
    )
    state_normalizer = bundle.normalizer(1, dtype=DTYPE)
    validation_normalized, validation_native = load_normalized_trajectories(
        args.data_root,
        split="val",
        groups=PLANARDET_VAL_GROUPS,
        normalizer=state_normalizer,
        retain_native=True,
    )
    if validation_native is None:
        raise RuntimeError("validation native truth was not retained")
    controls = validation_control_summary(validation_normalized[0])
    validate_frozen_controls(controls, contract)

    _configure_determinism(contract.seed)
    device = _validate_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    process_started_at = time.monotonic()
    runtime_manifest = _runtime_manifest(device, seed=contract.seed)
    config = contract.frozen_training_config()
    excluded_groups = [
        group
        for group in PLANARDET_TRAIN_GROUPS
        if group not in contract.active_train_groups
    ]
    input_manifest: dict[str, Any] = {
        "schema": "w26_l4_planardet_arch_data_scaling_inputs_v1",
        "repository": repository,
        "revision": revision,
        "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        "open_entry_count": len(inventory),
        "normalizer_arrays_sha256": contract.normalizer_arrays_sha256,
        "normalizer_fit_population": "all_seven_released_train_trajectories",
        "data_audit_final_manifest_sha256": (contract.data_audit_final_manifest_sha256),
        "full_grid_preflight_result_sha256": contract.preflight_result_sha256,
        "ordered_active_train_indices": list(contract.active_train_indices),
        "ordered_active_train_groups": list(contract.active_train_groups),
        "excluded_supervised_train_groups": excluded_groups,
        "excluded_trajectory_arrays_opened_by_trainer": False,
        "ordered_validation_groups": list(PLANARDET_VAL_GROUPS),
        "validation_controls": controls,
        "metadata": metadata.summary(),
        "test_object_opened": False,
    }
    input_manifest["canonical_payload_sha256"] = canonical_json_sha256(input_manifest)
    provenance = {
        "config_digest": str(config["canonical_payload_sha256"]),
        "input_digest": str(input_manifest["canonical_payload_sha256"]),
        "source_digest": str(source_manifest["canonical_payload_sha256"]),
        "runtime_digest": str(runtime_manifest["canonical_payload_sha256"]),
    }
    run_signature = canonical_json_sha256(provenance)

    resume = _prepare_output_directory(
        args.output_dir,
        data_root=args.data_root,
        resume_checkpoint=args.resume_checkpoint,
    )
    for name, payload in (
        ("preregistration.json", preregistration),
        ("config.json", config),
        ("input_manifest.json", input_manifest),
        ("source_manifest.json", source_manifest),
        ("runtime_manifest.json", runtime_manifest),
    ):
        _write_or_validate_json(args.output_dir / name, payload, resume=resume)

    train_normalized, _ = load_normalized_trajectories(
        args.data_root,
        split="train",
        groups=contract.active_train_groups,
        normalizer=state_normalizer,
        retain_native=False,
    )
    model = _build_model(contract.architecture, metadata).to(device=device, dtype=DTYPE)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != contract.expected_parameter_count:
        raise RuntimeError("training parameter count differs from preregistration")
    coordinates_device = _coordinates_for_architecture(
        contract.architecture, bundle, device=device
    )
    optimizer, scheduler = _build_optimizer_and_scheduler(model, contract)
    normalizer_state = bundle.checkpoint_state()

    completed_step = 0
    best_step = 0
    best_score = math.inf
    best_model_state_sha256: str | None = None
    history: list[dict[str, Any]] = []
    elapsed_seconds_before = 0.0
    if resume:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("resume checkpoint root must be a mapping")
        (
            completed_step,
            best_step,
            best_score,
            best_model_state_sha256,
            history,
            elapsed_seconds_before,
        ) = _restore_last_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            contract=contract,
            expected_run_signature=run_signature,
        )
        if completed_step >= stop_after_step:
            raise ValueError("resume checkpoint already reached requested stop")
        if best_step > 0:
            if best_model_state_sha256 is None:
                raise ValueError("resume best checkpoint hash is missing")
            _validate_retained_best(
                args.output_dir / "best.pt",
                contract=contract,
                run_signature=run_signature,
                best_step=best_step,
                best_score=best_score,
                best_model_state_sha256=best_model_state_sha256,
                provenance=provenance,
            )

    validation_step_set = set(validation_steps(contract))
    interval_loss_sum = 0.0
    interval_group_sums: Counter[str] = Counter()
    interval_case_presentations: Counter[int] = Counter()
    interval_gradient_norm_sum = 0.0
    interval_gradient_norm_max = 0.0
    interval_steps = 0
    stopped_for_wall_budget = False
    for step in range(completed_step + 1, stop_after_step + 1):
        sample = scheduled_scaling_window(
            step,
            one_call_steps=contract.one_call_steps,
            seed=contract.seed,
            case_count=contract.train_trajectory_count,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        group_sums: Counter[str] = Counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for case_index in sample.case_indices:
            current = (
                train_normalized[case_index, sample.frame_start].unsqueeze(0).to(device)
            )
            target = (
                train_normalized[case_index, sample.frame_start + sample.calls]
                .unsqueeze(0)
                .to(device)
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = training_prediction(
                    model, current, coordinates_device, calls=sample.calls
                )
                loss, by_group = grouped_planardet_mse(prediction, target)
            if not bool(torch.isfinite(prediction).all()) or not bool(
                torch.isfinite(loss)
            ):
                raise RuntimeError("training proposal or loss is nonfinite")
            loss_sum += float(loss.detach().item())
            for name, value in by_group.items():
                group_sums[name] += float(value.detach().item())
            interval_case_presentations[case_index] += 1
            scaled_microbatch_loss(
                loss, effective_batch_size=PRESENTATIONS_PER_STEP
            ).backward()
        if not _all_finite_gradients(model):
            raise RuntimeError("training gradient is missing or nonfinite")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), math.inf, error_if_nonfinite=True
            )
            .detach()
            .item()
        )
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("optimizer produced a nonfinite parameter")
        scheduler.step()
        completed_step = step
        interval_loss_sum += loss_sum / EFFECTIVE_BATCH_SIZE
        for name, value in group_sums.items():
            interval_group_sums[name] += value / EFFECTIVE_BATCH_SIZE
        interval_gradient_norm_sum += gradient_norm
        interval_gradient_norm_max = max(interval_gradient_norm_max, gradient_norm)
        interval_steps += 1

        if step not in validation_step_set:
            continue
        validation_started = time.monotonic()
        validation = _run_teacher_validation(
            model,
            validation_normalized,
            validation_native,
            coordinates_device,
            bundle,
            x=metadata.x,
            y=metadata.y,
            device=device,
        )
        validation_seconds = time.monotonic() - validation_started
        score = float(validation["realm_npe_mean"])
        if not math.isfinite(score):
            raise RuntimeError("validation selection metric is nonfinite")
        eligible = step >= contract.checkpoint_eligible_from_step
        improved = eligible and score < best_score
        elapsed_total = elapsed_seconds_before + time.monotonic() - process_started_at
        row = {
            "completed_step": step,
            "phase": sample.phase,
            "calls": sample.calls,
            "frame_start": sample.frame_start,
            "phase_cycle": sample.phase_cycle,
            "last_step_case_order": list(sample.case_indices),
            "interval_case_presentations": {
                contract.active_train_groups[index]: interval_case_presentations[index]
                for index in range(contract.train_trajectory_count)
            },
            "learning_rate_used": learning_rate_used,
            "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "interval_optimizer_steps": interval_steps,
            "interval_total_presentations": interval_steps * PRESENTATIONS_PER_STEP,
            "interval_mean_train_grouped_loss": interval_loss_sum / interval_steps,
            "interval_mean_train_loss_by_group": {
                name: value / interval_steps
                for name, value in sorted(interval_group_sums.items())
            },
            "interval_mean_gradient_l2_norm": interval_gradient_norm_sum
            / interval_steps,
            "interval_max_gradient_l2_norm": interval_gradient_norm_max,
            "gradient_clipping_applied": False,
            "validation_seconds": validation_seconds,
            "checkpoint_eligible": eligible,
            "strict_improvement": improved,
            "validation": validation,
            "competence_gate": competence_gate(validation, contract),
            "elapsed_seconds_total": elapsed_total,
        }
        history.append(row)
        if improved:
            best_score = score
            best_step = step
            best_model_state_sha256 = structured_state_sha256(model.state_dict())
            _torch_save_atomic(
                args.output_dir / "best.pt",
                _best_checkpoint(
                    model,
                    contract=contract,
                    config=config,
                    run_signature=run_signature,
                    completed_step=step,
                    validation=validation,
                    normalizer_state=normalizer_state,
                    provenance=provenance,
                ),
            )
        _torch_save_atomic(
            args.output_dir / "last.pt",
            _last_checkpoint(
                model,
                optimizer,
                scheduler,
                contract=contract,
                run_signature=run_signature,
                completed_step=step,
                best_step=best_step,
                best_score=best_score,
                best_model_state_sha256=best_model_state_sha256,
                history=history,
                elapsed_seconds_total=elapsed_total,
                provenance=provenance,
            ),
        )
        _write_json_atomic(
            args.output_dir / "history.json",
            {
                "schema": "w26_l4_planardet_arch_data_scaling_history_v1",
                "rows": history,
            },
        )
        interval_loss_sum = 0.0
        interval_group_sums.clear()
        interval_case_presentations.clear()
        interval_gradient_norm_sum = 0.0
        interval_gradient_norm_max = 0.0
        interval_steps = 0
        if elapsed_total >= contract.max_wall_seconds and step < stop_after_step:
            stopped_for_wall_budget = True
            break

    complete = completed_step == contract.total_steps
    if complete and best_step == 0:
        raise RuntimeError("completed training has no eligible retained checkpoint")
    status: dict[str, Any] = {
        "schema": "w26_l4_planardet_arch_data_scaling_status_v1",
        "run_id": contract.run_id,
        "architecture": contract.architecture,
        "train_trajectory_count": contract.train_trajectory_count,
        "run_signature": run_signature,
        "complete": complete,
        "completed_step": completed_step,
        "target_steps": contract.total_steps,
        "total_presentations_completed": completed_step * PRESENTATIONS_PER_STEP,
        "best_step": best_step,
        "best_teacher_forced_realm_npe_mean": (
            best_score if math.isfinite(best_score) else None
        ),
        "stopped_for_registered_wall_budget": stopped_for_wall_budget,
        "elapsed_seconds_this_process": time.monotonic() - process_started_at,
        "peak_cuda_allocated_bytes_this_process": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_cuda_reserved_bytes_this_process": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "checkpoint_paths": {"best": "best.pt", "last": "last.pt"},
        "anti_claims": [
            "three-case normalization uses aggregate statistics from all seven released train trajectories",
            "one seed and one validation condition do not support a generic architecture claim",
            "validation is model-selection evidence and not untouched test evidence",
            "matched explicit regularization does not make architecture capacities equivalent",
            "pMax is cumulative and released fields do not establish complete conservation",
        ],
    }
    _write_json_atomic(args.output_dir / "status.json", status)
    if complete:
        hashes = {
            name: sha256_file(args.output_dir / name)
            for name in (
                "preregistration.json",
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
            "schema": "w26_l4_planardet_arch_data_scaling_final_hash_manifest_v1",
            "files": hashes,
            "self_hash_excluded": True,
            "test_object_opened": False,
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
        result = (
            run_preflight(args) if args.command == "preflight" else run_training(args)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "preflight" and result["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
