"""Train one preregistered REALM PlanarDet residual-PCNO checkpoint.

The command fails closed unless the pinned open manifest, closed A2 audit,
normalizer, full-grid smoke, current executable sources, and resource-dependent
preregistration all agree.  It never accepts or discovers a test object.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    parse_manifest_payload,
    predict_one_call,
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
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
    recursive_to_cpu,
    structured_state_sha256,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    ADAM_BETAS,
    ADAM_EPS,
    CHANNELS,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_PARAMETER_COUNTS,
    FC_DIM,
    MAX_LR,
    MODE_COUNTS_XY,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    VALIDATION_HORIZON,
    WEIGHT_DECAY,
    PlanarDetNormalizerBundle,
    PlanarDetTrainingContract,
    competence_gate,
    grouped_planardet_mse,
    load_normalized_trajectories,
    load_normalizer_bundle,
    planardet_boundary_mask,
    scaled_microbatch_loss,
    scheduled_step_window,
    summarize_planardet_predictions,
    training_prediction,
    validate_frozen_controls,
    validation_control_summary,
    validation_steps,
)

DEVICE_NAME = "cuda:0"
DTYPE = torch.float32
BEST_CHECKPOINT_SCHEMA = "w26_l4_planardet_pd0_a3_best_v1"
LAST_CHECKPOINT_SCHEMA = "w26_l4_planardet_pd0_a3_last_v1"
SMOKE_SCHEMA = "w26_l4_planardet_pd0_a2_full_grid_smoke_v1"
AUDIT_MANIFEST_SCHEMA = "w26_l4_planardet_pd0_a2_data_final_hash_manifest_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--data-audit-final-manifest", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume only from OUTPUT_DIR/last.pt under the identical contract",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="operational stop at a registered pre-final validation step",
    )
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON input: {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root must be an object: {path.name}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"atomic JSON temporary path already exists: {path.name}")
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _torch_save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"checkpoint temporary path already exists: {path.name}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _prepare_output_directory(
    output_dir: Path,
    *,
    data_root: Path,
    resume_checkpoint: Path | None,
) -> bool:
    if _is_within(output_dir, data_root) or _is_within(data_root, output_dir):
        raise ValueError("training output and exact data root must be disjoint")
    if resume_checkpoint is None:
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("new training output directory must be absent")
        output_dir.mkdir(parents=True)
        return False
    expected = output_dir / "last.pt"
    if resume_checkpoint.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("resume is permitted only from OUTPUT_DIR/last.pt")
    if output_dir.is_symlink() or not output_dir.is_dir() or not expected.is_file():
        raise ValueError("resume output directory or last.pt is missing")
    return True


def _configure_determinism(seed: int) -> None:
    workspace_config = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    if workspace_config != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _runtime_manifest(device: torch.device, *, seed: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a3_cuda_runtime_v1",
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


def _validate_a2_bindings(
    contract: PlanarDetTrainingContract,
    *,
    audit_manifest_path: Path,
    normalizer_arrays_path: Path,
    smoke_result_path: Path,
    source_manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if sha256_file(audit_manifest_path) != contract.data_audit_final_manifest_sha256:
        raise ValueError("closed A2 data-audit manifest SHA-256 differs")
    audit = _load_json(audit_manifest_path)
    if (
        audit.get("schema") != AUDIT_MANIFEST_SCHEMA
        or audit.get("open_manifest_sha256") != PLANARDET_OPEN_MANIFEST_SHA256
        or audit.get("self_hash_excluded") is not True
    ):
        raise ValueError("closed A2 data-audit manifest contract differs")
    files = audit.get("files")
    if (
        not isinstance(files, Mapping)
        or files.get("normalizer_arrays.npz") != contract.normalizer_arrays_sha256
    ):
        raise ValueError("closed A2 audit does not bind the preregistered normalizer")
    if sha256_file(normalizer_arrays_path) != contract.normalizer_arrays_sha256:
        raise ValueError("normalizer arrays SHA-256 differs from preregistration")

    if sha256_file(smoke_result_path) != contract.smoke_result_sha256:
        raise ValueError("full-grid smoke result SHA-256 differs from preregistration")
    smoke = _load_json(smoke_result_path)
    _validate_signed_payload(smoke, name="full-grid smoke")
    configuration = smoke.get("configuration")
    prior_width128_sha256 = smoke.get("prior_width128_result_sha256")
    width_ladder_bound = (
        prior_width128_sha256 is None
        if contract.width == 128
        else isinstance(prior_width128_sha256, str)
        and len(prior_width128_sha256) == 64
        and all(value in "0123456789abcdef" for value in prior_width128_sha256)
    )
    if (
        smoke.get("schema") != SMOKE_SCHEMA
        or smoke.get("status") != "pass"
        or smoke.get("passes_full_grid_step") is not True
        or smoke.get("passes_memory_gate") is not True
        or smoke.get("test_object_opened") is not False
        or smoke.get("dataset_array_opened") is not False
        or not isinstance(configuration, Mapping)
        or configuration.get("layers") != [contract.width] * 5
        or configuration.get("parameter_count")
        != EXPECTED_PARAMETER_COUNTS[contract.width]
        or not width_ladder_bound
        or smoke.get("source_manifest") != source_manifest
    ):
        raise ValueError("full-grid smoke does not authorize this exact run envelope")
    return audit, smoke


def _build_optimizer_and_scheduler(
    model: nn.Module,
    contract: PlanarDetTrainingContract,
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


def _all_finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _rng_payload(*, include_cuda: bool) -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def build_best_checkpoint(
    model: nn.Module,
    *,
    contract: PlanarDetTrainingContract,
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
        "completed_step": completed_step,
        "selection_metric": "teacher_forced_realm_npe_mean",
        "selection_value": float(validation["realm_npe_mean"]),
        "validation": dict(validation),
        "competence_gate": competence_gate(validation, contract),
        "model_config": asdict(
            RealmPCNOConfig(
                channels=CHANNELS,
                mode_counts_xy=MODE_COUNTS_XY,
                layers=(contract.width,) * 5,
                fc_dim=FC_DIM,
                zero_initialize_head=True,
            )
        ),
        "model_state": model_state,
        "model_state_sha256": structured_state_sha256(model_state),
        "normalizer_state": normalizer_state_cpu,
        "normalizer_state_sha256": structured_state_sha256(normalizer_state_cpu),
        "provenance": dict(provenance),
        "resume_supported": False,
        "test_object_opened": False,
    }


def build_last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    contract: PlanarDetTrainingContract,
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


def restore_last_checkpoint(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    contract: PlanarDetTrainingContract,
    expected_run_signature: str,
) -> tuple[int, int, float, str | None, list[dict[str, Any]], float]:
    if (
        checkpoint.get("schema") != LAST_CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != contract.run_id
        or checkpoint.get("run_signature") != expected_run_signature
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
        raise ValueError("resume checkpoint model-state digest differs")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_state = checkpoint.get("cuda_rng_state")
    if any(parameter.is_cuda for parameter in model.parameters()):
        if not isinstance(cuda_state, Sequence) or len(cuda_state) != 1:
            raise ValueError("resume checkpoint CUDA RNG state differs")
        torch.cuda.set_rng_state_all(list(cuda_state))
    elif cuda_state is not None:
        raise ValueError("CPU resume checkpoint unexpectedly contains CUDA RNG state")

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
        raise ValueError("resume checkpoint progress state differs")
    return completed, best_step, best_score, best_hash, list(history), elapsed


def _best_checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    contract: PlanarDetTrainingContract,
    run_signature: str,
    best_step: int,
    best_score: float,
    best_model_state_sha256: str,
    normalizer_state_sha256: str,
    provenance: Mapping[str, str],
) -> bool:
    model_state = checkpoint.get("model_state")
    normalizer_state = checkpoint.get("normalizer_state")
    return bool(
        checkpoint.get("schema") == BEST_CHECKPOINT_SCHEMA
        and checkpoint.get("run_id") == contract.run_id
        and checkpoint.get("run_signature") == run_signature
        and checkpoint.get("completed_step") == best_step
        and checkpoint.get("selection_value") == best_score
        and checkpoint.get("model_state_sha256") == best_model_state_sha256
        and isinstance(model_state, Mapping)
        and structured_state_sha256(model_state) == best_model_state_sha256
        and checkpoint.get("normalizer_state_sha256") == normalizer_state_sha256
        and isinstance(normalizer_state, Mapping)
        and structured_state_sha256(normalizer_state) == normalizer_state_sha256
        and checkpoint.get("provenance") == provenance
        and checkpoint.get("resume_supported") is False
        and checkpoint.get("test_object_opened") is False
    )


def _recover_current_best_validation(
    history: Sequence[Mapping[str, Any]],
    *,
    completed_step: int,
    best_step: int,
    best_score: float,
    current_model_state_sha256: str,
    best_model_state_sha256: str | None,
) -> Mapping[str, Any] | None:
    if (
        best_step != completed_step
        or best_model_state_sha256 != current_model_state_sha256
        or not history
        or history[-1].get("completed_step") != completed_step
    ):
        return None
    validation = history[-1].get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("realm_npe_mean") != best_score
    ):
        return None
    return validation


def _run_teacher_validation(
    model: nn.Module,
    validation_normalized: torch.Tensor,
    validation_native: torch.Tensor,
    coordinates_device: torch.Tensor,
    bundle: PlanarDetNormalizerBundle,
    *,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if (
        validation_normalized.shape[0] != 1
        or validation_native.shape != validation_normalized.shape
    ):
        raise ValueError("training validation requires exactly one matching trajectory")
    was_training = model.training
    model.eval()
    predictions = torch.empty_like(validation_normalized[0, 1:])
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            current = validation_normalized[0, frame].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposal = predict_one_call(
                    model,
                    current,
                    coordinates_device,
                    parameterization="residual",
                )
            if not bool(torch.isfinite(proposal).all()):
                raise RuntimeError("teacher validation produced a nonfinite proposal")
            predictions[frame].copy_(proposal[0].to(device="cpu", dtype=DTYPE))
    model.train(was_training)

    trajectory_normalizer = bundle.normalizer(1, dtype=DTYPE)
    decoded = trajectory_normalizer.decode(predictions, inverse_domain_policy="nan")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("teacher validation produced a nonfinite decoded proposal")
    boundary = planardet_boundary_mask(
        torch.from_numpy(np.asarray(x).copy()),
        torch.from_numpy(np.asarray(y).copy()),
    )
    summary = summarize_planardet_predictions(
        predictions,
        validation_normalized[0, 1:],
        decoded,
        validation_native[0, 1:],
        validation_native[0, :-1],
        bundle=bundle,
        boundary_mask=boundary,
    )
    return summary


def _validate_stop_after_step(
    value: int | None,
    contract: PlanarDetTrainingContract,
) -> int:
    if value is None:
        return contract.total_steps
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("stop-after-step must be an integer")
    if value not in validation_steps(contract) or value >= contract.total_steps:
        raise ValueError(
            "stop-after-step must be a registered pre-final validation step"
        )
    return value


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if _load_json(path) != payload:
            raise ValueError(f"resume manifest differs: {path.name}")
    else:
        _write_json_atomic(path, payload)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = _load_json(args.preregistration)
    contract = PlanarDetTrainingContract.from_payload(preregistration)
    stop_after_step = _validate_stop_after_step(args.stop_after_step, contract)

    manifest_payload = _load_json(args.manifest)
    if canonical_json_sha256(manifest_payload) != contract.open_manifest_payload_sha256:
        raise ValueError("open manifest payload SHA-256 differs from preregistration")
    repository, revision, entries = parse_manifest_payload(manifest_payload)
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    if (
        metadata.train_groups != PLANARDET_TRAIN_GROUPS
        or metadata.val_groups != PLANARDET_VAL_GROUPS
        or len(PLANARDET_TRAIN_GROUPS) != EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError("metadata split order differs from the training contract")

    source_manifest = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    _validate_a2_bindings(
        contract,
        audit_manifest_path=args.data_audit_final_manifest,
        normalizer_arrays_path=args.normalizer_arrays,
        smoke_result_path=args.smoke_result,
        source_manifest=source_manifest,
    )
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=contract.normalizer_arrays_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("training requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    process_started_at = time.monotonic()
    runtime_manifest = _runtime_manifest(device, seed=contract.seed)
    config = contract.frozen_training_config()
    input_manifest: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a3_inputs_v1",
        "repository": repository,
        "revision": revision,
        "open_manifest_sha256": manifest_summary["manifest_sha256"],
        "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        "open_entry_count": len(inventory),
        "open_total_bytes": sum(entry.size for entry in entries),
        "normalizer_arrays_sha256": contract.normalizer_arrays_sha256,
        "data_audit_final_manifest_sha256": (contract.data_audit_final_manifest_sha256),
        "full_grid_smoke_result_sha256": contract.smoke_result_sha256,
        "ordered_train_groups": list(PLANARDET_TRAIN_GROUPS),
        "ordered_validation_groups": list(PLANARDET_VAL_GROUPS),
        "validation_controls": controls,
        "metadata": metadata.summary(),
        "test_object_opened": False,
    }
    input_manifest["canonical_payload_sha256"] = canonical_json_sha256(input_manifest)
    provenance = {
        "config_digest": config["canonical_payload_sha256"],
        "input_digest": input_manifest["canonical_payload_sha256"],
        "source_digest": source_manifest["canonical_payload_sha256"],
        "runtime_digest": runtime_manifest["canonical_payload_sha256"],
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
        groups=PLANARDET_TRAIN_GROUPS,
        normalizer=state_normalizer,
        retain_native=False,
    )
    geometry = build_realm_regular_grid_geometry(
        metadata.canonical_coords_yx,
        domain_lengths_xy=DOMAIN_LENGTHS_XY,
        released_coordinate_order=CANONICAL_COORDINATE_ORDER,
    )
    model_config = RealmPCNOConfig(
        channels=CHANNELS,
        mode_counts_xy=MODE_COUNTS_XY,
        layers=(contract.width,) * 5,
        fc_dim=FC_DIM,
        zero_initialize_head=True,
    )
    model = RealmRegularGridPCNO(config=model_config, geometry=geometry).to(
        device=device, dtype=DTYPE
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNTS[contract.width]:
        raise RuntimeError("PCNO parameter count differs from preregistration")
    coordinates_device = bundle.canonical_coordinates_yx.unsqueeze(0).to(
        device=device, dtype=DTYPE
    )
    optimizer, scheduler = _build_optimizer_and_scheduler(model, contract)

    completed_step = 0
    best_step = 0
    best_score = math.inf
    best_model_state_sha256: str | None = None
    history: list[dict[str, Any]] = []
    elapsed_seconds_before = 0.0
    normalizer_state = bundle.checkpoint_state()
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
            elapsed_seconds_before,
        ) = restore_last_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            contract=contract,
            expected_run_signature=run_signature,
        )
        if completed_step >= stop_after_step:
            raise ValueError("resume checkpoint already reached the requested stop")
        if best_step > 0 and best_model_state_sha256 is not None:
            best_path = args.output_dir / "best.pt"
            loaded_best: Mapping[str, Any] | None = None
            if best_path.is_file():
                candidate = torch.load(
                    best_path, map_location="cpu", weights_only=False
                )
                if not isinstance(candidate, Mapping):
                    raise TypeError("best checkpoint root must be a mapping")
                loaded_best = candidate
            normalizer_hash = structured_state_sha256(normalizer_state)
            matches = loaded_best is not None and _best_checkpoint_matches(
                loaded_best,
                contract=contract,
                run_signature=run_signature,
                best_step=best_step,
                best_score=best_score,
                best_model_state_sha256=best_model_state_sha256,
                normalizer_state_sha256=normalizer_hash,
                provenance=provenance,
            )
            if not matches:
                recovered_validation = _recover_current_best_validation(
                    history,
                    completed_step=completed_step,
                    best_step=best_step,
                    best_score=best_score,
                    current_model_state_sha256=structured_state_sha256(
                        model.state_dict()
                    ),
                    best_model_state_sha256=best_model_state_sha256,
                )
                if recovered_validation is None:
                    raise ValueError("best and last checkpoint identities differ")
                _torch_save_atomic(
                    best_path,
                    build_best_checkpoint(
                        model,
                        contract=contract,
                        run_signature=run_signature,
                        completed_step=completed_step,
                        validation=recovered_validation,
                        normalizer_state=normalizer_state,
                        provenance=provenance,
                    ),
                )

    started_at = process_started_at
    validation_step_set = set(validation_steps(contract))
    interval_loss_sum = 0.0
    interval_group_sums: Counter[str] = Counter()
    interval_gradient_norm_sum = 0.0
    interval_gradient_norm_max = 0.0
    interval_steps = 0
    stopped_for_wall_budget = False
    for step in range(completed_step + 1, stop_after_step + 1):
        sample = scheduled_step_window(
            step,
            one_call_steps=contract.one_call_steps,
            seed=contract.seed,
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
                    model,
                    current,
                    coordinates_device,
                    calls=sample.calls,
                )
                loss, by_group = grouped_planardet_mse(prediction, target)
            if not bool(torch.isfinite(prediction).all()) or not bool(
                torch.isfinite(loss)
            ):
                raise RuntimeError("training produced a nonfinite proposal or loss")
            loss_sum += float(loss.detach().item())
            for name, value in by_group.items():
                group_sums[name] += float(value.detach().item())
            scaled_microbatch_loss(loss).backward()
        if not _all_finite_gradients(model):
            raise RuntimeError("training produced a missing or nonfinite gradient")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                math.inf,
                error_if_nonfinite=True,
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
        if improved:
            best_score = score
            best_step = step
            best_model_state_sha256 = structured_state_sha256(model.state_dict())
        elapsed_total = elapsed_seconds_before + time.monotonic() - started_at
        row = {
            "completed_step": step,
            "phase": sample.phase,
            "calls": sample.calls,
            "frame_start": sample.frame_start,
            "phase_cycle": sample.phase_cycle,
            "case_order": list(sample.case_indices),
            "learning_rate_used": learning_rate_used,
            "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "interval_optimizer_steps": interval_steps,
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
        last_payload = build_last_checkpoint(
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
        )
        _torch_save_atomic(args.output_dir / "last.pt", last_payload)
        if improved:
            _torch_save_atomic(
                args.output_dir / "best.pt",
                build_best_checkpoint(
                    model,
                    contract=contract,
                    run_signature=run_signature,
                    completed_step=step,
                    validation=validation,
                    normalizer_state=normalizer_state,
                    provenance=provenance,
                ),
            )
        _write_json_atomic(
            args.output_dir / "history.json",
            {
                "schema": "w26_l4_planardet_pd0_a3_history_v1",
                "rows": history,
            },
        )
        interval_loss_sum = 0.0
        interval_group_sums.clear()
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
        "schema": "w26_l4_planardet_pd0_a3_status_v1",
        "run_id": contract.run_id,
        "run_signature": run_signature,
        "complete": complete,
        "completed_step": completed_step,
        "target_steps": contract.total_steps,
        "best_step": best_step,
        "best_teacher_forced_realm_npe_mean": (
            best_score if math.isfinite(best_score) else None
        ),
        "stopped_for_registered_wall_budget": stopped_for_wall_budget,
        "elapsed_seconds_this_process": time.monotonic() - started_at,
        "peak_cuda_allocated_bytes_this_process": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_cuda_reserved_bytes_this_process": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "checkpoint_paths": {"best": "best.pt", "last": "last.pt"},
        "anti_claims": [
            "validation is model-selection evidence, not untouched test evidence",
            "pMax is a cumulative released diagnostic, not instantaneous pressure",
            "released fields do not establish complete physical conservation",
            "a partial or competence-failing run is not a strong PCNO baseline",
            "one seed and one validation condition do not support a generic architecture claim",
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
            "schema": "w26_l4_planardet_pd0_a3_training_final_hash_manifest_v1",
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
        status = run_training(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
