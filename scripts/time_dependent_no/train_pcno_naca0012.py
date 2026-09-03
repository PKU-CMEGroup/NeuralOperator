#!/usr/bin/env python3
"""Run the frozen three-seed NACA0012 PCNO memory smoke or training job."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import random
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.prepare_pcno_naca0012_dataset import (
    SOURCE_FILES,
    captured_source_records,
)

SOURCE_RECORDS_AT_IMPORT = captured_source_records()

from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACABDF2Dataset,
    NACANormalization,
    VerifiedNACAGeometry,
    build_naca_bdf2_dataset,
    build_naca_pcno,
    load_naca_baseline_contract,
    load_verified_naca_geometry,
    predict_normalized_residual,
    validate_naca_model_config,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

DATASET_SCHEMA = "time_dependent_no.su2_naca0012_pcno_dataset.v1"
SOURCE_MANIFEST_SCHEMA = "time_dependent_no.naca_pcno_source_manifest.v1"
FINAL_HASH_MANIFEST_SCHEMA = "time_dependent_no.naca_pcno_final_hash_manifest.v1"
AUTHORIZATION_SCHEMA = "time_dependent_no.su2_naca0012_pcno_execution_authorization.v1"
SMOKE_SCHEMA = "time_dependent_no.su2_naca0012_pcno_resource_smoke.v1"
TRAINING_SCHEMA = "time_dependent_no.su2_naca0012_pcno_training.v1"
SEEDS = (17, 29, 43)
OPEN_ROLES = ("train", "development")
RESOURCE_CONFIGS = {(4, 1), (2, 2), (1, 4)}
EXPECTED_DATASET_FILES = {
    "dataset_manifest.json",
    "geometry.npz",
    "normalization.json",
    "source_manifest.json",
    "train_states.npy",
    "train_frame_indices.npy",
    "development_states.npy",
    "development_frame_indices.npy",
    "replay_margin_states.npy",
    "replay_margin_frame_indices.npy",
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _load_self_hashed(path: Path, schema: str) -> dict[str, Any]:
    value, _ = _load_self_hashed_with_file_sha256(path, schema)
    return value


def _load_self_hashed_with_file_sha256(
    path: Path, schema: str
) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON is absent or aliased: {path}")
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unsupported JSON schema: {path}")
    observed = value.pop("canonical_payload_sha256", None)
    expected = _canonical_sha256(value)
    if observed != expected:
        raise ValueError(f"canonical payload hash differs: {path}")
    value["canonical_payload_sha256"] = observed
    return value, sha256(payload).hexdigest()


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical hash field was supplied twice")
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _json_number(value: float) -> float | str:
    if math.isinf(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    if math.isnan(value):
        return "NaN"
    return float(value)


def _running_status_payload(
    *, seed: int, epoch: int, best_epoch: int | None, best_score: float
) -> dict[str, Any]:
    return {
        "schema": TRAINING_SCHEMA,
        "status": "running",
        "seed": seed,
        "last_completed_epoch": epoch,
        "best_epoch": best_epoch,
        "best_development_score": _json_number(best_score)
        if best_epoch is not None
        else None,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _verify_record(root: Path, record: Mapping[str, Any], expected: str) -> Path:
    if record.get("relative_path") != expected:
        raise ValueError(f"artifact record path differs for {expected}")
    candidate = root / expected
    if candidate.is_symlink():
        raise ValueError(f"artifact is aliased: {expected}")
    path = candidate.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact escapes packet: {expected}") from error
    if not path.is_file():
        raise ValueError(f"artifact is absent or aliased: {expected}")
    if path.stat().st_size != record.get("bytes"):
        raise ValueError(f"artifact byte count differs: {expected}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"artifact hash differs: {expected}")
    return path


def _read_record_bytes(root: Path, record: Mapping[str, Any], expected: str) -> bytes:
    path = _verify_record(root, record, expected)
    payload = path.read_bytes()
    if len(payload) != record.get("bytes") or sha256(payload).hexdigest() != record.get(
        "sha256"
    ):
        raise ValueError(f"artifact changed while reading: {expected}")
    return payload


def _load_npy_snapshot(
    root: Path, record: Mapping[str, Any], expected: str
) -> np.ndarray:
    payload = _read_record_bytes(root, record, expected)
    snapshot = np.array(np.load(io.BytesIO(payload), allow_pickle=False), copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _load_self_hashed_snapshot(
    root: Path, record: Mapping[str, Any], expected: str, schema: str
) -> dict[str, Any]:
    payload = _read_record_bytes(root, record, expected)
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unsupported JSON schema: {expected}")
    observed = value.pop("canonical_payload_sha256", None)
    if observed != _canonical_sha256(value):
        raise ValueError(f"canonical payload hash differs: {expected}")
    value["canonical_payload_sha256"] = observed
    return value


def _load_dataset(
    root: Path, contract: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    VerifiedNACAGeometry,
    NACANormalization,
    dict[str, tuple[np.ndarray, np.ndarray, NACABDF2Dataset]],
    str,
]:
    if root.is_symlink():
        raise ValueError("dataset directory is aliased")
    dataset_root = root.resolve()
    if not dataset_root.is_dir():
        raise ValueError("dataset directory is absent or aliased")
    observed_names = {path.name for path in dataset_root.iterdir()}
    if observed_names != EXPECTED_DATASET_FILES | {"final_hash_manifest.json"}:
        raise ValueError("dataset packet has missing or unexpected files")
    final_path = dataset_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("dataset final-hash manifest is absent or aliased")
    final_bytes = final_path.read_bytes()
    dataset_packet_sha256 = sha256(final_bytes).hexdigest()
    final = json.loads(final_bytes)
    if (
        not isinstance(final, dict)
        or final.get("schema") != FINAL_HASH_MANIFEST_SCHEMA
        or final.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or set(final.get("files", {})) != EXPECTED_DATASET_FILES
    ):
        raise ValueError("dataset final-hash manifest differs")
    for name in sorted(EXPECTED_DATASET_FILES):
        _verify_record(dataset_root, final["files"][name], name)

    manifest = _load_self_hashed_snapshot(
        dataset_root,
        final["files"]["dataset_manifest.json"],
        "dataset_manifest.json",
        DATASET_SCHEMA,
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("contract", {}).get("sha256") != BASELINE_CONTRACT_SHA256
        or manifest.get("access")
        != {
            "materialized_population_roles": ["train", "development"],
            "diagnostic_sets": ["native_replay_margin"],
            "prospective_opened": False,
            "sealed_opened": False,
        }
        or set(manifest.get("roles", {})) != set(OPEN_ROLES)
    ):
        raise ValueError("dataset access or contract binding differs")
    claims = manifest.get("claim_boundary", {})
    if (
        claims.get("train_and_development_population_materialized") is not True
        or claims.get("pcno_training_authorized") is not False
        or claims.get("prospective_opened") is not False
        or claims.get("sealed_opened") is not False
    ):
        raise ValueError("dataset claim boundary differs")

    geometry_record = final["files"]["geometry.npz"]
    if manifest.get("geometry", {}).get("file") != geometry_record:
        raise ValueError("dataset geometry record differs from final hashes")
    geometry = load_verified_naca_geometry(
        contract,
        dataset_root / "geometry.npz",
        expected_sha256=geometry_record["sha256"],
        expected_bytes=geometry_record["bytes"],
    )
    normalization_value = _load_self_hashed_snapshot(
        dataset_root,
        final["files"]["normalization.json"],
        "normalization.json",
        "time_dependent_no.naca_normalization.v1",
    )
    if (
        manifest.get("normalization", {}).get("file")
        != final["files"]["normalization.json"]
    ):
        raise ValueError("normalization record differs from final hashes")
    if (
        normalization_value.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or normalization_value.get("fit_frames")
        != contract["normalization"]["fit_frames"]
        or normalization_value.get("fit_transitions")
        != contract["normalization"]["fit_transitions"]
    ):
        raise ValueError("normalization authority binding differs")
    normalization = NACANormalization.from_mapping(normalization_value)

    loaded: dict[str, tuple[np.ndarray, np.ndarray, NACABDF2Dataset]] = {}
    for role in OPEN_ROLES:
        role_record = manifest["roles"][role]
        states_name = f"{role}_states.npy"
        indices_name = f"{role}_frame_indices.npy"
        if (
            role_record["states"] != final["files"][states_name]
            or role_record["frame_indices"] != final["files"][indices_name]
        ):
            raise ValueError(f"{role} records differ from final hashes")
        states = _load_npy_snapshot(
            dataset_root, final["files"][states_name], states_name
        )
        frame_indices = _load_npy_snapshot(
            dataset_root, final["files"][indices_name], indices_name
        )
        if frame_indices.dtype != np.int64:
            raise ValueError(f"{role} frame indices must be int64")
        center_first, center_last = role_record[
            "dense_transition_center_indices_inclusive"
        ]
        centers = tuple(range(center_first, center_last + 1))
        if len(centers) != 238 or role_record.get("dense_transition_count") != 238:
            raise ValueError(f"{role} transition cardinality differs")
        dataset = build_naca_bdf2_dataset(
            contract,
            geometry,
            role,
            states,
            frame_indices,
            normalization,
        )
        loaded[role] = (states, frame_indices, dataset)

    source = _load_self_hashed_snapshot(
        dataset_root,
        final["files"]["source_manifest.json"],
        "source_manifest.json",
        SOURCE_MANIFEST_SCHEMA,
    )
    if (
        manifest.get("source_manifest", {}).get("file")
        != final["files"]["source_manifest.json"]
    ):
        raise ValueError("source-manifest record differs from final hashes")
    diagnostic = manifest.get("diagnostic_sets", {}).get("native_replay_margin", {})
    if (
        diagnostic.get("states") != final["files"]["replay_margin_states.npy"]
        or diagnostic.get("frame_indices")
        != final["files"]["replay_margin_frame_indices.npy"]
    ):
        raise ValueError("diagnostic records differ from final hashes")
    if source.get("contract_sha256") != BASELINE_CONTRACT_SHA256 or source.get(
        "source_set_sha256"
    ) != manifest.get("source_manifest", {}).get("source_set_sha256"):
        raise ValueError("dataset source binding differs")
    if set(source.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("dataset source manifest lacks the exact execution closure")
    if source.get("files") != SOURCE_RECORDS_AT_IMPORT:
        raise ValueError("dataset source differs from import-time execution bytes")
    if source.get("source_set_sha256") != _canonical_sha256(SOURCE_RECORDS_AT_IMPORT):
        raise ValueError("dataset source-set differs from import-time execution bytes")
    live_records: dict[str, Any] = {}
    for relative, record in source.get("files", {}).items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound source is absent or aliased: {relative}")
        payload = path.read_bytes()
        live_records[relative] = {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
        if live_records[relative] != record:
            raise ValueError(f"live source differs from dataset binding: {relative}")
    if _canonical_sha256(live_records) != source["source_set_sha256"]:
        raise ValueError("live source-set digest differs")
    for name in sorted(EXPECTED_DATASET_FILES):
        _verify_record(dataset_root, final["files"][name], name)
    if sha256_file(final_path) != dataset_packet_sha256:
        raise ValueError("dataset final-hash manifest changed during loading")
    return manifest, geometry, normalization, loaded, dataset_packet_sha256


def _runtime(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if cuda else None,
        "device_capability": list(torch.cuda.get_device_capability(device))
        if cuda
        else None,
    }


def _device_binding(device: torch.device) -> dict[str, Any]:
    """Identify the execution class that a resource smoke actually exercised."""

    if device.type != "cuda":
        return {
            "device_type": device.type,
            "cuda_device_index": None,
            "cuda_device_name": None,
            "cuda_device_capability": None,
            "cuda_runtime_version": None,
        }
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "device_type": "cuda",
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
        "cuda_runtime_version": torch.version.cuda,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-5,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def _geometry_slice(
    value: Mapping[str, torch.Tensor], size: int
) -> dict[str, torch.Tensor]:
    return {name: tensor[:size] for name, tensor in value.items()}


def _fourier_slice(
    value: tuple[torch.Tensor, ...], size: int
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor[:size] for tensor in value)


def _step_effective_batch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    microbatch_size: int,
) -> tuple[float, float]:
    group_size = int(previous.shape[0])
    if group_size not in (2, 4):
        raise ValueError("effective training group must contain two or four samples")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for offset in range(0, group_size, microbatch_size):
        stop = min(offset + microbatch_size, group_size)
        size = stop - offset
        previous_chunk = previous[offset:stop]
        current_chunk = current[offset:stop]
        target_chunk = target[offset:stop]
        prediction = predict_normalized_residual(
            model,
            previous_chunk,
            current_chunk,
            _geometry_slice(geometry_batch, size),
            normalization,
            fourier_tensors=_fourier_slice(fourier_tensors, size),
        )
        loss = torch.mean(torch.square(prediction - target_chunk))
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became nonfinite")
        weight = size / group_size
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
            raise FloatingPointError("training gradient became nonfinite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("gradient norm became nonfinite")
    optimizer.step()
    for parameter in model.parameters():
        if not torch.all(torch.isfinite(parameter)):
            raise FloatingPointError("model parameter became nonfinite")
    return weighted_loss, float(gradient_norm.detach().cpu())


def _development_score(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    device: torch.device,
    microbatch_size: int,
) -> tuple[float, list[float]]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for previous, current, target, _ in loader:
            group_size = int(previous.shape[0])
            for offset in range(0, group_size, microbatch_size):
                stop = min(offset + microbatch_size, group_size)
                size = stop - offset
                prediction = predict_normalized_residual(
                    model,
                    previous[offset:stop].to(device),
                    current[offset:stop].to(device),
                    _geometry_slice(geometry_batch, size),
                    normalization,
                    fourier_tensors=_fourier_slice(fourier_tensors, size),
                )
                error = prediction - target[offset:stop].to(device)
                per_sample = torch.sqrt(torch.mean(torch.square(error), dim=(1, 2)))
                per_sample = torch.where(
                    torch.isfinite(per_sample),
                    per_sample,
                    torch.full_like(per_sample, math.inf),
                )
                values.extend(float(item) for item in per_sample.cpu())
    if len(values) != 238:
        raise ValueError("development checkpoint score lacks 238 transitions")
    return float(np.median(np.asarray(values, dtype=np.float64))), values


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    seed: int,
    contract: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    presentation_generator: torch.Generator,
    best_score: float,
    best_epoch: int | None,
) -> dict[str, Any]:
    model_config = model.model_config()  # type: ignore[attr-defined]
    validate_naca_model_config(model_config, contract, geometry)
    return {
        "schema": "time_dependent_no.su2_naca0012_pcno_checkpoint.v1",
        "contract_sha256": BASELINE_CONTRACT_SHA256,
        "dataset_manifest_payload_sha256": dataset_manifest["canonical_payload_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
        "source_set_sha256": dataset_manifest["source_manifest"]["source_set_sha256"],
        "seed": seed,
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "presentation_generator": presentation_generator.get_state(),
        },
        "best_development_score": best_score,
        "best_epoch": best_epoch,
    }


def _selected_checkpoint_payload(
    checkpoint: Mapping[str, Any], development_score: float
) -> dict[str, Any]:
    """Strip optimizer state from the deployed, development-selected checkpoint."""

    return {
        "schema": checkpoint["schema"],
        "contract_sha256": checkpoint["contract_sha256"],
        "dataset_manifest_payload_sha256": checkpoint[
            "dataset_manifest_payload_sha256"
        ],
        "dataset_final_hash_manifest_sha256": checkpoint[
            "dataset_final_hash_manifest_sha256"
        ],
        "source_set_sha256": checkpoint["source_set_sha256"],
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "model_config": checkpoint["model_config"],
        "model_state_dict": checkpoint["model_state_dict"],
        "best_development_score": development_score,
        "best_epoch": checkpoint["epoch"],
        "model_only": True,
    }


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _validate_resource_config(arguments: argparse.Namespace) -> None:
    pair = (arguments.microbatch_size, arguments.gradient_accumulation_steps)
    if pair not in RESOURCE_CONFIGS:
        raise ValueError("resource configuration must be exactly 4x1, 2x2, or 1x4")


def _run_smoke(
    arguments: argparse.Namespace,
    contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    device: torch.device,
) -> dict[str, Any]:
    if (
        arguments.authorization is not None
        or arguments.resource_smoke_receipt is not None
    ):
        raise ValueError("resource smoke does not accept production authorization")
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("resource-smoke output or its parent is aliased")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"resource-smoke output already exists: {output}")
    output.mkdir(parents=True)
    _seed_everything(arguments.seed)
    model = build_naca_pcno(contract, geometry).to(device)
    optimizer = _optimizer(model)
    geometry_batch = geometry.expand(arguments.microbatch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(arguments.seed),
        num_workers=0,
        drop_last=False,
    )
    previous, current, target, _ = next(iter(loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss, gradient_norm = _step_effective_batch(
        model=model,
        optimizer=optimizer,
        previous=previous.to(device),
        current=current.to(device),
        target=target.to(device),
        geometry_batch=geometry_batch,
        fourier_tensors=fourier,
        normalization=normalization,
        microbatch_size=arguments.microbatch_size,
    )
    receipt = _self_hashed(
        {
            "schema": SMOKE_SCHEMA,
            "status": "succeeded",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": dataset_manifest["source_manifest"][
                "source_set_sha256"
            ],
            "seed": arguments.seed,
            "effective_batch_size": 4,
            "microbatch_size": arguments.microbatch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "full_resolution_num_nodes": geometry.num_nodes,
            "forward_backward_and_adamw_step_completed": True,
            "loss": loss,
            "gradient_norm_before_clip": gradient_norm,
            "production_device_binding": _device_binding(device),
            "runtime": _runtime(device),
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "scientific_hyperparameters_changed": False,
            "checkpoint_written": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    _write_json(output / "resource_smoke.json", receipt)
    return receipt


def _validate_production_authorization(
    arguments: argparse.Namespace,
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if arguments.authorization is None or arguments.resource_smoke_receipt is None:
        raise PermissionError(
            "production training requires authorization and a resource-smoke receipt"
        )
    if arguments.resource_smoke_receipt.is_symlink():
        raise PermissionError("resource-smoke receipt is aliased")
    smoke_path = arguments.resource_smoke_receipt.resolve()
    smoke, smoke_file_sha256 = _load_self_hashed_with_file_sha256(
        smoke_path, SMOKE_SCHEMA
    )
    pair = (smoke.get("microbatch_size"), smoke.get("gradient_accumulation_steps"))
    device_binding = _device_binding(device)
    smoke_seed = smoke.get("seed")
    smoke_loss = smoke.get("loss")
    smoke_gradient_norm = smoke.get("gradient_norm_before_clip")
    if (
        smoke.get("status") != "succeeded"
        or smoke.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or smoke.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or smoke.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or smoke.get("source_set_sha256")
        != dataset_manifest["source_manifest"]["source_set_sha256"]
        or smoke.get("forward_backward_and_adamw_step_completed") is not True
        or smoke.get("effective_batch_size") != 4
        or smoke.get("full_resolution_num_nodes")
        != dataset_manifest.get("geometry", {}).get("num_nodes")
        or isinstance(smoke_seed, bool)
        or smoke_seed not in SEEDS
        or isinstance(smoke_loss, bool)
        or not isinstance(smoke_loss, (int, float))
        or not math.isfinite(float(smoke_loss))
        or isinstance(smoke_gradient_norm, bool)
        or not isinstance(smoke_gradient_norm, (int, float))
        or not math.isfinite(float(smoke_gradient_norm))
        or smoke.get("scientific_hyperparameters_changed") is not False
        or smoke.get("checkpoint_written") is not False
        or smoke.get("prospective_opened") is not False
        or smoke.get("sealed_opened") is not False
        or pair != (arguments.microbatch_size, arguments.gradient_accumulation_steps)
        or smoke.get("production_device_binding") != device_binding
    ):
        raise PermissionError("resource-smoke receipt does not authorize this job")

    if arguments.authorization.is_symlink():
        raise PermissionError("production authorization is aliased")
    authorization, _ = _load_self_hashed_with_file_sha256(
        arguments.authorization.resolve(), AUTHORIZATION_SCHEMA
    )
    expected_actions = ["train_pcno_naca0012"]
    if (
        authorization.get("status") != "authorized"
        or authorization.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or authorization.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or authorization.get("dataset_final_hash_manifest_sha256")
        != dataset_packet_sha256
        or authorization.get("source_set_sha256")
        != dataset_manifest["source_manifest"]["source_set_sha256"]
        or authorization.get("resource_smoke_payload_sha256")
        != smoke["canonical_payload_sha256"]
        or authorization.get("resource_smoke_file_sha256") != smoke_file_sha256
        or authorization.get("production_device_binding") != device_binding
        or authorization.get("allowed_actions") != expected_actions
        or authorization.get("authorized_seeds") != list(SEEDS)
        or authorization.get("pcno_code_audited") is not True
        or authorization.get("pcno_training_authorized") is not True
        or authorization.get("prospective_opened") is not False
        or authorization.get("sealed_opened") is not False
        or not isinstance(authorization.get("authorized_by"), str)
        or not authorization["authorized_by"].strip()
    ):
        raise PermissionError("production authorization is absent or mismatched")
    return smoke, authorization


def _new_training_packet(
    output: Path,
    arguments: argparse.Namespace,
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    smoke: Mapping[str, Any],
    authorization: Mapping[str, Any],
    device: torch.device,
) -> None:
    output.mkdir(parents=True)
    config = _self_hashed(
        {
            "schema": TRAINING_SCHEMA,
            "status": "frozen_configuration",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "seed": arguments.seed,
            "epochs": 100,
            "effective_batch_size": 4,
            "microbatch_size": arguments.microbatch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "production_device_binding": _device_binding(device),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1.0e-3,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 1.0e-5,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
            },
            "gradient_clip_norm": 1.0,
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": 100,
                "eta_min": 1.0e-5,
                "step": "after_each_completed_epoch",
            },
            "checkpoint_epochs": list(range(5, 101, 5)),
            "selection": "strict_decrease_of_development_median_uniform_component_balanced_normalized_residual_rmse",
            "rollout_used_for_selection": False,
            "amp": False,
            "tf32": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    input_manifest = _self_hashed(
        {
            "schema": "time_dependent_no.su2_naca0012_pcno_training_inputs.v1",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": dataset_manifest["source_manifest"][
                "source_set_sha256"
            ],
            "resource_smoke_payload_sha256": smoke["canonical_payload_sha256"],
            "authorization_payload_sha256": authorization["canonical_payload_sha256"],
            "production_device_binding": _device_binding(device),
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    _write_json(output / "config.json", config)
    _write_json(output / "input_manifest.json", input_manifest)
    _write_json(
        output / "runtime_manifest.json",
        _self_hashed(
            {
                "schema": "time_dependent_no.su2_naca0012_pcno_runtime.v1",
                "runtime": _runtime(device),
            }
        ),
    )
    _write_json(
        output / "status.json",
        {"schema": TRAINING_SCHEMA, "status": "running", "seed": arguments.seed},
    )


def _write_final_hash_manifest(output: Path) -> None:
    files: dict[str, Any] = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "final_hash_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("training packet contains a nonregular entry")
        files[path.name] = {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(
        output / "final_hash_manifest.json",
        {
            "schema": FINAL_HASH_MANIFEST_SCHEMA,
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "files": files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
        },
    )


def _run_training(
    arguments: argparse.Namespace,
    contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    development_dataset: NACABDF2Dataset,
    device: torch.device,
    smoke: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("training output or its parent is aliased")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"training output already exists: {output}")
    _new_training_packet(
        output,
        arguments,
        dataset_manifest,
        dataset_packet_sha256,
        smoke,
        authorization,
        device,
    )

    _seed_everything(arguments.seed)
    model = build_naca_pcno(contract, geometry).to(device)
    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1.0e-5
    )
    best_score = math.inf
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []

    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    geometry_batch = geometry.expand(arguments.microbatch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)

    try:
        for epoch in range(1, 101):
            model.train()
            loss_sum = 0.0
            gradient_norms: list[float] = []
            presentations = 0
            optimizer_steps = 0
            for previous, current, target, _ in train_loader:
                group_size = int(previous.shape[0])
                loss, gradient_norm = _step_effective_batch(
                    model=model,
                    optimizer=optimizer,
                    previous=previous.to(device),
                    current=current.to(device),
                    target=target.to(device),
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier,
                    normalization=normalization,
                    microbatch_size=arguments.microbatch_size,
                )
                loss_sum += loss * group_size
                gradient_norms.append(gradient_norm)
                presentations += group_size
                optimizer_steps += 1
            if presentations != 238 or optimizer_steps != 60:
                raise ValueError("epoch presentation or optimizer-step count differs")

            development_score: float | None = None
            selected = False
            if epoch % 5 == 0:
                development_score, _ = _development_score(
                    model=model,
                    loader=development_loader,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier,
                    normalization=normalization,
                    device=device,
                    microbatch_size=arguments.microbatch_size,
                )
                if epoch == 5 or development_score < best_score:
                    best_score = development_score
                    best_epoch = epoch
                    selected = True
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_sample_mean_loss": loss_sum / presentations,
                    "maximum_gradient_norm_before_clip": float(max(gradient_norms)),
                    "presentations": presentations,
                    "optimizer_steps": optimizer_steps,
                    "learning_rate_after_scheduler_step": float(
                        scheduler.get_last_lr()[0]
                    ),
                    "development_checkpoint_score": _json_number(development_score)
                    if development_score is not None
                    else None,
                    "selected_as_best": selected,
                }
            )
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                seed=arguments.seed,
                contract=contract,
                geometry=geometry,
                dataset_manifest=dataset_manifest,
                dataset_packet_sha256=dataset_packet_sha256,
                presentation_generator=generator,
                best_score=best_score,
                best_epoch=best_epoch,
            )
            _atomic_torch_save(output / "last.pt", checkpoint)
            if epoch % 5 == 0 and selected:
                _atomic_torch_save(
                    output / "best.pt",
                    _selected_checkpoint_payload(checkpoint, best_score),
                )
            _write_json(
                output / "history.json",
                {
                    "schema": "time_dependent_no.su2_naca0012_pcno_history.v1",
                    "seed": arguments.seed,
                    "epochs": history,
                },
            )
            _write_json(
                output / "status.json",
                _running_status_payload(
                    seed=arguments.seed,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=best_score,
                ),
            )
    except Exception as error:
        _write_json(
            output / "status.json",
            {
                "schema": TRAINING_SCHEMA,
                "status": "failed_incomplete_not_scientific_evidence",
                "seed": arguments.seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "prospective_opened": False,
                "sealed_opened": False,
            },
        )
        raise

    if best_epoch not in range(5, 101, 5) or not (output / "best.pt").is_file():
        raise ValueError("training completed without a selected checkpoint")
    summary = _self_hashed(
        {
            "schema": TRAINING_SCHEMA,
            "status": "complete",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": dataset_manifest["source_manifest"][
                "source_set_sha256"
            ],
            "seed": arguments.seed,
            "completed_epochs": 100,
            "best_epoch": best_epoch,
            "best_development_score": _json_number(best_score),
            "checkpoint_selection_used_rollout": False,
            "prospective_opened": False,
            "sealed_opened": False,
            "scientific_claim_made": False,
        }
    )
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "status.json",
        {
            "schema": TRAINING_SCHEMA,
            "status": "complete",
            "seed": arguments.seed,
            "last_completed_epoch": 100,
            "best_epoch": best_epoch,
            "best_development_score": _json_number(best_score),
        },
    )
    _write_final_hash_manifest(output)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resource-smoke", action="store_true")
    parser.add_argument("--resource-smoke-receipt", type=Path)
    parser.add_argument("--authorization", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _validate_resource_config(arguments)
        contract = load_naca_baseline_contract(arguments.contract)
        (
            dataset_manifest,
            geometry,
            normalization,
            roles,
            dataset_packet_sha256,
        ) = _load_dataset(arguments.dataset_dir, contract)
        device = torch.device(arguments.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if arguments.resource_smoke:
            result = _run_smoke(
                arguments,
                contract,
                dataset_manifest,
                dataset_packet_sha256,
                geometry,
                normalization,
                roles["train"][2],
                device,
            )
        else:
            smoke, authorization = _validate_production_authorization(
                arguments, dataset_manifest, dataset_packet_sha256, device
            )
            result = _run_training(
                arguments,
                contract,
                dataset_manifest,
                dataset_packet_sha256,
                geometry,
                normalization,
                roles["train"][2],
                roles["development"][2],
                device,
                smoke,
                authorization,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 PCNO training failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
