"""Train the frozen B3B4 NACA0012 corrective-successor PCNO arms.

This entry point deliberately reuses the verified R0 dataset/model loader while
adding a second, immutable successor contract and a train-only calibration
packet.  It never opens prospective or sealed population files.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

_ENTRY_PATH = Path(__file__)
if _ENTRY_PATH.is_symlink() or not _ENTRY_PATH.is_file():
    raise RuntimeError("successor trainer source is absent or aliased")
REPO_ROOT = _ENTRY_PATH.resolve().parents[2]
SUCCESSOR_SOURCE = "scripts/time_dependent_no/train_pcno_naca0012_successor.py"
SUCCESSOR_CORE_SOURCE = "utility/time_dependent_no/pcno_naca0012_successor.py"
if _ENTRY_PATH.resolve() != (REPO_ROOT / SUCCESSOR_SOURCE).resolve():
    raise RuntimeError("successor trainer executed from an unexpected repository path")


def _stable_source_record(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is absent or aliased")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != len(payload)
        or after.st_size != len(payload)
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while it was captured")
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


_BOOTSTRAP_SOURCE_RECORDS = {
    relative: _stable_source_record(REPO_ROOT / relative, relative)
    for relative in (SUCCESSOR_CORE_SOURCE, SUCCESSOR_SOURCE)
}

import numpy as np
import torch
from torch.utils.data import DataLoader

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import train_pcno_naca0012 as r0_trainer
from utility.time_dependent_no.pcno_naca0012 import (
    NACABDF2Dataset,
    NACANormalization,
    VerifiedNACAGeometry,
    build_naca_pcno,
    load_naca_baseline_contract,
    predict_normalized_residual,
    validate_naca_model_config,
)
from utility.time_dependent_no.pcno_naca0012_successor import (
    HistoryNoise,
    RecoveryCalibration,
    TrainPathProjector,
    make_recovery_presentation,
    sample_iid_history_noise,
    sample_structured_history_noise,
    validate_successor_math_contract,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

EXPERIMENT_ID = "B3B4_NACA_CM_20260901A"
SUCCESSOR_CONTRACT_SCHEMA = "time_dependent_no.naca_corrective_successor_contract.v1"
SUCCESSOR_CONTRACT_FILE_SHA256 = (
    "b0f373d42ca98c0761a61f17310233d22461da1add28b8c8d20359ef7e1d6715"
)
R0_CONTRACT_SHA256 = "94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87"
PREREGISTRATION_SHA256 = (
    "650b71d0c3c1979e9e54d6be6a06c3869dc507a9010710dd31eece5b33e1f8cb"
)
DATASET_FINAL_SHA256 = (
    "1cb5fd2d751bdf1ca27cde5e1a1457973a525c19a98dbadd7cc3738c3efe2e4b"
)
TRAINING_ARMS = (
    "CLEAN",
    "IID_RECOVERY",
    "ERROR_SUBSPACE_RECOVERY",
    "DETACHED_PUSHFORWARD",
)
SEEDS = (17, 29, 43)
EXPECTED_PACKET_SCHEMAS = {
    "calibration": "time_dependent_no.naca_corrective_successor_calibration.v1",
    "training": "time_dependent_no.naca_corrective_successor_training.v1",
    "training_config": "time_dependent_no.naca_corrective_successor_training_config.v1",
    "training_inputs": "time_dependent_no.naca_corrective_successor_training_inputs.v1",
    "evaluation": "time_dependent_no.naca_corrective_successor_evaluation.v1",
    "evaluation_inputs": "time_dependent_no.naca_corrective_successor_evaluation_inputs.v1",
    "final_hash_manifest": "time_dependent_no.naca_corrective_successor_final_hash_manifest.v1",
    "resource_smoke": "time_dependent_no.naca_corrective_successor_resource_smoke.v1",
    "execution_authorization": "time_dependent_no.naca_corrective_successor_execution_authorization.v1",
    "prospective_freeze": "time_dependent_no.naca_corrective_successor_prospective_freeze.v1",
    "prospective_authorization": "time_dependent_no.naca_corrective_successor_prospective_authorization.v1",
}
CALIBRATION_FILES = {"calibration.json", "calibration_arrays.npz"}
TRAINING_FILES = {
    "best.pt",
    "config.json",
    "history.json",
    "input_manifest.json",
    "last.pt",
    "runtime_manifest.json",
    "status.json",
    "summary.json",
}
_canonical_sha256 = r0_trainer._canonical_sha256
_json_number = r0_trainer._json_number
_self_hashed = r0_trainer._self_hashed
_write_json = r0_trainer._write_json
_atomic_torch_save = r0_trainer._atomic_torch_save


def _capture_source_records() -> dict[str, dict[str, Any]]:
    records = {
        name: dict(record)
        for name, record in r0_trainer.SOURCE_RECORDS_AT_IMPORT.items()
    }
    for relative in (SUCCESSOR_CORE_SOURCE, SUCCESSOR_SOURCE):
        observed = _stable_source_record(REPO_ROOT / relative, relative)
        if observed != _BOOTSTRAP_SOURCE_RECORDS[relative]:
            raise RuntimeError(f"successor source changed during import: {relative}")
        records[relative] = dict(observed)
    return records


SOURCE_RECORDS_AT_IMPORT = _capture_source_records()
SOURCE_SET_SHA256 = _canonical_sha256(SOURCE_RECORDS_AT_IMPORT)


def _verify_live_sources() -> None:
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected in SOURCE_RECORDS_AT_IMPORT.items():
        observed[relative] = _stable_source_record(
            REPO_ROOT / relative, f"bound source {relative}"
        )
        if observed[relative] != expected:
            raise ValueError(f"bound source changed during execution: {relative}")
    if _canonical_sha256(observed) != SOURCE_SET_SHA256:
        raise ValueError("successor source-set hash differs")


def _load_successor_contract(
    path: Path, preregistration: Path, r0_contract_path: Path
) -> dict[str, Any]:
    for candidate, label in (
        (path, "successor contract"),
        (preregistration, "successor preregistration"),
        (r0_contract_path, "inherited R0 contract"),
    ):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"{label} is absent or aliased")
    payload = path.read_bytes()
    file_sha256 = sha256(payload).hexdigest()
    if file_sha256 != SUCCESSOR_CONTRACT_FILE_SHA256:
        raise ValueError("successor contract bytes differ from the frozen contract")
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != SUCCESSOR_CONTRACT_SCHEMA:
        raise ValueError("successor contract schema differs")
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("packet_schemas") != EXPECTED_PACKET_SCHEMAS
        or value.get("inherited_r0_contract_sha256") != R0_CONTRACT_SHA256
        or value.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or value.get("dataset", {}).get("final_hash_manifest_sha256")
        != DATASET_FINAL_SHA256
        or value.get("protection")
        != {
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
            "online_defect_trigger": False,
        }
    ):
        raise ValueError("successor contract provenance or protection differs")
    if set(value.get("parent_r0", {}).get("checkpoints", {})) != {
        "17",
        "29",
        "43",
    } or any(arm not in value.get("arms", {}) for arm in TRAINING_ARMS):
        raise ValueError("successor parent or arm inventory differs")
    if sha256_file(preregistration) != PREREGISTRATION_SHA256:
        raise ValueError("successor preregistration bytes differ")
    if sha256_file(r0_contract_path) != R0_CONTRACT_SHA256:
        raise ValueError("inherited R0 contract bytes differ")
    if r0_trainer.BASELINE_CONTRACT_SHA256 != R0_CONTRACT_SHA256:
        raise ValueError("runtime inherited R0 contract constant differs")
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_calibration(
    root: Path,
    successor: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    normalization: NACANormalization,
    num_nodes: int,
) -> tuple[
    dict[str, Any],
    RecoveryCalibration,
    str,
    dict[str, Any],
]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("calibration packet is absent or aliased")
    calibration_root = root.resolve()
    final_path = calibration_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("calibration final-hash manifest is absent or aliased")
    final_bytes = final_path.read_bytes()
    final_sha256 = sha256(final_bytes).hexdigest()
    final = json.loads(final_bytes)
    if (
        not isinstance(final, dict)
        or final.get("schema") != EXPECTED_PACKET_SCHEMAS["final_hash_manifest"]
        or final.get("experiment_id") != EXPERIMENT_ID
        or final.get("packet_role") != "calibration"
        or final.get("successor_contract_sha256") != SUCCESSOR_CONTRACT_FILE_SHA256
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or set(final.get("files", {})) != CALIBRATION_FILES
        or {path.name for path in calibration_root.iterdir()}
        != CALIBRATION_FILES | {"final_hash_manifest.json"}
    ):
        raise ValueError("calibration final-hash manifest differs")
    for name in sorted(CALIBRATION_FILES):
        r0_trainer._verify_record(calibration_root, final["files"][name], name)
    metadata = r0_trainer._load_self_hashed_snapshot(
        calibration_root,
        final["files"]["calibration.json"],
        "calibration.json",
        EXPECTED_PACKET_SCHEMAS["calibration"],
    )
    if (
        metadata.get("status") != "complete"
        or metadata.get("experiment_id") != EXPERIMENT_ID
        or metadata.get("successor_contract_sha256") != SUCCESSOR_CONTRACT_FILE_SHA256
        or metadata.get("inherited_r0_contract_sha256") != R0_CONTRACT_SHA256
        or metadata.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or metadata.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or metadata.get("parent_r0_evaluation_final_hash_manifest_sha256")
        != successor["parent_r0"]["evaluation_final_hash_manifest_sha256"]
        or metadata.get("fit_role") != "train_only"
        or not isinstance(metadata.get("runtime"), dict)
        or metadata.get("pca")
        != {
            "recovery_algorithm": "exact centered covariance-Gram eigendecomposition",
            "path_algorithm": "exact torch.linalg.svd",
        }
        or metadata.get("access")
        != {
            "train_opened": True,
            "development_opened": True,
            "prospective_opened": False,
            "sealed_opened": False,
        }
        or metadata.get("claim_boundary")
        != {
            "trusted_displaced_state_response_measured": False,
            "manifold_drift_established": False,
            "successor_rollout_opened": False,
        }
    ):
        raise ValueError("calibration authority or protection binding differs")
    parent_records = metadata.get("parent_checkpoints")
    if not isinstance(parent_records, dict) or set(parent_records) != {
        "17",
        "29",
        "43",
    }:
        raise ValueError("calibration parent-checkpoint inventory differs")
    for seed in SEEDS:
        record = parent_records[str(seed)]
        expected_parent = successor["parent_r0"]["checkpoints"][str(seed)]
        if (
            not isinstance(record, dict)
            or record.get("final_hash_manifest_sha256")
            != expected_parent["final_hash_manifest_sha256"]
            or record.get("checkpoint_sha256") != expected_parent["checkpoint_sha256"]
            or not isinstance(record.get("packet_name"), str)
            or not record["packet_name"].strip()
            or isinstance(record.get("best_epoch"), bool)
            or record.get("best_epoch") not in range(5, 101, 5)
            or isinstance(record.get("best_development_score"), bool)
            or not isinstance(record.get("best_development_score"), (int, float))
            or not math.isfinite(float(record["best_development_score"]))
        ):
            raise ValueError(f"calibration parent checkpoint differs: {seed}")
    diagnostics = metadata.get("teacher_forced_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"17", "29", "43"}:
        raise ValueError("calibration teacher-forced diagnostics differ")
    arrays_record = metadata.get("arrays")
    if (
        not isinstance(arrays_record, dict)
        or arrays_record.get("file") != final["files"]["calibration_arrays.npz"]
        or not isinstance(arrays_record.get("members"), dict)
    ):
        raise ValueError("calibration array binding differs")
    source = metadata.get("source")
    calibration_sources = (
        *r0_trainer.SOURCE_FILES,
        "utility/time_dependent_no/pcno_ripple_diagnostics.py",
        "utility/time_dependent_no/pcno_naca0012_successor.py",
        "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md",
        "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json",
        "scripts/time_dependent_no/calibrate_pcno_naca0012_successor.py",
    )
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("files"), dict)
        or set(source["files"]) != set(calibration_sources)
        or source.get("source_set_sha256") != _canonical_sha256(source["files"])
    ):
        raise ValueError("calibration source closure differs")
    for relative in calibration_sources:
        record = source["files"][relative]
        path = REPO_ROOT / relative
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"calibration source differs from live source: {relative}")
    arrays_bytes = r0_trainer._read_record_bytes(
        calibration_root,
        final["files"]["calibration_arrays.npz"],
        "calibration_arrays.npz",
    )
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    records = arrays_record["members"]
    if not isinstance(records, dict) or set(records) != set(arrays):
        raise ValueError("calibration array records differ")
    for name, value in arrays.items():
        expected_record = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "array_sha256": _array_sha256(value),
        }
        if not isinstance(records[name], dict) or any(
            records[name].get(key) != expected
            for key, expected in expected_record.items()
        ):
            raise ValueError(f"calibration array content record differs: {name}")
    recovery_mapping = {
        name.removeprefix("recovery__"): value
        for name, value in arrays.items()
        if name.startswith("recovery__")
    }
    path_mapping = {
        name.removeprefix("path__"): value
        for name, value in arrays.items()
        if name.startswith("path__")
    }
    if len(recovery_mapping) + len(path_mapping) != len(arrays):
        raise ValueError("calibration NPZ contains an unregistered prefix")
    recovery = RecoveryCalibration.from_mapping(recovery_mapping)
    path_projector = TrainPathProjector.from_mapping(path_mapping, normalization)
    if recovery.num_nodes != num_nodes or path_projector.num_nodes != num_nodes:
        raise ValueError("calibration mesh differs from the verified NACA mesh")
    summary = metadata.get("calibration")
    expected_summary = {
        "iid_field_std": recovery.iid_field_rms.tolist(),
        "iid_expected_history_pair_squared_norm": recovery.iid_expected_history_pair_energy,
        "error_pair_count": recovery.pair_sample_count,
        "error_pair_dimension": 2 * recovery.num_nodes * 5,
        "error_pair_rank16_capture_fraction": recovery.structured_captured_variance,
        "error_pair_energy_rescale": recovery.structured_energy_rescale,
        "error_pair_matched_expected_squared_norm": float(
            torch.sum(torch.square(recovery.structured_coefficient_std)).item()
        ),
        "structured_sampling_mean": "zero",
        "path_state_count": 238,
        "path_state_dimension": path_projector.num_nodes * 5,
        "path_rank": path_projector.rank,
        "path_rank_without_cap": path_projector.variance_rank_without_cap,
        "path_rank_cap_active": path_projector.rank_cap_active,
        "path_retained_variance_fraction": path_projector.captured_variance,
    }
    if not isinstance(summary, dict) or any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise ValueError("calibration scientific summary differs from core mappings")
    for name in sorted(CALIBRATION_FILES):
        r0_trainer._verify_record(calibration_root, final["files"][name], name)
    if sha256_file(final_path) != final_sha256:
        raise ValueError("calibration packet changed while loading")
    return metadata, recovery, final_sha256, dict(final["files"])


def _reverify_calibration(
    root: Path, files: Mapping[str, Mapping[str, Any]], final_sha256: str
) -> None:
    calibration_root = root.resolve()
    if (
        root.is_symlink()
        or not calibration_root.is_dir()
        or {path.name for path in calibration_root.iterdir()}
        != set(files) | {"final_hash_manifest.json"}
    ):
        raise ValueError("verified calibration packet changed")
    for name, record in files.items():
        r0_trainer._verify_record(calibration_root, record, name)
    if sha256_file(calibration_root / "final_hash_manifest.json") != final_sha256:
        raise ValueError("verified calibration final hash changed")


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    return _canonical_sha256(
        {name: _tensor_sha256(value) for name, value in model.state_dict().items()}
    )


def _generator_state_sha256(generator: torch.Generator) -> str:
    return _tensor_sha256(generator.get_state())


class _EventHasher:
    def __init__(self) -> None:
        self._digest = sha256()
        self.count = 0

    def add(self, value: Mapping[str, Any]) -> None:
        rendered = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._digest.update(rendered)
        self._digest.update(b"\n")
        self.count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _presentation_orders(seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    initial = _generator_state_sha256(generator)
    orders = torch.stack([torch.randperm(238, generator=generator) for _ in range(100)])
    final = _generator_state_sha256(generator)
    return orders, {
        "algorithm": "torch.randperm_explicit_epoch_order_v1",
        "seed": seed,
        "shape": [100, 238],
        "dtype": str(orders.dtype),
        "initial_generator_state_sha256": initial,
        "final_generator_state_sha256": final,
        "presentation_schedule_sha256": _tensor_sha256(orders),
    }


def _intervention_generator(seed: int) -> tuple[torch.Generator, dict[str, Any]]:
    intervention_seed = 2_026_090_100_003 + seed * 1_000_003
    generator = torch.Generator().manual_seed(intervention_seed)
    return generator, {
        "algorithm": "torch_cpu_generator_explicit_state_v1",
        "seed_rule": "2026090100003 + successor_seed * 1000003",
        "derived_seed": intervention_seed,
        "initial_generator_state_sha256": _generator_state_sha256(generator),
    }


def _gather_batch(
    dataset: NACABDF2Dataset, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    items = [dataset[int(index)] for index in indices.tolist()]
    previous = torch.stack([item[0] for item in items])
    current = torch.stack([item[1] for item in items])
    target = torch.stack([item[2] for item in items])
    centers = torch.tensor([item[3] for item in items], dtype=torch.int64)
    return previous, current, target, centers


def _preceding_states(dataset: NACABDF2Dataset, centers: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for center in centers.tolist():
        if center < 957:
            values.append(torch.zeros_like(dataset[0][0]))
            continue
        position = dataset.frame_indices.index(int(center) - 2)
        values.append(
            torch.from_numpy(
                np.array(dataset.states[position], dtype=np.float32, copy=True)
            )
        )
    return torch.stack(values)


def _noise_pair(
    *,
    arm: str,
    reference: torch.Tensor,
    calibration: RecoveryCalibration,
    generator: torch.Generator,
) -> HistoryNoise | None:
    if arm == "IID_RECOVERY":
        return sample_iid_history_noise(
            reference,
            calibration.iid_field_rms,
            generator=generator,
        )
    if arm == "ERROR_SUBSPACE_RECOVERY":
        return sample_structured_history_noise(
            calibration,
            int(reference.shape[0]),
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
    return None


def _update_perturbation_statistics(
    statistics: dict[str, Any], noise: HistoryNoise | None
) -> None:
    if noise is None:
        return
    value = (
        torch.stack((noise.previous_normalized, noise.current_normalized), dim=1)
        .detach()
        .to(dtype=torch.float64, device="cpu")
    )
    statistics["history_pair_count"] += int(value.shape[0])
    statistics["sum_pair_squared_norm"] += float(torch.sum(torch.square(value)).item())
    statistics["sum_field_squared"] += torch.sum(
        torch.square(value), dim=(0, 1, 2)
    ).numpy()
    statistics["field_coordinate_count"] += int(
        value.shape[0] * value.shape[1] * value.shape[2]
    )


def _final_perturbation_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    pair_count = int(statistics["history_pair_count"])
    coordinate_count = int(statistics["field_coordinate_count"])
    if pair_count == 0:
        return {
            "history_pair_count": 0,
            "mean_history_pair_squared_norm": 0.0,
            "per_field_rms_state_normalized_displacement": [0.0] * 5,
        }
    return {
        "history_pair_count": pair_count,
        "mean_history_pair_squared_norm": float(
            statistics["sum_pair_squared_norm"] / pair_count
        ),
        "per_field_rms_state_normalized_displacement": np.sqrt(
            statistics["sum_field_squared"] / coordinate_count
        ).tolist(),
    }


def _model_call(
    *,
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    event_hasher: _EventHasher,
    cost: dict[str, int],
    phase: str,
    kind: str,
    epoch: int,
    optimizer_step: int,
    microbatch: int,
    centers: torch.Tensor,
    gradient_bearing: bool,
) -> torch.Tensor:
    size = int(previous.shape[0])
    event_hasher.add(
        {
            "phase": phase,
            "kind": kind,
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "microbatch": microbatch,
            "centers": [int(value) for value in centers.tolist()],
            "sample_count": size,
            "gradient_bearing": gradient_bearing,
        }
    )
    counter = "training_model_calls" if gradient_bearing else "inference_model_calls"
    sample_counter = (
        "training_model_sample_calls"
        if gradient_bearing
        else "inference_model_sample_calls"
    )
    cost[counter] += 1
    cost[sample_counter] += size
    return predict_normalized_residual(
        model,
        previous,
        current,
        r0_trainer._geometry_slice(geometry_batch, size),
        normalization,
        fourier_tensors=r0_trainer._fourier_slice(fourier_tensors, size),
    )


def _build_detached_pushforward_exposure(
    previous: torch.Tensor,
    clean_future: torch.Tensor,
    prefix_prediction: torch.Tensor,
    normalization: NACANormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        previous.shape != clean_future.shape
        or previous.shape != prefix_prediction.shape
    ):
        raise ValueError("detached pushforward states and prefix must share shape")
    generated_current = (
        previous + normalization.decode_residual(prefix_prediction)
    ).detach()
    target = normalization.normalize_residual(clean_future - generated_current).detach()
    return generated_current, target


def _step_arm_batch(
    *,
    arm: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    previous: torch.Tensor,
    current: torch.Tensor,
    clean_target: torch.Tensor,
    centers: torch.Tensor,
    preceding: torch.Tensor | None,
    noise: HistoryNoise | None,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    microbatch_size: int,
    epoch: int,
    optimizer_step: int,
    event_hasher: _EventHasher,
    cost: dict[str, int],
) -> tuple[float, float]:
    group_size = int(previous.shape[0])
    if group_size not in (2, 4):
        raise ValueError("effective training group must contain two or four samples")
    if arm in ("IID_RECOVERY", "ERROR_SUBSPACE_RECOVERY") and noise is None:
        raise ValueError("recovery arm lacks a registered perturbation")
    if arm == "DETACHED_PUSHFORWARD" and preceding is None:
        raise ValueError("detached pushforward lacks preceding clean states")
    clean_future = current + normalization.decode_residual(clean_target)
    if noise is not None:
        recovery = make_recovery_presentation(
            previous,
            current,
            clean_future,
            HistoryNoise(
                noise.previous_normalized.to(previous.device),
                noise.current_normalized.to(current.device),
            ),
            normalization,
        )
    else:
        recovery = None
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for microbatch, offset in enumerate(range(0, group_size, microbatch_size)):
        stop = min(offset + microbatch_size, group_size)
        size = stop - offset
        center_chunk = centers[offset:stop]
        clean_prediction = _model_call(
            model=model,
            previous=previous[offset:stop],
            current=current[offset:stop],
            geometry_batch=geometry_batch,
            fourier_tensors=fourier_tensors,
            normalization=normalization,
            event_hasher=event_hasher,
            cost=cost,
            phase="train",
            kind="clean_gradient",
            epoch=epoch,
            optimizer_step=optimizer_step,
            microbatch=microbatch,
            centers=center_chunk,
            gradient_bearing=True,
        )
        clean_per_sample = torch.mean(
            torch.square(clean_prediction - clean_target[offset:stop]), dim=(1, 2)
        )
        objective = clean_per_sample
        if arm in ("IID_RECOVERY", "ERROR_SUBSPACE_RECOVERY"):
            assert recovery is not None
            recovery_prediction = _model_call(
                model=model,
                previous=recovery.previous[offset:stop],
                current=recovery.current[offset:stop],
                geometry_batch=geometry_batch,
                fourier_tensors=fourier_tensors,
                normalization=normalization,
                event_hasher=event_hasher,
                cost=cost,
                phase="train",
                kind="recovery_gradient",
                epoch=epoch,
                optimizer_step=optimizer_step,
                microbatch=microbatch,
                centers=center_chunk,
                gradient_bearing=True,
            )
            recovery_per_sample = torch.mean(
                torch.square(
                    recovery_prediction
                    - recovery.target_normalized_residual[offset:stop]
                ),
                dim=(1, 2),
            )
            objective = 0.5 * clean_per_sample + 0.5 * recovery_per_sample
        elif arm == "DETACHED_PUSHFORWARD":
            assert preceding is not None
            eligible = center_chunk >= 957
            if bool(torch.any(eligible)):
                eligible_centers = center_chunk[eligible]
                eligible_device = eligible.to(previous.device)
                with torch.no_grad():
                    prefix_prediction = _model_call(
                        model=model,
                        previous=preceding[offset:stop][eligible_device],
                        current=previous[offset:stop][eligible_device],
                        geometry_batch=geometry_batch,
                        fourier_tensors=fourier_tensors,
                        normalization=normalization,
                        event_hasher=event_hasher,
                        cost=cost,
                        phase="train",
                        kind="detached_prefix_inference",
                        epoch=epoch,
                        optimizer_step=optimizer_step,
                        microbatch=microbatch,
                        centers=eligible_centers,
                        gradient_bearing=False,
                    )
                predicted_current, exposure_target = (
                    _build_detached_pushforward_exposure(
                        previous[offset:stop][eligible_device],
                        clean_future[offset:stop][eligible_device],
                        prefix_prediction,
                        normalization,
                    )
                )
                exposure_prediction = _model_call(
                    model=model,
                    previous=previous[offset:stop][eligible_device],
                    current=predicted_current,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier_tensors,
                    normalization=normalization,
                    event_hasher=event_hasher,
                    cost=cost,
                    phase="train",
                    kind="detached_exposure_gradient",
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    microbatch=microbatch,
                    centers=eligible_centers,
                    gradient_bearing=True,
                )
                exposure_per_sample = torch.mean(
                    torch.square(exposure_prediction - exposure_target), dim=(1, 2)
                )
                objective = objective.clone()
                objective[eligible_device] = (
                    0.5 * clean_per_sample[eligible_device] + 0.5 * exposure_per_sample
                )
        loss = torch.mean(objective)
        if not torch.isfinite(loss):
            raise FloatingPointError("successor training loss became nonfinite")
        weight = size / group_size
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
            raise FloatingPointError("successor training gradient became nonfinite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("successor gradient norm became nonfinite")
    optimizer.step()
    for parameter in model.parameters():
        if not torch.all(torch.isfinite(parameter)):
            raise FloatingPointError("successor model parameter became nonfinite")
    return weighted_loss, float(gradient_norm.detach().cpu())


def _common_bindings(
    *,
    arm: str,
    seed: int,
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    calibration: Mapping[str, Any],
    calibration_final_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "seed": seed,
        "successor_contract_sha256": SUCCESSOR_CONTRACT_FILE_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "inherited_r0_contract_sha256": R0_CONTRACT_SHA256,
        "dataset_manifest_payload_sha256": dataset_manifest["canonical_payload_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
        "calibration_payload_sha256": calibration["canonical_payload_sha256"],
        "calibration_final_hash_manifest_sha256": calibration_final_sha256,
        "source_set_sha256": SOURCE_SET_SHA256,
        "prospective_opened": False,
        "sealed_opened": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }


def _write_final_hash_manifest(
    output: Path,
    *,
    arm: str,
    seed: int,
    calibration_final_sha256: str,
    initial_model_state_sha256: str,
    presentation_schedule_sha256: str,
) -> None:
    observed = {path.name for path in output.iterdir()}
    if observed != TRAINING_FILES:
        raise ValueError("successor training packet has missing or unexpected files")
    files: dict[str, Any] = {}
    for name in sorted(TRAINING_FILES):
        path = output / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != output:
            raise ValueError("successor training packet contains a nonregular entry")
        files[name] = {
            "relative_path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(
        output / "final_hash_manifest.json",
        {
            "schema": EXPECTED_PACKET_SCHEMAS["final_hash_manifest"],
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "seed": seed,
            "successor_contract_sha256": SUCCESSOR_CONTRACT_FILE_SHA256,
            "inherited_r0_contract_sha256": R0_CONTRACT_SHA256,
            "dataset_final_hash_manifest_sha256": DATASET_FINAL_SHA256,
            "calibration_final_hash_manifest_sha256": calibration_final_sha256,
            "paired_initialization_and_clean_order": True,
            "initial_model_state_sha256": initial_model_state_sha256,
            "presentation_schedule_sha256": presentation_schedule_sha256,
            "files": files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
            "online_defect_trigger": False,
        },
    )


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _checkpoint_payload(
    *,
    record_kind: str,
    model_config: Mapping[str, Any],
    model_state_dict: Mapping[str, torch.Tensor],
    epoch: int,
    best_epoch: int,
    best_score: float,
    bindings: Mapping[str, Any],
    model_call_schedule_sha256: str,
    rng_state_binding_sha256: str,
    model_call_cost_sha256: str,
    cost: Mapping[str, Any],
    model_only: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": EXPECTED_PACKET_SCHEMAS["training"],
        "record_kind": record_kind,
        **bindings,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_clean_development_score": best_score,
        "checkpoint_selection_metric": "clean_development_one_step_normalized_residual_rmse_median",
        "checkpoint_selection_used_rollout": False,
        "model_config": dict(model_config),
        "model_state_dict": dict(model_state_dict),
        "model_call_schedule_sha256": model_call_schedule_sha256,
        "rng_state_binding_sha256": rng_state_binding_sha256,
        "model_call_cost_sha256": model_call_cost_sha256,
        "training_model_calls": int(cost["training_model_calls"]),
        "development_selection_model_calls": int(
            cost["development_selection_model_calls"]
        ),
        "training_wall_time_seconds": float(cost["training_wall_time_seconds"]),
        "target_semantics": "normalized_(stored_clean_future_minus_presented_current)",
        "model_only": model_only,
    }
    if not model_only:
        if optimizer is None or scheduler is None:
            raise ValueError("resumable checkpoint lacks optimizer or scheduler")
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _record_development_calls(
    *,
    event_hasher: _EventHasher,
    cost: dict[str, int],
    epoch: int,
    dataset: NACABDF2Dataset,
    microbatch_size: int,
) -> None:
    centers = list(dataset.center_indices)
    for optimizer_step, group_start in enumerate(range(0, len(centers), 4)):
        group = centers[group_start : group_start + 4]
        for microbatch, offset in enumerate(range(0, len(group), microbatch_size)):
            chunk = group[offset : offset + microbatch_size]
            event_hasher.add(
                {
                    "phase": "clean_development_selection",
                    "kind": "clean_selection_inference",
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "microbatch": microbatch,
                    "centers": chunk,
                    "sample_count": len(chunk),
                    "gradient_bearing": False,
                }
            )
            cost["development_selection_model_calls"] += 1
            cost["development_selection_model_sample_calls"] += len(chunk)


def _canonical_development_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clean development score must be numeric")
    score = float(value)
    return math.inf if not math.isfinite(score) else score


def _select_development_checkpoint(
    *, epoch: int, score: float, best_epoch: int | None, best_score: float
) -> bool:
    if best_epoch is None:
        if epoch != 5:
            raise ValueError("the first development checkpoint must be epoch 5")
        return True
    return math.isfinite(score) and score < best_score


def _resource_smoke(
    *,
    arguments: argparse.Namespace,
    r0_contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    calibration_metadata: Mapping[str, Any],
    recovery_calibration: RecoveryCalibration,
    calibration_final_sha256: str,
    calibration_files: Mapping[str, Mapping[str, Any]],
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
    r0_trainer._seed_everything(arguments.seed)
    model = build_naca_pcno(r0_contract, geometry).to(device)
    initial_model_sha256 = _model_state_sha256(model)
    optimizer = r0_trainer._optimizer(model)
    orders, presentation = _presentation_orders(arguments.seed)
    intervention_generator, rng = _intervention_generator(arguments.seed)
    indices = orders[0, :4]
    previous, current, target, centers = _gather_batch(train_dataset, indices)
    preceding = (
        _preceding_states(train_dataset, centers)
        if arguments.arm == "DETACHED_PUSHFORWARD"
        else None
    )
    noise = _noise_pair(
        arm=arguments.arm,
        reference=current,
        calibration=recovery_calibration,
        generator=intervention_generator,
    )
    geometry_batch = geometry.expand(arguments.microbatch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)
    event_hasher = _EventHasher()
    cost = {
        "training_model_calls": 0,
        "training_model_sample_calls": 0,
        "inference_model_calls": 0,
        "inference_model_sample_calls": 0,
        "development_selection_model_calls": 0,
        "development_selection_model_sample_calls": 0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    loss, gradient_norm = _step_arm_batch(
        arm=arguments.arm,
        model=model,
        optimizer=optimizer,
        previous=previous.to(device),
        current=current.to(device),
        clean_target=target.to(device),
        centers=centers,
        preceding=preceding.to(device) if preceding is not None else None,
        noise=noise,
        geometry_batch=geometry_batch,
        fourier_tensors=fourier,
        normalization=normalization,
        microbatch_size=arguments.microbatch_size,
        epoch=1,
        optimizer_step=1,
        event_hasher=event_hasher,
        cost=cost,
    )
    wall_time = time.perf_counter() - started
    rng["final_generator_state_sha256"] = _generator_state_sha256(
        intervention_generator
    )
    common = _common_bindings(
        arm=arguments.arm,
        seed=arguments.seed,
        dataset_manifest=dataset_manifest,
        dataset_packet_sha256=dataset_packet_sha256,
        calibration=calibration_metadata,
        calibration_final_sha256=calibration_final_sha256,
    )
    receipt = _self_hashed(
        {
            "schema": EXPECTED_PACKET_SCHEMAS["resource_smoke"],
            "status": "succeeded",
            **common,
            "effective_batch_size": 4,
            "microbatch_size": arguments.microbatch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "full_resolution_num_nodes": geometry.num_nodes,
            "paired_initialization_and_clean_order": True,
            "initial_model_state_sha256": initial_model_sha256,
            "presentation_schedule_sha256": presentation[
                "presentation_schedule_sha256"
            ],
            "intervention_rng": rng,
            "model_call_schedule_sha256": event_hasher.hexdigest(),
            "training_model_calls": cost["training_model_calls"],
            "inference_model_calls": cost["inference_model_calls"],
            "smoke_wall_time_seconds": wall_time,
            "forward_backward_and_adamw_step_completed": True,
            "loss": loss,
            "gradient_norm_before_clip": gradient_norm,
            "production_device_binding": r0_trainer._device_binding(device),
            "runtime": r0_trainer._runtime(device),
            "cuda_peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "scientific_hyperparameters_changed": False,
            "checkpoint_written": False,
        }
    )
    _verify_live_sources()
    _reverify_calibration(
        arguments.calibration_dir, calibration_files, calibration_final_sha256
    )
    if (
        sha256_file(arguments.dataset_dir.resolve() / "final_hash_manifest.json")
        != dataset_packet_sha256
    ):
        raise ValueError("dataset packet changed during successor resource smoke")
    _write_json(output / "resource_smoke.json", receipt)
    return receipt


def _validate_smoke_production_bindings(
    smoke: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    initial_model_state_sha256: str,
    presentation_schedule_sha256: str,
    intervention_initial_generator_state_sha256: str,
) -> None:
    intervention_rng = smoke.get("intervention_rng")
    if (
        smoke.get("runtime") != runtime
        or smoke.get("initial_model_state_sha256") != initial_model_state_sha256
        or smoke.get("presentation_schedule_sha256") != presentation_schedule_sha256
        or not isinstance(intervention_rng, Mapping)
        or intervention_rng.get("initial_generator_state_sha256")
        != intervention_initial_generator_state_sha256
    ):
        raise PermissionError(
            "successor resource smoke does not replay the production initialization"
        )


def _validate_production_authorization(
    *,
    arguments: argparse.Namespace,
    common: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    if arguments.authorization is None or arguments.resource_smoke_receipt is None:
        raise PermissionError(
            "successor production training requires authorization and a resource-smoke receipt"
        )
    if arguments.resource_smoke_receipt.is_symlink():
        raise PermissionError("successor resource-smoke receipt is aliased")
    smoke, smoke_file_sha256 = r0_trainer._load_self_hashed_with_file_sha256(
        arguments.resource_smoke_receipt.resolve(),
        EXPECTED_PACKET_SCHEMAS["resource_smoke"],
    )
    shared_keys = (
        "experiment_id",
        "arm",
        "seed",
        "successor_contract_sha256",
        "preregistration_sha256",
        "inherited_r0_contract_sha256",
        "dataset_manifest_payload_sha256",
        "dataset_final_hash_manifest_sha256",
        "calibration_payload_sha256",
        "calibration_final_hash_manifest_sha256",
        "source_set_sha256",
        "prospective_opened",
        "sealed_opened",
        "online_solver_calls",
        "online_defect_trigger",
    )
    pair = (smoke.get("microbatch_size"), smoke.get("gradient_accumulation_steps"))
    smoke_loss = smoke.get("loss")
    smoke_gradient_norm = smoke.get("gradient_norm_before_clip")
    if (
        any(smoke.get(key) != common.get(key) for key in shared_keys)
        or smoke.get("status") != "succeeded"
        or smoke.get("effective_batch_size") != 4
        or smoke.get("paired_initialization_and_clean_order") is not True
        or smoke.get("forward_backward_and_adamw_step_completed") is not True
        or smoke.get("scientific_hyperparameters_changed") is not False
        or smoke.get("checkpoint_written") is not False
        or isinstance(smoke_loss, bool)
        or not isinstance(smoke_loss, (int, float))
        or not math.isfinite(float(smoke_loss))
        or isinstance(smoke_gradient_norm, bool)
        or not isinstance(smoke_gradient_norm, (int, float))
        or not math.isfinite(float(smoke_gradient_norm))
        or not isinstance(smoke.get("training_model_calls"), int)
        or smoke["training_model_calls"] < 1
        or smoke.get("production_device_binding") != r0_trainer._device_binding(device)
        or pair != (arguments.microbatch_size, arguments.gradient_accumulation_steps)
    ):
        raise PermissionError("successor resource-smoke receipt is mismatched")
    if arguments.authorization.is_symlink():
        raise PermissionError("successor execution authorization is aliased")
    authorization, authorization_file_sha256 = (
        r0_trainer._load_self_hashed_with_file_sha256(
            arguments.authorization.resolve(),
            EXPECTED_PACKET_SCHEMAS["execution_authorization"],
        )
    )
    authorized_arms = authorization.get("authorized_arms")
    authorized_seeds = authorization.get("authorized_seeds")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("successor_contract_sha256")
        != SUCCESSOR_CONTRACT_FILE_SHA256
        or authorization.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or authorization.get("inherited_r0_contract_sha256") != R0_CONTRACT_SHA256
        or authorization.get("dataset_final_hash_manifest_sha256")
        != DATASET_FINAL_SHA256
        or authorization.get("calibration_final_hash_manifest_sha256")
        != common["calibration_final_hash_manifest_sha256"]
        or authorization.get("source_set_sha256") != SOURCE_SET_SHA256
        or authorization.get("resource_smoke_payload_sha256")
        != smoke["canonical_payload_sha256"]
        or authorization.get("resource_smoke_file_sha256") != smoke_file_sha256
        or authorization.get("production_device_binding")
        != r0_trainer._device_binding(device)
        or authorization.get("allowed_actions") != ["train_pcno_naca0012_successor"]
        or not isinstance(authorized_arms, list)
        or len(authorized_arms) != len(set(authorized_arms))
        or any(arm not in TRAINING_ARMS for arm in authorized_arms)
        or arguments.arm not in authorized_arms
        or not isinstance(authorized_seeds, list)
        or len(authorized_seeds) != len(set(authorized_seeds))
        or any(seed not in SEEDS for seed in authorized_seeds)
        or arguments.seed not in authorized_seeds
        or authorization.get("pcno_code_audited") is not True
        or authorization.get("successor_training_authorized") is not True
        or authorization.get("prospective_opened") is not False
        or authorization.get("sealed_opened") is not False
        or authorization.get("online_solver_calls") is not False
        or authorization.get("online_defect_trigger") is not False
        or not isinstance(authorization.get("authorized_by"), str)
        or not authorization["authorized_by"].strip()
    ):
        raise PermissionError("successor execution authorization is mismatched")
    return smoke, smoke_file_sha256, authorization, authorization_file_sha256


def _run_training(
    *,
    arguments: argparse.Namespace,
    successor: Mapping[str, Any],
    r0_contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    development_dataset: NACABDF2Dataset,
    calibration_metadata: Mapping[str, Any],
    recovery_calibration: RecoveryCalibration,
    calibration_final_sha256: str,
    calibration_files: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    common = _common_bindings(
        arm=arguments.arm,
        seed=arguments.seed,
        dataset_manifest=dataset_manifest,
        dataset_packet_sha256=dataset_packet_sha256,
        calibration=calibration_metadata,
        calibration_final_sha256=calibration_final_sha256,
    )
    smoke, smoke_file_sha256, authorization, authorization_file_sha256 = (
        _validate_production_authorization(
            arguments=arguments, common=common, device=device
        )
    )
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("successor training output or its parent is aliased")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"successor training output already exists: {output}")

    r0_trainer._seed_everything(arguments.seed)
    model = build_naca_pcno(r0_contract, geometry).to(device)
    model_config = model.model_config()  # type: ignore[attr-defined]
    validate_naca_model_config(model_config, r0_contract, geometry)
    initial_model_state_sha256 = _model_state_sha256(model)
    torch_rng_after_initialization_sha256 = _tensor_sha256(torch.get_rng_state())
    optimizer = r0_trainer._optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1.0e-5
    )
    orders, presentation = _presentation_orders(arguments.seed)
    intervention_generator, intervention_rng = _intervention_generator(arguments.seed)
    initial_rng_binding = {
        "torch_rng_after_model_initialization_sha256": torch_rng_after_initialization_sha256,
        "presentation_initial_generator_state_sha256": presentation[
            "initial_generator_state_sha256"
        ],
        "intervention_initial_generator_state_sha256": intervention_rng[
            "initial_generator_state_sha256"
        ],
    }
    initial_rng_state_sha256 = _canonical_sha256(initial_rng_binding)
    production_runtime = r0_trainer._runtime(device)
    _validate_smoke_production_bindings(
        smoke,
        runtime=production_runtime,
        initial_model_state_sha256=initial_model_state_sha256,
        presentation_schedule_sha256=presentation["presentation_schedule_sha256"],
        intervention_initial_generator_state_sha256=intervention_rng[
            "initial_generator_state_sha256"
        ],
    )
    run_bindings = {
        **common,
        "paired_initialization_and_clean_order": True,
        "initial_model_state_sha256": initial_model_state_sha256,
        "presentation_schedule_sha256": presentation["presentation_schedule_sha256"],
    }
    target_semantics = {
        "model_output": "normalized_next_state_residual",
        "clean_target": "normalized_(stored_clean_future_minus_clean_current)",
        "recovery_target": "normalized_(stored_clean_future_minus_displaced_current)",
        "detached_pushforward_target": "normalized_(stored_clean_future_minus_detached_predicted_current)",
        "detached_prefix_gradient": "stopped",
        "error_pair_sampling_mean": "zero",
        "stored_error_pair_mean_used_for_sampling": False,
    }
    output.mkdir(parents=True)
    config = _self_hashed(
        {
            "schema": EXPECTED_PACKET_SCHEMAS["training_config"],
            "status": "frozen_configuration",
            **run_bindings,
            "arm_contract": successor["arms"][arguments.arm],
            "presentation": presentation,
            "initial_rng_state_sha256": initial_rng_state_sha256,
            "intervention_rng": intervention_rng,
            "target_semantics": target_semantics,
            "epochs": 100,
            "effective_batch_size": 4,
            "microbatch_size": arguments.microbatch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1.0e-3,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 1.0e-5,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "t_max": 100,
                "eta_min": 1.0e-5,
            },
            "gradient_clip_norm": 1.0,
            "checkpoint_epochs": list(range(5, 101, 5)),
            "checkpoint_selection": "lowest_clean_development_one_step_normalized_residual_rmse_median_earliest_tie",
            "rollout_used_for_selection": False,
            "amp": False,
            "tf32": False,
        }
    )
    inputs = _self_hashed(
        {
            "schema": EXPECTED_PACKET_SCHEMAS["training_inputs"],
            "status": "verified_inputs",
            **run_bindings,
            "train_frames_inclusive": [955, 1194],
            "train_centers_inclusive": [956, 1193],
            "development_frames_inclusive": [1233, 1472],
            "development_centers_inclusive": [1234, 1471],
            "training_population_role": "train",
            "selection_population_role": "development",
            "selection_uses_clean_one_step_only": True,
            "resource_smoke_payload_sha256": smoke["canonical_payload_sha256"],
            "resource_smoke_file_sha256": smoke_file_sha256,
            "authorization_payload_sha256": authorization["canonical_payload_sha256"],
            "authorization_file_sha256": authorization_file_sha256,
            "production_device_binding": r0_trainer._device_binding(device),
            "source_files": SOURCE_RECORDS_AT_IMPORT,
            "calibration_array_records": calibration_metadata["arrays"],
            "parent_r0": successor["parent_r0"],
        }
    )
    runtime = _self_hashed(
        {
            "schema": EXPECTED_PACKET_SCHEMAS["training"],
            "record_kind": "runtime_manifest",
            **run_bindings,
            "runtime": production_runtime,
            "production_device_binding": r0_trainer._device_binding(device),
        }
    )
    _write_json(output / "config.json", config)
    _write_json(output / "input_manifest.json", inputs)
    _write_json(output / "runtime_manifest.json", runtime)
    _write_json(
        output / "status.json",
        _self_hashed(
            {
                "schema": EXPECTED_PACKET_SCHEMAS["training"],
                "record_kind": "status",
                "status": "running",
                **run_bindings,
                "last_completed_epoch": 0,
            }
        ),
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
    event_hasher = _EventHasher()
    cost: dict[str, Any] = {
        "training_model_calls": 0,
        "training_model_sample_calls": 0,
        "inference_model_calls": 0,
        "inference_model_sample_calls": 0,
        "development_selection_model_calls": 0,
        "development_selection_model_sample_calls": 0,
    }
    perturbation_statistics: dict[str, Any] = {
        "history_pair_count": 0,
        "sum_pair_squared_norm": 0.0,
        "sum_field_squared": np.zeros(5, dtype=np.float64),
        "field_coordinate_count": 0,
    }
    best_score = math.inf
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    loop_started = time.perf_counter()
    try:
        for epoch in range(1, 101):
            model.train()
            loss_sum = 0.0
            gradient_norms: list[float] = []
            presentations = 0
            optimizer_steps = 0
            order = orders[epoch - 1]
            for start in range(0, 238, 4):
                indices = order[start : start + 4]
                previous, current, clean_target, centers = _gather_batch(
                    train_dataset, indices
                )
                group_size = int(previous.shape[0])
                preceding = (
                    _preceding_states(train_dataset, centers)
                    if arguments.arm == "DETACHED_PUSHFORWARD"
                    else None
                )
                noise = _noise_pair(
                    arm=arguments.arm,
                    reference=current,
                    calibration=recovery_calibration,
                    generator=intervention_generator,
                )
                _update_perturbation_statistics(perturbation_statistics, noise)
                total_optimizer_steps += 1
                loss, gradient_norm = _step_arm_batch(
                    arm=arguments.arm,
                    model=model,
                    optimizer=optimizer,
                    previous=previous.to(device),
                    current=current.to(device),
                    clean_target=clean_target.to(device),
                    centers=centers,
                    preceding=preceding.to(device) if preceding is not None else None,
                    noise=noise,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier,
                    normalization=normalization,
                    microbatch_size=arguments.microbatch_size,
                    epoch=epoch,
                    optimizer_step=total_optimizer_steps,
                    event_hasher=event_hasher,
                    cost=cost,
                )
                loss_sum += loss * group_size
                gradient_norms.append(gradient_norm)
                presentations += group_size
                optimizer_steps += 1
            if presentations != 238 or optimizer_steps != 60:
                raise ValueError("successor epoch presentation or step count differs")
            development_score: float | None = None
            selected = False
            if epoch % 5 == 0:
                observed_development_score, _ = r0_trainer._development_score(
                    model=model,
                    loader=development_loader,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier,
                    normalization=normalization,
                    device=device,
                    microbatch_size=arguments.microbatch_size,
                )
                _record_development_calls(
                    event_hasher=event_hasher,
                    cost=cost,
                    epoch=epoch,
                    dataset=development_dataset,
                    microbatch_size=arguments.microbatch_size,
                )
                development_score = _canonical_development_score(
                    observed_development_score
                )
                if _select_development_checkpoint(
                    epoch=epoch,
                    score=development_score,
                    best_epoch=best_epoch,
                    best_score=best_score,
                ):
                    best_score = development_score
                    best_epoch = epoch
                    best_state = _clone_state_dict(model)
                    selected = True
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_sample_mean_objective": loss_sum / presentations,
                    "maximum_gradient_norm_before_clip": float(max(gradient_norms)),
                    "clean_presentations": presentations,
                    "optimizer_steps": optimizer_steps,
                    "learning_rate_after_scheduler_step": float(
                        scheduler.get_last_lr()[0]
                    ),
                    "clean_development_one_step_normalized_residual_rmse_median": (
                        _json_number(development_score)
                        if development_score is not None
                        else None
                    ),
                    "selected_as_best": selected,
                }
            )
            _write_json(
                output / "history.json",
                _self_hashed(
                    {
                        "schema": EXPECTED_PACKET_SCHEMAS["training"],
                        "record_kind": "history",
                        **run_bindings,
                        "epochs": history,
                    }
                ),
            )
            _write_json(
                output / "status.json",
                _self_hashed(
                    {
                        "schema": EXPECTED_PACKET_SCHEMAS["training"],
                        "record_kind": "status",
                        "status": "running",
                        **run_bindings,
                        "last_completed_epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_clean_development_score": (
                            _json_number(best_score) if best_epoch is not None else None
                        ),
                    }
                ),
            )
            _atomic_torch_save(
                output / "last.pt",
                {
                    "schema": EXPECTED_PACKET_SCHEMAS["training"],
                    "record_kind": "incomplete_last_checkpoint",
                    **run_bindings,
                    "epoch": epoch,
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "incomplete_not_scientific_evidence": True,
                },
            )
    except Exception as error:
        _write_json(
            output / "status.json",
            _self_hashed(
                {
                    "schema": EXPECTED_PACKET_SCHEMAS["training"],
                    "record_kind": "status",
                    "status": "failed_incomplete_not_scientific_evidence",
                    **run_bindings,
                    "last_completed_epoch": len(history),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
        )
        raise

    training_wall_time_seconds = time.perf_counter() - loop_started
    if best_epoch is None or best_state is None or best_epoch not in range(5, 101, 5):
        raise ValueError("successor training completed without a selected checkpoint")
    intervention_rng["final_generator_state_sha256"] = _generator_state_sha256(
        intervention_generator
    )
    final_rng_binding = {
        **initial_rng_binding,
        "torch_rng_at_training_completion_sha256": _tensor_sha256(
            torch.get_rng_state()
        ),
        "presentation_final_generator_state_sha256": presentation[
            "final_generator_state_sha256"
        ],
        "intervention_final_generator_state_sha256": intervention_rng[
            "final_generator_state_sha256"
        ],
    }
    rng_state_binding_sha256 = _canonical_sha256(final_rng_binding)
    perturbation_summary = _final_perturbation_statistics(perturbation_statistics)
    cost.update(
        {
            "optimizer_steps": total_optimizer_steps,
            "clean_presentations": 100 * 238,
            "intervention_presentations": (
                100 * 238
                if arguments.arm in ("IID_RECOVERY", "ERROR_SUBSPACE_RECOVERY")
                else 100 * 237
                if arguments.arm == "DETACHED_PUSHFORWARD"
                else 0
            ),
            "calibration_parent_predictor_count": 3,
            "calibration_error_pair_bank_size": 3 * 237,
            "calibration_error_pair_rank": 16,
            "training_wall_time_seconds": training_wall_time_seconds,
        }
    )
    if event_hasher.count != (
        cost["training_model_calls"]
        + cost["inference_model_calls"]
        + cost["development_selection_model_calls"]
    ):
        raise ValueError("model-call event count differs from cost counters")
    model_call_schedule_sha256 = event_hasher.hexdigest()
    model_call_cost_sha256 = _canonical_sha256(cost)
    summary = _self_hashed(
        {
            "schema": EXPECTED_PACKET_SCHEMAS["training"],
            "record_kind": "summary",
            "status": "complete",
            "classification": "SCIENTIFIC_TRAINING_ARTIFACT",
            **run_bindings,
            "completed_epochs": 100,
            "best_epoch": best_epoch,
            "best_clean_development_score": _json_number(best_score),
            "checkpoint_selection_metric": "clean_development_one_step_normalized_residual_rmse_median",
            "checkpoint_selection_used_rollout": False,
            "initial_rng_state_sha256": initial_rng_state_sha256,
            "rng_state_binding": final_rng_binding,
            "rng_state_binding_sha256": rng_state_binding_sha256,
            "model_call_schedule_sha256": model_call_schedule_sha256,
            "model_call_cost": cost,
            "model_call_cost_sha256": model_call_cost_sha256,
            "training_model_calls": cost["training_model_calls"],
            "training_model_sample_calls": cost["training_model_sample_calls"],
            "detached_prefix_inference_model_calls": cost["inference_model_calls"],
            "detached_prefix_inference_model_sample_calls": cost[
                "inference_model_sample_calls"
            ],
            "development_selection_model_calls": cost[
                "development_selection_model_calls"
            ],
            "development_selection_model_sample_calls": cost[
                "development_selection_model_sample_calls"
            ],
            "training_wall_time_seconds": training_wall_time_seconds,
            "realized_perturbation_statistics": perturbation_summary,
            "target_semantics": target_semantics,
            "scientific_claim_made": False,
        }
    )
    last_checkpoint = _checkpoint_payload(
        record_kind="last_checkpoint",
        model_config=model_config,
        model_state_dict=_clone_state_dict(model),
        epoch=100,
        best_epoch=best_epoch,
        best_score=best_score,
        bindings=run_bindings,
        model_call_schedule_sha256=model_call_schedule_sha256,
        rng_state_binding_sha256=rng_state_binding_sha256,
        model_call_cost_sha256=model_call_cost_sha256,
        cost=cost,
        model_only=False,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    best_checkpoint = _checkpoint_payload(
        record_kind="selected_checkpoint",
        model_config=model_config,
        model_state_dict=best_state,
        epoch=best_epoch,
        best_epoch=best_epoch,
        best_score=best_score,
        bindings=run_bindings,
        model_call_schedule_sha256=model_call_schedule_sha256,
        rng_state_binding_sha256=rng_state_binding_sha256,
        model_call_cost_sha256=model_call_cost_sha256,
        cost=cost,
        model_only=True,
    )
    _atomic_torch_save(output / "last.pt", last_checkpoint)
    _atomic_torch_save(output / "best.pt", best_checkpoint)
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "status.json",
        _self_hashed(
            {
                "schema": EXPECTED_PACKET_SCHEMAS["training"],
                "record_kind": "status",
                "status": "complete",
                **run_bindings,
                "last_completed_epoch": 100,
                "best_epoch": best_epoch,
                "best_clean_development_score": _json_number(best_score),
            }
        ),
    )
    _verify_live_sources()
    _reverify_calibration(
        arguments.calibration_dir, calibration_files, calibration_final_sha256
    )
    if (
        sha256_file(arguments.dataset_dir.resolve() / "final_hash_manifest.json")
        != dataset_packet_sha256
    ):
        raise ValueError("dataset packet changed during successor training")
    _write_final_hash_manifest(
        output,
        arm=arguments.arm,
        seed=arguments.seed,
        calibration_final_sha256=calibration_final_sha256,
        initial_model_state_sha256=initial_model_state_sha256,
        presentation_schedule_sha256=presentation["presentation_schedule_sha256"],
    )
    return summary


def _validate_open_roles(
    roles: Mapping[str, tuple[np.ndarray, np.ndarray, NACABDF2Dataset]],
    successor: Mapping[str, Any],
) -> None:
    if set(roles) != {"train", "development"}:
        raise ValueError("successor loader exposed an unexpected population role")
    expected = {
        "train": ([955, 1194], [956, 1193]),
        "development": ([1233, 1472], [1234, 1471]),
    }
    for role, (frames, centers) in expected.items():
        states, frame_indices, dataset = roles[role]
        if (
            frame_indices.tolist() != list(range(frames[0], frames[1] + 1))
            or list(dataset.center_indices) != list(range(centers[0], centers[1] + 1))
            or states.shape[0] != 240
        ):
            raise ValueError(f"successor {role} population differs")
        if successor["dataset"][f"{role}_frames_inclusive"] != frames:
            raise ValueError(f"successor {role} frame contract differs")
        if successor["dataset"][f"{role}_centers_inclusive"] != centers:
            raise ValueError(f"successor {role} center contract differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--successor-contract", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=TRAINING_ARMS, required=True)
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
        r0_trainer._validate_resource_config(arguments)
        successor = _load_successor_contract(
            arguments.successor_contract,
            arguments.preregistration,
            arguments.contract,
        )
        r0_contract = load_naca_baseline_contract(arguments.contract)
        (
            dataset_manifest,
            geometry,
            normalization,
            roles,
            dataset_packet_sha256,
        ) = r0_trainer._load_dataset(arguments.dataset_dir, r0_contract)
        validate_successor_math_contract(successor, r0_contract)
        if dataset_packet_sha256 != DATASET_FINAL_SHA256:
            raise ValueError(
                "dataset final-hash manifest differs from successor contract"
            )
        _validate_open_roles(roles, successor)
        (
            calibration_metadata,
            recovery_calibration,
            calibration_final_sha256,
            calibration_files,
        ) = _load_calibration(
            arguments.calibration_dir,
            successor,
            dataset_manifest,
            dataset_packet_sha256,
            normalization,
            geometry.num_nodes,
        )
        _verify_live_sources()
        device = torch.device(arguments.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if arguments.resource_smoke:
            result = _resource_smoke(
                arguments=arguments,
                r0_contract=r0_contract,
                dataset_manifest=dataset_manifest,
                dataset_packet_sha256=dataset_packet_sha256,
                geometry=geometry,
                normalization=normalization,
                train_dataset=roles["train"][2],
                calibration_metadata=calibration_metadata,
                recovery_calibration=recovery_calibration,
                calibration_final_sha256=calibration_final_sha256,
                calibration_files=calibration_files,
                device=device,
            )
        else:
            result = _run_training(
                arguments=arguments,
                successor=successor,
                r0_contract=r0_contract,
                dataset_manifest=dataset_manifest,
                dataset_packet_sha256=dataset_packet_sha256,
                geometry=geometry,
                normalization=normalization,
                train_dataset=roles["train"][2],
                development_dataset=roles["development"][2],
                calibration_metadata=calibration_metadata,
                recovery_calibration=recovery_calibration,
                calibration_final_sha256=calibration_final_sha256,
                calibration_files=calibration_files,
                device=device,
            )
        _verify_live_sources()
        _reverify_calibration(
            arguments.calibration_dir,
            calibration_files,
            calibration_final_sha256,
        )
        if (
            sha256_file(arguments.dataset_dir.resolve() / "final_hash_manifest.json")
            != dataset_packet_sha256
        ):
            raise ValueError("dataset packet changed during successor execution")
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"NACA0012 corrective successor training failed: {error}", file=sys.stderr
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
