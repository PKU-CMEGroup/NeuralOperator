"""Train the six frozen NACA0012 corrective-extension arms.

This is a train/development-only entry point for
``B3B4_NACA_CM_EXT_20260902A``.  It reuses the verified R0 data, geometry, and
optimizer code, never calls SU2, and publishes a packet only by an atomic
directory rename after every registered file has been hashed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

_ENTRY_PATH = Path(__file__)
if _ENTRY_PATH.is_symlink() or not _ENTRY_PATH.is_file():
    raise RuntimeError("extension trainer source is absent or aliased")
REPO_ROOT = _ENTRY_PATH.resolve().parents[2]
TRAINER_SOURCE = "scripts/time_dependent_no/train_pcno_naca0012_corrective_extension.py"
CORE_SOURCE = "utility/time_dependent_no/pcno_naca0012_corrective_extension.py"
RELABEL_SOURCE = "utility/time_dependent_no/su2_naca0012_relabel.py"
if _ENTRY_PATH.resolve() != (REPO_ROOT / TRAINER_SOURCE).resolve():
    raise RuntimeError("extension trainer executed from an unexpected path")

import numpy as np
import torch

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import train_pcno_naca0012 as r0_trainer
from scripts.time_dependent_no import train_pcno_naca0012_successor as parent_trainer
from utility.time_dependent_no.pcno_naca0012 import (
    NACABDF2Dataset,
    NACANormalization,
    VerifiedNACAGeometry,
    build_naca_pcno,
    load_naca_baseline_contract,
    predict_normalized_residual,
    validate_naca_model_config,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    EXTENSION_CONTRACT_SHA256,
    EXTENSION_EXPERIMENT_ID,
    EXTENSION_LEARNED_ARMS,
    EXTENSION_PREREGISTRATION_SHA256,
    ExponentialMovingAverage,
    FourStepVPredictionScheduler,
    VerifiedExtensionContract,
    build_naca_pcno_refiner,
    build_refiner_input,
    group_examples_by_prefix_depth,
    load_extension_contract,
    make_detached_prefix_presentation,
    make_paired_displaced_presentation,
    make_refiner_training_presentation,
    paired_bank_sign_for_epoch,
    paired_clean_displaced_objective,
    realize_prefix_depths,
    refined_recurrent_step,
    sample_curriculum_requested_prefix_depth,
    sample_mp_pde_prefix_depth,
    sample_refiner_noise_tape,
    sample_refiner_timesteps,
    validate_extension_math_contract,
)
from utility.time_dependent_no.pcno_naca0012_successor import (
    validate_successor_math_contract,
)
from utility.time_dependent_no.su2_naca0012_relabel import (
    RELABEL_RECEIPT_SCHEMA,
    REQUIRED_RECEIPT_CHECKS,
    relabel_step_index,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

EXPERIMENT_ID = EXTENSION_EXPERIMENT_ID
ARMS = EXTENSION_LEARNED_ARMS
SEEDS = (17, 29, 43)
EMA_ARMS = (
    "CLEAN_EMA",
    "CURRICULUM_EMA_PUSHFORWARD_K13",
    "PCNO_PDEREFINER_K3_VPRED",
)
PUSHFORWARD_ARMS = (
    "MP_PDE_PUSHFORWARD_M01",
    "CURRICULUM_EMA_PUSHFORWARD_K13",
)
PAIRED_ARMS = ("PAIRED_RECOVERY", "DYNAMICS_RELABEL")
R0_CONTRACT_SHA256 = parent_trainer.R0_CONTRACT_SHA256
SUCCESSOR_CONTRACT_SHA256 = parent_trainer.SUCCESSOR_CONTRACT_FILE_SHA256
SUCCESSOR_PREREGISTRATION_SHA256 = parent_trainer.PREREGISTRATION_SHA256
SUCCESSOR_CALIBRATION_SHA256 = (
    "6a33f2e20a92dcfa58d16872b6abd2f6d2744731c8c678815618cafd373d0dbc"
)
DATASET_FINAL_SHA256 = parent_trainer.DATASET_FINAL_SHA256
RESOURCE_MANIFEST_SHA256 = (
    "7e20bb67ebcf09447f0badd4d10752a78f749f5c038bb444cedff81d39ff56f5"
)

PAIRED_BANK_SCHEMA = "time_dependent_no.naca_corrective_extension_paired_bank.v1"
PAIRED_BANK_CASE_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_paired_bank_case.v1"
)
PAIRED_BANK_FINAL_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_paired_bank_final_manifest.v1"
)
PAIRED_BANK_ARRAY_FILE = "paired_bank.npz"
PAIRED_BANK_METADATA_FILE = "bank.json"
RELABEL_PILOT_RECEIPT_SCHEMA = "time_dependent_no.naca_su2_relabel_pilot.v1"
RELABEL_PILOT_FINAL_SCHEMA = (
    "time_dependent_no.naca_su2_relabel_pilot_final_hash_manifest.v1"
)
RELABEL_PILOT_RECEIPT_FILE = "pilot_receipt.json"
PAIRED_BANK_ARRAY_KEYS = (
    "centers",
    "signs",
    "coordinates",
    "normalized_directions",
    "direction_draw_indices",
    "redraw_counts",
    "displaced_previous",
    "displaced_current",
    "solver_future",
)
PAIRED_BANK_BASE_SEED = 20260902
PAIRED_BANK_PILOT_CENTERS = (956, 1075, 1193)
PAIRED_BANK_REGISTERED_STD = np.asarray(
    [
        0.0023510525449798097,
        0.0017943288061936155,
        0.004045914459334123,
        0.002355418735336881,
        0.001718743054261789,
    ],
    dtype=np.float64,
)
PAIRED_BANK_RATIO_RANGE = (0.95, 1.05)
TRAINING_SCHEMA = "time_dependent_no.naca_corrective_extension_training.v1"
SMOKE_SCHEMA = "time_dependent_no.naca_corrective_extension_resource_smoke.v1"
AUTHORIZATION_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_execution_authorization.v1"
)
FINAL_MANIFEST_SCHEMA = (
    "time_dependent_no.naca_corrective_extension_training_final_manifest.v1"
)
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


def _capture_source_records() -> dict[str, dict[str, Any]]:
    records = {
        name: dict(record)
        for name, record in parent_trainer.SOURCE_RECORDS_AT_IMPORT.items()
    }
    for relative in (CORE_SOURCE, RELABEL_SOURCE, TRAINER_SOURCE):
        records[relative] = _stable_source_record(REPO_ROOT / relative, relative)
    return records


SOURCE_RECORDS_AT_IMPORT = _capture_source_records()
SOURCE_SET_SHA256 = _canonical_sha256(SOURCE_RECORDS_AT_IMPORT)


def _verify_live_sources() -> None:
    observed = {
        relative: _stable_source_record(REPO_ROOT / relative, relative)
        for relative in SOURCE_RECORDS_AT_IMPORT
    }
    if observed != SOURCE_RECORDS_AT_IMPORT:
        raise ValueError("bound extension source changed during execution")
    if _canonical_sha256(observed) != SOURCE_SET_SHA256:
        raise ValueError("extension source-set hash differs")


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    return _canonical_sha256(
        {name: _tensor_sha256(value) for name, value in model.state_dict().items()}
    )


def _generator_state_sha256(generator: torch.Generator) -> str:
    return _tensor_sha256(generator.get_state())


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_training_contract(payload: Mapping[str, Any]) -> None:
    """Validate every production field consumed outside the core primitives."""

    validate_extension_math_contract(payload)
    if payload.get("parents") != {
        "r0_contract_sha256": R0_CONTRACT_SHA256,
        "successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
        "successor_calibration_final_hash_manifest_sha256": SUCCESSOR_CALIBRATION_SHA256,
        "dataset_final_hash_manifest_sha256": DATASET_FINAL_SHA256,
        "resource_manifest_sha256": RESOURCE_MANIFEST_SHA256,
        "su2_release": "SU2 v8.5.0 Harrier",
        "su2_executable_sha256": "61a803e0baf49382210888cc32f14f4ea837681a0e6cc2df6b1d5a99ce5baf8c",
    }:
        raise ValueError("extension parent provenance differs")
    if payload.get("population") != {
        "train_frames_inclusive": [955, 1194],
        "train_centers_inclusive": [956, 1193],
        "development_frames_inclusive": [1233, 1472],
        "development_centers_inclusive": [1234, 1471],
        "prospective_frames_inclusive": [1511, 1750],
        "sealed_frames_inclusive": [1755, 1994],
    }:
        raise ValueError("extension population contract differs")
    if payload.get("optimization") != {
        "seeds": [17, 29, 43],
        "epochs": 100,
        "effective_batch_size": 4,
        "optimizer": "AdamW",
        "initial_learning_rate": 0.001,
        "weight_decay": 0.00001,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 100,
        "scheduler_eta_min": 0.00001,
        "gradient_clip_norm": 1.0,
        "checkpoint_epochs": list(range(5, 101, 5)),
        "rollout_used_for_selection": False,
    }:
        raise ValueError("extension optimization contract differs")
    expected_protection = {
        "train_opened": True,
        "development_opened": True,
        "prospective_opened": False,
        "sealed_opened": False,
        "offline_training_label_solver_calls": True,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }
    if payload.get("protection") != expected_protection:
        raise ValueError("extension protection contract differs")
    arms = payload["arms"]
    if arms["CLEAN_EMA"] != {
        "objective": "clean_only",
        "ema_decay": 0.995,
        "primary_deployment": "ema",
        "online_result_reported": True,
    }:
        raise ValueError("CLEAN_EMA contract differs")
    if arms["MP_PDE_PUSHFORWARD_M01"] != {
        "epoch_1_prefix_depth": 0,
        "later_prefix_depth_distribution": "uniform_batchwise_over_0_1",
        "sampling_unit": "one_requested_depth_per_minibatch_with_per_example_train_history_cap",
        "prefix_model": "current_online_model",
        "prefix_gradient": "stopped",
        "loss": "terminal_only",
        "target": "aligned_stored_clean_future",
        "inference_corrector": False,
    }:
        raise ValueError("literal MP-PDE contract differs")
    curriculum = arms["CURRICULUM_EMA_PUSHFORWARD_K13"]
    for key, expected in {
        "exposure_probability": "0.5_times_min_1_epoch_over_10",
        "exposed_prefix_depth_distribution": "uniform_over_1_2_3",
        "sampling_unit": "one_requested_depth_per_minibatch_with_per_example_train_history_cap",
        "prefix_model": "ema_eval_model",
        "ema_decay": 0.995,
        "prefix_gradient": "stopped",
        "loss": "terminal_only",
        "target": "aligned_stored_clean_future",
        "primary_deployment": "ema",
    }.items():
        if curriculum.get(key) != expected:
            raise ValueError(f"curriculum contract {key} differs")
    for arm, target in (
        ("PAIRED_RECOVERY", "stored_clean_future_from_undisplaced_pair"),
        ("DYNAMICS_RELABEL", "stored_su2_future_from_displaced_pair"),
    ):
        if arms[arm] != {
            "clean_loss_weight": 0.5,
            "displaced_loss_weight": 0.5,
            "displaced_inputs": "paired_query_bank",
            "target": target,
            "sign_schedule": "minus_on_odd_epochs_plus_on_even_epochs",
        }:
            raise ValueError(f"{arm} contract differs")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or (
        evaluation.get("pderefiner_development_selection_sampler_seed") != 101
        or evaluation.get("report_online_and_ema") is not True
        or evaluation.get("report_runtime_and_peak_memory") is not True
        or evaluation.get("common_evaluator_required") is not True
    ):
        raise ValueError("extension evaluation contract differs")


@dataclass(frozen=True)
class VerifiedPairedBank:
    centers: np.ndarray
    signs: np.ndarray
    coordinates: np.ndarray
    normalized_directions: np.ndarray
    direction_draw_indices: np.ndarray
    redraw_counts: np.ndarray
    displaced_previous: np.ndarray
    displaced_current: np.ndarray
    solver_future: np.ndarray
    final_manifest_sha256: str
    pilot_final_manifest_sha256: str
    file_records: Mapping[str, Mapping[str, Any]]
    root: Path
    pilot_final_manifest_path: Path

    def select(
        self, centers: torch.Tensor, sign: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if centers.dtype != torch.int64 or centers.ndim != 1:
            raise ValueError("bank centers must be an int64 vector")
        if sign not in (-1, 1):
            raise ValueError("bank sign must be -1 or +1")
        center_values = centers.detach().cpu().tolist()
        offsets = [int(value) - 956 for value in center_values]
        if any(offset < 0 or offset >= 238 for offset in offsets):
            raise ValueError("bank lookup escapes the train centers")
        sign_offset = 0 if sign == -1 else 1
        values = (
            self.displaced_previous[sign_offset, offsets],
            self.displaced_current[sign_offset, offsets],
            self.solver_future[sign_offset, offsets],
        )
        return tuple(
            torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)
            for value in values
        )  # type: ignore[return-value]


def _validate_relabel_pilot_final_manifest(path: Path, *, expected_sha256: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != "final_hash_manifest.json"
    ):
        raise ValueError("relabel-pilot final manifest is absent, aliased, or misnamed")
    payload_bytes = path.read_bytes()
    if sha256(payload_bytes).hexdigest() != expected_sha256:
        raise ValueError("relabel-pilot final-manifest SHA256 differs")
    final = json.loads(payload_bytes)
    files = final.get("files") if isinstance(final, Mapping) else None
    if (
        not isinstance(final, Mapping)
        or final.get("schema") != RELABEL_PILOT_FINAL_SCHEMA
        or final.get("experiment_id") != EXPERIMENT_ID
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or not isinstance(files, Mapping)
        or RELABEL_PILOT_RECEIPT_FILE not in files
    ):
        raise ValueError("relabel-pilot final manifest differs")
    root = path.resolve().parent
    observed_files = {
        candidate.resolve().relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "final_hash_manifest.json"
    }
    if observed_files != set(files):
        raise ValueError("relabel-pilot packet inventory differs")
    for relative, record in files.items():
        r0_trainer._verify_record(root, record, relative)
    receipt = r0_trainer._load_self_hashed_snapshot(
        root,
        files[RELABEL_PILOT_RECEIPT_FILE],
        RELABEL_PILOT_RECEIPT_FILE,
        RELABEL_PILOT_RECEIPT_SCHEMA,
    )
    gates = receipt.get("gates")
    cases = receipt.get("case_receipts")
    expected_gates = {
        "all_registered_cases",
        "all_case_receipts",
        "realized_displacement_scale",
        "zero_control",
        "repeatability",
        "auxiliary_invariance",
        "response_separation",
        "authority_immutable",
    }
    if (
        receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("status") != "pilot_succeeded"
        or receipt.get("scientifically_usable") is not True
        or receipt.get("paired_query_bank_unlocked") is not True
        or receipt.get("error") is not None
        or receipt.get("registered_case_count") != 11
        or receipt.get("attempted_solver_call_count") != 11
        or receipt.get("completed_case_receipt_count") != 11
        or not isinstance(cases, list)
        or len(cases) != 11
        or any(
            not isinstance(case, Mapping)
            or case.get("status") != "validation_succeeded"
            or case.get("scientifically_usable") is not True
            for case in cases
        )
        or not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or any(gates[name] is not True for name in expected_gates)
        or receipt.get("online_solver_calls") is not False
        or receipt.get("prospective_opened") is not False
        or receipt.get("sealed_opened") is not False
    ):
        raise ValueError("relabel-pilot receipt does not unlock the paired bank")
    for relative, record in files.items():
        r0_trainer._verify_record(root, record, relative)
    if sha256_file(path) != expected_sha256:
        raise ValueError("relabel-pilot packet changed while it was validated")


def _validate_nested_relabel_receipt(
    value: Any, *, center: int, expected_hash: str
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("paired-bank source solver receipt is not a mapping")
    payload = dict(value)
    observed_hash = payload.pop("canonical_payload_sha256", None)
    if observed_hash != expected_hash or observed_hash != _canonical_sha256(payload):
        raise ValueError("paired-bank source solver receipt hash differs")
    checks = payload.get("checks")
    if (
        payload.get("schema") != RELABEL_RECEIPT_SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "validation_succeeded"
        or payload.get("scientifically_usable") is not True
        or payload.get("error") is not None
        or payload.get("index_contract") != relabel_step_index(center).to_mapping()
        or not isinstance(checks, Mapping)
        or set(checks) != set(REQUIRED_RECEIPT_CHECKS)
        or any(checks[name] is not True for name in REQUIRED_RECEIPT_CHECKS)
        or not isinstance(payload.get("details"), Mapping)
        or payload.get("nu_tilde_acceptance_gate") is not False
        or payload.get("prospective_opened") is not False
        or payload.get("sealed_opened") is not False
    ):
        raise ValueError("paired-bank source solver receipt is not successful")


def _load_paired_bank(
    root: Path,
    *,
    expected_final_sha256: str,
    pilot_final_manifest_path: Path,
    expected_pilot_final_sha256: str,
    expected_coordinates: np.ndarray,
    train_states: np.ndarray,
    train_frame_indices: np.ndarray,
    normalization: NACANormalization,
) -> VerifiedPairedBank:
    """Load the separately generated train-only bank without invoking SU2."""

    expected_coordinates = np.asarray(expected_coordinates)
    train_states = np.asarray(train_states)
    train_frame_indices = np.asarray(train_frame_indices)
    if (
        expected_coordinates.ndim != 2
        or expected_coordinates.shape[1] != 2
        or not np.all(np.isfinite(expected_coordinates))
    ):
        raise ValueError("expected native coordinates differ")
    expected_num_nodes = int(expected_coordinates.shape[0])
    if (
        train_states.dtype != np.float64
        or train_states.shape != (240, expected_num_nodes, 5)
        or train_frame_indices.dtype != np.int64
        or not np.array_equal(train_frame_indices, np.arange(955, 1195, dtype=np.int64))
        or not np.all(np.isfinite(train_states))
    ):
        raise ValueError("raw train history differs from the paired-bank contract")
    expected_final_sha256 = _require_sha256(
        expected_final_sha256, "paired-bank final-manifest SHA256"
    )
    expected_pilot_final_sha256 = _require_sha256(
        expected_pilot_final_sha256, "pilot final-manifest SHA256"
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("paired-bank packet is absent or aliased")
    bank_root = root.resolve()
    final_path = bank_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("paired-bank final manifest is absent or aliased")
    final_bytes = final_path.read_bytes()
    if sha256(final_bytes).hexdigest() != expected_final_sha256:
        raise ValueError("paired-bank final-manifest SHA256 differs")
    final = json.loads(final_bytes)
    expected_files = {
        PAIRED_BANK_METADATA_FILE,
        PAIRED_BANK_ARRAY_FILE,
        *(f"case_receipts/case_{ordinal:05d}.json" for ordinal in range(476)),
    }
    observed_files = {
        candidate.resolve().relative_to(bank_root).as_posix()
        for candidate in bank_root.rglob("*")
        if candidate.is_file() and candidate.name != "final_hash_manifest.json"
    }
    if (
        not isinstance(final, dict)
        or final.get("schema") != PAIRED_BANK_FINAL_SCHEMA
        or final.get("experiment_id") != EXPERIMENT_ID
        or final.get("packet_role") != "paired_query_bank"
        or final.get("status") != "complete"
        or final.get("extension_contract_sha256") != EXTENSION_CONTRACT_SHA256
        or final.get("preregistration_sha256") != EXTENSION_PREREGISTRATION_SHA256
        or final.get("dataset_final_hash_manifest_sha256") != DATASET_FINAL_SHA256
        or final.get("pilot_final_hash_manifest_sha256") != expected_pilot_final_sha256
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or final.get("online_solver_calls") is not False
        or set(final.get("files", {})) != expected_files
        or observed_files != set(final.get("files", {}))
    ):
        raise ValueError("paired-bank final manifest differs")
    _validate_relabel_pilot_final_manifest(
        pilot_final_manifest_path,
        expected_sha256=expected_pilot_final_sha256,
    )
    for name in final["files"]:
        r0_trainer._verify_record(bank_root, final["files"][name], name)
    metadata = r0_trainer._load_self_hashed_snapshot(
        bank_root,
        final["files"][PAIRED_BANK_METADATA_FILE],
        PAIRED_BANK_METADATA_FILE,
        PAIRED_BANK_SCHEMA,
    )
    if (
        metadata.get("status") != "complete"
        or metadata.get("scientifically_usable") is not True
        or metadata.get("error") is not None
        or metadata.get("experiment_id") != EXPERIMENT_ID
        or metadata.get("packet_role") != "paired_query_bank"
        or metadata.get("extension_contract_sha256") != EXTENSION_CONTRACT_SHA256
        or metadata.get("preregistration_sha256") != EXTENSION_PREREGISTRATION_SHA256
        or metadata.get("dataset_final_hash_manifest_sha256") != DATASET_FINAL_SHA256
        or metadata.get("pilot_final_hash_manifest_sha256")
        != expected_pilot_final_sha256
        or metadata.get("population_role") != "train_only"
        or metadata.get("centers_inclusive") != [956, 1193]
        or metadata.get("center_count") != 238
        or metadata.get("base_seed") != PAIRED_BANK_BASE_SEED
        or metadata.get("directions_per_center") != 1
        or metadata.get("antithetic_signs") != [-1, 1]
        or metadata.get("input_count") != 476
        or metadata.get("registered_solver_call_count") != 476
        or metadata.get("attempted_solver_call_count") != 476
        or metadata.get("completed_case_receipt_count") != 476
        or metadata.get("all_solver_cases_succeeded") is not True
        or metadata.get("pilot_gate_passed") is not True
        or metadata.get("authority_immutable") is not True
        or metadata.get("offline_training_label_solver_calls") is not True
        or metadata.get("prospective_opened") is not False
        or metadata.get("sealed_opened") is not False
        or metadata.get("online_solver_calls") is not False
    ):
        raise ValueError("paired-bank metadata differs")
    pilot_validation = metadata.get("pilot_validation")
    expected_gates = {
        "all_registered_cases",
        "all_case_receipts",
        "realized_displacement_scale",
        "zero_control",
        "repeatability",
        "auxiliary_invariance",
        "response_separation",
        "authority_immutable",
    }
    if (
        not isinstance(pilot_validation, Mapping)
        or pilot_validation.get("status") != "pilot_succeeded"
        or pilot_validation.get("scientifically_usable") is not True
        or pilot_validation.get("paired_query_bank_unlocked") is not True
        or pilot_validation.get("registered_case_count") != 11
        or pilot_validation.get("case_statuses") != ["validation_succeeded"] * 11
        or pilot_validation.get("final_hash_manifest_sha256")
        != expected_pilot_final_sha256
        or pilot_validation.get("immutable_after_bank") is not True
        or not isinstance(pilot_validation.get("gates"), Mapping)
        or set(pilot_validation["gates"]) != expected_gates
        or any(pilot_validation["gates"][name] is not True for name in expected_gates)
    ):
        raise ValueError("paired-bank pilot validation differs")
    if metadata.get("state_schema") != {
        "dynamic_fields": ["Density", "Momentum_x", "Momentum_y", "Energy", "Nu_Tilde"],
        "history_slot_order": ["previous", "current"],
        "history_indices": "center_minus_1_and_center",
        "expected_output_index": "center_plus_1",
        "state_coordinate_system": (
            "raw_physical_conservative_variables_model_facing_float32"
        ),
        "direction_coordinate_system": "state_normalized",
        "coordinate_row_order": "native_su2_restart_row_order",
        "coordinate_dtype": "float32",
        "state_axes": ["sign", "center", "node", "field"],
        "direction_axes": ["center", "history_slot", "node", "field"],
    }:
        raise ValueError("paired-bank raw-state schema differs")
    direction_generation = metadata.get("direction_generation")
    redraw_log = (
        direction_generation.get("redraw_log")
        if isinstance(direction_generation, Mapping)
        else None
    )
    pilot_proof = (
        direction_generation.get("pilot_direction_proof")
        if isinstance(direction_generation, Mapping)
        else None
    )
    if (
        not isinstance(direction_generation, Mapping)
        or direction_generation.get("library") != "torch"
        or direction_generation.get("device") != "cpu"
        or direction_generation.get("generator") != "torch.Generator.manual_seed"
        or direction_generation.get("distribution_call") != "torch.randn"
        or direction_generation.get("dtype") != "float32"
        or direction_generation.get("draw_shape") != [2, expected_num_nodes, 5]
        or direction_generation.get("stream_order")
        != "all_238_primary_draws_then_ascending_center_redraws"
        or direction_generation.get("primary_draw_count") != 238
        or not isinstance(direction_generation.get("total_draw_count"), int)
        or not all(
            isinstance(direction_generation.get(name), str)
            and len(direction_generation[name]) == 64
            for name in (
                "initial_generator_state_sha256",
                "after_primary_generator_state_sha256",
                "final_generator_state_sha256",
            )
        )
        or direction_generation.get("redraw_policy")
        != "only_nonpositive_density_or_ideal_gas_pressure_before_any_solver_call"
        or direction_generation.get("redraw_limit_per_center") != 10_000
        or not isinstance(redraw_log, list)
        or direction_generation.get("redrawn_center_count") != len(redraw_log)
        or direction_generation.get("nu_tilde_acceptance_gate") is not False
        or not isinstance(pilot_proof, Mapping)
        or pilot_proof.get("passed") is not True
        or not isinstance(pilot_proof.get("records"), list)
        or len(pilot_proof["records"]) != 3
    ):
        raise ValueError("paired-bank deterministic direction contract differs")
    realized = metadata.get("realized_displacement")
    if (
        not isinstance(realized, Mapping)
        or realized.get("registered_per_field_state_normalized_std")
        != PAIRED_BANK_REGISTERED_STD.tolist()
        or realized.get("required_ratio_inclusive") != list(PAIRED_BANK_RATIO_RANGE)
        or realized.get("all_476_inputs_passed") is not True
        or not isinstance(realized.get("per_field_ratio_min"), list)
        or not isinstance(realized.get("per_field_ratio_max"), list)
        or len(realized["per_field_ratio_min"]) != 5
        or len(realized["per_field_ratio_max"]) != 5
        or any(
            not math.isfinite(float(value))
            or float(value) < PAIRED_BANK_RATIO_RANGE[0]
            or float(value) > PAIRED_BANK_RATIO_RANGE[1]
            for value in (
                *realized["per_field_ratio_min"],
                *realized["per_field_ratio_max"],
            )
        )
    ):
        raise ValueError("paired-bank realized displacement differs")
    expected_schedule_sha256 = sha256(
        json.dumps(
            [[center, sign] for center in range(956, 1194) for sign in (-1, 1)],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    if metadata.get("call_schedule") != {
        "order": "ascending_center_then_sign_minus_plus",
        "canonical_sha256": expected_schedule_sha256,
        "failure_policy": "stop_without_drop_or_replacement",
    }:
        raise ValueError("paired-bank call schedule differs")
    arrays_record = metadata.get("arrays")
    if (
        not isinstance(arrays_record, Mapping)
        or arrays_record.get("file") != final["files"][PAIRED_BANK_ARRAY_FILE]
        or set(arrays_record.get("members", {})) != set(PAIRED_BANK_ARRAY_KEYS)
    ):
        raise ValueError("paired-bank array registration differs")
    archive_bytes = r0_trainer._read_record_bytes(
        bank_root, final["files"][PAIRED_BANK_ARRAY_FILE], PAIRED_BANK_ARRAY_FILE
    )
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(PAIRED_BANK_ARRAY_KEYS):
            raise ValueError("paired-bank NPZ members differ")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    members = arrays_record["members"]
    for name, value in arrays.items():
        expected = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "array_sha256": _array_sha256(value),
        }
        if members.get(name) != expected:
            raise ValueError(f"paired-bank member record differs: {name}")
    if arrays["centers"].dtype != np.int64 or not np.array_equal(
        arrays["centers"], np.arange(956, 1194, dtype=np.int64)
    ):
        raise ValueError("paired-bank centers differ")
    if arrays["signs"].dtype != np.int8 or not np.array_equal(
        arrays["signs"], np.array([-1, 1], dtype=np.int8)
    ):
        raise ValueError("paired-bank signs differ")
    expected_coordinates32 = np.asarray(expected_coordinates, dtype=np.float32)
    if (
        arrays["coordinates"].dtype != np.float32
        or arrays["coordinates"].shape != (expected_num_nodes, 2)
        or not np.array_equal(
            arrays["coordinates"].view(np.uint32),
            expected_coordinates32.view(np.uint32),
        )
    ):
        raise ValueError("paired-bank native coordinate row order differs")
    direction_shape = (238, 2, expected_num_nodes, 5)
    if (
        arrays["normalized_directions"].dtype != np.float32
        or arrays["normalized_directions"].shape != direction_shape
        or not np.all(np.isfinite(arrays["normalized_directions"]))
        or arrays["direction_draw_indices"].dtype != np.int64
        or arrays["direction_draw_indices"].shape != (238,)
        or arrays["redraw_counts"].dtype != np.int32
        or arrays["redraw_counts"].shape != (238,)
        or np.any(arrays["redraw_counts"] < 0)
    ):
        raise ValueError("paired-bank direction arrays differ")
    total_draw_count = direction_generation["total_draw_count"]
    if (
        total_draw_count != 238 + int(np.sum(arrays["redraw_counts"]))
        or np.any(arrays["direction_draw_indices"] < 0)
        or np.any(arrays["direction_draw_indices"] >= total_draw_count)
        or np.unique(arrays["direction_draw_indices"]).size != 238
        or np.any(
            (arrays["redraw_counts"] == 0)
            & (arrays["direction_draw_indices"] != np.arange(238, dtype=np.int64))
        )
        or np.any(
            (arrays["redraw_counts"] > 0) & (arrays["direction_draw_indices"] < 238)
        )
    ):
        raise ValueError("paired-bank redraw accounting differs")
    # The stored directions are the immutable scientific inputs.  PyTorch CPU
    # normal sampling and serialized generator states are not bitwise portable
    # across platforms/builds, so a training host must not regenerate this
    # stream.  Packet/member hashes, draw accounting, and the bitwise pilot
    # proof below bind the stored directions to the preregistered generation.
    proof_by_center = {
        record.get("center"): record for record in pilot_proof["records"]
    }
    if set(proof_by_center) != set(PAIRED_BANK_PILOT_CENTERS):
        raise ValueError("paired-bank pilot direction proof centers differ")
    for center in PAIRED_BANK_PILOT_CENTERS:
        offset = center - 956
        record = proof_by_center[center]
        direction_hash = _array_sha256(arrays["normalized_directions"][offset])
        if (
            record.get("bank_center_offset") != offset
            or record.get("bank_draw_index")
            != int(arrays["direction_draw_indices"][offset])
            or record.get("redraw_count") != int(arrays["redraw_counts"][offset])
            or record.get("bank_direction_array_sha256") != direction_hash
            or record.get("pilot_direction_array_sha256") != direction_hash
            or record.get("bitwise_equal") is not True
            or record.get("used_primary_draw_without_redraw") is not True
        ):
            raise ValueError("paired-bank pilot direction proof differs")
    expected_state_shape = (2, 238, expected_num_nodes, 5)
    for name in ("displaced_previous", "displaced_current", "solver_future"):
        value = arrays[name]
        if (
            value.dtype != np.float32
            or value.shape != expected_state_shape
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"paired-bank state member differs: {name}")
        value.setflags(write=False)
    clean_history = torch.from_numpy(train_states.astype(np.float32, copy=False))
    state_scale = torch.tensor(normalization.state_scale, dtype=torch.float32).reshape(
        1, 1, -1
    )
    for center_offset in range(238):
        clean = clean_history[center_offset : center_offset + 2]
        direction = torch.from_numpy(arrays["normalized_directions"][center_offset])
        for sign_offset, sign in enumerate((-1, 1)):
            expected_displaced = clean + state_scale * (float(sign) * direction)
            observed_displaced = np.stack(
                (
                    arrays["displaced_previous"][sign_offset, center_offset],
                    arrays["displaced_current"][sign_offset, center_offset],
                ),
                axis=0,
            )
            if not np.array_equal(
                expected_displaced.numpy().view(np.uint32),
                observed_displaced.view(np.uint32),
            ):
                raise ValueError("paired-bank raw displaced history differs")
    case_receipts = metadata.get("case_receipts")
    if not isinstance(case_receipts, list) or len(case_receipts) != 476:
        raise ValueError("paired-bank case receipt index differs")
    for ordinal, case_record in enumerate(case_receipts):
        center = 956 + ordinal // 2
        sign = -1 if ordinal % 2 == 0 else 1
        relative = f"case_receipts/case_{ordinal:05d}.json"
        if (
            not isinstance(case_record, Mapping)
            or case_record.get("relative_path") != relative
            or case_record.get("ordinal") != ordinal
            or case_record.get("center") != center
            or case_record.get("sign") != sign
            or case_record.get("status") != "validation_succeeded"
            or case_record.get("scientifically_usable") is not True
        ):
            raise ValueError("paired-bank case schedule provenance differs")
        wrapper = r0_trainer._load_self_hashed_snapshot(
            bank_root,
            final["files"][relative],
            relative,
            PAIRED_BANK_CASE_SCHEMA,
        )
        expected_case = {
            "ordinal": ordinal,
            "center": center,
            "sign": sign,
            "center_offset": ordinal // 2,
            "sign_offset": ordinal % 2,
            "history_indices": [center - 1, center],
            "expected_output_index": center + 1,
        }
        if (
            wrapper.get("experiment_id") != EXPERIMENT_ID
            or wrapper.get("status") != "validation_succeeded"
            or wrapper.get("scientifically_usable") is not True
            or wrapper.get("case") != expected_case
            or wrapper.get("source_solver_receipt_schema") != RELABEL_RECEIPT_SCHEMA
            or wrapper.get("offline_training_label_solver_call") is not True
            or wrapper.get("online_solver_calls") is not False
            or wrapper.get("nu_tilde_acceptance_gate") is not False
            or wrapper.get("prospective_opened") is not False
            or wrapper.get("sealed_opened") is not False
            or wrapper.get("canonical_payload_sha256")
            != case_record.get("canonical_payload_sha256")
            or wrapper.get("source_solver_receipt_canonical_payload_sha256")
            != case_record.get("source_solver_receipt_canonical_payload_sha256")
        ):
            raise ValueError("paired-bank case receipt differs")
        center_offset = ordinal // 2
        sign_offset = ordinal % 2
        expected_slices = {
            "normalized_direction": _array_sha256(
                arrays["normalized_directions"][center_offset]
            ),
            "displaced_previous": _array_sha256(
                arrays["displaced_previous"][sign_offset, center_offset]
            ),
            "displaced_current": _array_sha256(
                arrays["displaced_current"][sign_offset, center_offset]
            ),
            "solver_future": _array_sha256(
                arrays["solver_future"][sign_offset, center_offset]
            ),
        }
        slices = wrapper.get("array_slices")
        if not isinstance(slices, Mapping) or set(slices) != set(expected_slices):
            raise ValueError("paired-bank case array provenance differs")
        for name, expected_hash in expected_slices.items():
            if (
                not isinstance(slices[name], Mapping)
                or slices[name].get("array_sha256") != expected_hash
            ):
                raise ValueError("paired-bank case array slice hash differs")
        direction_record = wrapper.get("direction")
        if (
            not isinstance(direction_record, Mapping)
            or direction_record.get("source_draw_index")
            != int(arrays["direction_draw_indices"][center_offset])
            or direction_record.get("redraw_count")
            != int(arrays["redraw_counts"][center_offset])
            or direction_record.get("redraw_occurred_before_any_solver_execution")
            is not bool(arrays["redraw_counts"][center_offset] > 0)
        ):
            raise ValueError("paired-bank case direction provenance differs")
        _validate_nested_relabel_receipt(
            wrapper.get("source_solver_receipt"),
            center=center,
            expected_hash=wrapper["source_solver_receipt_canonical_payload_sha256"],
        )
    for name in (
        "coordinates",
        "normalized_directions",
        "direction_draw_indices",
        "redraw_counts",
    ):
        arrays[name].setflags(write=False)
    arrays["centers"].setflags(write=False)
    arrays["signs"].setflags(write=False)
    for name in final["files"]:
        r0_trainer._verify_record(bank_root, final["files"][name], name)
    if sha256_file(final_path) != expected_final_sha256:
        raise ValueError("paired-bank packet changed while loading")
    return VerifiedPairedBank(
        centers=arrays["centers"],
        signs=arrays["signs"],
        coordinates=arrays["coordinates"],
        normalized_directions=arrays["normalized_directions"],
        direction_draw_indices=arrays["direction_draw_indices"],
        redraw_counts=arrays["redraw_counts"],
        displaced_previous=arrays["displaced_previous"],
        displaced_current=arrays["displaced_current"],
        solver_future=arrays["solver_future"],
        final_manifest_sha256=expected_final_sha256,
        pilot_final_manifest_sha256=expected_pilot_final_sha256,
        file_records={name: dict(record) for name, record in final["files"].items()},
        root=bank_root,
        pilot_final_manifest_path=pilot_final_manifest_path.resolve(),
    )


def _reverify_paired_bank(bank: VerifiedPairedBank) -> None:
    observed_files = {
        path.resolve().relative_to(bank.root.resolve()).as_posix()
        for path in bank.root.rglob("*")
        if path.is_file() and path.name != "final_hash_manifest.json"
    }
    if bank.root.is_symlink() or observed_files != set(bank.file_records):
        raise ValueError("verified paired-bank packet changed")
    for name, record in bank.file_records.items():
        r0_trainer._verify_record(bank.root, record, name)
    if (
        sha256_file(bank.root / "final_hash_manifest.json")
        != bank.final_manifest_sha256
    ):
        raise ValueError("verified paired-bank final manifest changed")
    if (
        bank.pilot_final_manifest_path.is_symlink()
        or not bank.pilot_final_manifest_path.is_file()
        or sha256_file(bank.pilot_final_manifest_path)
        != bank.pilot_final_manifest_sha256
    ):
        raise ValueError("verified relabel-pilot final manifest changed")


class _CallLedger:
    """Hash the call schedule while keeping explicit call and sample totals."""

    def __init__(self) -> None:
        self._digest = sha256()
        self.counts: dict[str, int] = {
            "training_model_calls": 0,
            "training_model_sample_calls": 0,
            "prefix_model_calls": 0,
            "prefix_model_sample_calls": 0,
            "development_online_model_calls": 0,
            "development_online_model_sample_calls": 0,
            "development_ema_model_calls": 0,
            "development_ema_model_sample_calls": 0,
        }

    def record(
        self,
        *,
        counter: str,
        kind: str,
        epoch: int,
        optimizer_step: int,
        centers: Sequence[int],
        gradient_bearing: bool,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        sample_counter = counter.removesuffix("_calls") + "_sample_calls"
        if counter not in self.counts or sample_counter not in self.counts:
            raise ValueError("unregistered model-call counter")
        event = {
            "counter": counter,
            "kind": kind,
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "centers": [int(value) for value in centers],
            "sample_count": len(centers),
            "gradient_bearing": gradient_bearing,
            "detail": dict(detail or {}),
        }
        encoded = json.dumps(
            event, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._digest.update(encoded)
        self._digest.update(b"\n")
        self.counts[counter] += 1
        self.counts[sample_counter] += len(centers)

    @property
    def event_count(self) -> int:
        return sum(
            value
            for key, value in self.counts.items()
            if key.endswith("_calls") and not key.endswith("sample_calls")
        )

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True)
class ExtensionStepDiagnostics:
    loss: float
    gradient_norm_before_clip: float
    requested_depth: int | None
    realized_depths: tuple[int, ...]
    refiner_timesteps: tuple[int, ...]
    paired_sign: int | None
    prefix_displacement_sum_squared: Mapping[int, tuple[float, ...]]
    prefix_displacement_coordinate_count: Mapping[int, int]


def _presentation_orders(seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    return parent_trainer._presentation_orders(seed)


def _intervention_generator(
    seed: int, *, device: torch.device
) -> tuple[torch.Generator, dict[str, Any]]:
    derived_seed = 2_026_090_200_003 + seed * 1_000_003
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(derived_seed)
    return generator, {
        "algorithm": "torch_generator_explicit_state_v1",
        "seed_rule": "2026090200003 + extension_seed * 1000003",
        "derived_seed": derived_seed,
        "device": str(generator_device),
        "initial_generator_state_sha256": _generator_state_sha256(generator),
    }


def _depth_generator(seed: int) -> tuple[torch.Generator, dict[str, Any]]:
    derived_seed = 2_026_090_210_003 + seed * 1_000_003
    generator = torch.Generator().manual_seed(derived_seed)
    return generator, {
        "algorithm": "torch_cpu_generator_explicit_state_v1",
        "seed_rule": "2026090210003 + extension_seed * 1000003",
        "derived_seed": derived_seed,
        "initial_generator_state_sha256": _generator_state_sha256(generator),
    }


def _gather_batch(
    dataset: NACABDF2Dataset, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return parent_trainer._gather_batch(dataset, indices)


def _clean_state(dataset: NACABDF2Dataset, frame_index: int) -> torch.Tensor:
    try:
        position = dataset.frame_indices.index(int(frame_index))
    except ValueError as error:
        raise ValueError("clean state escapes the open dataset role") from error
    return torch.from_numpy(
        np.array(dataset.states[position], dtype=np.float32, copy=True)
    )


def _clean_sequence_batch(
    dataset: NACABDF2Dataset, centers: Sequence[int], depth: int
) -> torch.Tensor:
    sequences = []
    for center in centers:
        frames = range(int(center) - depth - 1, int(center) + 2)
        sequences.append(
            torch.stack([_clean_state(dataset, frame) for frame in frames])
        )
    result = torch.stack(sequences)
    if result.shape[1] != depth + 3:
        raise AssertionError("clean prefix sequence alignment differs")
    return result


def _geometry_for(
    geometry_batch: Mapping[str, torch.Tensor], size: int
) -> dict[str, torch.Tensor]:
    return r0_trainer._geometry_slice(geometry_batch, size)


def _fourier_for(
    fourier_tensors: tuple[torch.Tensor, ...], size: int
) -> tuple[torch.Tensor, ...]:
    return r0_trainer._fourier_slice(fourier_tensors, size)


def _baseline_call(
    *,
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    centers: Sequence[int],
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    ledger: _CallLedger,
    counter: str,
    kind: str,
    epoch: int,
    optimizer_step: int,
    gradient_bearing: bool,
    detail: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    size = int(previous.shape[0])
    ledger.record(
        counter=counter,
        kind=kind,
        epoch=epoch,
        optimizer_step=optimizer_step,
        centers=centers,
        gradient_bearing=gradient_bearing,
        detail=detail,
    )
    return predict_normalized_residual(
        model,  # type: ignore[arg-type]
        previous,
        current,
        _geometry_for(geometry_batch, size),
        normalization,
        fourier_tensors=_fourier_for(fourier_tensors, size),
    )


def _refiner_call(
    *,
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    candidate: torch.Tensor,
    timesteps: torch.Tensor,
    centers: Sequence[int],
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    ledger: _CallLedger,
    counter: str,
    kind: str,
    epoch: int,
    optimizer_step: int,
    gradient_bearing: bool,
) -> torch.Tensor:
    size = int(previous.shape[0])
    ledger.record(
        counter=counter,
        kind=kind,
        epoch=epoch,
        optimizer_step=optimizer_step,
        centers=centers,
        gradient_bearing=gradient_bearing,
        detail={"timesteps": [int(value) for value in timesteps.tolist()]},
    )
    # Mixed per-example scheduler indices use the same shared refiner call.
    features = build_refiner_input(
        previous,
        current,
        candidate,
        timesteps,
        _geometry_for(geometry_batch, size),
        normalization,
    )
    return model(
        features,
        _geometry_for(geometry_batch, size),
        fourier_tensors=_fourier_for(fourier_tensors, size),
    )


def _draw_requested_depth(arm: str, epoch: int, generator: torch.Generator) -> int:
    if arm == "MP_PDE_PUSHFORWARD_M01":
        return sample_mp_pde_prefix_depth(epoch, generator)
    if arm == "CURRICULUM_EMA_PUSHFORWARD_K13":
        return sample_curriculum_requested_prefix_depth(epoch, generator)
    raise ValueError("requested depth applies only to pushforward arms")


def _optimizer_step_with_ema(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    clip_norm: float = 1.0,
) -> float:
    """Clip, update online parameters, and only then update the EMA copy."""

    for parameter in model.parameters():
        if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
            raise FloatingPointError("extension training gradient became nonfinite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("extension gradient norm became nonfinite")
    optimizer.step()
    for parameter in model.parameters():
        if not torch.all(torch.isfinite(parameter)):
            raise FloatingPointError("extension model parameter became nonfinite")
    if ema is not None:
        ema.update(model)
    return float(gradient_norm.detach().cpu())


def _prefix_displacement_statistics(
    *,
    presented_current: torch.Tensor,
    aligned_clean_current: torch.Tensor,
    normalization: NACANormalization,
    depth: int,
) -> tuple[dict[int, tuple[float, ...]], dict[int, int]]:
    difference = normalization.normalize_state(
        presented_current
    ) - normalization.normalize_state(aligned_clean_current)
    squared = torch.sum(
        torch.square(difference.detach().to(torch.float64).cpu()), dim=(0, 1)
    )
    return {depth: tuple(float(value) for value in squared)}, {
        depth: int(difference.shape[0] * difference.shape[1])
    }


def _merge_prefix_statistics(
    target_sum: dict[int, list[float]],
    target_count: dict[int, int],
    values: Mapping[int, tuple[float, ...]],
    counts: Mapping[int, int],
) -> None:
    for depth, field_values in values.items():
        accumulator = target_sum.setdefault(depth, [0.0] * 5)
        for field, value in enumerate(field_values):
            accumulator[field] += float(value)
        target_count[depth] = target_count.get(depth, 0) + int(counts[depth])


def _step_extension_batch(
    *,
    arm: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    train_dataset: NACABDF2Dataset,
    previous: torch.Tensor,
    current: torch.Tensor,
    clean_target: torch.Tensor,
    centers: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    microbatch_size: int,
    epoch: int,
    optimizer_step: int,
    depth_generator: torch.Generator,
    intervention_generator: torch.Generator,
    scheduler: FourStepVPredictionScheduler | None,
    paired_bank: VerifiedPairedBank | None,
    ledger: _CallLedger,
    requested_depth_override: int | None = None,
) -> ExtensionStepDiagnostics:
    """Execute one effective batch with exact sample weighting."""

    if arm not in ARMS:
        raise ValueError("unknown extension arm")
    group_size = int(previous.shape[0])
    if group_size not in (2, 4) or centers.shape != (group_size,):
        raise ValueError("effective training group must contain two or four samples")
    if microbatch_size not in (1, 2, 4):
        raise ValueError("microbatch size differs from the frozen resource choices")
    if arm in EMA_ARMS and ema is None:
        raise ValueError("EMA arm lacks its EMA state")
    if arm not in EMA_ARMS and ema is not None:
        raise ValueError("non-EMA arm received an EMA state")
    if arm in PAIRED_ARMS and paired_bank is None:
        raise ValueError("paired arm lacks the frozen bank")
    if arm == "PCNO_PDEREFINER_K3_VPRED" and scheduler is None:
        raise ValueError("PDE-Refiner arm lacks its scheduler")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    requested_depth: int | None = None
    realized_depths: tuple[int, ...] = ()
    refiner_timesteps: tuple[int, ...] = ()
    paired_sign: int | None = None
    displacement_sum: dict[int, list[float]] = {}
    displacement_count: dict[int, int] = {}

    if arm in PUSHFORWARD_ARMS:
        requested_depth = (
            _draw_requested_depth(arm, epoch, depth_generator)
            if requested_depth_override is None
            else int(requested_depth_override)
        )
        if requested_depth < 0 or requested_depth > (
            1 if arm == "MP_PDE_PUSHFORWARD_M01" else 3
        ):
            raise ValueError("requested pushforward depth is outside the arm contract")
        center_values = [int(value) for value in centers.tolist()]
        realized_depths = realize_prefix_depths(requested_depth, center_values)
        prefix_source = model if arm == "MP_PDE_PUSHFORWARD_M01" else ema.model  # type: ignore[union-attr]
        groups = group_examples_by_prefix_depth(realized_depths)
        for depth, positions in groups.items():
            for offset in range(0, len(positions), microbatch_size):
                selected = positions[offset : offset + microbatch_size]
                selected_centers = [center_values[index] for index in selected]
                clean_sequence = _clean_sequence_batch(
                    train_dataset, selected_centers, depth
                ).to(previous.device)
                presentation = make_detached_prefix_presentation(
                    prefix_source,
                    clean_sequence,
                    depth,
                    _geometry_for(geometry_batch, len(selected)),
                    normalization,
                    fourier_tensors=_fourier_for(fourier_tensors, len(selected)),
                )
                for prefix_index in range(depth):
                    ledger.record(
                        counter="prefix_model_calls",
                        kind=(
                            "detached_online_prefix"
                            if arm == "MP_PDE_PUSHFORWARD_M01"
                            else "detached_ema_prefix"
                        ),
                        epoch=epoch,
                        optimizer_step=optimizer_step,
                        centers=selected_centers,
                        gradient_bearing=False,
                        detail={
                            "requested_depth": requested_depth,
                            "realized_depth": depth,
                            "prefix_index": prefix_index + 1,
                        },
                    )
                values, counts = _prefix_displacement_statistics(
                    presented_current=presentation.current,
                    aligned_clean_current=presentation.aligned_clean_current,
                    normalization=normalization,
                    depth=depth,
                )
                _merge_prefix_statistics(
                    displacement_sum, displacement_count, values, counts
                )
                prediction = _baseline_call(
                    model=model,
                    previous=presentation.previous,
                    current=presentation.current,
                    centers=selected_centers,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier_tensors,
                    normalization=normalization,
                    ledger=ledger,
                    counter="training_model_calls",
                    kind="terminal_pushforward_gradient",
                    epoch=epoch,
                    optimizer_step=optimizer_step,
                    gradient_bearing=True,
                    detail={
                        "requested_depth": requested_depth,
                        "realized_depth": depth,
                    },
                )
                per_sample = torch.mean(
                    torch.square(prediction - presentation.target_normalized_residual),
                    dim=(1, 2),
                )
                loss = torch.sum(per_sample) / group_size
                if not torch.isfinite(loss):
                    raise FloatingPointError("pushforward loss became nonfinite")
                loss.backward()
                total_loss += float(torch.sum(per_sample.detach()).cpu()) / group_size
    elif arm in PAIRED_ARMS:
        assert paired_bank is not None
        paired_sign = paired_bank_sign_for_epoch(epoch)
        displaced_previous, displaced_current, solver_future = paired_bank.select(
            centers, paired_sign, device=previous.device
        )
        clean_future = torch.stack(
            [
                _clean_state(train_dataset, int(center) + 1)
                for center in centers.tolist()
            ]
        ).to(previous.device)
        target_kind = "recovery" if arm == "PAIRED_RECOVERY" else "dynamics_relabel"
        displaced = make_paired_displaced_presentation(
            displaced_previous,
            displaced_current,
            clean_future,
            solver_future,
            normalization,
            target_kind=target_kind,
        )
        for offset in range(0, group_size, microbatch_size):
            stop = min(offset + microbatch_size, group_size)
            center_chunk = [int(value) for value in centers[offset:stop].tolist()]
            clean_prediction = _baseline_call(
                model=model,
                previous=previous[offset:stop],
                current=current[offset:stop],
                centers=center_chunk,
                geometry_batch=geometry_batch,
                fourier_tensors=fourier_tensors,
                normalization=normalization,
                ledger=ledger,
                counter="training_model_calls",
                kind="paired_clean_gradient",
                epoch=epoch,
                optimizer_step=optimizer_step,
                gradient_bearing=True,
                detail={"sign": paired_sign},
            )
            displaced_prediction = _baseline_call(
                model=model,
                previous=displaced.previous[offset:stop],
                current=displaced.current[offset:stop],
                centers=center_chunk,
                geometry_batch=geometry_batch,
                fourier_tensors=fourier_tensors,
                normalization=normalization,
                ledger=ledger,
                counter="training_model_calls",
                kind=f"paired_{target_kind}_gradient",
                epoch=epoch,
                optimizer_step=optimizer_step,
                gradient_bearing=True,
                detail={"sign": paired_sign},
            )
            clean_per_sample = torch.mean(
                torch.square(clean_prediction - clean_target[offset:stop]), dim=(1, 2)
            )
            displaced_per_sample = torch.mean(
                torch.square(
                    displaced_prediction
                    - displaced.target_normalized_residual[offset:stop]
                ),
                dim=(1, 2),
            )
            clean_loss = torch.sum(clean_per_sample) / (stop - offset)
            displaced_loss = torch.sum(displaced_per_sample) / (stop - offset)
            component = paired_clean_displaced_objective(clean_loss, displaced_loss)
            loss = component * ((stop - offset) / group_size)
            if not torch.isfinite(loss):
                raise FloatingPointError("paired loss became nonfinite")
            loss.backward()
            total_loss += float(loss.detach().cpu())
    elif arm == "PCNO_PDEREFINER_K3_VPRED":
        assert scheduler is not None
        timesteps = sample_refiner_timesteps(
            group_size, generator=intervention_generator, device=previous.device
        )
        noise = torch.randn(
            clean_target.shape,
            dtype=clean_target.dtype,
            device=clean_target.device,
            generator=intervention_generator,
        )
        presentation = make_refiner_training_presentation(
            clean_target, noise, timesteps, scheduler
        )
        refiner_timesteps = tuple(int(value) for value in timesteps.tolist())
        for offset in range(0, group_size, microbatch_size):
            stop = min(offset + microbatch_size, group_size)
            center_chunk = [int(value) for value in centers[offset:stop].tolist()]
            prediction = _refiner_call(
                model=model,
                previous=previous[offset:stop],
                current=current[offset:stop],
                candidate=presentation.noised_candidate[offset:stop],
                timesteps=timesteps[offset:stop],
                centers=center_chunk,
                geometry_batch=geometry_batch,
                fourier_tensors=fourier_tensors,
                normalization=normalization,
                ledger=ledger,
                counter="training_model_calls",
                kind="refiner_v_prediction_gradient",
                epoch=epoch,
                optimizer_step=optimizer_step,
                gradient_bearing=True,
            )
            per_sample = torch.mean(
                torch.square(prediction - presentation.velocity_target[offset:stop]),
                dim=(1, 2),
            )
            loss = torch.sum(per_sample) / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError("refiner loss became nonfinite")
            loss.backward()
            total_loss += float(torch.sum(per_sample.detach()).cpu()) / group_size
    else:
        for offset in range(0, group_size, microbatch_size):
            stop = min(offset + microbatch_size, group_size)
            center_chunk = [int(value) for value in centers[offset:stop].tolist()]
            prediction = _baseline_call(
                model=model,
                previous=previous[offset:stop],
                current=current[offset:stop],
                centers=center_chunk,
                geometry_batch=geometry_batch,
                fourier_tensors=fourier_tensors,
                normalization=normalization,
                ledger=ledger,
                counter="training_model_calls",
                kind="clean_gradient",
                epoch=epoch,
                optimizer_step=optimizer_step,
                gradient_bearing=True,
            )
            per_sample = torch.mean(
                torch.square(prediction - clean_target[offset:stop]), dim=(1, 2)
            )
            loss = torch.sum(per_sample) / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError("clean loss became nonfinite")
            loss.backward()
            total_loss += float(torch.sum(per_sample.detach()).cpu()) / group_size

    gradient_norm = _optimizer_step_with_ema(
        model=model, optimizer=optimizer, ema=ema, clip_norm=1.0
    )
    return ExtensionStepDiagnostics(
        loss=total_loss,
        gradient_norm_before_clip=gradient_norm,
        requested_depth=requested_depth,
        realized_depths=realized_depths,
        refiner_timesteps=refiner_timesteps,
        paired_sign=paired_sign,
        prefix_displacement_sum_squared={
            depth: tuple(values) for depth, values in displacement_sum.items()
        },
        prefix_displacement_coordinate_count=dict(displacement_count),
    )


def _baseline_development_score(
    *,
    model: torch.nn.Module,
    dataset: NACABDF2Dataset,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    device: torch.device,
    microbatch_size: int,
    ledger: _CallLedger,
    counter: str,
    epoch: int,
) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(dataset), dtype=torch.int64)
        for start in range(0, len(dataset), 4):
            previous, current, target, centers = _gather_batch(
                dataset, all_indices[start : start + 4]
            )
            group_size = int(previous.shape[0])
            for offset in range(0, group_size, microbatch_size):
                stop = min(offset + microbatch_size, group_size)
                center_chunk = [int(value) for value in centers[offset:stop].tolist()]
                prediction = _baseline_call(
                    model=model,
                    previous=previous[offset:stop].to(device),
                    current=current[offset:stop].to(device),
                    centers=center_chunk,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier_tensors,
                    normalization=normalization,
                    ledger=ledger,
                    counter=counter,
                    kind="clean_one_step_selection",
                    epoch=epoch,
                    optimizer_step=0,
                    gradient_bearing=False,
                )
                error = prediction - target[offset:stop].to(device)
                per_sample = torch.sqrt(torch.mean(torch.square(error), dim=(1, 2)))
                per_sample = torch.where(
                    torch.isfinite(per_sample),
                    per_sample,
                    torch.full_like(per_sample, math.inf),
                )
                values.extend(float(value) for value in per_sample.cpu())
    if len(values) != len(dataset):
        raise ValueError("development score lacks registered transitions")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _refiner_development_score(
    *,
    model: torch.nn.Module,
    dataset: NACABDF2Dataset,
    geometry_batch: Mapping[str, torch.Tensor],
    fourier_tensors: tuple[torch.Tensor, ...],
    normalization: NACANormalization,
    scheduler: FourStepVPredictionScheduler,
    device: torch.device,
    microbatch_size: int,
    ledger: _CallLedger,
    counter: str,
    epoch: int,
    sampler_seed: int = 101,
) -> tuple[float, str]:
    """Fixed-seed four-call one-step score used for refiner selection."""

    if sampler_seed != 101:
        raise ValueError("refiner checkpoint selection requires sampler seed 101")
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(sampler_seed)
    tape_digest = sha256()
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(dataset), dtype=torch.int64)
        for start in range(0, len(dataset), 4):
            previous, current, target, centers = _gather_batch(
                dataset, all_indices[start : start + 4]
            )
            group_size = int(previous.shape[0])
            for offset in range(0, group_size, microbatch_size):
                stop = min(offset + microbatch_size, group_size)
                previous_chunk = previous[offset:stop].to(device)
                current_chunk = current[offset:stop].to(device)
                center_chunk = [int(value) for value in centers[offset:stop].tolist()]
                tape = sample_refiner_noise_tape(current_chunk, generator=generator)
                for value in (
                    tape.initial,
                    tape.reverse_t3,
                    tape.reverse_t2,
                    tape.reverse_t1,
                ):
                    tape_digest.update(_tensor_sha256(value).encode("ascii"))
                result = refined_recurrent_step(
                    model,
                    previous_chunk,
                    current_chunk,
                    _geometry_for(geometry_batch, stop - offset),
                    normalization,
                    scheduler,
                    tape,
                    fourier_tensors=_fourier_for(fourier_tensors, stop - offset),
                )
                if result.model_calls != 4:
                    raise ValueError("refiner development step did not use four calls")
                for timestep in (3, 2, 1, 0):
                    ledger.record(
                        counter=counter,
                        kind="refiner_fixed_seed_selection",
                        epoch=epoch,
                        optimizer_step=0,
                        centers=center_chunk,
                        gradient_bearing=False,
                        detail={"sampler_seed": 101, "timestep": timestep},
                    )
                error = result.final_normalized_residual - target[offset:stop].to(
                    device
                )
                per_sample = torch.sqrt(torch.mean(torch.square(error), dim=(1, 2)))
                per_sample = torch.where(
                    torch.isfinite(per_sample),
                    per_sample,
                    torch.full_like(per_sample, math.inf),
                )
                values.extend(float(value) for value in per_sample.cpu())
    if len(values) != len(dataset):
        raise ValueError("refiner development score lacks registered transitions")
    return (
        float(np.median(np.asarray(values, dtype=np.float64))),
        tape_digest.hexdigest(),
    )


def _ema_distance(
    model: torch.nn.Module, ema: ExponentialMovingAverage | None
) -> dict[str, float] | None:
    if ema is None:
        return None
    online = dict(model.named_parameters())
    averaged = dict(ema.model.named_parameters())
    if tuple(online) != tuple(averaged):
        raise ValueError("EMA parameter layout differs")
    difference_squared = 0.0
    online_squared = 0.0
    for name, value in online.items():
        target = averaged[name].to(value.device)
        difference_squared += float(
            torch.sum(
                torch.square(
                    value.detach().to(torch.float64) - target.detach().to(torch.float64)
                )
            ).cpu()
        )
        online_squared += float(
            torch.sum(torch.square(value.detach().to(torch.float64))).cpu()
        )
    distance = math.sqrt(difference_squared)
    return {
        "l2": distance,
        "relative_to_online_l2": distance / max(math.sqrt(online_squared), 1.0e-30),
    }


def _checkpoint_payload(
    *,
    record_kind: str,
    arm: str,
    seed: int,
    epoch: int,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer | None,
    learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    intervention_generator: torch.Generator,
    depth_generator: torch.Generator,
    bindings: Mapping[str, Any],
    best_epoch: int,
    best_score: float,
    model_only: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": TRAINING_SCHEMA,
        "record_kind": record_kind,
        **dict(bindings),
        "arm": arm,
        "seed": seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_development_score": _json_number(best_score),
        "checkpoint_selection_used_rollout": False,
        "model_config": model.model_config(),  # type: ignore[attr-defined]
        "online_model_state_dict": _clone_state_dict(model),
        "ema_state_dict": (
            {
                name: value.detach().cpu().clone()
                for name, value in ema.state_dict().items()
            }
            if ema is not None
            else None
        ),
        "ema_num_updates": int(ema.num_updates.item()) if ema is not None else None,
        "rng_state": {
            "intervention": intervention_generator.get_state().cpu(),
            "depth": depth_generator.get_state().cpu(),
        },
        "refiner_scheduler": (
            {"betas": list(FourStepVPredictionScheduler().betas)}
            if arm == "PCNO_PDEREFINER_K3_VPRED"
            else None
        ),
        "model_only": model_only,
    }
    if not model_only:
        if optimizer is None or learning_rate_scheduler is None:
            raise ValueError("resumable checkpoint lacks optimizer or scheduler")
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = learning_rate_scheduler.state_dict()
    return payload


def _restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer,
    learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler,
    intervention_generator: torch.Generator,
    depth_generator: torch.Generator,
) -> None:
    """Strict resumable-checkpoint restore seam, also used by CPU tests."""

    if (
        payload.get("schema") != TRAINING_SCHEMA
        or payload.get("arm") != arm
        or payload.get("seed") != seed
        or payload.get("model_only") is not False
        or not isinstance(payload.get("online_model_state_dict"), Mapping)
        or not isinstance(payload.get("optimizer_state_dict"), Mapping)
        or not isinstance(payload.get("scheduler_state_dict"), Mapping)
        or not isinstance(payload.get("rng_state"), Mapping)
    ):
        raise ValueError("extension checkpoint identity or resume state differs")
    if (ema is None) != (payload.get("ema_state_dict") is None):
        raise ValueError("extension checkpoint EMA inventory differs")
    model.load_state_dict(payload["online_model_state_dict"], strict=True)
    if ema is not None:
        ema.load_state_dict(payload["ema_state_dict"], strict=True)
        if int(ema.num_updates.item()) != payload.get("ema_num_updates"):
            raise ValueError("restored EMA update count differs")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    learning_rate_scheduler.load_state_dict(payload["scheduler_state_dict"])
    intervention_generator.set_state(payload["rng_state"]["intervention"].cpu())
    depth_generator.set_state(payload["rng_state"]["depth"].cpu())


def _new_staging_directory(output: Path) -> tuple[Path, Path]:
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError("extension output or its parent is aliased")
    final = output.resolve()
    if final.exists():
        raise FileExistsError(f"extension output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.incomplete-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()
    return staging, final


def _commit_staging_directory(staging: Path, final: Path) -> None:
    if final.exists() or staging.is_symlink() or not staging.is_dir():
        raise FileExistsError("atomic extension packet destination is unavailable")
    os.replace(staging, final)


def _write_final_hash_manifest(
    staging: Path,
    *,
    arm: str,
    seed: int,
    bindings: Mapping[str, Any],
    initial_model_state_sha256: str,
    presentation_schedule_sha256: str,
) -> None:
    if {path.name for path in staging.iterdir()} != TRAINING_FILES:
        raise ValueError("extension packet has missing or unexpected files")
    files: dict[str, Any] = {}
    for name in sorted(TRAINING_FILES):
        path = staging / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != staging:
            raise ValueError("extension packet contains a nonregular entry")
        files[name] = {
            "relative_path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(
        staging / "final_hash_manifest.json",
        {
            "schema": FINAL_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "arm": arm,
            "seed": seed,
            **dict(bindings),
            "paired_initialization_and_clean_order_where_architectures_allow": True,
            "initial_model_state_sha256": initial_model_state_sha256,
            "presentation_schedule_sha256": presentation_schedule_sha256,
            "files": files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
            "online_solver_calls": False,
        },
    )


def _common_bindings(
    *,
    arm: str,
    seed: int,
    dataset_manifest: Mapping[str, Any],
    dataset_final_sha256: str,
    calibration_metadata: Mapping[str, Any],
    calibration_final_sha256: str,
    paired_bank: VerifiedPairedBank | None,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "seed": seed,
        "extension_contract_sha256": EXTENSION_CONTRACT_SHA256,
        "extension_preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
        "inherited_r0_contract_sha256": R0_CONTRACT_SHA256,
        "inherited_successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
        "dataset_manifest_payload_sha256": dataset_manifest["canonical_payload_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_final_sha256,
        "successor_calibration_payload_sha256": calibration_metadata[
            "canonical_payload_sha256"
        ],
        "successor_calibration_final_hash_manifest_sha256": calibration_final_sha256,
        "paired_bank_final_hash_manifest_sha256": (
            paired_bank.final_manifest_sha256 if paired_bank is not None else None
        ),
        "relabel_pilot_final_hash_manifest_sha256": (
            paired_bank.pilot_final_manifest_sha256 if paired_bank is not None else None
        ),
        "source_set_sha256": SOURCE_SET_SHA256,
        "train_opened": True,
        "development_opened": True,
        "prospective_opened": False,
        "sealed_opened": False,
        "offline_training_label_solver_calls": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }


def _reverify_bound_inputs(
    *,
    arguments: argparse.Namespace,
    calibration_files: Mapping[str, Mapping[str, Any]],
    calibration_final_sha256: str,
    dataset_final_sha256: str,
    paired_bank: VerifiedPairedBank | None,
    resource_smoke_file_sha256: str | None = None,
    authorization_file_sha256: str | None = None,
) -> None:
    """Recheck every mutable input immediately before atomic publication."""

    load_extension_contract(
        arguments.extension_contract, arguments.extension_preregistration
    )
    exact_files = (
        (arguments.contract, R0_CONTRACT_SHA256, "R0 contract"),
        (
            arguments.successor_contract,
            SUCCESSOR_CONTRACT_SHA256,
            "successor contract",
        ),
        (
            arguments.successor_preregistration,
            SUCCESSOR_PREREGISTRATION_SHA256,
            "successor preregistration",
        ),
    )
    for path, expected, label in exact_files:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} changed during extension execution")
    parent_trainer._reverify_calibration(
        arguments.calibration_dir,
        calibration_files,
        calibration_final_sha256,
    )
    if (
        sha256_file(arguments.dataset_dir.resolve() / "final_hash_manifest.json")
        != dataset_final_sha256
    ):
        raise ValueError("dataset packet changed during extension execution")
    _verify_live_sources()
    if paired_bank is not None:
        _reverify_paired_bank(paired_bank)
    optional_inputs = (
        (
            arguments.resource_smoke_receipt,
            resource_smoke_file_sha256,
            "resource-smoke receipt",
        ),
        (
            arguments.authorization,
            authorization_file_sha256,
            "execution authorization",
        ),
    )
    for path, expected, label in optional_inputs:
        if expected is None:
            if path is not None:
                raise ValueError(f"unexpected {label} during input reverification")
            continue
        if path is None or path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} disappeared during extension execution")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} changed during extension execution")


def _validate_smoke_replay_bindings(
    smoke: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device_binding: Mapping[str, Any],
    initial_model_state_sha256: str,
    presentation_schedule_sha256: str,
    intervention_initial_state_sha256: str,
    depth_initial_state_sha256: str,
    expected_num_nodes: int,
) -> None:
    shared = tuple(common)
    rng = smoke.get("rng")
    if (
        any(smoke.get(key) != common.get(key) for key in shared)
        or smoke.get("runtime") != runtime
        or smoke.get("production_device_binding") != device_binding
        or smoke.get("initial_model_state_sha256") != initial_model_state_sha256
        or smoke.get("presentation_schedule_sha256") != presentation_schedule_sha256
        or smoke.get("full_resolution_num_nodes") != expected_num_nodes
        or not isinstance(rng, Mapping)
        or rng.get("intervention_initial_generator_state_sha256")
        != intervention_initial_state_sha256
        or rng.get("depth_initial_generator_state_sha256") != depth_initial_state_sha256
    ):
        raise PermissionError(
            "extension resource smoke does not replay the production bindings"
        )


def _validate_production_authorization(
    *,
    arguments: argparse.Namespace,
    common: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    if arguments.authorization is None or arguments.resource_smoke_receipt is None:
        raise PermissionError(
            "extension production training requires authorization and a resource smoke"
        )
    if arguments.resource_smoke_receipt.is_symlink():
        raise PermissionError("extension resource-smoke receipt is aliased")
    smoke, smoke_file_sha256 = r0_trainer._load_self_hashed_with_file_sha256(
        arguments.resource_smoke_receipt.resolve(), SMOKE_SCHEMA
    )
    numeric = (smoke.get("loss"), smoke.get("gradient_norm_before_clip"))
    call_cost = smoke.get("model_call_cost")
    expected_forced_depth = (
        1
        if arguments.arm == "MP_PDE_PUSHFORWARD_M01"
        else 3
        if arguments.arm == "CURRICULUM_EMA_PUSHFORWARD_K13"
        else None
    )
    if (
        any(smoke.get(key) != value for key, value in common.items())
        or smoke.get("status") != "succeeded"
        or smoke.get("effective_batch_size") != 4
        or smoke.get("forward_backward_and_adamw_step_completed") is not True
        or smoke.get("checkpoint_written") is not False
        or smoke.get("scientific_hyperparameters_changed") is not False
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        )
        or (smoke.get("microbatch_size"), smoke.get("gradient_accumulation_steps"))
        != (arguments.microbatch_size, arguments.gradient_accumulation_steps)
        or smoke.get("production_device_binding") != r0_trainer._device_binding(device)
        or smoke.get("smoke_forced_requested_depth") != expected_forced_depth
        or smoke.get("ema_updated_after_optimizer") is not (arguments.arm in EMA_ARMS)
        or not isinstance(call_cost, Mapping)
        or not isinstance(call_cost.get("training_model_calls"), int)
        or call_cost["training_model_calls"] < 1
        or (
            arguments.arm == "MP_PDE_PUSHFORWARD_M01"
            and call_cost.get("prefix_model_sample_calls", 0) < 3
        )
        or (
            arguments.arm == "CURRICULUM_EMA_PUSHFORWARD_K13"
            and call_cost.get("prefix_model_sample_calls", 0) < 6
        )
        or (
            arguments.arm == "PCNO_PDEREFINER_K3_VPRED"
            and (
                call_cost.get("development_ema_model_calls") != 4
                or call_cost.get("development_ema_model_sample_calls")
                != 4 * arguments.microbatch_size
            )
        )
    ):
        raise PermissionError("extension resource-smoke receipt is mismatched")
    if arguments.authorization.is_symlink():
        raise PermissionError("extension execution authorization is aliased")
    authorization, authorization_file_sha256 = (
        r0_trainer._load_self_hashed_with_file_sha256(
            arguments.authorization.resolve(), AUTHORIZATION_SCHEMA
        )
    )
    arms = authorization.get("authorized_arms")
    seeds = authorization.get("authorized_seeds")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("extension_contract_sha256") != EXTENSION_CONTRACT_SHA256
        or authorization.get("extension_preregistration_sha256")
        != EXTENSION_PREREGISTRATION_SHA256
        or authorization.get("dataset_final_hash_manifest_sha256")
        != DATASET_FINAL_SHA256
        or authorization.get("successor_calibration_final_hash_manifest_sha256")
        != SUCCESSOR_CALIBRATION_SHA256
        or authorization.get("paired_bank_final_hash_manifest_sha256")
        != common["paired_bank_final_hash_manifest_sha256"]
        or authorization.get("relabel_pilot_final_hash_manifest_sha256")
        != common["relabel_pilot_final_hash_manifest_sha256"]
        or authorization.get("source_set_sha256") != SOURCE_SET_SHA256
        or authorization.get("resource_smoke_payload_sha256")
        != smoke["canonical_payload_sha256"]
        or authorization.get("resource_smoke_file_sha256") != smoke_file_sha256
        or authorization.get("production_device_binding")
        != r0_trainer._device_binding(device)
        or authorization.get("allowed_actions")
        != ["train_pcno_naca0012_corrective_extension"]
        or not isinstance(arms, list)
        or len(arms) != len(set(arms))
        or any(arm not in ARMS for arm in arms)
        or arguments.arm not in arms
        or not isinstance(seeds, list)
        or len(seeds) != len(set(seeds))
        or any(seed not in SEEDS for seed in seeds)
        or arguments.seed not in seeds
        or authorization.get("extension_code_audited") is not True
        or authorization.get("extension_training_authorized") is not True
        or authorization.get("train_opened") is not True
        or authorization.get("development_opened") is not True
        or authorization.get("prospective_opened") is not False
        or authorization.get("sealed_opened") is not False
        or authorization.get("online_solver_calls") is not False
        or not isinstance(authorization.get("authorized_by"), str)
        or not authorization["authorized_by"].strip()
    ):
        raise PermissionError("extension execution authorization is mismatched")
    return smoke, smoke_file_sha256, authorization, authorization_file_sha256


def _build_model_state(
    *,
    arm: str,
    seed: int,
    extension: VerifiedExtensionContract,
    r0_contract: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    device: torch.device,
) -> tuple[torch.nn.Module, ExponentialMovingAverage | None, dict[str, Any]]:
    r0_trainer._seed_everything(seed)
    if arm == "PCNO_PDEREFINER_K3_VPRED":
        model = build_naca_pcno_refiner(extension, geometry).to(device)
    else:
        model = build_naca_pcno(r0_contract, geometry).to(device)  # type: ignore[arg-type]
        validate_naca_model_config(model.model_config(), r0_contract, geometry)  # type: ignore[arg-type]
    ema = (
        ExponentialMovingAverage(model, decay=0.995).to(device)
        if arm in EMA_ARMS
        else None
    )
    return model, ema, model.model_config()  # type: ignore[attr-defined]


def _resource_smoke(
    *,
    arguments: argparse.Namespace,
    extension: VerifiedExtensionContract,
    r0_contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_final_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    calibration_metadata: Mapping[str, Any],
    calibration_final_sha256: str,
    calibration_files: Mapping[str, Mapping[str, Any]],
    paired_bank: VerifiedPairedBank | None,
    device: torch.device,
) -> dict[str, Any]:
    if (
        arguments.authorization is not None
        or arguments.resource_smoke_receipt is not None
    ):
        raise ValueError("resource smoke does not accept production authorization")
    staging, final = _new_staging_directory(arguments.output_dir)
    model, ema, _ = _build_model_state(
        arm=arguments.arm,
        seed=arguments.seed,
        extension=extension,
        r0_contract=r0_contract,
        geometry=geometry,
        device=device,
    )
    initial_model_sha256 = _model_state_sha256(model)
    optimizer = r0_trainer._optimizer(model)
    orders, presentation = _presentation_orders(arguments.seed)
    intervention_generator, intervention_rng = _intervention_generator(
        arguments.seed, device=device
    )
    depth_generator, depth_rng = _depth_generator(arguments.seed)
    indices = orders[0, :4]
    previous, current, target, centers = _gather_batch(train_dataset, indices)
    previous = previous.to(device)
    current = current.to(device)
    target = target.to(device)
    geometry_batch = geometry.expand(arguments.microbatch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    ledger = _CallLedger()
    scheduler = (
        FourStepVPredictionScheduler()
        if arguments.arm == "PCNO_PDEREFINER_K3_VPRED"
        else None
    )
    forced_depth = (
        1
        if arguments.arm == "MP_PDE_PUSHFORWARD_M01"
        else 3
        if arguments.arm == "CURRICULUM_EMA_PUSHFORWARD_K13"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    try:
        model.train()
        diagnostics = _step_extension_batch(
            arm=arguments.arm,
            model=model,
            optimizer=optimizer,
            ema=ema,
            train_dataset=train_dataset,
            previous=previous,
            current=current,
            clean_target=target,
            centers=centers,
            geometry_batch=geometry_batch,
            fourier_tensors=fourier,
            normalization=normalization,
            microbatch_size=arguments.microbatch_size,
            epoch=2 if arguments.arm == "MP_PDE_PUSHFORWARD_M01" else 10,
            optimizer_step=1,
            depth_generator=depth_generator,
            intervention_generator=intervention_generator,
            scheduler=scheduler,
            paired_bank=paired_bank,
            ledger=ledger,
            requested_depth_override=forced_depth,
        )
        if arguments.arm == "PCNO_PDEREFINER_K3_VPRED":
            assert scheduler is not None
            evaluation_model = ema.model if ema is not None else model
            evaluation_model.eval()
            with torch.no_grad():
                tape = sample_refiner_noise_tape(
                    current[: arguments.microbatch_size],
                    generator=intervention_generator,
                )
                result = refined_recurrent_step(
                    evaluation_model,
                    previous[: arguments.microbatch_size],
                    current[: arguments.microbatch_size],
                    _geometry_for(geometry_batch, arguments.microbatch_size),
                    normalization,
                    scheduler,
                    tape,
                    fourier_tensors=_fourier_for(fourier, arguments.microbatch_size),
                )
            if result.model_calls != 4:
                raise ValueError("refiner resource smoke lacks four inference calls")
            smoke_centers = [
                int(value) for value in centers[: arguments.microbatch_size].tolist()
            ]
            for timestep in (3, 2, 1, 0):
                ledger.record(
                    counter="development_ema_model_calls",
                    kind="refiner_resource_smoke_inference",
                    epoch=0,
                    optimizer_step=1,
                    centers=smoke_centers,
                    gradient_bearing=False,
                    detail={"timestep": timestep},
                )
        _synchronize(device)
        wall_time = time.perf_counter() - started
        common = _common_bindings(
            arm=arguments.arm,
            seed=arguments.seed,
            dataset_manifest=dataset_manifest,
            dataset_final_sha256=dataset_final_sha256,
            calibration_metadata=calibration_metadata,
            calibration_final_sha256=calibration_final_sha256,
            paired_bank=paired_bank,
        )
        receipt = _self_hashed(
            {
                "schema": SMOKE_SCHEMA,
                "status": "succeeded",
                **common,
                "effective_batch_size": 4,
                "microbatch_size": arguments.microbatch_size,
                "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
                "full_resolution_num_nodes": geometry.num_nodes,
                "paired_initialization_and_clean_order_where_architectures_allow": True,
                "initial_model_state_sha256": initial_model_sha256,
                "presentation_schedule_sha256": presentation[
                    "presentation_schedule_sha256"
                ],
                "rng": {
                    "intervention_initial_generator_state_sha256": intervention_rng[
                        "initial_generator_state_sha256"
                    ],
                    "depth_initial_generator_state_sha256": depth_rng[
                        "initial_generator_state_sha256"
                    ],
                },
                "smoke_forced_requested_depth": forced_depth,
                "model_call_schedule_sha256": ledger.hexdigest(),
                "model_call_cost": dict(ledger.counts),
                "smoke_wall_time_seconds": wall_time,
                "forward_backward_and_adamw_step_completed": True,
                "loss": diagnostics.loss,
                "gradient_norm_before_clip": diagnostics.gradient_norm_before_clip,
                "ema_updated_after_optimizer": ema is not None,
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
        _reverify_bound_inputs(
            arguments=arguments,
            calibration_files=calibration_files,
            calibration_final_sha256=calibration_final_sha256,
            dataset_final_sha256=dataset_final_sha256,
            paired_bank=paired_bank,
        )
        _write_json(staging / "resource_smoke.json", receipt)
        _commit_staging_directory(staging, final)
        return receipt
    except Exception as error:
        _write_json(
            staging / "status.json",
            {
                "schema": SMOKE_SCHEMA,
                "status": "failed_incomplete_not_scientific_evidence",
                "error_type": type(error).__name__,
                "error": str(error),
                "prospective_opened": False,
                "sealed_opened": False,
            },
        )
        raise


def _run_training(
    *,
    arguments: argparse.Namespace,
    extension: VerifiedExtensionContract,
    r0_contract: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_final_sha256: str,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    train_dataset: NACABDF2Dataset,
    development_dataset: NACABDF2Dataset,
    calibration_metadata: Mapping[str, Any],
    calibration_final_sha256: str,
    calibration_files: Mapping[str, Mapping[str, Any]],
    paired_bank: VerifiedPairedBank | None,
    device: torch.device,
) -> dict[str, Any]:
    common = _common_bindings(
        arm=arguments.arm,
        seed=arguments.seed,
        dataset_manifest=dataset_manifest,
        dataset_final_sha256=dataset_final_sha256,
        calibration_metadata=calibration_metadata,
        calibration_final_sha256=calibration_final_sha256,
        paired_bank=paired_bank,
    )
    smoke, smoke_file_sha256, authorization, authorization_file_sha256 = (
        _validate_production_authorization(
            arguments=arguments, common=common, device=device
        )
    )
    model, ema, model_config = _build_model_state(
        arm=arguments.arm,
        seed=arguments.seed,
        extension=extension,
        r0_contract=r0_contract,
        geometry=geometry,
        device=device,
    )
    initial_model_state_sha256 = _model_state_sha256(model)
    optimizer = r0_trainer._optimizer(model)
    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1.0e-5
    )
    orders, presentation = _presentation_orders(arguments.seed)
    intervention_generator, intervention_rng = _intervention_generator(
        arguments.seed, device=device
    )
    depth_generator, depth_rng = _depth_generator(arguments.seed)
    runtime = r0_trainer._runtime(device)
    device_binding = r0_trainer._device_binding(device)
    _validate_smoke_replay_bindings(
        smoke,
        common=common,
        runtime=runtime,
        device_binding=device_binding,
        initial_model_state_sha256=initial_model_state_sha256,
        presentation_schedule_sha256=presentation["presentation_schedule_sha256"],
        intervention_initial_state_sha256=intervention_rng[
            "initial_generator_state_sha256"
        ],
        depth_initial_state_sha256=depth_rng["initial_generator_state_sha256"],
        expected_num_nodes=geometry.num_nodes,
    )
    staging, final = _new_staging_directory(arguments.output_dir)
    geometry_batch = geometry.expand(arguments.microbatch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    refiner_scheduler = (
        FourStepVPredictionScheduler()
        if arguments.arm == "PCNO_PDEREFINER_K3_VPRED"
        else None
    )
    ledger = _CallLedger()
    input_manifest = _self_hashed(
        {
            "schema": TRAINING_SCHEMA,
            "record_kind": "input_manifest",
            **common,
            "resource_smoke": {
                "file_name": arguments.resource_smoke_receipt.name,
                "file_sha256": smoke_file_sha256,
                "payload_sha256": smoke["canonical_payload_sha256"],
            },
            "execution_authorization": {
                "file_name": arguments.authorization.name,
                "file_sha256": authorization_file_sha256,
                "payload_sha256": authorization["canonical_payload_sha256"],
                "authorized_by": authorization["authorized_by"],
            },
            "source_files": SOURCE_RECORDS_AT_IMPORT,
            "source_set_sha256": SOURCE_SET_SHA256,
            "paired_bank_schema": (
                PAIRED_BANK_SCHEMA if paired_bank is not None else None
            ),
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    config = _self_hashed(
        {
            "schema": TRAINING_SCHEMA,
            "record_kind": "config",
            **common,
            "model_config": model_config,
            "epochs": 100,
            "effective_batch_size": 4,
            "microbatch_size": arguments.microbatch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1.0e-3,
                "weight_decay": 1.0e-5,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "t_max": 100,
                "eta_min": 1.0e-5,
            },
            "gradient_clip_norm": 1.0,
            "checkpoint_epochs": list(range(5, 101, 5)),
            "ema_decay": 0.995 if ema is not None else None,
            "checkpoint_selection_metric": (
                "fixed_seed_101_four_call_development_normalized_residual_rmse_median"
                if arguments.arm == "PCNO_PDEREFINER_K3_VPRED"
                else "clean_development_one_step_normalized_residual_rmse_median"
            ),
            "checkpoint_selection_deployment": "ema" if ema is not None else "online",
            "checkpoint_selection_used_rollout": False,
        }
    )
    runtime_manifest = _self_hashed(
        {
            "schema": TRAINING_SCHEMA,
            "record_kind": "runtime_manifest",
            **common,
            "runtime": runtime,
            "device_binding": device_binding,
        }
    )
    _write_json(staging / "input_manifest.json", input_manifest)
    _write_json(staging / "config.json", config)
    _write_json(staging / "runtime_manifest.json", runtime_manifest)
    _write_json(
        staging / "history.json",
        _self_hashed(
            {
                "schema": TRAINING_SCHEMA,
                "record_kind": "history",
                **common,
                "epochs": [],
            }
        ),
    )
    _write_json(
        staging / "status.json",
        _self_hashed(
            {
                "schema": TRAINING_SCHEMA,
                "record_kind": "status",
                "status": "running_incomplete_not_scientific_evidence",
                **common,
                "last_completed_epoch": 0,
            }
        ),
    )

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch: int | None = None
    total_optimizer_steps = 0
    requested_depth_counts = {str(depth): 0 for depth in range(4)}
    realized_depth_counts = {str(depth): 0 for depth in range(4)}
    prefix_displacement_sum = {depth: [0.0] * 5 for depth in range(4)}
    prefix_displacement_count = {depth: 0 for depth in range(4)}
    refiner_timestep_counts = {str(timestep): 0 for timestep in range(4)}
    paired_sign_presentations = {"-1": 0, "1": 0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    try:
        for epoch in range(1, 101):
            model.train()
            epoch_loss = 0.0
            gradient_norms: list[float] = []
            presentations = 0
            optimizer_steps = 0
            epoch_requested = {str(depth): 0 for depth in range(4)}
            epoch_realized = {str(depth): 0 for depth in range(4)}
            order = orders[epoch - 1]
            for start in range(0, 238, 4):
                indices = order[start : start + 4]
                previous, current, clean_target, centers = _gather_batch(
                    train_dataset, indices
                )
                group_size = int(previous.shape[0])
                total_optimizer_steps += 1
                diagnostics = _step_extension_batch(
                    arm=arguments.arm,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    train_dataset=train_dataset,
                    previous=previous.to(device),
                    current=current.to(device),
                    clean_target=clean_target.to(device),
                    centers=centers,
                    geometry_batch=geometry_batch,
                    fourier_tensors=fourier,
                    normalization=normalization,
                    microbatch_size=arguments.microbatch_size,
                    epoch=epoch,
                    optimizer_step=total_optimizer_steps,
                    depth_generator=depth_generator,
                    intervention_generator=intervention_generator,
                    scheduler=refiner_scheduler,
                    paired_bank=paired_bank,
                    ledger=ledger,
                )
                epoch_loss += diagnostics.loss * group_size
                gradient_norms.append(diagnostics.gradient_norm_before_clip)
                presentations += group_size
                optimizer_steps += 1
                if diagnostics.requested_depth is not None:
                    key = str(diagnostics.requested_depth)
                    requested_depth_counts[key] += 1
                    epoch_requested[key] += 1
                for depth in diagnostics.realized_depths:
                    key = str(depth)
                    realized_depth_counts[key] += 1
                    epoch_realized[key] += 1
                for timestep in diagnostics.refiner_timesteps:
                    refiner_timestep_counts[str(timestep)] += 1
                if diagnostics.paired_sign is not None:
                    paired_sign_presentations[str(diagnostics.paired_sign)] += (
                        group_size
                    )
                for (
                    depth,
                    values,
                ) in diagnostics.prefix_displacement_sum_squared.items():
                    for field, value in enumerate(values):
                        prefix_displacement_sum[depth][field] += float(value)
                    prefix_displacement_count[depth] += int(
                        diagnostics.prefix_displacement_coordinate_count[depth]
                    )
            if presentations != 238 or optimizer_steps != 60:
                raise ValueError("extension epoch presentation or step count differs")
            online_score: float | None = None
            ema_score: float | None = None
            refiner_online_tape_sha256: str | None = None
            refiner_ema_tape_sha256: str | None = None
            selected = False
            if epoch % 5 == 0:
                if arguments.arm == "PCNO_PDEREFINER_K3_VPRED":
                    assert refiner_scheduler is not None and ema is not None
                    online_score, refiner_online_tape_sha256 = (
                        _refiner_development_score(
                            model=model,
                            dataset=development_dataset,
                            geometry_batch=geometry_batch,
                            fourier_tensors=fourier,
                            normalization=normalization,
                            scheduler=refiner_scheduler,
                            device=device,
                            microbatch_size=arguments.microbatch_size,
                            ledger=ledger,
                            counter="development_online_model_calls",
                            epoch=epoch,
                        )
                    )
                    ema_score, refiner_ema_tape_sha256 = _refiner_development_score(
                        model=ema.model,
                        dataset=development_dataset,
                        geometry_batch=geometry_batch,
                        fourier_tensors=fourier,
                        normalization=normalization,
                        scheduler=refiner_scheduler,
                        device=device,
                        microbatch_size=arguments.microbatch_size,
                        ledger=ledger,
                        counter="development_ema_model_calls",
                        epoch=epoch,
                    )
                    if refiner_online_tape_sha256 != refiner_ema_tape_sha256:
                        raise ValueError(
                            "refiner development models did not share noise tape"
                        )
                else:
                    online_score = _baseline_development_score(
                        model=model,
                        dataset=development_dataset,
                        geometry_batch=geometry_batch,
                        fourier_tensors=fourier,
                        normalization=normalization,
                        device=device,
                        microbatch_size=arguments.microbatch_size,
                        ledger=ledger,
                        counter="development_online_model_calls",
                        epoch=epoch,
                    )
                    if ema is not None:
                        ema_score = _baseline_development_score(
                            model=ema.model,
                            dataset=development_dataset,
                            geometry_batch=geometry_batch,
                            fourier_tensors=fourier,
                            normalization=normalization,
                            device=device,
                            microbatch_size=arguments.microbatch_size,
                            ledger=ledger,
                            counter="development_ema_model_calls",
                            epoch=epoch,
                        )
                primary_score = ema_score if ema is not None else online_score
                assert primary_score is not None
                primary_score = parent_trainer._canonical_development_score(
                    primary_score
                )
                if parent_trainer._select_development_checkpoint(
                    epoch=epoch,
                    score=primary_score,
                    best_epoch=best_epoch,
                    best_score=best_score,
                ):
                    best_score = primary_score
                    best_epoch = epoch
                    selected = True
                    best_payload = _checkpoint_payload(
                        record_kind="selected_checkpoint",
                        arm=arguments.arm,
                        seed=arguments.seed,
                        epoch=epoch,
                        model=model,
                        ema=ema,
                        optimizer=None,
                        learning_rate_scheduler=None,
                        intervention_generator=intervention_generator,
                        depth_generator=depth_generator,
                        bindings=common,
                        best_epoch=epoch,
                        best_score=best_score,
                        model_only=True,
                    )
                    _atomic_torch_save(staging / "best.pt", best_payload)
            learning_rate_scheduler.step()
            ema_distance = _ema_distance(model, ema)
            history.append(
                {
                    "epoch": epoch,
                    "train_sample_mean_objective": epoch_loss / presentations,
                    "maximum_gradient_norm_before_clip": float(max(gradient_norms)),
                    "presentations": presentations,
                    "optimizer_steps": optimizer_steps,
                    "learning_rate_after_scheduler_step": float(
                        learning_rate_scheduler.get_last_lr()[0]
                    ),
                    "requested_depth_minibatch_counts": epoch_requested,
                    "realized_depth_sample_counts": epoch_realized,
                    "paired_sign": (
                        paired_bank_sign_for_epoch(epoch)
                        if arguments.arm in PAIRED_ARMS
                        else None
                    ),
                    "ema_online_parameter_distance": ema_distance,
                    "online_development_score": (
                        _json_number(online_score) if online_score is not None else None
                    ),
                    "ema_development_score": (
                        _json_number(ema_score) if ema_score is not None else None
                    ),
                    "refiner_development_noise_tape_sha256": refiner_ema_tape_sha256,
                    "selected_as_best": selected,
                }
            )
            if best_epoch is None:
                checkpoint_best_epoch = 0
                checkpoint_best_score = math.inf
            else:
                checkpoint_best_epoch = best_epoch
                checkpoint_best_score = best_score
            last_payload = _checkpoint_payload(
                record_kind="last_checkpoint",
                arm=arguments.arm,
                seed=arguments.seed,
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                learning_rate_scheduler=learning_rate_scheduler,
                intervention_generator=intervention_generator,
                depth_generator=depth_generator,
                bindings=common,
                best_epoch=checkpoint_best_epoch,
                best_score=checkpoint_best_score,
                model_only=False,
            )
            _atomic_torch_save(staging / "last.pt", last_payload)
            _write_json(
                staging / "history.json",
                _self_hashed(
                    {
                        "schema": TRAINING_SCHEMA,
                        "record_kind": "history",
                        **common,
                        "epochs": history,
                    }
                ),
            )
            _write_json(
                staging / "status.json",
                _self_hashed(
                    {
                        "schema": TRAINING_SCHEMA,
                        "record_kind": "status",
                        "status": "running_incomplete_not_scientific_evidence",
                        **common,
                        "last_completed_epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_development_score": (
                            _json_number(best_score) if best_epoch is not None else None
                        ),
                    }
                ),
            )
        _synchronize(device)
        wall_time = time.perf_counter() - started
        if best_epoch is None or not (staging / "best.pt").is_file():
            raise ValueError("extension training completed without selected checkpoint")
        prefix_rms = {
            str(depth): (
                [
                    math.sqrt(value / prefix_displacement_count[depth])
                    for value in prefix_displacement_sum[depth]
                ]
                if prefix_displacement_count[depth] > 0
                else [0.0] * 5
            )
            for depth in range(4)
        }
        cost = {
            **ledger.counts,
            "model_call_event_count": ledger.event_count,
            "optimizer_steps": total_optimizer_steps,
            "training_presentations": 100 * 238,
            "requested_depth_minibatch_counts": requested_depth_counts,
            "realized_depth_sample_counts": realized_depth_counts,
            "refiner_timestep_sample_counts": refiner_timestep_counts,
            "paired_sign_presentations": paired_sign_presentations,
            "training_wall_time_seconds": wall_time,
            "cuda_peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        }
        summary = _self_hashed(
            {
                "schema": TRAINING_SCHEMA,
                "record_kind": "summary",
                "status": "complete",
                "classification": "SCIENTIFIC_TRAINING_ARTIFACT",
                **common,
                "completed_epochs": 100,
                "best_epoch": best_epoch,
                "best_development_score": _json_number(best_score),
                "checkpoint_selection_deployment": (
                    "ema" if ema is not None else "online"
                ),
                "checkpoint_selection_used_rollout": False,
                "ema_update_count": (
                    int(ema.num_updates.item()) if ema is not None else None
                ),
                "final_ema_online_parameter_distance": _ema_distance(model, ema),
                "requested_depth_minibatch_counts": requested_depth_counts,
                "realized_depth_sample_counts": realized_depth_counts,
                "prefix_displacement_per_field_state_normalized_rms_by_depth": prefix_rms,
                "refiner_timestep_sample_counts": refiner_timestep_counts,
                "paired_sign_presentations": paired_sign_presentations,
                "model_call_schedule_sha256": ledger.hexdigest(),
                "model_call_cost": cost,
                "model_call_cost_sha256": _canonical_sha256(cost),
                "training_wall_time_seconds": wall_time,
                "scientific_claim_made": False,
            }
        )
        _write_json(staging / "summary.json", summary)
        _write_json(
            staging / "status.json",
            _self_hashed(
                {
                    "schema": TRAINING_SCHEMA,
                    "record_kind": "status",
                    "status": "complete",
                    **common,
                    "last_completed_epoch": 100,
                    "best_epoch": best_epoch,
                    "best_development_score": _json_number(best_score),
                }
            ),
        )
        _reverify_bound_inputs(
            arguments=arguments,
            calibration_files=calibration_files,
            calibration_final_sha256=calibration_final_sha256,
            dataset_final_sha256=dataset_final_sha256,
            paired_bank=paired_bank,
            resource_smoke_file_sha256=smoke_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
        _write_final_hash_manifest(
            staging,
            arm=arguments.arm,
            seed=arguments.seed,
            bindings=common,
            initial_model_state_sha256=initial_model_state_sha256,
            presentation_schedule_sha256=presentation["presentation_schedule_sha256"],
        )
        _commit_staging_directory(staging, final)
        return summary
    except Exception as error:
        _write_json(
            staging / "status.json",
            _self_hashed(
                {
                    "schema": TRAINING_SCHEMA,
                    "record_kind": "status",
                    "status": "failed_incomplete_not_scientific_evidence",
                    **common,
                    "last_completed_epoch": len(history),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
        )
        raise


def _validate_open_roles(
    roles: Mapping[str, tuple[np.ndarray, np.ndarray, NACABDF2Dataset]],
    extension: Mapping[str, Any],
) -> None:
    if set(roles) != {"train", "development"}:
        raise ValueError("extension loader exposed an unexpected population role")
    expected = {
        "train": ([955, 1194], [956, 1193]),
        "development": ([1233, 1472], [1234, 1471]),
    }
    population = extension["population"]
    for role, (frames, centers) in expected.items():
        states, frame_indices, dataset = roles[role]
        if (
            frame_indices.tolist() != list(range(frames[0], frames[1] + 1))
            or list(dataset.center_indices) != list(range(centers[0], centers[1] + 1))
            or states.shape[0] != 240
            or population[f"{role}_frames_inclusive"] != frames
            or population[f"{role}_centers_inclusive"] != centers
        ):
            raise ValueError(f"extension {role} population differs")


def _load_optional_paired_bank(
    arguments: argparse.Namespace,
    *,
    expected_coordinates: np.ndarray,
    train_states: np.ndarray,
    train_frame_indices: np.ndarray,
    normalization: NACANormalization,
) -> VerifiedPairedBank | None:
    values = (
        arguments.paired_bank_dir,
        arguments.paired_bank_final_sha256,
        arguments.pilot_final_hash_manifest,
        arguments.pilot_final_sha256,
    )
    if arguments.arm not in PAIRED_ARMS:
        if any(value is not None for value in values):
            raise ValueError(
                "nonpaired extension arm must not receive paired-bank inputs"
            )
        return None
    if any(value is None for value in values):
        raise ValueError(
            "paired extension arm requires bank and pilot paths and exact SHA256s"
        )
    return _load_paired_bank(
        arguments.paired_bank_dir,
        expected_final_sha256=arguments.paired_bank_final_sha256,
        pilot_final_manifest_path=arguments.pilot_final_hash_manifest,
        expected_pilot_final_sha256=arguments.pilot_final_sha256,
        expected_coordinates=expected_coordinates,
        train_states=train_states,
        train_frame_indices=train_frame_indices,
        normalization=normalization,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--successor-contract", type=Path, required=True)
    parser.add_argument("--successor-preregistration", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--extension-preregistration", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--paired-bank-dir", type=Path)
    parser.add_argument("--paired-bank-final-sha256")
    parser.add_argument("--pilot-final-hash-manifest", type=Path)
    parser.add_argument("--pilot-final-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
        extension = load_extension_contract(
            arguments.extension_contract, arguments.extension_preregistration
        )
        extension_payload = extension.payload()
        _validate_training_contract(extension_payload)
        successor = parent_trainer._load_successor_contract(
            arguments.successor_contract,
            arguments.successor_preregistration,
            arguments.contract,
        )
        r0_contract = load_naca_baseline_contract(arguments.contract)
        validate_successor_math_contract(successor, r0_contract)
        (
            dataset_manifest,
            geometry,
            normalization,
            roles,
            dataset_final_sha256,
        ) = r0_trainer._load_dataset(arguments.dataset_dir, r0_contract)
        if dataset_final_sha256 != DATASET_FINAL_SHA256:
            raise ValueError("dataset final manifest differs from extension contract")
        _validate_open_roles(roles, extension_payload)
        (
            calibration_metadata,
            _recovery_calibration,
            calibration_final_sha256,
            calibration_files,
        ) = parent_trainer._load_calibration(
            arguments.calibration_dir,
            successor,
            dataset_manifest,
            dataset_final_sha256,
            normalization,
            geometry.num_nodes,
        )
        if calibration_final_sha256 != SUCCESSOR_CALIBRATION_SHA256:
            raise ValueError("successor calibration differs from extension contract")
        paired_bank = _load_optional_paired_bank(
            arguments,
            expected_coordinates=geometry.native_coordinates,
            train_states=roles["train"][0],
            train_frame_indices=roles["train"][1],
            normalization=normalization,
        )
        _verify_live_sources()
        device = torch.device(arguments.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if arguments.resource_smoke:
            result = _resource_smoke(
                arguments=arguments,
                extension=extension,
                r0_contract=r0_contract,
                dataset_manifest=dataset_manifest,
                dataset_final_sha256=dataset_final_sha256,
                geometry=geometry,
                normalization=normalization,
                train_dataset=roles["train"][2],
                calibration_metadata=calibration_metadata,
                calibration_final_sha256=calibration_final_sha256,
                calibration_files=calibration_files,
                paired_bank=paired_bank,
                device=device,
            )
        else:
            result = _run_training(
                arguments=arguments,
                extension=extension,
                r0_contract=r0_contract,
                dataset_manifest=dataset_manifest,
                dataset_final_sha256=dataset_final_sha256,
                geometry=geometry,
                normalization=normalization,
                train_dataset=roles["train"][2],
                development_dataset=roles["development"][2],
                calibration_metadata=calibration_metadata,
                calibration_final_sha256=calibration_final_sha256,
                calibration_files=calibration_files,
                paired_bank=paired_bank,
                device=device,
            )
        _verify_live_sources()
        parent_trainer._reverify_calibration(
            arguments.calibration_dir,
            calibration_files,
            calibration_final_sha256,
        )
        if (
            sha256_file(arguments.dataset_dir.resolve() / "final_hash_manifest.json")
            != dataset_final_sha256
        ):
            raise ValueError("dataset packet changed during extension execution")
        if paired_bank is not None:
            _reverify_paired_bank(paired_bank)
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"NACA0012 corrective extension training failed: {error}", file=sys.stderr
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
