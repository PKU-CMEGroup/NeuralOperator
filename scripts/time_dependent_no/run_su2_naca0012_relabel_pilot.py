#!/usr/bin/env python3
"""Run the frozen eleven-case train-only SU2 NACA0012 relabeling pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.time_dependent_no import train_pcno_naca0012 as r0_trainer
from scripts.time_dependent_no import (
    train_pcno_naca0012_successor as successor_trainer,
)
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    load_naca_baseline_contract,
    validate_parent_evidence,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    validate_extension_math_contract,
)
from utility.time_dependent_no.su2_naca0012_relabel import (
    EXPERIMENT_ID,
    REQUIRED_RECEIPT_CHECKS,
    RelabelStepIndex,
    load_verified_relabel_receipt,
    naca_state_admissibility,
    relabel_config_overrides,
    relabel_step_index,
    validate_relabel_history,
    validate_relabel_output,
    write_fail_closed_receipt,
    write_restart_from_verified_template,
)
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    audit_unsteady_naca0012_bundle,
    parse_su2_config,
    read_su2_binary_restart,
    sha256_file,
)

EXTENSION_CONTRACT_FILE_SHA256 = (
    "3d786bafff23cc08bbb4434a1f89253af3edde92e6637bcdcad8c263e97ae02c"
)
EXTENSION_PREREGISTRATION_SHA256 = (
    "1001e5f35b5d0aef811ec97cd5323c83ea67ca0a7eeed35220405117e47f61d1"
)
R0_CONTRACT_SHA256 = "94069e65ef520d31860735b6c17f2b7515da1c4acd34d68518a2a16b3181fa87"
SUCCESSOR_CONTRACT_SHA256 = (
    "b0f373d42ca98c0761a61f17310233d22461da1add28b8c8d20359ef7e1d6715"
)
DATASET_FINAL_SHA256 = (
    "1cb5fd2d751bdf1ca27cde5e1a1457973a525c19a98dbadd7cc3738c3efe2e4b"
)
CALIBRATION_FINAL_SHA256 = (
    "6a33f2e20a92dcfa58d16872b6abd2f6d2744731c8c678815618cafd373d0dbc"
)
RESOURCE_MANIFEST_SHA256 = (
    "7e20bb67ebcf09447f0badd4d10752a78f749f5c038bb444cedff81d39ff56f5"
)
SU2_EXECUTABLE_SHA256 = (
    "61a803e0baf49382210888cc32f14f4ea837681a0e6cc2df6b1d5a99ce5baf8c"
)
PILOT_CENTERS = (956, 1075, 1193)
TRAIN_CENTERS = tuple(range(956, 1194))
PILOT_SIGNS = (-1, 0, 1)
BASE_SEED = 20260902
PER_FIELD_NORMALIZED_STD = np.asarray(
    [
        0.0023510525449798097,
        0.0017943288061936155,
        0.004045914459334123,
        0.002355418735336881,
        0.001718743054261789,
    ],
    dtype=np.float64,
)
OVERALL_NORMALIZED_RMS = 0.002593012258034671
ZERO_OVERALL_LIMIT = 0.00002593012258034671
ZERO_FIELD_LIMITS = np.asarray(
    [
        0.0000235105254497981,
        0.000017943288061936155,
        0.00004045914459334123,
        0.00002355418735336881,
        0.00001718743054261789,
    ],
    dtype=np.float64,
)
REALIZED_RMS_RATIO_RANGE = (0.95, 1.05)
MIN_RESPONSE_RATIO = 4.0
PILOT_RECEIPT_SCHEMA = "time_dependent_no.naca_su2_relabel_pilot.v1"
PILOT_FINAL_HASH_SCHEMA = (
    "time_dependent_no.naca_su2_relabel_pilot_final_hash_manifest.v1"
)
PILOT_CONFIG_FILENAME = "relabel.cfg"
PILOT_RECEIPT_FILENAME = "pilot_receipt.json"
PILOT_FINAL_HASH_FILENAME = "final_hash_manifest.json"
PILOT_SOURCE_FILES = (
    "scripts/time_dependent_no/run_su2_naca0012_relabel_pilot.py",
    "utility/time_dependent_no/su2_naca0012_relabel.py",
    "utility/time_dependent_no/pcno_naca0012_corrective_extension.py",
)

_DYNAMIC_INDICES = tuple(
    NACA_NATIVE_RESTART_FIELDS.index(field) for field in NACA_DYNAMIC_FIELDS
)
_COORDINATE_INDICES = tuple(
    NACA_NATIVE_RESTART_FIELDS.index(field) for field in ("x", "y")
)


@dataclass(frozen=True)
class PilotCase:
    ordinal: int
    center: int
    sign: int
    variant: str

    @property
    def name(self) -> str:
        sign_name = {-1: "m1", 0: "z0", 1: "p1"}[self.sign]
        return (
            f"case_{self.ordinal:02d}_center_{self.center:05d}_"
            f"sign_{sign_name}_{self.variant}"
        )

    @property
    def step(self) -> RelabelStepIndex:
        return relabel_step_index(self.center)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "center": self.center,
            "sign": self.sign,
            "variant": self.variant,
        }


@dataclass(frozen=True)
class NativeFrame:
    index: int
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    bytes: int
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {"file": self.path.name, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class PilotAuthority:
    extension: Mapping[str, Any]
    normalization: NACANormalization
    clean_states: Mapping[int, np.ndarray]
    native_frames: Mapping[int, NativeFrame]
    coordinates: np.ndarray
    config_path: Path
    mesh_path: Path
    executable_path: Path
    snapshots: Mapping[str, FileSnapshot]
    authority_summary: Mapping[str, Any]


@dataclass(frozen=True)
class CaseExecution:
    case: PilotCase
    directory: Path
    receipt: Mapping[str, Any]
    output_state: np.ndarray | None
    output_path: Path | None
    realized_field_rms: np.ndarray | None


def pilot_cases() -> tuple[PilotCase, ...]:
    cases: list[PilotCase] = []
    for center in PILOT_CENTERS:
        for sign in PILOT_SIGNS:
            cases.append(PilotCase(len(cases), center, sign, "base"))
    cases.append(PilotCase(len(cases), 1075, 1, "repeat"))
    cases.append(PilotCase(len(cases), 1075, 1, "auxiliary_swap"))
    if len(cases) != 11:
        raise AssertionError("pilot schedule must contain exactly eleven cases")
    return tuple(cases)


def _require_absolute(path: str | Path, label: str) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return supplied


def _snapshot(path: Path, label: str) -> FileSnapshot:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is absent or aliased")
    return FileSnapshot(path.resolve(), path.stat().st_size, sha256_file(path))


def _verify_snapshot(snapshot: FileSnapshot, label: str) -> None:
    observed = _snapshot(snapshot.path, label)
    if observed != snapshot:
        raise RuntimeError(f"{label} changed after authority verification")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical hash must not be supplied")
    result["canonical_payload_sha256"] = sha256(_canonical_bytes(result)).hexdigest()
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_self_hashed_json(path: Path, schema: str) -> dict[str, Any]:
    snapshot = _snapshot(path, path.name)
    value = json.loads(snapshot.path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unsupported JSON schema: {path.name}")
    observed = value.pop("canonical_payload_sha256", None)
    if observed != sha256(_canonical_bytes(value)).hexdigest():
        raise ValueError(f"canonical payload hash differs: {path.name}")
    value["canonical_payload_sha256"] = observed
    _verify_snapshot(snapshot, path.name)
    return value


def _load_extension_contract(
    contract_path: Path, preregistration_path: Path
) -> dict[str, Any]:
    contract = _snapshot(contract_path, "extension contract")
    preregistration = _snapshot(preregistration_path, "extension preregistration")
    if contract.sha256 != EXTENSION_CONTRACT_FILE_SHA256:
        raise ValueError("extension contract differs from the frozen file hash")
    if preregistration.sha256 != EXTENSION_PREREGISTRATION_SHA256:
        raise ValueError("extension preregistration differs from the frozen file hash")
    value = json.loads(contract.path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError("extension contract must be a JSON object")
    validate_extension_math_contract(value)
    parents = value.get("parents")
    expected_parents = {
        "r0_contract_sha256": R0_CONTRACT_SHA256,
        "successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
        "successor_calibration_final_hash_manifest_sha256": CALIBRATION_FINAL_SHA256,
        "dataset_final_hash_manifest_sha256": DATASET_FINAL_SHA256,
        "resource_manifest_sha256": RESOURCE_MANIFEST_SHA256,
        "su2_release": "SU2 v8.5.0 Harrier",
        "su2_executable_sha256": SU2_EXECUTABLE_SHA256,
    }
    if value.get("preregistration_sha256") != preregistration.sha256:
        raise ValueError("extension preregistration binding differs")
    if parents != expected_parents:
        raise ValueError("extension parent bindings differ")
    displacement = value.get("displacement")
    if not isinstance(displacement, Mapping):
        raise TypeError("extension displacement contract is absent")
    if (
        displacement.get("coordinate_system") != "state_normalized"
        or displacement.get("distribution")
        != "independent_gaussian_all_nodes_fields_and_history_slots"
        or displacement.get("per_field_standard_deviation")
        != PER_FIELD_NORMALIZED_STD.tolist()
        or displacement.get("overall_equal_entry_rms") != OVERALL_NORMALIZED_RMS
        or displacement.get("scale_multiplier") != 1.0
        or displacement.get("rollout_tuned") is not False
    ):
        raise ValueError("extension displacement contract differs")
    _validate_pilot_contract(value.get("relabeling_pilot"))
    protection = value.get("protection")
    if (
        not isinstance(protection, Mapping)
        or protection.get("train_opened") is not True
        or protection.get("offline_training_label_solver_calls") is not True
        or any(
            protection.get(key) is not False
            for key in (
                "prospective_opened",
                "sealed_opened",
                "online_solver_calls",
                "online_defect_trigger",
            )
        )
    ):
        raise PermissionError("extension protection boundary differs")
    _verify_snapshot(contract, "extension contract")
    _verify_snapshot(preregistration, "extension preregistration")
    return value


def _validate_pilot_contract(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("relabeling pilot contract is absent")
    exact = {
        "centers": list(PILOT_CENTERS),
        "base_seed": BASE_SEED,
        "rng": {
            "library": "torch",
            "device": "cpu",
            "generator": "torch.Generator.manual_seed",
            "distribution_call": "torch.randn",
            "dtype": "float32",
            "stream_order": "one_ascending_center_stream_956_through_1193",
            "draw_shape": "2_by_num_nodes_by_5_previous_then_current",
            "pilot_directions_identical_to_corresponding_query_bank_entries": True,
        },
        "signs": list(PILOT_SIGNS),
        "one_step_call_count": 11,
        "deterministic_repeat_case": {"center": 1075, "sign": 1},
        "auxiliary_column_probe": {
            "center": 1075,
            "sign": 1,
            "same_dynamic_bdf2_pair": True,
            "alternate_auxiliary_templates": (
                "swap_native_1074_and_1075_auxiliary_columns_between_history_slots"
            ),
            "required_evolved_output_comparison": "bitwise_equal",
            "full_native_output_comparison": "record_only",
        },
        "absolute_indexing": {
            "input_history_indices": "center_minus_1_and_center",
            "restart_iter": "center_plus_1",
            "time_iter": "center_plus_2",
            "expected_output_index": "center_plus_1",
        },
        "zero_control_max_overall_state_normalized_rms": ZERO_OVERALL_LIMIT,
        "zero_control_max_per_field_state_normalized_rms": ZERO_FIELD_LIMITS.tolist(),
        "zero_control_reference": "stored_native_float64_clean_successor",
        "zero_control_inputs": (
            "clean_states_rounded_to_authoritative_model_facing_float32"
        ),
        "realized_field_rms_ratio_inclusive": list(REALIZED_RMS_RATIO_RANGE),
        "response_metric": (
            "state_normalized_equal_entry_rms_signed_output_minus_same_center_zero_output"
        ),
        "response_population": "six_values_from_three_centers_and_two_nonzero_signs",
        "response_aggregation": "median",
        "median_response_to_max_zero_discrepancy_min_ratio": MIN_RESPONSE_RATIO,
        "repeat_required_comparisons": [
            "full_native_output_file_bytes_equal",
            "five_evolved_output_arrays_bitwise_equal",
        ],
        "repeat_history_comparison": "convergence_contract_only_not_bitwise",
        "inner_iter": 10,
        "convergence_field": "REL_RMS_DENSITY",
        "convergence_residual_minval": -3.0,
        "require_final_relrms_density_lte": -3.0,
        "require_positive_density": True,
        "require_positive_ideal_gas_pressure": True,
        "nu_tilde_sign_policy": "report_without_clip_mask_or_rejection",
        "require_all_solver_cases": True,
        "failure_policy": (
            "stop_without_silent_drop_replacement_or_inner_iteration_change"
        ),
    }
    if dict(value) != exact:
        raise ValueError("relabeling pilot contract differs from the frozen semantics")


def _trajectory_record(storage: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    files = storage.get("files")
    if not isinstance(files, list):
        raise TypeError("trajectory storage files must be a list")
    matches = [record for record in files if record.get("index") == index]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError(f"trajectory storage has no unique train frame {index}")
    return matches[0]


def _native_frame(
    *,
    trajectory_root: Path,
    storage: Mapping[str, Any],
    index: int,
    expected_coordinates: np.ndarray,
    expected_state: np.ndarray,
) -> NativeFrame:
    if index < 955 or index > 1194:
        raise PermissionError("pilot attempted to open a nontrain trajectory frame")
    record = _trajectory_record(storage, index)
    expected_name = f"trajectory_flow_{index:05d}.dat"
    path = trajectory_root / expected_name
    if (
        record.get("file") != expected_name
        or isinstance(record.get("bytes"), bool)
        or not isinstance(record.get("bytes"), int)
        or not isinstance(record.get("sha256"), str)
    ):
        raise ValueError(f"trajectory record {index} differs")
    observed = _snapshot(path, f"train trajectory frame {index}")
    if observed.bytes != record["bytes"] or observed.sha256 != record["sha256"]:
        raise ValueError(f"train trajectory frame {index} differs from storage")
    restart = read_su2_binary_restart(path)
    if restart.fields != NACA_NATIVE_RESTART_FIELDS or restart.num_fields != 19:
        raise ValueError(f"train trajectory frame {index} is not native-19")
    if not np.array_equal(
        restart.values[:, _COORDINATE_INDICES].view(np.uint64),
        expected_coordinates.view(np.uint64),
    ):
        raise ValueError(f"train trajectory frame {index} coordinates differ")
    dynamic = restart.values[:, _DYNAMIC_INDICES]
    if expected_state.dtype != np.float64 or not np.array_equal(
        dynamic.view(np.uint64), expected_state.view(np.uint64)
    ):
        raise ValueError(f"dataset and trajectory state differ at frame {index}")
    return NativeFrame(index, observed.path, observed.bytes, observed.sha256)


def _resource_path(
    root: Path, resources: Mapping[str, Any], role: str
) -> tuple[Path, Mapping[str, Any]]:
    record = resources.get(role)
    if not isinstance(record, Mapping):
        raise TypeError(f"resource manifest lacks {role}")
    relative = Path(str(record.get("file")))
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError(f"resource path is unsafe: {role}")
    path = root / relative
    observed = _snapshot(path, f"resource {role}")
    if observed.bytes != record.get("bytes") or observed.sha256 != record.get("sha256"):
        raise ValueError(f"resource differs from its manifest: {role}")
    return observed.path, record


def _load_bound_successor_contract(
    successor_contract_path: Path, baseline_contract_path: Path
) -> tuple[dict[str, Any], str]:
    """Load the parent successor through its complete three-file authority."""

    successor_preregistration_path = successor_contract_path.with_name(
        "B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md"
    )
    successor = successor_trainer._load_successor_contract(
        successor_contract_path,
        successor_preregistration_path,
        baseline_contract_path,
    )
    successor_sha256 = _snapshot(successor_contract_path, "successor contract").sha256
    return successor, successor_sha256


def _load_pilot_authority(
    *,
    extension_contract_path: Path,
    extension_preregistration_path: Path,
    successor_contract_path: Path,
    baseline_contract_path: Path,
    dataset_dir: Path,
    calibration_dir: Path,
    trajectory_dir: Path,
    resource_manifest_path: Path,
    resource_dir: Path,
    executable_path: Path,
) -> PilotAuthority:
    for path, label in (
        (dataset_dir, "dataset directory"),
        (calibration_dir, "calibration directory"),
        (trajectory_dir, "trajectory directory"),
        (resource_dir, "resource directory"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} is absent or aliased")
    extension = _load_extension_contract(
        extension_contract_path, extension_preregistration_path
    )
    successor, successor_sha256 = _load_bound_successor_contract(
        successor_contract_path, baseline_contract_path
    )
    if successor_sha256 != SUCCESSOR_CONTRACT_SHA256:
        raise ValueError("successor contract differs from extension binding")
    baseline_snapshot = _snapshot(baseline_contract_path, "R0 contract")
    if baseline_snapshot.sha256 != R0_CONTRACT_SHA256:
        raise ValueError("R0 contract differs from extension binding")
    baseline = load_naca_baseline_contract(baseline_contract_path)
    context = validate_parent_evidence(
        baseline,
        repository_root=REPO_ROOT,
        resource_root=resource_dir,
    )
    if context.trajectory_root.resolve() != trajectory_dir.resolve():
        raise ValueError("trajectory directory differs from verified parent evidence")
    dataset_manifest, geometry, normalization, roles, dataset_sha256 = (
        r0_trainer._load_dataset(dataset_dir, baseline)
    )
    if dataset_sha256 != DATASET_FINAL_SHA256:
        raise ValueError("dataset final hash differs from extension binding")
    calibration, recovery, calibration_sha256, _ = successor_trainer._load_calibration(
        calibration_dir,
        successor,
        dataset_manifest,
        dataset_sha256,
        normalization,
        geometry.num_nodes,
    )
    if calibration_sha256 != CALIBRATION_FINAL_SHA256:
        raise ValueError("calibration final hash differs from extension binding")
    observed_scale = recovery.iid_field_rms.numpy()
    if not np.array_equal(observed_scale, PER_FIELD_NORMALIZED_STD):
        raise ValueError("calibration IID field scale differs from extension")
    if math.sqrt(float(np.mean(np.square(observed_scale)))) != OVERALL_NORMALIZED_RMS:
        raise ValueError("calibration aggregate IID scale differs from extension")

    resource_manifest = _snapshot(resource_manifest_path, "resource manifest")
    if resource_manifest.sha256 != RESOURCE_MANIFEST_SHA256:
        raise ValueError("resource manifest differs from extension binding")
    stage0 = audit_unsteady_naca0012_bundle(
        resource_dir=resource_dir, manifest_path=resource_manifest.path
    )
    if stage0.get("status") != "stage0_valid":
        raise ValueError("Stage-0 resource audit did not pass")
    resource_value = json.loads(resource_manifest.path.read_text(encoding="utf-8"))
    resources = resource_value.get("resources")
    if not isinstance(resources, Mapping):
        raise TypeError("resource manifest lacks resources")
    config_path, _ = _resource_path(resource_dir, resources, "config")
    mesh_path, _ = _resource_path(resource_dir, resources, "mesh")
    release = resource_value.get("upstream", {}).get("target_replay_release", {})
    executable_contract = release.get("windows_mpi_asset", {}).get("executable", {})
    executable = _snapshot(executable_path, "SU2 executable")
    if (
        release.get("tag") != "v8.5.0"
        or executable.sha256 != SU2_EXECUTABLE_SHA256
        or executable_contract.get("sha256") != SU2_EXECUTABLE_SHA256
        or executable_contract.get("bytes") != executable.bytes
    ):
        raise ValueError("SU2 executable differs from the frozen release asset")

    trajectory_root = trajectory_dir.resolve()
    storage_path = trajectory_root / "trajectory_storage_manifest.json"
    storage_snapshot = _snapshot(storage_path, "trajectory storage manifest")
    parent_trajectory = baseline["parent_evidence"]["trajectory"]
    if storage_snapshot.sha256 != parent_trajectory["storage_manifest_file_sha256"]:
        raise ValueError("trajectory storage manifest differs from R0 binding")
    storage = _load_self_hashed_json(
        storage_path, "time_dependent_no.su2_naca0012_trajectory_storage.v1"
    )
    if (
        storage.get("canonical_payload_sha256")
        != parent_trajectory["storage_manifest_payload_sha256"]
    ):
        raise ValueError("trajectory storage payload differs from R0 binding")
    trajectory_receipt_snapshot = _snapshot(
        trajectory_root / "trajectory_receipt.json", "trajectory receipt"
    )
    if (
        trajectory_receipt_snapshot.sha256
        != parent_trajectory["trajectory_receipt_file_sha256"]
    ):
        raise ValueError("trajectory receipt differs from R0 binding")

    train_states, train_indices, _ = roles["train"]
    positions = {int(index): offset for offset, index in enumerate(train_indices)}
    required_indices = sorted(
        {
            index
            for center in PILOT_CENTERS
            for index in (center - 1, center, center + 1)
        }
    )
    coordinates = np.array(geometry.native_coordinates, dtype=np.float64, copy=True)
    clean_states: dict[int, np.ndarray] = {}
    native_frames: dict[int, NativeFrame] = {}
    for index in required_indices:
        state = np.array(train_states[positions[index]], dtype=np.float64, copy=True)
        native_frames[index] = _native_frame(
            trajectory_root=trajectory_root,
            storage=storage,
            index=index,
            expected_coordinates=coordinates,
            expected_state=state,
        )
        state.setflags(write=False)
        clean_states[index] = state
    coordinates.setflags(write=False)

    snapshots = {
        "extension_contract": _snapshot(extension_contract_path, "extension contract"),
        "extension_preregistration": _snapshot(
            extension_preregistration_path, "extension preregistration"
        ),
        "successor_contract": _snapshot(successor_contract_path, "successor contract"),
        "baseline_contract": baseline_snapshot,
        "dataset_final_hash_manifest": _snapshot(
            dataset_dir / "final_hash_manifest.json", "dataset final manifest"
        ),
        "calibration_final_hash_manifest": _snapshot(
            calibration_dir / "final_hash_manifest.json", "calibration final manifest"
        ),
        "resource_manifest": resource_manifest,
        "trajectory_receipt": trajectory_receipt_snapshot,
        "trajectory_storage_manifest": storage_snapshot,
        "resource_config": _snapshot(config_path, "resource config"),
        "resource_mesh": _snapshot(mesh_path, "resource mesh"),
        "su2_executable": executable,
        **{
            f"train_frame_{index}": FileSnapshot(frame.path, frame.bytes, frame.sha256)
            for index, frame in native_frames.items()
        },
        **{
            f"pilot_source_{offset}": _snapshot(
                REPO_ROOT / relative, f"pilot source {relative}"
            )
            for offset, relative in enumerate(PILOT_SOURCE_FILES)
        },
    }
    for label, snapshot in snapshots.items():
        _verify_snapshot(snapshot, label)
    return PilotAuthority(
        extension=extension,
        normalization=normalization,
        clean_states=clean_states,
        native_frames=native_frames,
        coordinates=coordinates,
        config_path=config_path,
        mesh_path=mesh_path,
        executable_path=executable.path,
        snapshots=snapshots,
        authority_summary={
            "extension_contract_sha256": EXTENSION_CONTRACT_FILE_SHA256,
            "extension_preregistration_sha256": EXTENSION_PREREGISTRATION_SHA256,
            "r0_contract_sha256": R0_CONTRACT_SHA256,
            "successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "normalization_payload_sha256": dataset_manifest["normalization"][
                "canonical_payload_sha256"
            ],
            "calibration_final_hash_manifest_sha256": calibration_sha256,
            "calibration_payload_sha256": calibration["canonical_payload_sha256"],
            "resource_manifest_sha256": resource_manifest.sha256,
            "trajectory_receipt_sha256": trajectory_receipt_snapshot.sha256,
            "trajectory_storage_manifest_sha256": storage_snapshot.sha256,
            "su2_executable_sha256": executable.sha256,
            "pilot_sources": {
                relative: snapshots[f"pilot_source_{offset}"].to_mapping()
                for offset, relative in enumerate(PILOT_SOURCE_FILES)
            },
        },
    )


def _reverify_authority(authority: PilotAuthority) -> bool:
    try:
        for label, snapshot in authority.snapshots.items():
            _verify_snapshot(snapshot, label)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def generate_pilot_directions(num_nodes: int) -> dict[int, np.ndarray]:
    """Generate selected directions from the frozen full train-center stream."""

    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASE_SEED)
    scale = torch.tensor(PER_FIELD_NORMALIZED_STD, dtype=torch.float32)
    selected: dict[int, np.ndarray] = {}
    for center in TRAIN_CENTERS:
        direction = torch.randn(
            (2, num_nodes, len(NACA_DYNAMIC_FIELDS)),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        direction.mul_(scale)
        if center in PILOT_CENTERS:
            selected[center] = direction.numpy().copy()
    if tuple(selected) != PILOT_CENTERS:
        raise AssertionError("pilot direction stream did not select all centers")
    return selected


def _render_config(source_text: str, overrides: Mapping[str, str]) -> str:
    rendered: list[str] = []
    observed: set[str] = set()
    for raw_line in source_text.splitlines():
        active = raw_line.split("%", maxsplit=1)[0].strip()
        if "=" in active:
            key = active.split("=", maxsplit=1)[0].strip().upper()
            if key in overrides:
                if key in observed:
                    raise ValueError(f"duplicate relabel override target {key}")
                rendered.append(f"{key}= {overrides[key]}")
                observed.add(key)
                continue
        rendered.append(raw_line)
    missing = [key for key in overrides if key not in observed]
    if missing:
        rendered.extend(("", "% Frozen train-only relabeling overrides"))
        rendered.extend(f"{key}= {overrides[key]}" for key in missing)
    return "\n".join(rendered) + "\n"


def _file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if root is not None:
        record["relative_path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return record


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    little_endian = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "little_endian_c_order_sha256": sha256(
            little_endian.tobytes(order="C")
        ).hexdigest(),
    }


def _input_states(
    authority: PilotAuthority,
    case: PilotCase,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    step = case.step
    clean = torch.from_numpy(
        np.stack(
            (
                authority.clean_states[step.previous_index],
                authority.clean_states[step.current_index],
            ),
            axis=0,
        ).astype(np.float32, copy=False)
    )
    normalized_direction = torch.from_numpy(direction)
    if normalized_direction.dtype != torch.float32 or tuple(
        normalized_direction.shape
    ) != tuple(clean.shape):
        raise ValueError("pilot direction differs from the BDF2 state shape")
    if case.sign == 0:
        displaced = clean.clone()
    else:
        state_scale = torch.tensor(
            authority.normalization.state_scale, dtype=torch.float32
        ).reshape(1, 1, -1)
        displaced = clean + state_scale * (float(case.sign) * normalized_direction)
    displaced_array = displaced.numpy().copy()
    clean_array = clean.numpy()
    state_scale64 = authority.normalization.state_scale.reshape(1, 1, -1)
    realized = (displaced_array.astype(np.float64) - clean_array.astype(np.float64)) / (
        state_scale64
    )
    field_rms = np.sqrt(np.mean(np.square(realized), axis=(0, 1)))
    return (
        displaced_array[0],
        displaced_array[1],
        {
            "clean_float32": _array_record(clean_array),
            "displaced_float32": _array_record(displaced_array),
            "normalized_direction": _array_record(direction),
            "realized_field_rms": field_rms.tolist(),
            "realized_field_rms_ratio": (field_rms / PER_FIELD_NORMALIZED_STD).tolist(),
        },
    )


def _case_template_indices(case: PilotCase) -> tuple[int, int]:
    step = case.step
    if case.variant == "auxiliary_swap":
        return step.current_index, step.previous_index
    return step.previous_index, step.current_index


def _case_checks() -> dict[str, bool]:
    return {name: False for name in REQUIRED_RECEIPT_CHECKS}


def _run_case(
    *,
    staging_root: Path,
    authority: PilotAuthority,
    case: PilotCase,
    direction: np.ndarray,
    timeout_seconds: float,
    process_runner: Callable[..., Any],
) -> CaseExecution:
    case_root = staging_root / case.name
    case_root.mkdir()
    step = case.step
    checks = _case_checks()
    details: dict[str, Any] = {"case": case.to_mapping()}
    error_message: str | None = None
    output_state: np.ndarray | None = None
    output_path: Path | None = None
    realized_field_rms: np.ndarray | None = None
    input_paths = [case_root / name for name in step.input_filenames]
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    staged_tracked_paths: tuple[Path, ...] = ()
    staged_inputs_before: dict[str, Any] | None = None
    config_path: Path | None = None
    config_before: dict[str, Any] | None = None
    try:
        previous, current, displacement = _input_states(authority, case, direction)
        realized_field_rms = np.asarray(
            displacement["realized_field_rms"], dtype=np.float64
        )
        details["displacement"] = displacement
        input_reports = [
            naca_state_admissibility(previous),
            naca_state_admissibility(current),
        ]
        details["input_admissibility"] = input_reports
        checks["input_finite"] = all(
            report["finite"]["passed"] for report in input_reports
        )
        checks["input_positive_density"] = all(
            report["density"]["passed"] for report in input_reports
        )
        checks["input_positive_pressure"] = all(
            report["ideal_gas_pressure"]["passed"] for report in input_reports
        )
        if not all(
            checks[name]
            for name in (
                "input_finite",
                "input_positive_density",
                "input_positive_pressure",
            )
        ):
            raise ValueError("displaced BDF2 input failed admissibility without redraw")

        template_indices = _case_template_indices(case)
        writes = []
        for state, template_index, destination in zip(
            (previous, current), template_indices, input_paths, strict=True
        ):
            template = authority.native_frames[template_index]
            writes.append(
                write_restart_from_verified_template(
                    template_path=template.path,
                    expected_template_sha256=template.sha256,
                    destination_path=destination.resolve(),
                    authoritative_state=state,
                )
            )
        details["restart_writes"] = writes
        checks["restart_write_verified"] = all(
            record["source_immutable"]
            and record["non_evolved_fields_bitwise_preserved"]
            and record["replaced_fields"] == list(NACA_DYNAMIC_FIELDS)
            for record in writes
        )

        staged_mesh = case_root / authority.mesh_path.name
        staged_source_config = case_root / "upstream_unsteady_naca0012.cfg"
        shutil.copyfile(authority.mesh_path, staged_mesh)
        shutil.copyfile(authority.config_path, staged_source_config)
        staged_tracked_paths = (
            *input_paths,
            staged_mesh,
            staged_source_config,
        )
        staged_inputs_before = {
            path.name: _file_record(path) for path in staged_tracked_paths
        }
        overrides = {
            **relabel_config_overrides(case.center),
            "MESH_FILENAME": staged_mesh.name,
        }
        config_path = case_root / PILOT_CONFIG_FILENAME
        config_path.write_text(
            _render_config(
                staged_source_config.read_text(encoding="utf-8", errors="strict"),
                overrides,
            ),
            encoding="utf-8",
            newline="\n",
        )
        parsed = parse_su2_config(config_path)
        if any(parsed.get(key) != value for key, value in overrides.items()):
            raise ValueError("rendered relabel config differs from frozen overrides")
        config_before = _file_record(config_path)
        details["config"] = {**config_before, "overrides": overrides}

        environment = os.environ.copy()
        selected_environment = {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"}
        environment.update(selected_environment)
        command = [
            str(authority.executable_path),
            "--threads",
            "1",
            PILOT_CONFIG_FILENAME,
        ]
        details["process"] = {
            "attempted": False,
            "argv": ["<bound-su2-executable>", "--threads", "1", PILOT_CONFIG_FILENAME],
            "execution_mode": "direct_single_mpi_rank_one_openmp_thread",
            "selected_environment": selected_environment,
            "return_code": None,
            "duration_seconds": None,
            "timeout_seconds": timeout_seconds,
        }
        started = time.perf_counter()
        with (
            stdout_path.open("x", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("x", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            details["process"]["attempted"] = True
            try:
                completed = process_runner(
                    command,
                    cwd=case_root,
                    env=environment,
                    shell=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                )
            finally:
                details["process"]["duration_seconds"] = time.perf_counter() - started
        return_code = getattr(completed, "returncode", None)
        checks["process_exit_zero"] = return_code == 0
        details["process"]["return_code"] = return_code
        if not checks["process_exit_zero"]:
            raise RuntimeError(f"SU2 exited with code {return_code}")

        expected_output = case_root / step.output_filename
        output_candidates = sorted(
            path for path in case_root.glob("*.dat") if path not in input_paths
        )
        checks["exactly_one_expected_output"] = output_candidates == [expected_output]
        if not checks["exactly_one_expected_output"]:
            raise ValueError("solver did not produce exactly one expected restart")
        history_candidates = sorted(case_root.glob("*.csv"))
        expected_history = case_root / step.history_filename
        if history_candidates != [expected_history]:
            raise ValueError("solver did not produce exactly one expected history")

        output_state, output_record = validate_relabel_output(
            expected_output.resolve(),
            step=step,
            expected_coordinates=authority.coordinates,
        )
        history_record = validate_relabel_history(expected_history.resolve(), step=step)
        output_path = expected_output
        details["output"] = output_record
        details["history"] = history_record
        checks["output_finite"] = output_record["admissibility"]["finite"]["passed"]
        checks["output_positive_density"] = output_record["admissibility"]["density"][
            "passed"
        ]
        checks["output_positive_pressure"] = output_record["admissibility"][
            "ideal_gas_pressure"
        ]["passed"]
        checks["convergence_relrms_density"] = history_record["convergence"]["passed"]

        staged_inputs_after = {
            path.name: _file_record(path)
            for path in (*input_paths, staged_mesh, staged_source_config)
        }
        if (
            staged_inputs_after != staged_inputs_before
            or _file_record(config_path) != config_before
        ):
            checks["restart_write_verified"] = False
            raise RuntimeError("solver changed a staged input or frozen config")
    except Exception as error:  # noqa: BLE001 - preserve every failed case
        error_message = f"{type(error).__name__}: {error}"
        details["exception"] = error_message
    if staged_inputs_before is not None and config_path is not None:
        try:
            staged_inputs_after = {
                path.name: _file_record(path) for path in staged_tracked_paths
            }
            staged_unchanged = staged_inputs_after == staged_inputs_before
            config_unchanged = (
                config_before is not None and _file_record(config_path) == config_before
            )
        except (OSError, TypeError, ValueError):
            staged_unchanged = False
            config_unchanged = False
        if not staged_unchanged or not config_unchanged:
            checks["restart_write_verified"] = False
            if error_message is None:
                error_message = "RuntimeError: solver changed a staged input or config"
    checks["source_immutable"] = _reverify_authority(authority)
    if not checks["source_immutable"] and error_message is None:
        error_message = "RuntimeError: verified authority changed during the case"
    receipt_path = case_root / "receipt.json"
    receipt = write_fail_closed_receipt(
        receipt_path.resolve(),
        center=case.center,
        checks=checks,
        details=details,
        error=error_message,
    )
    verified_receipt = load_verified_relabel_receipt(receipt_path.resolve())
    if verified_receipt != receipt:
        raise RuntimeError("case receipt changed during verification")
    if receipt["scientifically_usable"] is not True:
        output_state = None
        output_path = None
    return CaseExecution(
        case=case,
        directory=case_root,
        receipt=receipt,
        output_state=output_state,
        output_path=output_path,
        realized_field_rms=realized_field_rms,
    )


def _state_normalized_rms(
    difference: np.ndarray, normalization: NACANormalization
) -> tuple[float, np.ndarray]:
    array = np.asarray(difference, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(NACA_DYNAMIC_FIELDS):
        raise ValueError("state difference must have shape [N,5]")
    normalized = array / normalization.state_scale.reshape(1, -1)
    per_field = np.sqrt(np.mean(np.square(normalized), axis=0))
    overall = math.sqrt(float(np.mean(np.square(normalized))))
    return overall, per_field


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(
        np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )
    )


def evaluate_pilot_gates(
    executions: Sequence[CaseExecution], authority: PilotAuthority
) -> tuple[dict[str, bool], dict[str, Any]]:
    schedule = pilot_cases()
    if tuple(execution.case for execution in executions) != schedule:
        raise ValueError("executed cases differ from the frozen schedule")
    if any(
        execution.receipt["scientifically_usable"] is not True
        for execution in executions
    ):
        raise ValueError("a failed case cannot enter pilot metric evaluation")
    by_key = {
        (execution.case.center, execution.case.sign, execution.case.variant): execution
        for execution in executions
    }

    realized_records: list[dict[str, Any]] = []
    realized_passed = True
    for center in PILOT_CENTERS:
        for sign in (-1, 1):
            observed = by_key[(center, sign, "base")].realized_field_rms
            if observed is None:
                raise ValueError("signed pilot case lacks realized displacement")
            ratios = observed / PER_FIELD_NORMALIZED_STD
            passed = bool(
                np.all(ratios >= REALIZED_RMS_RATIO_RANGE[0])
                and np.all(ratios <= REALIZED_RMS_RATIO_RANGE[1])
            )
            realized_passed = realized_passed and passed
            realized_records.append(
                {
                    "center": center,
                    "sign": sign,
                    "field_rms": observed.tolist(),
                    "ratios": ratios.tolist(),
                    "passed": passed,
                }
            )

    zero_records: list[dict[str, Any]] = []
    zero_passed = True
    zero_outputs: dict[int, np.ndarray] = {}
    for center in PILOT_CENTERS:
        execution = by_key[(center, 0, "base")]
        if execution.output_state is None:
            raise ValueError("zero-control case lacks output state")
        zero_outputs[center] = execution.output_state
        overall, per_field = _state_normalized_rms(
            execution.output_state - authority.clean_states[center + 1],
            authority.normalization,
        )
        passed = bool(
            overall <= ZERO_OVERALL_LIMIT and np.all(per_field <= ZERO_FIELD_LIMITS)
        )
        zero_passed = zero_passed and passed
        zero_records.append(
            {
                "center": center,
                "overall_state_normalized_rms": overall,
                "per_field_state_normalized_rms": per_field.tolist(),
                "passed": passed,
            }
        )

    response_records: list[dict[str, Any]] = []
    response_values: list[float] = []
    for center in PILOT_CENTERS:
        for sign in (-1, 1):
            output = by_key[(center, sign, "base")].output_state
            if output is None:
                raise ValueError("signed pilot case lacks output state")
            overall, per_field = _state_normalized_rms(
                output - zero_outputs[center], authority.normalization
            )
            response_values.append(overall)
            response_records.append(
                {
                    "center": center,
                    "sign": sign,
                    "overall_state_normalized_rms": overall,
                    "per_field_state_normalized_rms": per_field.tolist(),
                }
            )
    median_response = float(np.median(np.asarray(response_values, dtype=np.float64)))
    largest_zero = max(
        record["overall_state_normalized_rms"] for record in zero_records
    )
    response_passed = bool(median_response > MIN_RESPONSE_RATIO * largest_zero)
    ratio = None if largest_zero == 0.0 else median_response / largest_zero

    base = by_key[(1075, 1, "base")]
    repeat = by_key[(1075, 1, "repeat")]
    auxiliary = by_key[(1075, 1, "auxiliary_swap")]
    if any(
        execution.output_state is None or execution.output_path is None
        for execution in (base, repeat, auxiliary)
    ):
        raise ValueError("comparison case lacks a validated output")
    repeat_bytes = base.output_path.read_bytes() == repeat.output_path.read_bytes()
    repeat_evolved = _bitwise_equal(base.output_state, repeat.output_state)
    auxiliary_evolved = _bitwise_equal(base.output_state, auxiliary.output_state)
    auxiliary_full = base.output_path.read_bytes() == auxiliary.output_path.read_bytes()
    repeat_passed = repeat_bytes and repeat_evolved
    auxiliary_passed = auxiliary_evolved

    gates = {
        "all_registered_cases": len(executions) == 11,
        "all_case_receipts": all(
            execution.receipt["scientifically_usable"] is True
            for execution in executions
        ),
        "realized_displacement_scale": realized_passed,
        "zero_control": zero_passed,
        "repeatability": repeat_passed,
        "auxiliary_invariance": auxiliary_passed,
        "response_separation": response_passed,
        "authority_immutable": _reverify_authority(authority),
    }
    metrics = {
        "realized_displacement": {
            "required_ratio_inclusive": list(REALIZED_RMS_RATIO_RANGE),
            "cases": realized_records,
        },
        "zero_control": {
            "reference": "stored_native_float64_clean_successor",
            "overall_limit": ZERO_OVERALL_LIMIT,
            "per_field_limits": ZERO_FIELD_LIMITS.tolist(),
            "cases": zero_records,
        },
        "trusted_response": {
            "definition": (
                "state_normalized_equal_entry_rms_signed_output_minus_same_center_zero_output"
            ),
            "values": response_records,
            "median": median_response,
            "largest_zero_control_overall_rms": largest_zero,
            "median_to_largest_zero_ratio": ratio,
            "required_strict_ratio": MIN_RESPONSE_RATIO,
            "passed": response_passed,
        },
        "repeatability": {
            "full_native_output_file_bytes_equal": repeat_bytes,
            "five_evolved_output_arrays_bitwise_equal": repeat_evolved,
            "history_policy": "convergence_contract_only_not_bitwise",
            "passed": repeat_passed,
        },
        "auxiliary_probe": {
            "templates": "swap_native_1074_and_1075_auxiliary_columns_between_history_slots",
            "five_evolved_output_arrays_bitwise_equal": auxiliary_evolved,
            "full_native_output_file_bytes_equal_record_only": auxiliary_full,
            "passed": auxiliary_passed,
        },
    }
    return gates, metrics


def _write_direction_arrays(
    root: Path, directions: Mapping[int, np.ndarray]
) -> dict[str, Any]:
    centers = np.asarray(PILOT_CENTERS, dtype=np.int64)
    values = np.stack([directions[center] for center in PILOT_CENTERS], axis=0)
    centers_path = root / "pilot_centers.npy"
    values_path = root / "pilot_normalized_directions.npy"
    np.save(centers_path, centers, allow_pickle=False)
    np.save(values_path, values, allow_pickle=False)
    return {
        "centers": _file_record(centers_path, root),
        "normalized_directions": {
            **_file_record(values_path, root),
            **_array_record(values),
            "stream": "one_ascending_center_stream_956_through_1193",
            "base_seed": BASE_SEED,
            "torch_version": torch.__version__,
        },
    }


def _write_final_hash_manifest(root: Path) -> tuple[dict[str, Any], str]:
    files = {
        path.resolve().relative_to(root.resolve()).as_posix(): _file_record(path, root)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != PILOT_FINAL_HASH_FILENAME
    }
    manifest = {
        "schema": PILOT_FINAL_HASH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    path = root / PILOT_FINAL_HASH_FILENAME
    _atomic_json(path, manifest)
    for relative, record in files.items():
        observed = _file_record(root / relative, root)
        if observed != record:
            raise RuntimeError(f"pilot artifact changed during closure: {relative}")
    return manifest, sha256_file(path)


def run_su2_naca0012_relabel_pilot(
    *,
    extension_contract_path: str | Path,
    extension_preregistration_path: str | Path,
    successor_contract_path: str | Path,
    baseline_contract_path: str | Path,
    dataset_dir: str | Path,
    calibration_dir: str | Path,
    trajectory_dir: str | Path,
    resource_manifest_path: str | Path,
    resource_dir: str | Path,
    executable_path: str | Path,
    output_dir: str | Path,
    timeout_seconds: float = 3600.0,
    process_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify authority, run at most the frozen eleven cases, and close a packet."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    paths = {
        "extension_contract_path": _require_absolute(
            extension_contract_path, "extension_contract_path"
        ),
        "extension_preregistration_path": _require_absolute(
            extension_preregistration_path, "extension_preregistration_path"
        ),
        "successor_contract_path": _require_absolute(
            successor_contract_path, "successor_contract_path"
        ),
        "baseline_contract_path": _require_absolute(
            baseline_contract_path, "baseline_contract_path"
        ),
        "dataset_dir": _require_absolute(dataset_dir, "dataset_dir"),
        "calibration_dir": _require_absolute(calibration_dir, "calibration_dir"),
        "trajectory_dir": _require_absolute(trajectory_dir, "trajectory_dir"),
        "resource_manifest_path": _require_absolute(
            resource_manifest_path, "resource_manifest_path"
        ),
        "resource_dir": _require_absolute(resource_dir, "resource_dir"),
        "executable_path": _require_absolute(executable_path, "executable_path"),
    }
    output = _require_absolute(output_dir, "output_dir").resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent is absent or aliased")
    authority = _load_pilot_authority(**paths)
    staging = output.with_name(f".{output.name}.{uuid4().hex}.staging")
    staging.mkdir()
    runner = process_runner or subprocess.run
    attempted_solver_call_count = 0

    def counted_runner(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempted_solver_call_count
        attempted_solver_call_count += 1
        return runner(*args, **kwargs)

    executions: list[CaseExecution] = []
    direction_records: dict[str, Any] = {}
    gates = {
        "all_registered_cases": False,
        "all_case_receipts": False,
        "realized_displacement_scale": False,
        "zero_control": False,
        "repeatability": False,
        "auxiliary_invariance": False,
        "response_separation": False,
        "authority_immutable": False,
    }
    metrics: dict[str, Any] = {}
    error_message: str | None = None
    try:
        directions = generate_pilot_directions(authority.coordinates.shape[0])
        direction_records = _write_direction_arrays(staging, directions)
        for case in pilot_cases():
            execution = _run_case(
                staging_root=staging,
                authority=authority,
                case=case,
                direction=directions[case.center],
                timeout_seconds=timeout_seconds,
                process_runner=counted_runner,
            )
            executions.append(execution)
            if execution.receipt["scientifically_usable"] is not True:
                error_message = (
                    f"registered case {case.name} failed; remaining cases were not run"
                )
                break
        if error_message is None:
            gates, metrics = evaluate_pilot_gates(executions, authority)
            if not all(gates.values()):
                failed = [name for name, passed in gates.items() if not passed]
                error_message = "pilot gates failed: " + ", ".join(failed)
    except Exception as error:  # noqa: BLE001 - close a fail-closed packet
        error_message = f"{type(error).__name__}: {error}"
        gates["authority_immutable"] = _reverify_authority(authority)

    gates["authority_immutable"] = _reverify_authority(authority)
    succeeded = bool(error_message is None and all(gates.values()))
    receipt = _self_hashed(
        {
            "schema": PILOT_RECEIPT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "pilot_succeeded" if succeeded else "pilot_failed",
            "scientifically_usable": succeeded,
            "paired_query_bank_unlocked": succeeded,
            "error": error_message,
            "registered_case_count": 11,
            "attempted_solver_call_count": attempted_solver_call_count,
            "completed_case_receipt_count": len(executions),
            "schedule": [case.to_mapping() for case in pilot_cases()],
            "case_receipts": [
                {
                    "case": execution.case.to_mapping(),
                    "relative_path": (
                        execution.directory.relative_to(staging).as_posix()
                        + "/receipt.json"
                    ),
                    "status": execution.receipt["status"],
                    "scientifically_usable": execution.receipt["scientifically_usable"],
                    "canonical_payload_sha256": execution.receipt[
                        "canonical_payload_sha256"
                    ],
                }
                for execution in executions
            ],
            "directions": direction_records,
            "gates": gates,
            "metrics": metrics,
            "authority": dict(authority.authority_summary),
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "rng_device": "cpu",
            },
            "failure_policy": (
                "stop_without_silent_drop_replacement_or_inner_iteration_change"
            ),
            "nu_tilde_acceptance_gate": False,
            "online_solver_calls": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    _atomic_json(staging / PILOT_RECEIPT_FILENAME, receipt)
    final_manifest, final_sha256 = _write_final_hash_manifest(staging)
    os.replace(staging, output)
    stored = _load_self_hashed_json(
        output / PILOT_RECEIPT_FILENAME, PILOT_RECEIPT_SCHEMA
    )
    if stored != receipt:
        raise RuntimeError("pilot receipt changed during packet publication")
    if sha256_file(output / PILOT_FINAL_HASH_FILENAME) != final_sha256:
        raise RuntimeError("pilot final-hash manifest changed during publication")
    for relative, record in final_manifest["files"].items():
        if _file_record(output / relative, output) != record:
            raise RuntimeError(f"pilot artifact changed during publication: {relative}")
    return {
        "receipt": stored,
        "packet": {
            "directory": output.name,
            "final_hash_manifest_sha256": final_sha256,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    docs = REPO_ROOT / "docs/time_dependent_no"
    parser.add_argument(
        "--extension-contract",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json",
    )
    parser.add_argument(
        "--extension-preregistration",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md",
    )
    parser.add_argument(
        "--successor-contract",
        type=Path,
        default=docs / "B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json",
    )
    parser.add_argument(
        "--baseline-contract",
        type=Path,
        default=docs / "R0_NACA_PCNO_BASELINE_CONTRACT.json",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--resource-manifest",
        type=Path,
        default=docs / "R0_SU2_NACA_RESOURCE_MANIFEST.json",
    )
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--su2-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        result = run_su2_naca0012_relabel_pilot(
            extension_contract_path=arguments.extension_contract,
            extension_preregistration_path=arguments.extension_preregistration,
            successor_contract_path=arguments.successor_contract,
            baseline_contract_path=arguments.baseline_contract,
            dataset_dir=arguments.dataset_dir,
            calibration_dir=arguments.calibration_dir,
            trajectory_dir=arguments.trajectory_dir,
            resource_manifest_path=arguments.resource_manifest,
            resource_dir=arguments.resource_dir,
            executable_path=arguments.su2_executable,
            output_dir=arguments.output_dir,
            timeout_seconds=arguments.timeout_seconds,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA0012 relabeling pilot failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["receipt"]["scientifically_usable"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
