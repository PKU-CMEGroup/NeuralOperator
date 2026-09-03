"""Build the train-only calibration packet for the NACA corrective successor."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
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
    SOURCE_FILES as R0_SOURCE_FILES,
)
from scripts.time_dependent_no.prepare_pcno_naca0012_dataset import (
    captured_source_records,
)

CONTRACT_SCHEMA = "time_dependent_no.naca_corrective_successor_contract.v1"
EXPERIMENT_ID = "B3B4_NACA_CM_20260901A"
SUCCESSOR_CONTRACT_FILE_SHA256 = (
    "b0f373d42ca98c0761a61f17310233d22461da1add28b8c8d20359ef7e1d6715"
)
DATASET_FINAL_HASH_MANIFEST_SHA256 = (
    "1cb5fd2d751bdf1ca27cde5e1a1457973a525c19a98dbadd7cc3738c3efe2e4b"
)
EXPECTED_PARENT_SEEDS = (17, 29, 43)
SUCCESSOR_SOURCE_FILES = (
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
    "utility/time_dependent_no/pcno_naca0012_successor.py",
    "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md",
    "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_CONTRACT.json",
    "scripts/time_dependent_no/calibrate_pcno_naca0012_successor.py",
)
SOURCE_FILES = (*R0_SOURCE_FILES, *SUCCESSOR_SOURCE_FILES)


def _read_source_records(
    source_files: Sequence[str], repository_root: Path
) -> dict[str, dict[str, Any]]:
    """Read every source once so byte count and digest describe one snapshot."""

    records: dict[str, dict[str, Any]] = {}
    for relative in source_files:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"required calibration source is absent or aliased: {relative}"
            )
        payload = path.read_bytes()
        records[relative] = {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    return records


# R0 captures its closure before importing repository code. Capture every
# successor-only dependency before importing or using those modules below.
SOURCE_RECORDS_AT_IMPORT = {
    **captured_source_records(),
    **_read_source_records(SUCCESSOR_SOURCE_FILES, REPO_ROOT),
}
if tuple(SOURCE_RECORDS_AT_IMPORT) != SOURCE_FILES:
    raise RuntimeError("calibration import-time source inventory differs")

from scripts.time_dependent_no.evaluate_pcno_naca0012 import (
    _reverify_dataset_packet,
    _reverify_packet_files,
    _verify_training_packet,
)
from scripts.time_dependent_no.train_pcno_naca0012 import (
    DATASET_SCHEMA,
    FINAL_HASH_MANIFEST_SCHEMA,
    _fourier_slice,
    _geometry_slice,
    _load_dataset,
)
from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    VerifiedNACAGeometry,
    load_naca_baseline_contract,
    predict_normalized_residual,
)
from utility.time_dependent_no.pcno_naca0012_successor import (
    fit_recovery_calibration,
    fit_train_path_projector,
    validate_successor_math_contract,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    node_highpass_field,
)
from utility.time_dependent_no.su2_restart_contract import (
    sha256_file,
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _verify_live_sources() -> None:
    observed = _read_source_records(SOURCE_FILES, REPO_ROOT)
    if observed != SOURCE_RECORDS_AT_IMPORT:
        raise RuntimeError("calibration source files changed during execution")


def _read_stable_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent or aliased")
    payload = path.read_bytes()
    if path.read_bytes() != payload:
        raise RuntimeError(f"{label} changed while reading")
    return payload


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "canonical_payload_sha256" in result:
        raise ValueError("canonical payload hash was supplied twice")
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"calibration artifact is absent or aliased: {path}")
    return {"bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "array_sha256": _array_digest(array),
    }


def _reverify_bound_inputs(
    dataset_root: Path,
    dataset_final_sha256: str,
    parent_packets: Sequence[tuple[int, Path, Mapping[str, Mapping[str, Any]], str]],
) -> None:
    """Reverify every retained input packet immediately before promotion."""

    if tuple(sorted(seed for seed, _, _, _ in parent_packets)) != EXPECTED_PARENT_SEEDS:
        raise ValueError("retained parent packet seed inventory differs")
    _reverify_dataset_packet(dataset_root, dataset_final_sha256)
    for seed, root, files, final_sha256 in parent_packets:
        if "best.pt" not in files:
            raise ValueError(f"retained parent checkpoint is absent: {seed}")
        _reverify_packet_files(root, files, final_sha256)


def _verify_staged_packet(
    root: Path,
    expected_calibration: Mapping[str, Any],
    expected_final: Mapping[str, Any],
) -> None:
    """Reopen and close the exact three-file calibration packet."""

    if root.is_symlink():
        raise ValueError("staged calibration packet is aliased")
    packet_root = root.resolve()
    expected_names = {
        "calibration.json",
        "calibration_arrays.npz",
        "final_hash_manifest.json",
    }
    if (
        not packet_root.is_dir()
        or {path.name for path in packet_root.iterdir()} != expected_names
    ):
        raise ValueError("staged calibration packet inventory differs")

    final_path = packet_root / "final_hash_manifest.json"
    final_bytes = _read_stable_regular_file(
        final_path, "staged calibration final-hash manifest"
    )
    final = json.loads(final_bytes)
    if not isinstance(final, dict):
        raise TypeError("staged calibration final-hash manifest is not an object")
    files = final.get("files")
    if (
        not isinstance(files, Mapping)
        or set(files) != {"calibration.json", "calibration_arrays.npz"}
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
    ):
        raise ValueError("staged calibration final-hash closure differs")
    if final != dict(expected_final):
        raise ValueError("staged calibration final-hash manifest differs")

    snapshots: dict[str, bytes] = {}
    for name in ("calibration.json", "calibration_arrays.npz"):
        record = files[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"relative_path", "bytes", "sha256"}
            or record.get("relative_path") != name
        ):
            raise ValueError(f"staged calibration file record differs: {name}")
        payload = _read_stable_regular_file(
            packet_root / name, f"staged calibration file {name}"
        )
        if len(payload) != record.get("bytes") or sha256(
            payload
        ).hexdigest() != record.get("sha256"):
            raise ValueError(f"staged calibration file hash differs: {name}")
        snapshots[name] = payload

    calibration = json.loads(snapshots["calibration.json"])
    if not isinstance(calibration, dict):
        raise TypeError("staged calibration metadata is not an object")
    observed_payload_sha256 = calibration.pop("canonical_payload_sha256", None)
    if observed_payload_sha256 != _canonical_sha256(calibration):
        raise ValueError("staged calibration self-hash differs")
    calibration["canonical_payload_sha256"] = observed_payload_sha256
    if calibration != dict(expected_calibration):
        raise ValueError("staged calibration metadata differs")
    arrays_record = calibration.get("arrays")
    if (
        not isinstance(arrays_record, Mapping)
        or arrays_record.get("file") != files["calibration_arrays.npz"]
        or not isinstance(arrays_record.get("members"), Mapping)
        or calibration.get("successor_contract_sha256")
        != final.get("successor_contract_sha256")
    ):
        raise ValueError("staged calibration cross-record binding differs")

    with np.load(
        io.BytesIO(snapshots["calibration_arrays.npz"]), allow_pickle=False
    ) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(
            arrays_record["members"]
        ):
            raise ValueError("staged calibration NPZ member inventory differs")
        for name in archive.files:
            value = np.array(archive[name], copy=True)
            record = arrays_record["members"][name]
            if not isinstance(record, Mapping) or record != _array_record(value):
                raise ValueError(f"staged calibration NPZ member differs: {name}")

    if {path.name for path in packet_root.iterdir()} != expected_names:
        raise RuntimeError("staged calibration packet changed during closure")
    if (
        _read_stable_regular_file(final_path, "staged calibration final-hash manifest")
        != final_bytes
    ):
        raise RuntimeError("staged calibration final-hash manifest changed")


def _load_successor_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_stable_regular_file(path, "successor contract")
    file_sha256 = sha256(payload).hexdigest()
    if file_sha256 != SUCCESSOR_CONTRACT_FILE_SHA256:
        raise ValueError("successor contract bytes differ from the frozen contract")
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported successor contract")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("successor experiment identity differs")
    if value.get("protection") != {
        "prospective_opened": False,
        "sealed_opened": False,
        "online_solver_calls": False,
        "online_defect_trigger": False,
    }:
        raise ValueError("successor protection boundary differs")
    calibration = value.get("calibration", {})
    if (
        calibration.get("fit_role") != "train_only"
        or calibration.get("parent_seeds") != list(EXPECTED_PARENT_SEEDS)
        or calibration.get("iid_multiplier") != 1.0
        or calibration.get("structured_rank") != 16
        or calibration.get("structured_pair_count_per_seed") != 237
        or calibration.get("path_pca_variance_target") != 0.999
        or calibration.get("path_pca_rank_cap") != 32
        or calibration.get("path_fit_frames_inclusive") != [956, 1193]
        or calibration.get("path_fit_state_count") != 238
    ):
        raise ValueError("successor calibration constants differ")
    preregistration = (
        REPO_ROOT
        / "docs/time_dependent_no/B3B4_NACA_CORRECTIVE_SUCCESSOR_PREREGISTRATION.md"
    )
    if sha256_file(preregistration) != value.get("preregistration_sha256"):
        raise ValueError("successor preregistration hash differs")
    return value, file_sha256


def _preflight_open_dataset(root: Path, successor: Mapping[str, Any]) -> None:
    """Validate identity and role metadata before any state array is loaded."""

    if root.is_symlink():
        raise ValueError("open dataset directory is aliased")
    dataset_root = root.resolve()
    if not dataset_root.is_dir():
        raise ValueError("open dataset directory is absent or aliased")
    dataset_contract = successor.get("dataset")
    if not isinstance(dataset_contract, Mapping):
        raise TypeError("successor dataset contract is invalid")
    if (
        dataset_contract.get("final_hash_manifest_sha256")
        != DATASET_FINAL_HASH_MANIFEST_SHA256
    ):
        raise ValueError("successor open-dataset identity differs")

    final_path = dataset_root / "final_hash_manifest.json"
    final_bytes = _read_stable_regular_file(
        final_path, "open dataset final-hash manifest"
    )
    if sha256(final_bytes).hexdigest() != DATASET_FINAL_HASH_MANIFEST_SHA256:
        raise ValueError("open dataset final-hash manifest differs")
    final = json.loads(final_bytes)
    files = final.get("files") if isinstance(final, Mapping) else None
    if (
        not isinstance(final, Mapping)
        or final.get("schema") != FINAL_HASH_MANIFEST_SCHEMA
        or final.get("contract_sha256") != successor.get("inherited_r0_contract_sha256")
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or not isinstance(files, Mapping)
    ):
        raise ValueError("open dataset final-hash manifest metadata differs")
    manifest_record = files.get("dataset_manifest.json")
    if (
        not isinstance(manifest_record, Mapping)
        or manifest_record.get("relative_path") != "dataset_manifest.json"
        or isinstance(manifest_record.get("bytes"), bool)
        or not isinstance(manifest_record.get("bytes"), int)
        or not isinstance(manifest_record.get("sha256"), str)
    ):
        raise ValueError("open dataset manifest record differs")

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest_bytes = _read_stable_regular_file(manifest_path, "open dataset manifest")
    if (
        len(manifest_bytes) != manifest_record["bytes"]
        or sha256(manifest_bytes).hexdigest() != manifest_record["sha256"]
    ):
        raise ValueError("open dataset manifest differs from final hashes")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("open dataset manifest is not an object")
    observed_payload_sha256 = manifest.pop("canonical_payload_sha256", None)
    if observed_payload_sha256 != _canonical_sha256(manifest):
        raise ValueError("open dataset manifest payload hash differs")
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("contract", {}).get("sha256")
        != successor.get("inherited_r0_contract_sha256")
        or manifest.get("access")
        != {
            "materialized_population_roles": ["train", "development"],
            "diagnostic_sets": ["native_replay_margin"],
            "prospective_opened": False,
            "sealed_opened": False,
        }
    ):
        raise ValueError("open dataset authority or access metadata differs")

    expected_roles = {
        "train": ([955, 1194], [956, 1193]),
        "development": ([1233, 1472], [1234, 1471]),
    }
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(expected_roles):
        raise ValueError("open dataset role inventory differs")
    for role, (frames, centers) in expected_roles.items():
        record = roles[role]
        if (
            not isinstance(record, Mapping)
            or record.get("owned_frame_indices_inclusive") != frames
            or record.get("frame_count") != 240
            or record.get("dense_transition_center_indices_inclusive") != centers
            or record.get("dense_transition_count") != 238
            or dataset_contract.get(f"{role}_frames_inclusive") != frames
            or dataset_contract.get(f"{role}_centers_inclusive") != centers
        ):
            raise ValueError(f"open dataset {role} role or frame inventory differs")
    if final_path.read_bytes() != final_bytes:
        raise RuntimeError("open dataset final-hash manifest changed during preflight")


def _unique_undirected_edges(directed: np.ndarray, num_nodes: int) -> np.ndarray:
    edges = np.asarray(directed, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("directed edges must have shape [E,2]")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("directed edge index is outside the mesh")
    ordered = np.sort(edges, axis=1)
    ordered = ordered[ordered[:, 0] != ordered[:, 1]]
    unique = np.unique(ordered, axis=0)
    if unique.shape[0] * 2 != edges.shape[0]:
        raise ValueError("NACA directed edges are not an exact symmetric doubling")
    return unique


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


def _teacher_forced_errors(
    *,
    model: torch.nn.Module,
    dataset: Any,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    geometry_batch = geometry.expand(4, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)
    state_scale = torch.as_tensor(
        normalization.state_scale, dtype=torch.float32, device=device
    )
    residual_scale = torch.as_tensor(
        normalization.residual_scale, dtype=torch.float32, device=device
    )
    errors: list[np.ndarray] = []
    centers: list[int] = []
    total_error_energy = 0.0
    total_increment_energy = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for previous, current, target, center in loader:
            size = int(previous.shape[0])
            prediction = predict_normalized_residual(
                model,
                previous.to(device),
                current.to(device),
                _geometry_slice(geometry_batch, size),
                normalization,
                fourier_tensors=_fourier_slice(fourier, size),
            )
            target_device = target.to(device)
            state_error = (prediction - target_device) * residual_scale / state_scale
            increment = target_device * residual_scale / state_scale
            if not torch.all(torch.isfinite(state_error)):
                raise FloatingPointError("teacher-forced state error is nonfinite")
            errors.append(state_error.cpu().numpy().astype(np.float32, copy=False))
            centers.extend(int(value) for value in center)
            total_error_energy += float(torch.sum(torch.square(state_error)).cpu())
            total_increment_energy += float(torch.sum(torch.square(increment)).cpu())
            count += int(state_error.numel())
    if centers != list(range(956, 1194)):
        raise ValueError("teacher-forced calibration centers differ")
    result = np.concatenate(errors, axis=0)
    if result.shape != (238, geometry.num_nodes, 5):
        raise ValueError("teacher-forced error bank shape differs")
    return result, {
        "normalized_state_error_rms": math.sqrt(total_error_energy / count),
        "normalized_clean_increment_rms": math.sqrt(total_increment_energy / count),
    }


def _highpass_summary(errors: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    highpass_energy = 0.0
    total_energy = 0.0
    for error in errors:
        highpass = node_highpass_field(error, edges)
        highpass_energy += float(np.sum(np.square(highpass, dtype=np.float64)))
        total_energy += float(np.sum(np.square(error, dtype=np.float64)))
    ratio = highpass_energy / total_energy if total_energy > 0.0 else math.nan
    return {
        "normalized_state_error_energy": total_energy,
        "graph_highpass_error_energy": highpass_energy,
        "graph_highpass_to_total_energy_ratio": ratio,
    }


def _build_arrays(
    *,
    error_banks: Mapping[int, np.ndarray],
    train_states: np.ndarray,
    train_frames: np.ndarray,
    normalization: NACANormalization,
    parent_contract: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stacked_errors = np.stack(
        [error_banks[seed] for seed in EXPECTED_PARENT_SEEDS], axis=0
    )
    recovery = fit_recovery_calibration(
        stacked_errors,
        range(956, 1194),
        EXPECTED_PARENT_SEEDS,
        parent_contract,
    )
    projector = fit_train_path_projector(
        train_states,
        train_frames,
        normalization,
        parent_contract,
    )
    arrays = {
        **{f"recovery__{name}": value for name, value in recovery.to_mapping().items()},
        **{f"path__{name}": value for name, value in projector.to_mapping().items()},
    }
    summary = {
        "iid_field_std": recovery.iid_field_rms.tolist(),
        "iid_expected_history_pair_squared_norm": (
            recovery.iid_expected_history_pair_energy
        ),
        "error_pair_count": recovery.pair_sample_count,
        "error_pair_dimension": 2 * recovery.num_nodes * 5,
        "error_pair_rank16_capture_fraction": (recovery.structured_captured_variance),
        "error_pair_energy_rescale": recovery.structured_energy_rescale,
        "error_pair_matched_expected_squared_norm": float(
            torch.sum(torch.square(recovery.structured_coefficient_std)).item()
        ),
        "structured_sampling_mean": "zero",
        "path_state_count": int(projector.normalized_path.shape[0]),
        "path_state_dimension": int(
            projector.normalized_path.shape[1] * projector.normalized_path.shape[2]
        ),
        "path_rank": projector.rank,
        "path_rank_without_cap": projector.variance_rank_without_cap,
        "path_rank_cap_active": projector.rank_cap_active,
        "path_retained_variance_fraction": projector.captured_variance,
    }
    return arrays, summary


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    _verify_live_sources()
    successor, successor_sha256 = _load_successor_contract(arguments.successor_contract)
    if sha256_file(arguments.baseline_contract) != successor.get(
        "inherited_r0_contract_sha256"
    ):
        raise ValueError("baseline contract hash differs from successor binding")
    baseline = load_naca_baseline_contract(arguments.baseline_contract)
    validate_successor_math_contract(successor, baseline)
    _preflight_open_dataset(arguments.dataset_dir, successor)
    (
        dataset_manifest,
        geometry,
        normalization,
        roles,
        dataset_packet_sha256,
    ) = _load_dataset(arguments.dataset_dir, baseline)
    if dataset_packet_sha256 != successor["dataset"]["final_hash_manifest_sha256"]:
        raise ValueError("dataset packet differs from successor binding")

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(arguments.training_dir) != 3:
        raise ValueError("exactly three parent training packets are required")

    unique_edges = _unique_undirected_edges(geometry.directed_edges, geometry.num_nodes)
    parent_records: dict[str, Any] = {}
    parent_packet_snapshots: list[
        tuple[int, Path, Mapping[str, Mapping[str, Any]], str]
    ] = []
    error_banks: dict[int, np.ndarray] = {}
    diagnostic: dict[str, Any] = {}
    for training_dir in arguments.training_dir:
        (
            seed,
            model,
            summary,
            final_sha256,
            packet_name,
            training_root,
            training_files,
        ) = _verify_training_packet(
            training_dir,
            dataset_manifest,
            dataset_packet_sha256,
            baseline,
            geometry,
            device,
        )
        expected_parent = successor["parent_r0"]["checkpoints"][str(seed)]
        if final_sha256 != expected_parent["final_hash_manifest_sha256"]:
            raise ValueError(f"parent training manifest differs for seed {seed}")
        checkpoint_path = training_dir.resolve() / "best.pt"
        if sha256_file(checkpoint_path) != expected_parent["checkpoint_sha256"]:
            raise ValueError(f"parent checkpoint differs for seed {seed}")
        if seed in error_banks:
            raise ValueError(f"duplicate parent seed: {seed}")
        errors, seed_summary = _teacher_forced_errors(
            model=model,
            dataset=roles["train"][2],
            geometry=geometry,
            normalization=normalization,
            device=device,
        )
        error_banks[seed] = errors
        diagnostic[str(seed)] = {
            **seed_summary,
            **_highpass_summary(errors, unique_edges),
        }
        parent_records[str(seed)] = {
            "packet_name": packet_name,
            "final_hash_manifest_sha256": final_sha256,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": summary["best_epoch"],
            "best_development_score": summary["best_development_score"],
        }
        parent_packet_snapshots.append(
            (seed, training_root, training_files, final_sha256)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if tuple(sorted(error_banks)) != EXPECTED_PARENT_SEEDS:
        raise ValueError("parent seed set differs")

    train_frames = np.asarray(roles["train"][1])
    if not np.array_equal(train_frames, np.arange(955, 1195, dtype=np.int64)):
        raise ValueError("train frame indices differ")
    path_frames = train_frames[1:-1]
    train_states = np.asarray(roles["train"][0])[1:-1]
    if train_states.shape != (238, geometry.num_nodes, 5):
        raise ValueError("train-state path geometry shape differs")
    arrays, calibration_summary = _build_arrays(
        error_banks=error_banks,
        train_states=train_states,
        train_frames=path_frames,
        normalization=normalization,
        parent_contract=baseline,
    )

    output = arguments.output_dir.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError("calibration output or staging path already exists")
    staging.mkdir(parents=True)
    arrays_path = staging / "calibration_arrays.npz"
    np.savez(arrays_path, **arrays)
    array_records = {name: _array_record(value) for name, value in arrays.items()}

    source_records = {
        relative: dict(record) for relative, record in SOURCE_RECORDS_AT_IMPORT.items()
    }
    source_set_sha256 = _canonical_sha256(source_records)
    calibration = _self_hashed(
        {
            "schema": successor["packet_schemas"]["calibration"],
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "successor_contract_sha256": successor_sha256,
            "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "parent_r0_evaluation_final_hash_manifest_sha256": successor["parent_r0"][
                "evaluation_final_hash_manifest_sha256"
            ],
            "parent_checkpoints": parent_records,
            "fit_role": "train_only",
            "teacher_forced_diagnostics": diagnostic,
            "calibration": calibration_summary,
            "pca": {
                "recovery_algorithm": "exact centered covariance-Gram eigendecomposition",
                "path_algorithm": "exact torch.linalg.svd",
            },
            "arrays": {
                "file": {
                    "relative_path": "calibration_arrays.npz",
                    **_file_record(arrays_path),
                },
                "members": array_records,
            },
            "source": {
                "files": source_records,
                "source_set_sha256": source_set_sha256,
            },
            "runtime": _runtime(device),
            "access": {
                "train_opened": True,
                "development_opened": True,
                "prospective_opened": False,
                "sealed_opened": False,
            },
            "claim_boundary": {
                "trusted_displaced_state_response_measured": False,
                "manifold_drift_established": False,
                "successor_rollout_opened": False,
            },
        }
    )
    calibration_path = staging / "calibration.json"
    _write_json(calibration_path, calibration)
    final = {
        "schema": successor["packet_schemas"]["final_hash_manifest"],
        "experiment_id": EXPERIMENT_ID,
        "packet_role": "calibration",
        "successor_contract_sha256": successor_sha256,
        "files": {
            "calibration.json": {
                "relative_path": "calibration.json",
                **_file_record(calibration_path),
            },
            "calibration_arrays.npz": {
                "relative_path": "calibration_arrays.npz",
                **_file_record(arrays_path),
            },
        },
        "self_hash_excluded": True,
        "prospective_opened": False,
        "sealed_opened": False,
    }
    _write_json(staging / "final_hash_manifest.json", final)
    _verify_live_sources()
    _reverify_bound_inputs(
        arguments.dataset_dir,
        dataset_packet_sha256,
        parent_packet_snapshots,
    )
    _verify_staged_packet(staging, calibration, final)
    os.replace(staging, output)
    return {
        "status": "complete",
        "output": output.name,
        "successor_contract_sha256": successor_sha256,
        "final_hash_manifest_sha256": sha256_file(output / "final_hash_manifest.json"),
        "calibration": calibration_summary,
        "prospective_opened": False,
        "sealed_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-contract", type=Path, required=True)
    parser.add_argument("--successor-contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _run(arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA successor calibration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
