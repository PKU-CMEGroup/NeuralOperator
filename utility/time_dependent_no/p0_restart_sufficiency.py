"""Fail-closed P0-A1 plumbing for restart-sufficiency experiments.

P0-A1 is deliberately narrower than the stored-state restart experiment.  It
verifies frozen manifests and exact input bytes, and exercises synthetic
shape, timing, admissibility, and boundary-accounting contracts.  It never
deserializes a checkpoint, constructs a model, or advances the native solver.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    state_is_admissible,
)

A0_ARTIFACT_MANIFEST_SHA256 = (
    "14b5b48c8d6ea754341ece82cb1e3cf26ef53c350807b2620df89c1eb77d8eb7"
)
A0_MEMBER_MAPPING_SHA256 = (
    "393a5035ee9f32b38df8700d8f6808a005397a9fb3d46f6a498ea45139d29bda"
)
CASE_MANIFEST_SHA256 = (
    "1452e03f905b1fd7a50ba7f2c09143faec259b5cab08e12a0d2538deebb554f7"
)
NATIVE_SOLVER_MANIFEST_SHA256 = (
    "9424413ecd99d2980b2f6cfbae9cfc06c442a07d3237b2c6caa072be423525e4"
)
NATIVE_SOLVER_MAPPING_SHA256 = (
    "293af994cfd3656f9ea72120bec9d32a921c6f3a4939f71daa9ae67fdd48e8cf"
)
DATASET_MANIFEST_SHA256 = (
    "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c"
)
SPLIT_DIGEST = "ae4be7f0e5fcfb305106173625c62c9a96634a716061207afb35fc9bda724d5a"

A1_SOURCE_MANIFEST_SCHEMA = "p0_restart_sufficiency_a1_source_manifest_v1"
A1_SUMMARY_SCHEMA = "p0_restart_sufficiency_a1_preflight_v1"
A1_ARTIFACT_MANIFEST_SCHEMA = "p0_restart_sufficiency_a1_artifact_manifest_v1"
CASE_MANIFEST_SCHEMA = "p0_restart_sufficiency_case_manifest_v1"
ARTIFACT_MANIFEST_SCHEMA = "p0_restart_sufficiency_artifact_manifest_v1"
NATIVE_SOLVER_MANIFEST_SCHEMA = "p0_native_solver_source_manifest_v1"
EXECUTION_CONTRACT_SCHEMA = "p0_restart_sufficiency_execution_contract_v1"

CHANNEL_ORDER = ("rho", "rho*u", "rho*v", "E")
CHANNEL_SCALES = (0.0752340287, 0.0546168404, 0.0435805767, 0.2345080528)
NATIVE_NX = 250
NATIVE_NY = 100
NATIVE_NODES = NATIVE_NX * NATIVE_NY
STORED_FRAMES = 61
STORED_FRAME_SPACING = 0.01
COMMON_HORIZON = 0.02
BOUNDARY_BALANCE_TOLERANCE = 1.0e-10

CHECKPOINT_RECORDS = {
    "checkpoints/d044/best.pt": {
        "bytes": 229_920_355,
        "sha256": ("c5e468c7045bf5ff8ccdd5f222c19bd63dab15af54b0461a58e26af5e17f678f"),
    },
    "checkpoints/d060/best.pt": {
        "bytes": 229_920_931,
        "sha256": ("95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9"),
    },
}

FIXED_CASES: tuple[dict[str, Any], ...] = (
    {
        "trajectory_id": "sv_e06_y00",
        "frame": 10,
        "physical_start_time": 0.1,
        "split": "validation",
        "split_group_id": "position_ood_y00",
        "state_relative_path": ("traj_sv_e06_y00_de0eaf81/states_conservative.npy"),
        "state_shape": [61, 25000, 4],
        "state_dtype": "<f4",
        "state_file_sha256": (
            "d1efd778a6fd47878b06685375de79c788003df360ca9492073a0cb22fc9c792"
        ),
        "input_frame_sha256": (
            "6681c8c8008180d3a31ce3863ed608bb50b1872eb26fd03e70bb699336c6cbec"
        ),
        "reference_f_plus_2_sha256": (
            "3cac3b8c2cd4c258330916d45052881d253462f0dbc33f2582665df73c516700"
        ),
        "eligibility_f_plus_4_sha256": (
            "e9c5aabe790943d67fee956d8ecf4c4ca3acb97c446fb09e21ddbcf611ec563d"
        ),
    },
    {
        "trajectory_id": "sv_e11_y08",
        "frame": 30,
        "physical_start_time": 0.3,
        "split": "validation",
        "split_group_id": "position_ood_y08",
        "state_relative_path": ("traj_sv_e11_y08_70576637/states_conservative.npy"),
        "state_shape": [61, 25000, 4],
        "state_dtype": "<f4",
        "state_file_sha256": (
            "356cf53a302c55774a9ff350ba61a6c90b9aa04df945df601dfb9be6be1149bc"
        ),
        "input_frame_sha256": (
            "e7f4f4ffd8b4cf270a509034a1b6539df86e68b6393f0cb9e83283fd77772896"
        ),
        "reference_f_plus_2_sha256": (
            "22a15515bace5297188e8297c04cbd9f4cabf4f5b17455f36af25d55889a7167"
        ),
        "eligibility_f_plus_4_sha256": (
            "d19970950fee4b0508165cc3c4371c12d528cf341071c5301f4d3f46180254d4"
        ),
    },
    {
        "trajectory_id": "sv_e05_y00",
        "frame": 50,
        "physical_start_time": 0.5,
        "split": "validation",
        "split_group_id": "position_ood_y00",
        "state_relative_path": ("traj_sv_e05_y00_331b8ff3/states_conservative.npy"),
        "state_shape": [61, 25000, 4],
        "state_dtype": "<f4",
        "state_file_sha256": (
            "2a7d2595d93ad3035acd3aa9096bbbfe61ee67f5bb92e23daf425c5e981e9c75"
        ),
        "input_frame_sha256": (
            "b4a807dca89e71694c571bbb8bc979e0f1fabb64d27f87292dc7fdd6dea20878"
        ),
        "reference_f_plus_2_sha256": (
            "5d57105f5c06c319ea0a0aeaa3a66e9e2d9dbb62604dec6518bab08fc06d9a44"
        ),
        "eligibility_f_plus_4_sha256": (
            "0348fe8d86d9fc686f6e1e13ac045edd84d54329812c50be12c364731b5170f4"
        ),
    },
)

A1_REQUIRED_SOURCE_MEMBERS = frozenset(
    {
        "utility/time_dependent_no/p0_restart_sufficiency.py",
        "scripts/time_dependent_no/preflight_p0_restart_sufficiency.py",
        "tests/time_dependent_no/test_p0_restart_sufficiency.py",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {Path(path).name}")
    return payload


def load_bound_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return load_json_object(path)


def resolve_manifest_member(root: Path, relative_path: str) -> Path:
    """Resolve one manifest member without permitting an escape from ``root``."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe manifest member path: {relative_path!r}")
    root = root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"manifest member escapes its root: {relative_path!r}"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"manifest member is not a file: {relative_path!r}")
    return candidate


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def validate_artifact_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("unexpected A0 artifact-manifest schema")
    if payload.get("root") != "inputs":
        raise ValueError("A0 artifact manifest must be rooted at inputs")
    if payload.get("self_exclusion") != "manifests/artifact_manifest.json":
        raise ValueError("unexpected artifact-manifest self exclusion")
    members = payload.get("members")
    if not isinstance(members, dict) or len(members) != 1945:
        raise ValueError("A0 artifact manifest must contain exactly 1,945 members")
    if payload.get("member_count") != len(members):
        raise ValueError("artifact member_count does not match members")

    byte_total = 0
    for relative, record in members.items():
        if not isinstance(relative, str):
            raise ValueError("artifact member names must be strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact member path: {relative!r}")
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise ValueError(f"invalid artifact record for {relative}")
        if not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise ValueError(f"invalid byte count for {relative}")
        _validate_sha256(record["sha256"], label=f"{relative} SHA-256")
        byte_total += record["bytes"]
    if byte_total != 4_415_212_852 or payload.get("total_member_bytes") != byte_total:
        raise ValueError("artifact byte total does not match the frozen A0 receipt")
    observed_mapping = canonical_json_sha256(members)
    if observed_mapping != A0_MEMBER_MAPPING_SHA256:
        raise ValueError("artifact member mapping does not match frozen A0")
    if payload.get("canonical_member_mapping_sha256") != observed_mapping:
        raise ValueError("artifact mapping digest is internally inconsistent")

    for relative, expected in CHECKPOINT_RECORDS.items():
        if members.get(relative) != expected:
            raise ValueError(f"checkpoint record drifted: {relative}")
    for case in FIXED_CASES:
        relative = "data/full_trajectory_root/" + case["state_relative_path"]
        expected = {
            "bytes": 24_400_128,
            "sha256": case["state_file_sha256"],
        }
        if members.get(relative) != expected:
            raise ValueError(f"selected state record drifted: {case['trajectory_id']}")
    required_manifests = {
        "manifests/case_manifest.json": {
            "bytes": 3057,
            "sha256": CASE_MANIFEST_SHA256,
        },
        "manifests/native_solver_source_manifest.json": {
            "bytes": 1398,
            "sha256": NATIVE_SOLVER_MANIFEST_SHA256,
        },
    }
    for relative, expected in required_manifests.items():
        if members.get(relative) != expected:
            raise ValueError(f"required manifest record drifted: {relative}")
    return members


def verify_artifact_member(
    *,
    inputs_root: Path,
    artifact_members: Mapping[str, Any],
    relative_path: str,
) -> Path:
    """Rehash one physically opened A0 member against the frozen inventory."""

    record = artifact_members.get(relative_path)
    if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
        raise ValueError(f"missing artifact record for {relative_path}")
    path = resolve_manifest_member(inputs_root, relative_path)
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"artifact member size drifted: {relative_path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact member bytes drifted: {relative_path}")
    return path


def validate_case_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if payload.get("schema") != CASE_MANIFEST_SCHEMA:
        raise ValueError("unexpected case-manifest schema")
    if payload.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256:
        raise ValueError("case manifest references the wrong dataset")
    if payload.get("open_grouped_split_digest") != SPLIT_DIGEST:
        raise ValueError("case manifest references the wrong split")
    if payload.get("eligible_validation_count") != 24:
        raise ValueError("eligible validation population drifted")
    if payload.get("access_receipt") != {
        "historical_test_members_referenced": False,
        "only_selected_state_paths_referenced": True,
        "selected_splits": ["validation"],
        "strength_ood_members_referenced": False,
    }:
        raise ValueError("case-manifest access boundary is not closed")
    if payload.get("selection_contract") != {
        "assigned_frames": [10, 30, 50],
        "population": "open validation only",
        "selection_performed_before_frame_bytes_were_read": True,
        "sort_key": 'sha256("P0-RS-20260825|" + trajectory_id)',
        "take": 3,
    }:
        raise ValueError("field-blind selection contract drifted")
    cases = payload.get("cases")
    if cases != list(FIXED_CASES):
        raise ValueError("case inventory or fixed case metadata drifted")
    return FIXED_CASES


def validate_execution_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != EXECUTION_CONTRACT_SCHEMA:
        raise ValueError("unexpected execution-contract schema")
    expected_state = {
        "channel_order": list(CHANNEL_ORDER),
        "channel_scales": list(CHANNEL_SCALES),
        "domain": [[0.0, 2.0], [0.0, 1.0]],
        "flattening": "index = y * 250 + x",
        "gamma": 1.4,
        "input_transform": "none",
        "shape": [NATIVE_NODES, 4],
    }
    if payload.get("state") != expected_state:
        raise ValueError("state shape/order/channel contract drifted")
    expected_horizon = {
        "d044": "two raw stride-1 residual calls",
        "d060": "one raw stride-2 residual call",
        "delta_t": COMMON_HORIZON,
        "native_solver": {
            "output_times": [0.0, COMMON_HORIZON],
            "t_final": COMMON_HORIZON,
        },
        "reference": "stored frame f+2",
        "stored_frame_spacing": STORED_FRAME_SPACING,
    }
    if payload.get("common_horizon") != expected_horizon:
        raise ValueError("one-stride timing contract drifted")
    roots = payload.get("source_import_roots_must_remain_separate")
    if not isinstance(roots, dict) or set(roots) != {
        "historical_model_evaluator",
        "native_solver",
        "reason",
    }:
        raise ValueError(
            "historical evaluator and native solver roots are not separated"
        )


def _validate_hash_mapping(
    mapping: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{label} must be a nonempty object")
    validated: dict[str, str] = {}
    for relative, digest in mapping.items():
        if not isinstance(relative, str):
            raise ValueError(f"{label} member names must be strings")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe {label} member path: {relative!r}")
        validated[relative] = _validate_sha256(digest, label=f"{relative} SHA-256")
    return validated


def validate_a1_source_manifest(
    payload: Mapping[str, Any],
    *,
    source_root: Path,
) -> dict[str, str]:
    if payload.get("schema") != A1_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unexpected A1 source-manifest schema")
    members = _validate_hash_mapping(payload.get("members"), label="A1 source")
    if payload.get("member_count") != len(members):
        raise ValueError("A1 source member_count is inconsistent")
    mapping_digest = canonical_json_sha256(members)
    if payload.get("canonical_member_mapping_sha256") != mapping_digest:
        raise ValueError("A1 source mapping digest is inconsistent")
    if not A1_REQUIRED_SOURCE_MEMBERS.issubset(members):
        missing = sorted(A1_REQUIRED_SOURCE_MEMBERS.difference(members))
        raise ValueError(f"A1 source manifest is missing required members: {missing}")
    observed_python = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file()
    }
    if observed_python != set(members):
        raise ValueError("A1 source root and manifested Python inventory differ")
    for relative, expected in members.items():
        path = resolve_manifest_member(source_root, relative)
        if sha256_file(path) != expected:
            raise ValueError(f"A1 source member drifted: {relative}")
    return members


def validate_native_solver_manifest(
    payload: Mapping[str, Any],
    *,
    source_root: Path,
) -> dict[str, str]:
    if payload.get("schema") != NATIVE_SOLVER_MANIFEST_SCHEMA:
        raise ValueError("unexpected native-solver source-manifest schema")
    if payload.get("missing_project_imports") != []:
        raise ValueError("native-solver source closure has unresolved imports")
    if payload.get("dynamic_import_sites") != []:
        raise ValueError("native-solver source closure has dynamic imports")
    if payload.get("registered_seed_member_count") != 3:
        raise ValueError("native-solver seed inventory drifted")
    members = _validate_hash_mapping(
        payload.get("transitive_members"), label="native-solver source"
    )
    if len(members) != 10 or payload.get("transitive_member_count") != 10:
        raise ValueError("native-solver closure must contain exactly ten members")
    mapping_digest = canonical_json_sha256(members)
    if mapping_digest != NATIVE_SOLVER_MAPPING_SHA256:
        raise ValueError("native-solver mapping does not match frozen A0")
    if payload.get("canonical_transitive_mapping_sha256") != mapping_digest:
        raise ValueError("native-solver source digest is internally inconsistent")
    for relative, expected in members.items():
        path = resolve_manifest_member(source_root, relative)
        if sha256_file(path) != expected:
            raise ValueError(f"native-solver source member drifted: {relative}")
    return members


def prepare_solver_input(stored_frame: np.ndarray) -> np.ndarray:
    """Copy one raw stored frame without changing its bytes or representation."""

    frame = np.asarray(stored_frame)
    if frame.shape != (NATIVE_NODES, 4):
        raise ValueError(f"expected state shape {(NATIVE_NODES, 4)}, got {frame.shape}")
    if frame.dtype.str != "<f4":
        raise ValueError(f"expected little-endian float32 input, got {frame.dtype.str}")
    if not np.isfinite(frame).all():
        raise ValueError("stored solver input contains a non-finite value")
    prepared = np.array(frame, copy=True, order="C")
    if not prepared.flags.c_contiguous:
        raise AssertionError("prepared solver input is not C-contiguous")
    if raw_array_sha256(prepared) != raw_array_sha256(frame):
        raise AssertionError("solver-input preparation changed stored bytes")
    return prepared


def replay_native_layout(state: np.ndarray) -> np.ndarray:
    """Round-trip the registered row-major ``y * nx + x`` layout."""

    state = np.asarray(state)
    if state.shape != (NATIVE_NODES, 4):
        raise ValueError(f"expected state shape {(NATIVE_NODES, 4)}, got {state.shape}")
    grid = state.reshape(NATIVE_NY, NATIVE_NX, 4, order="C")
    replayed = grid.reshape(NATIVE_NODES, 4, order="C")
    if not np.array_equal(state, replayed, equal_nan=True):
        raise AssertionError("row-major layout replay changed the state")
    return replayed


def boundary_balance_residual(
    delta_integral: Sequence[float] | np.ndarray,
    boundary_exchange: Sequence[float] | np.ndarray,
) -> float:
    delta = np.asarray(delta_integral, dtype=np.float64)
    exchange = np.asarray(boundary_exchange, dtype=np.float64)
    if delta.shape != (4,) or exchange.shape != (4,):
        raise ValueError("boundary accounting requires two four-channel vectors")
    if not np.isfinite(delta).all() or not np.isfinite(exchange).all():
        raise ValueError("boundary accounting inputs must be finite")
    numerator = np.linalg.norm(delta + exchange)
    denominator = max(np.linalg.norm(delta), np.linalg.norm(exchange), 1.0)
    return float(numerator / denominator)


def require_boundary_balance(
    delta_integral: Sequence[float] | np.ndarray,
    boundary_exchange: Sequence[float] | np.ndarray,
) -> float:
    residual = boundary_balance_residual(delta_integral, boundary_exchange)
    if residual > BOUNDARY_BALANCE_TOLERANCE:
        raise ValueError(
            f"boundary-balance residual {residual:.6e} exceeds "
            f"{BOUNDARY_BALANCE_TOLERANCE:.1e}"
        )
    return residual


def validate_synthetic_admissibility_rejection() -> dict[str, bool]:
    config = ShockVortexFVConfig(
        nx=NATIVE_NX,
        ny=NATIVE_NY,
        coarse_nx=NATIVE_NX,
        coarse_ny=NATIVE_NY,
        t_final=COMMON_HORIZON,
        output_times=(0.0, COMMON_HORIZON),
    ).validated()
    base = torch.tensor([[1.0, 0.2, 0.0, 2.52]], dtype=torch.float64)
    invalid_nonfinite = base.clone()
    invalid_nonfinite[0, 1] = torch.nan
    invalid_density = base.clone()
    invalid_density[0, 0] = -1.0
    invalid_pressure = base.clone()
    invalid_pressure[0, 3] = 0.0
    checks = {
        "admissible_state_accepted": state_is_admissible(base, config),
        "nonfinite_state_rejected": not state_is_admissible(invalid_nonfinite, config),
        "negative_density_rejected": not state_is_admissible(invalid_density, config),
        "nonpositive_pressure_rejected": not state_is_admissible(
            invalid_pressure, config
        ),
    }
    if not all(checks.values()):
        raise AssertionError("synthetic admissibility rejection contract failed")
    return checks


def _validate_selected_case(
    *,
    inputs_root: Path,
    artifact_members: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    relative = "data/full_trajectory_root/" + str(case["state_relative_path"])
    state_path = resolve_manifest_member(inputs_root, relative)
    artifact_record = artifact_members[relative]
    if state_path.stat().st_size != artifact_record["bytes"]:
        raise ValueError(f"selected state size drifted: {case['trajectory_id']}")
    if sha256_file(state_path) != case["state_file_sha256"]:
        raise ValueError(f"selected state file drifted: {case['trajectory_id']}")

    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    if list(states.shape) != case["state_shape"]:
        raise ValueError(f"selected state shape drifted: {case['trajectory_id']}")
    if states.dtype.str != case["state_dtype"]:
        raise ValueError(f"selected state dtype drifted: {case['trajectory_id']}")
    frame = int(case["frame"])
    input_frame = states[frame]
    reference_frame = states[frame + 2]
    eligibility_frame = states[frame + 4]
    observed_hashes = {
        "input": raw_array_sha256(input_frame),
        "reference_f_plus_2": raw_array_sha256(reference_frame),
        "eligibility_f_plus_4": raw_array_sha256(eligibility_frame),
    }
    expected_hashes = {
        "input": case["input_frame_sha256"],
        "reference_f_plus_2": case["reference_f_plus_2_sha256"],
        "eligibility_f_plus_4": case["eligibility_f_plus_4_sha256"],
    }
    if observed_hashes != expected_hashes:
        raise ValueError(f"selected frame bytes drifted: {case['trajectory_id']}")

    prepared = prepare_solver_input(input_frame)
    replayed = replay_native_layout(prepared)
    prepared_hash = raw_array_sha256(replayed)
    if prepared_hash != case["input_frame_sha256"]:
        raise AssertionError("solver input bytes differ from the selected stored frame")
    config = ShockVortexFVConfig(
        nx=NATIVE_NX,
        ny=NATIVE_NY,
        coarse_nx=NATIVE_NX,
        coarse_ny=NATIVE_NY,
        t_final=COMMON_HORIZON,
        output_times=(0.0, COMMON_HORIZON),
    ).validated()
    admissible = state_is_admissible(
        torch.as_tensor(prepared, dtype=torch.float64, device="cpu"), config
    )
    if not admissible:
        raise ValueError(f"selected input is inadmissible: {case['trajectory_id']}")
    return {
        "trajectory_id": case["trajectory_id"],
        "frame": frame,
        "state_file_sha256": case["state_file_sha256"],
        "stored_input_sha256": case["input_frame_sha256"],
        "prepared_solver_input_sha256": prepared_hash,
        "solver_input_bytes_equal_stored_input_bytes": True,
        "shape": list(prepared.shape),
        "dtype": prepared.dtype.str,
        "c_contiguous": bool(prepared.flags.c_contiguous),
        "row_major_layout_replayed": True,
        "admissible_fp64_check": True,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _prepare_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("A1 output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ValueError("A1 output directory must be fresh and empty")
    else:
        output_dir.mkdir(parents=True)


def run_a1_preflight(
    *,
    inputs_root: Path,
    source_root: Path,
    a1_source_manifest_path: Path,
    output_dir: Path,
    owner_authorized_a1: bool,
) -> tuple[dict[str, Any], Path]:
    """Run the bounded P0-A1 preflight and write its immutable-ready receipt."""

    if not owner_authorized_a1:
        raise ValueError("P0-A1 requires explicit owner authorization")
    inputs_root = inputs_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    artifact_path = resolve_manifest_member(
        inputs_root, "manifests/artifact_manifest.json"
    )
    case_path = resolve_manifest_member(inputs_root, "manifests/case_manifest.json")
    native_manifest_path = resolve_manifest_member(
        inputs_root, "manifests/native_solver_source_manifest.json"
    )
    execution_path = resolve_manifest_member(
        inputs_root, "manifests/execution_contract.json"
    )

    artifact = load_bound_json(artifact_path, A0_ARTIFACT_MANIFEST_SHA256)
    artifact_members = validate_artifact_manifest(artifact)
    cases = validate_case_manifest(load_bound_json(case_path, CASE_MANIFEST_SHA256))
    verify_artifact_member(
        inputs_root=inputs_root,
        artifact_members=artifact_members,
        relative_path="manifests/execution_contract.json",
    )
    validate_execution_contract(load_json_object(execution_path))
    a1_source_manifest = load_json_object(a1_source_manifest_path)
    a1_members = validate_a1_source_manifest(
        a1_source_manifest, source_root=source_root
    )
    native_manifest = load_bound_json(
        native_manifest_path, NATIVE_SOLVER_MANIFEST_SHA256
    )
    native_members = validate_native_solver_manifest(
        native_manifest, source_root=source_root
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    synthetic_layout = np.arange(
        NATIVE_NODES * len(CHANNEL_ORDER), dtype=np.float32
    ).reshape(NATIVE_NODES, len(CHANNEL_ORDER))
    replay_native_layout(synthetic_layout)
    admissibility_checks = validate_synthetic_admissibility_rejection()
    synthetic_delta = np.asarray([1.25, -0.75, 0.5, 2.0], dtype=np.float64)
    boundary_residual = require_boundary_balance(synthetic_delta, -synthetic_delta)
    case_receipts = [
        _validate_selected_case(
            inputs_root=inputs_root,
            artifact_members=artifact_members,
            case=case,
        )
        for case in cases
    ]

    summary: dict[str, Any] = {
        "schema": A1_SUMMARY_SCHEMA,
        "stage": "P0-A1",
        "status": "pass",
        "claim_boundary": (
            "Synthetic and exact-input plumbing only; no checkpoint, model, "
            "native-solver advance, training, or scientific metric was executed."
        ),
        "authorization": {
            "owner_authorized_a1": True,
            "a2_authorized": False,
        },
        "activity": {
            "checkpoint_files_opened": False,
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "native_solver_executed": False,
            "training_executed": False,
            "scientific_metric_computed": False,
            "field_selection_performed": False,
            "selected_state_files_opened": 3,
        },
        "bound_inputs": {
            "a0_artifact_manifest_sha256": A0_ARTIFACT_MANIFEST_SHA256,
            "a0_member_mapping_sha256": A0_MEMBER_MAPPING_SHA256,
            "case_manifest_sha256": CASE_MANIFEST_SHA256,
            "native_solver_manifest_sha256": NATIVE_SOLVER_MANIFEST_SHA256,
            "native_solver_mapping_sha256": NATIVE_SOLVER_MAPPING_SHA256,
            "a1_source_manifest_sha256": sha256_file(a1_source_manifest_path),
            "a1_source_mapping_sha256": canonical_json_sha256(a1_members),
            "a1_source_member_count": len(a1_members),
            "native_solver_source_member_count": len(native_members),
        },
        "contracts": {
            "channel_order": list(CHANNEL_ORDER),
            "channel_scales": list(CHANNEL_SCALES),
            "shape": [NATIVE_NODES, 4],
            "flattening": "index = y * 250 + x",
            "stored_frame_spacing": STORED_FRAME_SPACING,
            "common_horizon": COMMON_HORIZON,
            "reference_offset_frames": 2,
            "d044_calls": 2,
            "d044_stride": 1,
            "d060_calls": 1,
            "d060_stride": 2,
            "native_solver_output_times": [0.0, COMMON_HORIZON],
            "input_transform": "none",
        },
        "synthetic_checks": {
            "shape_order_channel_replay": True,
            "one_stride_timing": True,
            "boundary_balance_residual": boundary_residual,
            "boundary_balance_tolerance": BOUNDARY_BALANCE_TOLERANCE,
            **admissibility_checks,
        },
        "cases": case_receipts,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "admissibility_check_dtype": "float64",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    summary["payload_sha256"] = canonical_json_sha256(summary)

    _prepare_output_directory(output_dir)
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    artifact_members_out = {
        "summary.json": {
            "bytes": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        }
    }
    output_manifest = {
        "schema": A1_ARTIFACT_MANIFEST_SCHEMA,
        "root": "output_dir",
        "self_exclusion": "artifact_manifest.json",
        "member_count": len(artifact_members_out),
        "total_member_bytes": sum(
            record["bytes"] for record in artifact_members_out.values()
        ),
        "members": artifact_members_out,
        "canonical_member_mapping_sha256": canonical_json_sha256(artifact_members_out),
    }
    atomic_write_json(output_dir / "artifact_manifest.json", output_manifest)
    return summary, summary_path
