"""Shared fail-closed contracts for the P0-A2 restart-sufficiency gate.

This module contains only artifact, metric, and repeatability plumbing.  The
native solver and historical model evaluator stay in separate import roots and
are called by separate entry points.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from utility.time_dependent_no.p0_restart_sufficiency import (
    BOUNDARY_BALANCE_TOLERANCE,
    CHANNEL_SCALES,
    FIXED_CASES,
    NATIVE_NODES,
    canonical_json_sha256,
    load_json_object,
    raw_array_sha256,
    resolve_manifest_member,
    sha256_file,
)

A1_CLOSEOUT_MANIFEST_SHA256 = (
    "8fd2192d84e287da98c890a653dda2f367791ec8d3b8ca889cd13626257f93c6"
)
A1_CLOSEOUT_MAPPING_SHA256 = (
    "35dfc745f07f19a9a3c2a2be4ad04e3ca2296835eb145f77864591f0a02b4c32"
)
A1_SUMMARY_PAYLOAD_SHA256 = (
    "addff7cb47f129d850226b7fdd1b596e91951d40e7d5b82b5290af9b64aaf330"
)

HISTORICAL_EVALUATOR_MAPPING_SHA256 = (
    "9bda94598e09cdabb56731ccbcfe8c52bd1a829b6cb169b478ed0952f946e675"
)
NATIVE_SOLVER_MAPPING_SHA256 = (
    "293af994cfd3656f9ea72120bec9d32a921c6f3a4939f71daa9ae67fdd48e8cf"
)

A2_SOURCE_MANIFEST_SCHEMA = "p0_restart_sufficiency_a2_source_manifest_v1"
A2_SOLVER_SUMMARY_SCHEMA = "p0_restart_sufficiency_a2_solver_process_v1"
A2_SOLVER_ARTIFACT_SCHEMA = "p0_restart_sufficiency_a2_solver_artifact_manifest_v1"
A2_MODEL_SUMMARY_SCHEMA = "p0_restart_sufficiency_a2_model_bias_gate_v1"
A2_MODEL_ARTIFACT_SCHEMA = "p0_restart_sufficiency_a2_model_artifact_manifest_v1"

PRIMARY_DISTANCE_TOLERANCE = 1.0e-12
BIAS_RATIO_LIMIT = 0.25
SOURCE_ROLES = ("native_solver", "historical_evaluator")
PROCESS_ROLES = ("primary", "fresh_repeat")

EXPECTED_BASE_MAPPING = {
    "native_solver": NATIVE_SOLVER_MAPPING_SHA256,
    "historical_evaluator": HISTORICAL_EVALUATOR_MAPPING_SHA256,
}
REQUIRED_A2_SOURCE_MEMBERS = {
    "native_solver": frozenset(
        {
            "utility/time_dependent_no/p0_restart_sufficiency.py",
            "utility/time_dependent_no/p0_restart_sufficiency_a2.py",
            "scripts/time_dependent_no/run_p0_restart_sufficiency_a2_solver.py",
            "tests/time_dependent_no/test_p0_restart_sufficiency_a2.py",
        }
    ),
    "historical_evaluator": frozenset(
        {
            "utility/time_dependent_no/p0_restart_sufficiency.py",
            "utility/time_dependent_no/p0_restart_sufficiency_a2.py",
            "scripts/time_dependent_no/evaluate_p0_restart_sufficiency_a2.py",
        }
    ),
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def prepare_fresh_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(path.iterdir()):
            raise ValueError("output directory must be fresh and empty")
    else:
        path.mkdir(parents=True)


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_member_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("artifact members must be an object")
    records: dict[str, dict[str, Any]] = {}
    for relative, record in value.items():
        if not isinstance(relative, str):
            raise ValueError("artifact member names must be strings")
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"unsafe artifact member path: {relative!r}")
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise ValueError(f"invalid artifact record: {relative}")
        if not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise ValueError(f"invalid artifact byte count: {relative}")
        _validate_sha256(record["sha256"], label=f"{relative} SHA-256")
        records[relative] = dict(record)
    return records


def build_artifact_manifest(
    root: Path,
    *,
    schema: str,
    self_exclusion: str = "artifact_manifest.json",
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    members: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == self_exclusion:
            continue
        members[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema": schema,
        "root": "output_dir",
        "self_exclusion": self_exclusion,
        "member_count": len(members),
        "total_member_bytes": sum(record["bytes"] for record in members.values()),
        "members": members,
        "canonical_member_mapping_sha256": canonical_json_sha256(members),
    }


def validate_artifact_tree(
    root: Path,
    *,
    expected_schema: str,
    require_read_only: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = root / "artifact_manifest.json"
    payload = load_json_object(manifest_path)
    if payload.get("schema") != expected_schema:
        raise ValueError("unexpected artifact-manifest schema")
    if payload.get("root") != "output_dir":
        raise ValueError("artifact manifest has the wrong root contract")
    if payload.get("self_exclusion") != "artifact_manifest.json":
        raise ValueError("artifact manifest has the wrong self exclusion")
    members = _validate_member_records(payload.get("members"))
    if payload.get("member_count") != len(members):
        raise ValueError("artifact member_count is inconsistent")
    if payload.get("total_member_bytes") != sum(
        record["bytes"] for record in members.values()
    ):
        raise ValueError("artifact byte total is inconsistent")
    if payload.get("canonical_member_mapping_sha256") != canonical_json_sha256(members):
        raise ValueError("artifact member mapping digest is inconsistent")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != set(members):
        raise ValueError("artifact tree and manifested inventory differ")
    for relative, record in members.items():
        path = resolve_manifest_member(root, relative)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"artifact size drifted: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact bytes drifted: {relative}")
        if require_read_only and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f"artifact remains writable: {relative}")
    if require_read_only and stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
        raise ValueError("artifact manifest remains writable")
    if require_read_only and stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise ValueError("artifact root remains writable")
    return payload


def validate_a1_closeout(a1_root: Path) -> dict[str, Any]:
    a1_root = a1_root.resolve(strict=True)
    manifest_path = a1_root / "execution/execution_artifact_manifest.json"
    if sha256_file(manifest_path) != A1_CLOSEOUT_MANIFEST_SHA256:
        raise ValueError("A1 closeout manifest does not match the promoted receipt")
    payload = load_json_object(manifest_path)
    if payload.get("schema") != "p0_restart_sufficiency_a1_closeout_manifest_v1":
        raise ValueError("unexpected A1 closeout-manifest schema")
    members = _validate_member_records(payload.get("members"))
    if len(members) != 9 or payload.get("member_count") != 9:
        raise ValueError("A1 closeout must contain exactly nine members")
    if canonical_json_sha256(members) != A1_CLOSEOUT_MAPPING_SHA256:
        raise ValueError("A1 closeout member mapping drifted")
    for relative, record in members.items():
        path = resolve_manifest_member(a1_root, relative)
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"A1 closeout member drifted: {relative}")
    receipt = load_json_object(a1_root / "execution/execution_receipt.json")
    if (
        receipt.get("attempt") != "P0-A1-v2"
        or receipt.get("status") != "complete_pass"
        or receipt.get("summary_payload_sha256") != A1_SUMMARY_PAYLOAD_SHA256
        or receipt.get("activity", {}).get("native_solver_executed") is not False
        or receipt.get("activity", {}).get("checkpoint_deserialized") is not False
    ):
        raise ValueError("promoted A1 execution receipt is not closed")
    return receipt


def _validate_hash_mapping(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a nonempty object")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if not isinstance(relative, str):
            raise ValueError(f"{label} member names must be strings")
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"unsafe {label} path: {relative!r}")
        result[relative] = _validate_sha256(digest, label=f"{relative} SHA-256")
    return result


def validate_a2_source_manifest(
    payload: Mapping[str, Any],
    *,
    source_root: Path,
    source_role: str,
) -> dict[str, str]:
    if source_role not in SOURCE_ROLES:
        raise ValueError(f"unsupported A2 source role: {source_role}")
    if payload.get("schema") != A2_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unexpected A2 source-manifest schema")
    if payload.get("source_role") != source_role:
        raise ValueError("A2 source role is inconsistent")
    base_members = _validate_hash_mapping(payload.get("base_members"), label="base")
    expected_base = EXPECTED_BASE_MAPPING[source_role]
    if canonical_json_sha256(base_members) != expected_base:
        raise ValueError("A2 base source mapping does not match A0")
    if payload.get("base_mapping_sha256") != expected_base:
        raise ValueError("A2 base source mapping declaration is inconsistent")
    members = _validate_hash_mapping(payload.get("members"), label="A2 source")
    if payload.get("member_count") != len(members):
        raise ValueError("A2 source member_count is inconsistent")
    if payload.get("canonical_member_mapping_sha256") != canonical_json_sha256(members):
        raise ValueError("A2 source mapping digest is inconsistent")
    if not set(base_members).issubset(members):
        raise ValueError("A2 source manifest omits a frozen base member")
    if any(members[name] != digest for name, digest in base_members.items()):
        raise ValueError("A2 source manifest changes a frozen base member")
    required = REQUIRED_A2_SOURCE_MEMBERS[source_role]
    if not required.issubset(members):
        raise ValueError(
            f"A2 source manifest misses required members: {sorted(required - set(members))}"
        )
    source_root = source_root.resolve(strict=True)
    observed = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file()
    }
    if observed != set(members):
        raise ValueError("A2 source root and manifested Python inventory differ")
    for relative, expected in members.items():
        if sha256_file(resolve_manifest_member(source_root, relative)) != expected:
            raise ValueError(f"A2 source member drifted: {relative}")
    return members


def scaled_rms_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray = CHANNEL_SCALES,
) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64)
    if first.shape != second.shape or first.shape != (weights.size, scale.size):
        raise ValueError("states, volumes, and component scales do not align")
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(scale).all()
        or np.any(weights <= 0.0)
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "scaled RMS inputs must be finite with positive weights/scales"
        )
    scaled = (first - second) / scale.reshape(1, -1)
    return float(
        np.sqrt(np.einsum("n,nc,nc->", weights, scaled, scaled) / weights.sum())
    )


def weighted_relative_l2_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    volumes: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or pred.shape != (weights.size, 4):
        raise ValueError("relative-L2 states and volumes do not align")
    error = pred - truth
    numerator = np.einsum("n,nc,nc->c", weights, error, error)
    denominator = np.einsum("n,nc,nc->c", weights, truth, truth)
    per_channel = [
        None if float(base) <= 0.0 else float(np.sqrt(value / base))
        for value, base in zip(numerator, denominator, strict=True)
    ]
    total_denominator = float(denominator.sum())
    return {
        "unscaled_physical_volume_relative_l2": (
            None
            if total_denominator <= 0.0
            else float(np.sqrt(float(numerator.sum()) / total_denominator))
        ),
        "per_channel_physical_volume_relative_l2": per_channel,
    }


def primitive_fields(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    conservative = np.asarray(state, dtype=np.float64)
    if conservative.ndim != 2 or conservative.shape[1] != 4:
        raise ValueError("conservative state must have shape [N,4]")
    rho = conservative[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = conservative[:, 1] / rho
        v = conservative[:, 2] / rho
        internal = (
            conservative[:, 3]
            - 0.5
            * (np.square(conservative[:, 1]) + np.square(conservative[:, 2]))
            / rho
        )
        pressure = (float(gamma) - 1.0) * internal
    return np.stack((rho, u, v, pressure), axis=-1)


def admissibility_summary(state: np.ndarray, *, gamma: float = 1.4) -> dict[str, Any]:
    conservative = np.asarray(state, dtype=np.float64)
    primitive = primitive_fields(conservative, gamma=gamma)
    rho = primitive[:, 0]
    pressure = primitive[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        internal = (
            conservative[:, 3]
            - 0.5
            * (np.square(conservative[:, 1]) + np.square(conservative[:, 2]))
            / conservative[:, 0]
        )
    finite = np.isfinite(conservative).all(axis=-1) & np.isfinite(primitive).all(
        axis=-1
    )
    admissible = finite & (rho > 0.0) & (internal > 0.0) & (pressure > 0.0)
    return {
        "finite": bool(finite.all()),
        "admissible": bool(admissible.all()),
        "admissible_fraction": float(admissible.mean()),
        "minimum_density": float(np.nanmin(rho)),
        "minimum_internal_energy": float(np.nanmin(internal)),
        "minimum_pressure": float(np.nanmin(pressure)),
    }


def primitive_relative_l2_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    volumes: np.ndarray,
    gamma: float = 1.4,
) -> dict[str, float | None]:
    pred = primitive_fields(prediction, gamma=gamma)
    truth = primitive_fields(reference, gamma=gamma)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if pred.shape[0] != weights.size:
        raise ValueError("primitive fields and volumes do not align")

    def relative(indices: tuple[int, ...]) -> float | None:
        delta = pred[:, indices] - truth[:, indices]
        baseline = truth[:, indices]
        numerator = float(np.einsum("n,nc,nc->", weights, delta, delta))
        denominator = float(np.einsum("n,nc,nc->", weights, baseline, baseline))
        return None if denominator <= 0.0 else float(np.sqrt(numerator / denominator))

    return {
        "density_relative_l2": relative((0,)),
        "velocity_vector_relative_l2": relative((1, 2)),
        "pressure_relative_l2": relative((3,)),
    }


def integrated_conservative_change(
    final_state: np.ndarray,
    initial_state: np.ndarray,
    *,
    volumes: np.ndarray,
) -> np.ndarray:
    final = np.asarray(final_state, dtype=np.float64)
    initial = np.asarray(initial_state, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if final.shape != initial.shape or final.shape != (weights.size, 4):
        raise ValueError("conservative states and volumes do not align")
    return np.einsum("n,nc->c", weights, final - initial)


def boundary_balance_residual(
    delta_integral: np.ndarray,
    boundary_exchange: np.ndarray,
) -> float:
    delta = np.asarray(delta_integral, dtype=np.float64)
    exchange = np.asarray(boundary_exchange, dtype=np.float64)
    if delta.shape != (4,) or exchange.shape != (4,):
        raise ValueError("boundary balance requires two four-channel vectors")
    if not np.isfinite(delta).all() or not np.isfinite(exchange).all():
        raise ValueError("boundary balance inputs must be finite")
    return float(
        np.linalg.norm(delta + exchange)
        / max(np.linalg.norm(delta), np.linalg.norm(exchange), 1.0)
    )


def bias_gate_row(
    *,
    baseline: float,
    d044: float,
    d060: float,
    separation: float,
) -> dict[str, Any]:
    values = (baseline, d044, d060, separation)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("bias-gate distances must be finite and nonnegative")
    model_denominator = min(d044, d060)
    model_resolved = model_denominator > 0.0
    separation_resolved = separation > 0.0
    model_ratio = baseline / model_denominator if model_resolved else None
    separation_ratio = baseline / separation if separation_resolved else None
    model_pass = model_ratio is not None and model_ratio <= BIAS_RATIO_LIMIT
    separation_pass = (
        separation_ratio is not None and separation_ratio <= BIAS_RATIO_LIMIT
    )
    return {
        "baseline_scaled_rms": baseline,
        "d044_scaled_rms": d044,
        "d060_scaled_rms": d060,
        "model_separation_scaled_rms": separation,
        "model_defect_denominator_resolved": model_resolved,
        "model_separation_denominator_resolved": separation_resolved,
        "baseline_over_min_model_defect": model_ratio,
        "baseline_over_model_separation": separation_ratio,
        "ratio_limit": BIAS_RATIO_LIMIT,
        "model_defect_gate_pass": model_pass,
        "model_separation_gate_pass": separation_pass,
        "pass": model_pass and separation_pass,
    }


def case_contract() -> list[dict[str, Any]]:
    return [
        {
            "trajectory_id": case["trajectory_id"],
            "frame": case["frame"],
            "input_frame_sha256": case["input_frame_sha256"],
            "reference_f_plus_2_sha256": case["reference_f_plus_2_sha256"],
        }
        for case in FIXED_CASES
    ]


def validate_solver_process_summary(
    payload: Mapping[str, Any], *, process_role: str
) -> list[dict[str, Any]]:
    if process_role not in PROCESS_ROLES:
        raise ValueError(f"unsupported process role: {process_role}")
    if payload.get("schema") != A2_SOLVER_SUMMARY_SCHEMA:
        raise ValueError("unexpected A2 solver-summary schema")
    if payload.get("stage") != "P0-A2" or payload.get("status") != "pass":
        raise ValueError("A2 solver process did not complete its gates")
    if payload.get("process_role") != process_role:
        raise ValueError("A2 solver process role is inconsistent")
    if payload.get("cases_contract") != case_contract():
        raise ValueError("A2 solver case contract drifted")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(FIXED_CASES):
        raise ValueError("A2 solver summary must contain exactly three cases")
    for expected, row in zip(FIXED_CASES, cases, strict=True):
        if (
            row.get("trajectory_id") != expected["trajectory_id"]
            or row.get("frame") != expected["frame"]
        ):
            raise ValueError("A2 solver case ordering drifted")
        if row.get("stored_input_sha256") != expected["input_frame_sha256"]:
            raise ValueError("A2 solver input hash drifted")
        if row.get("same_process_state_bytes_identical") is not True:
            raise ValueError("same-process solver states are not byte-identical")
        if row.get("same_process_exchange_bytes_identical") is not True:
            raise ValueError("same-process boundary exchange is not byte-identical")
        if row.get("rejected_attempts") != 0 or row.get("face_fallbacks") != 0:
            raise ValueError("A2 solver used a retry or face fallback")
        if row.get("boundary_balance_residual", math.inf) > BOUNDARY_BALANCE_TOLERANCE:
            raise ValueError("A2 solver boundary-balance gate failed")
        if row.get("finite") is not True or row.get("admissible") is not True:
            raise ValueError("A2 solver output is not finite and admissible")
    return [dict(row) for row in cases]


def compare_solver_process_summaries(
    primary: Mapping[str, Any],
    fresh_repeat: Mapping[str, Any],
    *,
    primary_states: Mapping[str, np.ndarray],
    repeat_states: Mapping[str, np.ndarray],
    volumes: np.ndarray,
) -> list[dict[str, Any]]:
    primary_rows = validate_solver_process_summary(primary, process_role="primary")
    repeat_rows = validate_solver_process_summary(
        fresh_repeat, process_role="fresh_repeat"
    )
    for key in (
        "a0_artifact_manifest_sha256",
        "a1_closeout_manifest_sha256",
        "source_manifest_sha256",
        "source_mapping_sha256",
        "solver_config_sha256",
    ):
        if primary.get("bindings", {}).get(key) != fresh_repeat.get("bindings", {}).get(
            key
        ):
            raise ValueError(f"solver process binding differs: {key}")
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(primary_rows, repeat_rows, strict=True):
        case_id = str(left["trajectory_id"])
        if left["stored_input_sha256"] != right["stored_input_sha256"]:
            raise ValueError("solver process input hashes differ")
        for key in ("accepted_steps", "rejected_attempts", "face_fallbacks"):
            if left.get(key) != right.get(key):
                raise ValueError(f"solver process count differs: {case_id}/{key}")
        distance = scaled_rms_distance(
            primary_states[case_id],
            repeat_states[case_id],
            volumes=volumes,
        )
        if distance > PRIMARY_DISTANCE_TOLERANCE:
            raise ValueError(
                f"fresh-process primary distance exceeds tolerance: {case_id}"
            )
        comparisons.append(
            {
                "trajectory_id": case_id,
                "primary_distance_scaled_rms": distance,
                "primary_distance_tolerance": PRIMARY_DISTANCE_TOLERANCE,
                "pass": True,
            }
        )
    return comparisons


def require_solver_balance(residual: float) -> None:
    if not math.isfinite(residual) or residual > BOUNDARY_BALANCE_TOLERANCE:
        raise ValueError(
            f"boundary-balance residual {residual:.6e} exceeds "
            f"{BOUNDARY_BALANCE_TOLERANCE:.1e}"
        )


def require_exact_array_bytes(
    left: np.ndarray, right: np.ndarray, *, label: str
) -> None:
    first = np.asarray(left)
    second = np.asarray(right)
    if (
        first.shape != second.shape
        or first.dtype != second.dtype
        or raw_array_sha256(first) != raw_array_sha256(second)
    ):
        raise ValueError(f"same-process {label} bytes differ")


def expected_solver_array_shape() -> tuple[int, int, int]:
    return (2, NATIVE_NODES, 4)
