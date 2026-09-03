"""Fail-closed native replay and evaluation for the pinned SU2 NACA case."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    NACA_REPLAY_CASE_SCHEMA,
    NACA_REPLAY_CONFIG_FILENAME,
    NACA_REPLAY_CONFIG_OVERRIDES,
    NACA_REPLAY_OUTPUT_FILENAME,
    NACA_RESTART_FIELDS,
    RESOURCE_MANIFEST_SCHEMA,
    SU2Mesh,
    SU2Restart,
    parse_su2_config,
    parse_su2_mesh,
    read_su2_binary_restart,
    sha256_file,
)

NACA_NATIVE_REPLAY_SCHEMA = "time_dependent_no.su2_naca0012_native_replay.v1"
NACA_REPLAY_COMPARISON_SCHEMA = "time_dependent_no.su2_naca0012_replay_comparison.v1"
NACA_NATIVE_RECEIPT_FILENAME = "native_replay_receipt.json"
NACA_STDOUT_FILENAME = "su2_stdout.log"
NACA_STDERR_FILENAME = "su2_stderr.log"
NACA_ALLOWED_NATIVE_EXTRA_FIELDS = ("Velocity_x", "Velocity_y")
NACA_NATIVE_RESTART_FIELDS = (
    *NACA_RESTART_FIELDS[:11],
    *NACA_ALLOWED_NATIVE_EXTRA_FIELDS,
    *NACA_RESTART_FIELDS[11:],
)

_VERSION_PATTERN = re.compile(r"SU2 v(?P<version>\d+\.\d+\.\d+)")
_STAGED_AUTHORITY_KEYS = {
    "license": "license",
    "mesh": "mesh",
    "restart_00497": "restart_00497",
    "restart_00498": "restart_00498",
    "upstream_config": "config",
}


class NativeReplayError(RuntimeError):
    """A native replay failed before it could support a scientific claim."""

    def __init__(self, message: str, *, receipt_path: Path | None = None):
        super().__init__(message)
        self.receipt_path = receipt_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _with_payload_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["canonical_payload_sha256"] = sha256(
        _canonical_payload_bytes(result)
    ).hexdigest()
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = _with_payload_sha256(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic receipt temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(rendered, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return rendered


def load_verified_json_receipt(
    path: str | Path, *, expected_schema: str
) -> dict[str, Any]:
    """Load a JSON receipt and verify its schema and canonical payload digest."""

    receipt_path = Path(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"{receipt_path}: unsupported receipt schema")
    observed = payload.pop("canonical_payload_sha256", None)
    expected = sha256(_canonical_payload_bytes(payload)).hexdigest()
    if observed != expected:
        raise ValueError(f"{receipt_path}: canonical payload SHA256 differs")
    payload["canonical_payload_sha256"] = observed
    return payload


def _safe_case_file(case_root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or relative.name != str(relative):
        raise ValueError(f"unsafe case filename: {relative_name}")
    candidate = (case_root / relative).resolve()
    if candidate.parent != case_root or candidate.is_symlink():
        raise ValueError(f"case file escapes or aliases the case root: {relative_name}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _file_record_or_error(path: Path) -> dict[str, Any] | None:
    """Capture best-effort evidence without preventing a failure receipt."""

    try:
        if not path.is_file():
            return None
        return _file_record(path)
    except OSError as error:
        return {
            "file": path.name,
            "record_error": f"{type(error).__name__}: {error}",
        }


def _evaluator_source_inventory() -> dict[str, dict[str, Any]]:
    source_root = Path(__file__).resolve().parent
    source_names = ("su2_native_replay.py", "su2_restart_contract.py")
    return {name: _file_record(source_root / name) for name in source_names}


def _python_runtime_record() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": " ".join(platform.python_build()),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }


def _case_inventory(case_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for candidate in sorted(case_root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"case contains a symbolic link: {candidate}")
        if candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if case_root not in resolved.parents:
            raise ValueError(f"case artifact escapes the case root: {candidate}")
        relative = candidate.relative_to(case_root).as_posix()
        inventory[relative] = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return inventory


def _verify_prepared_case(
    *,
    case_dir: str | Path,
    target_path: str | Path,
    executable_path: str | Path,
    expected_executable_sha256: str,
) -> dict[str, Any]:
    case_input = Path(case_dir)
    target_input = Path(target_path)
    executable_input = Path(executable_path)
    if not case_input.is_absolute():
        raise ValueError("case_dir must be absolute")
    if not target_input.is_absolute():
        raise ValueError("target_path must be absolute")
    if not executable_input.is_absolute():
        raise ValueError("executable_path must be absolute")

    case_root = case_input.resolve()
    target = target_input.resolve()
    executable = executable_input.resolve()
    if not case_root.is_dir() or case_root.is_symlink():
        raise ValueError(f"case directory is absent or aliased: {case_root}")
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"canonical target is absent or aliased: {target}")
    if not executable.is_file() or executable.is_symlink():
        raise ValueError(f"SU2 executable is absent or aliased: {executable}")
    if target == case_root or case_root in target.parents:
        raise ValueError("canonical target must remain outside the solver case")

    contract_path = _safe_case_file(case_root, "replay_case.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != NACA_REPLAY_CASE_SCHEMA
        or contract.get("status") != "prepared_not_executed"
    ):
        raise ValueError("case is not a frozen prepared NACA replay")
    for flag in (
        "native_solver_executed",
        "restart_sufficiency_claimed",
        "trusted_transition_claimed",
    ):
        if contract.get(flag) is not False:
            raise ValueError(f"prepared case has an invalid claim flag: {flag}")

    authority = contract.get("stage0_authority_binding")
    if not isinstance(authority, dict):
        raise TypeError("prepared case lacks its Stage-0 authority")
    authority_resources = authority.get("resource_sha256")
    if not isinstance(authority_resources, dict):
        raise TypeError("prepared case lacks authority resource hashes")

    replay_record = contract.get("replay_config")
    if not isinstance(replay_record, dict):
        raise TypeError("prepared case lacks replay_config")
    replay_config = _safe_case_file(case_root, str(replay_record.get("file")))
    if replay_config.name != NACA_REPLAY_CONFIG_FILENAME:
        raise ValueError("prepared case uses an unexpected replay config name")
    if _file_record(replay_config) != {
        "file": replay_config.name,
        "bytes": replay_record.get("bytes"),
        "sha256": replay_record.get("sha256"),
    }:
        raise ValueError("prepared replay config differs from its receipt")
    parsed_config = parse_su2_config(replay_config)
    for key, expected in NACA_REPLAY_CONFIG_OVERRIDES.items():
        if parsed_config.get(key) != expected:
            raise ValueError(f"replay config {key} differs from the frozen value")

    staged_inputs = contract.get("staged_inputs")
    if not isinstance(staged_inputs, dict):
        raise TypeError("prepared case lacks staged_inputs")
    for role, authority_key in _STAGED_AUTHORITY_KEYS.items():
        record = staged_inputs.get(role)
        if not isinstance(record, dict):
            raise TypeError(f"prepared case lacks staged input {role}")
        candidate = _safe_case_file(case_root, str(record.get("file")))
        observed = _file_record(candidate)
        if observed != {
            "file": candidate.name,
            "bytes": record.get("bytes"),
            "sha256": record.get("sha256"),
        }:
            raise ValueError(f"staged input differs from its receipt: {role}")
        if observed["sha256"] != authority_resources.get(authority_key):
            raise ValueError(f"staged input differs from Stage-0 authority: {role}")

    manifest_record = contract.get("staged_resource_manifest")
    if not isinstance(manifest_record, dict):
        raise TypeError("prepared case lacks its staged resource manifest")
    staged_manifest = _safe_case_file(case_root, str(manifest_record.get("file")))
    if sha256_file(staged_manifest) != manifest_record.get("sha256"):
        raise ValueError("staged resource manifest differs from its receipt")
    if manifest_record.get("sha256") != authority.get("manifest_sha256"):
        raise ValueError("staged resource manifest differs from Stage-0 authority")
    manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != RESOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError("staged resource manifest has an unsupported schema")
    manifest_upstream = manifest.get("upstream")
    manifest_resources = manifest.get("resources")
    if not isinstance(manifest_upstream, dict) or not isinstance(
        manifest_resources, dict
    ):
        raise TypeError("staged resource manifest lacks authority records")
    for name, record in manifest_resources.items():
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise TypeError(f"staged manifest resource {name} lacks SHA256")
    manifest_authority = {
        "manifest_file": staged_manifest.name,
        "manifest_sha256": sha256_file(staged_manifest),
        "tutorial_commit": manifest_upstream.get("tutorial_commit"),
        "target_replay_release": manifest_upstream.get("target_replay_release"),
        "resource_sha256": {
            name: record["sha256"] for name, record in manifest_resources.items()
        },
    }
    if _canonical_payload_bytes(authority) != _canonical_payload_bytes(
        manifest_authority
    ):
        raise ValueError(
            "prepared Stage-0 authority differs from the staged resource manifest"
        )

    target_record = contract.get("external_reference_target")
    if not isinstance(target_record, dict):
        raise TypeError("prepared case lacks its external target receipt")
    if target_record.get("copied_into_case") is not False:
        raise ValueError("prepared case does not preserve target isolation")
    if (
        target.name != target_record.get("file")
        or target.name != "restart_flow_00499.dat"
    ):
        raise ValueError("external target filename differs from the frozen target")
    target_sha256 = sha256_file(target)
    if target_sha256 != target_record.get("sha256"):
        raise ValueError("external target differs from the prepared case receipt")
    if target_sha256 != authority_resources.get("restart_00499"):
        raise ValueError("external target differs from Stage-0 authority")

    if contract.get("expected_output") != NACA_REPLAY_OUTPUT_FILENAME:
        raise ValueError("prepared case declares an unexpected replay output")
    for absent_name in (
        NACA_REPLAY_OUTPUT_FILENAME,
        "restart_flow_00499.dat",
        NACA_NATIVE_RECEIPT_FILENAME,
        NACA_STDOUT_FILENAME,
        NACA_STDERR_FILENAME,
    ):
        if (case_root / absent_name).exists():
            raise FileExistsError(f"case is not fresh; found {absent_name}")

    release = manifest_upstream.get("target_replay_release")
    if not isinstance(release, dict) or release.get("tag") != "v8.5.0":
        raise ValueError("prepared case lacks the pinned SU2 v8.5.0 release")
    asset = release.get("windows_mpi_asset")
    if not isinstance(asset, dict):
        raise TypeError("prepared case lacks the frozen Windows MPI asset")
    executable_record = asset.get("executable")
    if not isinstance(executable_record, dict):
        raise TypeError("prepared case lacks the frozen executable identity")
    if expected_executable_sha256 != executable_record.get("sha256"):
        raise ValueError("requested executable SHA256 differs from the manifest")
    observed_executable = _file_record(executable)
    if observed_executable["sha256"] != expected_executable_sha256:
        raise ValueError("SU2 executable SHA256 differs from the requested identity")
    if observed_executable["bytes"] != executable_record.get("bytes"):
        raise ValueError("SU2 executable size differs from the official asset")

    return {
        "case_root": case_root,
        "target": target,
        "target_sha256": target_sha256,
        "executable": executable,
        "executable_record": observed_executable,
        "contract": contract,
        "contract_path": contract_path,
        "replay_config": replay_config,
        "staged_manifest": staged_manifest,
        "manifest": manifest,
        "release": release,
        "baseline_inventory": _case_inventory(case_root),
        "mesh_points": parse_su2_mesh(
            case_root / contract["staged_inputs"]["mesh"]["file"]
        ).num_points,
    }


def _probe_su2_release(
    executable: Path, *, cwd: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--help"],
        cwd=cwd,
        env=dict(environment),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    match = _VERSION_PATTERN.search(combined)
    version = match.group("version") if match else None
    stdout_bytes = completed.stdout.encode("utf-8")
    stderr_bytes = completed.stderr.encode("utf-8")
    return {
        "argv": ["<bound-su2-executable>", "--help"],
        "return_code": completed.returncode,
        "version": version,
        "banner": next(
            (line for line in combined.splitlines() if "SU2 v" in line), None
        ),
        "stdout": {
            "bytes": len(stdout_bytes),
            "sha256": sha256(stdout_bytes).hexdigest(),
        },
        "stderr": {
            "bytes": len(stderr_bytes),
            "sha256": sha256(stderr_bytes).hexdigest(),
        },
        "error": None,
    }


def lumped_vertex_areas(mesh: SU2Mesh) -> np.ndarray:
    """Compute positive primal-cell areas lumped equally to element vertices."""

    if mesh.dimension != 2:
        raise ValueError("lumped area weights require a two-dimensional mesh")
    weights = np.zeros(mesh.num_points, dtype=np.float64)
    for vtk_type, nodes in mesh.elements:
        if vtk_type not in (5, 9):
            raise ValueError(f"unsupported volume element for area weights: {vtk_type}")
        coordinates = mesh.points[np.asarray(nodes, dtype=np.int64)]
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("mesh contains a nonpositive or nonfinite cell area")
        weights[np.asarray(nodes, dtype=np.int64)] += area / len(nodes)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("lumped vertex areas must be finite and positive")
    return weights


def lumped_marker_lengths(mesh: SU2Mesh, marker: str) -> np.ndarray:
    """Compute edge length lumped equally to vertices on one boundary marker."""

    elements = mesh.marker_elements.get(marker)
    if not elements:
        raise ValueError(f"mesh lacks marker {marker}")
    weights = np.zeros(mesh.num_points, dtype=np.float64)
    for vtk_type, nodes in elements:
        if vtk_type != 3 or len(nodes) != 2:
            raise ValueError(f"marker {marker} contains a non-line element")
        indices = np.asarray(nodes, dtype=np.int64)
        length = float(
            np.linalg.norm(mesh.points[indices[1]] - mesh.points[indices[0]])
        )
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"marker {marker} contains a nonpositive edge length")
        weights[indices] += 0.5 * length
    return weights


def _weights_record(weights: np.ndarray) -> dict[str, Any]:
    little_endian = np.asarray(weights, dtype="<f8")
    return {
        "encoding": "little-endian-float64",
        "sha256": sha256(little_endian.tobytes(order="C")).hexdigest(),
        "sum": float(np.sum(weights)),
        "minimum_positive": float(np.min(weights[weights > 0.0])),
        "maximum": float(np.max(weights)),
        "nonzero_count": int(np.count_nonzero(weights)),
    }


def _field_error_metrics(
    *,
    fields: Sequence[str],
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if reference_values.shape != candidate_values.shape:
        raise ValueError("restart arrays have different shapes")
    if weights.shape != (reference_values.shape[0],):
        raise ValueError("weight vector does not match restart point count")
    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("metric weights have no positive entries")
    selected_weights = weights[positive]
    weight_sum = float(np.sum(selected_weights))
    result: dict[str, dict[str, Any]] = {}
    for field in fields:
        index = NACA_RESTART_FIELDS.index(field)
        reference = reference_values[positive, index]
        candidate = candidate_values[positive, index]
        error = candidate - reference
        reference_energy = float(np.dot(selected_weights, reference * reference))
        error_energy = float(np.dot(selected_weights, error * error))
        relative_l2 = (
            math.sqrt(error_energy / reference_energy)
            if reference_energy > 0.0
            else None
        )
        result[field] = {
            "reference_weighted_rms": math.sqrt(reference_energy / weight_sum),
            "error_weighted_rms": math.sqrt(error_energy / weight_sum),
            "weighted_relative_l2": relative_l2,
            "relative_l2_null_reason": (
                None if relative_l2 is not None else "zero_reference_energy"
            ),
            "linf": float(np.max(np.abs(error))),
        }
    return result


def _dynamic_aggregate(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    values = [metrics[field]["weighted_relative_l2"] for field in NACA_DYNAMIC_FIELDS]
    if any(value is None for value in values):
        return {
            "component_balanced_relative_l2": None,
            "maximum_dynamic_field_relative_l2": None,
        }
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "component_balanced_relative_l2": float(np.sqrt(np.mean(numeric * numeric))),
        "maximum_dynamic_field_relative_l2": float(np.max(numeric)),
    }


def _surface_force_proxy(
    mesh: SU2Mesh,
    values: np.ndarray,
    *,
    angle_of_attack_degrees: float,
) -> dict[str, float | str]:
    cp_index = NACA_RESTART_FIELDS.index("Pressure_Coefficient")
    cfx_index = NACA_RESTART_FIELDS.index("Skin_Friction_Coefficient_x")
    cfy_index = NACA_RESTART_FIELDS.index("Skin_Friction_Coefficient_y")
    marker_edges: list[tuple[int, int]] = []
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for vtk_type, nodes in mesh.marker_elements.get("airfoil", ()):
        if vtk_type != 3 or len(nodes) != 2:
            raise ValueError("airfoil marker contains a non-line element")
        first, second = nodes
        if first in successor or second in predecessor:
            raise ValueError("airfoil marker is not one consistently directed loop")
        successor[first] = second
        predecessor[second] = first
        marker_edges.append((first, second))
    if len(marker_edges) < 3 or set(successor) != set(predecessor):
        raise ValueError("airfoil marker is not a closed directed loop")
    start = marker_edges[0][0]
    visited: set[int] = set()
    current = start
    for _ in marker_edges:
        if current in visited:
            raise ValueError("airfoil marker closes before visiting every edge")
        visited.add(current)
        current = successor[current]
    if current != start or len(visited) != len(marker_edges):
        raise ValueError("airfoil marker is not one closed directed loop")

    twice_signed_area = float(
        sum(
            mesh.points[first, 0] * mesh.points[second, 1]
            - mesh.points[second, 0] * mesh.points[first, 1]
            for first, second in marker_edges
        )
    )
    if not math.isfinite(twice_signed_area) or abs(twice_signed_area) <= 1.0e-15:
        raise ValueError("airfoil marker has zero or nonfinite signed area")
    orientation_sign = 1.0 if twice_signed_area > 0.0 else -1.0

    force = np.zeros(2, dtype=np.float64)
    for first, second in marker_edges:
        delta = mesh.points[second] - mesh.points[first]
        length = float(np.linalg.norm(delta))
        outward_body_normal_ds = orientation_sign * np.asarray(
            [delta[1], -delta[0]], dtype=np.float64
        )
        cp = 0.5 * (values[first, cp_index] + values[second, cp_index])
        friction = 0.5 * (
            values[first, [cfx_index, cfy_index]]
            + values[second, [cfx_index, cfy_index]]
        )
        force += -cp * outward_body_normal_ds + friction * length
    alpha = math.radians(angle_of_attack_degrees)
    streamwise = force[0] * math.cos(alpha) + force[1] * math.sin(alpha)
    cross_stream = -force[0] * math.sin(alpha) + force[1] * math.cos(alpha)
    return {
        "airfoil_orientation": (
            "counterclockwise" if orientation_sign > 0.0 else "clockwise"
        ),
        "twice_signed_area": twice_signed_area,
        "body_force_x_proxy": float(force[0]),
        "body_force_y_proxy": float(force[1]),
        "streamwise_body_force_proxy": float(streamwise),
        "cross_stream_body_force_proxy": float(cross_stream),
    }


def _aligned_canonical_values(
    restart: SU2Restart, *, label: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    if restart.fields != NACA_NATIVE_RESTART_FIELDS:
        raise ValueError(f"{label} restart has an unsupported field schema")
    indices = [restart.fields.index(field) for field in NACA_RESTART_FIELDS]
    return restart.values[:, indices], NACA_ALLOWED_NATIVE_EXTRA_FIELDS


def _evaluate_aligned_restart_values(
    *,
    mesh: SU2Mesh,
    reference_path: Path,
    candidate_path: Path,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    interface: Mapping[str, Any],
    angle_of_attack_degrees: float,
) -> dict[str, Any]:
    if reference_values.shape != (mesh.num_points, len(NACA_RESTART_FIELDS)):
        raise ValueError("reference restart point count differs from the mesh")
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate restart point count differs from the reference")

    coordinate_indices = [
        NACA_RESTART_FIELDS.index("x"),
        NACA_RESTART_FIELDS.index("y"),
    ]
    reference_coordinate_error = float(
        np.max(np.abs(reference_values[:, coordinate_indices] - mesh.points))
    )
    candidate_coordinate_error = float(
        np.max(np.abs(candidate_values[:, coordinate_indices] - mesh.points))
    )
    if reference_coordinate_error > 1.0e-12:
        raise ValueError("reference coordinates differ from the mesh")
    if candidate_coordinate_error > 1.0e-12:
        raise ValueError("candidate coordinates differ from the mesh")

    volume_weights = lumped_vertex_areas(mesh)
    state_fields = NACA_RESTART_FIELDS[2:]
    field_metrics = _field_error_metrics(
        fields=state_fields,
        reference_values=reference_values,
        candidate_values=candidate_values,
        weights=volume_weights,
    )
    boundary_metrics: dict[str, Any] = {}
    for marker in ("airfoil", "farfield"):
        marker_weights = lumped_marker_lengths(mesh, marker)
        boundary_metrics[marker] = {
            "weights": _weights_record(marker_weights),
            "fields": _field_error_metrics(
                fields=state_fields,
                reference_values=reference_values,
                candidate_values=candidate_values,
                weights=marker_weights,
            ),
        }

    reference_force = _surface_force_proxy(
        mesh,
        reference_values,
        angle_of_attack_degrees=angle_of_attack_degrees,
    )
    candidate_force = _surface_force_proxy(
        mesh,
        candidate_values,
        angle_of_attack_degrees=angle_of_attack_degrees,
    )
    force_component_keys = (
        "body_force_x_proxy",
        "body_force_y_proxy",
        "streamwise_body_force_proxy",
        "cross_stream_body_force_proxy",
    )
    return {
        "reference": _file_record(reference_path),
        "candidate": _file_record(candidate_path),
        "interface": dict(interface),
        "coordinate_max_abs_error": {
            "reference_vs_mesh": reference_coordinate_error,
            "candidate_vs_mesh": candidate_coordinate_error,
            "candidate_vs_reference": float(
                np.max(
                    np.abs(
                        candidate_values[:, coordinate_indices]
                        - reference_values[:, coordinate_indices]
                    )
                )
            ),
        },
        "volume_weights": _weights_record(volume_weights),
        "fields": field_metrics,
        "dynamic_aggregate": _dynamic_aggregate(field_metrics),
        "boundary": boundary_metrics,
        "surface_force_proxy": {
            "scope": (
                "orientation-normalized pressure-plus-skin-friction body-force "
                "functional from restart fields; not claimed as SU2 integrated "
                "lift or drag"
            ),
            "reference": reference_force,
            "candidate": candidate_force,
            "absolute_difference": {
                key: abs(float(candidate_force[key]) - float(reference_force[key]))
                for key in force_component_keys
            },
        },
        "lift_drag_gate_complete": False,
    }


def evaluate_restart_replay(
    *,
    mesh_path: str | Path,
    reference_path: str | Path,
    candidate_path: str | Path,
    angle_of_attack_degrees: float = 17.0,
) -> dict[str, Any]:
    """Compare native output with the strict canonical 17-field target."""

    mesh = parse_su2_mesh(mesh_path)
    reference = read_su2_binary_restart(reference_path)
    candidate = read_su2_binary_restart(candidate_path)
    if reference.fields != NACA_RESTART_FIELDS:
        raise ValueError("reference restart fields differ from the NACA contract")
    candidate_values, candidate_extra_fields = _aligned_canonical_values(
        candidate, label="candidate"
    )
    if (
        candidate.header[0] != reference.header[0]
        or candidate.header[2:] != reference.header[2:]
    ):
        raise ValueError("candidate restart header differs from the reference")
    return _evaluate_aligned_restart_values(
        mesh=mesh,
        reference_path=Path(reference_path),
        candidate_path=Path(candidate_path),
        reference_values=reference.values,
        candidate_values=candidate_values,
        interface={
            "reference_schema": "canonical_17",
            "candidate_schema": "su2_v8.5_native_19",
            "reference_fields": list(reference.fields),
            "candidate_fields": list(candidate.fields),
            "candidate_extra_fields": list(candidate_extra_fields),
            "canonical_field_to_candidate_index": {
                field: candidate.fields.index(field) for field in NACA_RESTART_FIELDS
            },
            "excluded_candidate_fields": list(candidate_extra_fields),
            "excluded_fields_enter_metrics": False,
            "canonical_fields_name_aligned": True,
            "reference_contract_strict": True,
        },
        angle_of_attack_degrees=angle_of_attack_degrees,
    )


def evaluate_restart_repeat(
    *,
    mesh_path: str | Path,
    first_path: str | Path,
    second_path: str | Path,
    angle_of_attack_degrees: float = 17.0,
) -> dict[str, Any]:
    """Compare two native outputs without weakening the canonical target contract."""

    mesh = parse_su2_mesh(mesh_path)
    first = read_su2_binary_restart(first_path)
    second = read_su2_binary_restart(second_path)
    if first.fields != second.fields:
        raise ValueError("native repeat restart field schemas differ")
    if first.header != second.header:
        raise ValueError("native repeat restart headers differ")
    first_values, first_extra_fields = _aligned_canonical_values(
        first, label="first native output"
    )
    second_values, second_extra_fields = _aligned_canonical_values(
        second, label="second native output"
    )
    return _evaluate_aligned_restart_values(
        mesh=mesh,
        reference_path=Path(first_path),
        candidate_path=Path(second_path),
        reference_values=first_values,
        candidate_values=second_values,
        interface={
            "reference_schema": "su2_v8.5_native_19",
            "candidate_schema": "su2_v8.5_native_19",
            "reference_fields": list(first.fields),
            "candidate_fields": list(second.fields),
            "reference_extra_fields": list(first_extra_fields),
            "candidate_extra_fields": list(second_extra_fields),
            "canonical_field_to_reference_index": {
                field: first.fields.index(field) for field in NACA_RESTART_FIELDS
            },
            "canonical_field_to_candidate_index": {
                field: second.fields.index(field) for field in NACA_RESTART_FIELDS
            },
            "excluded_reference_fields": list(first_extra_fields),
            "excluded_candidate_fields": list(second_extra_fields),
            "excluded_fields_enter_metrics": False,
            "canonical_fields_name_aligned": True,
            "comparison_scope": "native-output repeatability, not canonical-target error",
        },
        angle_of_attack_degrees=angle_of_attack_degrees,
    )


def run_native_naca0012_replay(
    *,
    case_dir: str | Path,
    target_path: str | Path,
    executable_path: str | Path,
    expected_executable_sha256: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute and evaluate one fresh prepared replay, preserving all evidence."""

    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    prepared = _verify_prepared_case(
        case_dir=case_dir,
        target_path=target_path,
        executable_path=executable_path,
        expected_executable_sha256=expected_executable_sha256,
    )
    run_id = str(uuid4())
    evaluator_sources_at_start = _evaluator_source_inventory()
    python_runtime = _python_runtime_record()
    case_root: Path = prepared["case_root"]
    target: Path = prepared["target"]
    executable: Path = prepared["executable"]
    receipt_path = case_root / NACA_NATIVE_RECEIPT_FILENAME
    stdout_path = case_root / NACA_STDOUT_FILENAME
    stderr_path = case_root / NACA_STDERR_FILENAME
    selected_environment = {
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
    }
    environment = os.environ.copy()
    environment.update(selected_environment)

    release_tag = prepared["release"]["tag"]
    expected_version = str(release_tag).removeprefix("v")
    command = [
        str(executable),
        "--threads",
        "1",
        NACA_REPLAY_CONFIG_FILENAME,
    ]
    probe: dict[str, Any] = {
        "argv": ["<bound-su2-executable>", "--help"],
        "return_code": None,
        "version": None,
        "banner": None,
        "stdout": None,
        "stderr": None,
        "error": None,
    }
    probe_error: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    return_code: int | None = None
    timed_out = False
    process_error: str | None = None
    process_attempted = False
    process_started = False
    with (
        stdout_path.open("x", encoding="utf-8", newline="\n") as stdout_handle,
        stderr_path.open("x", encoding="utf-8", newline="\n") as stderr_handle,
    ):
        try:
            probe = _probe_su2_release(
                executable,
                cwd=case_root,
                environment=environment,
            )
        except Exception as error:  # noqa: BLE001 - preserve failed probe evidence
            probe_error = f"{type(error).__name__}: {error}"
            probe["error"] = probe_error
        if probe_error is None and (
            probe["return_code"] != 0 or probe["version"] != expected_version
        ):
            probe_error = "SU2 banner does not match the pinned release"
            probe["error"] = probe_error

        if probe_error is None:
            process_attempted = True
            started_at = _utc_now()
            start_clock = time.perf_counter()
            try:
                completed = subprocess.run(
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
                return_code = completed.returncode
                process_started = True
            except subprocess.TimeoutExpired:
                timed_out = True
                process_started = True
                process_error = f"timeout after {timeout_seconds:g} seconds"
            except OSError as error:
                process_error = f"{type(error).__name__}: {error}"
            duration_seconds = time.perf_counter() - start_clock
            ended_at = _utc_now()
        else:
            process_error = probe_error

    evaluation: dict[str, Any] | None = None
    evaluation_error: str | None = None
    status = "release_probe_failed" if probe_error is not None else "process_failed"
    after_inventory: dict[str, dict[str, Any]] = {}
    new_artifacts: dict[str, dict[str, Any]] = {}
    target_after: dict[str, Any] | None = None
    executable_after: dict[str, Any] | None = None
    evaluator_sources_after_process: dict[str, dict[str, Any]] | None = None
    evaluator_sources_before_evaluation: dict[str, dict[str, Any]] | None = None
    evaluator_sources_after_evaluation: dict[str, dict[str, Any]] | None = None
    try:
        after_inventory = _case_inventory(case_root)
        new_artifacts = {
            relative: record
            for relative, record in after_inventory.items()
            if relative not in prepared["baseline_inventory"]
        }
        for relative, record in prepared["baseline_inventory"].items():
            if after_inventory.get(relative) != record:
                raise ValueError(
                    f"prepared case input changed during execution: {relative}"
                )
        target_after = _file_record(target)
        if target_after["sha256"] != prepared["target_sha256"]:
            raise ValueError("canonical target changed during execution")
        executable_after = _file_record(executable)
        if executable_after["sha256"] != expected_executable_sha256:
            raise ValueError("SU2 executable changed during execution")
        evaluator_sources_after_process = _evaluator_source_inventory()
        if evaluator_sources_after_process != evaluator_sources_at_start:
            raise ValueError("evaluator source changed during execution")
        if probe_error is not None:
            raise RuntimeError(probe_error)
        if timed_out:
            raise RuntimeError(process_error)
        if process_error is not None:
            raise RuntimeError(process_error)
        if return_code != 0:
            raise RuntimeError(f"SU2 exited with code {return_code}")
        if (case_root / "restart_flow_00499.dat").exists():
            raise ValueError("solver case contains the forbidden canonical target name")
        replay_outputs = sorted(case_root.glob("replay_flow_*.dat"))
        if replay_outputs != [case_root / NACA_REPLAY_OUTPUT_FILENAME]:
            raise ValueError("solver did not produce exactly replay_flow_00499.dat")
        evaluator_sources_before_evaluation = _evaluator_source_inventory()
        if evaluator_sources_before_evaluation != evaluator_sources_at_start:
            raise ValueError("evaluator source changed before evaluation")
        try:
            evaluation = evaluate_restart_replay(
                mesh_path=(
                    case_root
                    / prepared["contract"]["staged_inputs"]["mesh"]["file"]
                ),
                reference_path=target,
                candidate_path=replay_outputs[0],
                angle_of_attack_degrees=float(
                    parse_su2_config(prepared["replay_config"])["AOA"]
                ),
            )
        finally:
            evaluator_sources_after_evaluation = _evaluator_source_inventory()
        if evaluator_sources_after_evaluation != evaluator_sources_at_start:
            raise ValueError("evaluator source changed during evaluation")
        status = "execution_and_evaluation_succeeded"
    except Exception as error:  # noqa: BLE001 - preserve a receipt for any failed gate
        evaluation_error = f"{type(error).__name__}: {error}"
        if probe_error is not None:
            status = "release_probe_failed"
        elif return_code == 0 and not timed_out and process_error is None:
            status = "evaluation_failed"
        else:
            status = "process_failed"

    nodes = prepared["contract"]["stage0_authority_binding"]
    mesh_points = prepared["mesh_points"]
    baseline_inventory = prepared["baseline_inventory"]
    output_record = _file_record_or_error(case_root / NACA_REPLAY_OUTPUT_FILENAME)
    stdout_record = _file_record_or_error(stdout_path)
    stderr_record = _file_record_or_error(stderr_path)
    receipt = {
        "schema": NACA_NATIVE_REPLAY_SCHEMA,
        "run_id": run_id,
        "status": status,
        "stage0_authority_binding": nodes,
        "prepared_case": {
            "receipt_file": prepared["contract_path"].name,
            "receipt_sha256": baseline_inventory["replay_case.json"]["sha256"],
            "replay_config_sha256": baseline_inventory[
                NACA_REPLAY_CONFIG_FILENAME
            ]["sha256"],
            "baseline_inventory": baseline_inventory,
        },
        "runtime_identity": {
            "executable": {
                **prepared["executable_record"],
                "absolute_path": str(executable),
            },
            "release_probe": probe,
            "release_asset": prepared["release"]["windows_mpi_asset"],
            "release_identity_verified": (
                probe_error is None
                and probe["version"] == expected_version
                and prepared["executable_record"]["sha256"]
                == prepared["release"]["windows_mpi_asset"]["executable"]["sha256"]
            ),
            "selected_environment": selected_environment,
            "argv": [
                "<bound-su2-executable>",
                "--threads",
                "1",
                NACA_REPLAY_CONFIG_FILENAME,
            ],
            "execution_mode": "direct_single_mpi_rank_one_openmp_thread",
        },
        "process": {
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "duration_seconds": duration_seconds,
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "return_code": return_code,
            "error": process_error,
            "attempted": process_attempted,
            "started": process_started,
            "stdout": stdout_record,
            "stderr": stderr_record,
        },
        "external_target": {
            "file": target.name,
            "sha256_before": prepared["target_sha256"],
            "record_after": target_after,
            "sha256_after": (
                target_after.get("sha256") if target_after is not None else None
            ),
            "copied_into_case": False,
        },
        "executable_record_after": executable_after,
        "evaluator_identity": {
            "source_at_start": evaluator_sources_at_start,
            "source_after_process": evaluator_sources_after_process,
            "source_before_evaluation": evaluator_sources_before_evaluation,
            "source_after_evaluation": evaluator_sources_after_evaluation,
            "unchanged_through_evaluation": (
                evaluation is not None
                and evaluator_sources_after_process == evaluator_sources_at_start
                and evaluator_sources_before_evaluation == evaluator_sources_at_start
                and evaluator_sources_after_evaluation == evaluator_sources_at_start
            ),
            "python_runtime": python_runtime,
        },
        "new_solver_artifacts": new_artifacts,
        "expected_output": output_record,
        "evaluation": evaluation,
        "evaluation_error": evaluation_error,
        "throughput": {
            "transitions_per_second": (
                1.0 / duration_seconds
                if duration_seconds is not None and duration_seconds > 0.0
                else None
            ),
            "nodes_per_second": (
                mesh_points / duration_seconds
                if duration_seconds is not None and duration_seconds > 0.0
                else None
            ),
        },
        "process_attempted": process_attempted,
        "process_started": process_started,
        "execution_succeeded": (
            return_code == 0 and not timed_out and process_error is None
        ),
        "output_contract_valid": (
            evaluation is not None and status == "execution_and_evaluation_succeeded"
        ),
        "replay_evaluated": evaluation is not None,
        "restart_sufficiency_claimed": False,
        "trusted_transition_claimed": False,
        "boundary_and_force_gate_complete": False,
    }
    rendered_receipt = _write_json_atomic(receipt_path, receipt)
    if status != "execution_and_evaluation_succeeded":
        raise NativeReplayError(
            evaluation_error or process_error or "native replay failed",
            receipt_path=receipt_path,
        )
    return rendered_receipt


def compare_native_replay_receipts(
    *,
    first_receipt_path: str | Path,
    second_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare two independently prepared and successfully evaluated replays."""

    first_path = Path(first_receipt_path).resolve()
    second_path = Path(second_receipt_path).resolve()
    destination = Path(output_path).resolve()
    comparator_sources_at_start = _evaluator_source_inventory()
    comparator_python_runtime = _python_runtime_record()
    if first_path == second_path:
        raise ValueError("replay comparison requires two distinct receipts")
    if first_path.parent == second_path.parent:
        raise ValueError("replay comparison requires two distinct case directories")
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    first = load_verified_json_receipt(
        first_path, expected_schema=NACA_NATIVE_REPLAY_SCHEMA
    )
    second = load_verified_json_receipt(
        second_path, expected_schema=NACA_NATIVE_REPLAY_SCHEMA
    )
    for label, receipt in (("first", first), ("second", second)):
        if receipt.get("status") != "execution_and_evaluation_succeeded":
            raise ValueError(f"{label} replay receipt is not successful")
        if receipt.get("replay_evaluated") is not True:
            raise ValueError(f"{label} replay was not evaluated")
        if not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]:
            raise ValueError(f"{label} replay lacks a run identifier")
        evaluator_identity = receipt.get("evaluator_identity")
        if (
            not isinstance(evaluator_identity, dict)
            or evaluator_identity.get("unchanged_through_evaluation") is not True
            or evaluator_identity.get("source_at_start")
            != evaluator_identity.get("source_after_process")
            or evaluator_identity.get("source_at_start")
            != evaluator_identity.get("source_before_evaluation")
            or evaluator_identity.get("source_at_start")
            != evaluator_identity.get("source_after_evaluation")
        ):
            raise ValueError(f"{label} replay lacks stable evaluator identity")
        if evaluator_identity["source_at_start"] != comparator_sources_at_start:
            raise ValueError(f"{label} replay evaluator differs from live source")
        if evaluator_identity.get("python_runtime") != comparator_python_runtime:
            raise ValueError(f"{label} replay Python runtime differs from comparator")
    if first["run_id"] == second["run_id"]:
        raise ValueError("replay comparison rejects copied or repeated run evidence")
    if first["canonical_payload_sha256"] == second["canonical_payload_sha256"]:
        raise ValueError("replay comparison rejects identical receipt payloads")
    if sha256_file(first_path) == sha256_file(second_path):
        raise ValueError("replay comparison rejects identical receipt files")

    identity_pairs = {
        "stage0_authority_binding": (
            first["stage0_authority_binding"],
            second["stage0_authority_binding"],
        ),
        "replay_config_sha256": (
            first["prepared_case"]["replay_config_sha256"],
            second["prepared_case"]["replay_config_sha256"],
        ),
        "executable_sha256": (
            first["runtime_identity"]["executable"]["sha256"],
            second["runtime_identity"]["executable"]["sha256"],
        ),
        "release_version": (
            first["runtime_identity"]["release_probe"]["version"],
            second["runtime_identity"]["release_probe"]["version"],
        ),
        "selected_environment": (
            first["runtime_identity"]["selected_environment"],
            second["runtime_identity"]["selected_environment"],
        ),
        "execution_mode": (
            first["runtime_identity"]["execution_mode"],
            second["runtime_identity"]["execution_mode"],
        ),
        "target_sha256": (
            first["external_target"]["sha256_after"],
            second["external_target"]["sha256_after"],
        ),
        "volume_weight_sha256": (
            first["evaluation"]["volume_weights"]["sha256"],
            second["evaluation"]["volume_weights"]["sha256"],
        ),
        "evaluator_source": (
            first["evaluator_identity"]["source_at_start"],
            second["evaluator_identity"]["source_at_start"],
        ),
        "python_runtime": (
            first["evaluator_identity"]["python_runtime"],
            second["evaluator_identity"]["python_runtime"],
        ),
    }
    mismatches = [
        key
        for key, (first_value, second_value) in identity_pairs.items()
        if first_value != second_value
    ]
    if mismatches:
        raise ValueError(f"replay identities differ: {', '.join(mismatches)}")

    live_outputs: list[Path] = []
    live_output_records: list[dict[str, Any]] = []
    live_meshes: list[Path] = []
    for label, receipt_path, receipt in (
        ("first", first_path, first),
        ("second", second_path, second),
    ):
        case_root = receipt_path.parent
        baseline = receipt["prepared_case"]["baseline_inventory"]
        if not isinstance(baseline, dict):
            raise TypeError(f"{label} replay lacks a baseline inventory")
        for relative, stored_record in baseline.items():
            live_baseline = _safe_case_file(case_root, relative)
            observed = {
                "bytes": live_baseline.stat().st_size,
                "sha256": sha256_file(live_baseline),
            }
            if observed != stored_record:
                raise ValueError(
                    f"{label} replay baseline differs from receipt: {relative}"
                )

        expected_output = receipt.get("expected_output")
        if not isinstance(expected_output, dict):
            raise TypeError(f"{label} replay lacks its expected output record")
        if expected_output.get("file") != NACA_REPLAY_OUTPUT_FILENAME:
            raise ValueError(f"{label} replay declares an unsafe output filename")
        live_output = _safe_case_file(case_root, NACA_REPLAY_OUTPUT_FILENAME)
        live_output_record = _file_record(live_output)
        if live_output_record != expected_output:
            raise ValueError(f"{label} replay output differs from its receipt")
        if receipt["evaluation"].get("candidate") != expected_output:
            raise ValueError(f"{label} replay evaluation output differs from receipt")

        mesh_sha256 = receipt["stage0_authority_binding"]["resource_sha256"][
            "mesh"
        ]
        mesh_matches = [
            relative
            for relative, record in baseline.items()
            if record.get("sha256") == mesh_sha256
        ]
        if len(mesh_matches) != 1:
            raise ValueError(f"{label} replay does not identify one baseline mesh")
        live_mesh = _safe_case_file(case_root, mesh_matches[0])
        live_outputs.append(live_output)
        live_output_records.append(live_output_record)
        live_meshes.append(live_mesh)

    first_output, second_output = live_outputs
    first_output_record, second_output_record = live_output_records
    first_mesh, second_mesh = live_meshes
    if first_output == second_output or first_mesh == second_mesh:
        raise ValueError("replay comparison requires independently staged artifacts")
    comparator_sources_before_evaluation = _evaluator_source_inventory()
    if comparator_sources_before_evaluation != comparator_sources_at_start:
        raise ValueError("comparator source changed before live evaluation")
    try:
        numerical_repeat = evaluate_restart_repeat(
            mesh_path=first_mesh,
            first_path=first_output,
            second_path=second_output,
        )
    finally:
        comparator_sources_after_evaluation = _evaluator_source_inventory()
    if comparator_sources_after_evaluation != comparator_sources_at_start:
        raise ValueError("comparator source changed during live evaluation")
    output_sha256_equal = (
        first_output_record["sha256"] == second_output_record["sha256"]
    )
    comparison = {
        "schema": NACA_REPLAY_COMPARISON_SCHEMA,
        "status": "comparison_complete",
        "first_receipt": {
            "file": first_path.name,
            "sha256": sha256_file(first_path),
            "canonical_payload_sha256": first["canonical_payload_sha256"],
        },
        "second_receipt": {
            "file": second_path.name,
            "sha256": sha256_file(second_path),
            "canonical_payload_sha256": second["canonical_payload_sha256"],
        },
        "identity_match": True,
        "live_output_records": {
            "first": first_output_record,
            "second": second_output_record,
        },
        "output_sha256_equal": output_sha256_equal,
        "bitwise_deterministic": output_sha256_equal,
        "numerical_repeat": numerical_repeat,
        "comparator_identity": {
            "source_at_start": comparator_sources_at_start,
            "source_before_evaluation": comparator_sources_before_evaluation,
            "source_after_evaluation": comparator_sources_after_evaluation,
            "unchanged_through_evaluation": True,
            "python_runtime": comparator_python_runtime,
        },
        "timing": {
            "first_duration_seconds": first["process"]["duration_seconds"],
            "second_duration_seconds": second["process"]["duration_seconds"],
        },
        "restart_sufficiency_claimed": False,
        "trusted_transition_claimed": False,
        "boundary_and_force_gate_complete": False,
    }
    return _write_json_atomic(destination, comparison)
