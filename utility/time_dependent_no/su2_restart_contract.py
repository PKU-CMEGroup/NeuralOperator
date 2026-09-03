"""Strict readers for the SU2 Unsteady NACA0012 readiness resources.

This module validates file identity and the mesh/restart state interface.  It
does not run SU2 and therefore cannot establish restart sufficiency, solver
accuracy, or a trusted transition.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

RESOURCE_MANIFEST_SCHEMA = "time_dependent_no.su2_naca0012_resources.v1"
SU2_BINARY_MAGIC = 535532
SU2_HEADER_INTS = 5
SU2_FIELD_NAME_BYTES = 33
NACA_DYNAMIC_FIELDS = (
    "Density",
    "Momentum_x",
    "Momentum_y",
    "Energy",
    "Nu_Tilde",
)
NACA_CONFIG_CONTRACT = {
    "SOLVER": "RANS",
    "KIND_TURB_MODEL": "SA",
    "RESTART_SOL": "YES",
    "RESTART_ITER": "499",
    "TIME_DOMAIN": "YES",
    "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
    "TIME_STEP": "5e-4",
    "INNER_ITER": "10",
    "MESH_FILENAME": "unsteady_naca0012_mesh.su2",
    "SOLUTION_FILENAME": "restart_flow",
}
NACA_RESTART_INDICES = (497, 498, 499)
NACA_BDF2_HISTORY_INDICES = (497, 498)
NACA_REPLAY_TARGET_INDEX = 499
NACA_STAGE0_CLAIM_BOUNDARY = {
    "native_solver_executed": False,
    "restart_sufficiency_claimed": False,
    "trusted_transition_claimed": False,
}
NACA_REPLAY_CONFIG_OVERRIDES = {
    "RESTART_ITER": "499",
    "TIME_ITER": "500",
    "SOLUTION_FILENAME": "restart_flow",
    "RESTART_FILENAME": "replay_flow",
    "WINDOW_CAUCHY_CRIT": "NO",
    "OUTPUT_FILES": "( RESTART )",
    "OUTPUT_WRT_FREQ": "( 1 )",
    "WRT_RESTART_COMPACT": "NO",
}
NACA_REPLAY_CONFIG_FILENAME = "replay.cfg"
NACA_REPLAY_OUTPUT_FILENAME = "replay_flow_00499.dat"
NACA_REPLAY_CASE_SCHEMA = "time_dependent_no.su2_naca0012_replay_case.v1"
NACA_RESTART_FIELDS = (
    "x",
    "y",
    *NACA_DYNAMIC_FIELDS,
    "Pressure",
    "Temperature",
    "Mach",
    "Pressure_Coefficient",
    "Laminar_Viscosity",
    "Skin_Friction_Coefficient_x",
    "Skin_Friction_Coefficient_y",
    "Heat_Flux",
    "Y_Plus",
    "Eddy_Viscosity",
)

_VTK_NODE_COUNT = {
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quadrilateral
    10: 4,  # tetrahedron
    12: 8,  # hexahedron
    13: 6,  # prism
    14: 5,  # pyramid
}
_RESTART_INDEX = re.compile(r"_(\d+)\.dat$")


@dataclass(frozen=True)
class SU2Mesh:
    """The bounded mesh information needed by the readiness audit."""

    dimension: int
    num_elements: int
    num_points: int
    points: np.ndarray
    elements: tuple[tuple[int, tuple[int, ...]], ...]
    element_type_counts: Mapping[int, int]
    marker_elements: Mapping[str, tuple[tuple[int, tuple[int, ...]], ...]]
    marker_element_counts: Mapping[str, int]


@dataclass(frozen=True)
class SU2Restart:
    """One native SU2 binary restart and its point-major values."""

    path: Path
    byte_order: str
    header: tuple[int, ...]
    fields: tuple[str, ...]
    values: np.ndarray
    sha256: str

    @property
    def num_fields(self) -> int:
        return int(self.header[1])

    @property
    def num_points(self) -> int:
        return int(self.header[2])


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_su2_config(path: str | Path) -> dict[str, str]:
    """Parse the scalar/list assignments used by the pinned SU2 config."""

    config_path = Path(path)
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8", errors="strict").splitlines(),
        start=1,
    ):
        line = raw_line.split("%", maxsplit=1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{config_path}:{line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        key = key.upper()
        if not key or not value:
            raise ValueError(f"{config_path}:{line_number}: empty key or value")
        if key in result:
            raise ValueError(f"{config_path}:{line_number}: duplicate key {key}")
        result[key] = value
    if not result:
        raise ValueError(f"{config_path} contains no assignments")
    return result


def _mesh_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        line = raw_line.split("%", maxsplit=1)[0].strip()
        if line:
            lines.append((line_number, line))
    return lines


def _assignment(
    lines: list[tuple[int, str]], cursor: int, expected_key: str, path: Path
) -> tuple[str, int]:
    if cursor >= len(lines):
        raise ValueError(f"{path}: missing {expected_key}")
    line_number, line = lines[cursor]
    if "=" not in line:
        raise ValueError(f"{path}:{line_number}: expected {expected_key}=...")
    key, value = (part.strip() for part in line.split("=", maxsplit=1))
    if key.upper() != expected_key:
        raise ValueError(
            f"{path}:{line_number}: expected {expected_key}, observed {key}"
        )
    return value, cursor + 1


def _element_nodes(line: str, path: Path, line_number: int) -> tuple[int, ...]:
    try:
        values = [int(item) for item in line.split()]
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: noninteger element row") from error
    if not values or values[0] not in _VTK_NODE_COUNT:
        observed = values[0] if values else None
        raise ValueError(f"{path}:{line_number}: unsupported VTK type {observed}")
    node_count = _VTK_NODE_COUNT[values[0]]
    if len(values) < node_count + 1:
        raise ValueError(f"{path}:{line_number}: truncated element row")
    return tuple(values[1 : node_count + 1])


def parse_su2_mesh(path: str | Path) -> SU2Mesh:
    """Parse and validate the ordered sections of an ASCII SU2 mesh."""

    mesh_path = Path(path)
    lines = _mesh_lines(mesh_path)
    cursor = 0
    value, cursor = _assignment(lines, cursor, "NDIME", mesh_path)
    dimension = int(value)
    if dimension not in (2, 3):
        raise ValueError(f"{mesh_path}: unsupported NDIME={dimension}")

    value, cursor = _assignment(lines, cursor, "NELEM", mesh_path)
    num_elements = int(value)
    if num_elements < 1:
        raise ValueError(f"{mesh_path}: NELEM must be positive")
    element_types: Counter[int] = Counter()
    elements: list[tuple[int, tuple[int, ...]]] = []
    referenced_nodes: set[int] = set()
    for _ in range(num_elements):
        if cursor >= len(lines):
            raise ValueError(f"{mesh_path}: truncated element block")
        line_number, line = lines[cursor]
        cursor += 1
        vtk_type = int(line.split()[0])
        nodes = _element_nodes(line, mesh_path, line_number)
        element_types[vtk_type] += 1
        elements.append((vtk_type, nodes))
        referenced_nodes.update(nodes)

    value, cursor = _assignment(lines, cursor, "NPOIN", mesh_path)
    num_points = int(value)
    if num_points < 1:
        raise ValueError(f"{mesh_path}: NPOIN must be positive")
    points = np.empty((num_points, dimension), dtype=np.float64)
    for point_row in range(num_points):
        if cursor >= len(lines):
            raise ValueError(f"{mesh_path}: truncated point block")
        line_number, line = lines[cursor]
        cursor += 1
        values = line.split()
        if len(values) < dimension:
            raise ValueError(f"{mesh_path}:{line_number}: truncated point row")
        try:
            points[point_row] = [float(item) for item in values[:dimension]]
            # SU2's FVM reader associates coordinates with their row index and
            # ignores any trailing label.  Parse that label only for syntax.
            if len(values) > dimension:
                int(values[dimension])
        except ValueError as error:
            raise ValueError(f"{mesh_path}:{line_number}: {error}") from error
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{mesh_path}: point coordinates contain nonfinite values")
    if referenced_nodes and (
        min(referenced_nodes) < 0 or max(referenced_nodes) >= num_points
    ):
        raise ValueError(f"{mesh_path}: volume element references an unknown point")

    value, cursor = _assignment(lines, cursor, "NMARK", mesh_path)
    num_markers = int(value)
    marker_counts: dict[str, int] = {}
    marker_elements: dict[str, tuple[tuple[int, tuple[int, ...]], ...]] = {}
    for _ in range(num_markers):
        marker, cursor = _assignment(lines, cursor, "MARKER_TAG", mesh_path)
        if marker in marker_counts:
            raise ValueError(f"{mesh_path}: duplicate marker {marker}")
        count_text, cursor = _assignment(lines, cursor, "MARKER_ELEMS", mesh_path)
        marker_count = int(count_text)
        marker_counts[marker] = marker_count
        elements_for_marker: list[tuple[int, tuple[int, ...]]] = []
        for _ in range(marker_count):
            if cursor >= len(lines):
                raise ValueError(f"{mesh_path}: truncated marker {marker}")
            line_number, line = lines[cursor]
            cursor += 1
            vtk_type = int(line.split()[0])
            nodes = _element_nodes(line, mesh_path, line_number)
            elements_for_marker.append((vtk_type, nodes))
            if min(nodes) < 0 or max(nodes) >= num_points:
                raise ValueError(
                    f"{mesh_path}:{line_number}: marker references unknown point"
                )
        marker_elements[marker] = tuple(elements_for_marker)
    if cursor != len(lines):
        line_number, line = lines[cursor]
        raise ValueError(f"{mesh_path}:{line_number}: unexpected trailing row {line}")

    return SU2Mesh(
        dimension=dimension,
        num_elements=num_elements,
        num_points=num_points,
        points=points,
        elements=tuple(elements),
        element_type_counts=dict(sorted(element_types.items())),
        marker_elements=marker_elements,
        marker_element_counts=marker_counts,
    )


def read_su2_binary_restart(path: str | Path) -> SU2Restart:
    """Read one native SU2 binary restart using its source-defined layout."""

    restart_path = Path(path)
    with restart_path.open("rb") as handle:
        raw_header = handle.read(4 * SU2_HEADER_INTS)
        if len(raw_header) != 4 * SU2_HEADER_INTS:
            raise ValueError(f"{restart_path}: truncated binary header")
        header_array = np.frombuffer(raw_header, dtype="<i4")
        if int(header_array[0]) != SU2_BINARY_MAGIC:
            raise ValueError(f"{restart_path}: invalid SU2 binary magic")
        byte_order = "little"
        float_dtype = np.dtype("<f8")
        header = tuple(int(value) for value in header_array)
        num_fields, num_points = header[1], header[2]
        if num_fields < 1 or num_points < 1:
            raise ValueError(f"{restart_path}: invalid field or point count")
        if header[3:] != (0, 0):
            raise ValueError(f"{restart_path}: unsupported nonzero reserved header")
        expected_size = (
            4 * SU2_HEADER_INTS
            + num_fields * SU2_FIELD_NAME_BYTES
            + num_fields * num_points * np.dtype("<f8").itemsize
        )
        if restart_path.stat().st_size != expected_size:
            raise ValueError(
                f"{restart_path}: file size is inconsistent with binary header"
            )

        fields: list[str] = []
        for field_index in range(num_fields):
            raw_name = handle.read(SU2_FIELD_NAME_BYTES)
            if len(raw_name) != SU2_FIELD_NAME_BYTES:
                raise ValueError(f"{restart_path}: truncated field-name block")
            encoded, separator, padding = raw_name.partition(b"\0")
            if not separator or any(padding):
                raise ValueError(
                    f"{restart_path}: field {field_index} is not NUL padded"
                )
            try:
                name = encoded.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{restart_path}: field {field_index} is not ASCII"
                ) from error
            if not name:
                raise ValueError(f"{restart_path}: field {field_index} is empty")
            fields.append(name)
        if len(set(fields)) != len(fields):
            raise ValueError(f"{restart_path}: duplicate field names")

        expected_values = num_fields * num_points
        values = np.fromfile(handle, dtype=float_dtype, count=expected_values)
        if values.size != expected_values:
            raise ValueError(f"{restart_path}: truncated point-data block")
        if handle.read(1):
            raise ValueError(f"{restart_path}: unexpected trailing bytes")
    values = values.astype(np.float64, copy=False).reshape(num_points, num_fields)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{restart_path}: restart values contain nonfinite entries")
    return SU2Restart(
        path=restart_path,
        byte_order=byte_order,
        header=header,
        fields=tuple(fields),
        values=values,
        sha256=sha256_file(restart_path),
    )


def load_resource_manifest(
    manifest_path: str | Path, resource_dir: str | Path
) -> dict[str, Any]:
    """Load a manifest and independently verify every declared local resource."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != RESOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError(f"{path}: unsupported resource manifest schema")
    resources = payload.get("resources")
    if not isinstance(resources, dict) or not resources:
        raise ValueError(f"{path}: resources must be a nonempty object")
    root = Path(resource_dir).resolve()
    for resource_name, record in resources.items():
        if not isinstance(record, dict):
            raise TypeError(f"{path}: resource {resource_name} is not an object")
        relative = Path(str(record.get("file")))
        candidate = (root / relative).resolve()
        if candidate.parent != root or relative.name != str(relative):
            raise ValueError(f"{path}: resource {resource_name} has an unsafe path")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        expected_size = int(record.get("bytes", -1))
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"{candidate}: byte size differs from manifest")
        if sha256_file(candidate) != record.get("sha256"):
            raise ValueError(f"{candidate}: SHA256 differs from manifest")
    return payload


def _require_config(config: Mapping[str, str], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        observed = config.get(str(key).upper())
        if observed != str(value):
            raise ValueError(
                f"config {str(key).upper()}={observed!r}, expected {str(value)!r}"
            )


def _require_naca_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Reject manifests that weaken the frozen NACA state/restart semantics."""

    if manifest.get("config_contract") != NACA_CONFIG_CONTRACT:
        raise ValueError(
            "manifest config_contract differs from the frozen NACA contract"
        )
    restart = manifest.get("restart_contract")
    if not isinstance(restart, dict):
        raise TypeError("manifest restart_contract must be an object")
    if tuple(restart.get("indices", ())) != NACA_RESTART_INDICES:
        raise ValueError(
            "manifest restart indices differ from the frozen NACA contract"
        )
    if tuple(restart.get("bdf2_history_indices", ())) != NACA_BDF2_HISTORY_INDICES:
        raise ValueError("manifest BDF2 histories differ from the frozen NACA contract")
    if restart.get("one_step_replay_target_index") != NACA_REPLAY_TARGET_INDEX:
        raise ValueError("manifest replay target differs from the frozen NACA contract")
    if tuple(restart.get("fields", ())) != NACA_RESTART_FIELDS:
        raise ValueError("manifest fields differ from the frozen NACA contract")
    if tuple(restart.get("dynamic_fields", ())) != NACA_DYNAMIC_FIELDS:
        raise ValueError("manifest dynamic fields differ from the frozen NACA contract")
    if manifest.get("claim_boundary") != NACA_STAGE0_CLAIM_BOUNDARY:
        raise ValueError(
            "manifest claim boundary differs from the frozen Stage-0 scope"
        )


def _restart_index(path: Path) -> int:
    match = _RESTART_INDEX.search(path.name)
    if match is None:
        raise ValueError(f"cannot infer restart index from {path.name}")
    return int(match.group(1))


def audit_unsteady_naca0012_bundle(
    *, resource_dir: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Validate the pinned Stage-0 resource and state contract.

    Passing this audit establishes byte identity and interface completeness
    only.  The returned claim flags deliberately remain false until SU2 is run.
    """

    root = Path(resource_dir)
    manifest_file = Path(manifest_path)
    manifest = load_resource_manifest(manifest_file, root)
    _require_naca_manifest_contract(manifest)
    resources = manifest["resources"]
    config = parse_su2_config(root / resources["config"]["file"])
    expected_config = manifest.get("config_contract")
    if not isinstance(expected_config, dict):
        raise TypeError("manifest config_contract must be an object")
    _require_config(config, expected_config)

    mesh = parse_su2_mesh(root / resources["mesh"]["file"])
    expected_mesh = manifest.get("mesh_contract")
    if not isinstance(expected_mesh, dict):
        raise TypeError("manifest mesh_contract must be an object")
    if mesh.dimension != int(expected_mesh.get("dimension", -1)):
        raise ValueError("mesh dimension differs from manifest")
    if mesh.num_elements != int(expected_mesh.get("num_elements", -1)):
        raise ValueError("mesh element count differs from manifest")
    if mesh.num_points != int(expected_mesh.get("num_points", -1)):
        raise ValueError("mesh point count differs from manifest")
    expected_element_types = {
        int(key): int(value)
        for key, value in expected_mesh.get("element_type_counts", {}).items()
    }
    if mesh.element_type_counts != expected_element_types:
        raise ValueError("mesh element types differ from manifest")
    if mesh.marker_element_counts != expected_mesh.get("marker_element_counts"):
        raise ValueError("mesh marker contract differs from manifest")

    restart_contract = manifest.get("restart_contract")
    if not isinstance(restart_contract, dict):
        raise TypeError("manifest restart_contract must be an object")
    expected_fields = tuple(restart_contract.get("fields", ()))
    expected_indices = tuple(int(item) for item in restart_contract.get("indices", ()))
    if expected_fields != NACA_RESTART_FIELDS:
        raise ValueError("manifest does not declare the frozen NACA restart fields")
    if expected_indices != NACA_RESTART_INDICES:
        raise ValueError("NACA BDF2 audit requires histories 497/498 and target 499")

    restarts: list[SU2Restart] = []
    for index in expected_indices:
        key = f"restart_{index:05d}"
        if key not in resources:
            raise ValueError(f"manifest is missing resource {key}")
        restart = read_su2_binary_restart(root / resources[key]["file"])
        if _restart_index(restart.path) != index:
            raise ValueError(f"{restart.path}: restart index differs from manifest")
        if restart.fields != expected_fields:
            raise ValueError(f"{restart.path}: field order differs from manifest")
        if restart.num_points != mesh.num_points:
            raise ValueError(f"{restart.path}: point count differs from mesh")
        restarts.append(restart)

    coordinate_indices = [expected_fields.index("x"), expected_fields.index("y")]
    coordinate_errors = []
    for restart in restarts:
        coordinates = restart.values[:, coordinate_indices]
        error = float(np.max(np.abs(coordinates - mesh.points)))
        coordinate_errors.append(error)
        if error > 1.0e-12:
            raise ValueError(f"{restart.path}: coordinates do not match the mesh")

    dynamic_indices = [expected_fields.index(name) for name in NACA_DYNAMIC_FIELDS]
    ranges: dict[str, dict[str, list[float]]] = {}
    for index, restart in zip(expected_indices, restarts, strict=True):
        state = restart.values[:, dynamic_indices]
        ranges[str(index)] = {
            "minimum": [float(value) for value in np.min(state, axis=0)],
            "maximum": [float(value) for value in np.max(state, axis=0)],
        }

    restart_iter = int(config["RESTART_ITER"])
    history_indices = (restart_iter - 2, restart_iter - 1)
    if (
        history_indices != NACA_BDF2_HISTORY_INDICES
        or restart_iter != NACA_REPLAY_TARGET_INDEX
    ):
        raise ValueError("restart indices do not form the declared BDF2 replay triplet")

    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        upstream = {}

    return {
        "schema": "time_dependent_no.su2_naca0012_stage0_audit.v1",
        "status": "stage0_valid",
        "claim_scope": "resource identity and state-interface validation only",
        "authority_binding": {
            "manifest_file": manifest_file.name,
            "manifest_sha256": sha256_file(manifest_file),
            "tutorial_commit": upstream.get("tutorial_commit"),
            "target_replay_release": upstream.get("target_replay_release"),
            "resource_sha256": {
                name: record["sha256"] for name, record in resources.items()
            },
        },
        "native_solver_executed": False,
        "restart_sufficiency_claimed": False,
        "trusted_transition_claimed": False,
        "mesh": {
            "dimension": mesh.dimension,
            "num_elements": mesh.num_elements,
            "num_points": mesh.num_points,
            "element_type_counts": {
                str(key): value for key, value in mesh.element_type_counts.items()
            },
            "marker_element_counts": dict(mesh.marker_element_counts),
        },
        "restart": {
            "indices": list(expected_indices),
            "bdf2_history_indices": list(history_indices),
            "one_step_replay_target_index": restart_iter,
            "fields": list(expected_fields),
            "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
            "coordinate_max_abs_error": coordinate_errors,
            "dynamic_field_ranges": ranges,
            "files": [
                {
                    "file": restart.path.name,
                    "sha256": restart.sha256,
                    "byte_order": restart.byte_order,
                    "header": list(restart.header),
                }
                for restart in restarts
            ],
        },
    }


def _render_config_with_overrides(
    source_text: str, overrides: Mapping[str, str]
) -> str:
    """Preserve an SU2 config while replacing or appending exact assignments."""

    rendered: list[str] = []
    observed: set[str] = set()
    for raw_line in source_text.splitlines():
        active = raw_line.split("%", maxsplit=1)[0].strip()
        if "=" in active:
            key = active.split("=", maxsplit=1)[0].strip().upper()
            if key in overrides:
                if key in observed:
                    raise ValueError(f"duplicate replay override target {key}")
                rendered.append(f"{key}= {overrides[key]}")
                observed.add(key)
                continue
        rendered.append(raw_line)

    missing = [key for key in overrides if key not in observed]
    if missing:
        rendered.extend(("", "% Frozen one-step replay overrides"))
        rendered.extend(f"{key}= {overrides[key]}" for key in missing)
    return "\n".join(rendered) + "\n"


def prepare_unsteady_naca0012_replay_case(
    *,
    resource_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create an isolated, non-executing 497/498-to-499 SU2 replay case.

    The canonical 499 target is deliberately not copied. The generated output
    stem is distinct from the input stem, so a later native run cannot overwrite
    the reference even if paths are confused.
    """

    resource_root = Path(resource_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"replay output directory already exists: {destination}")
    if destination == resource_root or resource_root in destination.parents:
        raise ValueError("replay output directory must be outside the resource bundle")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"replay output parent does not exist: {destination.parent}"
        )

    stage0 = audit_unsteady_naca0012_bundle(
        resource_dir=resource_root,
        manifest_path=manifest_path,
    )
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    resources = manifest["resources"]
    authority = stage0["authority_binding"]
    authority_resources = authority["resource_sha256"]
    target_record = resources["restart_00499"]
    target_path = resource_root / target_record["file"]
    target_sha256 = sha256_file(target_path)
    if target_sha256 != authority_resources["restart_00499"]:
        raise ValueError("canonical replay target differs from Stage-0 authority")

    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(f"stale replay staging directory exists: {staging}")
    staging.mkdir()
    try:
        staged_sources = {
            "license": ("license", "LICENSE"),
            "mesh": ("mesh", resources["mesh"]["file"]),
            "restart_00497": (
                "restart_00497",
                resources["restart_00497"]["file"],
            ),
            "restart_00498": (
                "restart_00498",
                resources["restart_00498"]["file"],
            ),
            "upstream_config": (
                "config",
                "upstream_unsteady_naca0012.cfg",
            ),
        }
        staged_files: dict[str, dict[str, Any]] = {}
        for role, (resource_key, staged_name) in staged_sources.items():
            source = resource_root / resources[resource_key]["file"]
            staged = staging / staged_name
            shutil.copyfile(source, staged)
            digest = sha256_file(staged)
            if digest != authority_resources[resource_key]:
                raise ValueError(f"staged replay input differs from authority: {role}")
            staged_files[role] = {
                "file": staged.name,
                "bytes": staged.stat().st_size,
                "sha256": digest,
            }

        staged_manifest = staging / manifest_file.name
        shutil.copyfile(manifest_file, staged_manifest)
        staged_manifest_sha256 = sha256_file(staged_manifest)
        if staged_manifest_sha256 != authority["manifest_sha256"]:
            raise ValueError("staged resource manifest differs from Stage-0 authority")
        upstream_config = staging / staged_files["upstream_config"]["file"]
        replay_text = _render_config_with_overrides(
            upstream_config.read_text(encoding="utf-8", errors="strict"),
            NACA_REPLAY_CONFIG_OVERRIDES,
        )
        replay_config = staging / NACA_REPLAY_CONFIG_FILENAME
        replay_config.write_text(replay_text, encoding="utf-8", newline="\n")
        parsed_replay = parse_su2_config(replay_config)
        _require_config(parsed_replay, NACA_REPLAY_CONFIG_OVERRIDES)

        forbidden_target = staging / target_record["file"]
        if forbidden_target.exists():
            raise ValueError(
                "canonical replay target was copied into the solver directory"
            )
        if (staging / NACA_REPLAY_OUTPUT_FILENAME).exists():
            raise ValueError("replay output already exists before native execution")
        if sha256_file(target_path) != authority_resources["restart_00499"]:
            raise ValueError("canonical replay target changed during case preparation")

        contract = {
            "schema": NACA_REPLAY_CASE_SCHEMA,
            "status": "prepared_not_executed",
            "stage0_authority_binding": stage0["authority_binding"],
            "replay_config": {
                "file": replay_config.name,
                "bytes": replay_config.stat().st_size,
                "sha256": sha256_file(replay_config),
                "overrides": dict(NACA_REPLAY_CONFIG_OVERRIDES),
            },
            "staged_inputs": staged_files,
            "staged_resource_manifest": {
                "file": staged_manifest.name,
                "sha256": staged_manifest_sha256,
            },
            "external_reference_target": {
                "file": target_record["file"],
                "sha256": target_sha256,
                "copied_into_case": False,
            },
            "expected_native_command": ["SU2_CFD", replay_config.name],
            "expected_output": NACA_REPLAY_OUTPUT_FILENAME,
            "native_solver_executed": False,
            "restart_sufficiency_claimed": False,
            "trusted_transition_claimed": False,
        }
        contract_path = staging / "replay_case.json"
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return contract
