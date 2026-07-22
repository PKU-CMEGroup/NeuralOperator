"""Recover and validate the physical mesh contract behind CPG bump data.

The released HDF5 files contain point coordinates, a graph, and node labels,
but omit the raw mesh lineage and boundary geometry.  The extracted bump cases
contain an Abaqus mesh and ASCII VTU snapshots.  This module provides the small
set of parsers and checks needed to establish whether those three
representations describe the same points and primal mesh edges.

The recovered graph is a vertex graph.  Nothing here promotes graph vertices
to finite-volume control volumes or primal graph edges to physical control-
volume faces.  Derived vertex-area weights and nodal normals are geometric
diagnostics, not proof of the measure used by the released learned update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np


MESH_CONTRACT_SCHEMA = "cpg_bump_mesh_contract_v1"
PHYSICAL_BOUNDARY_SETS = ("BOTTOM", "WALL", "TOP", "RIGHT", "LEFT")
WALL_BOUNDARY_SETS = ("BOTTOM", "WALL", "TOP")
NORMAL_NODE = 0
WALL_NODE = 1
OUTFLOW_NODE = 2
INFLOW_NODE = 3


@dataclass(frozen=True)
class AbaqusElement:
    """One observed Abaqus element."""

    element_id: int
    element_type: str
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class AbaqusMesh:
    """Minimal Abaqus mesh representation used by the CPG audit."""

    node_ids: np.ndarray
    points: np.ndarray
    elements: tuple[AbaqusElement, ...]
    element_sets: Mapping[str, tuple[int, ...]]
    node_sets: Mapping[str, tuple[int, ...]]

    @property
    def num_nodes(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class VTUFrame:
    """Point fields and static mesh arrays from one ASCII VTU snapshot."""

    points: np.ndarray
    point_data: Mapping[str, np.ndarray]
    connectivity: np.ndarray
    offsets: np.ndarray
    cell_types: np.ndarray
    number_of_points: int
    number_of_cells: int


@dataclass(frozen=True)
class BoundaryStencil:
    """Sparse current-interior to boundary extrapolation stencil."""

    target_nodes: np.ndarray
    target_rows: np.ndarray
    source_nodes: np.ndarray
    weights: np.ndarray
    fallback_target_count: int


GRAPH_BOUNDARY_GEOMETRY_ATOL = 5.0e-6


def parse_abaqus_mesh(path: str | Path) -> AbaqusMesh:
    """Parse the node, element, ELSET, and NSET blocks used by bump cases."""

    mesh_path = Path(path)
    lines = mesh_path.read_text(encoding="utf-8", errors="strict").splitlines()
    nodes: dict[int, tuple[float, float]] = {}
    elements: dict[int, AbaqusElement] = {}
    element_sets: dict[str, set[int]] = {}
    node_sets: dict[str, set[int]] = {}
    section: str | None = None
    parameters: dict[str, str] = {}
    flags: set[str] = set()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            section, parameters, flags = _parse_abaqus_header(line)
            if section not in {"NODE", "ELEMENT", "ELSET", "NSET"}:
                section = None
            continue

        values = [item.strip() for item in line.split(",") if item.strip()]
        try:
            if section == "NODE":
                if len(values) < 3:
                    raise ValueError("node row has fewer than x/y coordinates")
                node_id = int(values[0])
                if node_id in nodes:
                    raise ValueError(f"duplicate node id {node_id}")
                nodes[node_id] = (float(values[1]), float(values[2]))
            elif section == "ELEMENT":
                if len(values) < 3:
                    raise ValueError("element row has fewer than two nodes")
                element_id = int(values[0])
                if element_id in elements:
                    raise ValueError(f"duplicate element id {element_id}")
                element_type = parameters.get("TYPE", "").upper()
                node_ids = tuple(int(item) for item in values[1:])
                elements[element_id] = AbaqusElement(
                    element_id=element_id,
                    element_type=element_type,
                    node_ids=node_ids,
                )
                implicit_set = parameters.get("ELSET")
                if implicit_set:
                    element_sets.setdefault(implicit_set.upper(), set()).add(element_id)
            elif section in {"ELSET", "NSET"}:
                set_key = "ELSET" if section == "ELSET" else "NSET"
                set_name = parameters.get(set_key)
                if not set_name:
                    raise ValueError(f"{section} header has no {set_key} name")
                target = element_sets if section == "ELSET" else node_sets
                parsed = _parse_set_values(values, generate="GENERATE" in flags)
                target.setdefault(set_name.upper(), set()).update(parsed)
        except ValueError as error:
            raise ValueError(f"{mesh_path}:{line_number}: {error}") from error

    if not nodes:
        raise ValueError(f"{mesh_path} contains no nodes")
    if not elements:
        raise ValueError(f"{mesh_path} contains no elements")

    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    points = np.asarray([nodes[int(node_id)] for node_id in node_ids], dtype=np.float64)
    element_tuple = tuple(elements[element_id] for element_id in sorted(elements))
    known_nodes = set(nodes)
    for element in element_tuple:
        missing = set(element.node_ids) - known_nodes
        if missing:
            raise ValueError(
                f"element {element.element_id} references unknown nodes {sorted(missing)}"
            )

    return AbaqusMesh(
        node_ids=node_ids,
        points=points,
        elements=element_tuple,
        element_sets={
            name: tuple(sorted(values)) for name, values in element_sets.items()
        },
        node_sets={name: tuple(sorted(values)) for name, values in node_sets.items()},
    )


def mesh_primal_edges(mesh: AbaqusMesh) -> np.ndarray:
    """Return sorted unique zero-based edges of all quadrilateral cells."""

    node_index = _node_index(mesh)
    edges: set[tuple[int, int]] = set()
    for element in _volume_elements(mesh):
        indices = [node_index[node_id] for node_id in element.node_ids]
        for left, right in zip(indices, indices[1:] + indices[:1], strict=True):
            if left == right:
                raise ValueError(f"element {element.element_id} has a zero-length edge")
            edges.add((min(left, right), max(left, right)))
    if not edges:
        raise ValueError("mesh contains no quadrilateral primal edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def expected_cpg_node_types(mesh: AbaqusMesh) -> np.ndarray:
    """Recover CPG node labels from named bump boundary node sets.

    Wall labels are assigned first.  Right and left boundary labels then take
    precedence at their corner nodes, matching the observed CPG convention.
    """

    missing = [name for name in PHYSICAL_BOUNDARY_SETS if name not in mesh.node_sets]
    if missing:
        raise ValueError(f"mesh is missing physical NSETs {missing}")
    node_index = _node_index(mesh)
    types = np.full(mesh.num_nodes, NORMAL_NODE, dtype=np.int64)
    for name in WALL_BOUNDARY_SETS:
        types[_set_indices(mesh.node_sets[name], node_index, name)] = WALL_NODE
    types[_set_indices(mesh.node_sets["RIGHT"], node_index, "RIGHT")] = OUTFLOW_NODE
    types[_set_indices(mesh.node_sets["LEFT"], node_index, "LEFT")] = INFLOW_NODE
    right = set(mesh.node_sets["RIGHT"])
    left = set(mesh.node_sets["LEFT"])
    if right & left:
        raise ValueError("left and right boundary node sets overlap")
    return types


def recover_boundary_geometry(mesh: AbaqusMesh) -> dict[str, np.ndarray]:
    """Recover outward boundary normals and diagnostic vertex-area weights."""

    element_by_id = {element.element_id: element for element in mesh.elements}
    node_index = _node_index(mesh)
    volume_elements = _volume_elements(mesh)
    adjacency: dict[tuple[int, int], list[AbaqusElement]] = {}
    for element in volume_elements:
        indices = [node_index[node_id] for node_id in element.node_ids]
        for left, right in zip(indices, indices[1:] + indices[:1], strict=True):
            adjacency.setdefault((min(left, right), max(left, right)), []).append(
                element
            )

    expected_types = expected_cpg_node_types(mesh)
    normal_sums = {
        code: np.zeros((mesh.num_nodes, 2), dtype=np.float64)
        for code in (WALL_NODE, OUTFLOW_NODE, INFLOW_NODE)
    }
    nodal_measures = {
        code: np.zeros(mesh.num_nodes, dtype=np.float64)
        for code in (WALL_NODE, OUTFLOW_NODE, INFLOW_NODE)
    }
    boundary_edges: list[tuple[int, int]] = []
    boundary_normals: list[np.ndarray] = []
    boundary_lengths: list[float] = []
    boundary_codes: list[int] = []
    used_line_elements: set[int] = set()

    for name in PHYSICAL_BOUNDARY_SETS:
        if name not in mesh.element_sets:
            raise ValueError(f"mesh is missing physical ELSET {name}")
        code = _boundary_code(name)
        nset = set(mesh.node_sets[name])
        for element_id in mesh.element_sets[name]:
            element = element_by_id.get(element_id)
            if element is None:
                raise ValueError(
                    f"ELSET {name} references unknown element {element_id}"
                )
            if len(element.node_ids) != 2:
                raise ValueError(f"ELSET {name} contains non-line element {element_id}")
            if not set(element.node_ids).issubset(nset):
                raise ValueError(
                    f"ELSET {name} element {element_id} is inconsistent with its NSET"
                )
            used_line_elements.add(element_id)
            left = node_index[element.node_ids[0]]
            right = node_index[element.node_ids[1]]
            canonical = (min(left, right), max(left, right))
            neighbors = adjacency.get(canonical, [])
            if len(neighbors) != 1:
                raise ValueError(
                    f"boundary edge {canonical} has {len(neighbors)} adjacent quads"
                )
            edge_normal, edge_length = _outward_edge_normal(
                mesh.points, left, right, neighbors[0], node_index
            )
            boundary_edges.append((left, right))
            boundary_normals.append(edge_normal)
            boundary_lengths.append(edge_length)
            boundary_codes.append(code)
            for node in (left, right):
                normal_sums[code][node] += edge_length * edge_normal
                nodal_measures[code][node] += 0.5 * edge_length

    all_line_elements = {
        element.element_id for element in mesh.elements if len(element.node_ids) == 2
    }
    if used_line_elements != all_line_elements:
        missing = sorted(all_line_elements - used_line_elements)
        extra = sorted(used_line_elements - all_line_elements)
        raise ValueError(
            "physical boundary sets do not partition line elements: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    node_normals = np.zeros((mesh.num_nodes, 2), dtype=np.float64)
    node_measures = np.zeros(mesh.num_nodes, dtype=np.float64)
    node_normal_coherence = np.zeros(mesh.num_nodes, dtype=np.float64)
    for code in (WALL_NODE, OUTFLOW_NODE, INFLOW_NODE):
        selected = expected_types == code
        vectors = normal_sums[code][selected]
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0.0):
            bad = np.flatnonzero(selected)[norms <= 0.0]
            raise ValueError(f"boundary nodes {bad[:8].tolist()} lack normals")
        node_normals[selected] = vectors / norms[:, None]
        node_measures[selected] = nodal_measures[code][selected]
        node_normal_coherence[selected] = norms / (2.0 * nodal_measures[code][selected])

    primal_boundary = {edge for edge, cells in adjacency.items() if len(cells) == 1}
    recovered_boundary = {
        (min(left, right), max(left, right)) for left, right in boundary_edges
    }
    if recovered_boundary != primal_boundary:
        raise ValueError("named line elements do not equal the quad-mesh boundary")

    return {
        "node_type": expected_types,
        "node_normal": node_normals,
        "node_boundary_normal_coherence": node_normal_coherence,
        "node_boundary_measure": node_measures,
        "lumped_vertex_area": lumped_vertex_areas(mesh),
        "boundary_edges": np.asarray(boundary_edges, dtype=np.int64),
        "boundary_edge_normal": np.asarray(boundary_normals, dtype=np.float64),
        "boundary_edge_length": np.asarray(boundary_lengths, dtype=np.float64),
        "boundary_edge_node_type": np.asarray(boundary_codes, dtype=np.int64),
    }


def _graph_boundary_cycle(
    points: np.ndarray, edges: Any, types: np.ndarray
) -> tuple[np.ndarray, list[list[int]]]:
    canonical = np.sort(_normalize_edges(edges, points.shape[0]), axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError("graph repeats an undirected edge")
    canonical = canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]
    adjacency: list[list[int]] = [[] for _ in range(points.shape[0])]
    for left, right in canonical:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    boundary = canonical[
        (types[canonical[:, 0]] != NORMAL_NODE)
        & (types[canonical[:, 1]] != NORMAL_NODE)
    ]
    boundary_nodes = np.flatnonzero(types != NORMAL_NODE)
    neighbors: list[list[int]] = [[] for _ in range(points.shape[0])]
    for left, right in boundary:
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    if any(len(neighbors[node]) != 2 for node in boundary_nodes):
        raise ValueError("non-normal induced graph is not a simple boundary cycle")
    visited = {int(boundary_nodes[0])}
    frontier = list(visited)
    while frontier:
        for neighbor in neighbors[frontier.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if len(visited) != boundary_nodes.size:
        raise ValueError("non-normal induced graph has multiple boundary components")
    return boundary, adjacency


def _graph_boundary_edge_normal(
    points: np.ndarray,
    types: np.ndarray,
    adjacency: list[list[int]],
    left: int,
    right: int,
) -> tuple[np.ndarray, float]:
    vector = points[right] - points[left]
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"boundary edge {(left, right)} has zero length")
    normal = np.asarray([-vector[1], vector[0]], dtype=np.float64) / length
    interior = sorted(
        {
            neighbor
            for neighbor in adjacency[left] + adjacency[right]
            if types[neighbor] == NORMAL_NODE
        }
    )
    if not interior:
        raise ValueError(f"boundary edge {(left, right)} has no adjacent normal node")
    inward = np.mean(points[interior], axis=0) - 0.5 * (points[left] + points[right])
    orientation = float(normal @ inward)
    if orientation > 0.0:
        normal, orientation = -normal, -orientation
    if orientation >= -1.0e-14:
        raise ValueError(f"cannot orient boundary edge {(left, right)} from the graph")
    return normal, length


def recover_graph_boundary_geometry(
    *, pos: Any, edges: Any, node_type: Any
) -> dict[str, np.ndarray]:
    """Recover nodal boundary geometry from a released CPG vertex graph.

    This supports the causal nodal boundary counterfactual. It does not
    identify graph nodes as control volumes or graph edges as physical faces.
    """

    points = np.asarray(pos, dtype=np.float64)
    types = np.asarray(node_type).reshape(-1).astype(np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pos must have shape (num_nodes, 2)")
    if types.shape != (points.shape[0],):
        raise ValueError("node_type must contain one label per point")
    supported = {NORMAL_NODE, WALL_NODE, OUTFLOW_NODE, INFLOW_NODE}
    if not set(np.unique(types)).issubset(supported):
        raise ValueError("node_type contains an unsupported CPG boundary label")
    codes = (WALL_NODE, OUTFLOW_NODE, INFLOW_NODE)
    if any(not np.any(types == code) for code in codes):
        raise ValueError("node_type must contain wall, outflow, and inflow nodes")

    boundary_edges, adjacency = _graph_boundary_cycle(points, edges, types)
    normal_sums = {
        code: np.zeros((points.shape[0], 2), dtype=np.float64) for code in codes
    }
    nodal_measures = {
        code: np.zeros(points.shape[0], dtype=np.float64) for code in codes
    }
    edge_normals: list[np.ndarray] = []
    edge_lengths: list[float] = []
    edge_codes: list[int] = []
    for left, right in boundary_edges:
        left_i, right_i = int(left), int(right)
        normal, length = _graph_boundary_edge_normal(
            points, types, adjacency, left_i, right_i
        )
        left_type, right_type = int(types[left_i]), int(types[right_i])
        if left_type == right_type == INFLOW_NODE:
            code = INFLOW_NODE
        elif left_type == right_type == OUTFLOW_NODE:
            code = OUTFLOW_NODE
        elif WALL_NODE in {left_type, right_type}:
            code = WALL_NODE
        else:
            raise ValueError(
                f"boundary edge {(left_i, right_i)} has incompatible endpoint "
                f"types {(left_type, right_type)}"
            )
        edge_normals.append(normal)
        edge_lengths.append(length)
        edge_codes.append(code)
        for node in (left_i, right_i):
            normal_sums[code][node] += length * normal
            nodal_measures[code][node] += 0.5 * length

    node_normals = np.zeros_like(points)
    node_measures = np.zeros(points.shape[0], dtype=np.float64)
    coherence = np.zeros(points.shape[0], dtype=np.float64)
    for code in codes:
        selected = types == code
        vectors = normal_sums[code][selected]
        norms = np.linalg.norm(vectors, axis=1)
        measures = nodal_measures[code][selected]
        if np.any(norms <= 0.0) or np.any(measures <= 0.0):
            raise ValueError(f"boundary type {code} lacks valid nodal geometry")
        node_normals[selected] = vectors / norms[:, None]
        node_measures[selected] = measures
        coherence[selected] = norms / (2.0 * measures)

    return {
        "node_type": types,
        "node_normal": node_normals,
        "node_boundary_normal_coherence": coherence,
        "node_boundary_measure": node_measures,
        "boundary_edges": boundary_edges.astype(np.int64, copy=False),
        "boundary_edge_normal": np.asarray(edge_normals, dtype=np.float64),
        "boundary_edge_length": np.asarray(edge_lengths, dtype=np.float64),
        "boundary_edge_node_type": np.asarray(edge_codes, dtype=np.int64),
    }


def lumped_vertex_areas(mesh: AbaqusMesh) -> np.ndarray:
    """Distribute each quadrilateral polygon area equally to its vertices."""

    node_index = _node_index(mesh)
    weights = np.zeros(mesh.num_nodes, dtype=np.float64)
    for element in _volume_elements(mesh):
        indices = np.asarray([node_index[node_id] for node_id in element.node_ids])
        polygon = mesh.points[indices]
        area = 0.5 * abs(
            float(
                np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
            )
        )
        if area <= 0.0:
            raise ValueError(f"quadrilateral {element.element_id} has nonpositive area")
        weights[indices] += area / len(indices)
    if np.any(weights <= 0.0):
        bad = np.flatnonzero(weights <= 0.0)
        raise ValueError(
            f"mesh vertices {bad[:8].tolist()} have no positive area weight"
        )
    return weights


def validate_hdf_mesh_identity(
    mesh: AbaqusMesh,
    *,
    hdf_pos: Any,
    hdf_edges: Any,
    hdf_node_type: Any,
    position_atol: float = 2.0e-6,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate exact HDF node/label identity and primal-edge set equality."""

    pos = np.asarray(hdf_pos, dtype=np.float64)
    if pos.shape != mesh.points.shape:
        raise ValueError(
            f"HDF positions have shape {pos.shape}; mesh has {mesh.points.shape}"
        )
    position_error = float(np.max(np.abs(pos - mesh.points)))
    if position_error > position_atol:
        raise ValueError(
            f"HDF positions do not match sorted Abaqus nodes (max {position_error})"
        )

    edges = _normalize_edges(hdf_edges, mesh.num_nodes)
    canonical = np.sort(edges, axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError("HDF graph repeats a primal edge")
    canonical = canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]
    expected_edges = mesh_primal_edges(mesh)
    if not np.array_equal(canonical, expected_edges):
        expected_set = {tuple(edge) for edge in expected_edges.tolist()}
        actual_set = {tuple(edge) for edge in canonical.tolist()}
        raise ValueError(
            "HDF graph is not the quadrilateral primal-edge set: "
            f"missing={len(expected_set - actual_set)}, "
            f"extra={len(actual_set - expected_set)}"
        )

    node_type = np.asarray(hdf_node_type).reshape(-1).astype(np.int64)
    expected_type = expected_cpg_node_types(mesh)
    if node_type.shape != expected_type.shape:
        raise ValueError("HDF node labels do not match the mesh node count")
    if not np.array_equal(node_type, expected_type):
        mismatch = np.flatnonzero(node_type != expected_type)
        raise ValueError(
            f"HDF node labels disagree with named boundary sets at "
            f"{mismatch.size} nodes"
        )

    geometry = recover_boundary_geometry(mesh)
    graph_geometry = recover_graph_boundary_geometry(
        pos=pos,
        edges=edges,
        node_type=node_type,
    )
    physical_boundary = {
        tuple(sorted((int(left), int(right))))
        for left, right in geometry["boundary_edges"]
    }
    graph_boundary = {
        tuple(sorted((int(left), int(right))))
        for left, right in graph_geometry["boundary_edges"]
    }
    if graph_boundary != physical_boundary:
        raise ValueError("graph-native boundary cycle differs from Abaqus boundary")
    graph_normal_error = float(
        np.max(np.abs(graph_geometry["node_normal"] - geometry["node_normal"]))
    )
    graph_coherence_error = float(
        np.max(
            np.abs(
                graph_geometry["node_boundary_normal_coherence"]
                - geometry["node_boundary_normal_coherence"]
            )
        )
    )
    graph_geometry_atol = max(GRAPH_BOUNDARY_GEOMETRY_ATOL, 2.5 * position_atol)
    if graph_normal_error > graph_geometry_atol:
        raise ValueError(
            "graph-native nodal normals differ from Abaqus normals "
            f"(max {graph_normal_error})"
        )
    if graph_coherence_error > graph_geometry_atol:
        raise ValueError(
            "graph-native normal coherence differs from Abaqus geometry "
            f"(max {graph_coherence_error})"
        )
    volumes = _volume_elements(mesh)
    euler_characteristic = mesh.num_nodes - expected_edges.shape[0] + len(volumes)
    summary = {
        "schema": MESH_CONTRACT_SCHEMA,
        "num_nodes": mesh.num_nodes,
        "num_volume_quads": len(volumes),
        "num_primal_edges": int(expected_edges.shape[0]),
        "num_boundary_edges": int(geometry["boundary_edges"].shape[0]),
        "planar_euler_characteristic": int(euler_characteristic),
        "hdf_to_abaqus_node_mapping": "verified_sorted_node_id_index_identity",
        "hdf_graph_mapping": "verified_quadrilateral_primal_edge_set",
        "hdf_node_type_mapping": "verified_named_boundary_node_sets",
        "graph_native_boundary_mapping": "verified_against_abaqus_boundary",
        "graph_native_normal_max_abs_error": graph_normal_error,
        "graph_native_normal_coherence_max_abs_error": graph_coherence_error,
        "position_max_abs_error": position_error,
        "node_type_counts": {
            str(code): int(np.count_nonzero(node_type == code)) for code in range(4)
        },
        "graph_node_control_volume_status": "not_established_point_sample",
        "graph_edge_control_volume_face_status": "not_established_primal_edge_only",
        "lumped_vertex_area_status": "derived_geometry_not_validated_solver_measure",
    }
    return summary, {"primal_edges": expected_edges, **geometry}


def read_ascii_vtu(path: str | Path) -> VTUFrame:
    """Read the ASCII arrays used by the extracted Trixi VTU snapshots."""

    vtu_path = Path(path)
    root = ET.parse(vtu_path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{vtu_path} contains no UnstructuredGrid Piece")
    number_of_points = int(piece.attrib["NumberOfPoints"])
    number_of_cells = int(piece.attrib["NumberOfCells"])

    point_data: dict[str, np.ndarray] = {}
    point_data_element = piece.find("PointData")
    if point_data_element is None:
        raise ValueError(f"{vtu_path} contains no PointData")
    for data_array in point_data_element.findall("DataArray"):
        name = data_array.attrib.get("Name")
        if name:
            point_data[name] = _read_ascii_data_array(data_array, vtu_path)

    points_element = piece.find("Points/DataArray")
    if points_element is None:
        raise ValueError(f"{vtu_path} contains no point coordinates")
    point_values = _read_ascii_data_array(points_element, vtu_path)
    components = int(points_element.attrib.get("NumberOfComponents", "3"))
    if point_values.size != number_of_points * components:
        raise ValueError(f"{vtu_path} point coordinate count is inconsistent")
    points = point_values.reshape(number_of_points, components)[:, :2]

    cells_element = piece.find("Cells")
    if cells_element is None:
        raise ValueError(f"{vtu_path} contains no Cells")
    cell_arrays = {
        element.attrib.get("Name", ""): _read_ascii_data_array(element, vtu_path)
        for element in cells_element.findall("DataArray")
    }
    required = ("connectivity", "offsets", "types")
    missing = [name for name in required if name not in cell_arrays]
    if missing:
        raise ValueError(f"{vtu_path} is missing cell arrays {missing}")
    connectivity = cell_arrays["connectivity"].astype(np.int64)
    offsets = cell_arrays["offsets"].astype(np.int64)
    cell_types = cell_arrays["types"].astype(np.uint8)
    if offsets.shape != (number_of_cells,) or cell_types.shape != (number_of_cells,):
        raise ValueError(f"{vtu_path} cell metadata count is inconsistent")
    return VTUFrame(
        points=points,
        point_data=point_data,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
        number_of_points=number_of_points,
        number_of_cells=number_of_cells,
    )


def validate_vtu_mesh_identity(
    mesh: AbaqusMesh, frame: VTUFrame, *, position_atol: float = 1.0e-10
) -> dict[str, Any]:
    """Validate point order and every line/quad cell against the Abaqus mesh."""

    if frame.points.shape != mesh.points.shape:
        raise ValueError("VTU and Abaqus point counts differ")
    position_error = float(np.max(np.abs(frame.points - mesh.points)))
    if position_error > position_atol:
        raise ValueError(f"VTU points do not match Abaqus nodes (max {position_error})")

    node_index = _node_index(mesh)
    expected_cells = [
        np.asarray(
            [node_index[node_id] for node_id in element.node_ids], dtype=np.int64
        )
        for element in mesh.elements
    ]
    expected_connectivity = np.concatenate(expected_cells)
    expected_offsets = np.cumsum([cell.size for cell in expected_cells], dtype=np.int64)
    expected_types = np.asarray(
        [_vtk_cell_type(element) for element in mesh.elements], dtype=np.uint8
    )
    if not np.array_equal(frame.connectivity, expected_connectivity):
        raise ValueError("VTU connectivity does not match Abaqus element order")
    if not np.array_equal(frame.offsets, expected_offsets):
        raise ValueError("VTU offsets do not match Abaqus element order")
    if not np.array_equal(frame.cell_types, expected_types):
        raise ValueError("VTU cell types do not match Abaqus element types")
    return {
        "vtu_to_abaqus_node_mapping": "verified_index_identity",
        "vtu_to_abaqus_cell_mapping": "verified_element_id_order_identity",
        "vtu_position_max_abs_error": position_error,
        "number_of_points": frame.number_of_points,
        "number_of_cells": frame.number_of_cells,
    }


def vtu_primitive(frame: VTUFrame) -> np.ndarray:
    """Return VTU point primitives in CPG order [rho, v1, v2, pressure]."""

    missing = [
        name for name in ("rho", "v1", "v2", "p") if name not in frame.point_data
    ]
    if missing:
        raise ValueError(f"VTU point data is missing primitive fields {missing}")
    primitive = np.column_stack(
        [frame.point_data[name] for name in ("rho", "v1", "v2", "p")]
    ).astype(np.float64)
    if primitive.shape != (frame.number_of_points, 4):
        raise ValueError("VTU primitive fields have inconsistent point counts")
    return primitive


def primitive_error(reference: Any, candidate: Any) -> dict[str, Any]:
    """Return absolute and relative differences for two primitive point fields."""

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("primitive arrays must share shape (num_nodes, 4)")
    difference = left - right
    channel_norm = np.linalg.norm(left, axis=0)
    relative = np.divide(
        np.linalg.norm(difference, axis=0),
        channel_norm,
        out=np.full(4, np.inf, dtype=np.float64),
        where=channel_norm > 0.0,
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "channel_max_abs": np.max(np.abs(difference), axis=0).tolist(),
        "channel_relative_l2": relative.tolist(),
    }


def audit_bump_julia_config(path: str | Path) -> dict[str, Any]:
    """Recover the fixed physical and boundary settings from an observed case."""

    source = Path(path).read_text(encoding="utf-8", errors="strict")
    result = {
        "gamma": _julia_number(source, r"const\s+gamma\s*=\s*([0-9.eE+-]+)"),
        "rho_inf": _julia_number(source, r"const\s+rho_inf\s*=\s*([0-9.eE+-]+)"),
        "p_inf": _julia_number(source, r"const\s+p_inf\s*=\s*([0-9.eE+-]+)"),
        "polydeg": int(_julia_number(source, r"polydeg\s*=\s*([0-9]+)")),
        "save_dt": _julia_number(
            source, r"SaveSolutionCallback\s*\(\s*dt\s*=\s*([0-9.eE+-]+)"
        ),
    }
    required_evidence = {
        "inflow": "BoundaryConditionDirichlet(initial_condition_mach3_flow)",
        "outflow": "flux = Trixi.flux(u_inner, normal_direction, equations)",
        "wall": "boundary_condition_slip_wall",
        "mesh": "P4estMesh{2}",
    }
    missing = [name for name, token in required_evidence.items() if token not in source]
    if missing:
        raise ValueError(
            f"Bump.jl is missing expected configuration evidence {missing}"
        )
    result["boundary_contract"] = {
        "inflow": "supersonic_dirichlet_freestream",
        "wall": "slip_wall",
        "outflow": "inner_state_physical_flux",
    }
    return result


def freestream_primitive(
    mach: float, *, gamma: float = 1.4, rho_inf: float = 1.4, p_inf: float = 1.0
) -> np.ndarray:
    """Return the bump-case freestream primitive state."""

    if mach <= 0.0 or gamma <= 1.0 or rho_inf <= 0.0 or p_inf <= 0.0:
        raise ValueError("freestream parameters must be physical")
    sound_speed = np.sqrt(gamma * p_inf / rho_inf)
    return np.asarray([rho_inf, mach * sound_speed, 0.0, p_inf], dtype=np.float64)


def build_boundary_stencil(
    *,
    pos: Any,
    edges: Any,
    node_type: Any,
    node_normal: Any,
    max_source_hops: int = 3,
) -> BoundaryStencil:
    """Build a causal first-order interior extrapolation stencil.

    Wall and outflow point states use adjacent normal nodes.  Neighbors lying
    inward relative to the recovered outward normal are preferred.  A target
    falls back to all adjacent normal nodes only when no strictly inward normal
    neighbor exists.
    """

    points = np.asarray(pos, dtype=np.float64)
    types = np.asarray(node_type).reshape(-1).astype(np.int64)
    normals = np.asarray(node_normal, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pos must have shape (num_nodes, 2)")
    if types.shape != (points.shape[0],) or normals.shape != points.shape:
        raise ValueError("node_type and node_normal must match positions")
    if max_source_hops < 1:
        raise ValueError("max_source_hops must be positive")
    graph_edges = _normalize_edges(edges, points.shape[0])
    adjacency: list[list[int]] = [[] for _ in range(points.shape[0])]
    for left, right in graph_edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))

    targets = np.flatnonzero((types == WALL_NODE) | (types == OUTFLOW_NODE))
    source_nodes: list[int] = []
    target_rows: list[int] = []
    weights: list[float] = []
    fallback_count = 0
    for row, target in enumerate(targets):
        candidates = _nearest_normal_nodes(
            adjacency, types, int(target), max_hops=max_source_hops
        )
        if candidates.size == 0:
            raise ValueError(
                f"boundary node {target} has no normal node within "
                f"{max_source_hops} graph hops"
            )
        displacement = points[candidates] - points[target]
        inward = displacement @ normals[target] < -1.0e-12
        selected = candidates[inward]
        selected_displacement = displacement[inward]
        if selected.size == 0:
            selected = candidates
            selected_displacement = displacement
            fallback_count += 1
        distance = np.linalg.norm(selected_displacement, axis=1)
        if np.any(distance <= 0.0):
            raise ValueError(f"boundary node {target} has a coincident graph neighbor")
        inverse = 1.0 / distance
        normalized = inverse / np.sum(inverse)
        source_nodes.extend(int(value) for value in selected)
        target_rows.extend([row] * selected.size)
        weights.extend(float(value) for value in normalized)
    return BoundaryStencil(
        target_nodes=targets.astype(np.int64),
        target_rows=np.asarray(target_rows, dtype=np.int64),
        source_nodes=np.asarray(source_nodes, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        fallback_target_count=fallback_count,
    )


def _nearest_normal_nodes(
    adjacency: list[list[int]],
    node_type: np.ndarray,
    source: int,
    *,
    max_hops: int,
) -> np.ndarray:
    visited = {source}
    frontier = {source}
    for _ in range(max_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier -= visited
        candidates = sorted(
            node for node in next_frontier if node_type[node] == NORMAL_NODE
        )
        if candidates:
            return np.asarray(candidates, dtype=np.int64)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return np.empty(0, dtype=np.int64)


def apply_causal_nodal_boundaries(
    primitive: Any,
    *,
    node_type: Any,
    node_normal: Any,
    wall_normal_coherence: Any | None = None,
    stencil: BoundaryStencil,
    mach: float,
    gamma: float = 1.4,
    rho_inf: float = 1.4,
    p_inf: float = 1.0,
) -> np.ndarray:
    """Apply current-state-only inflow, slip-wall, and outflow point states.

    This is a causal nodal boundary counterfactual for the released graph, not
    an exact replay of Trixi's DG surface flux.  It applies no positivity floor
    and therefore does not hide an inadmissible recurrent interior state.
    """

    state = np.asarray(primitive, dtype=np.float64)
    types = np.asarray(node_type).reshape(-1).astype(np.int64)
    normals = np.asarray(node_normal, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("primitive must have shape (num_nodes, 4)")
    if types.shape != (state.shape[0],) or normals.shape != (state.shape[0], 2):
        raise ValueError("node_type and node_normal must match primitive")
    if wall_normal_coherence is None:
        coherence = np.ones(state.shape[0], dtype=np.float64)
    else:
        coherence = np.asarray(wall_normal_coherence, dtype=np.float64).reshape(-1)
        if coherence.shape != (state.shape[0],):
            raise ValueError("wall_normal_coherence must match primitive")
    if stencil.target_rows.size:
        extrapolated = np.zeros((stencil.target_nodes.size, 4), dtype=np.float64)
        np.add.at(
            extrapolated,
            stencil.target_rows,
            state[stencil.source_nodes] * stencil.weights[:, None],
        )
    else:
        extrapolated = np.empty((0, 4), dtype=np.float64)

    output = state.copy()
    output[stencil.target_nodes] = extrapolated
    wall = types[stencil.target_nodes] == WALL_NODE
    wall_nodes = stencil.target_nodes[wall]
    wall_normal = normals[wall_nodes]
    wall_velocity = output[wall_nodes, 1:3]
    output[wall_nodes, 1:3] = wall_velocity - (
        np.sum(wall_velocity * wall_normal, axis=1, keepdims=True) * wall_normal
    )
    sharp_wall = (types == WALL_NODE) & (coherence < 0.95)
    output[sharp_wall, 1:3] = 0.0
    output[types == INFLOW_NODE] = freestream_primitive(
        mach, gamma=gamma, rho_inf=rho_inf, p_inf=p_inf
    )
    return output


def build_torch_boundary_policy(
    *,
    torch: Any,
    device: Any,
    node_type: Any,
    node_normal: Any,
    wall_normal_coherence: Any,
    stencil: BoundaryStencil,
    mach: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Move one validated causal nodal boundary policy to a Torch device."""

    types = np.asarray(node_type).reshape(-1).astype(np.int64)
    normals = np.asarray(node_normal, dtype=np.float64)
    coherence = np.asarray(wall_normal_coherence, dtype=np.float64).reshape(-1)
    if normals.shape != (types.size, 2) or coherence.shape != (types.size,):
        raise ValueError("boundary geometry must match node_type")
    target_type = types[stencil.target_nodes]
    wall_rows = np.flatnonzero(target_type == WALL_NODE)
    sharp_wall_rows = np.flatnonzero(
        (target_type == WALL_NODE) & (coherence[stencil.target_nodes] < 0.95)
    )
    inflow_nodes = np.flatnonzero(types == INFLOW_NODE)
    stream = freestream_primitive(
        mach,
        gamma=float(config["gamma"]),
        rho_inf=float(config["rho_inf"]),
        p_inf=float(config["p_inf"]),
    )
    return {
        "num_nodes": int(types.size),
        "target_nodes": torch.as_tensor(
            stencil.target_nodes, dtype=torch.long, device=device
        ),
        "target_rows": torch.as_tensor(
            stencil.target_rows, dtype=torch.long, device=device
        ),
        "source_nodes": torch.as_tensor(
            stencil.source_nodes, dtype=torch.long, device=device
        ),
        "weights": torch.as_tensor(stencil.weights, dtype=torch.float32, device=device),
        "wall_rows": torch.as_tensor(wall_rows, dtype=torch.long, device=device),
        "sharp_wall_rows": torch.as_tensor(
            sharp_wall_rows, dtype=torch.long, device=device
        ),
        "target_normals": torch.as_tensor(
            normals[stencil.target_nodes], dtype=torch.float32, device=device
        ),
        "inflow_nodes": torch.as_tensor(inflow_nodes, dtype=torch.long, device=device),
        "freestream": torch.as_tensor(stream, dtype=torch.float32, device=device),
    }


def apply_torch_boundary_policy(
    torch: Any, state: Any, policy: Mapping[str, Any]
) -> Any:
    """Apply one single-graph or concatenated causal nodal policy."""

    if state.ndim != 2 or state.shape != (policy["num_nodes"], 4):
        raise ValueError("state must have shape (num_nodes, 4)")
    target_nodes = policy["target_nodes"]
    extrapolated = state.new_zeros((target_nodes.numel(), 4))
    if policy["source_nodes"].numel():
        extrapolated.index_add_(
            0,
            policy["target_rows"],
            state[policy["source_nodes"]]
            * policy["weights"].to(dtype=state.dtype)[:, None],
        )
    output = state.clone()
    output[target_nodes] = extrapolated
    wall_rows = policy["wall_rows"]
    if wall_rows.numel():
        wall_nodes = target_nodes[wall_rows]
        wall_normal = policy["target_normals"][wall_rows].to(dtype=state.dtype)
        wall_velocity = output[wall_nodes, 1:3]
        output[wall_nodes, 1:3] = wall_velocity - (
            torch.sum(wall_velocity * wall_normal, dim=1, keepdim=True) * wall_normal
        )
    sharp_wall_rows = policy["sharp_wall_rows"]
    if sharp_wall_rows.numel():
        output[target_nodes[sharp_wall_rows], 1:3] = 0.0
    inflow_nodes = policy["inflow_nodes"]
    stream = policy["freestream"].to(dtype=state.dtype)
    if stream.ndim == 1:
        if stream.shape != (4,):
            raise ValueError("freestream must have shape (4,) or (num_inflow, 4)")
    elif stream.shape != (inflow_nodes.numel(), 4):
        raise ValueError("freestream must have shape (4,) or (num_inflow, 4)")
    output[inflow_nodes] = stream
    return output


def _parse_abaqus_header(line: str) -> tuple[str, dict[str, str], set[str]]:
    tokens = [token.strip() for token in line[1:].split(",")]
    section = tokens[0].upper()
    parameters: dict[str, str] = {}
    flags: set[str] = set()
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            parameters[key.strip().upper()] = value.strip().upper()
        elif token:
            flags.add(token.upper())
    return section, parameters, flags


def _parse_set_values(values: list[str], *, generate: bool) -> list[int]:
    integers = [int(value) for value in values]
    if not generate:
        return integers
    if len(integers) not in {2, 3}:
        raise ValueError("GENERATE set row requires start, stop, and optional step")
    start, stop = integers[:2]
    step = integers[2] if len(integers) == 3 else 1
    if step <= 0 or stop < start:
        raise ValueError("invalid GENERATE set range")
    return list(range(start, stop + 1, step))


def _node_index(mesh: AbaqusMesh) -> dict[int, int]:
    return {int(node_id): index for index, node_id in enumerate(mesh.node_ids)}


def _volume_elements(mesh: AbaqusMesh) -> list[AbaqusElement]:
    volume = [element for element in mesh.elements if len(element.node_ids) == 4]
    if not volume:
        raise ValueError("mesh contains no quadrilateral volume elements")
    return volume


def _set_indices(
    node_ids: tuple[int, ...], node_index: Mapping[int, int], name: str
) -> np.ndarray:
    try:
        return np.asarray([node_index[node_id] for node_id in node_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            f"NSET {name} references unknown node {error.args[0]}"
        ) from error


def _boundary_code(name: str) -> int:
    if name in WALL_BOUNDARY_SETS:
        return WALL_NODE
    if name == "RIGHT":
        return OUTFLOW_NODE
    if name == "LEFT":
        return INFLOW_NODE
    raise ValueError(f"unknown physical boundary set {name}")


def _outward_edge_normal(
    points: np.ndarray,
    left: int,
    right: int,
    adjacent: AbaqusElement,
    node_index: Mapping[int, int],
) -> tuple[np.ndarray, float]:
    delta = points[right] - points[left]
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("boundary edge has zero length")
    normal = np.asarray([delta[1], -delta[0]], dtype=np.float64) / length
    cell_indices = [node_index[node_id] for node_id in adjacent.node_ids]
    centroid = np.mean(points[cell_indices], axis=0)
    midpoint = 0.5 * (points[left] + points[right])
    orientation = float(np.dot(midpoint - centroid, normal))
    if orientation < 0.0:
        normal = -normal
        orientation = -orientation
    if orientation <= 1.0e-14:
        raise ValueError("cannot orient boundary normal away from adjacent quad")
    return normal, length


def _normalize_edges(edges: Any, num_nodes: int) -> np.ndarray:
    array = np.asarray(edges)
    if array.ndim != 2:
        raise ValueError("edges must be a two-dimensional array")
    if array.shape[1] != 2 and array.shape[0] == 2:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError("edges must have shape (num_edges, 2)")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("edges must contain integer indices")
    array = array.astype(np.int64, copy=False)
    if array.size and (array.min() < 0 or array.max() >= num_nodes):
        raise ValueError("edge index lies outside the mesh")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("graph contains a self edge")
    return array


def _read_ascii_data_array(element: ET.Element, path: Path) -> np.ndarray:
    if element.attrib.get("format", "ascii").lower() != "ascii":
        raise ValueError(f"{path} uses a non-ASCII DataArray")
    values = np.fromstring(element.text or "", sep=" ")
    if values.size == 0:
        name = element.attrib.get("Name", "unnamed")
        raise ValueError(f"{path} DataArray {name} is empty")
    return values


def _vtk_cell_type(element: AbaqusElement) -> int:
    if len(element.node_ids) == 2:
        return 3
    if len(element.node_ids) == 4:
        return 9
    raise ValueError(
        f"unsupported Abaqus element {element.element_id} with "
        f"{len(element.node_ids)} nodes"
    )


def _julia_number(source: str, pattern: str) -> float:
    match = re.search(pattern, source)
    if match is None:
        raise ValueError(f"Bump.jl does not match required pattern {pattern!r}")
    return float(match.group(1))
