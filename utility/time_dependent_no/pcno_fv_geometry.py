"""Map validated finite-volume cells and faces to PCNO graph geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCNOFiniteVolumeGeometry:
    """PCNO tensors plus an explicit physical mesh-to-graph map."""

    nodes: np.ndarray
    edges: np.ndarray
    node_type: np.ndarray
    node_measures: np.ndarray
    node_weights: np.ndarray
    node_rhos: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    mesh_cell_to_graph_node: np.ndarray
    face_to_directed_edge: np.ndarray
    minimum_stencil_singular_value: float
    maximum_stencil_condition_number: float
    maximum_coordinate_gradient_error: float


def _validated_array(
    value: np.ndarray,
    *,
    name: str,
    ndim: int,
    last_dim: int | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if last_dim is not None and array.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have final dimension {last_dim}, got {array.shape}"
        )
    return array


def build_pcno_finite_volume_geometry(
    *,
    cell_centers: np.ndarray,
    cell_volume: np.ndarray,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    face_boundary_tag: np.ndarray,
    boundary_tag_names: tuple[str, ...],
    gradient_rcond: float = 1.0e-12,
) -> PCNOFiniteVolumeGeometry:
    """Build deterministic PCNO geometry from one oriented FV mesh.

    Node codes are local to this dynamic benchmark: ``0`` is an interior cell,
    ``1`` touches a y-symmetry face, ``2`` touches an x-extrapolation face, and
    ``3`` touches both.  Interior physical faces become exactly two directed
    PCNO edges.  Gradient weights are the deterministic least-squares
    pseudoinverse on those face-neighbor stencils.
    """

    if not np.isfinite(gradient_rcond) or gradient_rcond <= 0.0:
        raise ValueError("gradient_rcond must be positive and finite")
    nodes = _validated_array(
        cell_centers, name="cell_centers", ndim=2, last_dim=2
    ).astype(np.float64, copy=False)
    volumes = _validated_array(cell_volume, name="cell_volume", ndim=1).astype(
        np.float64, copy=False
    )
    owner = _validated_array(face_owner, name="face_owner", ndim=1).astype(
        np.int64, copy=False
    )
    neighbor = _validated_array(face_neighbor, name="face_neighbor", ndim=1).astype(
        np.int64, copy=False
    )
    boundary_tag = _validated_array(
        face_boundary_tag, name="face_boundary_tag", ndim=1
    ).astype(np.int64, copy=False)
    num_nodes = nodes.shape[0]
    num_faces = owner.size
    if volumes.shape != (num_nodes,):
        raise ValueError("cell_volume must have one value per cell center")
    if neighbor.shape != (num_faces,) or boundary_tag.shape != (num_faces,):
        raise ValueError("face arrays must have identical leading dimensions")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(volumes)):
        raise ValueError("cell centers and volumes must be finite")
    if np.any(volumes <= 0.0):
        raise ValueError("cell volumes must be positive")
    if np.any(owner < 0) or np.any(owner >= num_nodes):
        raise ValueError("face owner index lies outside the cell axis")
    if np.any(neighbor < -1):
        raise ValueError("boundary face neighbors must use the -1 sentinel")
    interior = neighbor >= 0
    if np.any(neighbor[interior] >= num_nodes):
        raise ValueError("face neighbor index lies outside the cell axis")
    if np.any(owner[interior] == neighbor[interior]):
        raise ValueError("interior face cannot connect a cell to itself")
    required_boundary_names = {"interior", "x_min", "x_max", "y_min", "y_max"}
    if (
        len(boundary_tag_names) != len(required_boundary_names)
        or set(boundary_tag_names) != required_boundary_names
        or boundary_tag_names[0] != "interior"
    ):
        raise ValueError(
            "boundary_tag_names must be interior, x_min, x_max, y_min, and y_max"
        )
    if np.any(boundary_tag < 0) or np.any(boundary_tag >= len(boundary_tag_names)):
        raise ValueError("face boundary tag lies outside boundary_tag_names")
    if np.any(boundary_tag[interior] != 0) or np.any(boundary_tag[~interior] == 0):
        raise ValueError("interior connectivity and boundary tags disagree")

    raw_edges = np.stack((owner[interior], neighbor[interior]), axis=-1)
    canonical_edges = np.sort(raw_edges, axis=1)
    if np.unique(canonical_edges, axis=0).shape[0] != canonical_edges.shape[0]:
        raise ValueError("multiple physical faces connect the same cell pair")
    edge_order = np.lexsort((canonical_edges[:, 1], canonical_edges[:, 0]))
    edges = canonical_edges[edge_order]

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for first, second in edges.tolist():
        adjacency[first].append(second)
        adjacency[second].append(first)
    if any(not neighbors for neighbors in adjacency):
        raise ValueError("every FV cell must have at least one interior-face neighbor")

    directed_rows: list[tuple[int, int]] = []
    gradient_rows: list[np.ndarray] = []
    minimum_singular = np.inf
    maximum_condition = 0.0
    maximum_gradient_error = 0.0
    for target, unsorted_neighbors in enumerate(adjacency):
        neighbors = np.asarray(sorted(unsorted_neighbors), dtype=np.int64)
        delta = nodes[neighbors] - nodes[target]
        singular_values = np.linalg.svd(delta, compute_uv=False)
        if singular_values.size != 2 or singular_values[-1] <= 0.0:
            raise ValueError(f"cell {target} has a rank-deficient face stencil")
        minimum_singular = min(minimum_singular, float(singular_values[-1]))
        maximum_condition = max(
            maximum_condition,
            float(singular_values[0] / singular_values[-1]),
        )
        weights = np.linalg.pinv(delta, rcond=float(gradient_rcond)).T
        coordinate_gradient = weights.T @ delta
        maximum_gradient_error = max(
            maximum_gradient_error,
            float(np.max(np.abs(coordinate_gradient - np.eye(2)))),
        )
        directed_rows.extend((target, int(source)) for source in neighbors)
        gradient_rows.extend(weights)
    if maximum_gradient_error > 1.0e-10:
        raise ValueError(
            "gradient pseudoinverse does not reproduce coordinate gradients: "
            f"maximum error {maximum_gradient_error:.3e}"
        )

    directed_edges = np.asarray(directed_rows, dtype=np.int64)
    edge_gradient_weights = np.asarray(gradient_rows, dtype=np.float64)
    directed_lookup = {
        (int(target), int(source)): index
        for index, (target, source) in enumerate(directed_edges.tolist())
    }
    face_to_directed_edge = np.full((num_faces, 2), -1, dtype=np.int64)
    for face_index in np.flatnonzero(interior):
        first = int(owner[face_index])
        second = int(neighbor[face_index])
        face_to_directed_edge[face_index] = (
            directed_lookup[(first, second)],
            directed_lookup[(second, first)],
        )
    if np.any(face_to_directed_edge[interior] < 0):
        raise AssertionError("interior face is missing a directed graph edge")

    x_boundary_tags = {
        index
        for index, name in enumerate(boundary_tag_names)
        if name in {"x_min", "x_max"}
    }
    y_boundary_tags = {
        index
        for index, name in enumerate(boundary_tag_names)
        if name in {"y_min", "y_max"}
    }
    if len(x_boundary_tags) != 2 or len(y_boundary_tags) != 2:
        raise ValueError("boundary_tag_names do not define both x and y boundaries")
    touches_x = np.zeros(num_nodes, dtype=bool)
    touches_y = np.zeros(num_nodes, dtype=bool)
    boundary_faces = np.flatnonzero(~interior)
    for face_index in boundary_faces:
        tag = int(boundary_tag[face_index])
        cell = int(owner[face_index])
        touches_x[cell] |= tag in x_boundary_tags
        touches_y[cell] |= tag in y_boundary_tags
    node_type = 2 * touches_x.astype(np.int64) + touches_y.astype(np.int64)

    node_measures = volumes[:, None]
    total_volume = float(np.sum(volumes))
    node_weights = node_measures / total_volume
    node_rhos = node_weights / node_measures
    if not np.isclose(node_weights.sum(), 1.0, rtol=0.0, atol=1.0e-14):
        raise AssertionError(
            "normalized physical cell-volume weights do not sum to one"
        )

    return PCNOFiniteVolumeGeometry(
        nodes=nodes,
        edges=edges,
        node_type=node_type,
        node_measures=node_measures,
        node_weights=node_weights,
        node_rhos=node_rhos,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
        mesh_cell_to_graph_node=np.arange(num_nodes, dtype=np.int64),
        face_to_directed_edge=face_to_directed_edge,
        minimum_stencil_singular_value=float(minimum_singular),
        maximum_stencil_condition_number=float(maximum_condition),
        maximum_coordinate_gradient_error=float(maximum_gradient_error),
    )
