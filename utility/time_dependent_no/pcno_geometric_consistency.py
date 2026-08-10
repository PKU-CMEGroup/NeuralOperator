"""Deterministic bump query-graph and rigid-rotation diagnostics for PCNO.

The coarse object in this module preserves reconstructed proxy mass only.  It
does not represent a coarse PDE solve or a physically conservative finite-
volume restriction.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pcno.geo_utility import compute_edge_gradient_weights
from pcno.pcno import compute_Fourier_bases

BUMP_NODE_TYPE_NAMES = {
    0: "normal",
    1: "wall",
    2: "outflow",
    3: "inflow",
}
ROTATION_90_CCW = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64)


@dataclass(frozen=True)
class BumpQueryGraph:
    """One deterministic proxy-mass query graph derived from a native graph."""

    nodes: np.ndarray
    node_measures: np.ndarray
    node_weights: np.ndarray
    node_rhos: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    node_type: np.ndarray
    fine_to_coarse: np.ndarray
    anchor_indices: np.ndarray
    fine_proxy_mass: np.ndarray
    cluster_proxy_mass: np.ndarray
    radius: int
    gradient_rcond: float

    @property
    def fine_node_count(self) -> int:
        return int(self.fine_to_coarse.size)

    @property
    def coarse_node_count(self) -> int:
        return int(self.nodes.shape[0])

    def restrict(self, values: np.ndarray) -> np.ndarray:
        """Proxy-mass average a field whose last two axes are node, component."""

        array = np.asarray(values)
        if array.ndim < 2 or array.shape[-2] != self.fine_node_count:
            raise ValueError("values must end in [fine_node, component]")
        leading = array.shape[:-2]
        components = int(array.shape[-1])
        flat = np.asarray(array, dtype=np.float64).reshape(
            (-1, self.fine_node_count, components)
        )
        result = np.zeros(
            (flat.shape[0], self.coarse_node_count, components), dtype=np.float64
        )
        for batch_index in range(flat.shape[0]):
            np.add.at(
                result[batch_index],
                self.fine_to_coarse,
                flat[batch_index] * self.fine_proxy_mass[:, None],
            )
        result /= self.cluster_proxy_mass[None, :, None]
        return result.reshape(leading + (self.coarse_node_count, components))

    def prolong(self, values: np.ndarray) -> np.ndarray:
        """Piecewise-constant inject a coarse field to its native clusters."""

        array = np.asarray(values)
        if array.ndim < 2 or array.shape[-2] != self.coarse_node_count:
            raise ValueError("values must end in [coarse_node, component]")
        return np.take(array, self.fine_to_coarse, axis=-2)


def _validated_bump_node_type(node_type: np.ndarray, node_count: int) -> np.ndarray:
    codes = np.asarray(node_type, dtype=np.int64).reshape(-1)
    if codes.shape != (node_count,):
        raise ValueError("node_type must contain one code per node")
    unsupported = sorted({int(value) for value in np.unique(codes)} - set(range(4)))
    if unsupported:
        raise ValueError(f"unsupported bump node types: {unsupported}")
    return codes


def canonical_undirected_edges(
    directed_edges: np.ndarray, node_count: int
) -> np.ndarray:
    """Return sorted unique undirected pairs after strict index validation."""

    edges = np.asarray(directed_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("directed_edges must have shape [E,2]")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= node_count):
        raise ValueError("edge index lies outside the node array")
    pairs = np.sort(edges, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not pairs.size:
        raise ValueError("graph contains no non-self edges")
    return np.unique(pairs, axis=0)


def symmetric_directed_edges(undirected_edges: np.ndarray) -> np.ndarray:
    """Expand undirected pairs into a lexicographically sorted symmetric graph."""

    pairs = np.asarray(undirected_edges, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("undirected_edges must have shape [E,2]")
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return directed[order]


def _adjacency(undirected_edges: np.ndarray, node_count: int) -> list[list[int]]:
    adjacency = [[] for _ in range(node_count)]
    for left, right in np.asarray(undirected_edges, dtype=np.int64):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def _mark_radius(
    start: int,
    *,
    radius: int,
    code: int,
    node_type: np.ndarray,
    adjacency: list[list[int]],
    covered: np.ndarray,
) -> None:
    frontier = [start]
    visited = {start}
    covered[start] = True
    for _ in range(radius):
        following: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if node_type[neighbor] != code or neighbor in visited:
                    continue
                visited.add(neighbor)
                covered[neighbor] = True
                following.append(neighbor)
        frontier = following
        if not frontier:
            break


def _type_stratified_assignment(
    node_type: np.ndarray,
    adjacency: list[list[int]],
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    covered = np.zeros(node_type.size, dtype=bool)
    anchors: list[int] = []
    for node in range(node_type.size):
        if covered[node]:
            continue
        anchors.append(node)
        _mark_radius(
            node,
            radius=radius,
            code=int(node_type[node]),
            node_type=node_type,
            adjacency=adjacency,
            covered=covered,
        )

    best_distance = np.full(node_type.size, np.iinfo(np.int64).max, dtype=np.int64)
    best_anchor = np.full(node_type.size, np.iinfo(np.int64).max, dtype=np.int64)
    queue: list[tuple[int, int, int]] = []
    for anchor in anchors:
        best_distance[anchor] = 0
        best_anchor[anchor] = anchor
        heapq.heappush(queue, (0, anchor, anchor))
    while queue:
        distance, anchor, node = heapq.heappop(queue)
        if distance != best_distance[node] or anchor != best_anchor[node]:
            continue
        for neighbor in adjacency[node]:
            if node_type[neighbor] != node_type[node]:
                continue
            candidate = (distance + 1, anchor)
            incumbent = (int(best_distance[neighbor]), int(best_anchor[neighbor]))
            if candidate < incumbent:
                best_distance[neighbor], best_anchor[neighbor] = candidate
                heapq.heappush(queue, (candidate[0], candidate[1], neighbor))

    if bool((best_anchor == np.iinfo(np.int64).max).any()):
        raise RuntimeError("type-stratified graph assignment left nodes unassigned")
    anchor_indices = np.asarray(anchors, dtype=np.int64)
    coarse_for_anchor = {anchor: index for index, anchor in enumerate(anchors)}
    fine_to_coarse = np.asarray(
        [coarse_for_anchor[int(anchor)] for anchor in best_anchor], dtype=np.int64
    )
    if not np.array_equal(node_type, node_type[anchor_indices][fine_to_coarse]):
        raise RuntimeError("query aggregation mixed family-local node types")
    if int(best_distance.max()) > radius:
        raise RuntimeError("query aggregation exceeded its declared graph radius")
    return anchor_indices, fine_to_coarse


def _validate_connected(edges: np.ndarray, node_count: int) -> None:
    adjacency = _adjacency(canonical_undirected_edges(edges, node_count), node_count)
    reached = np.zeros(node_count, dtype=bool)
    stack = [0]
    reached[0] = True
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if not reached[neighbor]:
                reached[neighbor] = True
                stack.append(neighbor)
    if not bool(reached.all()):
        raise ValueError("query graph is disconnected")


def regenerate_differential_weights(
    nodes: np.ndarray,
    directed_edges: np.ndarray,
    *,
    rcond: float = 1.0e-3,
    require_rank_two: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recompute PCNO least-squares edge weights in the supplied edge order."""

    positions = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(directed_edges, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("nodes must have shape [N,2]")
    canonical_undirected_edges(edges, positions.shape[0])
    if not np.isfinite(positions).all() or not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("nodes and rcond must be finite, with positive rcond")
    if np.unique(edges, axis=0).shape[0] != edges.shape[0]:
        raise ValueError("directed graph contains duplicate edges")

    weights = np.empty((edges.shape[0], 2), dtype=np.float64)
    ranks = np.zeros(positions.shape[0], dtype=np.int64)
    singular_ratios = np.zeros(positions.shape[0], dtype=np.float64)
    sources = edges[:, 0]
    for node in range(positions.shape[0]):
        edge_indices = np.flatnonzero(sources == node)
        if edge_indices.size == 0:
            raise ValueError(f"node {node} has no outgoing differential edges")
        offsets = positions[edges[edge_indices, 1]] - positions[node]
        u, singular, vt = np.linalg.svd(offsets, full_matrices=False)
        cutoff = float(rcond) * float(singular[0])
        retained = singular > cutoff
        retained[2:] = False
        ranks[node] = int(retained.sum())
        singular_ratios[node] = (
            0.0 if singular[0] == 0.0 else float(singular[-1] / singular[0])
        )
        inverse = np.zeros_like(singular)
        inverse[retained] = 1.0 / singular[retained]
        pseudo_inverse = vt.T @ (inverse[:, None] * u.T)
        weights[edge_indices] = pseudo_inverse.T
    if require_rank_two and int(ranks.min()) < 2:
        deficient = np.flatnonzero(ranks < 2)[:10].tolist()
        raise ValueError(f"differential geometry is rank deficient at {deficient}")
    return weights, {
        "minimum_rank": int(ranks.min()),
        "maximum_rank": int(ranks.max()),
        "minimum_singular_ratio": float(singular_ratios.min()),
        "median_singular_ratio": float(np.median(singular_ratios)),
    }


def regenerate_element_differential_geometry(
    nodes: np.ndarray,
    elements: np.ndarray,
    *,
    rcond: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the D041 vertex/element graph and least-squares weights."""

    positions = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elements, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("nodes must have shape [N,2]")
    if elems.ndim != 2 or elems.shape[1] < 4:
        raise ValueError("elements must contain dimensionality and node indices")
    directed_edges, gradient_weights, _ = compute_edge_gradient_weights(
        positions,
        elems,
        mesh_type="vertex_centered",
        adjacent_type="element",
        rcond=float(rcond),
    )
    return (
        np.asarray(directed_edges, dtype=np.int64),
        np.asarray(gradient_weights, dtype=np.float64),
    )


def build_bump_query_graph(
    nodes: np.ndarray,
    directed_edges: np.ndarray,
    node_measures: np.ndarray,
    node_type: np.ndarray,
    *,
    radius: int = 2,
    gradient_rcond: float = 1.0e-3,
) -> BumpQueryGraph:
    """Build the declared deterministic proxy-mass bump query graph."""

    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("nodes must have shape [N,2]")
    if radius < 1:
        raise ValueError("radius must be positive")
    codes = _validated_bump_node_type(node_type, positions.shape[0])
    measures = np.asarray(node_measures, dtype=np.float64)
    if measures.ndim == 2 and measures.shape[1] == 1:
        measures = measures[:, 0]
    if (
        measures.shape != (positions.shape[0],)
        or not np.isfinite(measures).all()
        or bool((measures <= 0.0).any())
    ):
        raise ValueError("native proxy measures must be finite and positive")

    native_pairs = canonical_undirected_edges(directed_edges, positions.shape[0])
    adjacency = _adjacency(native_pairs, positions.shape[0])
    anchors, fine_to_coarse = _type_stratified_assignment(
        codes, adjacency, radius=radius
    )
    coarse_count = anchors.size
    coarse_mass = np.zeros(coarse_count, dtype=np.float64)
    np.add.at(coarse_mass, fine_to_coarse, measures)
    if bool((coarse_mass <= 0.0).any()):
        raise RuntimeError("query graph contains a zero-mass cluster")

    mapped = np.sort(fine_to_coarse[native_pairs], axis=1)
    mapped = mapped[mapped[:, 0] != mapped[:, 1]]
    quotient_pairs = np.unique(mapped, axis=0)
    quotient_edges = symmetric_directed_edges(quotient_pairs)
    _validate_connected(quotient_edges, coarse_count)
    gradient_weights, _ = regenerate_differential_weights(
        positions[anchors],
        quotient_edges,
        rcond=gradient_rcond,
        require_rank_two=True,
    )
    total_mass = float(coarse_mass.sum())
    query = BumpQueryGraph(
        nodes=np.array(positions[anchors], copy=True),
        node_measures=coarse_mass[:, None],
        node_weights=(coarse_mass / total_mass)[:, None],
        node_rhos=np.full((coarse_count, 1), 1.0 / total_mass, dtype=np.float64),
        directed_edges=quotient_edges,
        edge_gradient_weights=gradient_weights,
        node_type=np.array(codes[anchors], copy=True),
        fine_to_coarse=fine_to_coarse,
        anchor_indices=anchors,
        fine_proxy_mass=np.array(measures, copy=True),
        cluster_proxy_mass=coarse_mass,
        radius=int(radius),
        gradient_rcond=float(gradient_rcond),
    )
    return query


def rotation_center(nodes: np.ndarray) -> np.ndarray:
    """Return the outcome-independent bounding-box midpoint of one geometry."""

    positions = np.asarray(nodes, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("nodes must have shape [N,2]")
    return 0.5 * (positions.min(axis=0) + positions.max(axis=0))


def rotate_points(
    points: np.ndarray,
    center: np.ndarray,
    *,
    matrix: np.ndarray = ROTATION_90_CCW,
    inverse: bool = False,
) -> np.ndarray:
    """Apply or invert ``x'=c+Q(x-c)`` using row-vector storage."""

    values = np.asarray(points, dtype=np.float64)
    pivot = np.asarray(center, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.shape[-1] != 2 or pivot.shape != (2,) or transform.shape != (2, 2):
        raise ValueError("rotation expects [...,2] points, a 2-vector, and 2x2 Q")
    active = transform.T if inverse else transform
    return pivot + (values - pivot) @ active.T


def rotate_euler_field(
    field: np.ndarray,
    *,
    matrix: np.ndarray = ROTATION_90_CCW,
    inverse: bool = False,
) -> np.ndarray:
    """Rotate momentum in a conservative state or residual; keep rho and E."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("Euler field must end in four conservative components")
    transform = np.asarray(matrix, dtype=np.float64)
    active = transform.T if inverse else transform
    result = np.array(values, copy=True)
    result[..., 1:3] = values[..., 1:3] @ active.T
    return result


def transformed_phase_origin(
    center: np.ndarray, *, matrix: np.ndarray = ROTATION_90_CCW
) -> np.ndarray:
    """Return the image of the original coordinate origin under rotation."""

    pivot = np.asarray(center, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    return pivot - pivot @ transform.T


def rotate_fourier_modes(
    modes: np.ndarray, *, matrix: np.ndarray = ROTATION_90_CCW
) -> np.ndarray:
    """Transport physical wavevectors while preserving learned mode ordering."""

    values = np.asarray(modes, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("modes must have shape [K,2,M]")
    return np.einsum("ij,kjw->kiw", transform, values)


def transported_fourier_tensors(
    rotated_nodes: torch.Tensor,
    node_weights: torch.Tensor,
    modes: torch.Tensor,
    *,
    center: np.ndarray,
    matrix: np.ndarray = ROTATION_90_CCW,
) -> tuple[torch.Tensor, ...]:
    """Build basis tensors for transported modes and the transformed origin."""

    transform = torch.as_tensor(
        matrix, dtype=rotated_nodes.dtype, device=rotated_nodes.device
    )
    pivot = torch.as_tensor(
        center, dtype=rotated_nodes.dtype, device=rotated_nodes.device
    )
    origin = pivot - torch.mv(transform, pivot)
    phase_nodes = rotated_nodes - origin.reshape(1, 1, 2)
    rotated_modes = torch.einsum("ij,kjw->kiw", transform.to(dtype=modes.dtype), modes)
    bases = compute_Fourier_bases(phase_nodes, rotated_modes)
    weighted = tuple(
        torch.einsum("bxkw,bxw->bxkw", basis, node_weights) for basis in bases
    )
    return (*bases, *weighted)


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    """Return a float64 relative L2 with an explicit tiny denominator floor."""

    numerator = float(np.linalg.norm(np.asarray(value, dtype=np.float64).ravel()))
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=np.float64).ravel()))
    return numerator / max(denominator, 1.0e-30)
