"""PCNO adapter helpers for CPG-style Euler graph frames.

These utilities intentionally stop at tensor-shape/data-contract conversion.
They do not decide final model features or claim physical conservation weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ArrayLike = Any


@dataclass(frozen=True)
class PcnoFrameBatch:
    """Batched NumPy arrays following the existing PCNO training convention."""

    x: np.ndarray
    y: np.ndarray
    node_mask: np.ndarray
    nodes: np.ndarray
    node_weights: np.ndarray
    directed_edges: np.ndarray
    edge_gradient_weights: np.ndarray
    metadata: dict[str, Any]

    @property
    def aux(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.node_mask,
            self.nodes,
            self.node_weights,
            self.directed_edges,
            self.edge_gradient_weights,
        )


def make_equal_node_weights(
    num_nodes: int,
    *,
    nmeasures: int = 1,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return normalized equal node weights with shape ``(N, nmeasures)``."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if nmeasures <= 0:
        raise ValueError("nmeasures must be positive")
    value = 1.0 / float(num_nodes * nmeasures)
    return np.full((num_nodes, nmeasures), value, dtype=dtype)


def normalize_edge_array(edges: ArrayLike, *, num_nodes: int) -> np.ndarray:
    """Return valid integer graph edges as ``(E, 2)``."""

    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.ndim != 2:
        raise ValueError("edges must have shape (E, 2) or (2, E)")
    if edge_array.shape[1] != 2 and edge_array.shape[0] == 2:
        edge_array = edge_array.T
    if edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2) or (2, E)")
    valid = (
        (edge_array[:, 0] >= 0)
        & (edge_array[:, 1] >= 0)
        & (edge_array[:, 0] < num_nodes)
        & (edge_array[:, 1] < num_nodes)
        & (edge_array[:, 0] != edge_array[:, 1])
    )
    return edge_array[valid]


def compute_graph_edge_gradient_weights(
    positions: ArrayLike,
    edges: ArrayLike,
    *,
    make_undirected: bool = True,
    rcond: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build PCNO directed edges and least-squares gradient weights.

    The returned ``directed_edges`` follow the PCNO convention
    ``(target_node, source_neighbor)``. For a scalar field ``f``, the gradient
    estimate at a target node is ``sum_e w_e * (f[source] - f[target])``.
    """

    nodes = np.asarray(positions, dtype=np.float64)
    if nodes.ndim != 2:
        raise ValueError("positions must have shape (num_nodes, ndim)")
    num_nodes, ndim = nodes.shape
    if num_nodes == 0 or ndim == 0:
        raise ValueError("positions must be nonempty")
    if rcond < 0.0:
        raise ValueError("rcond must be nonnegative")

    edge_array = normalize_edge_array(edges, num_nodes=num_nodes)
    adjacency = [set() for _ in range(num_nodes)]
    for a, b in edge_array:
        adjacency[int(a)].add(int(b))
        if make_undirected:
            adjacency[int(b)].add(int(a))

    directed_edges: list[list[int]] = []
    weights: list[np.ndarray] = []
    for target, neighbors_set in enumerate(adjacency):
        neighbors = sorted(neighbors_set)
        if not neighbors:
            continue
        dx = nodes[neighbors] - nodes[target]
        edge_weights = _least_squares_gradient_weights(
            dx,
            rrank=min(ndim, len(neighbors)),
            rcond=rcond,
        )
        for source, weight in zip(neighbors, edge_weights, strict=True):
            directed_edges.append([target, source])
            weights.append(weight)

    if not directed_edges:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, ndim), dtype=np.float64),
        )
    return np.asarray(directed_edges, dtype=np.int64), np.asarray(weights)


def make_pcno_frame_batch(
    frame: dict[str, np.ndarray],
    *,
    include_positions_in_x: bool = True,
    make_undirected: bool = True,
    rcond: float = 1.0e-3,
) -> PcnoFrameBatch:
    """Convert one CPG graph frame to a batch of PCNO-compatible arrays."""

    positions = np.asarray(frame["pos"], dtype=np.float32)
    features = np.asarray(frame["x"], dtype=np.float32)
    target = np.asarray(frame["y"], dtype=np.float32)
    if positions.ndim != 2:
        raise ValueError("frame['pos'] must have shape (N, ndim)")
    if features.ndim != 2 or target.ndim != 2:
        raise ValueError("frame['x'] and frame['y'] must be node-feature matrices")
    if positions.shape[0] != features.shape[0] or target.shape[0] != features.shape[0]:
        raise ValueError("frame arrays disagree on node count")

    x_features = (
        np.concatenate((positions, features), axis=-1)
        if include_positions_in_x
        else features
    )
    directed_edges, edge_gradient_weights = compute_graph_edge_gradient_weights(
        positions,
        frame["edges"],
        make_undirected=make_undirected,
        rcond=rcond,
    )
    num_nodes = positions.shape[0]
    node_weights = make_equal_node_weights(num_nodes)

    metadata = {
        "num_nodes": int(num_nodes),
        "num_edges_input": int(normalize_edge_array(frame["edges"], num_nodes=num_nodes).shape[0]),
        "num_directed_edges": int(directed_edges.shape[0]),
        "ndim": int(positions.shape[1]),
        "x_includes_positions": bool(include_positions_in_x),
        "node_weight_policy": "equal_normalized",
        "edge_gradient_policy": "least_squares_from_graph_edges",
        "make_undirected": bool(make_undirected),
        "rcond": float(rcond),
    }
    return PcnoFrameBatch(
        x=x_features[None, ...],
        y=target[None, ...],
        node_mask=np.ones((1, num_nodes, 1), dtype=np.float32),
        nodes=positions[None, ...],
        node_weights=node_weights[None, ...],
        directed_edges=directed_edges[None, ...],
        edge_gradient_weights=edge_gradient_weights.astype(np.float32)[None, ...],
        metadata=metadata,
    )


def make_pcno_euler7_frame_batch(
    frame: dict[str, np.ndarray],
    *,
    current_primitives: ArrayLike | None = None,
    node_rho_policy: str = "ones",
    make_undirected: bool = True,
    rcond: float = 1.0e-3,
) -> PcnoFrameBatch:
    """Convert a CPG frame to the original Euler PCNO 7-channel layout.

    The collaborator checkpoint follows ``scripts/2d_Euler_eq/forward_train.py``:
    ``x = [x, y, node_rho, rho, v1, v2, pres]``.  CPG HDF5 files do not expose
    validated cell measures, so ``node_rho_policy='ones'`` is the conservative
    default for the integration-density feature while equal normalized node
    weights are still used in the PCNO auxiliary measure.
    """

    positions = np.asarray(frame["pos"], dtype=np.float32)
    target = np.asarray(frame["y"], dtype=np.float32)
    if current_primitives is None:
        current = np.asarray(frame["current_primitives"], dtype=np.float32)
    else:
        current = np.asarray(current_primitives, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("frame['pos'] must have shape (N, 2)")
    if current.shape != target.shape or current.ndim != 2 or current.shape[1] != 4:
        raise ValueError("current_primitives and frame['y'] must have shape (N, 4)")
    if positions.shape[0] != current.shape[0]:
        raise ValueError("positions and primitives disagree on node count")

    num_nodes = positions.shape[0]
    node_weights = make_equal_node_weights(num_nodes)
    if node_rho_policy == "ones":
        node_rho = np.ones((num_nodes, 1), dtype=np.float32)
    elif node_rho_policy == "node_weights":
        node_rho = node_weights.astype(np.float32)
    else:
        raise ValueError("node_rho_policy must be 'ones' or 'node_weights'")
    directed_edges, edge_gradient_weights = compute_graph_edge_gradient_weights(
        positions,
        frame["edges"],
        make_undirected=make_undirected,
        rcond=rcond,
    )
    x_features = np.concatenate((positions, node_rho, current), axis=-1)
    metadata = {
        "num_nodes": int(num_nodes),
        "num_edges_input": int(normalize_edge_array(frame["edges"], num_nodes=num_nodes).shape[0]),
        "num_directed_edges": int(directed_edges.shape[0]),
        "ndim": 2,
        "x_layout": "pos2_node_rho1_primitives4",
        "node_rho_policy": node_rho_policy,
        "node_weight_policy": "equal_normalized",
        "edge_gradient_policy": "least_squares_from_graph_edges",
        "make_undirected": bool(make_undirected),
        "rcond": float(rcond),
    }
    return PcnoFrameBatch(
        x=x_features[None, ...],
        y=target[None, ...],
        node_mask=np.ones((1, num_nodes, 1), dtype=np.float32),
        nodes=positions[None, ...],
        node_weights=node_weights[None, ...],
        directed_edges=directed_edges[None, ...],
        edge_gradient_weights=edge_gradient_weights.astype(np.float32)[None, ...],
        metadata=metadata,
    )

def _least_squares_gradient_weights(
    dx: np.ndarray,
    *,
    rrank: int,
    rcond: float,
) -> np.ndarray:
    """Return per-neighbor gradient weights from the pseudo-inverse of ``dx``."""

    if dx.ndim != 2:
        raise ValueError("dx must be a matrix")
    num_neighbors, ndim = dx.shape
    if num_neighbors == 0:
        return np.empty((0, ndim), dtype=np.float64)
    u, singular_values, vt = np.linalg.svd(dx, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.zeros((num_neighbors, ndim), dtype=np.float64)
    keep = singular_values > (rcond * singular_values[0])
    keep[max(rrank, 0) :] = False
    inv = np.divide(
        1.0,
        singular_values,
        where=keep,
        out=np.zeros_like(singular_values),
    )
    pinv = vt.T @ (inv[:, None] * u.T)
    return pinv.T
