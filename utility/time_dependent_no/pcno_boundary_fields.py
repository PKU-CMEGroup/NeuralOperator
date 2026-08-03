"""Resolution-independent bounded boundary fields for PCNO inputs.

The fields in this module are volume descriptors.  For a semantic boundary
subset ``Gamma_k`` and a fixed physical width ``ell``, the corresponding
channel is

    B_k^ell(x) = rho(distance(x, Gamma_k) / ell),

where ``rho`` is a compactly supported, bounded cubic collar.  These channels
are deliberately not scaled as diffuse surface measures and must not be used
as quadrature weights for boundary integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

BOUNDARY_FIELD_CONTRACT_SCHEMA = "pcno_bounded_semantic_collar_v1"
COMPACT_CUBIC_KERNEL = "compact_cubic_c1_v1"


@dataclass(frozen=True)
class BoundaryFieldData:
    """One static boundary-feature array and its complete data contract."""

    values: np.ndarray
    contract: dict[str, Any]


def compact_cubic_collar(
    distance: np.ndarray | Sequence[float], physical_width: float
) -> np.ndarray:
    """Evaluate a C1 compact collar with support of fixed physical width.

    The kernel is ``1 - 3 r**2 + 2 r**3`` for ``0 <= r < 1`` and zero for
    ``r >= 1``.  It has value one and zero derivative at the boundary, and
    value and derivative zero at the edge of the collar.
    """

    width = float(physical_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("physical_width must be positive and finite")
    values = np.asarray(distance, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("distance must be nonnegative and finite")
    ratio = values / width
    collar = np.zeros_like(ratio)
    supported = ratio < 1.0
    r = ratio[supported]
    collar[supported] = 1.0 - 3.0 * np.square(r) + 2.0 * np.power(r, 3)
    return collar


def point_to_segment_distance(
    points: np.ndarray,
    segments: np.ndarray,
    *,
    point_chunk_size: int = 4096,
) -> np.ndarray:
    """Return the minimum Euclidean distance from each point to line segments."""

    query = np.asarray(points, dtype=np.float64)
    lines = np.asarray(segments, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if lines.ndim != 3 or lines.shape[1:] != (2, 2) or lines.shape[0] == 0:
        raise ValueError("segments must have shape [M, 2, 2] with M > 0")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(lines)):
        raise ValueError("points and segments must be finite")
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive")
    starts = lines[:, 0]
    directions = lines[:, 1] - starts
    squared_lengths = np.einsum("md,md->m", directions, directions)
    if np.any(squared_lengths <= 0.0):
        raise ValueError("boundary segments must have positive length")

    result = np.empty(query.shape[0], dtype=np.float64)
    for first in range(0, query.shape[0], point_chunk_size):
        chunk = query[first : first + point_chunk_size]
        offsets = chunk[:, None, :] - starts[None, :, :]
        parameter = np.einsum("nmd,md->nm", offsets, directions)
        parameter = np.clip(parameter / squared_lengths[None, :], 0.0, 1.0)
        closest = starts[None, :, :] + parameter[..., None] * directions[None, :, :]
        squared_distance = np.sum(np.square(chunk[:, None, :] - closest), axis=-1)
        result[first : first + chunk.shape[0]] = np.sqrt(
            np.min(squared_distance, axis=1)
        )
    return result


def semantic_collar_fields(
    points: np.ndarray,
    segments_by_semantic: Mapping[str, np.ndarray],
    *,
    physical_width: float,
    family: str,
    boundary_geometry_provenance: str,
    corner_policy: str,
    semantic_factorization: str,
) -> BoundaryFieldData:
    """Build one bounded physical collar for each named boundary subset."""

    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    names = tuple(str(name) for name in segments_by_semantic)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("semantic boundary names must be nonempty and unique")
    if "normal" in names or "interior" in names:
        raise ValueError("interior/normal is represented by absence, not a collar")

    distances = []
    for name in names:
        segments = np.asarray(segments_by_semantic[name], dtype=np.float64)
        distances.append(point_to_segment_distance(query, segments))
    distance_matrix = np.stack(distances, axis=-1)
    values = compact_cubic_collar(distance_matrix, physical_width).astype(
        np.float32, copy=False
    )
    contract = {
        "schema": BOUNDARY_FIELD_CONTRACT_SCHEMA,
        "continuum_object": "bounded_volume_descriptor",
        "representation": "semantic_physical_collar",
        "family": str(family),
        "channel_names": list(names),
        "physical_width": float(physical_width),
        "width_units": "source_coordinate_units",
        "kernel": COMPACT_CUBIC_KERNEL,
        "support_radius_in_widths": 1.0,
        "value_range": [0.0, 1.0],
        "surface_measure_scaling": False,
        "physical_width_fixed_under_resolution_change": True,
        "pointwise_amplitude_fixed_under_resolution_change": True,
        "rotation_behavior": "scalar_rigid_motion_invariant",
        "boundary_geometry_provenance": str(boundary_geometry_provenance),
        "corner_policy": str(corner_policy),
        "semantic_factorization": str(semantic_factorization),
        "physical_boundary_policy_changed": False,
    }
    return BoundaryFieldData(values=values, contract=contract)


def factorized_rectangle_boundary_fields(
    points: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    physical_width: float,
    family: str = "dynamic_shock_vortex_fv",
) -> BoundaryFieldData:
    """Build overlapping y-symmetry and x-extrapolation collars on a rectangle."""

    bounds = np.asarray([x_min, x_max, y_min, y_max], dtype=np.float64)
    if not np.all(np.isfinite(bounds)) or not x_min < x_max or not y_min < y_max:
        raise ValueError("rectangle bounds must be finite and strictly ordered")
    x_segments = np.asarray(
        [
            [[x_min, y_min], [x_min, y_max]],
            [[x_max, y_min], [x_max, y_max]],
        ],
        dtype=np.float64,
    )
    y_segments = np.asarray(
        [
            [[x_min, y_min], [x_max, y_min]],
            [[x_min, y_max], [x_max, y_max]],
        ],
        dtype=np.float64,
    )
    result = semantic_collar_fields(
        points,
        {
            "y_symmetry": y_segments,
            "x_extrapolation": x_segments,
        },
        physical_width=physical_width,
        family=family,
        boundary_geometry_provenance="validated_axis_aligned_domain_bounds",
        corner_policy="overlap_of_independent_x_and_y_descriptors",
        semantic_factorization=(
            "two_overlapping_physical_descriptors_not_four_exclusive_node_codes"
        ),
    )
    result.contract["boundary_geometry_exact_for_declared_rectangle"] = True
    result.contract["domain_bounds"] = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }
    return result


def recover_tagged_boundary_cycle(
    edges: np.ndarray,
    node_type: np.ndarray,
) -> np.ndarray:
    """Recover the simple non-normal cycle without inferring boundary normals."""

    raw_edges = np.asarray(edges, dtype=np.int64)
    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError("edges must have shape [M, 2]")
    if np.any(raw_edges < 0) or np.any(raw_edges >= types.size):
        raise ValueError("edge index is outside the node-type array")
    if np.any(raw_edges[:, 0] == raw_edges[:, 1]):
        raise ValueError("graph edges must not contain self loops")
    canonical = np.sort(raw_edges, axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError("graph repeats an undirected edge")
    canonical = canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]
    boundary = canonical[(types[canonical[:, 0]] != 0) & (types[canonical[:, 1]] != 0)]
    boundary_nodes = np.flatnonzero(types != 0)
    if boundary_nodes.size < 3 or boundary.shape[0] != boundary_nodes.size:
        raise ValueError("non-normal induced graph is not a simple boundary cycle")
    adjacency = {int(node): [] for node in boundary_nodes}
    for left, right in boundary:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("non-normal induced graph is not a simple boundary cycle")
    visited = {int(boundary_nodes[0])}
    frontier = list(visited)
    while frontier:
        for neighbor in adjacency[frontier.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if len(visited) != boundary_nodes.size:
        raise ValueError("non-normal induced graph has multiple boundary components")
    return boundary


def tagged_polyline_boundary_fields(
    points: np.ndarray,
    boundary_edges: np.ndarray,
    node_type: np.ndarray,
    *,
    semantic_codes: Mapping[str, int],
    physical_width: float,
    family: str = "supersonic_bump_graph",
) -> BoundaryFieldData:
    """Build semantic collars from a tagged graph-boundary polyline.

    An edge whose endpoints have two different semantic codes is included in
    both semantic subsets.  This makes the finite-resolution corner overlap
    explicit instead of imposing a wall/inflow/outflow precedence.  The
    duplicated tangential length is a graph-boundary proxy error that should
    shrink with boundary-edge length under a genuine mesh refinement.
    """

    query = np.asarray(points, dtype=np.float64)
    edges = np.asarray(boundary_edges, dtype=np.int64)
    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        raise ValueError("boundary_edges must have shape [M, 2] with M > 0")
    if types.shape != (query.shape[0],):
        raise ValueError("node_type must have one code per point")
    if np.any(edges < 0) or np.any(edges >= query.shape[0]):
        raise ValueError("boundary edge index is outside the point array")
    if np.any(types[edges] == 0):
        raise ValueError("graph boundary edges must not contain normal nodes")
    codes = {str(name): int(code) for name, code in semantic_codes.items()}
    if not codes or len(set(codes.values())) != len(codes):
        raise ValueError("semantic codes must be nonempty and one-to-one")
    unknown = sorted(set(np.unique(types[edges])) - set(codes.values()))
    if unknown:
        raise ValueError(f"boundary edges contain undeclared semantic codes: {unknown}")

    edge_types = types[edges]
    segments_by_semantic: dict[str, np.ndarray] = {}
    for name, code in codes.items():
        selected = np.any(edge_types == code, axis=1)
        if not np.any(selected):
            raise ValueError(f"semantic boundary {name!r} has no boundary segment")
        segments_by_semantic[name] = query[edges[selected]]
    result = semantic_collar_fields(
        query,
        segments_by_semantic,
        physical_width=physical_width,
        family=family,
        boundary_geometry_provenance="released_graph_boundary_cycle_polyline_proxy",
        corner_policy="mixed_endpoint_edges_belong_to_both_semantic_subsets",
        semantic_factorization="wall_inflow_outflow_independent_channels",
    )
    result.contract["semantic_source_codes"] = codes
    result.contract["boundary_geometry_is_mesh_derived_proxy"] = True
    result.contract["resolution_transfer_claim_requires_common_geometry_provenance"] = (
        True
    )
    return result
