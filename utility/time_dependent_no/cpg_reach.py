"""Graph-dependency and characteristic-reach helpers for released CPGNet.

The pinned CPGNet release uses a symmetric directed message graph.  Current
state information is local before its flow processor, expands by one graph hop
in each processor layer, is decoded on both orientations of each unique edge,
and is finally aggregated at the target node.  Consequently, a model with L
flow-processor layers has an exact architectural state-support set equal to the
undirected (L + 1)-hop ball around a target output node.

This module compares that code-traced support with Euler characteristic travel
on the recovered dataset graph.  It deliberately does not identify graph nodes
as control volumes or graph edges as physical faces: the public HDF5 contract
does not contain the volumes, measures, or solver-to-graph map needed for that
claim.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
from typing import Any, Sequence

import numpy as np

from utility.time_dependent_no.cpg_release import (
    CPG_REFERENCE_COMMIT,
    CPG_REFERENCE_RUNTIME_SHA256,
)


ArrayLike = Any
DEFAULT_GAMMA = 1.4
REACH_SCHEMA = "cpg_characteristic_reach_v1"

_DEPENDENCY_SOURCE_FILES = (
    "modelEdgeUpd/simulator.py",
    "modelEdgeUpd/modelEU.py",
    "modelEdgeUpd/convFlow.py",
    "modelEdgeUpd/convReconstruct.py",
    "modelEdgeUpd/conserveUpd.py",
    "utils/to_undirected.py",
)


@dataclass(frozen=True)
class CPGDependencyTrace:
    """Pinned code-trace contract for current-state dependency support."""

    message_passing_layers: int
    final_output_radius: int
    reference_commit: str = CPG_REFERENCE_COMMIT

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "support_kind": "exact_architectural_current_state_support",
                "support_set": "undirected graph ball around target output node",
                "source_sha256": {
                    path: CPG_REFERENCE_RUNTIME_SHA256[path]
                    for path in _DEPENDENCY_SOURCE_FILES
                },
                "trace": [
                    "solution encoder is node-local",
                    "four-layer edge encoder receives geometry only",
                    "each flow EdgeConv expands node state support by one hop",
                    "directed reconstruction uses target-node and edge latents",
                    "one Rusanov flux pairs both orientations of each unique edge",
                    "target-node aggregation adds one endpoint hop",
                ],
                "active_derivative_caveat": (
                    "trained weights and inactive nonlinearities may make the "
                    "state-specific active derivative support a strict subset"
                ),
            }
        )
        return payload


def cpg_dependency_trace(message_passing_layers: int = 12) -> CPGDependencyTrace:
    """Return the pinned structural trace for a release-style CPG model."""

    if isinstance(message_passing_layers, bool) or message_passing_layers < 0:
        raise ValueError("message_passing_layers must be a nonnegative integer")
    if int(message_passing_layers) != message_passing_layers:
        raise ValueError("message_passing_layers must be an integer")
    layers = int(message_passing_layers)
    return CPGDependencyTrace(layers, layers + 1)


def normalize_edges(edges: ArrayLike, num_nodes: int) -> np.ndarray:
    """Return an integer edge array with shape (E, 2), rejecting bad indices."""

    array = np.asarray(edges)
    if array.ndim != 2:
        raise ValueError(f"edges must be two-dimensional, got {array.shape}")
    if array.shape[1] != 2 and array.shape[0] == 2:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("edges must contain integer indices")
    array = array.astype(np.int64, copy=False)
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if array.size and (array.min() < 0 or array.max() >= num_nodes):
        raise ValueError("edges contain node indices outside the graph")
    return array


def normalize_scale_positions(pos: ArrayLike) -> np.ndarray:
    """Reproduce torch_geometric.transforms.NormalizeScale for 2D positions."""

    points = _validate_positions(pos)
    centered = points - np.mean(points, axis=0, keepdims=True)
    maximum = float(np.max(np.abs(centered)))
    if maximum == 0.0:
        return centered
    return centered * (0.999999 / maximum)


def validate_release_graph_mapping(
    *,
    raw_pos: ArrayLike,
    raw_edges: ArrayLike,
    model_pos: ArrayLike,
    directed_edges: ArrayLike,
    edge_attr_before_model: ArrayLike,
    atol: float = 2.0e-5,
) -> dict[str, Any]:
    """Validate dataset-index identity and the exact released message stencil.

    The check establishes that HDF5 nodes and unique edges map by identity to
    model nodes and to the first half of the directed model graph.  It also
    verifies the PyG position normalization and Cartesian/distance features.
    It does not establish a control-volume or physical-face interpretation.
    """

    raw_points = _validate_positions(raw_pos)
    nodes = raw_points.shape[0]
    unique = normalize_edges(raw_edges, nodes)
    if np.any(unique[:, 0] == unique[:, 1]):
        raise ValueError("raw graph contains self edges")
    canonical = np.sort(unique, axis=1)
    if np.unique(canonical, axis=0).shape[0] != unique.shape[0]:
        raise ValueError("raw graph repeats an undirected edge")

    full = normalize_edges(directed_edges, nodes)
    expected_full = np.concatenate((unique, unique[:, ::-1]), axis=0)
    if not np.array_equal(full, expected_full):
        raise ValueError(
            "directed model edges are not raw unique edges followed by reversals"
        )

    transformed = _validate_positions(model_pos)
    expected_pos = normalize_scale_positions(raw_points)
    if not np.allclose(transformed, expected_pos, rtol=0.0, atol=atol):
        error = float(np.max(np.abs(transformed - expected_pos)))
        raise ValueError(f"model positions do not match NormalizeScale (max {error})")

    features = np.asarray(edge_attr_before_model, dtype=np.float64)
    if features.shape != (full.shape[0], 3):
        raise ValueError(
            "edge_attr_before_model must contain Cartesian x/y and distance; "
            f"got {features.shape}"
        )
    displacement = transformed[full[:, 0]] - transformed[full[:, 1]]
    distance = np.linalg.norm(displacement, axis=1)
    expected_features = np.column_stack((displacement, distance))
    if not np.allclose(features, expected_features, rtol=1.0e-5, atol=atol):
        error = float(np.max(np.abs(features - expected_features)))
        raise ValueError(f"edge features do not map to the model stencil (max {error})")

    raw_lengths = np.linalg.norm(
        raw_points[unique[:, 1]] - raw_points[unique[:, 0]], axis=1
    )
    if not np.all(np.isfinite(raw_lengths)) or np.any(raw_lengths <= 0.0):
        raise ValueError("raw graph contains a nonpositive or nonfinite edge length")
    return {
        "dataset_node_to_model_node": "verified_index_identity",
        "dataset_unique_edge_to_model_edge": "verified_first_half_identity",
        "reverse_edge_layout": "verified_second_half_exact_reversal",
        "model_position_transform": "verified_pyg_normalize_scale",
        "model_cartesian_feature": "verified_source_minus_target",
        "model_distance_feature": "verified",
        "num_nodes": int(nodes),
        "num_unique_edges": int(unique.shape[0]),
        "raw_edge_length_min": float(raw_lengths.min()),
        "raw_edge_length_median": float(np.median(raw_lengths)),
        "raw_edge_length_max": float(raw_lengths.max()),
        "control_volume_mapping": "missing",
        "physical_face_mapping": "missing",
        "physical_volume_and_face_measure_fields": "missing",
    }


def graph_hop_distances(
    edges: ArrayLike,
    num_nodes: int,
    sources: int | Sequence[int] | np.ndarray,
    *,
    max_hops: int | None = None,
) -> np.ndarray:
    """Compute undirected shortest-hop distances from one or more sources."""

    unique = normalize_edges(edges, num_nodes)
    if isinstance(sources, (int, np.integer)):
        source_array = np.asarray([sources], dtype=np.int64)
    else:
        source_array = np.asarray(sources, dtype=np.int64).reshape(-1)
    if source_array.size == 0:
        raise ValueError("at least one source node is required")
    if source_array.min() < 0 or source_array.max() >= num_nodes:
        raise ValueError("source node is outside the graph")
    if max_hops is not None and max_hops < 0:
        raise ValueError("max_hops must be nonnegative")

    adjacency = _undirected_adjacency(unique, num_nodes)
    distance = np.full(num_nodes, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for source in np.unique(source_array):
        distance[source] = 0
        queue.append(int(source))
    while queue:
        node = queue.popleft()
        if max_hops is not None and distance[node] >= max_hops:
            continue
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def cpg_dependency_mask(
    edges: ArrayLike,
    num_nodes: int,
    target_node: int,
    *,
    message_passing_layers: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact architectural support and hop distances for one output."""

    radius = cpg_dependency_trace(message_passing_layers).final_output_radius
    distance = graph_hop_distances(edges, num_nodes, target_node, max_hops=radius)
    return distance >= 0, distance


def euler_characteristic_travel_times(
    pos: ArrayLike,
    edges: ArrayLike,
    primitive_samples: ArrayLike,
    *,
    gamma: float = DEFAULT_GAMMA,
    speed_epsilon: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """Compute oriented Euler and symmetric-envelope edge travel times.

    primitive_samples has shape (S, N, 4).  For every directed traversal u to
    v, the unit direction is (x_v - x_u) / ell.  Speeds are maximized over all
    supplied samples and both edge endpoints.  Nonpositive one-way speeds yield
    infinite travel time.  The usual absolute-normal-velocity expression is
    returned only as a separate symmetric fastest-wave envelope.
    """

    points = _validate_positions(pos)
    nodes = points.shape[0]
    unique = normalize_edges(edges, nodes)
    states = np.asarray(primitive_samples, dtype=np.float64)
    if states.ndim == 2:
        states = states[None, ...]
    if states.ndim != 3 or states.shape[1:] != (nodes, 4):
        raise ValueError(
            f"primitive_samples must have shape (S, {nodes}, 4), got {states.shape}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("primitive samples contain nonfinite values")
    if np.any(states[..., 0] <= 0.0) or np.any(states[..., 3] <= 0.0):
        raise ValueError("primitive samples require positive density and pressure")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    if speed_epsilon <= 0.0:
        raise ValueError("speed_epsilon must be positive")

    directed = np.concatenate((unique, unique[:, ::-1]), axis=0)
    delta = points[directed[:, 1]] - points[directed[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("graph contains a nonpositive or nonfinite edge length")
    normals = delta / lengths[:, None]

    endpoint = states[:, directed, :]
    density = endpoint[..., 0]
    velocity = endpoint[..., 1:3]
    pressure = endpoint[..., 3]
    sound_speed = np.sqrt(gamma * pressure / density)
    normal_velocity = np.sum(velocity * normals[None, :, None, :], axis=-1)

    minus_speed = np.max(normal_velocity - sound_speed, axis=(0, 2))
    contact_speed = np.max(normal_velocity, axis=(0, 2))
    plus_speed = np.max(normal_velocity + sound_speed, axis=(0, 2))
    symmetric_speed = np.max(np.abs(normal_velocity) + sound_speed, axis=(0, 2))
    return {
        "directed_edges": directed,
        "length": np.concatenate(
            (lengths[: unique.shape[0]], lengths[: unique.shape[0]])
        ),
        "unit_direction_source_to_target": normals,
        "minus_speed_max": minus_speed,
        "contact_speed_max": contact_speed,
        "plus_speed_max": plus_speed,
        "plus_travel_time": _travel_time(lengths, plus_speed, speed_epsilon),
        "symmetric_speed_max": symmetric_speed,
        "symmetric_travel_time": _travel_time(lengths, symmetric_speed, speed_epsilon),
    }


def build_incoming_travel_adjacency(
    directed_edges: ArrayLike,
    travel_time: ArrayLike,
    num_nodes: int,
) -> list[list[tuple[int, float]]]:
    """Build reversed adjacency for predecessor-to-target shortest paths."""

    arcs = normalize_edges(directed_edges, num_nodes)
    weights = np.asarray(travel_time, dtype=np.float64)
    if weights.shape != (arcs.shape[0],):
        raise ValueError(f"travel_time must have shape ({arcs.shape[0]},)")
    if np.any(weights < 0.0) or np.any(np.isnan(weights)):
        raise ValueError("travel times must be nonnegative or infinite")
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for (source, target), weight in zip(arcs, weights, strict=True):
        if np.isfinite(weight):
            incoming[int(target)].append((int(source), float(weight)))
    return incoming


def minimum_predecessor_times(
    incoming: Sequence[Sequence[tuple[int, float]]],
    target_node: int,
    *,
    cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum one-way time from every predecessor to a target node."""

    nodes = len(incoming)
    if target_node < 0 or target_node >= nodes:
        raise ValueError("target node is outside the graph")
    if cutoff is not None and cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")
    distance = np.full(nodes, np.inf, dtype=np.float64)
    next_hop = np.full(nodes, -1, dtype=np.int64)
    distance[target_node] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(target_node))]
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance != distance[node]:
            continue
        if cutoff is not None and current_distance > cutoff:
            break
        for predecessor, edge_time in incoming[node]:
            candidate = current_distance + edge_time
            if cutoff is not None and candidate > cutoff:
                continue
            if candidate < distance[predecessor]:
                distance[predecessor] = candidate
                next_hop[predecessor] = node
                heapq.heappush(heap, (candidate, predecessor))
    return distance, next_hop


def compare_characteristic_and_model_reach(
    *,
    edges: ArrayLike,
    num_nodes: int,
    target_node: int,
    directed_predecessor_time: ArrayLike,
    symmetric_predecessor_time: ArrayLike,
    macro_dt: float,
    injected_mask: ArrayLike | None = None,
    message_passing_layers: int = 12,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare exact CPG support with directed and symmetric physical cones."""

    if macro_dt <= 0.0:
        raise ValueError("macro_dt must be positive")
    directed_time = np.asarray(directed_predecessor_time, dtype=np.float64)
    symmetric_time = np.asarray(symmetric_predecessor_time, dtype=np.float64)
    if directed_time.shape != (num_nodes,) or symmetric_time.shape != (num_nodes,):
        raise ValueError("predecessor time arrays must have shape (num_nodes,)")
    dependency, dependency_distance = cpg_dependency_mask(
        edges,
        num_nodes,
        target_node,
        message_passing_layers=message_passing_layers,
    )
    directed = directed_time <= macro_dt * (1.0 + 1.0e-12)
    symmetric = symmetric_time <= macro_dt * (1.0 + 1.0e-12)
    directed_uncovered = directed & ~dependency
    symmetric_uncovered = symmetric & ~dependency

    if injected_mask is None:
        injected = np.zeros(num_nodes, dtype=bool)
    else:
        injected = np.asarray(injected_mask, dtype=bool)
        if injected.shape != (num_nodes,):
            raise ValueError("injected_mask must have shape (num_nodes,)")

    full_hops = graph_hop_distances(edges, num_nodes, target_node)
    summary = {
        "target_node": int(target_node),
        "model_dependency_radius": int(
            cpg_dependency_trace(message_passing_layers).final_output_radius
        ),
        "model_dependency_count": int(np.count_nonzero(dependency)),
        "model_dependency_injected_count": int(np.count_nonzero(dependency & injected)),
        "target_distance_from_injected_nodes": _target_distance_from_sources(
            edges, num_nodes, target_node, injected
        ),
        "directed_predecessor_count": int(np.count_nonzero(directed)),
        "directed_predecessor_injected_count": int(
            np.count_nonzero(directed & injected)
        ),
        "directed_uncovered_count": int(np.count_nonzero(directed_uncovered)),
        "directed_max_graph_hops": _masked_max(full_hops, directed),
        "directed_uncovered_min_time": _masked_min(directed_time, directed_uncovered),
        "symmetric_predecessor_count": int(np.count_nonzero(symmetric)),
        "symmetric_uncovered_count": int(np.count_nonzero(symmetric_uncovered)),
        "symmetric_max_graph_hops": _masked_max(full_hops, symmetric),
        "coverage_status": (
            "endpoint_sampled_directed_cone_within_model_support"
            if not np.any(directed_uncovered)
            else "endpoint_sampled_directed_cone_exceeds_model_support"
        ),
    }
    arrays = {
        "dependency_mask": dependency,
        "dependency_hop_distance": dependency_distance,
        "directed_predecessor_mask": directed,
        "directed_uncovered_mask": directed_uncovered,
        "symmetric_predecessor_mask": symmetric,
        "symmetric_uncovered_mask": symmetric_uncovered,
    }
    return summary, arrays


def node_pressure_jump_scores(primitive: ArrayLike, edges: ArrayLike) -> np.ndarray:
    """Return maximum incident absolute pressure jump for every graph node."""

    state = np.asarray(primitive, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("primitive must have shape (num_nodes, 4)")
    unique = normalize_edges(edges, state.shape[0])
    jump = np.abs(state[unique[:, 0], 3] - state[unique[:, 1], 3])
    score = np.zeros(state.shape[0], dtype=np.float64)
    np.maximum.at(score, unique[:, 0], jump)
    np.maximum.at(score, unique[:, 1], jump)
    return score


def select_reach_targets(
    *,
    pos: ArrayLike,
    edges: ArrayLike,
    primitive: ArrayLike,
    normal_mask: ArrayLike,
    boundary_distance: ArrayLike,
    dependency_radius: int,
    count_per_region: int = 2,
) -> list[dict[str, Any]]:
    """Select deterministic shock, smooth, upstream, and downstream targets."""

    points = _validate_positions(pos)
    nodes = points.shape[0]
    normal = np.asarray(normal_mask, dtype=bool)
    distance = np.asarray(boundary_distance, dtype=np.int64)
    if normal.shape != (nodes,) or distance.shape != (nodes,):
        raise ValueError("normal_mask and boundary_distance must match node count")
    if count_per_region < 1:
        raise ValueError("count_per_region must be positive")
    normal_ids = np.flatnonzero(normal)
    if normal_ids.size == 0:
        raise ValueError("no normal nodes are available for target selection")
    far_ids = normal_ids[distance[normal_ids] > dependency_radius]
    bulk = far_ids if far_ids.size else normal_ids
    score = node_pressure_jump_scores(primitive, edges)

    orders = {
        "shock": _rank_nodes(normal_ids, -score[normal_ids]),
        "smooth": _rank_nodes(bulk, score[bulk]),
        "upstream": _rank_nodes(bulk, points[bulk, 0]),
        "downstream": _rank_nodes(bulk, -points[bulk, 0]),
    }
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for region, order in orders.items():
        added = 0
        for node in order:
            node_int = int(node)
            if node_int in used:
                continue
            selected.append(
                {
                    "target_node": node_int,
                    "region": region,
                    "pressure_jump_score": float(score[node_int]),
                    "boundary_graph_distance": int(distance[node_int]),
                }
            )
            used.add(node_int)
            added += 1
            if added == count_per_region:
                break
    return selected


def construct_pointwise_admissible_causal_pair(
    *,
    base_primitive: ArrayLike,
    pos: ArrayLike,
    edges: ArrayLike,
    target_node: int,
    dependency_mask: ArrayLike,
    predecessor_time: ArrayLike,
    next_hop: ArrayLike,
    macro_dt: float,
    normal_mask: ArrayLike,
    amplitude: float = 0.01,
    patch_hops: int = 0,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, Any] | None:
    """Construct a graph-state causal-pair candidate outside model support.

    The perturbation follows the local positive acoustic characteristic toward
    the target and keeps density and pressure positive.  It is pointwise
    admissible and leaves all non-normal and in-support nodes unchanged.  It is
    not a validated DG initial condition until solver degrees of freedom and
    graph nodes are explicitly mapped.
    """

    state_a = np.asarray(base_primitive, dtype=np.float64)
    points = _validate_positions(pos)
    if state_a.shape != (points.shape[0], 4):
        raise ValueError("base_primitive must have shape (num_nodes, 4)")
    if np.any(state_a[:, 0] <= 0.0) or np.any(state_a[:, 3] <= 0.0):
        raise ValueError("base state must have positive density and pressure")
    if not (0.0 < amplitude < 0.25):
        raise ValueError("amplitude must lie in (0, 0.25)")
    if patch_hops < 0:
        raise ValueError("patch_hops must be nonnegative")
    nodes = state_a.shape[0]
    dependency = np.asarray(dependency_mask, dtype=bool)
    travel = np.asarray(predecessor_time, dtype=np.float64)
    successor = np.asarray(next_hop, dtype=np.int64)
    normal = np.asarray(normal_mask, dtype=bool)
    if any(
        array.shape != (nodes,) for array in (dependency, travel, successor, normal)
    ):
        raise ValueError("pair masks and predecessor arrays must match node count")

    candidate = (
        np.isfinite(travel)
        & (travel <= macro_dt * (1.0 + 1.0e-12))
        & ~dependency
        & normal
        & (successor >= 0)
    )
    candidate_ids = np.flatnonzero(candidate)
    if candidate_ids.size == 0:
        return None
    source = int(candidate_ids[np.argmin(travel[candidate_ids])])
    patch_distance = graph_hop_distances(edges, nodes, source, max_hops=patch_hops)
    perturb = (patch_distance >= 0) & candidate
    weights = np.zeros(nodes, dtype=np.float64)
    weights[perturb] = 1.0 - (
        patch_distance[perturb].astype(np.float64) / (patch_hops + 1.0)
    )

    state_b = state_a.copy()
    for node in np.flatnonzero(perturb):
        following = int(successor[node])
        direction = points[following] - points[node]
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("causal path contains a zero-length hop")
        direction /= norm
        rho, _, _, pressure = state_a[node]
        sound_speed = float(np.sqrt(gamma * pressure / rho))
        strength = amplitude * weights[node]
        state_b[node, 0] = rho * (1.0 + strength)
        state_b[node, 1:3] += strength * sound_speed * direction
        state_b[node, 3] = pressure * (1.0 + gamma * strength)

    if not np.array_equal(state_a[dependency], state_b[dependency]):
        raise RuntimeError("causal pair changed a node inside model dependency support")
    if np.any(state_b[:, 0] <= 0.0) or np.any(state_b[:, 3] <= 0.0):
        raise RuntimeError("causal pair construction violated positivity")
    return {
        "state_a": state_a,
        "state_b": state_b,
        "perturbation_mask": perturb,
        "source_node": source,
        "target_node": int(target_node),
        "source_to_target_time": float(travel[source]),
        "amplitude": float(amplitude),
        "patch_hops": int(patch_hops),
        "perturbed_node_count": int(np.count_nonzero(perturb)),
        "admissibility": "pointwise_positive_density_and_pressure",
        "solver_initial_condition_status": "unverified_graph_to_dg_map",
        "required_causal_label": (
            "reference target difference from the validated solver under legal "
            "boundaries; nominal dataset error is not a causal label"
        ),
    }


def _travel_time(
    lengths: np.ndarray, speed: np.ndarray, speed_epsilon: float
) -> np.ndarray:
    out = np.full(lengths.shape, np.inf, dtype=np.float64)
    allowed = speed > speed_epsilon
    out[allowed] = lengths[allowed] / speed[allowed]
    return out


def _validate_positions(pos: ArrayLike) -> np.ndarray:
    points = np.asarray(pos, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"positions must have shape (N, dim>=2), got {points.shape}")
    points = points[:, :2]
    if points.shape[0] < 1 or not np.all(np.isfinite(points)):
        raise ValueError("positions must be nonempty and finite")
    return points


def _undirected_adjacency(edges: np.ndarray, num_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return adjacency


def _target_distance_from_sources(
    edges: ArrayLike,
    num_nodes: int,
    target_node: int,
    sources: np.ndarray,
) -> int | None:
    source_ids = np.flatnonzero(sources)
    if source_ids.size == 0:
        return None
    distance = graph_hop_distances(edges, num_nodes, source_ids)
    value = int(distance[target_node])
    return None if value < 0 else value


def _masked_max(values: np.ndarray, mask: np.ndarray) -> int | None:
    selected = values[mask]
    selected = selected[selected >= 0]
    return None if selected.size == 0 else int(np.max(selected))


def _masked_min(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return None if selected.size == 0 else float(np.min(selected))


def _rank_nodes(nodes: np.ndarray, primary: np.ndarray) -> np.ndarray:
    order = np.lexsort((nodes, primary))
    return nodes[order]
