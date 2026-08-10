from __future__ import annotations

import numpy as np
import torch

from pcno.pcno import compute_Fourier_bases
from utility.time_dependent_no.pcno_geometric_consistency import (
    BUMP_NODE_TYPE_NAMES,
    ROTATION_90_CCW,
    _type_stratified_assignment,
    build_bump_query_graph,
    regenerate_differential_weights,
    regenerate_element_differential_geometry,
    rotate_euler_field,
    rotate_points,
    rotation_center,
    symmetric_directed_edges,
    transported_fourier_tensors,
)


def _grid_graph(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [(float(x), float(y)) for y in range(height) for x in range(width)]
    )
    pairs = []
    for index, (x, y) in enumerate(nodes):
        for other in range(index + 1, nodes.shape[0]):
            ox, oy = nodes[other]
            if max(abs(x - ox), abs(y - oy)) == 1.0:
                pairs.append((index, other))
    return nodes, symmetric_directed_edges(np.asarray(pairs, dtype=np.int64))


def _quad_elements(width: int, height: int) -> np.ndarray:
    rows = []
    for y in range(height - 1):
        for x in range(width - 1):
            lower_left = y * width + x
            rows.append(
                (
                    2,
                    lower_left,
                    lower_left + 1,
                    lower_left + width + 1,
                    lower_left + width,
                )
            )
    return np.asarray(rows, dtype=np.int64)


def test_radius_cover_traverses_vertices_covered_by_an_earlier_anchor():
    # The 0-1-2-3 path plus branch 2-4 exposes a global-covered/local-BFS mixup:
    # after anchor 0 covers 0,1,2, anchor 3 must traverse 2 to cover 4.
    adjacency = [[1], [0, 2], [1, 3, 4], [2], [2]]
    anchors, assignment = _type_stratified_assignment(
        np.zeros(5, dtype=np.int64), adjacency, radius=2
    )
    np.testing.assert_array_equal(anchors, np.asarray([0, 3]))
    assert assignment.shape == (5,)


def test_bump_query_graph_is_type_stratified_and_proxy_mass_preserving():
    nodes, edges = _grid_graph(11, 11)
    node_type = np.zeros(nodes.shape[0], dtype=np.int64)
    node_type[nodes[:, 1] == 0.0] = 1
    node_type[nodes[:, 0] == 0.0] = 3
    node_type[nodes[:, 0] == 10.0] = 2
    measures = (1.0 + 0.01 * nodes[:, 0])[:, None]

    query = build_bump_query_graph(
        nodes,
        edges,
        measures,
        node_type,
        radius=2,
    )

    assert query.coarse_node_count < query.fine_node_count
    assert query.radius == 2
    assert BUMP_NODE_TYPE_NAMES == {
        0: "normal",
        1: "wall",
        2: "outflow",
        3: "inflow",
    }
    np.testing.assert_array_equal(query.node_type[query.fine_to_coarse], node_type)
    np.testing.assert_allclose(
        query.node_measures.sum(), measures.sum(), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(query.node_weights.sum(), 1.0, rtol=0.0, atol=1.0e-12)

    field = np.stack(
        (np.ones(nodes.shape[0]), nodes[:, 0], nodes[:, 1], nodes[:, 0] ** 2),
        axis=-1,
    )
    restricted = query.restrict(field)
    prolonged = query.prolong(restricted)
    assert restricted.shape == (query.coarse_node_count, 4)
    assert prolonged.shape == field.shape
    np.testing.assert_allclose(query.restrict(np.ones_like(field)), 1.0)
    native_integral = np.einsum("n,nc->c", measures[:, 0], field)
    coarse_integral = np.einsum("n,nc->c", query.node_measures[:, 0], restricted)
    np.testing.assert_allclose(coarse_integral, native_integral, rtol=1.0e-13)

    directed = set(map(tuple, query.directed_edges.tolist()))
    assert all((right, left) in directed for left, right in directed)


def test_rotation_round_trip_differential_and_fourier_covariance():
    nodes, edges = _grid_graph(4, 4)
    elements = _quad_elements(4, 4)
    nodes = nodes + np.asarray(
        [
            (0.03 * np.sin(index), 0.02 * np.cos(2 * index))
            for index in range(nodes.shape[0])
        ]
    )
    center = rotation_center(nodes)
    rotated_nodes = rotate_points(nodes, center)
    np.testing.assert_allclose(
        rotate_points(rotated_nodes, center, inverse=True), nodes, atol=1.0e-14
    )

    weights, native_rank = regenerate_differential_weights(nodes, edges)
    regenerated_edges, regenerated_weights = regenerate_element_differential_geometry(
        nodes, elements
    )
    rotated_edges, rotated_weights = regenerate_element_differential_geometry(
        rotated_nodes, elements
    )
    np.testing.assert_array_equal(regenerated_edges, edges)
    np.testing.assert_array_equal(rotated_edges, edges)
    np.testing.assert_allclose(regenerated_weights, weights, atol=2.0e-13)
    _, rotated_rank = regenerate_differential_weights(rotated_nodes, rotated_edges)
    assert native_rank["minimum_rank"] == 2
    assert rotated_rank["minimum_rank"] == 2
    np.testing.assert_allclose(
        rotated_weights,
        weights @ ROTATION_90_CCW.T,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    rng = np.random.default_rng(20260809)
    state = rng.normal(size=(3, nodes.shape[0], 4))
    rotated_state = rotate_euler_field(state)
    np.testing.assert_allclose(
        rotate_euler_field(rotated_state, inverse=True), state, atol=1.0e-14
    )
    np.testing.assert_allclose(
        np.square(rotated_state[..., 1:3]).sum(axis=-1),
        np.square(state[..., 1:3]).sum(axis=-1),
        atol=1.0e-13,
    )

    modes = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.0], [2.0]],
            [[1.5], [-0.5]],
        ],
        dtype=torch.float64,
    )
    torch_nodes = torch.as_tensor(nodes, dtype=torch.float64).unsqueeze(0)
    torch_rotated = torch.as_tensor(rotated_nodes, dtype=torch.float64).unsqueeze(0)
    node_weights = torch.full(
        (1, nodes.shape[0], 1), 1.0 / nodes.shape[0], dtype=torch.float64
    )
    original = compute_Fourier_bases(torch_nodes, modes)
    transported = transported_fourier_tensors(
        torch_rotated,
        node_weights,
        modes,
        center=center,
    )
    for expected, actual in zip(original, transported[:3]):
        torch.testing.assert_close(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    for basis, weighted in zip(transported[:3], transported[3:]):
        torch.testing.assert_close(
            weighted,
            basis * node_weights[:, :, None, :],
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    modes_before = modes.clone()
    fixed_bases = compute_Fourier_bases(torch_rotated, modes)
    fixed = (
        *fixed_bases,
        *(basis * node_weights[:, :, None, :] for basis in fixed_bases),
    )
    torch.testing.assert_close(modes, modes_before, rtol=0.0, atol=0.0)
    assert (
        max(
            float((left - right).abs().max()) for left, right in zip(fixed, transported)
        )
        > 0.1
    )
