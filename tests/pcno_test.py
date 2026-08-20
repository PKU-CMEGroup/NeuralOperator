from unittest.mock import patch

import numpy as np
import torch

from pcno.geo_utility import (
    compute_edge_gradient_weights,
    compute_elem_adjacent_list,
    compute_node_adjacent_list,
    compute_node_measures,
    compute_node_weights,
    compute_triangle_area_,
    convert_structured_data,
    preprocess_data_mesh,
)
from pcno.pcno import (
    PCNO,
    _compute_Fourier_bases_and_weights,
    compute_flat_edge_indices,
    compute_gradient,
    compute_neighbor_degree,
    graph_neighbor_average,
)

#####################################################################
# PCNO CODE TESTS
#####################################################################


def _compute_gradient_materialized_reference(
    f: torch.Tensor,
    directed_edges: torch.Tensor,
    edge_gradient_weights: torch.Tensor,
) -> torch.Tensor:
    f = f.permute(0, 2, 1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    target, source = directed_edges[..., 0], directed_edges[..., 1]
    batch_index = torch.arange(batch_size, device=f.device).unsqueeze(1)
    message = torch.einsum(
        "bed,bec->becd",
        edge_gradient_weights,
        f[batch_index, source] - f[batch_index, target],
    ).reshape(batch_size, max_nedges, in_channels * ndims)
    gradients = torch.zeros(
        batch_size,
        max_nnodes,
        in_channels * ndims,
        dtype=message.dtype,
        device=message.device,
    )
    gradients.scatter_add_(
        dim=1,
        src=message,
        index=target.unsqueeze(2).repeat(1, 1, in_channels * ndims),
    )
    return gradients.permute(0, 2, 1)


def test_compute_gradient_matches_materialized_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(20260801)
    features = torch.randn(2, 3, 5, dtype=torch.float64, generator=generator)
    directed_edges = torch.tensor(
        [
            [[0, 1], [0, 2], [1, 0], [1, 3], [2, 4], [4, 2]],
            [[0, 4], [2, 1], [2, 3], [3, 0], [3, 4], [4, 1]],
        ],
        dtype=torch.int64,
    )
    edge_gradient_weights = torch.randn(
        2, 6, 2, dtype=torch.float64, generator=generator
    )
    output_gradient = torch.randn(2, 6, 5, dtype=torch.float64, generator=generator)

    actual_features = features.clone().requires_grad_(True)
    actual_weights = edge_gradient_weights.clone().requires_grad_(True)
    actual = compute_gradient(actual_features, directed_edges, actual_weights)
    (actual * output_gradient).sum().backward()

    reference_features = features.clone().requires_grad_(True)
    reference_weights = edge_gradient_weights.clone().requires_grad_(True)
    reference = _compute_gradient_materialized_reference(
        reference_features, directed_edges, reference_weights
    )
    (reference * output_gradient).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_features.grad, reference_features.grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual_weights.grad, reference_weights.grad, rtol=0.0, atol=0.0
    )

    chunked_features = features.clone().requires_grad_(True)
    with patch("pcno.pcno._GRADIENT_MESSAGE_CHUNK_BYTES", 32):
        chunked = compute_gradient(
            chunked_features,
            directed_edges,
            edge_gradient_weights,
            flat_edge_indices=compute_flat_edge_indices(
                directed_edges,
                features.shape[-1],
            ),
        )
        (chunked * output_gradient).sum().backward()

    torch.testing.assert_close(chunked, reference, rtol=1.0e-15, atol=1.0e-15)
    torch.testing.assert_close(
        chunked_features.grad,
        reference_features.grad,
        rtol=1.0e-15,
        atol=1.0e-15,
    )


def test_flat_graph_gradient_passes_first_and_second_derivative_checks() -> None:
    directed_edges = torch.tensor(
        [[[0, 1], [0, 2], [1, 0], [2, 3], [3, 1]]],
        dtype=torch.int64,
    )
    edge_gradient_weights = torch.randn(
        1,
        directed_edges.shape[1],
        2,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(20260815),
    )
    flat_edge_indices = compute_flat_edge_indices(directed_edges, nnodes=4)
    values = torch.randn(
        1,
        2,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(20260816),
        requires_grad=True,
    )

    def function(tensor: torch.Tensor) -> torch.Tensor:
        return compute_gradient(
            tensor,
            directed_edges,
            edge_gradient_weights,
            flat_edge_indices=flat_edge_indices,
        )

    with patch("pcno.pcno._GRADIENT_MESSAGE_CHUNK_BYTES", 16):
        assert torch.autograd.gradcheck(function, (values,))
        assert torch.autograd.gradgradcheck(function, (values,))


def test_homogeneous_fourier_tensors_match_materialized_batch_and_share_storage() -> None:
    generator = torch.Generator().manual_seed(20260802)
    batch_size = 3
    base_nodes = torch.rand(1, 5, 2, dtype=torch.float64, generator=generator)
    base_weights = torch.rand(1, 5, 1, dtype=torch.float64, generator=generator)
    modes = torch.tensor(
        [
            [[1.0], [0.5]],
            [[0.25], [-0.75]],
            [[-0.5], [1.25]],
        ],
        dtype=torch.float64,
    )
    expanded_nodes = base_nodes.expand(batch_size, -1, -1)
    expanded_weights = base_weights.expand(batch_size, -1, -1)

    actual = _compute_Fourier_bases_and_weights(
        expanded_nodes,
        expanded_weights,
        modes,
    )
    reference = _compute_Fourier_bases_and_weights(
        expanded_nodes.clone(),
        expanded_weights.clone(),
        modes,
    )

    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(
            actual_tensor,
            reference_tensor,
            rtol=0.0,
            atol=0.0,
        )
        assert actual_tensor.stride(0) == 0
        assert actual_tensor.untyped_storage().nbytes() == (
            actual_tensor[0].numel() * actual_tensor.element_size()
        )


def test_neighbor_degree_matches_materialized_batch_and_shares_storage() -> None:
    batch_size = 4
    base_edges = torch.tensor(
        [[[0, 1], [0, 2], [1, 0], [2, 3], [2, 4], [4, 1]]],
        dtype=torch.int64,
    )
    expanded_edges = base_edges.expand(batch_size, -1, -1)

    actual = compute_neighbor_degree(
        expanded_edges,
        nnodes=5,
        dtype=torch.float64,
    )
    reference = compute_neighbor_degree(
        expanded_edges.clone(),
        nnodes=5,
        dtype=torch.float64,
    )

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    assert actual.stride(0) == 0
    assert actual.untyped_storage().nbytes() == (
        actual[0].numel() * actual.element_size()
    )


def test_graph_neighbor_average_reuses_degree_without_changing_gradients() -> None:
    generator = torch.Generator().manual_seed(20260802)
    features = torch.randn(2, 3, 5, dtype=torch.float64, generator=generator)
    directed_edges = torch.tensor(
        [
            [[0, 1], [0, 2], [1, 0], [2, 3], [2, 4], [4, 1]],
            [[0, 4], [1, 2], [1, 3], [3, 0], [3, 4], [4, 2]],
        ],
        dtype=torch.int64,
    )
    output_gradient = torch.randn(2, 3, 5, dtype=torch.float64, generator=generator)
    neighbor_degree = compute_neighbor_degree(
        directed_edges,
        nnodes=features.shape[-1],
        dtype=features.dtype,
    )

    actual_features = features.clone().requires_grad_(True)
    actual = graph_neighbor_average(
        actual_features,
        directed_edges,
        iterations=2,
        neighbor_degree=neighbor_degree,
    )
    (actual * output_gradient).sum().backward()

    reference_features = features.clone().requires_grad_(True)
    reference = graph_neighbor_average(
        reference_features,
        directed_edges,
        iterations=2,
    )
    (reference * output_gradient).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_features.grad,
        reference_features.grad,
        rtol=0.0,
        atol=0.0,
    )

    flat_features = features.clone().requires_grad_(True)
    flat = graph_neighbor_average(
        flat_features,
        directed_edges,
        iterations=2,
        neighbor_degree=neighbor_degree,
        flat_edge_indices=compute_flat_edge_indices(
            directed_edges,
            features.shape[-1],
        ),
    )
    (flat * output_gradient).sum().backward()

    torch.testing.assert_close(flat, reference, rtol=1.0e-15, atol=1.0e-15)
    torch.testing.assert_close(
        flat_features.grad,
        reference_features.grad,
        rtol=1.0e-15,
        atol=1.0e-15,
    )


def test_flat_graph_average_passes_first_and_second_derivative_checks() -> None:
    directed_edges = torch.tensor(
        [[[0, 1], [0, 2], [1, 0], [2, 3], [3, 1]]],
        dtype=torch.int64,
    )
    nnodes = 4
    neighbor_degree = compute_neighbor_degree(
        directed_edges,
        nnodes,
        dtype=torch.float64,
    )
    flat_edge_indices = compute_flat_edge_indices(directed_edges, nnodes)
    values = torch.randn(
        1,
        2,
        nnodes,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(20260802),
        requires_grad=True,
    )

    def function(tensor: torch.Tensor) -> torch.Tensor:
        return graph_neighbor_average(
            tensor,
            directed_edges,
            iterations=2,
            neighbor_degree=neighbor_degree,
            flat_edge_indices=flat_edge_indices,
        )

    with patch("pcno.pcno._GRAPH_MESSAGE_CHUNK_BYTES", 16):
        assert torch.autograd.gradcheck(function, (values,))
        assert torch.autograd.gradgradcheck(function, (values,))


def test_pcno_homogeneous_geometry_matches_materialized_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(20260802)
    batch_size, nnodes = 3, 5
    modes = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
            [[1.0], [1.0]],
        ],
        dtype=torch.float64,
    )
    base_node_mask = torch.ones(1, nnodes, 1, dtype=torch.float64)
    base_nodes = torch.rand(1, nnodes, 2, dtype=torch.float64, generator=generator)
    base_weights = torch.rand(
        1,
        nnodes,
        1,
        dtype=torch.float64,
        generator=generator,
    )
    base_edges = torch.tensor(
        [[[0, 1], [0, 2], [1, 0], [2, 3], [2, 4], [3, 1], [4, 2]]],
        dtype=torch.int64,
    )
    base_edge_weights = torch.randn(
        1,
        base_edges.shape[1],
        2,
        dtype=torch.float64,
        generator=generator,
    )
    expanded_aux = [
        base_node_mask.expand(batch_size, -1, -1),
        base_nodes.expand(batch_size, -1, -1),
        base_weights.expand(batch_size, -1, -1),
        base_edges.expand(batch_size, -1, -1),
        base_edge_weights.expand(batch_size, -1, -1),
    ]
    materialized_aux = [tensor.clone() for tensor in expanded_aux]

    torch.manual_seed(20260802)
    actual_model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=3,
        out_dim=2,
    ).double()
    reference_model = PCNO(
        ndims=2,
        modes=modes,
        nmeasures=1,
        layers=[4, 4, 4],
        fc_dim=5,
        in_dim=3,
        out_dim=2,
    ).double()
    reference_model.load_state_dict(actual_model.state_dict())

    inputs = torch.randn(
        batch_size,
        nnodes,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    output_gradient = torch.randn(
        batch_size,
        nnodes,
        2,
        dtype=torch.float64,
        generator=generator,
    )

    actual_inputs = inputs.clone().requires_grad_(True)
    actual = actual_model(actual_inputs, expanded_aux)
    (actual * output_gradient).sum().backward()

    reference_inputs = inputs.clone().requires_grad_(True)
    reference = reference_model(reference_inputs, materialized_aux)
    (reference * output_gradient).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_inputs.grad,
        reference_inputs.grad,
        rtol=0.0,
        atol=0.0,
    )
    for (actual_name, actual_parameter), (
        reference_name,
        reference_parameter,
    ) in zip(actual_model.named_parameters(), reference_model.named_parameters()):
        assert actual_name == reference_name
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=0.0,
            atol=0.0,
        )



def test_compute_triangle_area_supports_2d_and_3d_points():
    points_2d = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    points_3d = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])

    assert np.isclose(compute_triangle_area_(points_2d), 3.0)
    assert np.isclose(compute_triangle_area_(points_3d), 3.0)


def test_preprocess_data_mesh_preserves_time_node_major_features():
    nodes_list = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    elems_list = [np.array([[2, 0, 1, 2, 3]], dtype=np.int64)]
    features_list = [np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)]

    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(
        nodes_list, elems_list, features_list, mesh_type="vertex_centered", adjacent_type="element"
    )

    assert nnodes.tolist() == [4]
    assert features.shape == (1, 2, 4, 3)
    np.testing.assert_allclose(features[0], features_list[0])


def test_convert_structured_data():
    # 2 dim test
    elem_dim=2
    Lx, Ly = 1.0, 2.0
    Npx, Npy = 2, 3
    grid_1d_x, grid_1d_y = np.linspace(0, Lx, Npx), np.linspace(0, Ly, Npy)
    grid_x, grid_y = np.meshgrid(grid_1d_x, grid_1d_y, indexing="ij")
    ndata = 2
    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 4, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,1,4,3],[elem_dim,1,2,5,4]])) == 0)
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type='vertex_centered', adjacent_type='element')
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = True)
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.all(np.isclose(node_measures - np.tile(np.array([1.0/4,1.0/2,1.0/4,1.0/4,1.0/2,1.0/4])[:,np.newaxis], (ndata,1,1)), 0)))
    assert(np.all(np.isclose(node_equal_measures - np.tile(np.array([1.0/3,1.0/3,1.0/3,1.0/3,1.0/3,1.0/3])[:,np.newaxis], (ndata,1,1)), 0)))
    assert(np.all(np.isclose(node_equal_weights - np.tile(np.array([1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6])[:,np.newaxis], (ndata,1,1)), 0.0)))
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)


    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 3, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,1,4],[elem_dim,0,4,3],[elem_dim,1,2,5],[elem_dim,1,5,4]])) == 0)
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type='vertex_centered', adjacent_type='element')
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_measures - np.tile(np.array([1.0/3,1.0/2,1.0/6,1.0/6,1.0/2,1.0/3])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)


    # 3 dim test
    elem_dim=3
    Lx, Ly, Lz = 1.0, 2.0, 3.0
    Npx, Npy, Npz = 2, 3, 2
    grid_1d_x, grid_1d_y, grid_1d_z = np.linspace(0, Lx, Npx), np.linspace(0, Ly, Npy), np.linspace(0, Lz, Npz)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d_x, grid_1d_y, grid_1d_z, indexing="ij")
    ndata = 2
    features = np.zeros((ndata, Npx, Npy, Npz, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata,1,1,1)), np.tile(grid_y, (ndata,1,1,1)), np.tile(grid_z, (ndata,1,1,1))], features, nnodes_per_elem = 8, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,6,8,2,1,7,9,3],[elem_dim,2,8,10,4,3,9,11,5]])) == 0)
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type='vertex_centered', adjacent_type='element')
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = True)
    assert(np.linalg.norm(nnodes - Npx * Npy * Npz) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0,0],[0,0,3],[0,1,0],[0,1,3],[0,2,0],[0,2,3],[1,0,0],[1,0,3],[1,1,0],[1,1,3],[1,2,0],[1,2,3]]), (ndata,1,1,1))) == 0)
    assert(np.all(np.isclose(node_measures - np.tile(np.array([3.0/8,3.0/8,6.0/8,6.0/8,3.0/8,3.0/8,3.0/8,3.0/8,6.0/8,6.0/8,3.0/8,3.0/8])[:,np.newaxis], (ndata,1,1,1)), 0)))
    assert(np.all(np.isclose(node_equal_measures - np.tile(np.array([1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2])[:,np.newaxis], (ndata,1,1,1)), 0)))
    assert(np.all(np.isclose(node_equal_weights - np.tile(np.array([1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12])[:,np.newaxis], (ndata,1,1,1)), 0.0)))
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy*Npz, 1)), nodes), axis=2)) == 0)


    features = np.zeros((ndata, Npx, Npy, Npz, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata,1,1,1)), np.tile(grid_y, (ndata,1,1,1)), np.tile(grid_z, (ndata,1,1,1))], features, nnodes_per_elem = 4, feature_include_coords = True)
    #                                                           0 1 2 3 4 5 6 7            0 1 2  3 4 5 6  7
    # assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,6,8,2,1,7,9,3],[elem_dim,2,8,10,4,3,9,11,5]])) == 0)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,6,2,7],[elem_dim,0,2,7,3],[elem_dim,0,1,7,3],[elem_dim,6,8,2,7],[elem_dim,8,7,9,3],[elem_dim,8,2,7,3],
                                                    [elem_dim,2,8,4,9],[elem_dim,2,4,9,5],[elem_dim,2,3,9,5],[elem_dim,8,10,4,9],[elem_dim,10,9,11,5],[elem_dim,10,4,9,5]])) == 0)
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type='vertex_centered', adjacent_type='element')
    assert(np.linalg.norm(nnodes - Npx * Npy * Npz) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0,0],[0,0,3],[0,1,0],[0,1,3],[0,2,0],[0,2,3],[1,0,0],[1,0,3],[1,1,0],[1,1,3],[1,2,0],[1,2,3]]), (ndata,1,1,1))) == 0)
    assert(np.linalg.norm(node_measures - np.tile(np.array([3.0/8,1.0/8,7.0/8,5.0/8,1.0/2,1.0/2,1.0/4,3.0/4,5.0/8,7.0/8,3.0/8,1.0/8])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy*Npz, 1)), nodes), axis=2)) == 0)

def adjacent_list_test():
    """Test compute_node_adjacent_list and compute_elem_adjacent_list function with various mesh types"""
    
    # 4--a--5-----6
    # |     |   e | \
    # |b    |d    | f \ 
    # 0--c--1-----2----3
    nodes = np.array([[0.0,0.0], [1.0,0.0], [2.0,0.0], [3.0,0.0],[0.0,1.0], [1.0,1.0], [2.0,1.0]])
    elems = np.array([
        [1, -1, -1, 4, 5],  # line a
        [1, -1, -1, 0, 4],  # line b
        [1, -1, -1, 0, 1],  # line c
        [1, -1, -1, 1, 5],  # line d
        [2,  1, 2, 6, 5],   # quad e
        [2, -1, 2, 3, 6],   # triangle f
    ])
    adjacent_list_vertex_centered = {'element': [set([1,4]), set([0, 2, 5, 6]), set([1, 3, 6, 5]), set([2, 6]), set([0, 5]), set([4, 1, 6, 2]), set([2, 3, 5, 1])],
                                     'edge': [set([1,4]), set([0, 2, 5]), set([1, 3, 6]), set([2, 6]), set([0, 5]), set([4, 1, 6]), set([2, 3, 5])]}
    adjacent_list_cell_centered = {'node': [set([1,3,4]), set([0, 2]), set([1, 3, 4]), set([0,2,4]), set([0,2,3,5]), set([4])],
                                   'edge': [set(), set(), set(), set([4]), set([3, 5]), set([4])],
                                   'face': [set([1,3,4]), set([0, 2]), set([1, 3, 4]), set([0,2,4]), set([3,5]), set([4])]}
    
    for adjacent_type in ['element', 'edge']:
        adj_list = compute_node_adjacent_list(nodes, elems, adjacent_type)
        assert adj_list == adjacent_list_vertex_centered[adjacent_type]
    for adjacent_type in ['node', 'edge', 'face']:
        adj_list = compute_elem_adjacent_list(elems, adjacent_type)
        assert adj_list == adjacent_list_cell_centered[adjacent_type]
    
    

    #     9 -------- 10 ------11
    #    /|         /|        /|
    #   6 -------- 7 ------- 8 |
    #   | |        | |       | |
    # z | 3 -------|-4 ------| 5 
    #   |/y        |/        |/  \
    #   0 ----x----1 --------2-----12       
    # two cube and a triangle
    nodes = np.array([[0.0,0.0,0.0], [1.0,0.0,0.0], [2.0,0.0,0.0], [0.0,1.0,0.0], [1.0,1.0,0.0], [2.0,1.0,0.0], 
                      [0.0,0.0,1.0], [1.0,0.0,1.0], [2.0,0.0,1.0], [0.0,1.0,1.0], [1.0,1.0,1.0], [2.0,1.0,1.0], 
                      [3.0,0.0,0.0]])
    elems = np.array([[3, 0, 1,  4, 3, 6, 7, 10, 9],   # cube a
                      [3, 1, 2,  5, 4, 7, 8, 11,10],  # cube b
                      [2, 2, 12, 5,-1, -1,-1, -1,-1]])  # line c
    adjacent_list_vertex_centered = {'element': [set([1,3,4,6,7,9,10]), set([0,2,3,4,5,6,7,8,9,10,11]), set([1,4,5,7,8,10,11,12]), 
                                                 set([0,1,4,6,7,9,10]), set([0,1,2,3,5,6,7,8,9,10,11]), set([1,2,4,7,8,10,11,12]),
                                                 set([0,1,3,4,7,9,10]), set([0,1,2,3,4,5,6,8,9,10,11]), set([1,2,4,5,7,10,11]),
                                                 set([0,1,3,4,6,7,10]), set([0,1,2,3,4,5,6,7,8,9,11]), set([1,2,4,5,7,8,10]),
                                                 set([2,5])],
                                     'edge': [set([1,3,6]), set([0,2,4,7]), set([1,12,5,8]), set([0,4,9]), set([1,3,5,10]), set([2,4,11,12]), 
                                              set([0,7,9]), set([1,6,8,10]), set([2,7,11]), set([3,6,10]), set([4,7,9,11]), set([5,8,10]), set([2,5])]}
    adjacent_list_cell_centered = {'node': [set([1]), set([0, 2]), set([1])],
                                   'edge': [set([1]), set([0, 2]), set([1])],
                                   'face': [set([1]), set([0]), set([1])]}
    
    for adjacent_type in ['element', 'edge']:
        adj_list = compute_node_adjacent_list(nodes, elems, adjacent_type)
        assert adj_list == adjacent_list_vertex_centered[adjacent_type]
    for adjacent_type in ['node', 'edge', 'face']:
        adj_list = compute_elem_adjacent_list(elems, adjacent_type)
        assert adj_list == adjacent_list_cell_centered[adjacent_type]
    
    
def gradient_test(ndims = 2):
    ################################
    # Preprocess
    ################################
    #nnodes by ndims
    if ndims == 2:
        nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]])
    else: 
        nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],[0.5,0.5,1.0]])
    nnodes, ndims = nodes.shape
    elem_dim = 2
    elems = np.array([[elem_dim,0,1,4],[elem_dim,2,4,1],[elem_dim,2,3,4],[elem_dim,0,4,3]], dtype=np.int64)
    # (nedges, 2), (nedges, ndims)
    directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes, elems, mesh_type='vertex_centered', adjacent_type='element', rcond=1e-3)
    nedges = directed_edges.shape[0]
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights)

    ################################
    # Construct features
    ################################
    nchannels = 4
    # features is a nchannels by nnodes array, for each channel, the gradient 
    # is gradients[i, :], and the gradient is constant for all nodes
    gradients = np.random.rand(nchannels, ndims)
    features =  gradients @ nodes.T
    # nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
    features_gradients_ref = np.repeat(gradients.reshape(1,-1), nnodes, axis=0)
    if ndims == 3:
        # remove the gradient in the normal direction
        features_gradients_ref[:,2::ndims] = 0.0

    features = torch.from_numpy(features).permute(1,0)  #nx by nchannels
    
    ################################
    # Online computation
    ################################
    # Message passing: compute f_source - f_target for each edge
    target, source = directed_edges.T  # source and target nodes of edges
    message = torch.einsum('ed,ec->ecd', edge_gradient_weights, features[source] - features[target]).reshape(nedges, nchannels*ndims)
    features_gradients = torch.zeros(nnodes, nchannels*ndims, dtype=message.dtype)
    features_gradients.scatter_add_(dim=0,  src=message, index=target.unsqueeze(1).repeat(1,nchannels*ndims))
    
    print("gradient error is ", np.linalg.norm(features_gradients-features_gradients_ref))
    assert(np.allclose(features_gradients-features_gradients_ref, 0.0, rtol=1e-15))


def batch_gradient_test(ndims = 2):
    ################################
    # Preprocess
    ################################
    batch_size = 2
    if ndims == 2:
        elem_dims = [2,2]
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]]), \
                    np.array([[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]])]
        
        elems_list = [np.array([[elem_dims[0],0,1,4],[elem_dims[0],2,4,1],[elem_dims[0],2,3,4],[elem_dims[0],0,4,3]], dtype=np.int64), \
                    np.array([[elem_dims[1],0,1,2],[elem_dims[1],0,2,3]], dtype=np.int64)]
    else:
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0, 0.0],[1.0,0.0, 0.0],[1.0,1.0, 0.0],[0.0,1.0, 0.0],[0.5,0.5, 0.0]]), \
                      np.array([[1.0,0.0, 0.0],[1.0,1.0, 0.0],[0.0,1.0, 0.0],[1.0,0.0, 1.0]])]
        elem_dims = [2,3]
        elems_list = [np.array([[elem_dims[0],0,1,4],[elem_dims[0],2,4,1],[elem_dims[0],2,3,4],[elem_dims[0],0,4,3]], dtype=np.int64), \
                    np.array([[elem_dims[1],0,1,2, 3]], dtype=np.int64)]
    max_nnodes = max([nodes.shape[0] for nodes in nodes_list])

    # batch by ndims by nnodes
    grids = np.zeros((batch_size,ndims,max_nnodes))
    for b in range(batch_size):
        grids[b,:,:nodes_list[b].shape[0]] = nodes_list[b].T


    directed_edges_list, edge_weights_list = [], []
    for b in range(batch_size):
        directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes_list[b], elems_list[b], mesh_type='vertex_centered', adjacent_type='element', rcond=1e-3)
        directed_edges_list.append(directed_edges)
        edge_weights_list.append(edge_gradient_weights) 
    max_nedges = max([directed_edges.shape[0] for directed_edges in directed_edges_list])
    
    #padding with zero
    directed_edges = np.zeros((batch_size, max_nedges, 2), dtype=np.int64)
    edge_gradient_weights = np.zeros((batch_size, max_nedges, ndims))
    for b in range(batch_size):
        directed_edges[b, :directed_edges_list[b].shape[0], :] = directed_edges_list[b]
        edge_gradient_weights[b, :edge_weights_list[b].shape[0], :] = edge_weights_list[b]
    # batch_size by ndims by max_nnodes 
    grids = torch.from_numpy(grids)
    # batch_size by max_edges by 2
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    # batch_size by max_edges by ndims
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights)

    ################################
    # Construct features
    ################################
    nchannels = 5
    # features is a batch_size by nchannels by max_nnodes array, 
    # for each channel, the gradient is gradients[i, :], 
    # and the gradient is constant for all nodes
    gradients = np.random.rand(batch_size, nchannels, ndims)
    # grids = batch_size, ndims, nnodes
    features =  np.einsum('bcd,bdn->bcn', gradients, grids)
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
    features_gradients_ref = np.zeros((batch_size, nchannels * ndims, max_nnodes))
    for b in range(batch_size):
        features_gradients_ref[b,:,:nodes_list[b].shape[0]] = np.tile(gradients[b,:,:].flatten(), (nodes_list[b].shape[0],1)).T
    for i, elem_dim in enumerate(elem_dims):
        if ndims == 3 and elem_dim == 2:
            # remove the gradient in the normal direction
            features_gradients_ref[i,2::ndims,:] = 0.0
    # batch_size, nnodes, nchannels
    features = torch.from_numpy(features)  
    ##############################
    # Online computation
    ##############################
    features_gradients = compute_gradient(features, directed_edges, edge_gradient_weights)

    
    for b in range(batch_size):
        print("batch gradient[%d] error is "%b, np.linalg.norm(features_gradients[b,...]-features_gradients_ref[b,...]))

    assert(np.allclose(features_gradients-features_gradients_ref, 0.0, rtol=1e-15))
    
    print("When the point and its neighbors are on the a degenerated plane, the gradient in the normal direction is not known")





def node_measures_test():
    elem_dim = 2
    elems = np.array([[elem_dim, 0,1,2],[elem_dim, 0,2,3]])
    nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]) 
    
    assert(np.all(np.isclose(compute_node_measures(nodes, elems) - np.array([[1.0/3], [1.0/6], [1.0/3], [1.0/6]]), 0)))
    
    elem_dim = 2
    elems = np.array([[elem_dim, 0,1,2],[elem_dim, 0,2,3]])
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    assert(np.all(np.isclose(compute_node_measures(nodes, elems) - np.array([[1.0/3], [1.0/6], [1.0/3], [1.0/6]]), 0)))
    
    elem_dim = 1 
    elems = np.array([[elem_dim,0,1],[elem_dim,1,2],[elem_dim,2,3]])
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    assert(np.all(np.isclose(compute_node_measures(nodes, elems) - np.array([[0.5], [1.0], [1.0], [0.5]]), 0)))
    
    elem_dim = 3 
    elems = np.array([[elem_dim,0,1,2,4],[elem_dim,0,2,3,4]])
    nodes = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) 
    assert(np.all(np.isclose(compute_node_measures(nodes, elems) - np.array([[1.0/12.0], [1.0/24.0], [1.0/12.0], [1.0/24.0], [1.0/12.0]]), 0)))
    
    elem_dim = 3 
    elems = np.array([[elem_dim,0,1,2,3,4,5,6,7],[elem_dim,4,5,6,7,8,9,10,11]])
    nodes = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],
                      [0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0], 
                      [0.0,0.0,2.0],[1.0,0.0,2.0],[1.0,1.0,2.0],[0.0,1.0,2.0]]) 
    assert(np.all(np.isclose(compute_node_measures(nodes, elems) - np.array([[1.0/8.0], [1.0/8.0], [1.0/8.0], [1.0/8.0], [1.0/4.0], [1.0/4.0], [1.0/4.0], [1.0/4.0],[1.0/8.0], [1.0/8.0], [1.0/8.0], [1.0/8.0],]), 0)))
    


def preprocess_data_mesh_test():
    """
    Random two meshes
    Mesh 1:
            |\
            |  \
            ______
    ______________

    Mesh 2:
            ______
            |\   |
            |  \ |
            ______
    ______________
    """
    
    nodes_list = [np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]]), 
                  np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])]
    elems_list = [np.array([[1, -1, 0, 1],[1, -1, 1, 3],[2, 1, 3, 2]], dtype=int), 
                  np.array([[1, -1, 0, 1],[1, -1, 1, 3],[2, 1, 3, 2],[2, 2, 3, 4]], dtype=int)]
    
    ##########################################################
    # mesh_type='vertex_centered'
    ########################################################## 
    mesh_type = 'vertex_centered'
    # features are coordinates
    features_list = [np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]]), 
                  np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])]
    
    ##########################################################
    ##  adjacent_type='element'
    ########################################################## 
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type=mesh_type, adjacent_type='element')
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)

    # compute_gradient require features = float[batch_size, in_channels, nnodes]
    features_gradients = compute_gradient(torch.from_numpy(features).permute(0,2,1), torch.from_numpy(directed_edges.astype(np.int64)), torch.from_numpy(edge_gradient_weights))

    assert(np.linalg.norm(nnodes - np.array([4, 5])) == 0)
    assert(np.all(np.isclose(node_mask - np.stack((np.array([[1],[1],[1],[1],[0]]), 
                                                   np.array([[1],[1],[1],[1],[1]])), axis=0), 0)))
    assert(np.all(np.isclose(nodes - np.stack((np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 0.0]]), 
                                               np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])), axis=0), 0)))
    assert(np.all(np.isclose(features - np.stack((np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 0.0]]), 
                                                  np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])), axis=0), 0)))
    
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f1_z f2_x f2_y,.....
    # the first node has dimension 1
    assert(np.all(np.isclose(features_gradients.permute(0,2,1) - np.stack((np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 
                                                                           np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), axis=0), 0)))
    
    assert(np.all(np.isclose(node_measures - np.stack((np.array([[1.0/2, 0],[1.0, 1.0/6],[0, 1.0/6],[1.0/2, 1.0/6],[0, 0]]), 
                                                       np.array([[1.0/2, 0],[1.0, 1.0/6],[0, 1.0/3],[1.0/2, 1.0/3],[0, 1.0/6]])), axis=0), 0)))
    assert(np.all(np.isclose(node_weights - np.stack((np.array([[1.0/4, 0],[1.0/2, 1.0/3],[0, 1.0/3],[1.0/4, 1.0/3],[0, 0]]), 
                                                       np.array([[1.0/4, 0],[1.0/2, 1.0/6],[0, 1.0/3],[1.0/4, 1.0/3],[0, 1.0/6]])), axis=0)/2.0, 0)))
    
    assert(np.all(np.isclose(node_equal_measures - np.stack((np.array([[2.0/3, 0],[2.0/3, 1.0/6],[0, 1.0/6],[2.0/3, 1.0/6],[0, 0]]), 
                                                       np.array([[2.0/3, 0],[2.0/3, 1.0/4],[0, 1.0/4],[2.0/3, 1.0/4],[0, 1.0/4]])), axis=0), 0)))
    assert(np.all(np.isclose(node_equal_weights - np.stack((np.array([[1.0/3, 0],[1.0/3, 1.0/3],[0, 1.0/3],[1.0/3, 1.0/3],[0, 0]]), 
                                                       np.array([[1.0/3, 0],[1.0/3, 1.0/4],[0, 1.0/4],[1.0/3, 1.0/4],[0, 1.0/4]])), axis=0)/2.0, 0)))
    
    ##########################################################
    ##  adjacent_type='edge'
    ########################################################## 
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type=mesh_type, adjacent_type='edge')
    # compute_gradient require features = float[batch_size, in_channels, nnodes]
    features_gradients = compute_gradient(torch.from_numpy(features).permute(0,2,1), torch.from_numpy(directed_edges.astype(np.int64)), torch.from_numpy(edge_gradient_weights))
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f1_z f2_x f2_y,.....
    # the first node has dimension 1
    assert(np.all(np.isclose(features_gradients.permute(0,2,1) - np.stack((np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 
                                                                           np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), axis=0), 0)))
    
    
    ##########################################################
    # mesh_type='cell_centered'
    ########################################################## 
    mesh_type='cell_centered'
    # features are coordinates
    features_list = [np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0]]), 
                     np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[2.0/3, 2.0/3, 0.0]])]
    
    ##########################################################
    ## adjacent_type='node'
    ########################################################## 
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type=mesh_type, adjacent_type='node')
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)

    # compute_gradient require features = float[batch_size, in_channels, nnodes]
    features_gradients = compute_gradient(torch.from_numpy(features).permute(0,2,1), torch.from_numpy(directed_edges.astype(np.int64)), torch.from_numpy(edge_gradient_weights))

    assert(np.linalg.norm(nnodes - np.array([3, 4])) == 0)
    assert(np.all(np.isclose(node_mask - np.stack((np.array([[1],[1],[1],[0]]), 
                                                   np.array([[1],[1],[1],[1]])), axis=0), 0)))
    assert(np.all(np.isclose(nodes - np.stack((np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[0.0, 0.0, 0.0]]), 
                                               np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[2.0/3, 2.0/3, 0.0]])), axis=0), 0)))
    assert(np.all(np.isclose(features - np.stack((np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[0.0, 0.0, 0.0]]), 
                                                  np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[2.0/3, 2.0/3, 0.0]])), axis=0), 0)))
    
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f1_z f2_x f2_y,.....
    # the first node has dimension 1
    assert(np.all(np.isclose(features_gradients.permute(0,2,1) - np.stack((np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 
                                                                           np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), axis=0), 0)))
    
    assert(np.all(np.isclose(node_measures - np.stack((np.array([[1.0, 0],[1.0, 0.0],[0, 1.0/2],[0, 0]]), 
                                                       np.array([[1.0, 0],[1.0, 0.0],[0, 1.0/2],[0, 1.0/2]])), axis=0), 0)))
    assert(np.all(np.isclose(node_weights - np.stack((np.array([[1.0/2, 0],[1.0/2, 0.0],[0, 1.0], [0, 0]]), 
                                                       np.array([[1.0/2, 0],[1.0/2, 0.0],[0, 1.0/2], [0, 1.0/2]])), axis=0)/2.0, 0)))
    
    assert(np.all(np.isclose(node_equal_measures - np.stack((np.array([[1.0, 0],[1.0, 0.0],[0, 1.0/2], [0, 0]]), 
                                                       np.array([[1.0, 0],[1.0, 0.0],[0, 1.0/2],[0, 1.0/2]])), axis=0), 0)))
    assert(np.all(np.isclose(node_equal_weights - np.stack((np.array([[1.0/2, 0],[1.0/2, 0.0],[0, 1.0], [0, 0]]), 
                                                       np.array([[1.0/2, 0],[1.0/2, 0.0],[0, 1.0/2], [0, 1.0/2]])), axis=0)/2.0, 0)))
    
    
    ##########################################################
    ## adjacent_type='face'
    ########################################################## 
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type=mesh_type, adjacent_type='faces')
    # compute_gradient require features = float[batch_size, in_channels, nnodes]
    features_gradients = compute_gradient(torch.from_numpy(features).permute(0,2,1), torch.from_numpy(directed_edges.astype(np.int64)), torch.from_numpy(edge_gradient_weights))

    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f1_z f2_x f2_y,.....
    # the first node has dimension 1
    assert(np.all(np.isclose(features_gradients.permute(0,2,1) - np.stack((np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 
                                                                           np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), axis=0), 0)))
    

    ##########################################################
    ## mesh_type='cell_centered' , adjacent_type='edge'
    ########################################################## 
    """
    Test mesh_type='cell_centered' , adjacent_type='edge'
    1) when there is no adjacent node, the gradient is 0
    2) when the edge is the boundary on the element, the gradient is the normal gradient
    Mesh 1:
            3|-----|4
             |     |
     0______ 1______2
    ______________

    Mesh 2:
             
            3|\   
             |  \ 
    0______ 1______2
             |  /
            4|/
    ______________
    """
    mesh_type='cell_centered'
    nodes_list = [np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0]]), 
                  np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, -1.0, 0.0]])]
    elems_list = [np.array([[1, -1, -1, 0, 1],[1, -1, -1, 1, 2],[2, 1, 2, 4, 3]], dtype=int), 
                  np.array([[1, -1,  0, 1],[1, -1,  1, 2],[2, 1, 2, 3],[2, 1, 2, 4]], dtype=int)]
    # features are coordinates
    features_list = [np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[0.5, 0.5, 0.0]]), 
                     np.array([[-0.5, 0.0, 0.0],[0.5, 0.0, 0.0],[1.0/3, 1.0/3, 0.0],[1.0/3, -1.0/3, 0.0]])]
    
    
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type=mesh_type, adjacent_type='edge')
    # compute_gradient require features = float[batch_size, in_channels, nnodes]
    
    features_gradients = compute_gradient(torch.from_numpy(features).permute(0,2,1), torch.from_numpy(directed_edges.astype(np.int64)), torch.from_numpy(edge_gradient_weights))

    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f1_z f2_x f2_y,.....
    # the first node has dimension 1
    assert(np.all(np.isclose(features_gradients.permute(0,2,1) - np.stack((np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 
                                                                           np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), axis=0), 0)))
    
if __name__ == "__main__":
    node_measures_test()
    adjacent_list_test()
    
    test_convert_structured_data()

    print("2d gradient test")
    gradient_test(ndims = 2)
    batch_gradient_test(ndims=2)
    print("3d gradient test")
    gradient_test(ndims = 3)
    batch_gradient_test(ndims=3)

    preprocess_data_mesh_test()
