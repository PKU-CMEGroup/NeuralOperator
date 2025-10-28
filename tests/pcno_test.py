import numpy as np
import torch
from pcno.geo_utility import convert_structured_data, compute_node_weights, preprocess_data_mesh, compute_node_measures, compute_edge_gradient_weights
from pcno.geo_utility import compute_elem_adjacent_list, compute_node_adjacent_list, sample_close_node_pairs
from pcno.pcno import compute_gradient, compute_Fourier_modes, compute_Fourier_bases
from pcno.pcno import SpectralConv, SpectralConvLocal
#####################################################################
# PCNO CODE TESTS
#####################################################################




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
    
def speconv_test():
    ndim, ndata = 2, 2
    nodes = np.stack([np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 1.0],[1.0, 0.0],[0.0, 0.0]]), 
                      np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 1.0],[1.0, 0.0],[0.5, 0.5]])], axis=0)
    nnodes = np.array([4,5]) 
    # nmeasure = 2
    node_weights = np.repeat(np.stack([np.array([1.0,1.0,2.0,3.0,0.0]), 
                             np.array([1.0,1.0,2.0,3.0,4.0])], axis=0)[:, :, np.newaxis], 2, axis=2)
    dist_threshold = 10.0
    max_nedges = 100
    directed_edges, nedges, directed_edge_node_weights = sample_close_node_pairs(nodes, nnodes, node_weights, dist_threshold, max_nedges)
    # assert np.array_equal(nedges, [[12, 12],[20,20]]), f"Expected [12, 20], but got {nedges}"
    
    kx_max, ky_max = 12, 12
    Lx, Ly = 1.0, 2.0
    modes = compute_Fourier_modes(ndim, [kx_max, ky_max, kx_max, ky_max], [Lx, Ly, Lx, Ly])
    modes = torch.tensor(modes, dtype=torch.float32)
    nodes = torch.tensor(nodes, dtype=torch.float32)
    node_weights = torch.tensor(node_weights, dtype=torch.float32)
    directed_edges = torch.tensor(directed_edges, dtype=torch.int64)
    directed_edge_node_weights = torch.tensor(directed_edge_node_weights, dtype=torch.float32)
    

    bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, modes) 
    wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
    wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
    wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
            
    in_channels, out_channels = 4, 4
     
    speconv = SpectralConv(in_channels, out_channels, modes)
    speconvlocal = SpectralConvLocal(in_channels, out_channels, modes)
    speconvlocal.weights_c, speconvlocal.weights_s, speconvlocal.weights_0 = speconv.weights_c, speconv.weights_s, speconv.weights_0
    x = torch.rand(ndata, in_channels, max(nnodes)) # float[batch_size, in_channels, nnodes]
    
    x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
    x2 = speconvlocal(x, bases_c, bases_s, bases_0, directed_edges, directed_edge_node_weights)
    diff = x1 - x2
    error = 0.0
    for i in range(ndata):
        error += np.linalg.norm(diff[i, :, :nnodes[i]].detach().numpy())/np.linalg.norm(x1[i, :, :nnodes[i]].detach().numpy())
    assert(error < 1.0e-6), f"Error is {error}"
    

# def speconv_test():
#     ndim = 2
#     nodes = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 1.0],[1.0, 0.0]]).reshape((1,-1,ndim))
#     nnodes = np.array([4]) 
#     # nmeasure = 2
#     # node_weights = np.array([1.0,1.0,2.0,3.0]).reshape((1,-1,1))
#     node_weights = np.array([1.0,1.0,1.0,1.0]).reshape((1,-1,1))
#     dist_threshold = 10.0
#     max_nedges = 100
#     directed_edges, nedges, directed_edge_node_weights = sample_close_node_pairs(nodes, nnodes, node_weights, dist_threshold, max_nedges)
#     # assert np.array_equal(nedges, [[12]]), f"Expected [12, 20], but got {nedges}"
    
#     kx_max, ky_max = 2, 2
#     Lx, Ly = 1.0, 2.0
#     modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
#     modes = torch.tensor(modes, dtype=torch.float32)
#     nodes = torch.tensor(nodes, dtype=torch.float32)
#     node_weights = torch.tensor(node_weights, dtype=torch.float32)
#     directed_edges = torch.tensor(directed_edges, dtype=torch.int64)
#     directed_edge_node_weights = torch.tensor(directed_edge_node_weights, dtype=torch.float32)
    

#     bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, modes) 
#     wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
#     wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
#     wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
            
#     in_channels, out_channels = 1, 1
     
#     speconv = SpectralConv(in_channels, out_channels, modes)
#     speconvlocal = SpectralConvLocal(in_channels, out_channels, modes)
#     speconvlocal.weights_c, speconvlocal.weights_s, speconvlocal.weights_0 = speconv.weights_c, speconv.weights_s, speconv.weights_0
#     x = torch.rand(1, in_channels, 4) # float[batch_size, in_channels, nnodes]
    
#     x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
#     x2 = speconvlocal(x, bases_c, bases_s, bases_0, directed_edges, directed_edge_node_weights)
#     print("x1 = ", x1)
#     print("x2 = ", x2)
#     print(x1 - x2)
    
if __name__ == "__main__":
    speconv_test()
    # node_measures_test()
    # adjacent_list_test()
    
    # test_convert_structured_data()

    # print("2d gradient test")
    # gradient_test(ndims = 2)
    # batch_gradient_test(ndims=2)
    # print("3d gradient test")
    # gradient_test(ndims = 3)
    # batch_gradient_test(ndims=3)

    # preprocess_data_mesh_test()


