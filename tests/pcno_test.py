import numpy as np
import torch
from pcno.geo_utility import convert_structured_data, compute_node_weights, preprocess_data, compute_node_measures
from pcno.pcno import compute_edge_gradient_weights, compute_gradient

#####################################################################
# PCNO CODE TESTS
#####################################################################




def test_convert_structured_data():
    elem_dim=2
    Lx, Ly = 1.0, 2.0
    Npx, Npy = 2, 3
    grid_1d_x, grid_1d_y = np.linspace(0, Lx, Npx), np.linspace(0, Ly, Npy)
    grid_x, grid_y = np.meshgrid(grid_1d_x, grid_1d_y)
    grid_x, grid_y = grid_x.T, grid_y.T
    ndata = 2
    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 4, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,1,4,3],[elem_dim,1,2,5,4]])) == 0)
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list)
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
    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list)
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_measures - np.tile(np.array([1.0/3,1.0/2,1.0/6,1.0/6,1.0/2,1.0/3])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)




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
    directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes, elems, rcond=1e-3)
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
        directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes_list[b], elems_list[b], rcond=1e-3)
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





def test_node_measures():
    elem_dim = 2
    elems = np.array([[elem_dim, 0,1,2],[elem_dim, 0,2,3]])
    nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]) 
    print(compute_node_measures(nodes, elems))
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
    


def test_preprocess_data():
    """
    Two meshes
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
    # features include coordinates
    features_list = [np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]]), 
                  np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])]
    
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
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
    
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
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
    



if __name__ == "__main__":

    test_node_measures()
    test_convert_structured_data()

    print("2d gradient test")
    gradient_test(ndims = 2)
    batch_gradient_test(ndims=2)
    print("3d gradient test")
    gradient_test(ndims = 3)
    batch_gradient_test(ndims=3)

    test_preprocess_data()


