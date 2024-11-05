import numpy as np

def compute_triangle_area_(points):
    ab = points[1, :] - points[0,:]
    ac = points[2, :] - points[0,:]
    cross_product = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_product)


def compute_tetrahedron_volume_(points):
    ab = points[1, :] - points[0,:]
    ac = points[2, :] - points[0,:]
    ad = points[3, :] - points[0,:]
    # Calculate the scalar triple product
    volume = abs(np.dot(np.cross(ab, ac), ad)) / 6
    return volume

def compute_weight_per_elem_(points, weight_type):
    '''
    Compute element weight (length, area or volume)
    for 2-point  element, compute its length
    for 3-point  element, compute its area
    for 4-point  element, compute its area if weight_type="area"; compute its volume if weight_type="volume"
    equally assign it to its nodes
    
        Parameters: 
            points : float[npoints, ndims]
            weight_type : string, "length", "area", or "volume"
    
        Returns:
            s : float
    '''
    
    npoints, ndims = points.shape
    if npoints == 2: 
        s = np.linalg.norm(points[0, :] - points[1, :])
    elif npoints == 3:
        s = compute_triangle_area_(points)
    elif npoints == 4:
        assert(npoints == 3 or npoints == 4)
        if weight_type == "area":
            s = compute_triangle_area_(points[:3,:]) + compute_triangle_area_(points[1:,:])
        elif weight_type == "volume":
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("weight type ", weight_type,  "is not recognized")
    else:   
        raise ValueError("npoints ", npoints,  "is not recognized")
    return s


def compute_node_weights(nodes, elems, weight_type):
    '''
    Compute node weights (length, area or volume for each node), 
    for each element, compute its length, area or volume, 
    equally assign it to its nodes.
    
    * When weight_type is None, all nodes are of equal weight

    # TODO compute as FEM mass matrix

        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem]
            type  : string, "length", "area", "volume", or None

            * When node_weight_type is None, all nodes are of equal weight, S/N

            # TODO set 1/N

            * The elems array can have some padding numbers, for example, when
            we have both line segments and triangles, the padding values are
            -1 or any negative integers.
        Return :
            weights : float[nnodes]
    '''
    nnodes = nodes.shape[0]
    weights = np.zeros(nnodes)
    for e in elems:
        e = e[e >= 0]
        ne = len(e)
        s = compute_weight_per_elem_(nodes[e, :], weight_type)
        weights[e] += s/ne 

    if weights is None:
        weights = sum(weights)/nnodes
    return weights


def compute_edge_gradient_weights(nodes, elems, rcond = 10.0):
    '''
    Compute weights for gradient computation  
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)     =          :
       :                                :
    xj - x                        f(xj) - f(x)
    
    in matrix form   dx  nable f(x)   = df.
    
    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    We store directed edges (x, x1), (x, x2), ..., (x, xj)
    And its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]
    Then the gradient can be efficiently computed with scatter_add
    
    TODO: what will happen, when these points are a degerated plane?


        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem]
            rcond : float, truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value
            
            * The elems array can have some padding numbers, for example, when
            we have both line segments and triangles, the padding values are
            -1 or any negative integers.

        Return :

            directed_edges : int[nedges,2]
            edge_gradient_weights   : float[nedges, ndims]

            * the directed_edges (and adjacent list) include all node pairs that share the element
            
    '''

    nnodes, ndims = nodes.shape
    nelems, _ = elems.shape
    # Initialize adjacency list as a list of sets
    adj_list = [set() for _ in range(nnodes)]
    # Use a set to store unique directed edges

    # Loop through each element and create directed edges
    for e in elems:
        e = e[e >= 0]
        nnodes_per_elem = len(e)
        for i in range(nnodes_per_elem):
            # Add each node's neighbors to its set
            adj_list[e[i]].update([e[j] for j in range(nnodes_per_elem) if j != i])
    
    directed_edges = []
    edge_gradient_weights = [] 
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b,:] - nodes[a,:]
            directed_edges.append([a,b])
        edge_gradient_weights.append(np.linalg.pinv(dx, rcond=rcond).T)
        
    directed_edges = np.array(directed_edges, dtype=int)
    edge_gradient_weights = np.concatenate(edge_gradient_weights, axis=0)
    return directed_edges, edge_gradient_weights, adj_list



def preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area"):
    '''
    Compute node weights (length, area or volume for each node), 
    for each element, compute its length, area or volume, 
    equally assign it to its nodes.

        Parameters:  
            nodes_list :     list of float[nnodes, ndims]
            elems_list :     list int[nelems, max_num_of_nodes_per_elem]
            features_list  : list of float[nnodes, nfeatures]
            node_weight_type : "length", "area", "volumn", None

            * When node_weight_type is None, all nodes are of equal weight, S/N

            * The elems array can have some padding numbers, for example, when
            we have both line segments and triangles, the padding values are
            -1 or any negative integers. 

        Return :
            nnodes         :  int
            node_mask      :  int[ndata, max_nnodes, 1]               (1 for node, 0 for padding)
            nodes          :  float[ndata, max_nnodes, ndims]      (padding 0)
            node_weights   :  float[ndata, max_nnodes, 1]               (padding 0)   
            features       :  float[ndata, max_nnodes, nfeatures]  (padding 0)   
            directed_edges :  float[ndata, max_nedges, 2]          (padding 0)   
            edge_gradient_weights   :  float[ndata, max_nedges, ndims]      (padding 0)  
    '''
    ndata = len(nodes_list)
    ndims, nfeatures = nodes_list[0].shape[1], features_list[0].shape[1]
    nnodes = np.array([nodes.shape[0] for nodes in nodes_list], dtype=int)
    max_nnodes = max(nnodes)

    print("Preprocessing data : computing node_mask")
    node_mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        node_mask[i,:nnodes[i], :] = 1

    print("Preprocessing data : computing nodes")
    nodes = np.zeros((ndata, max_nnodes, ndims))
    for i in range(ndata):
        nodes[i,:nnodes[i], :] = nodes_list[i]
    
    print("Preprocessing data : computing node_weights")
    node_weights = np.zeros((ndata, max_nnodes, 1))
    for i in range(ndata):
        node_weights[i,:nnodes[i], 0] = compute_node_weights(nodes_list[i], elems_list[i], node_weight_type)

    print("Preprocessing data : computing features")
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        features[i,:nnodes[i],:] = features_list[i] 

    print("Preprocessing data : computing directed_edges and edge_gradient_weights")
    directed_edges_list, edge_gradient_weights_list = [], []
    for i in range(ndata):
        directed_edges, edge_gradient_weights, edge_adj_list = compute_edge_gradient_weights(nodes_list[i], elems_list[i])
        directed_edges_list.append(directed_edges) 
        edge_gradient_weights_list.append(edge_gradient_weights)   
    nedges = np.array([directed_edges.shape[0] for directed_edges in directed_edges_list])
    max_nedges = max(nedges)
    directed_edges, edge_gradient_weights = np.zeros((ndata, max_nedges, 2),dtype=int), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        directed_edges[i,:nedges[i],:] = directed_edges_list[i]
        edge_gradient_weights[i,:nedges[i],:] = edge_gradient_weights_list[i] 

    return nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights 


def convert_structured_data(coords_list, features, nnodes_per_elem = 3, feature_include_coords = True):
    '''
    Convert structured data, to unstructured data
                    ny-1                                                                  ny-1   2ny-1
                    ny-2                                                                  ny-2    .
                    .                                                                       .     .
    y direction     .          nodes are ordered from left to right/bottom to top           .     .
                    .                                                                       .     .
                    1                                                                       1     ny+1
                    0                                                                       0     ny
                        0 - 1 - 2 - ... - nx-1   (x direction)

        Parameters:  
            coords_list            :  list of ndims float[nnodes, nx, ny], for each dimension
            features               :  float[nelems, nx, ny, nfeatures]
            nnodes_per_elem        :  int, nnodes_per_elem = 3: triangle mesh; nnodes_per_elem = 4: quad mesh
            feature_include_coords :  boolean, whether treating coordinates as features

        Return :  
            nodes_list :     list of float[nnodes, ndims]
            elems_list :     list int[nelems, max_num_of_nodes_per_elem]
            features_list  : list of float[nnodes, nfeatures]
    '''
    print("convert_structured_data so far only supports 2d problems")
    ndims = len(coords_list)
    assert(ndims == 2) 
    coordx, coordy = coords_list
    ndata, nx, ny  = coords_list[0].shape
    nnodes, nelems = nx*ny, (nx-1)*(ny-1)*(5 - nnodes_per_elem)
    nodes = np.stack((coordx.reshape((ndata, nnodes)), coordy.reshape((ndata, nnodes))), axis=2)
    if feature_include_coords :
        nfeatures = features.shape[-1] + ndims
        features = np.concatenate((features.reshape((ndata, nnodes, -1)), nodes), axis=-1)
    else :
        nfeatures = features.shape[-1]
        features = features.reshape((ndata, nnodes, -1))

    elems = np.zeros((nelems, nnodes_per_elem), dtype=int)
    for i in range(nx-1):
        for j in range(ny-1):
            ie = i*(ny-1) + j 
            if nnodes_per_elem == 4:
                elems[ie, :] = i*ny+j, i*ny+j+1, (i+1)*ny+j+1, (i+1)*ny+j
            else:
                elems[2*ie, :]   = i*ny+j, i*ny+j+1, (i+1)*ny+j+1
                elems[2*ie+1, :] = i*ny+j, (i+1)*ny+j+1, (i+1)*ny+j

    elems = np.tile(elems, (ndata, 1, 1))

    nodes_list = [nodes[i,...] for i in range(ndata)]
    elems_list = [elems[i,...] for i in range(ndata)]
    features_list = [features[i,...] for i in range(ndata)]

    return nodes_list, elems_list, features_list  


def test_node_weights():
    elems = np.array([[0,1,2],[0,2,3]])
    type = "area"
    nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]) 
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/3, 1.0/6, 1.0/3, 1.0/6])) < 1e-15)

    elems = np.array([[0,1,2],[0,2,3]])
    type = "area"
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/3, 1.0/6, 1.0/3, 1.0/6])) < 1e-15)
    

    elems = np.array([[0,1],[1,2],[2,3]])
    type = "length"
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([0.5, 1.0, 1.0, 0.5])) < 1e-15)
    

    elems = np.array([[0,1,2,4],[0,2,3,4]])
    type = "volume"
    nodes = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) 
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/12.0, 1.0/24.0, 1.0/12.0, 1.0/24.0, 1.0/12.0])) < 1e-15)
    


def test_convert_structured_data():
    Lx, Ly = 1.0, 2.0
    Npx, Npy = 2, 3
    grid_1d_x, grid_1d_y = np.linspace(0, Lx, Npx), np.linspace(0, Ly, Npy)
    grid_x, grid_y = np.meshgrid(grid_1d_x, grid_1d_y)
    grid_x, grid_y = grid_x.T, grid_y.T
    ndata = 2
    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 4, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[0,1,4,3],[1,2,5,4]])) == 0)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_weights - np.tile(np.array([1.0/4,1.0/2,1.0/4,1.0/4,1.0/2,1.0/4])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)

    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 3, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])) == 0)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_weights - np.tile(np.array([1.0/3,1.0/2,1.0/6,1.0/6,1.0/2,1.0/3])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)


if __name__ == "__main__":
    test_node_weights()
    test_convert_structured_data()
