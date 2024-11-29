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

def compute_weight_per_elem_(points, elem_dim):
    '''
    Compute element weight (length, area or volume)
    for 2-point  element, compute its length
    for 3-point  element, compute its area
    for 4-point  element, compute its area if elem_dim=2; compute its volume if elem_dim=3
    equally assign it to its nodes
    
        Parameters: 
            points : float[npoints, ndims]
            elem_dim : int
    
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
        if elem_dim == 2:
            s = compute_triangle_area_(points[:3,:]) + compute_triangle_area_(points[1:,:])
        elif elem_dim == 3:
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("elem dim ", elem_dim,  "is not recognized")
    else:   
        raise ValueError("npoints ", npoints,  "is not recognized")
    return s


def compute_node_weights(nodes, elems, weight_type):
    '''
    Compute node weights normalized by the total weight 
    (length, area or volume for each node), 
    For each element, compute its length, area or volume, 
    equally assign it to its nodes.
    
    When weight_type is None, all nodes are of equal weight (1/N)

    # TODO compute as FEM mass matrix

        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            type  : "area" or None

            * When node_weight_type is None, all nodes are of equal weight, S/N, S is the total weight (i.e., area)

            # TODO set 1/N

            
        Return :
            weights : float[nnodes]
    '''
    nnodes = nodes.shape[0]
    weights = np.zeros(nnodes)
    if weight_type is None:
        weights = 1.0/nnodes
    else:
        for elem in elems:
            elem_dim, e = elem[0], elem[1:]
            e = e[e >= 0]
            ne = len(e)
            if ne != 0:
                s = compute_weight_per_elem_(nodes[e, :], elem_dim)
                weights[e] += s/ne

        weights /= sum(weights)

    return weights

def pinv(a, rrank, rcond=1e-3):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

        Parameters:
            a : float[M, N]
                Matrix to be pseudo-inverted.
            rrank : int
                Maximum rank
            rcond : float, optional
                Cutoff for small singular values.
                Singular values less than or equal to
                ``rcond * largest_singular_value`` are set to zero.
                Default: ``1e-3``.

        Returns:
            B : float[N, M]
                The pseudo-inverse of `a`. 

    """
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    # discard small singular values
    cutoff = rcond * s[0]
    large = s > cutoff
    large[rrank:] = False
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0

    res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
    return res


def compute_edge_gradient_weights(nodes, elems, rcond = 1e-3):
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
    
    When these points are on a degerated plane or surface, the gradient towards the 
    normal direction is 0.


        Parameters:  
            nodes : float[nnodes, ndims]
                    elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                            The first entry is elem_dim, the dimensionality of the element.
                            The elems array can have some padding numbers, for example, when
                            we have both line segments and triangles, the padding values are
                            -1 or any negative integers.
            rcond : float, truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value
            

        Return :

            directed_edges : int[nedges,2]
            edge_gradient_weights   : float[nedges, ndims]

            * the directed_edges (and adjacent list) include all node pairs that share the element
            
    '''

    nnodes, ndims = nodes.shape
    if ndims == 1:
        nelems = elems.shape[0]
    else:
        nelems, _ = elems.shape
    # Initialize adjacency list as a list of sets
    # Use a set to store unique directed edges
    adj_list = [set() for _ in range(nnodes)]
    
    # Initialize node_dims to store the maximum dimensionality at that node
    node_dims = np.zeros(nnodes, dtype=int)
    # Loop through each element and create directed edges
    for elem in elems:
        elem_dim, e = elem[0], elem[1:]
        node_dims[e] = np.maximum(node_dims[e], elem_dim)
        e = e[e >= 0]
        nnodes_per_elem = len(e)
        for i in range(nnodes_per_elem):
            # Add each node's neighbors to its set
            adj_list[e[i]].update([e[j] for j in range(nnodes_per_elem) if j != i])
    
    directed_edges = []
    edge_gradient_weights = [] 
    '''
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b,:] - nodes[a,:]
            directed_edges.append([a,b])
        edge_gradient_weights.append(pinv(dx, rrank=node_dims[a], rcond=rcond).T)
    '''
    for a in range(nnodes):
        if len(adj_list[a]) != 0:
            dx = np.zeros((len(adj_list[a]), ndims))
            for i, b in enumerate(adj_list[a]):
                dx[i, :] = nodes[b,:] - nodes[a,:]
                directed_edges.append([a,b])
            edge_gradient_weights.append(pinv(dx, rrank=node_dims[a], rcond=rcond).T)
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
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            features_list  : list of float[nnodes, nfeatures]
            node_weight_type : "length", "area", "volumn", None

            * When node_weight_type is None, all nodes are of equal weight, S/N, S is the total weight (i.e., area)


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
            feature_include_coords :  boolean, whether treating coordinates as features, if coordinates
                                      are treated as features, they are concatenated at the end

        Return :  
            nodes_list :     list of float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.

                    `ndims = 2`
                    ================================================================
                    3 - 4 - 5
                    |   |   | square grid (nnodes_per_elem = 4)
                    0 - 1 - 2
                    ================================================================
                    3 — 4 — 5
                    | / | / | triangle grid (nnodes_per_elem = 3)
                    0 - 1 - 2
                    ================================================================
            features_list  : list of float[nnodes, nfeatures]
    '''
    ndims = len(coords_list)
    if ndims == 1:
        coordx = coords_list[0]
        ndata, nx  = coords_list[0].shape
        nnodes, nelems = nx, nx-1
        nodes = coordx.reshape((ndata, nnodes, 1))
        if feature_include_coords :
            nfeatures = features.shape[-1] + ndims
            features = np.concatenate((features.reshape((ndata, nnodes, -1)), nodes), axis=-1)
        else :
            nfeatures = features.shape[-1]
            features = features.reshape((ndata, nnodes, -1))
        
        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx-1):
            ie = i
            elems[i, :] = 2, i, i+1
        
        elems = np.tile(elems, (ndata, 1, 1))
        '''
        elems = -np.ones((ndata, nelems, nnodes_per_elem + 1), dtype=int)
        for n in range(ndata):
            no = 0
            i, j = 0, 1
            while i < nx-1:
                while j < nx-1 and nodes[n, j, 0] == 0:
                    j += 1
                if j == nx-1:
                    break
                else:
                    elems[n, no, :] = 2, i, j
                    no += 1
                    i, j = j, j+1
        '''
        nodes_list = [nodes[i,...] for i in range(ndata)]
        elems_list = [elems[i,...] for i in range(ndata)]
        features_list = [features[i,...] for i in range(ndata)]
        
    elif ndims == 2:
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

        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx-1):
            for j in range(ny-1):
                ie = i*(ny-1) + j 
                if nnodes_per_elem == 4:
                    elems[ie, :] = 2, i*ny+j, i*ny+j+1, (i+1)*ny+j+1, (i+1)*ny+j
                else:
                    elems[2*ie, :]   = 2, i*ny+j, i*ny+j+1, (i+1)*ny+j+1
                    elems[2*ie+1, :] = 2, i*ny+j, (i+1)*ny+j+1, (i+1)*ny+j

        elems = np.tile(elems, (ndata, 1, 1))

        nodes_list = [nodes[i,...] for i in range(ndata)]
        elems_list = [elems[i,...] for i in range(ndata)]
        features_list = [features[i,...] for i in range(ndata)]
    
    else:
        print(f"dim {ndims} is not supported")

    return nodes_list, elems_list, features_list  


def test_node_weights():
    elem_dim = 2
    elems = np.array([[elem_dim, 0,1,2],[elem_dim, 0,2,3]])
    nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]) 
    type = "area"
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/3, 1.0/6, 1.0/3, 1.0/6])) < 1e-15)
    type = None
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/4, 1.0/4, 1.0/4, 1.0/4])) < 1e-15)

    elem_dim = 2
    elems = np.array([[elem_dim, 0,1,2],[elem_dim, 0,2,3]])
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    type = "area"
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/3, 1.0/6, 1.0/3, 1.0/6])) < 1e-15)
    type = None
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/4, 1.0/4, 1.0/4, 1.0/4])) < 1e-15)
    
    elem_dim = 1 
    elems = np.array([[elem_dim,0,1],[elem_dim,1,2],[elem_dim,2,3]])
    nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]) 
    type = "area"
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([0.5, 1.0, 1.0, 0.5])) < 1e-15)
    type = None
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([0.75, 0.75, 0.75, 0.75])) < 1e-15)
    
    elem_dim = 3 
    elems = np.array([[elem_dim,0,1,2,4],[elem_dim,0,2,3,4]])
    nodes = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) 
    type = "area"
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/12.0, 1.0/24.0, 1.0/12.0, 1.0/24.0, 1.0/12.0])) < 1e-15)
    type = None
    assert(np.linalg.norm(compute_node_weights(nodes, elems, type) - np.array([1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0])) < 1e-15)
    


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
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_weights - np.tile(np.array([1.0/4,1.0/2,1.0/4,1.0/4,1.0/2,1.0/4])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)

    features = np.zeros((ndata, Npx, Npy, 1)) # all zeros data
    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(grid_y, (ndata, 1, 1))], features, nnodes_per_elem = 3, feature_include_coords = True)
    assert(np.linalg.norm(elems_list[0] - np.array([[elem_dim,0,1,4],[elem_dim,0,4,3],[elem_dim,1,2,5],[elem_dim,1,5,4]])) == 0)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights  = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    assert(np.linalg.norm(nnodes - Npx * Npy) == 0)
    assert(np.linalg.norm(node_mask - 1) == 0)
    assert(np.linalg.norm(nodes - np.tile(np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]), (ndata, 1, 1))) == 0)
    assert(np.linalg.norm(node_weights - np.tile(np.array([1.0/3,1.0/2,1.0/6,1.0/6,1.0/2,1.0/3])[:,np.newaxis], (ndata,1,1))) == 0)
    assert(np.linalg.norm(features - np.concatenate((np.zeros((ndata, Npx*Npy, 1)), nodes), axis=2)) == 0)


if __name__ == "__main__":
    test_node_weights()
    test_convert_structured_data()
