import numpy as np
from tqdm import tqdm

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

def compute_measure_per_elem_(points, elem_dim):
    '''
    Compute element measure (length, area or volume)
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


def compute_node_measures(nodes, elems):
    '''
    Compute node measures  (separate length, area and volume ... for each node), 
    For each element, compute its length, area or volume s, 
    equally assign it to its ne nodes (measures[:] += s/ne).

        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            
        Return :
            measures : float[nnodes, nmeasures]
                       padding NaN for nodes that do not have measures
                       nmeasures >= 1: number of measures with different dimensionalities
                       For example, if there are both lines and triangles, nmeasures = 2
            
    '''
    nnodes, ndims = nodes.shape
    measures = np.full((nnodes, ndims), np.NaN)
    measure_types = [False] * ndims
    for elem in elems:
        elem_dim, e = elem[0], elem[1:]
        e = e[e >= 0]
        ne = len(e)
        # compute measure based on elem_dim
        s = compute_measure_per_elem_(nodes[e, :], elem_dim)
        # assign it to cooresponding measures
        measures[e, elem_dim-1] = np.nan_to_num(measures[e, elem_dim-1], nan=0.0)
        measures[e, elem_dim-1] += s/ne 
        measure_types[elem_dim - 1] = True

    # return only nonzero measures
    return measures[:, measure_types]

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


def compute_edge_gradient_weights_helper(nodes, node_dims, adj_list, rcond = 1e-3):
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
            node_dims : int[nnodes], the intrisic dimensionality of the node
                        if the node is on a volume, it is 3
                        if the node is on a surface, it is 2
                        if the node is on a line, it is 1
                        if it is on different type of elements, take the maximum

                                
            adj_list : list of set, saving neighbors for each nodes
            rcond : float, truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value
            

        Return :

            directed_edges : int[nedges,2]
            edge_gradient_weights   : float[nedges, ndims]

            * the directed_edges (and adjacent list) include all node pairs that share the element
            
    '''

    nnodes, ndims = nodes.shape
    directed_edges = []
    edge_gradient_weights = [] 
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b,:] - nodes[a,:]
            directed_edges.append([a,b])
        edge_gradient_weights.append(pinv(dx, rrank=node_dims[a], rcond=rcond).T)
        
    directed_edges = np.array(directed_edges, dtype=int)
    edge_gradient_weights = np.concatenate(edge_gradient_weights, axis=0)
    return directed_edges, edge_gradient_weights, adj_list


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
    
    return compute_edge_gradient_weights_helper(nodes, node_dims, adj_list, rcond = rcond)




def preprocess_data(nodes_list, elems_list, features_list):
    '''
    Compute node measures (length, area or volume for each node), 
    for each element, compute its length, area or volume, 
    equally assign it to its nodes.

        Parameters:  
            nodes_list :  list of float[nnodes, ndims]
            elems_list :  list of float[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            features_list  : list of float[nnodes, nfeatures]
            

        Return :
            nnodes         :  int
            node_mask      :  int[ndata, max_nnodes, 1]               (1 for node, 0 for padding)
            nodes          :  float[ndata, max_nnodes, ndims]      (padding 0)
            node_measures  :  float[ndata, max_nnodes, 1]               (padding NaN)   
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
    
    print("Preprocessing data : computing node_measures")
    # The mesh might have elements with different dimensionalities (e.g., 1D edges, 2D faces, 3D volumes).
    # If any mesh includes both 1D and 2D elements, it is assumed that all meshes in the dataset will also include both types of elements.
    # This ensures uniformity in processing and avoids inconsistencies in element handling.
    node_measures = np.full((ndata, max_nnodes, ndims), np.NaN)
    nmeasures = 0
    for i in tqdm(range(ndata)):
        measures = compute_node_measures(nodes_list[i], elems_list[i])
        if i == 0:
            nmeasures = measures.shape[1]
        node_measures[i,:nnodes[i], :nmeasures] = measures
    node_measures = node_measures[...,:nmeasures]

    print("Preprocessing data : computing features")
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        features[i,:nnodes[i],:] = features_list[i] 

    print("Preprocessing data : computing directed_edges and edge_gradient_weights")
    directed_edges_list, edge_gradient_weights_list = [], []
    for i in tqdm(range(ndata)):
        directed_edges, edge_gradient_weights, edge_adj_list = compute_edge_gradient_weights(nodes_list[i], elems_list[i])
        directed_edges_list.append(directed_edges) 
        edge_gradient_weights_list.append(edge_gradient_weights)   
    nedges = np.array([directed_edges.shape[0] for directed_edges in directed_edges_list])
    max_nedges = max(nedges)
    directed_edges, edge_gradient_weights = np.zeros((ndata, max_nedges, 2),dtype=int), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        directed_edges[i,:nedges[i],:] = directed_edges_list[i]
        edge_gradient_weights[i,:nedges[i],:] = edge_gradient_weights_list[i] 

    return nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights 




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

    return nodes_list, elems_list, features_list  




def compute_node_weights(nnodes,  node_measures,  equal_measure = False):
    '''
    Compute node weights based on node measures (length, area, volume,......).

    This function calculates weights for each node using its corresponding measures. 
    If `equal_measure` is set to True, the node measures are recomputed as equal, |Omega|/n, 
    where `|Omega|` is the total measure and `n` is the number of nodes with nonzero measures.
    Node weights are computed such that their sum equals 1, using the formula:
        node_weight = node_measure / sum(node_measures)

    If there are several types of measures, compute weights for each type of measures, and normalize it by nmeasures
    node_weight = 1/nmeasures * node_measure / sum(node_measures)

    Parameters:
        nnodes int[ndata]: 
            Number of nodes for each data instance.
        
        node_measures float[ndata, max_nnodes, nmeasures]: 
            Each value corresponds to the measure of a node.
            Padding with NaN is used for indices greater than or equal to the number of nodes (`nnodes`), or nodes do not have measure

        equal_measure (bool, optional): 
            If True, node measures are uniformly distributed as |Omega|/n. Default is False.

    Returns:
        node_measures float[ndata, max_nnodes, nmeasures]: 
            Updated array of node measures with shape, maintaining the same padding structure (But with padding 0).
            If equal_measure is False, the measures remains unchanged
        node_weights float[ndata, max_nnodes, nmeasures]: 
            Array of computed node weights, maintaining the same padding structure.
    '''
        
    ndata, max_nnodes, nmeasures = node_measures.shape
    node_measures_new = np.zeros((ndata, max_nnodes, nmeasures))
    node_weights = np.zeros((ndata, max_nnodes, nmeasures))
    if equal_measure:
        for i in range(ndata):
            for j in range(nmeasures):
                # take average for nonzero measure nodes
                indices = np.isfinite(node_measures[i, :, j])  
                node_measures_new[i, indices, j] = sum(node_measures[i, indices, j])/sum(indices)
    else:
        # replace all NaN value to 0
        node_measures_new = np.nan_to_num(node_measures, nan=0.0)


    # node weight is the normalization of node measure
    for i in range(ndata):
        for j in range(nmeasures):
            node_weights[i, :nnodes[i], j] = 1.0/nmeasures * node_measures_new[i, :nnodes[i], j]/sum(node_measures_new[i, :nnodes[i], j])
    
    return node_measures_new, node_weights







