import numpy as np
import matplotlib.pyplot as plt


def compute_triangle_area_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    cross_product = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_product)


def compute_tetrahedron_volume_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    ad = points[3, :] - points[0, :]
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
        assert (npoints == 3 or npoints == 4)
        if elem_dim == 2:
            s = compute_triangle_area_(
                points[:3, :]) + compute_triangle_area_(points[1:, :])
        elif elem_dim == 3:
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("elem dim ", elem_dim, "is not recognized")
    else:
        raise ValueError("npoints ", npoints, "is not recognized")
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
        weights = 1.0 / nnodes
    else:
        for elem in elems:
            elem_dim, e = elem[0], elem[1:]
            e = e[e >= 0]
            ne = len(e)
            s = compute_weight_per_elem_(nodes[e, :], elem_dim)
            weights[e] += s / ne

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

    res = np.matmul(np.transpose(vt), np.multiply(
        s[..., np.newaxis], np.transpose(u)))
    return res


def compute_edge_gradient_weights(nodes, elems, rcond=1e-3):
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
            adj_list[e[i]].update([e[j]
                                  for j in range(nnodes_per_elem) if j != i])

    directed_edges = []
    edge_gradient_weights = []
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b, :] - nodes[a, :]
            directed_edges.append([a, b])
        edge_gradient_weights.append(
            pinv(dx, rrank=node_dims[a], rcond=rcond).T)

    directed_edges = np.array(directed_edges, dtype=int)
    edge_gradient_weights = np.concatenate(edge_gradient_weights, axis=0)
    return directed_edges, edge_gradient_weights, adj_list


def ahmed_data(nodes_list, elems_list, elemfeats_list, infos_list, node_weight_type=None):

    ndata = len(nodes_list)
    ndims = nodes_list[0].shape[1]
    nnodes = np.array([nodes.shape[0] for nodes in nodes_list], dtype=int)
    max_nnodes = max(nnodes)

    print("Preprocessing data : computing node_mask", flush=True)
    node_mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        node_mask[i, :nnodes[i], :] = 1

    print("Preprocessing data : computing nodes", flush=True)
    nodes = np.zeros((ndata, max_nnodes, ndims))
    for i in range(ndata):
        nodes[i, :nnodes[i], :] = nodes_list[i]

    print("Preprocessing data : computing node_weights", flush=True)
    node_weights = np.zeros((ndata, max_nnodes, 1))
    for i in range(ndata):
        node_weights[i, :nnodes[i], 0] = compute_node_weights(
            nodes_list[i], elems_list[i], node_weight_type)

    print("Preprocessing data : computing node features from elemfeats", flush=True)
    nelemfeats = elemfeats_list[0].shape[-1]
    ninfos = len(infos_list[0])
    nfeatures = nelemfeats + ninfos
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        n_neighbor_elems = np.zeros(nnodes[i], dtype=int)
        for (tri, feat) in zip(elems_list[i], elemfeats_list[i]):
            features[i, tri[1:], :1] += feat
            n_neighbor_elems[tri[1:]] += 1
        features[i, :nnodes[i], 0] = features[i,
                                              :nnodes[i], 0] / n_neighbor_elems
        infos = np.tile(np.array(list(infos_list[i].values())), (nnodes[i], 1))
        features[i, :nnodes[i], 1:] = infos

    print("Preprocessing data : computing directed_edges and edge_gradient_weights", flush=True)
    directed_edges_list, edge_gradient_weights_list = [], []
    for i in range(ndata):
        directed_edges, edge_gradient_weights, edge_adj_list = compute_edge_gradient_weights(
            nodes_list[i], elems_list[i])
        # print(directed_edges.shape)
        directed_edges_list.append(directed_edges)
        edge_gradient_weights_list.append(edge_gradient_weights)
    nedges = np.array([directed_edges.shape[0]
                      for directed_edges in directed_edges_list])
    max_nedges = max(nedges)
    directed_edges, edge_gradient_weights = np.zeros(
        (ndata, max_nedges, 2), dtype=int), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        directed_edges[i, :nedges[i], :] = directed_edges_list[i]
        edge_gradient_weights[i, :nedges[i], :] = edge_gradient_weights_list[i]

    return nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights
