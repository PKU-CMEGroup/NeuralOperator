import numpy as np
from math import prod
from tqdm import tqdm
from typing import List, Set, Union

def compute_triangle_area_(points:np.ndarray) -> float:
    ab = points[1, :] - points[0,:]
    ac = points[2, :] - points[0,:]
    cross_product = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_product)


def compute_tetrahedron_volume_(points:np.ndarray) -> float:
    ab = points[1, :] - points[0,:]
    ac = points[2, :] - points[0,:]
    ad = points[3, :] - points[0,:]
    # Calculate the scalar triple product
    volume = abs(np.dot(np.cross(ab, ac), ad)) / 6
    return volume

def compute_measure_per_elem_(points:np.ndarray, elem_dim:int) -> float:
    '''
    Compute element measure (length, area or volume)
    for 2-point  element, compute its length
    for 3-point  element, compute its area
    for 4-point  element, compute its area if elem_dim=2; compute its volume if elem_dim=3
    for 8-point  element, compute its volume require elem_dim=3
    equally assign it to its nodes
    
        Parameters: 
            points : float[npoints, ndims]
            elem_dim : int
    
        Returns:
            s : float
        
        Require:
            When computing the measure for a 4-point 2D quadrilateral or an 8-point 3D hexahedron, 
            we first decompose them into simplices, compute their individual measures, and then sum the results.
            For quad, the nodes are in clockwise or counterclockwise
              1 ---------2
             /          /
            0 -------- 3
            For hex, the nodes are  in the following order
              7 -------- 6
             /|         /|
            4 -------- 5 |
            | |        | |
            | 3 -------|-2
            |/         |/
            0 -------- 1
    '''
    
    npoints, ndims = points.shape
    if npoints == 2: 
        s = np.linalg.norm(points[0, :] - points[1, :])
    elif npoints == 3:
        s = compute_triangle_area_(points)
    elif npoints == 4:
        assert(elem_dim == 2 or elem_dim == 3)
        if elem_dim == 2:
            s = compute_triangle_area_(points[:3,:]) + compute_triangle_area_(points[1:,:])
        elif elem_dim == 3:
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("elem dim ", elem_dim,  "is not recognized")
    elif npoints == 8:
        assert(elem_dim == 3)
        s = (compute_tetrahedron_volume_(points[[0,1,3,5],:]) + 
             compute_tetrahedron_volume_(points[[0,3,5,7],:]) +
             compute_tetrahedron_volume_(points[[0,4,5,7],:]) + 
             compute_tetrahedron_volume_(points[[1,2,3,5],:]) +
             compute_tetrahedron_volume_(points[[2,5,6,7],:]) + 
             compute_tetrahedron_volume_(points[[2,3,5,7],:]))
    else:   
        raise ValueError("npoints ", npoints,  "is not recognized")
    return s


def pinv(a:np.ndarray, rrank:int, rcond:float = 1e-3) -> np.ndarray:
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



def compute_node_measures(nodes:np.ndarray, elems:np.ndarray) -> np.ndarray:
    '''
    For vertex-centered mesh.

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
    measures = np.full((nnodes, ndims), np.nan)
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


def compute_elem_measures(nodes:np.ndarray, elems:np.ndarray) -> np.ndarray:
    '''
    For cell-centered mesh.

    Compute elem measures  (separate length, area and volume ... for each element), 
    
        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            
        Return :
            measures : float[nelems, nmeasures]
                       padding NaN for elements that do not have measures
                       nmeasures >= 1: number of measures with different dimensionalities
                       For example, if there are both lines and triangles, nmeasures = 2
            
    '''
    nelems, ndims = elems.shape[0], nodes.shape[1]
    measures = np.full((nelems, ndims), np.nan)
    measure_types = [False] * ndims
    for i, elem in enumerate(elems):
        elem_dim, e = elem[0], elem[1:]
        e = e[e >= 0]
        # compute measure based on elem_dim
        measures[i, elem_dim-1] = compute_measure_per_elem_(nodes[e, :], elem_dim)
        measure_types[elem_dim - 1] = True

    # return only nonzero measures
    return measures[:, measure_types]



def compute_node_adjacent_list(nodes:np.ndarray, elems:np.ndarray, adjacent_type:str = "element") -> List[Set[int]]:
    """
    Compute node adjacency list for a given mesh.
    
        Parameters:  
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            
            adj_type: str
                    'edge' : nodes share one edge
                    'element' : nodes share one element
    
    
        Returns:
            adj_list: List[Set[int]]
                      each set contains adjacency node indices for each node
            
        Raises:
            ValueError: If unsupported adjacency type is specified
            When the adjacency type is 'edge', element nodes must follow a specific ordering:
            - For 2D elements, nodes should be listed in clockwise or counterclockwise order.
            - For 3D elements, only 4-node tetrahedra and 8-node hexahedra are supported, and the nodes must be ordered as specified.
    """
    nnodes, ndims = nodes.shape
    nelems, _ = elems.shape
    # Initialize adjacency list as a list of sets
    # Use a set to store unique directed edges
    adj_list = [set() for _ in range(nnodes)]
    
    if adjacent_type == "element":
        # Element-based adjacency: connect nodes that share elements
        # Loop through each element and create directed edges
        for elem in elems:
            elem_dim, e = elem[0], elem[1:]
            e = e[e >= 0]
            ne = len(e)
            for i in range(ne):
                # Add each node's neighbors to its set
                adj_list[e[i]].update([e[j] for j in range(ne) if j != i])
    
    elif adjacent_type == "edge":
        # Edge-based adjacency: connect nodes that share edges
        # Loop through each element and create directed edges
        for elem in elems:
            elem_dim, e = elem[0], elem[1:]
            e = e[e >= 0]
            ne = len(e)
            if elem_dim == 1 or (elem_dim == 3 and ne == 4):
                for i in range(ne):
                    adj_list[e[i]].update([e[j] for j in range(ne) if j != i])
            elif elem_dim == 2:
                for i in range(ne):
                    adj_list[e[i]].update([e[i-1],e[(i+1)%ne]])
            elif (elem_dim == 3 and ne == 8):
                # nodes are ordered as 
                #   7 -------- 6
                #  /|         /|
                # 4 -------- 5 |
                # | |        | |
                # | 3 -------|-2
                # |/         |/
                # 0 -------- 1

                adj_list[e[0]].update([e[1],e[3],e[4]])
                adj_list[e[1]].update([e[0],e[2],e[5]])
                adj_list[e[2]].update([e[1],e[3],e[6]])
                adj_list[e[3]].update([e[0],e[2],e[7]])
                adj_list[e[4]].update([e[0],e[5],e[7]])
                adj_list[e[5]].update([e[1],e[4],e[6]])
                adj_list[e[6]].update([e[2],e[5],e[7]])
                adj_list[e[7]].update([e[3],e[4],e[6]])

            else:
                raise ValueError(f"Unsupported element: element dimensionality is {elem_dim}, number of nodes is {ne}.")
    

    else:
        raise ValueError(f"Unsupported adjacency type: {adjacent_type}. Use 'edge' or 'element'")
    
    return adj_list


def compute_elem_adjacent_list(elems:np.ndarray, adjacent_type:str = "node") -> List[Set[int]]:
    """
    Compute element adjacency list for a given mesh.

            Parameters:  
            elems : int[nelems, max_num_of_nodes_per_elem+1]. 
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            
            adjacent_type: str
                    'node' : elements share at least one node
                    'edge' : elements share an edge (≥2 nodes in 2D/3D)
                    'face' : elements share a face (≥1 nodes for 1D element, ≥2 nodes for 2D element, ≥3 nodes for 3D element)
    
        Returns:
            adj_list: List[Set[int]]
                      each set contains adjacent element indices for each element
            
    """
    nelems, _ = elems.shape
    adj_list = [set() for _ in range(nelems)]
    
    # Step 1: build initial adjacency based on shared nodes
    node2elems = {}
    for ie, elem in enumerate(elems):
        e = elem[1:]
        e = e[e >= 0]
        for n in e:
            node2elems.setdefault(n, []).append(ie)
    # Populate adjacency list: elements sharing the same node are neighbors
    for e_list in node2elems.values():
        for i in range(len(e_list)):
            ei = e_list[i]
            adj_list[ei].update([ej for j, ej in enumerate(e_list) if j != i])

    if adjacent_type == "node":
        return adj_list

    # Step 2: filter adjacency for edge or face connections
    for ie1, e1_neigh in enumerate(adj_list):
        e1_dim, e1 = elems[ie1, 0], elems[ie1, 1:]
        e1 = e1[e1 >= 0]
        for ie2 in list(e1_neigh):
            e2 = elems[ie2, 1:]
            e2 = e2[e2 >= 0]
            common = len(set(e1) & set(e2))  # number of shared nodes

            if adjacent_type == "edge":
                # Edge adjacency: must share at least 2 nodes
                if common < 2:
                    e1_neigh.remove(ie2)

            elif adjacent_type == "face":
                # when e1 is 1d element, face means share 1 node
                # when e1 is 2d element, face means share an edge (2 nodes)
                # when e1 is 3d element, face means share an face (at least 3 nodes)
                if common < 2 and e1_dim == 2:
                    e1_neigh.remove(ie2)
                elif common < 3 and e1_dim == 3:
                    e1_neigh.remove(ie2)

    return adj_list

  
def compute_edge_gradient_weights_helper(nodes:np.ndarray, node_dims:np.ndarray, adj_list:List[Set[int]], rcond:float = 1e-3):
    '''
    Compute weights for gradient computation  
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)     =          :
       :                                :
    xj - x                        f(xj) - f(x)
    
    in matrix form   dx  nabla f(x)   = df.
    
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


def compute_edge_gradient_weights(nodes:np.ndarray, elems:np.ndarray, mesh_type:str, adjacent_type:str, rcond:float = 1e-3):
    '''
    Compute weights for gradient computation for quantities stored at each node

    When mesh_type = "vertex_centered", nodes are element vertices; 
    When mesh_type = "cell_centered", nodes are element centers.

    The function first construct the node_dims to store the (maximum) dimensionality at that node, 
    and the adjacent list for the nodes, then call compute_edge_gradient_weights_helper to compute 
    weights, as following


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

            mesh_type : str
                    Specifies the type of mesh:
                    - 'vertex_centered': nodes correspond to element vertices.
                    - 'cell_centered': nodes correspond to element centers.

            adjacent_type : str
                    Determines how adjacency is computed:
                    for 'vertex_centered' meshes: adjacency is based on nodes sharing an 'edge' or an 'element'.
                    for 'cell_centered' meshes, adjacency is based on elements sharing a 'node', 'edge', or 'face'.
            
            rcond : float, 
                    Truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value
            
        Return :

            directed_edges : int[nedges,2]
                All directed edges between adjacent nodes.

            edge_gradient_weights   : float[nedges, ndims]
               Gradient weights for each directed edge.

    '''
  
    nnodes, ndims = nodes.shape
    # Initialize node_dims to store the (maximum) dimensionality at that node
    node_dims = np.zeros(nnodes, dtype=int)
    
    # Compute node_dims, and construct adjacent list
    if mesh_type == "vertex_centered":
        # Loop through each element and compute the maximum dimensionality at that node
        for elem in elems:
            elem_dim, e = elem[0], elem[1:]
            node_dims[e] = np.maximum(node_dims[e], elem_dim)
        adj_list = compute_node_adjacent_list(nodes, elems, adjacent_type)

    elif mesh_type == "cell_centered":
        # Initialize node_dims to store the maximum dimensionality at that node
        node_dims[:] = elems[:,0]
        adj_list = compute_elem_adjacent_list(elems, adjacent_type)
    
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}. Use 'vertex_centered' or 'cell_centered'")
    

    return compute_edge_gradient_weights_helper(nodes, node_dims, adj_list, rcond = rcond)



def preprocess_data_point_cloud(nodes_list:List[np.ndarray], features_list:List[np.ndarray], node_measures_list:List[np.ndarray], adjacent_list:List[List[Set]], rcond:float = 1e-3):
    '''
    Preprocesses raw point cloud data.
    '''
    raise NotImplementedError(
        "Point cloud preprocessing not implemented. "
    )
    return


def preprocess_data_mesh(vertices_list:List[np.ndarray], elems_list:List[np.ndarray], features_list:List[np.ndarray], mesh_type: str, adjacent_type:str, rcond:float = 1e-3):
    '''
    Preprocesses a batch of meshes (vertex- or cell-centered). 
    We have the vertex coordinates and element connectivity of each mesh.
    For cell-centered meshes, features are stored on each element (or cell).
    For vertex-centered meshes, features are stored on each node (or vertex).

    This function converts list to numpy array and computes:
    - Node (vertex or cell center) coordinates and measures (length, area, volume, equal to the cell size)
    - Directed edges connecting nodes and gradient weights for finite difference operations
    - Proper padding and masking for batched processing


        Parameters:  
            vertices_list :  list of float[nvertices, ndims]
                    Original vertex coordinates for each mesh in the batch.
            
            elems_list :  list of int[nelems, max_num_of_nodes_per_elem+1]. 
                    Element connectivity.
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.

            features_list  : list of float[nelems, nfeatures]
                    Node features (on vertices or cell centers) depending on mesh_type.

            mesh_type : str
                    Specifies the type of mesh:
                    - 'vertex_centered': nodes correspond to element vertices.
                    - 'cell_centered': nodes correspond to element centers.

            adjacent_type : str
                    Determines how adjacency is computed:
                    for 'vertex_centered' meshes: adjacency is based on nodes sharing an 'edge' or an 'element'.
                    for 'cell_centered' meshes, adjacency is based on elements sharing a 'node', 'edge', or 'face'.
            
            rcond : float
                    Truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value
            

        Return :
            nnodes : int[ndata]
                    Number of nodes per mesh.
            node_mask : int[ndata, max_nnodes, 1]         
                    Mask indicating valid nodes (1) vs padding (0).     
            nodes : float[ndata, max_nnodes, ndims]     
                    Node coordinates (vertex positions or cell centers). Padded with 0.
            node_measures : float[ndata, max_nnodes, 1] 
                    Node measures (length, area, volume). Padded with NaN.             
            features : float[ndata, max_nnodes, nfeatures]  
                    Node (on vertices or cell centers) features. Padded with 0.  
            directed_edges :  float[ndata, max_nedges, 2]         
                    Directed edges between nodes for gradient computation. Padded with 0.  
            edge_gradient_weights   :  float[ndata, max_nedges, ndims]    
                    Gradient weights for each direced edge. Padded with 0.
    '''

    ndata = len(vertices_list)
    ndims, nfeatures = vertices_list[0].shape[1], features_list[0].shape[1]
    
    # Determine number of nodes per sample
    if mesh_type == "vertex_centered":
        nnodes = np.array([nodes.shape[0] for nodes in vertices_list], dtype=int)
    elif mesh_type == "cell_centered":
        nnodes = np.array([elems.shape[0] for elems in elems_list], dtype=int)
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")
    max_nnodes = nnodes.max()


    print("Preprocessing data : computing node_mask")
    node_mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        node_mask[i,:nnodes[i], :] = 1


    print("Preprocessing data : computing node coordinates")
    nodes = np.zeros((ndata, max_nnodes, ndims))
    if mesh_type == "vertex_centered":
        for i in range(ndata):
            nodes[i, :nnodes[i], :] = vertices_list[i]
    else:
        for i in range(ndata):
            for j in range(nnodes[i]):
                e = elems_list[i][j,1:]
                e = e[e >= 0]
                nodes[i, j, :] = np.mean(vertices_list[i][e, :], axis=0)
    
    print("Preprocessing data : computing node measures")
    # The mesh might have elements with different dimensionalities (e.g., 1D edges, 2D faces, 3D volumes).
    # If any mesh includes both 1D and 2D elements, it is assumed that all meshes in the dataset will also include both types of elements.
    # This ensures uniformity in processing and avoids inconsistencies in element handling.
    node_measures = np.full((ndata, max_nnodes, ndims), np.nan)
    nmeasures = 0
    for i in tqdm(range(ndata)):
        if mesh_type == "vertex_centered":
            measures = compute_node_measures(vertices_list[i], elems_list[i]) 
        else:
            measures = compute_elem_measures(vertices_list[i], elems_list[i])
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
        directed_edges, edge_gradient_weights, edge_adj_list = compute_edge_gradient_weights(nodes[i, :nnodes[i], :], elems_list[i], mesh_type=mesh_type, adjacent_type=adjacent_type, rcond=rcond)
        directed_edges_list.append(directed_edges) 
        edge_gradient_weights_list.append(edge_gradient_weights)   
    
    # Pad edges to max_nedges
    nedges = np.array([directed_edges.shape[0] for directed_edges in directed_edges_list])
    max_nedges = max(nedges)
    directed_edges, edge_gradient_weights = np.zeros((ndata, max_nedges, 2),dtype=int), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        directed_edges[i,:nedges[i],:] = directed_edges_list[i]
        edge_gradient_weights[i,:nedges[i],:] = edge_gradient_weights_list[i] 

    return nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights 





def convert_structured_data(coords_list, features, nnodes_per_elem = 3, feature_include_coords = True):
    '''
    Convert structured data, to unstructured data, support both 2d and 3d coordinates
    coords_list stores x, y, (z) coordinates of each points in list of ndims float[nnodes, nx, ny, (nz)], coords_list[i] is as following
                    nz-1       ny-1                                                         
                    nz-2     ny-2                                                            
                    .       .                                                               
    z direction     .      .   (y direction)            
                    .     1                                                                 
                    1    .                                                                    
                    0   0                                                                    
                        0 - 1 - 2 - ... - nx-1   (x direction)
    For example, it can be generated as
    grid_1d_x, grid_1d_y, grid_1d_z = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), np.linspace(0, Lz, nz)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d_x, grid_1d_y, grid_1d_z, indexing="ij")
    coords_list = [grid_x, grid_y, grid_z]
    
    Then we order the nodes by iterating z, then y, then x (reshape)
    for i in range(nx-1): 
        for j in range(ny-1): 
            for k in range(nz-1): 
                id = i*ny*nz + j*nz + k
    For example when nx=ny=nz, the ordering is as following
          3 -------- 7
         /|         /|
        1 -------- 5 |
        | |        | |
      z | 2 -------|-6
        |/y        |/
        0 ----x----4      


        Parameters:  
            coords_list            :  list of ndims float[nnodes, nx, ny, (nz)], for each dimension coords_list[0], coords_list[1],... are x, y,... coordinates
            features               :  float[nelems, nx, ny, (nz), nfeatures], features on each point
            nnodes_per_elem        :  int, describing element type
                                      nnodes_per_elem = 3: 2d triangle mesh; 
                                      nnodes_per_elem = 4: 2d quad mesh or 3d tetrahedron mesh
                                      nnodes_per_elem = 8: 3d hexahedron mesh
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
    ndims = len(coords_list)
    print("convert_structured_data for ", ndims, " problems")
    ndata, *dims = coords_list[0].shape
    nnodes = prod(dims)
    # construct nodes
    nodes = np.stack((coords_list[i].reshape((ndata, nnodes)) for i in range(ndims)), axis=2)
    # construct features
    if feature_include_coords :
        nfeatures = features.shape[-1] + ndims
        features = np.concatenate((features.reshape((ndata, nnodes, -1)), nodes), axis=-1)
    else :
        nfeatures = features.shape[-1]
        features = features.reshape((ndata, nnodes, -1))
    
    # construct elements
    if (ndims == 2): # triange (nnodes_per_elem = 3), quad (nnodes_per_elem = 4)
        assert(nnodes_per_elem == 4 or nnodes_per_elem == 3)
        nx, ny = dims
        nelems = (nx-1)*(ny-1)*(1 if nnodes_per_elem == 4 else 2)
        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx-1):
            for j in range(ny-1):
                ie = i*(ny-1) + j   #element id
                #node ids clockwise
                #   1 ---------2
                #  /          /
                # 0 -------- 3
                ins = [i*ny+j, i*ny+j+1, (i+1)*ny+j+1, (i+1)*ny+j]  
                if nnodes_per_elem == 4:
                    elems[ie, :] = 2, ins[0], ins[1], ins[2], ins[3]
                else:
                    elems[2*ie, :]   = 2, ins[0], ins[1], ins[2]
                    elems[2*ie+1, :] = 2, ins[0], ins[2], ins[3]
    elif (ndims == 3): # tetrahedron (nnodes_per_elem = 4), cubic (nnodes_per_elem = 8)
        assert(nnodes_per_elem == 8 or nnodes_per_elem == 4)
        nx, ny, nz = dims
        nelems = (nx-1)*(ny-1)*(nz-1)*(1 if nnodes_per_elem == 8 else 6)
        elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    ie = i*(ny-1)*(nz-1) + j*(nz-1) + k #element id
                    # node ids for k, and k+1 in counterclockwise
                    #   7 -------- 6
                    #  /|         /|
                    # 4 -------- 5 |
                    # | |        | |
                    # | 3 -------|-2
                    # |/         |/
                    # 0 -------- 1
                    ins = [i*ny*nz+j*nz+k,     (i+1)*ny*nz+j*nz+k,     (i+1)*ny*nz+(j+1)*nz+k,     i*ny*nz+(j+1)*nz+k, 
                           i*ny*nz+j*nz+(k+1), (i+1)*ny*nz+j*nz+(k+1), (i+1)*ny*nz+(j+1)*nz+(k+1), i*ny*nz+(j+1)*nz+(k+1)]
                    if nnodes_per_elem == 8:
                        elems[ie, :] = 3, ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7]
                    else:
                        elems[6*ie, :]   = 3, ins[0], ins[1], ins[3], ins[5]
                        elems[6*ie+1, :] = 3, ins[0], ins[3], ins[5], ins[7]
                        elems[6*ie+2, :] = 3, ins[0], ins[4], ins[5], ins[7]
                        elems[6*ie+3, :] = 3, ins[1], ins[2], ins[3], ins[5]
                        elems[6*ie+4, :] = 3, ins[2], ins[5], ins[6], ins[7]
                        elems[6*ie+5, :] = 3, ins[2], ins[3], ins[5], ins[7]


    elems = np.tile(elems, (ndata, 1, 1))
    nodes_list = [nodes[i,...] for i in range(ndata)]
    elems_list = [elems[i,...] for i in range(ndata)]
    features_list = [features[i,...] for i in range(ndata)]

    return nodes_list, elems_list, features_list  




def compute_node_weights(nnodes:np.ndarray,  node_measures:np.ndarray,  equal_measure:bool = False):
    '''
    Compute node weights based on node measures (length, area, volume,......).

    This function calculates measures and weights (normalized measures) for each node using its corresponding measures. 
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







