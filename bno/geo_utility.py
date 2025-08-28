import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from pcno.geo_utility import compute_edge_gradient_weights, compute_node_measures


def compute_min_point_distance(boundary, query):
    tree = cKDTree(boundary)
    distances, _ = tree.query(query, k=1)
    return distances


def neighbor_search_edge_csr(data: np.ndarray, queries: np.ndarray, radius: float):
    '''
    Perform neighborhood search as a preprocessing step in CSR format
    '''

    dists = np.linalg.norm(
        queries[:, None, :] - data[None, :, :], axis=-1)
    in_nbr = dists <= radius  # mask where True means within radius

    # Get neighbor indices
    nbr_indices = np.nonzero(in_nbr)[1]

    # cumulative sum for CSR format
    nbrhd_sizes = np.cumsum(np.sum(in_nbr, axis=1))
    splits = np.insert(nbrhd_sizes, 0, 0)

    return nbr_indices.astype(np.int64), splits.astype(np.int64)


def neighbor_search_edge(data: np.ndarray, queries: np.ndarray, radius: float):

    dists = np.linalg.norm(
        queries[:, None, :] - data[None, :, :], axis=-1)
    in_nbr = dists <= radius

    tgt, src = np.nonzero(in_nbr)
    nedge = len(tgt)
    directed_edges = np.zeros((nedge, 2))
    directed_edges[:, 0] = src
    directed_edges[:, 1] = tgt

    return directed_edges.astype(np.int64)


def _compute_node_weights(nnodes, node_measures, equal_weights=False):
    '''
    This function calculates weights and rhos for each node using its corresponding measures.
            v(x) = ∫ u(x)rho(x) dx 
                 ≈ ∑ u(x_i)rho(x_i)m(x_i) 
                 = ∑ u(x_i)w(x_i)
            w(x_i) = rho(x_i)m(x_i)
    Node weights are computed such that their sum equals 1, because:
              1  = ∫ rho(x) dx 
                 ≈ ∑ rho(x_i)m(x_i)
                 = ∑ w(x_i)
    If there are several types of measures, compute weights for each type of measures, and normalize it by nmeasures
            w_k(x_i)= w_k(x_i)/nmeasures 

    Parameters:
        nnodes int[ndata]: 
            Number of nodes for each data instance.

        node_measures float[ndata, max_nnodes, nmeasures]: 
            Each value corresponds to the measure of a node.
            Padding with NaN is used for indices greater than or equal to the number of nodes (`nnodes`), or nodes do not have measure

        equal_weights bool:
            - True
                    w(x_i)=1/nnodes
              and we can recover rho by
                    rho(x_i) = w(x_i)/m(x_i) = 1/(m(x_i)*nnodes)
            - False 
                    rho(x_i)=1/|Omega|
               and we can compute w by
                    w(x_i) = rho(x_i)m(x_i) = m(x_i)/|Omega|

    Returns:
        node_weights float[ndata, max_nnodes, nmeasures]: 
            Array of computed node weights, maintaining the same padding structure.
        node_rhos   float[ndata, max_nnodes, nmeasures]: 
            Array of computed node rhos, maintaining the same padding structure.
    '''

    ndata, max_nnodes, nmeasures = node_measures.shape
    node_weights = np.zeros((ndata, max_nnodes, nmeasures))
    node_rhos = np.zeros((ndata, max_nnodes, nmeasures))
    if equal_weights:
        for i in range(ndata):
            n = nnodes[i]
            for j in range(nmeasures):
                # take average for nonzero measure nodes
                S = sum(node_measures[i, :n, j])
                node_weights[i, :n, j] = 1 / n
                node_rhos[i, :n, j] = 1 / (node_measures[i, :n, j] * n)

    else:
        for i in range(ndata):
            n = nnodes[i]
            for j in range(nmeasures):
                S = sum(node_measures[i, :n, j])
                node_rhos[i, :n, j] = 1 / S
                node_weights[i, :n, j] = node_measures[i, :n, j] / S

    node_weights = node_weights / nmeasures
    node_rhos = node_rhos / nmeasures

    return node_weights, node_rhos


def compute_edge_neighbor_weights(nedges, nnodes, directed_edges, equal_weights=True):

    ndata = nedges.shape[0]
    max_nedges = np.max(nedges)
    edge_weights = np.zeros((ndata, max_nedges))

    if equal_weights:
        for i in range(ndata):
            tgt = directed_edges[i, :nedges[i], 1]
            counts = np.bincount(tgt, minlength=nnodes[i])
            edge_weights[i, :nedges[i]] = 1 / counts[tgt]
    else:
        raise NotImplementedError(
            "Computation for non-equal weights is not yet implemented")

    return edge_weights


def _preprocess_all(nodes_list, elems_list, features_list):

    ndata = len(nodes_list)
    ndims, nfeatures = nodes_list[0].shape[1], features_list[0].shape[1]
    nnodes = np.array([nodes.shape[0] for nodes in nodes_list], dtype=int)
    max_nnodes = max(nnodes)

    print("computing mask")
    mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        mask[i, :nnodes[i], :] = 1

    print("computing nodes")
    nodes = np.zeros((ndata, max_nnodes, ndims))
    for i in range(ndata):
        nodes[i, :nnodes[i], :] = nodes_list[i]

    print("computing features")
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        features[i, :nnodes[i], :] = features_list[i]

    print("computing measures")
    measures = np.full((ndata, max_nnodes, ndims), np.nan)
    nmeasures = 0
    for i in tqdm(range(ndata)):
        if i == 0:
            nmeasures = compute_node_measures(nodes_list[i], elems_list[i]).shape[1]
        measures[i, :nnodes[i], :nmeasures] = compute_node_measures(nodes_list[i], elems_list[i])
    measures = measures[..., :nmeasures]

    print("computing edges and gradient weights")
    edges_list, edgeweights_list = [], []
    for i in tqdm(range(ndata)):
        edges, edgeweights, edge_adj_list = compute_edge_gradient_weights(
            nodes_list[i], elems_list[i])
        edges_list.append(edges)
        edgeweights_list.append(edgeweights)
    nedges = np.array([es.shape[0]for es in edges_list])
    max_nedges = max(nedges)

    edges, edgeweights = np.zeros(
        (ndata, max_nedges, 2), dtype=int), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        edges[i, :nedges[i], :] = edges_list[i]
        edgeweights[i, :nedges[i], :] = edgeweights_list[i]

    print("computing weights and rhos")
    weights, rhos = _compute_node_weights(nnodes, measures[..., :nmeasures], equal_weights=False)
    equal_weights, equal_rhos = _compute_node_weights(
        nnodes, measures[..., :nmeasures], equal_weights=True)
    return nnodes, nodes, mask, measures, weights, equal_weights, rhos, equal_rhos, features, nedges, edges, edgeweights


def _preprocess_boundary(nodes_list, elems_list, features_list):

    ndata = len(nodes_list)
    ndims, nfeatures = nodes_list[0].shape[1], features_list[0].shape[1]
    nnodes = np.array([nodes.shape[0] for nodes in nodes_list], dtype=int)
    max_nnodes = max(nnodes)

    print("computing mask")
    mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        mask[i, :nnodes[i], :] = 1

    print("computing nodes")
    nodes = np.zeros((ndata, max_nnodes, ndims))
    for i in range(ndata):
        nodes[i, :nnodes[i], :] = nodes_list[i]

    print("computing measures")
    measures = np.full((ndata, max_nnodes, ndims), np.nan)
    nmeasures = 0
    for i in tqdm(range(ndata)):
        if i == 0:
            nmeasures = compute_node_measures(nodes_list[i], elems_list[i]).shape[1]
        measures[i, :nnodes[i], :nmeasures] = compute_node_measures(nodes_list[i], elems_list[i])

    print("computing features")
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        features[i, :nnodes[i], :] = features_list[i]

    print("computing weights and rhos")
    weights, rhos = _compute_node_weights(
        nnodes, measures[..., :nmeasures], equal_weights=False)
    equal_weights, equal_rhos = _compute_node_weights(
        nnodes, measures[..., :nmeasures], equal_weights=True)

    return nnodes, nodes, mask, measures, weights, equal_weights, rhos, equal_rhos, features


def preprocess_data_mesh(nodes_all_list, elems_all_list, features_all_list, nodes_boundary_list, elems_boundary_list, features_boundary_list, radius, should_find_index=True, tol=1e-05):

    ndata = len(nodes_all_list)

    print(f"Preprocessing data in the entire domain...", flush=True)
    nnodes_all, nodes_all, mask_all, measures_all, weights_all, equal_weights_all, rhos_all, equal_rhos_all, features_all, nedges_all, edges_all, edgeweights_all = _preprocess_all(
        nodes_all_list, elems_all_list, features_all_list)

    print(f"Preprocessing data on the boundary...", flush=True)
    nnodes_boundary, nodes_boundary, mask_boundary, measures_boundary, weights_boundary, equal_weights_boundary, rhos_boundary, equal_rhos_boundary, features_boundary = _preprocess_boundary(
        nodes_boundary_list, elems_boundary_list, features_boundary_list)

    print(f"Preprocessing data in the boundary neighbor...", flush=True)
    print("computing point-wise SDF")
    n, max_nnodes, _ = nodes_all.shape
    sdf = np.full((n, max_nnodes, 1), 0.0, dtype=np.float64)

    for i in tqdm(range(n)):
        nodes_src = nodes_boundary[i, :nnodes_boundary[i], :]
        nodes_query = nodes_all[i, :nnodes_all[i], :]
        sdf[i, :nnodes_all[i], 0] = compute_min_point_distance(nodes_src, nodes_query)
    features_all = np.concatenate([features_all, sdf], axis=-1)

    print("computing indices")
    indices = -np.ones((ndata, max(nnodes_boundary), 1), dtype=np.int32)
    if should_find_index:
        for i in tqdm(range(ndata)):
            edges = neighbor_search_edge(
                data=nodes_boundary[i, :nnodes_boundary[i], :], queries=nodes_all[i, :nnodes_all[i], :], radius=tol)
            assert edges.shape[0] == nnodes_boundary[i], (
                f"Sample {i} mismatch: expected {nnodes_boundary[i]} boundary points, but only found {edges.shape[0]}. Some boundary points cannot be mapped to indices in the entire field."
            )
            tgt_indices = edges[:, 1]
            indices[i, :nnodes_boundary[i], 0] = tgt_indices
    else:
        for i in range(ndata):
            indices[i, :nnodes_boundary[i], 0] = np.arange(0, nnodes_boundary[i])

    print("computing boundary neighbor edges and weights")
    edges_list = []
    for i in tqdm(range(ndata)):
        edges = neighbor_search_edge(
            data=nodes_all[i, :nnodes_all[i], :], queries=nodes_boundary[i, :nnodes_boundary[i], :], radius=radius)
        edges_list.append(edges)
    nedges_boundary = np.array([edges.shape[0] for edges in edges_list])
    edges_boundary = np.zeros((ndata, max(nedges_boundary), 2), dtype=np.int64)
    for i in range(ndata):
        edges_boundary[i, :nedges_boundary[i], :] = edges_list[i]
    edgeweights_boundary = compute_edge_neighbor_weights(
        nedges_boundary, nnodes_boundary, edges_boundary, equal_weights=True)

    data_dict = dict(r=radius,
                     nnodes_all=nnodes_all, nodes_all=nodes_all, mask_all=mask_all,
                     measures_all=measures_all,
                     weights_all=weights_all, rhos_all=rhos_all,
                     equal_weights_all=equal_weights_all, equal_rhos_all=equal_rhos_all,
                     features_all=features_all,
                     nedges_all=nedges_all,
                     edges_all=edges_all,
                     edgeweights_all=edgeweights_all,
                     indices=indices,
                     nnodes_boundary=nnodes_boundary, nodes_boundary=nodes_boundary, mask_boundary=mask_boundary,
                     measures_boundary=measures_boundary,
                     weights_boundary=weights_boundary, rhos_boundary=rhos_boundary,
                     equal_weights_boundary=equal_weights_boundary, equal_rhos_boundary=equal_rhos_boundary,
                     features_boundary=features_boundary,
                     nedges_boundary=nedges_boundary,
                     edges_boundary=edges_boundary,
                     edgeweights_boundary=edgeweights_boundary)

    return data_dict


def _pad_and_concat(array_list, padding=0):

    ndim = array_list[0].ndim
    if ndim == 0:
        return np.array(array_list)

    total_ndata = sum(a.shape[0] for a in array_list)
    if ndim == 1:
        return np.concatenate(array_list, axis=0)
    max_m = max(a.shape[1] for a in array_list)

    if ndim == 2:
        result = np.full((total_ndata, max_m), padding, dtype=array_list[0].dtype)
    else:
        result = np.full((total_ndata, max_m) + array_list[0].shape[2:], padding, dtype=array_list[0].dtype)

    i = 0
    for a in array_list:
        n, m = a.shape[0], a.shape[1]
        if ndim == 2:
            result[i:i + n, :m] = a
        else:
            result[i:i + n, :m, ...] = a
        i += n

    return result


def mix_data(data_folder, m_trains, shape_types_train, m_tests=None, shape_types_test=None, model_name='bno'):

    temp = {}

    # training data
    for m, shape in zip(m_trains, shape_types_train):
        data = np.load(data_folder + f'{model_name}_{shape}_data.npz')
        for key in data.files:
            if key not in temp:
                temp[key] = []
            ndim = data[key].ndim
            if ndim == 0:
                temp[key].append(data[key])
            else:
                temp[key].append(data[key][:m] if ndim == 1 else data[key][:m, ...])

    # testing data
    if m_tests is not None:
        for m, shape in zip(m_tests, shape_types_test):
            data = np.load(data_folder + f'{model_name}_{shape}_data.npz')
            for key in data.files:
                ndim = data[key].ndim
                if ndim == 0:
                    temp[key].append(data[key])
                else:
                    temp[key].append(data[key][-m:] if ndim == 1 else data[key][-m:, ...])

    mixed_data = {}
    for key, array_list in temp.items():
        if key in {"measures_all", "measures_boundary"}:
            padding_value = np.nan
        else:
            padding_value = 0
        mixed_data[key] = _pad_and_concat(array_list, padding_value)

    return mixed_data
