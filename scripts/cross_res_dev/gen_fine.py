import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEDIAN_DATA_PATH = os.path.join(BASE_DIR, "median_data.npz")
MEDIAN_ELEMS_PATH = os.path.join(BASE_DIR, "median_elems.npy")

NODES_COARSE_PATH = os.path.join(BASE_DIR, "nodes_coarse.npy")
ELEMS_COARSE_PATH = os.path.join(BASE_DIR, "elems_coarse.npy")
FEATURES_COARSE_PATH = os.path.join(BASE_DIR, "features_coarse.npy")

NODES_FINE_PATH = os.path.join(BASE_DIR, "nodes_fine.npy")
ELEMS_FINE_PATH = os.path.join(BASE_DIR, "elems_fine.npy")
FEATURES_FINE_PATH = os.path.join(BASE_DIR, "features_fine.npy")

IDX_CORR_PATH = os.path.join(BASE_DIR, "idx_corr.npy")
MESH_COMPARE_PATH = os.path.join(BASE_DIR, "mesh_compare.png")


def load_npz_required(path, key):
    """
    Load a required array from an npz file.
    """
    data = np.load(path)

    if key not in data:
        raise KeyError(f"Expected key '{key}' in {path}, got keys {list(data.keys())}.")
    return data[key]


def get_edge_midpoint(i, j, nodes, fine_nodes, edge_to_mid, features=None, fine_features=None):
    """
    Return the midpoint-node index of edge (i, j). The same edge shared by two
    triangles is only created once. If features are provided, the midpoint
    feature is the average of the two endpoint features.
    """
    if i > j:
        i, j = j, i
    edge = (int(i), int(j))

    if edge in edge_to_mid:
        return edge_to_mid[edge]

    midpoint = 0.5 * (nodes[i] + nodes[j])
    mid_idx = len(fine_nodes)
    fine_nodes.append(midpoint)
    edge_to_mid[edge] = mid_idx

    if features is not None and fine_features is not None:
        fine_features.append(0.5 * (features[i] + features[j]))

    return mid_idx


def refine_tri_mesh(nodes, elems, features=None):
    """
    Refine each triangular element into four smaller triangles.

    Coarse triangle:
        (a, b, c)

    Midpoints:
        ab, bc, ca

    Fine triangles:
        (a, ab, ca),
        (ab, b, bc),
        (ca, bc, c),
        (ab, bc, ca)

    Returns:
        nodes_fine : np.ndarray [n_fine_nodes, dim]
        elems_fine : np.ndarray [n_fine_elems, 3]
        features_fine : np.ndarray [n_fine_nodes, n_features] or None
            Fine-node features. Original coarse features are preserved first;
            edge-midpoint features are averaged from the two endpoint features.
        idx_corr : np.ndarray [n_fine_nodes]
            idx_corr[j] = i if fine node j is exactly coarse node i; otherwise -1.
            Since coarse nodes are kept at the beginning, idx_corr[:n_coarse] is
            np.arange(n_coarse).
    """
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)

    if features is not None:
        features = np.asarray(features)
        if features.shape[0] != nodes.shape[0]:
            raise ValueError(
                f"features must have the same first dimension as nodes, got "
                f"features.shape={features.shape}, nodes.shape={nodes.shape}."
            )

    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError(
            f"Only triangular elems with shape [n_elems, 3] are supported, got {elems.shape}."
        )

    n_coarse = nodes.shape[0]
    fine_nodes = [nodes[i].copy() for i in range(n_coarse)]
    fine_features = None
    if features is not None:
        fine_features = [features[i].copy() for i in range(n_coarse)]

    fine_elems = []
    edge_to_mid = {}

    for elem in elems:
        a, b, c = map(int, elem)

        ab = get_edge_midpoint(a, b, nodes, fine_nodes, edge_to_mid, features, fine_features)
        bc = get_edge_midpoint(b, c, nodes, fine_nodes, edge_to_mid, features, fine_features)
        ca = get_edge_midpoint(c, a, nodes, fine_nodes, edge_to_mid, features, fine_features)

        fine_elems.append([a, ab, ca])
        fine_elems.append([ab, b, bc])
        fine_elems.append([ca, bc, c])
        fine_elems.append([ab, bc, ca])

    nodes_fine = np.asarray(fine_nodes, dtype=nodes.dtype)
    elems_fine = np.asarray(fine_elems, dtype=elems.dtype)

    features_fine = None
    if fine_features is not None:
        features_fine = np.asarray(fine_features, dtype=features.dtype)

    idx_corr = -np.ones(nodes_fine.shape[0], dtype=np.int64)
    idx_corr[:n_coarse] = np.arange(n_coarse, dtype=np.int64)

    return nodes_fine, elems_fine, features_fine, idx_corr


def plot_mesh(ax, nodes, elems, title):
    """
    Plot a triangular mesh on a given matplotlib axis.
    """
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)

    if nodes.shape[1] < 2:
        raise ValueError(f"nodes must have at least 2 coordinates for visualization, got {nodes.shape}.")

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems.astype(np.int64))
    ax.triplot(triangulation, linewidth=0.7)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_mesh_comparison(nodes_coarse, elems_coarse, nodes_fine, elems_fine, save_path):
    """
    Save a side-by-side visualization of the coarse and fine meshes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    plot_mesh(
        axes[0],
        nodes_coarse,
        elems_coarse,
        f"Coarse mesh: {nodes_coarse.shape[0]} nodes, {elems_coarse.shape[0]} elems",
    )
    plot_mesh(
        axes[1],
        nodes_fine,
        elems_fine,
        f"Fine mesh: {nodes_fine.shape[0]} nodes, {elems_fine.shape[0]} elems",
    )

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    nodes_coarse = load_npz_required(MEDIAN_DATA_PATH, "nodes")[0, ...]
    x = load_npz_required(MEDIAN_DATA_PATH, "x")[0, ...]
    y = load_npz_required(MEDIAN_DATA_PATH, "y")[0, ...]
    features_coarse = np.concatenate([x, y], axis=-1)

    elems_coarse = np.load(MEDIAN_ELEMS_PATH)[:,1:]

    nodes_fine, elems_fine, features_fine, idx_corr = refine_tri_mesh(
        nodes_coarse,
        elems_coarse,
        features=features_coarse,
    )

    np.save(NODES_COARSE_PATH, nodes_coarse)
    np.save(ELEMS_COARSE_PATH, elems_coarse)
    np.save(FEATURES_COARSE_PATH, features_coarse)
    np.save(NODES_FINE_PATH, nodes_fine)
    np.save(ELEMS_FINE_PATH, elems_fine)
    np.save(FEATURES_FINE_PATH, features_fine)
    np.save(IDX_CORR_PATH, idx_corr)
    save_mesh_comparison(nodes_coarse, elems_coarse, nodes_fine, elems_fine, MESH_COMPARE_PATH)

    print("Loaded median coarse mesh files:")
    print(f"  {MEDIAN_DATA_PATH}: nodes {nodes_coarse.shape}, features {features_coarse.shape}")
    print(f"  {MEDIAN_ELEMS_PATH}: elems {elems_coarse.shape}")
    print("Saved coarse and fine mesh files:")
    print(f"  {NODES_COARSE_PATH}: {nodes_coarse.shape}")
    print(f"  {ELEMS_COARSE_PATH}: {elems_coarse.shape}")
    print(f"  {FEATURES_COARSE_PATH}: {features_coarse.shape}")
    print(f"  {NODES_FINE_PATH}: {nodes_fine.shape}")
    print(f"  {ELEMS_FINE_PATH}: {elems_fine.shape}")
    print(f"  {FEATURES_FINE_PATH}: {features_fine.shape}")
    print(f"  {IDX_CORR_PATH}: {idx_corr.shape}")
    print(f"  {MESH_COMPARE_PATH}")
    print("Coarse nodes and features are preserved as the first entries in the fine arrays.")


if __name__ == "__main__":
    main()
