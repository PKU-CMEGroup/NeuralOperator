import numpy as np
import math
from tqdm import tqdm
import os

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from pcno.geo_utility import compute_node_measures
from pcno.geo_utility import preprocess_data_mesh, compute_node_weights

def random_polar_curve(N, k=4, r0_range = 0, freq_range = 0.5):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_base = r0_range * np.random.uniform(-1, 1)
    r = np.full_like(t, r_base)
    for i in range(1, k + 1):
        a_sin = freq_range*np.random.uniform(-1, 1)
        a_cos = freq_range*np.random.uniform(-1, 1)
        r += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
    r = np.tanh(r) + 1.5
    x = r * np.cos(t)
    y = r * np.sin(t)
    nodes = np.stack([x, y], axis=1)
    return nodes

def sequential_elems(N):
    """
    Generates a sequence of elements represented as arrays of three integers.
    Each element is constructed as follows:
        - The first entry is set to 1 (elem_dim).
        - The second entry is the current index (i).
        - The third entry is the next index in sequence, wrapping around to 0 at the end ((i + 1) % N).
    Parameters
    ----------
    N : int
        The number of elements to generate.
    Returns
    -------
    numpy.ndarray
        An array of shape (N, 3), where each row represents an element as described above.
    """

    elem_dim = 1
    idx = np.arange(N)
    next_idx = (idx + 1) % N
    elems = np.stack([np.full(N, elem_dim, dtype=int), idx, next_idx], axis=1)
    return elems

def smooth_feature_f(N, k=6):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.zeros(N)
    for i in range(1, k + 1):
        amp1 = np.random.uniform(0.5, 1.5)/math.sqrt(i)
        amp2 = np.random.uniform(0.5, 1.5)/math.sqrt(i)
        freq = i
        f += amp1 * np.sin(freq * t)
        f += amp2 * np.cos(freq * t)
    f = np.tanh(f) + 0.5 * np.sin(f)
    return f.reshape(-1, 1)

def kernel(x, y, kernel_type = 'log'):
    if kernel_type == 'log':
        return np.log(np.linalg.norm(x - y)+1e-6)
    elif kernel_type == 'inv':
        return 1.0 / (np.linalg.norm(x - y) + 1e-6)
    elif kernel_type == 'grad_log':
        diff = y - x
        norm_sq = np.dot(diff, diff) + 1e-6
        return diff / norm_sq
    else:
        raise ValueError("Unknown kernel type")

def compute_unit_normals(nodes, elems):
    """
    Compute outward unit normals for a closed planar curve.
    Args:
        nodes (np.ndarray): (N,2) array of ordered points along the closed curve.
        elems (np.ndarray): (M,3) element connectivity (only 1D edges used). Not strictly needed if nodes are ordered.
    Returns:
        np.ndarray: (N,2) outward unit normal vectors.
    """
    nodes = np.asarray(nodes)
    N = nodes.shape[0]

    # Tangent via centered difference (assumes cyclic ordering)
    prev_idx = (np.arange(N) - 1) % N
    next_idx = (np.arange(N) + 1) % N
    tangents = nodes[next_idx] - nodes[prev_idx]

    # Normalize tangents
    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tan_norm[tan_norm == 0.0] = 1.0
    tangents /= tan_norm

    # Signed area to determine orientation (CCW => area > 0)
    x = nodes[:, 0]
    y = nodes[:, 1]
    area = 0.5 * np.sum(x * y[np.roll(np.arange(N), -1)] - x[np.roll(np.arange(N), -1)] * y)
    ccw = area > 0

    # For CCW, outward normal = (ty, -tx); for CW reverse
    tx = tangents[:, 0]
    ty = tangents[:, 1]
    if ccw:
        normals = np.stack([ty, -tx], axis=1)
    else:
        normals = np.stack([-ty, tx], axis=1)

    # Normalize (robustness)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norm[n_norm == 0.0] = 1.0
    normals /= n_norm
    return normals

def curve_integral_g(node_list, f, kernel_type, node_measure, normal_vector=None, approaching_direction=None):
    """
    Computes the integral of a kernel function over a set of nodes, optionally using the normal vector for differentiation.
    This function evaluates either:
        ∫ k(x, y) g(y) dy
    or
        ∫ ∂_n_y k(x, y) g(y) dy
    depending on whether the normal_vector is provided.
    Args:
        node_list (np.ndarray): Array of node coordinates with shape (N, ndim).
        f (np.ndarray): Function values at each node, shape (N, 1).
        kernel_type (str): Type of kernel to use in the computation.
        node_measure (np.ndarray): Measure (e.g., weight or length) associated with each node, shape (N,).
        normal_vector (np.ndarray, optional): Normal vectors at each node, shape (N, ndim). If provided, computes the normal derivative of the kernel.
    Returns:
        np.ndarray: Resulting integral values at each node, shape (N, 1).
    """
    N = node_list.shape[0]
    g = np.zeros((N, 1))
    for i in range(N):
        x = node_list[i]
        for j in range(N):
            if j != i:
                y = node_list[j]
                if normal_vector is not None:
                    kxy = np.dot(kernel(x, y, kernel_type), normal_vector[j])
                else:
                    kxy = kernel(x, y, kernel_type)
                g[i, 0] += kxy * f[j, 0] * node_measure[j]
        if approaching_direction is not None and kernel_type == 'grad_log':
            if approaching_direction == 'interior':
                g[i, 0] -= (-2*math.pi) * 1/2*f[i, 0]
            elif approaching_direction == 'exterior':
                g[i, 0] += (-2*math.pi) * 1/2*f[i, 0]
            else:
                raise ValueError("approaching_direction must be 'interior' or 'exterior'")
    return g



def generate_curves_data(n_data, N, r0_range=0, freq_range=0.5, k_curve=4, k_feature=6, kernel_type='log', approaching_direction = None):
    nodes_list = []
    elems_list = []
    features_list = []
    for _ in tqdm(range(n_data), desc="Generating curves data"):
        nodes = random_polar_curve(N, k=k_curve, r0_range=r0_range, freq_range=freq_range)
        elems = sequential_elems(N)
        if kernel_type in ['grad_log']:
            normal_vector = compute_unit_normals(nodes, elems)
        else:
            normal_vector = None
        f = smooth_feature_f(N, k=k_feature)
        g = curve_integral_g(nodes, f, kernel_type, compute_node_measures(nodes, elems)[:,0], normal_vector=normal_vector, approaching_direction = approaching_direction)
        features = np.concatenate([f, g], axis=1)
        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
    return nodes_list, elems_list, features_list


def generate_curves_data_test(n_data, N, k_curve=4):
    nodes_list = []
    elems_list = []
    features_list = []
    for _ in tqdm(range(n_data), desc="Generating curves data"):
        nodes = random_polar_curve(N, k=k_curve)
        elems = sequential_elems(N)

        normal_vector = compute_unit_normals(nodes, elems)

        f1 = nodes[:, 0:1] + nodes[:, 1:2]  # Use x-coordinate as feature
        g1 = curve_integral_g(nodes, f1, 'grad_log', compute_node_measures(nodes, elems)[:,0], normal_vector=normal_vector, approaching_direction = 'interior')
        f2 = np.sum(normal_vector, axis=1, keepdims=True)
        g2 = curve_integral_g(nodes, f2, 'log', compute_node_measures(nodes, elems)[:,0], approaching_direction = 'interior')
        g = -(- g1 + g2)/(2*np.pi)
        features = np.concatenate([f1, g], axis=1)
        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
    return nodes_list, elems_list, features_list


def visualize_curve(nodes, features, elems):
    plt.figure(figsize=(16, 6))

    # 左图：曲线及外法向量
    plt.subplot(1, 3, 1)
    plt.plot(nodes[:, 0], nodes[:, 1], color='blue', alpha=0.5)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=20)
    normals = compute_unit_normals(nodes, elems)    
    plt.quiver(nodes[:, 0], nodes[:, 1], normals[:, 0], normals[:, 1], color='red', scale=20, width=0.005, alpha=0.7)
    plt.title('Random Polar Curve with Outward Normals')
    plt.axis('equal')

    # 中图: feature f
    plt.subplot(1, 3, 2)
    scatter_f = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 0], cmap='viridis', s=40)
    plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
    plt.colorbar(scatter_f, label='f(x)')
    plt.title('Feature f(x) on Random Polar Curve')
    plt.axis('equal')
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

    # 右图: feature g
    plt.subplot(1, 3, 3)
    scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 1], cmap='viridis', s=40)
    plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
    plt.colorbar(scatter_g, label='g(x)')
    plt.title('Feature g(x) on Random Polar Curve')
    plt.axis('equal')
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    np.random.seed(0)
    n_data = 1
    N = 1000
    r0_range = 0.5
    freq_range = 1
    k_curve = 3
    k_feature = 3
    nodes_list, elems_list, features_list = generate_curves_data(n_data, N, r0_range=r0_range, freq_range=freq_range, k_curve=k_curve, k_feature=k_feature, kernel_type='grad_log'
                                                                #  ,approaching_direction = 'interior'
                                                                 )
    # nodes_list, elems_list, features_list = generate_curves_data_test(n_data, N, k_curve=k_curve)
    print("nodes_list(array) shape:", np.array(nodes_list).shape)
    print("elems_list(array) shape:", np.array(elems_list).shape)
    print("features_list(array) shape:", np.array(features_list).shape)
    # np.savez(f"../../data/curve/curve_data_{k_curve}_{k_feature}.npz", nodes_list=nodes_list, elems_list=elems_list, features_list=features_list)
    visualize_curve(nodes_list[0], features_list[0], elems_list[0])

    # nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = "vertex_centered", adjacent_type="edge")
    # node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    # node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    # np.savez_compressed("../../data/curve/pcno_curve_data_1_1_3_3_grad.npz", \
    #                     nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
    #                     node_measures_raw = node_measures_raw, \
    #                     node_measures=node_measures, node_weights=node_weights, \
    #                     node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
    #                     features=features, \
    #                     directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 