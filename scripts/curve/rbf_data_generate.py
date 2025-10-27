import numpy as np
import math
from tqdm import tqdm
import os

import sys
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from pcno.geo_utility import compute_node_measures
from pcno.geo_utility import preprocess_data_mesh, compute_node_weights

def random_polar_curve_batch(Ndata, N, k=4, r0_scale = 1, freq_scale = 1, deform = True, deform_configs = []):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    t = np.repeat(t[np.newaxis, ...], Ndata, axis=0)
    r_base = r0_scale * np.random.uniform(-1, 1, (Ndata, 1))
    r = np.repeat(r_base, N, axis = -1)
    for i in range(1, k + 1):
        a_sin = np.random.uniform(-1, 1, (Ndata,1))
        a_sin = np.repeat(a_sin, N, axis = -1) * freq_scale / math.sqrt(i)
        a_cos = np.random.uniform(-1, 1, (Ndata,1))
        a_cos = np.repeat(a_cos, N, axis = -1) * freq_scale / math.sqrt(i)
        r += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
    r = np.tanh(r) + 1.5
    x = r * np.cos(t)
    y = r * np.sin(t)
    print(x.shape,y.shape)
    nodes = np.stack([x, y], axis=-1)
    print(nodes.shape)
    if deform:
        for i in range(Ndata):
            nodes[i, ...] = deform_rbf(nodes[i, ...], *deform_configs)
    return nodes


def deform_rbf(nodes, M=50, sigma=1, epsilon = 0.1, bbox=[-3,3,-3,3]):
    """
    M: 基函数个数
    bbox: [xmin,xmax,ymin,ymax]  基心采样区域
    sigma: 高斯半径

    rbf(x) = sum_{i=1}^{M} w_i exp(-||x-c_i||^2/(2*sigma^2))
    weights w_i ~ N(0,1) * U(0.2,1.0)
    nodes <- nodes + epsilon * rbf(nodes)

    """

    xmin,xmax,ymin,ymax = bbox
    centers = np.column_stack([
        np.random.uniform(xmin, xmax, size=M),
        np.random.uniform(ymin, ymax, size=M),
    ])

    weights = np.random.randn(M, 2)
    weights *= np.random.uniform(0.2, 1.0, size=(M,1))

    def field(pts):
        # pts: (N,2)
        # u(pts) (N,2)
        d2 = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,M)
        K = np.exp(-0.5 * d2 / (sigma**2))  # (N,M)
        u = K.dot(weights)  # (N,2)
        u = u / (np.linalg.norm(weights, axis=1).mean() + 1e-8)
        return u

    return nodes + epsilon * field(nodes)


def smooth_feature_f(Ndata, N, k=6):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    t = np.repeat(t[np.newaxis, ...], Ndata, axis=0)
    f = np.zeros((Ndata,N))
    for i in range(1, k + 1):
        a_sin = np.random.uniform(0.5, 1.5, (Ndata,1))
        a_sin = np.repeat(a_sin, N, axis = -1) * freq_scale / math.sqrt(i)
        a_cos = np.random.uniform(0.5, 1.5, (Ndata,1))
        a_cos = np.repeat(a_cos, N, axis = -1) * freq_scale / math.sqrt(i)
        f += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
    f = np.tanh(f) + 0.5 * np.sin(f)
    return f[...,np.newaxis]

def compute_unit_normals(nodes):
    """
    Compute outward unit normals for a closed planar curve.
    Args:
        nodes (np.ndarray): (N,2) array of ordered points along the closed curve.
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
    

def generate_curves_data(n_data, N, r0_scale=0, freq_scale=0.5, k_curve=4, k_feature=6, kernel_type='log', approaching_direction = None, deform = True, deform_configs = []):
    normals_list = []
    nodes = random_polar_curve_batch(n_data, N, k=k_curve, r0_scale=r0_scale, freq_scale=freq_scale, deform=deform, deform_configs=deform_configs)
    elems = np.stack((np.arange(N), np.roll(np.arange(N), -1)), axis=1)
    fin = smooth_feature_f(n_data, N, k=k_feature)
    fout = np.zeros_like(fin)
    for i in tqdm(range(n_data), desc="Generating curves data"):
        normal_vector = compute_unit_normals(nodes)
        if kernel_type == "log":
            kernel = log_kernel
        elif kernel_type == "nebla_log":
            kernel = nebla_log_normal_kernel
        else:
            raise NotImplementedError("没定义过啥是 "+kernel_type+" kernel! ")
        fout[i,...] = integral_operator(nodes[i,...], elems, kernel, fin[i,...], nodes[i,...])[...,np.newaxis]
        normals_list.append(normal_vector)
    features_list = [np.concatenate((fin[i], fout[i]), axis=-1) for i in range(len(fin))]
    return nodes, features_list, normals_list
    
def integral_operator(ypoints, elems, kernel, fin, xpoints, num_quad_points=3):
    """
    Compute fout(x) = ∫ fin(y) * kernel(x , y) dy for each node x in 'xpoints'
    
    Parameters:
    - ypoints: ndarray of shape (my,) : coordinates of nodes on the boundary 
    - elems: ndarray of shape (melems, 2) : indices of nodes forming elements, the outward is on the right
    - kernel: function with vectorization R^dx * R^dy -> R^{dx \times dy} : kernel(x , y)
    - fin: array of shape (my) : input function values on each node y
    - xpoints: ndarray of shape (mx,) : coordinates of nodes for computing fout
    - num_quad_points: int : number of Gaussian quadrature points
    
    Returns:
    - fout: ndarray of shape (mx,) : result of integral at each node x
    """
    my, mx = len(ypoints), len(xpoints)
    fout = np.zeros(mx)
    
    # Quadrature points and weights on reference element [-1, 1]
    quad_points, quad_weights = np.polynomial.legendre.leggauss(num_quad_points)
        
    
    # Loop over all elements
    for elem in elems:
        p0, p1 = ypoints[elem[0]], ypoints[elem[1]]
        J = (p1 - p0) / 2  # Jacobian of the affine map

        normal = np.array([J[1], -J[0]]) / np.linalg.norm(J)  # outward unit normal, pointing to the right 
        f0, f1 = fin[elem[0]], fin[elem[1]]

        # compute the contribution from this element to all points
        y = (p0 + p1)/2 + np.outer(quad_points, J)
        f = (f0 + f1)/2 + (f1 - f0) / 2 * quad_points
        K = kernel(xpoints, y, normal)
        fout += K.dot( (quad_weights * f) * np.linalg.norm(J) )
    
    return fout

def log_kernel(x, y, n): # 这里的n其实是法向量，只是在这个核函数中没用到
    # ln(||x - y||)
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # shape (mx, my, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)    # shape (mx, my)

    return np.log(distance_matrix)
    
def nebla_log_normal_kernel(x, y, n):
    # (x - y) · n / ||y - x||^2
    diff = - x[:, np.newaxis, :] + y[np.newaxis, :, :]  # shape (mx, my, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)    # shape (mx, my)
    dot_product = np.einsum('ijk,k->ij', diff, n)      # shape (mx, my)

    return dot_product / (distance_matrix**2) 

def smooth_kernel(x, y, n):
    # (x-y) · n
    diff = -x[:, np.newaxis, :] + y[np.newaxis, :, :]
    dot_product = np.einsum("ijk,k->ij", diff, n)

    return dot_product


def visualize_curve(nodes, features, elems, figurename = ''):
    plt.figure(figsize=(16, 6))

    # 左图：曲线及外法向量
    plt.subplot(1, 3, 1)
    plt.plot(nodes[:, 0], nodes[:, 1], color='blue', alpha=0.5)
    plt.plot(nodes[[-1,0],0], nodes[[-1,0],1], color = "blue", alpha=0.5)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=20)
    normals = compute_unit_normals(nodes)    
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
    if figurename:
        plt.savefig(figurename)
    else:
        plt.show()



if __name__ == "__main__":
    np.random.seed(10000)
    epsilon = 0.03
    n_data = 10
    N = 1000
    r0_scale = 1
    freq_scale = 1
    k_curve = 10
    k_feature = 10
    kernel_type = 'grad_log'
    deform = True
    deform_configs = [100, 1, 0.1, [-2.5,2.5,-2.5,2.5]]   # M, sigma, epsilon, bbox
    elems = np.stack((np.arange(N), np.roll(np.arange(N), -1)), axis=1)
    elems_list = [elems] * N
    nodes_list, features_list, normals_list = generate_curves_data(
        n_data, N, r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve, k_feature=k_feature,
                                                                kernel_type='log',
                                                                deform = deform, 
                                                                deform_configs = deform_configs
                                                                #approaching_direction = 'interior'
                                                                )
    
    print("nodes_list(array) shape:", np.array(nodes_list).shape)
    print("elems_list(array) shape:", np.array(elems_list).shape)
    print("features_list(array) shape:", np.array(features_list).shape)
    np.savez(f"curve_data_{k_curve}_{k_feature}_withnorm_1000.npz", nodes_list=nodes_list, elems_list=elems_list, features_list=features_list, normals_list=normals_list)
    visualize_curve(nodes_list[4], features_list[4], elems_list[4]
                    # , figurename = f'figures/deformed.png'
                    )