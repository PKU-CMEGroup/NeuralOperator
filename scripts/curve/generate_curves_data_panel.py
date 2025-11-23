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

def random_polar_curve(N, k=4, r0_scale = 1, freq_scale = 1, deform = True, deform_configs = []):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_base = r0_scale * np.random.uniform(-1, 1)
    r = np.full_like(t, r_base)
    for i in range(1, k + 1):
        a_sin = freq_scale/math.sqrt(i)*np.random.uniform(-1, 1)
        a_cos = freq_scale/math.sqrt(i)*np.random.uniform(-1, 1)
        r += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
    r = np.tanh(r) + 1.5
    x = r * np.cos(t)
    y = r * np.sin(t)
    nodes = np.stack([x, y], axis=1)
    if deform:
        nodes = deform_rbf(nodes, *deform_configs)
    return nodes


def deform_rbf(nodes, M=50, sigma=1, epsilon = 0.1, bbox=[-3,3,-3,3]):
    """
    M: number of RBF centers
    bbox: [xmin,xmax,ymin,ymax]  sampling region for centers
    sigma: Gaussian radius

    rbf(x) = sum_{i=1}^{M} w_i exp(-||x-c_i||^2/(2*sigma^2))
    weights w_i ~ N(0,1) * U(0.2,1.0)
    nodes <- nodes + epsilon * rbf(nodes)

    Args:
        nodes: (N,2)
    Returns:
        nodes_deformed: (N,2)
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


    for _ in range(10):
        nodes = nodes + epsilon/10 * field(nodes)

    abs_max = np.amax(np.abs(nodes))
    if abs_max > 2.5:
        eps = np.random.uniform(0.0, 0.2)
        nodes = nodes / (abs_max / 2.5) * (1-eps)

    return nodes





def smooth_feature_f(points, f_random_config, num_features=1):
    '''
    points: (N,2)
    f_random_config: [random_dim, k(if 1d)]
    Returns:
    f: (N, num_features)
    '''
    def smooth_feature_f_2d(points, M = 200, sigma = (0.5, 1.5)):
        '''
        points: (N,2)
        Returns:
        f: (N,1)
        '''

        xmin, xmax = points[:,0].min(), points[:,0].max()
        ymin, ymax = points[:,1].min(), points[:,1].max()
        centers = np.column_stack([
            np.random.uniform(xmin-1, xmax+1, size=M),
            np.random.uniform(ymin-1, ymax+1, size=M),
        ])

        if isinstance(sigma, (list, tuple)):
            sigmas = np.random.uniform(sigma[0], sigma[1], size=(1, M))
        else:
            sigmas = np.full((1, M), sigma)

        weights = np.random.randn(M, 1)
        weights *= np.random.uniform(0.2, 1, size=(M,1))
        d2 = np.sum((points[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,M)
        K = np.exp(-0.5 * d2 / (sigmas**2))  # (N,M)
        f = K.dot(weights)  # (N,1)
        f = np.tanh(f) + 0.5 * np.sin(f) + 0.2*f
        return f.reshape(-1)


    def smooth_feature_f_1d(N, k=6):
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        f = np.zeros(N)
        for i in range(1, k + 1):
            a_sin = np.random.uniform(0.5, 1.5)/math.sqrt(i)
            a_cos = np.random.uniform(0.5, 1.5)/math.sqrt(i)
            f += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
        f = np.tanh(f) + 0.5 * np.sin(f)
        return f

    N = points.shape[0]
    f_list = []
    for _ in range(num_features):
        if f_random_config[0] == "1d":
            f_list.append(smooth_feature_f_1d(N, k = f_random_config[1]))  # (N)
        elif f_random_config[0] == "2d":
            f_list.append(smooth_feature_f_2d(points))  # (N)
        else:
            raise ValueError("Unknown f_random_config")

    return np.stack(f_list, axis=-1)  # (N, feature_dim)


class PanelGeometry:
    """
    Assume curve vertices are ordered counter-clockwise.
    """
    def __init__(self, vertices: np.ndarray):
        """
        Args:
            vertices (ndarray) (n, 2)
        """

        self.n_panels = vertices.shape[0]
        self.vertices = vertices
        self._compute_panel_properties()

    def _compute_panel_properties(self):
        '''
        panel_midpoints: (n,2)
        panel_lengths: (n,)
        panel_cosines: (n,)
        panel_sines: (n,)
        normals: (n,2)
        elems: (n,3)  [1, i, i+1]
        '''

        vertices = self.vertices
        self.panel_midpoints = (np.roll(vertices, -1, axis=0) + vertices) / 2.0
        
        d = np.roll(vertices, -1, axis=0) - vertices
        self.panel_lengths = np.sqrt(d[:,0]**2 + d[:,1]**2)

        self.panel_cosines = d[:,0] / self.panel_lengths
        self.panel_sines = d[:,1] / self.panel_lengths
        self.out_normals = np.column_stack((self.panel_sines, -self.panel_cosines))
        self.elems = np.stack([np.full(self.n_panels, 1, dtype=int), np.arange(self.n_panels), (np.arange(self.n_panels) + 1) % self.n_panels], axis=1)
    
    def compute_points_kernel_coeffs(self, points: np.ndarray, kernel_type: str):
        '''
        coeff_i = int_{panel_i} K(point, y) dy

        Args:
            points: (N, 2)  
            kernel_type: 'log' or 'grad_log' or 'stokes'
        Returns:
            coeffs: (N, n_panels, dim_kernel)
        '''
        x0 = points[:, 0]  # N
        y0 = points[:, 1]  # N
        x0_stars = self.panel_cosines*(x0[:, None] - self.vertices[None, :,0]) + self.panel_sines*(y0[:, None] - self.vertices[None, :,1])  # N, n_panels
        y0_stars = -self.panel_sines*(x0[:, None] - self.vertices[None, :,0]) + self.panel_cosines*(y0[:, None] - self.vertices[None, :,1])  # N, n_panels

        r_lengths = np.sqrt((x0[:, None] - self.vertices[None, :,0])**2+(y0[:, None] - self.vertices[None, :,1])**2)  # N, n_panels
        r_lengths_roll = np.roll(r_lengths, -1, axis=1)  # N, n_panels
        collinear_mask = np.isclose(np.abs(y0_stars), 0.0, atol=1e-10, rtol=1e-10)
        

        if kernel_type == "log": 
            # k(x,y) = ln(|x-y|) * (-1/2pi)
            coeffs = ( (self.panel_lengths[None,...]  - x0_stars)*np.log(r_lengths_roll)  + x0_stars*np.log(r_lengths) - self.panel_lengths[None,...])
            coeffs[~collinear_mask] += y0_stars[~collinear_mask] * (np.arctan(((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask]) / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])) 
            coeffs = -coeffs[...,np.newaxis]/(2*math.pi) 
        elif kernel_type == "grad_log": 
            # k(x,y) = (y-x)ny /|x-y|^2  * (-1/2pi)
            coeffs = np.zeros_like(x0_stars)  # N, n_panels
            coeffs[~collinear_mask] = np.arctan((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask] / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])
            coeffs = -coeffs[...,np.newaxis]/(2*math.pi)
        elif kernel_type == "stokes": 
            # k(x,y) = (-ln|x-y| I + (x-y)(x-y)^T/|x-y|^2 ) ny / (4pi)
            coeffs0 = (self.panel_lengths[None,...]  - x0_stars)*np.log(r_lengths_roll)  + x0_stars*np.log(r_lengths) - self.panel_lengths[None,...]
            angle_term = np.zeros_like(x0_stars)  # N, n_panels
            angle_term[~collinear_mask] = y0_stars[~collinear_mask] * (np.arctan(((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask]) / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])) 
            coeffs0[~collinear_mask] += angle_term[~collinear_mask]

            coeffs_11 = np.zeros_like(x0_stars) + self.panel_lengths[None,...]
            coeffs_11[~collinear_mask] += -angle_term[~collinear_mask]

            coeffs_12 = np.zeros_like(x0_stars)  # N, n_panels
            coeffs_12[~collinear_mask] = y0_stars[~collinear_mask] * (np.log(r_lengths) - np.log(r_lengths_roll))[~collinear_mask]

            coeffs_22 = np.zeros_like(x0_stars)
            coeffs_22[~collinear_mask] += angle_term[~collinear_mask]

            coeffs = np.stack([coeffs_11, coeffs_12, coeffs_12, coeffs_22], axis=-1)  # N, n_panels, 4
            Q = np.stack([self.panel_cosines, self.panel_sines, -self.panel_sines, self.panel_cosines], axis=-1).reshape(self.n_panels, 2,2)  # n_panels, 2,2
            coeffs = np.einsum('ijkl,jlm->ijkm', coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2,2), Q)  # N, n_panels, 2,2
            coeffs = np.einsum('jkl,ijkm->ijlm', Q, coeffs)  # N, n_panels, 2,2
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 4)
            
            coeffs[...,0] -= coeffs0
            coeffs[...,3] -= coeffs0
            coeffs = coeffs/(4*math.pi)

        return coeffs
    
    def compute_kernel_integral(self, points: np.ndarray, f: np.ndarray, kernel_type: str):
        '''
        g(x) = int_{curve} K(x,y) f(y) dy

        Args:
            points: (N, 2)
            f: (n_panels, n_features) 
            kernel_type: 'log' or 'grad_log' or 'stokes'
        Returns:
            g: (n_panels, n_features) 
        '''
        coeffs = self.compute_points_kernel_coeffs(points, kernel_type)  # N, n_panels, dim_kernel
        if kernel_type == "log" or kernel_type == "grad_log":
            coeffs = coeffs[...,0]  # N, n_panels
            g = np.einsum('ij,jl->il', coeffs, f)  # N, n_features
        elif kernel_type == "stokes":
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2, 2)  # N, n_panels, 2,2
            assert f.shape[1] == 2, "f must have 2 features for stokes kernel"
            g = np.einsum('ijkl,jl->ik', coeffs, f)  # N, n_features


        return g


def generate_curves_data_panel(n_data, N, r0_scale=0, freq_scale=0.5, k_curve=4, f_random_config = ["2d"],kernel_type='log', deform = True, deform_configs = []):
    nodes_list = []
    elems_list = []
    features_list = []

    for index in tqdm(range(n_data), desc="Generating curves data"):
        nodes = random_polar_curve(N, k=k_curve, r0_scale=r0_scale, freq_scale=freq_scale, deform = deform, deform_configs= deform_configs)
        panel_geo = PanelGeometry(nodes)
        if kernel_type == 'stokes':
            num_features = 2
        else:
            num_features = 1
        f = smooth_feature_f(panel_geo.panel_midpoints, f_random_config, num_features=num_features)  # N, num_features

        g = panel_geo.compute_kernel_integral(panel_geo.panel_midpoints, f, kernel_type)  # N, num_features
        features = np.concatenate([f, panel_geo.out_normals, g], axis=1)  # N, 2+2*num_features
        elems = panel_geo.elems

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)

    return nodes_list, elems_list, features_list




def visualize_curve(nodes, features, elems, figurename = ''):

    if features.shape[-1] == 4:
        plt.figure(figsize=(16, 6))
        # Left plot: curve and outward normals
        plt.subplot(1, 3, 1)
        plt.plot(nodes[:, 0], nodes[:, 1], color='blue', alpha=0.5)
        # plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=20)
        normals = features[:, 1:3] 
        plt.quiver(nodes[:, 0], nodes[:, 1], normals[:, 0], normals[:, 1], color='red', scale=20, width=0.005, alpha=0.7)
        plt.title('Random Polar Curve with Outward Normals')
        plt.axis('equal')

        # Middle plot: feature f
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

        # Right plot: feature g
        plt.subplot(1, 3, 3)
        scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, -1], cmap='viridis', s=40)
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
    elif features.shape[-1] == 6:
        plt.figure(figsize=(16, 12))
        # Left plot: curve and outward normals
        plt.subplot(2, 3, 1)
        plt.plot(nodes[:, 0], nodes[:, 1], color='blue', alpha=0.5)
        # plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=20)
        normals = features[:, 2:4] 
        plt.quiver(nodes[:, 0], nodes[:, 1], normals[:, 0], normals[:, 1], color='red', scale=20, width=0.005, alpha=0.7)
        plt.title('Random Polar Curve with Outward Normals')
        plt.axis('equal')

        # Middle plot: feature f1
        plt.subplot(2, 3, 2)
        scatter_f = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 0], cmap='viridis', s=40)
        plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_f, label='f(x)')
        plt.title('Feature f_1(x) on Random Polar Curve')
        plt.axis('equal')
        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

        # Middle plot: feature f2
        plt.subplot(2, 3, 5)
        scatter_f = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 1], cmap='viridis', s=40)
        plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_f, label='f(x)')
        plt.title('Feature f_2(x) on Random Polar Curve')
        plt.axis('equal')
        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

        # Right plot: feature g1
        plt.subplot(2, 3, 3)
        scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 4], cmap='viridis', s=40)
        plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_g, label='g(x)')
        plt.title('Feature g_1(x) on Random Polar Curve')
        plt.axis('equal')
        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

        # Right plot: feature g2
        plt.subplot(2, 3, 6)
        scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 5], cmap='viridis', s=40)
        plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_g, label='g(x)')
        plt.title('Feature g_2(x) on Random Polar Curve')
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
    seed = 1000
    n_data = 1
    N = 1000
    r0_scale = 1
    freq_scale = 1
    k_curve = 5
    f_random_config = ["2d"]
    kernel_type = 'stokes'

    deform = True
    deform_configs = [200, 1, 0.1, [-2.5,2.5,-2.5,2.5]]   # M, sigma, epsilon, bbox

    check_g_with_previous = True
    check_all_with_previous = False
    visualization = True
    save_data_to_pcno_format = False
    quality_test = True
    



    np.random.seed(seed)
    nodes_list, elems_list, features_list = generate_curves_data_panel(
        n_data, N, r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve, f_random_config = f_random_config,
                                                                kernel_type = kernel_type,
                                                                deform = deform, 
                                                                deform_configs = deform_configs

                                                                )
    print("nodes_list(array) shape:", np.array(nodes_list).shape)
    print("elems_list(array) shape:", np.array(elems_list).shape)
    print("features_list(array) shape:", np.array(features_list).shape)

    features_at_nodes_list = (features_list + np.roll(features_list, 1, axis=1))/2



    if visualization:
        visualize_curve(nodes_list[0], features_at_nodes_list[0], elems_list[0]
                        , figurename = f'figures/panel.png'
                        )


    if check_g_with_previous:
        from generate_curves_data_batch import curve_integral_g_batch, compute_unit_normals
        n_feature = 2 if kernel_type == 'stokes' else 1
        g_new = curve_integral_g_batch(nodes_list[0][np.newaxis, ...], features_list[0][..., 0:n_feature][np.newaxis, ...], kernel_type,
                                                    compute_node_measures(nodes_list[0], elems_list[0])[:,0][np.newaxis, ...], normal_vector= compute_unit_normals(nodes_list[0])[np.newaxis, ...])

        features_list_new = features_list.copy()
        features_list_new[0][:,-n_feature:] = g_new[0]
        if visualization:
            visualize_curve(nodes_list[0], features_list_new[0], elems_list[0]
                            , figurename = f'figures/previous.png'
                            )


        print("Rel error in g:", np.linalg.norm(features_at_nodes_list[0][:,-n_feature:] - features_list_new[0][:,-n_feature:])/np.linalg.norm(features_at_nodes_list[0][:,-n_feature:]))
        if visualization:
            visualize_curve(nodes_list[0], features_at_nodes_list[0] - features_list_new[0], elems_list[0]
                            , figurename = f'figures/error.png'
                            )

    if check_all_with_previous:
        from generate_curves_data_batch import generate_curves_data_batch
        np.random.seed(seed)
        nodes_list_prev, elems_list_prev, features_list_prev = generate_curves_data_batch(
            n_data, n_batch = n_data, N = N, r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve, f_random_config = f_random_config,
                                                                    kernel_type = kernel_type,
                                                                    deform = deform, 
                                                                    deform_configs = deform_configs
                                                                    )
        print("Rel error in nodes:", np.linalg.norm(np.array(nodes_list) - np.array(nodes_list_prev))/np.linalg.norm(np.array(nodes_list_prev)))
        print("Rel error in f:", np.linalg.norm(np.array(features_at_nodes_list)[...,0] - np.array(features_list_prev)[...,0])/np.linalg.norm(np.array(features_list_prev)[...,0]))
        print("Rel error in normals:", np.linalg.norm(np.array(features_at_nodes_list)[...,1:3] - np.array(features_list_prev)[...,1:3])/np.linalg.norm(np.array(features_list_prev)[...,1:3]))
        print("Rel error in g:", np.linalg.norm(np.array(features_at_nodes_list)[...,-1] - np.array(features_list_prev)[...,-1])/np.linalg.norm(np.array(features_list_prev)[...,-1]))
        if visualization:
            visualize_curve(nodes_list_prev[0], features_list_prev[0], elems_list_prev[0]
                            , figurename = f'figures/log_2.png'
                            )
    if save_data_to_pcno_format:
        nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = "cell_centered", adjacent_type="nodes")
        node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
        node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
        np.savez_compressed(f"../../data/curve/pcno_curve_data_{r0_scale}_{freq_scale}_{k_curve}_{f_random_config[-1]}_{kernel_type}_panel.npz", \
                            nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                            node_measures_raw = node_measures_raw, \
                            node_measures=node_measures, node_weights=node_weights, \
                            node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                            features=features, \
                            directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    if quality_test:
        from quality_test import assess_curve_quality
        report = assess_curve_quality(nodes_list)