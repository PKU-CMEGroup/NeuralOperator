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
            kernel_type: 'sp_laplace' or 'dp_laplace' or 'stokes' or 'modified_dp_laplace'
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
        

        if kernel_type == "sp_laplace": 
            # k(x,y) = ln(|x-y|) * (-1/2pi)
            coeffs = ( (self.panel_lengths[None,...]  - x0_stars)*np.log(r_lengths_roll)  + x0_stars*np.log(r_lengths) - self.panel_lengths[None,...])
            coeffs[~collinear_mask] += y0_stars[~collinear_mask] * (np.arctan(((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask]) / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])) 
            coeffs = -coeffs[...,np.newaxis]/(2*math.pi) 
        elif kernel_type == "dp_laplace": 
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
            coeffs = np.einsum('Npld,pdk->Nplk', coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2,2), Q)  # N, n_panels, 2,2
            coeffs = np.einsum('pdk,Npdl->Npkl', Q, coeffs)  # N, n_panels, 2,2

            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 4)
            
            coeffs[...,0] -= coeffs0
            coeffs[...,3] -= coeffs0
            coeffs = coeffs/(4*math.pi)
        elif kernel_type == "modified_dp_laplace": 
            # k(x,y) = (y-x) /|x-y|^2 * (-1/2pi)  
            coeffs1 = np.log(r_lengths) - np.log(r_lengths_roll) # N, n_panels
            coeffs2 = np.zeros_like(x0_stars)  # N, n_panels
            coeffs2[~collinear_mask] = np.arctan((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask] / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])
            coeffs = np.stack([coeffs1, coeffs2], axis=-1)  # N, n_panels, 2
            coeffs = coeffs/(2*math.pi)
            Q = np.stack([self.panel_cosines, self.panel_sines, -self.panel_sines, self.panel_cosines], axis=-1).reshape(self.n_panels, 2,2)  # n_panels, 2,2
            coeffs = np.einsum('pdk,Npd->Npk', Q, coeffs)  # N, n_panels, 2



        return coeffs
    
    def compute_kernel_integral(self, points: np.ndarray, f: np.ndarray, kernel_type: str):
        '''
        g(x) = int_{curve} K(x,y) f(y) dy

        Args:
            points: (N, 2)
            f: (n_panels, n_features) 
            kernel_type: 'sp_laplace' or 'dp_laplace' or 'stokes' or 'fredholm_laplace'
        Returns:
            g: (n_panels, n_features) 
        '''
        if kernel_type != "fredholm_laplace":
            coeffs = self.compute_points_kernel_coeffs(points, kernel_type) # N, n_panels, dim_kernel
        if kernel_type == "sp_laplace" or kernel_type == "dp_laplace":
            coeffs = coeffs[...,0]  # N, n_panels
            g = np.einsum('Np,pk->Nk', coeffs, f)  # N, n_features
        elif kernel_type == "stokes":
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2, 2)  # N, n_panels, 2,2
            assert f.shape[1] == 2, "f must have 2 features for stokes kernel"
            g = np.einsum('Npkl,pl->Nk', coeffs, f)  # N, n_features
        elif kernel_type == "modified_dp_laplace":
            g = np.einsum('Npd,p->Nd', coeffs, f[...,0])  # N, n_features
        elif kernel_type == "fredholm_laplace":
            # For Fredholm formulation, the evaluation points must be panel midpoints
            # assert np.allclose(points, self.panel_midpoints), "For 'fredholm_laplace', query points must be panel midpoints." 
            # rhs = f/2.0 + self.compute_kernel_integral(points, f, 'dp_laplace') # RHS = (1/2) f + ∫ K_dp(x, y) f dy
            rhs = f
            coeffs = self.compute_points_kernel_coeffs(points, 'sp_laplace')
            coeffs = coeffs[...,0]
            g = np.linalg.solve(coeffs, rhs)  # Solve  ∫ K_sp(x, y) g dy = rhs



        return g

class PanelGeometryNew:
    """
    Assume curve vertices are ordered counter-clockwise.
    """
    def __init__(self, vertices: np.ndarray, elems: np.ndarray):
        """
        Args:
            vertices (ndarray) (n, 2)
            elems (ndarray) (m, 3) dtype = int 此处我们要求每条边必须要是逆时针方向
        """

        self.n_panels = elems.shape[0]
        self.elems = elems
        self.vertices = vertices
        self._compute_panel_properties()

    def _compute_panel_properties(self):
        '''
        panel_midpoints: (m,2)
        panel_lengths: (m,)
        panel_cosines: (m,)
        panel_sines: (m,)
        normals: (m,2)
        '''

        vertices = self.vertices
        elems = self.elems
        self.panel_midpoints = (vertices[elems[:,1]] + vertices[elems[:,2]]) / 2
        
        d = vertices[elems[:,2]] - vertices[elems[:,1]]
        self.panel_lengths = np.sqrt(d[:,0]**2 + d[:,1]**2)

        self.panel_cosines = d[:,0] / self.panel_lengths
        self.panel_sines = d[:,1] / self.panel_lengths
        self.out_normals = np.column_stack((self.panel_sines, -self.panel_cosines))
    
    def compute_points_kernel_coeffs(self, points: np.ndarray, kernel_type: str):
        '''
        coeff_i = int_{panel_i} K(point, y) dy

        Args:
            points: (N, 2)  
            kernel_type: 'sp_laplace' or 'dp_laplace' or 'stokes' or 'modified_dp_laplace'
        Returns:
            coeffs: (N, n_panels, dim_kernel)
        '''
        x0 = points[:, 0]  # N
        y0 = points[:, 1]  # N
        x0_stars = self.panel_cosines*(x0[:, None] - self.vertices[None, self.elems[:,1], 0]) + self.panel_sines*(y0[:, None] - self.vertices[None, self.elems[:,1], 1])  # N, n_panels
        y0_stars = -self.panel_sines*(x0[:, None] - self.vertices[None, self.elems[:,1], 0]) + self.panel_cosines*(y0[:, None] - self.vertices[None, self.elems[:,1], 1])  # N, n_panels

        r_lengths = np.sqrt((x0[:, None] - self.vertices[None, self.elems[:,1], 0])**2+(y0[:, None] - self.vertices[None, self.elems[:,1], 1])**2)  # N, n_panels
        r_lengths_roll = np.sqrt((x0[:, None] - self.vertices[None, self.elems[:,2], 0])**2+(y0[:, None] - self.vertices[None, self.elems[:,2], 1])**2)  # N, n_panels

        collinear_mask = np.isclose(np.abs(y0_stars), 0.0, atol=1e-10, rtol=1e-10)
        

        if kernel_type == "sp_laplace": 
            # k(x,y) = ln(|x-y|) * (-1/2pi)
            coeffs = ( (self.panel_lengths[None,...]  - x0_stars)*np.log(r_lengths_roll)  + x0_stars*np.log(r_lengths) - self.panel_lengths[None,...])
            coeffs[~collinear_mask] += y0_stars[~collinear_mask] * (np.arctan(((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask]) / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])) 
            coeffs = -coeffs[...,np.newaxis]/(2*math.pi) 
        elif kernel_type == "dp_laplace": 
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
            coeffs = np.einsum('Npld,pdk->Nplk', coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2,2), Q)  # N, n_panels, 2,2
            coeffs = np.einsum('pdk,Npdl->Npkl', Q, coeffs)  # N, n_panels, 2,2

            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 4)
            
            coeffs[...,0] -= coeffs0
            coeffs[...,3] -= coeffs0
            coeffs = coeffs/(4*math.pi)
        elif kernel_type == "modified_dp_laplace": 
            # k(x,y) = (y-x) /|x-y|^2 * (-1/2pi)  
            coeffs1 = np.log(r_lengths) - np.log(r_lengths_roll) # N, n_panels
            coeffs2 = np.zeros_like(x0_stars)  # N, n_panels
            coeffs2[~collinear_mask] = np.arctan((self.panel_lengths[None,...]  - x0_stars)[~collinear_mask] / y0_stars[~collinear_mask]) + np.arctan(x0_stars[~collinear_mask] / y0_stars[~collinear_mask])
            coeffs = np.stack([coeffs1, coeffs2], axis=-1)  # N, n_panels, 2
            coeffs = coeffs/(2*math.pi)
            Q = np.stack([self.panel_cosines, self.panel_sines, -self.panel_sines, self.panel_cosines], axis=-1).reshape(self.n_panels, 2,2)  # n_panels, 2,2
            coeffs = np.einsum('pdk,Npd->Npk', Q, coeffs)  # N, n_panels, 2

        return coeffs
    
    def compute_kernel_integral(self, points: np.ndarray, f: np.ndarray, kernel_type: str):
        '''
        g(x) = int_{curve} K(x,y) f(y) dy

        Args:
            points: (N, 2)
            f: (n_panels, n_features) 
            kernel_type: 'sp_laplace' or 'dp_laplace' or 'stokes' or 'fredholm_laplace'
        Returns:
            g: (n_panels, n_features) 
        '''
        if kernel_type != "fredholm_laplace":
            coeffs = self.compute_points_kernel_coeffs(points, kernel_type) # N, n_panels, dim_kernel
        if kernel_type == "sp_laplace" or kernel_type == "dp_laplace":
            coeffs = coeffs[...,0]  # N, n_panels
            g = np.einsum('Np,pk->Nk', coeffs, f)  # N, n_features
        elif kernel_type == "stokes":
            coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 2, 2)  # N, n_panels, 2,2
            assert f.shape[1] == 2, "f must have 2 features for stokes kernel"
            g = np.einsum('Npkl,pl->Nk', coeffs, f)  # N, n_features
        elif kernel_type == "modified_dp_laplace":
            g = np.einsum('Npd,p->Nd', coeffs, f[...,0])  # N, n_features
        elif kernel_type == "fredholm_laplace":
            # For Fredholm formulation, the evaluation points must be panel midpoints
            # assert np.allclose(points, self.panel_midpoints), "For 'fredholm_laplace', query points must be panel midpoints." 
            # rhs = f/2.0 + self.compute_kernel_integral(points, f, 'dp_laplace') # RHS = (1/2) f + ∫ K_dp(x, y) f dy
            rhs = f # RHS = f
            coeffs = self.compute_points_kernel_coeffs(points, 'sp_laplace')
            coeffs = coeffs[...,0]
            g = np.linalg.solve(coeffs, rhs)  # Solve  ∫ K_sp(x, y) g dy = rhs

        return g

def generate_curves_data_panel(n_data, N, r0_scale=0, freq_scale=0.5, k_curve=4, f_random_config = ["2d"],kernel_type='sp_laplace', deform = True, deform_configs = []):
    nodes_list = []
    elems_list = []
    features_list = []

    for index in tqdm(range(n_data), desc="Generating curves data"):
        nodes = random_polar_curve(N, k=k_curve, r0_scale=r0_scale, freq_scale=freq_scale, deform = deform, deform_configs= deform_configs)
        elems = np.stack([np.full(N, 1, dtype=int), np.arange(N), (np.arange(N) + 1) % N], axis=1)

        panel_geo = PanelGeometryNew(nodes,elems)
        if kernel_type == 'stokes':
            num_features = 2
        else:
            num_features = 1
        f = smooth_feature_f(panel_geo.panel_midpoints, f_random_config, num_features=num_features)  # N, num_features

        g = panel_geo.compute_kernel_integral(panel_geo.panel_midpoints, f, kernel_type)  # N, num_features
        features = np.concatenate([f, panel_geo.out_normals, g], axis=1)  # N, 2 + num_features_in + num_features_out
        elems = panel_geo.elems

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)

    return nodes_list, elems_list, features_list
def generate_curves_data_panel_two(n_data, N, r0_scale=0, freq_scale=0.5, k_curve=4, f_random_config = ["2d"],kernel_type='sp_laplace', deform = True, deform_configs = []):
    nodes_list = []
    elems_list = []
    features_list = []

    for index in tqdm(range(n_data), desc="Generating curves data"):
        nodes1 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale, freq_scale=freq_scale, deform = deform, deform_configs= deform_configs)
        nodes1[:,0] = (nodes1[:,0] + 2.5) * 0.49 - 2.5
        nodes2 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale, freq_scale=freq_scale, deform = deform, deform_configs= deform_configs)
        nodes2[:,0] = (nodes2[:,0] - 2.5) * 0.49 + 2.5
        nodes = np.concatenate([nodes1, nodes2], axis=0)  # 2N, 2

        elems1 = np.stack([np.full(N, 1, dtype=int), np.arange(N), (np.arange(N) + 1) % N], axis=1)
        elems2 = np.stack([np.full(N, 1, dtype=int), np.arange(N, 2*N), (np.arange(N, 2*N) + 1 - N) % N + N], axis=1)
        elems = np.concatenate([elems1, elems2], axis=0)  # 2N, 3

        panel_geo = PanelGeometryNew(nodes,elems)
        if kernel_type == 'stokes':
            num_features = 2
        else:
            num_features = 1
        f = smooth_feature_f(panel_geo.panel_midpoints, f_random_config, num_features=num_features)  # N, num_features

        g = panel_geo.compute_kernel_integral(panel_geo.panel_midpoints, f, kernel_type)  # N, num_features
        features = np.concatenate([f, panel_geo.out_normals, g], axis=1)  # N, 2 + num_features_in + num_features_out
        elems = panel_geo.elems

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)

    return nodes_list, elems_list, features_list




def visualize_curve(nodes, features, elems, kernel_type, figurename = ''):

    out_dim = 2 if kernel_type in ['stokes', 'modified_dp_laplace'] else 1
    in_dim = 2 if kernel_type == 'stokes' else 1
    plt.figure(figsize=(16, 6*out_dim))
    # Left plot: curve and outward normals
    plt.subplot(out_dim, 3, 1)
    # plt.plot(nodes[:, 0], nodes[:, 1], color='blue', alpha=0.5)
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='blue', linewidth=1, alpha=0.5)
    # plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=20)
    normals = features[:, in_dim:in_dim+2] 
    plt.quiver(nodes[:, 0], nodes[:, 1], normals[:, 0], normals[:, 1], color='red', scale=20, width=0.005, alpha=0.7)
    plt.title('Random Polar Curve with Outward Normals')
    plt.axis('equal')

    # Middle plot: feature f
    plt.subplot(out_dim, 3, 2)
    scatter_f = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 0], cmap='viridis', s=40)
    # plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
    plt.colorbar(scatter_f, label='f1(x)')
    plt.title('Feature f1(x) on Random Polar Curve')
    plt.axis('equal')
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

    if in_dim == 2:
        plt.subplot(out_dim, 3, 2+3)
        scatter_f = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 1], cmap='viridis', s=40)
        # plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_f, label='f2(x)')
        plt.title('Feature f2(x) on Random Polar Curve')
        plt.axis('equal')
        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)

    # Right plot: feature g
    plt.subplot(out_dim, 3, 3)
    scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, in_dim+2], cmap='viridis', s=40)
    # plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
    plt.colorbar(scatter_g, label='g1(x)')
    plt.title('Feature g1(x) on Random Polar Curve')
    plt.axis('equal')
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)
    if out_dim == 2:
        plt.subplot(out_dim, 3, 3+3)
        scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, in_dim+3], cmap='viridis', s=40)
        # plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
        plt.colorbar(scatter_g, label='g2(x)')
        plt.title('Feature g2(x) on Random Polar Curve')
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
    n_data = 10000
    N = 1000
    r0_scale = 1
    freq_scale = 1
    k_curve = 5
    f_random_config = ["2d"]
    kernel_type = 'stokes'  # 'sp_laplace' or 'dp_laplace' or 'stokes' or 'modified_dp_laplace' or 'fredholm_laplace'

    deform = True
    deform_configs = [200, 1, 0.1, [-2.5,2.5,-2.5,2.5]]   # M, sigma, epsilon, bbox

    visualization = True
    save_data_to_pcno_format = True
    
    two_circles = True  # generate two circles data for interaction kernel testing


    np.random.seed(seed)
    if not two_circles:
        nodes_list, elems_list, features_list = generate_curves_data_panel(
        n_data, N, r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve, f_random_config = f_random_config,
                                                                kernel_type = kernel_type,
                                                                deform = deform, 
                                                                deform_configs = deform_configs)
    else:
        nodes_list, elems_list, features_list = generate_curves_data_panel_two(
        n_data, N, r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve, f_random_config = f_random_config,
                                                                kernel_type = kernel_type,
                                                                deform = deform, 
                                                                deform_configs = deform_configs)

    print("nodes_list(array) shape:", np.array(nodes_list).shape)
    print("elems_list(array) shape:", np.array(elems_list).shape)
    print("features_list(array) shape:", np.array(features_list).shape)

    features_at_nodes_list = (features_list + np.roll(features_list, 1, axis=1))/2
    out_dim = 2 if kernel_type in ['stokes', 'modified_dp_laplace'] else 1
    in_dim = 1 if kernel_type in ['stokes'] else 1


    if visualization:
        visualize_curve(nodes_list[0], features_at_nodes_list[0], elems_list[0], kernel_type
                        , figurename = f'panel.png'
                        )

    if save_data_to_pcno_format:
        name = f"pcno_curve_data_{r0_scale}_{freq_scale}_{k_curve}_{f_random_config[-1]}_{kernel_type}_panel" + ("_two_circles" if two_circles else "") + ".npz"
        nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = "cell_centered", adjacent_type="nodes")
        node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
        node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
        np.savez_compressed(name, \
                            nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                            node_measures_raw = node_measures_raw, \
                            node_measures=node_measures, node_weights=node_weights, \
                            node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                            features=features, \
                            directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)