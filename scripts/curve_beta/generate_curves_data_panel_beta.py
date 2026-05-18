import numpy as np
import math
from tqdm import tqdm
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from pcno.geo_utility import preprocess_data_mesh, compute_node_weights


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------

def random_polar_curve(N, k=4, r0_scale=1, freq_scale=1, deform=True, deform_configs=[]):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_base = r0_scale * np.random.uniform(-1, 1)
    r = np.full_like(t, r_base)
    for i in range(1, k + 1):
        a_sin = freq_scale / math.sqrt(i) * np.random.uniform(-1, 1)
        a_cos = freq_scale / math.sqrt(i) * np.random.uniform(-1, 1)
        r += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
    r = np.tanh(r) + 1.5
    x = r * np.cos(t)
    y = r * np.sin(t)
    nodes = np.stack([x, y], axis=1)
    if deform:
        nodes = deform_rbf(nodes, *deform_configs)
    return nodes


def deform_rbf(nodes, M=50, sigma=1, epsilon=0.1, bbox=[-3, 3, -3, 3]):
    xmin, xmax, ymin, ymax = bbox
    centers = np.column_stack([
        np.random.uniform(xmin, xmax, size=M),
        np.random.uniform(ymin, ymax, size=M),
    ])
    weights = np.random.randn(M, 2)
    weights *= np.random.uniform(0.2, 1.0, size=(M, 1))

    def field(pts):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        K = np.exp(-0.5 * d2 / (sigma ** 2))
        u = K.dot(weights)
        u = u / (np.linalg.norm(weights, axis=1).mean() + 1e-8)
        return u

    for _ in range(10):
        nodes = nodes + epsilon / 10 * field(nodes)

    abs_max = np.amax(np.abs(nodes))
    if abs_max > 2.5:
        eps = np.random.uniform(0.0, 0.2)
        nodes = nodes / (abs_max / 2.5) * (1 - eps)

    return nodes


def smooth_feature_f(points, f_random_config, num_features=1):
    def smooth_feature_f_2d(points, M=200, sigma=(0.5, 1.5)):
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        centers = np.column_stack([
            np.random.uniform(xmin - 1, xmax + 1, size=M),
            np.random.uniform(ymin - 1, ymax + 1, size=M),
        ])
        if isinstance(sigma, (list, tuple)):
            sigmas = np.random.uniform(sigma[0], sigma[1], size=(1, M))
        else:
            sigmas = np.full((1, M), sigma)
        weights = np.random.randn(M, 1)
        weights *= np.random.uniform(0.2, 1, size=(M, 1))
        d2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        K = np.exp(-0.5 * d2 / (sigmas ** 2))
        f = K.dot(weights)
        f = np.tanh(f) + 0.5 * np.sin(f) + 0.2 * f
        return f.reshape(-1)

    def smooth_feature_f_1d(N, k=6):
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        f = np.zeros(N)
        for i in range(1, k + 1):
            a_sin = np.random.uniform(0.5, 1.5) / math.sqrt(i)
            a_cos = np.random.uniform(0.5, 1.5) / math.sqrt(i)
            f += a_sin * np.sin(i * t) + a_cos * np.cos(i * t)
        f = np.tanh(f) + 0.5 * np.sin(f)
        return f

    N = points.shape[0]
    f_list = []
    for _ in range(num_features):
        if f_random_config[0] == "1d":
            f_list.append(smooth_feature_f_1d(N, k=f_random_config[1]))
        elif f_random_config[0] == "2d":
            f_list.append(smooth_feature_f_2d(points))
        else:
            raise ValueError("Unknown f_random_config")
    return np.stack(f_list, axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Anisotropic PanelGeometry with beta parameter
# ---------------------------------------------------------------------------

class PanelGeometryBeta:
    """
    Panel geometry for the anisotropic single-layer Laplace kernel

        K_beta(x, y) = -1/(2*pi) * log( sqrt(x^2 + beta^2 * y^2) )

    where beta > 0 is an anisotropy parameter.  When beta = 1 this reduces
    to the standard isotropic sp_laplace kernel.

    Algorithm for integrating K_beta over a straight panel from A to B:
      1. Scale the y-coordinates of every point by 1/beta  (x -> x, y -> y/beta).
         In the scaled coordinate system the kernel becomes the standard
         isotropic -1/(2*pi) log(r) kernel.
      2. Apply the standard analytic panel integration formula in the scaled
         coordinate system.
      3. Multiply the result by  |AB|_orig / |AB|_scaled  (the ratio of the
         original panel length to its length after the y-scaling), so that
         the quadrature weight is expressed in terms of the original arc-length
         element ds.

    Vertices are assumed to be ordered counter-clockwise.
    """

    def __init__(self, vertices: np.ndarray, elems: np.ndarray,
                 beta: float = 1.0,
                 panel_component_ids: np.ndarray | None = None):
        """
        Args:
            vertices (ndarray): (n, 2)
            elems (ndarray): (m, 3), dtype int; each row [dummy, start_idx, end_idx]
            beta (float): anisotropy parameter beta > 0
            panel_component_ids (ndarray or None): (m,) component labels
        """
        assert beta > 0, "beta must be positive"
        self.beta = beta
        self.n_panels = elems.shape[0]
        self.elems = elems
        self.vertices = vertices
        self.panel_component_ids = panel_component_ids
        if self.panel_component_ids is not None:
            self.panel_component_ids = np.asarray(self.panel_component_ids, dtype=int)
            assert self.panel_component_ids.shape[0] == self.n_panels
        self._compute_panel_properties()

    def _compute_panel_properties(self):
        """
        Compute geometric properties of each panel in *original* coordinates.

        Also precompute the scaled-coordinate counterparts needed for the
        anisotropic kernel integration.
        """
        vertices = self.vertices
        elems = self.elems
        beta = self.beta

        # ---- original-coordinate properties --------------------------------
        self.panel_midpoints = (vertices[elems[:, 1]] + vertices[elems[:, 2]]) / 2
        d = vertices[elems[:, 2]] - vertices[elems[:, 1]]           # (m, 2)
        self.panel_lengths = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)  # (m,)
        self.panel_cosines = d[:, 0] / self.panel_lengths
        self.panel_sines   = d[:, 1] / self.panel_lengths
        self.out_normals   = np.column_stack((self.panel_sines, -self.panel_cosines))

        # ---- scaled-coordinate properties  (y -> y/beta) -------------------
        # scaled start / end vertices
        v_start_s = vertices[elems[:, 1]].copy()   # (m, 2)
        v_end_s   = vertices[elems[:, 2]].copy()   # (m, 2)
        v_start_s[:, 1] *= beta
        v_end_s[:, 1]   *= beta

        d_s = v_end_s - v_start_s                                       # (m, 2)
        self._panel_lengths_scaled = np.sqrt(d_s[:, 0] ** 2 + d_s[:, 1] ** 2)  # (m,)
        self._panel_cosines_scaled = d_s[:, 0] / self._panel_lengths_scaled
        self._panel_sines_scaled   = d_s[:, 1] / self._panel_lengths_scaled

        # ratio  |AB|_orig / |AB|_scaled  — this is the weight correction factor
        self._length_ratio = self.panel_lengths / self._panel_lengths_scaled  # (m,)

        # store scaled start vertices for local-frame construction
        self._v_start_scaled = v_start_s   # (m, 2)

    # ------------------------------------------------------------------
    # Core integration formula
    # ------------------------------------------------------------------

    def compute_points_kernel_coeffs(self, points: np.ndarray, kernel_type: str):
        """
        Compute  coeff_i = integral_{panel_i} K_beta(point, y) ds(y)
        for every evaluation point and every panel, using the scaled-
        coordinate trick described in the class docstring.

        Args:
            points: (N, 2)  evaluation points in *original* coordinates

        Returns:
            coeffs: (N, n_panels, 1)
        """
        beta = self.beta

        # Scale evaluation points: x unchanged, y -> y/beta
        pts_s = points.copy()                    # (N, 2)
        pts_s[:, 1] *= beta

        x0s = pts_s[:, 0]   # (N,)
        y0s = pts_s[:, 1]   # (N,)

        # Local coordinates in the *scaled* panel frame
        # (cos_s, sin_s are unit vectors along the scaled panel direction)
        cos_s = self._panel_cosines_scaled   # (m,)
        sin_s = self._panel_sines_scaled     # (m,)

        # Vector from scaled panel start to scaled evaluation point
        dx = x0s[:, None] - self._v_start_scaled[None, :, 0]   # (N, m)
        dy = y0s[:, None] - self._v_start_scaled[None, :, 1]   # (N, m)

        # Local tangential and normal coordinates in scaled frame
        x0_star = cos_s[None, :] * dx + sin_s[None, :] * dy   # (N, m)
        y0_star = -sin_s[None, :] * dx + cos_s[None, :] * dy  # (N, m)

        L_s = self._panel_lengths_scaled[None, :]  # (1, m)  scaled panel length

        # Distances from evaluation point to scaled panel endpoints
        r_start = np.sqrt(dx ** 2 + dy ** 2)                              # (N, m)
        r_end   = np.sqrt((x0s[:, None] - (self._v_start_scaled[None, :, 0] + cos_s[None, :] * self._panel_lengths_scaled[None, :])) ** 2 +
                          (y0s[:, None] - (self._v_start_scaled[None, :, 1] + sin_s[None, :] * self._panel_lengths_scaled[None, :])) ** 2)  # (N, m)

        # Collinear mask: evaluation point lies on the panel line
        collinear_mask = np.isclose(np.abs(y0_star), 0.0, atol=1e-10, rtol=1e-10)  # (N, m)

        if kernel_type == "sp_laplace":
            # Standard sp_laplace analytic formula (same as original code) applied
            # in the *scaled* coordinate system:
            #   I = (L - x0*) log(r_end) + x0* log(r_start) - L
            #       + y0* [ arctan((L - x0*)/y0*) + arctan(x0*/y0*) ]   (when not collinear)
            coeffs = ((L_s - x0_star) * np.log(r_end + 1e-300)
                    + x0_star * np.log(r_start + 1e-300)
                    - L_s)

            coeffs[~collinear_mask] += (
                y0_star[~collinear_mask] * (
                    np.arctan(((L_s - x0_star)[~collinear_mask]) / y0_star[~collinear_mask])
                    + np.arctan(x0_star[~collinear_mask] / y0_star[~collinear_mask])
                )
            )

            # Multiply by -1/(2*pi) (standard prefactor)
            # and by the length ratio  |AB|_orig / |AB|_scaled  (arc-length correction)
            coeffs = -coeffs / (2 * math.pi)                          # (N, m)
            coeffs = coeffs * self._length_ratio[None, :]             # (N, m)  correction

            return coeffs[..., np.newaxis]   # (N, m, 1)
        elif kernel_type == "dp_laplace": 
            # k(x,y) = (y-x)ny /|x-y|^2  * (-1/2pi)
            coeffs = np.zeros_like(x0_star)  # N, n_panels
            coeffs[~collinear_mask] = np.arctan((L_s - x0_star)[~collinear_mask] / y0_star[~collinear_mask]) + np.arctan(x0_star[~collinear_mask] / y0_star[~collinear_mask])
            coeffs = -coeffs / (2*math.pi)
            coeffs = coeffs * self._length_ratio[None, :]

            return coeffs[...,np.newaxis]

    # ------------------------------------------------------------------
    # Kernel integral  g(x) = int K_beta(x,y) f(y) ds(y)
    # ------------------------------------------------------------------

    def compute_kernel_integral(self, points: np.ndarray, f: np.ndarray, kernel_type: str):
        """
        Compute  g(x) = integral_{curve} K_beta(x, y) f(y) ds(y)

        Args:
            points: (N, 2)
            f: (n_panels, 1)  — single scalar feature per panel

        Returns:
            g: (N, 1)
        """
        coeffs = self.compute_points_kernel_coeffs(points, kernel_type)  # (N, n_panels, 1)
        coeffs = coeffs[..., 0]                              # (N, n_panels)
        g = np.einsum('Np,pk->Nk', coeffs, f)               # (N, 1)
        return g


# ---------------------------------------------------------------------------
# Data generation functions
# ---------------------------------------------------------------------------

def generate_curves_data_panel_beta(n_data, N, beta_range=(0.5, 1.0), r0_scale=0, freq_scale=0.5, k_curve=4, f_random_config=["2d"], kernel_type='sp_laplace', deform=True, deform_configs=[]):
    """
    Generate dataset for the anisotropic sp_laplace kernel.
    beta is drawn i.i.d. from Uniform(beta_range[0], beta_range[1]) for each sample.

    Args:
        n_data:      number of samples
        N:           number of panels per curve
        beta_range:  (beta_min, beta_max), both > 0

    Returns:
        nodes_list, elems_list, features_list, betas_list
        features layout per panel: [f, normal_x, normal_y, g]
        betas_list: list of length n_data, each entry is the scalar beta used
    """
    nodes_list    = []
    elems_list    = []
    features_list = []
    betas_list    = []

    beta_min, beta_max = beta_range
    for _ in tqdm(range(n_data), desc=f"Generating curves (beta~U[{beta_min},{beta_max}])"):
        beta = float(np.random.uniform(beta_min, beta_max))

        nodes = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                   freq_scale=freq_scale,
                                   deform=deform, deform_configs=deform_configs)
        elems = np.stack(
            [np.full(N, 1, dtype=int),
             np.arange(N),
             (np.arange(N) + 1) % N],
            axis=1)

        panel_geo = PanelGeometryBeta(
            nodes, elems, beta=beta,
            panel_component_ids=np.zeros(elems.shape[0], dtype=int))

        f = smooth_feature_f(panel_geo.panel_midpoints, f_random_config, num_features=1)  # (N, 1)
        g = panel_geo.compute_kernel_integral(panel_geo.panel_midpoints, f, kernel_type)  # (N, 1)

        features = np.concatenate([f, panel_geo.out_normals, g], axis=1)  # (N, 4)

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
        betas_list.append(beta)

    return nodes_list, elems_list, features_list, betas_list


def generate_curves_data_panel_beta_two(
        n_data, N,
        beta_range=(0.5, 1.0),
        r0_scale=0, freq_scale=0.5, k_curve=4,
        f_random_config=["2d"],
        kernel_type='sp_laplace',
        deform=True, deform_configs=[],
        compress_coeff=1.0):
    """
    Two-curve variant (analogous to generate_curves_data_panel_two).
    beta is drawn i.i.d. from Uniform(beta_range[0], beta_range[1]) for each sample.

    Returns:
        nodes_list, elems_list, features_list, betas_list
    """
    nodes_list    = []
    elems_list    = []
    features_list = []
    betas_list    = []

    beta_min, beta_max = beta_range
    for _ in tqdm(range(n_data), desc=f"Generating two-curve data (beta~U[{beta_min},{beta_max}])"):
        beta = float(np.random.uniform(beta_min, beta_max))
        nodes1 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                    freq_scale=freq_scale,
                                    deform=deform, deform_configs=deform_configs)
        nodes1[:, 0] = (nodes1[:, 0] + 2.5) * 0.49 - 2.5
        nodes1 = nodes1 * compress_coeff

        nodes2 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                    freq_scale=freq_scale,
                                    deform=deform, deform_configs=deform_configs)
        nodes2[:, 0] = (nodes2[:, 0] - 2.5) * 0.49 + 2.5
        nodes2 = nodes2 * compress_coeff

        nodes = np.concatenate([nodes1, nodes2], axis=0)

        elems1 = np.stack(
            [np.full(N, 1, dtype=int),
             np.arange(N),
             (np.arange(N) + 1) % N],
            axis=1)
        elems2 = np.stack(
            [np.full(N, 1, dtype=int),
             np.arange(N, 2 * N),
             (np.arange(N, 2 * N) + 1 - N) % N + N],
            axis=1)
        elems = np.concatenate([elems1, elems2], axis=0)

        comp_ids = np.concatenate(
            [np.zeros(elems1.shape[0], dtype=int),
             np.ones(elems2.shape[0], dtype=int)], axis=0)

        panel_geo = PanelGeometryBeta(nodes, elems, beta=beta,
                                      panel_component_ids=comp_ids)

        f = smooth_feature_f(panel_geo.panel_midpoints, f_random_config, num_features=1)
        g = panel_geo.compute_kernel_integral(panel_geo.panel_midpoints, f, kernel_type)

        features = np.concatenate([f, panel_geo.out_normals, g], axis=1)

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
        betas_list.append(beta)

    return nodes_list, elems_list, features_list, betas_list


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_curve_beta(nodes, features, elems, beta, figurename=''):
    """Visualise geometry, input feature f, and output feature g."""
    plt.figure(figsize=(16, 6))

    def draw_edges(ax, alpha=0.5, color='blue'):
        for elem in elems:
            idx = elem[1:]
            valid = idx[idx != -1]
            if len(valid) > 1:
                ax.plot(nodes[valid, 0], nodes[valid, 1],
                        color=color, linewidth=1, alpha=alpha)

    # --- geometry + normals
    ax1 = plt.subplot(1, 3, 1)
    draw_edges(ax1)
    normals = features[:, 1:3]
    ax1.quiver(nodes[:, 0], nodes[:, 1],
               normals[:, 0], normals[:, 1],
               color='red', scale=20, width=0.005, alpha=0.7)
    ax1.set_title(f'Curve + Outward Normals  (beta={beta:.3f})')
    ax1.set_aspect('equal')

    # --- input feature f
    ax2 = plt.subplot(1, 3, 2)
    sc = ax2.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 0],
                     cmap='viridis', s=40)
    plt.colorbar(sc, ax=ax2, label='f(x)')
    draw_edges(ax2, color='red', alpha=0.7)
    ax2.set_title('Input feature f(x)')
    ax2.set_aspect('equal')

    # --- output g
    ax3 = plt.subplot(1, 3, 3)
    sc = ax3.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 3],
                     cmap='viridis', s=40)
    plt.colorbar(sc, ax=ax3, label='g(x)')
    draw_edges(ax3, color='red', alpha=0.7)
    ax3.set_title('Output g(x) = K_beta * f')
    ax3.set_aspect('equal')

    plt.tight_layout()
    if figurename:
        plt.savefig(figurename)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed       = 1000
    n_data     = 50000
    N          = 1000
    beta_range = (0.5, 1.0)    # beta ~ Uniform(0.5, 1.0), sampled per sample
    r0_scale   = 1
    freq_scale = 1
    k_curve    = 5
    f_random_config = ["2d"]
    kernel_type = 'sp_laplace'

    deform         = True
    deform_configs = [200, 1, 0.1, [-2.5, 2.5, -2.5, 2.5]]

    visualization            = True
    save_data_to_pcno_format = True
    two_curves               = False

    np.random.seed(seed)

    if not two_curves:
        nodes_list, elems_list, features_list, betas_list = generate_curves_data_panel_beta(
            n_data, N,
            beta_range=beta_range,
            r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve,
            f_random_config=f_random_config,
            kernel_type=kernel_type,
            deform=deform, deform_configs=deform_configs)
    else:
        nodes_list, elems_list, features_list, betas_list = generate_curves_data_panel_beta_two(
            n_data, N,
            beta_range=beta_range,
            r0_scale=r0_scale, freq_scale=freq_scale, k_curve=k_curve,
            f_random_config=f_random_config,
            kernel_type=kernel_type,
            deform=deform, deform_configs=deform_configs,
            compress_coeff=0.9)

    betas_array = np.array(betas_list)   # (n_data,)

    print("nodes_list shape   :", np.array(nodes_list).shape)
    print("elems_list shape   :", np.array(elems_list).shape)
    print("features_list shape:", np.array(features_list).shape)
    print("betas shape        :", betas_array.shape,
          f"  min={betas_array.min():.4f}  max={betas_array.max():.4f}")

    features_at_nodes_list = (np.array(features_list) + np.roll(np.array(features_list), 1, axis=1)) / 2

    if visualization:
        visualize_curve_beta(
            nodes_list[0],
            features_at_nodes_list[0],
            elems_list[0],
            beta=betas_list[0],
            figurename='panel_beta.png')

    if save_data_to_pcno_format:
        tag  = "two_curves" if two_curves else "single"
        name = (f"../../data/curve_beta/pcno_curve_data"
                f"_{r0_scale}_{freq_scale}_{k_curve}_beta{beta_range}"
                f"_{f_random_config[-1]}_{kernel_type}_beta_random"
                f"_panel_{tag}_5000.npz")

        nnodes, node_mask, nodes, node_measures_raw, features, \
            directed_edges, edge_gradient_weights = preprocess_data_mesh(
                nodes_list, elems_list, features_list,
                mesh_type="cell_centered", adjacent_type="nodes")

        node_measures, node_weights = compute_node_weights(
            nnodes, node_measures_raw, equal_measure=False)
        node_equal_measures, node_equal_weights = compute_node_weights(
            nnodes, node_measures_raw, equal_measure=True)

        np.savez_compressed(
            name,
            nnodes=nnodes, node_mask=node_mask, nodes=nodes,
            node_measures_raw=node_measures_raw,
            node_measures=node_measures, node_weights=node_weights,
            node_equal_measures=node_equal_measures,
            node_equal_weights=node_equal_weights,
            features=features,
            directed_edges=directed_edges,
            edge_gradient_weights=edge_gradient_weights,
            betas=betas_array)   # shape (n_data,), one scalar beta per sample