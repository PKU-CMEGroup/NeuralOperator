import numpy as np
import math
from tqdm import tqdm
import os
import argparse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from scripts.subsonic_panelmethod.Subsonic_PanelMethod2D.geometry_beta import PanelGeometryBeta
from scripts.subsonic_panelmethod.Subsonic_PanelMethod2D.assembly import solve_panel_method

def random_polar_curve(N, k=4, r0_scale=1, freq_scale=1, deform=True, deform_configs=[]):
    t = np.linspace(0, -2 * np.pi, N, endpoint=False)
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

# ---------------------------------------------------------------------------
# Data generation functions
# ---------------------------------------------------------------------------

def generate_curves_data_panel_method_subsonic(
        n_data, N,
        beta_range=(0.5, 1.0),
        r0_scale=0, freq_scale=0.5, k_curve=4,
        kernel_type='panel_method',
        deform=True, deform_configs=[],
        ):
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

    if kernel_type in ["panel_method"]:
        for _ in tqdm(range(n_data), desc=f"Generating curves (beta~U[{beta_min},{beta_max}])"):
            # beta = \sqrt{1-Mach^2}
            beta = float(np.random.uniform(beta_min, beta_max))
            Mach = np.sqrt(1-beta**2)

            # nodes: (n_panels,2)
            nodes = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                    freq_scale=freq_scale,
                                    deform=deform, deform_configs=deform_configs)
            vertices = np.vstack([nodes,nodes[0]]) # vertices: (n_panels+1,2)
            # elems: (n_panels,3)
            elems = np.stack(
                [np.full(N, 1, dtype=int),
                np.arange(N),
                (np.arange(N) + 1) % N],
                axis=1)

            panel_geo = PanelGeometryBeta(vertices, generate_lift=False, beta=beta)
            # out_normals: (n_panels, 2)
            out_normals   = np.column_stack((panel_geo.panel_sines, -panel_geo.panel_cosines))

            _,_,_,Cp = solve_panel_method([panel_geo], 0.0, 1.0, Mach=Mach) # Cp: (n_panels,1)
            g = Cp[0][:,np.newaxis] # g: (n_panels, 1)
            features = np.concatenate([out_normals, g], axis=1)  # (N, 3)

            nodes_list.append(nodes)
            elems_list.append(elems)
            features_list.append(features)
            betas_list.append(beta)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    return nodes_list, elems_list, features_list, betas_list


def generate_curves_data_panel_method_subsonic_two(
        n_data, N,
        beta_range=(0.5, 1.0),
        r0_scale=0, freq_scale=0.5, k_curve=4,
        kernel_type='sp_laplace',
        deform=True, deform_configs=[],
        compress_coeff=1.0,
        ):
    """
    Two-curve variant.  Supports both single-beta and double-beta (fourier_twoparam) kernels.

    Returns:
        nodes_list, elems_list, features_list, betas_list
        For single-beta: betas_list is list of floats
        For double-beta: betas_list is list of (betax, betay) tuples
    """
    nodes_list    = []
    elems_list    = []
    features_list = []
    betas_list    = []

    beta_min, beta_max = beta_range
    is_double_beta = (kernel_type == "fourier_twoparam")

    if kernel_type in ("panel_method"):
        for _ in tqdm(range(n_data), desc=f"Generating two-curve data ({'2D beta' if is_double_beta else 'beta'})"):
            
            beta = float(np.random.uniform(beta_min, beta_max))
            Mach = np.sqrt(1-beta**2)

            # ---------- generate two curves ----------
            nodes1 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                        freq_scale=freq_scale,
                                        deform=deform, deform_configs=deform_configs)
            nodes1[:, 0] = (nodes1[:, 0] + 2.5) * 0.49 - 2.5
            nodes1 = nodes1 * compress_coeff
            vertices1 = np.vstack([nodes1,nodes1[0]]) # vertices1: (n_panels1+1,2) 

            nodes2 = random_polar_curve(N, k=k_curve, r0_scale=r0_scale,
                                        freq_scale=freq_scale,
                                        deform=deform, deform_configs=deform_configs)
            nodes2[:, 0] = (nodes2[:, 0] - 2.5) * 0.49 + 2.5
            nodes2 = nodes2 * compress_coeff
            vertices2 = np.vstack([nodes2,nodes2[0]]) # vertices2: (n_panels2+1,2)

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

            # ---------- choose geometry class ----------
            panel_geo1 = PanelGeometryBeta(vertices1, False, beta)
            panel_geo2 = PanelGeometryBeta(vertices2, False, beta)

            out_normals1   = np.column_stack((panel_geo1.panel_sines, -panel_geo1.panel_cosines))
            out_normals2   = np.column_stack((panel_geo2.panel_sines, -panel_geo2.panel_cosines))
            out_normals = np.vstack([out_normals1,out_normals2])


            # ---------- compute g ----------
            _,_,_,Cp = solve_panel_method([panel_geo1,panel_geo2],0.0,1.0,Mach=Mach)
            g = np.concatenate([pre[:,np.newaxis] for pre in Cp], axis=0) # (n_panels1+n_panels2,1)

            features = np.concatenate([out_normals, g], axis=1)
            nodes_list.append(nodes)
            elems_list.append(elems)
            features_list.append(features)
            betas_list.append(beta)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

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
    ax1 = plt.subplot(1, 2, 1)
    draw_edges(ax1)
    normals = features[:, 0:2]
    ax1.quiver(nodes[:, 0], nodes[:, 1],
               normals[:, 0], normals[:, 1],
               color='red', scale=20, width=0.005, alpha=0.7)
    if isinstance(beta, tuple):
        ax1.set_title(f'Curve + Outward Normals  (βx={beta[0]:.3f}, βy={beta[1]:.3f})')
    else:
        ax1.set_title(f'Curve + Outward Normals  (beta={beta:.3f})')
    ax1.set_aspect('equal')

    # --- output g
    ax2 = plt.subplot(1, 2, 2)
    sc = ax2.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 2],
                     cmap='viridis', s=40)
    plt.colorbar(sc, ax=ax2, label='g(x)')
    draw_edges(ax2, color='red', alpha=0.7)
    ax2.set_title('Output g(x) = K_beta * f')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if figurename:
        plt.savefig(figurename)
    else:
        plt.show()


def save_pcno_curve_beta_dataset(save_path, nodes_list, elems_list, features_list, betas_list):
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(
        nodes_list,
        elems_list,
        features_list,
        mesh_type="cell_centered",
        adjacent_type="nodes",
    )

    node_measures, node_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=False
    )
    node_equal_measures, node_equal_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=True
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(
        save_path,
        nnodes=nnodes,
        node_mask=node_mask,
        nodes=nodes,
        node_measures_raw=node_measures_raw,
        node_measures=node_measures,
        node_weights=node_weights,
        node_equal_measures=node_equal_measures,
        node_equal_weights=node_equal_weights,
        features=features,
        directed_edges=directed_edges,
        edge_gradient_weights=edge_gradient_weights,
        betas=np.asarray(betas_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate variable-geometry, beta-parameterized kernel datasets for curve_beta."
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n_data", type=int, default=1)
    parser.add_argument("--N", type=int, default=1000, help="Panels per curve")
    parser.add_argument("--beta_low", type=float, default=0.5)
    parser.add_argument("--beta_high", type=float, default=1.0)
    parser.add_argument("--r0_scale", type=float, default=1.0)
    parser.add_argument("--freq_scale", type=float, default=1.0)
    parser.add_argument("--k_curve", type=int, default=5)
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="panel_method",
        choices=["panel_method"],
    )
    parser.add_argument("--two_curves", action="store_true")
    parser.add_argument("--compress_coeff", type=float, default=0.9)
    parser.add_argument("--deform", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--deform_M", type=int, default=200)
    parser.add_argument("--deform_sigma", type=float, default=1.0)
    parser.add_argument("--deform_epsilon", type=float, default=0.1)
    parser.add_argument("--bbox", type=float, nargs=4, default=[-2.5, 2.5, -2.5, 2.5])
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--figure_path", type=str, default="panel_beta.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    beta_range = (args.beta_low, args.beta_high)
    if args.beta_low <= 0 or args.beta_high <= 0 or args.beta_low > args.beta_high:
        raise ValueError("Require 0 < beta_low <= beta_high.")

    deform = args.deform.lower() == "true"
    deform_configs = [args.deform_M, args.deform_sigma, args.deform_epsilon, args.bbox]

    if not args.two_curves:
        nodes_list, elems_list, features_list, betas_list = generate_curves_data_panel_method_subsonic(
            args.n_data,
            args.N,
            beta_range=beta_range,
            r0_scale=args.r0_scale,
            freq_scale=args.freq_scale,
            k_curve=args.k_curve,
            kernel_type=args.kernel_type,
            deform=deform,
            deform_configs=deform_configs,
        )
    else:
        nodes_list, elems_list, features_list, betas_list = generate_curves_data_panel_method_subsonic_two(
            args.n_data,
            args.N,
            beta_range=beta_range,
            r0_scale=args.r0_scale,
            freq_scale=args.freq_scale,
            k_curve=args.k_curve,
            kernel_type=args.kernel_type,
            deform=deform,
            deform_configs=deform_configs,
            compress_coeff=args.compress_coeff,
        )

    betas_array = np.asarray(betas_list, dtype=np.float32)
    print("nodes_list shape   :", np.array(nodes_list).shape)
    print("elems_list shape   :", np.array(elems_list).shape)
    print("features_list shape:", np.array(features_list).shape)
    print(
        "betas shape        :",
        betas_array.shape,
        f"  min={betas_array.min():.4f}  max={betas_array.max():.4f}",
    )

    if args.visualize:
        features_at_nodes_list = (np.array(features_list) + np.roll(np.array(features_list), 1, axis=1)) / 2
        visualize_curve_beta(
            nodes_list[0],
            features_at_nodes_list[0],
            elems_list[0],
            beta=betas_list[0],
            figurename=args.figure_path,
        )
        print(f"Saved preview figure to: {args.figure_path}")

    tag = "two_curves" if args.two_curves else "single"
    default_name = (
        f"../../data/curve_beta/mpcno_curve_data"
        f"_{args.r0_scale}_{args.freq_scale}_{args.k_curve}_beta{beta_range}"
        f"_{args.kernel_type}_beta_random"
        f"_panel_{tag}.npz"
    )
    save_path = args.save_path if args.save_path else default_name
    save_pcno_curve_beta_dataset(
        save_path=save_path,
        nodes_list=nodes_list,
        elems_list=elems_list,
        features_list=features_list,
        betas_list=betas_list,
    )
    print(f"Saved dataset to: {save_path}")
