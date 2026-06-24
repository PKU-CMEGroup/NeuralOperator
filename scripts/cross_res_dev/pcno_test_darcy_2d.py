import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_unnormalized_node_measures, compute_node_weights
from pcno.pcno import compute_Fourier_modes, compute_Fourier_bases, PCNO
from pcno.interpolation import interp


torch.set_printoptions(precision=8, sci_mode=True)
torch.manual_seed(42)
np.random.seed(42)


def plot_outputs(items, save_path, title="PCNO outputs comparison"):
    """
    Plot multiple mesh scalar fields in one row with a shared colorbar.

    Each item is a dict with keys:
        nodes, elems, values, title
    """
    prepared_items = []
    all_values = []

    for item in items:
        nodes = np.asarray(item["nodes"])
        elems = np.asarray(item["elems"])
        values = np.asarray(item["values"]).reshape(-1)

        if elems.shape[1] == 4:
            triangles = elems[:, 1:4]
        elif elems.shape[1] == 3:
            triangles = elems
        else:
            raise ValueError(f"Expected elems shape [n_elems, 3] or [n_elems, 4], got {elems.shape}.")

        if values.shape[0] != nodes.shape[0]:
            raise ValueError(f"values must have length {nodes.shape[0]}, got {values.shape[0]}.")

        triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles.astype(np.int64))
        prepared_items.append((triangulation, values, item["title"]))
        all_values.append(values)

    # Use the first item, usually the coarse output, to define the shared colorbar range.
    # This makes all cross-resolution outputs comparable against the coarse baseline scale.
    reference_values = all_values[0]
    value_min = float(np.min(reference_values))
    value_max = float(np.max(reference_values))
    value_pad = 0.05 * (value_max - value_min)
    value_min -= value_pad
    value_max += value_pad
    if np.isclose(value_min, value_max):
        value_min -= 1e-6
        value_max += 1e-6

    levels = np.linspace(value_min, value_max, 33)

    fig, axes = plt.subplots(1, len(prepared_items), figsize=(5 * len(prepared_items), 5), constrained_layout=True)
    if len(prepared_items) == 1:
        axes = [axes]

    contour = None
    for ax, (triangulation, values, subplot_title) in zip(axes, prepared_items):
        contour = ax.tricontourf(
            triangulation,
            values,
            levels=levels,
            vmin=value_min,
            vmax=value_max,
        )
        ax.triplot(triangulation, linewidth=0.25, alpha=0.35)
        ax.set_title(subplot_title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(contour, ax=axes, shrink=0.85, label="value")
    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_outputs_separate_colorbars(items, save_path, title="PCNO outputs comparison"):
    """
    Plot multiple mesh scalar fields in one row, each with its own colorbar.

    This version is useful for seeing the spatial structure of each output even
    when different methods have noticeably different value ranges.
    """
    prepared_items = []

    for item in items:
        nodes = np.asarray(item["nodes"])
        elems = np.asarray(item["elems"])
        values = np.asarray(item["values"]).reshape(-1)

        if elems.shape[1] == 4:
            triangles = elems[:, 1:4]
        elif elems.shape[1] == 3:
            triangles = elems
        else:
            raise ValueError(f"Expected elems shape [n_elems, 3] or [n_elems, 4], got {elems.shape}.")

        if values.shape[0] != nodes.shape[0]:
            raise ValueError(f"values must have length {nodes.shape[0]}, got {values.shape[0]}.")

        triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles.astype(np.int64))
        prepared_items.append((triangulation, values, item["title"]))

    fig, axes = plt.subplots(1, len(prepared_items), figsize=(5 * len(prepared_items), 5), constrained_layout=True)
    if len(prepared_items) == 1:
        axes = [axes]

    for ax, (triangulation, values, subplot_title) in zip(axes, prepared_items):
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if np.isclose(value_min, value_max):
            value_min -= 1e-6
            value_max += 1e-6
        levels = np.linspace(value_min, value_max, 33)

        contour = ax.tricontourf(
            triangulation,
            values,
            levels=levels,
            vmin=value_min,
            vmax=value_max,
        )
        ax.triplot(triangulation, linewidth=0.25, alpha=0.35)
        ax.set_title(f"{subplot_title}\n[{value_min:.3e}, {value_max:.3e}]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(contour, ax=ax, shrink=0.85, label="value")

    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def build_cross_knn_edges(nodes_in, nodes_out, k=3, eps=1e-8):
    batch_size, n_out, ndim = nodes_out.shape
    n_in = nodes_in.shape[1]
    k = min(k, n_in)

    dists = torch.cdist(nodes_out, nodes_in)  # [B, n_out, n_in]
    knn_idx = torch.topk(dists, k=k, dim=-1, largest=False).indices  # [B, n_out, k]

    target = torch.arange(n_out, device=nodes_out.device).view(1, n_out, 1)
    target = target.expand(batch_size, n_out, k)

    directed_edges_cross = torch.stack(
        [target.reshape(batch_size, -1), knn_idx.reshape(batch_size, -1)],
        dim=-1,
    ).long()

    batch_index = torch.arange(batch_size, device=nodes_in.device).view(batch_size, 1, 1)
    source_nodes = nodes_in[batch_index, knn_idx]  # [B, n_out, k, ndim]
    target_nodes = nodes_out.unsqueeze(2)          # [B, n_out, 1, ndim]
    dx = source_nodes - target_nodes               # [B, n_out, k, ndim]

    # Local least-squares pseudo-inverse weights:
    # coeff = DX @ inv(DX^T DX + eps I), shape [B, n_out, k, ndim].
    dx_t = dx.transpose(-2, -1)                    # [B, n_out, ndim, k]
    gram = dx_t @ dx                               # [B, n_out, ndim, ndim]
    eye = torch.eye(ndim, dtype=nodes_in.dtype, device=nodes_in.device)
    gram = gram + eps * eye.view(1, 1, ndim, ndim)

    coeff_t = torch.linalg.solve(gram, dx_t)       # [B, n_out, ndim, k]
    coeff = coeff_t.transpose(-2, -1)              # [B, n_out, k, ndim]

    edge_gradient_weights_cross = coeff.reshape(batch_size, -1, ndim)

    return directed_edges_cross, edge_gradient_weights_cross


def add_elem_dim(elems_np):
    elems_np = np.asarray(elems_np, dtype=np.int64)

    if elems_np.ndim != 2:
        raise ValueError(f"elems_np must be a 2D array, got shape {elems_np.shape}.")

    if elems_np.shape[1] == 3:
        return np.concatenate(
            (
                np.full((elems_np.shape[0], 1), 2, dtype=np.int64),
                elems_np,
            ),
            axis=1,
        )

    if elems_np.shape[1] == 4:
        return elems_np


def make_aux_from_mesh(nodes_np, elems_np, Ls, device):
    nodes_list = [nodes_np]
    elems_list = [add_elem_dim(elems_np)]
    features_list = [np.ones((nodes_np.shape[0], 1), dtype=np.float32)]

    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(
        nodes_list,
        elems_list,
        features_list,
        mesh_type="vertex_centered",
        adjacent_type="element",
    )

    measure_dims = np.array([2], dtype=np.int64)

    # node_measures, node_weights = compute_unnormalized_node_measures(
    #     nnodes,
    #     node_measures_raw,
    #     measure_dims=measure_dims,
    #     Ls=Ls,
    # )
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)

    node_measures_raw = torch.from_numpy(node_measures_raw).to(device)
    node_measures = torch.from_numpy(node_measures.astype(np.float32)).to(device)
    node_weights = torch.from_numpy(node_weights.astype(np.float32)).to(device)

    indices = torch.isfinite(node_measures_raw)
    node_rhos = node_weights.clone()
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    node_mask = torch.from_numpy(node_mask).to(device)
    nodes = torch.from_numpy(nodes.astype(np.float32)).to(device)
    node_weights = node_weights.to(device)
    node_rhos = node_rhos.to(device)
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64)).to(device)
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32)).to(device)

    aux = (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)
    return aux, node_rhos


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    data = np.load("./data/darcy/median_data.npz")

    nodes_coarse = np.load("./data/darcy/nodes_coarse.npy").astype(np.float32)
    elems_coarse = np.load("./data/darcy/elems_coarse.npy").astype(np.int64)
    features_coarse = np.load("./data/darcy/features_coarse.npy").astype(np.float32)
    nodes_fine = np.load("./data/darcy/nodes_fine.npy").astype(np.float32)
    elems_fine = np.load("./data/darcy/elems_fine.npy").astype(np.int64)

    print("Loaded meshes:")
    print(f"  coarse nodes: {nodes_coarse.shape}, coarse elems: {elems_coarse.shape}")
    print(f"  fine nodes  : {nodes_fine.shape}, fine elems  : {elems_fine.shape}")
    print(f"  coarse elems with dim marker: {add_elem_dim(elems_coarse).shape}")
    print(f"  fine elems with dim marker  : {add_elem_dim(elems_fine).shape}")

    Lx, Ly = 1.0, 1.0
    Ls = [Lx, Ly]
    aux_coarse, node_rhos_coarse = make_aux_from_mesh(nodes_coarse, elems_coarse, Ls, device)
    aux_fine, node_rhos_fine = make_aux_from_mesh(nodes_fine, elems_fine, Ls, device)
    
    # aux_coarse = (None, nodes_coarse)

    node_mask_c, nodes_c, node_weights_c, directed_edges_c, edge_gradient_weights_c = aux_coarse
    node_mask_f, nodes_f, node_weights_f, directed_edges_f, edge_gradient_weights_f = aux_fine

    directed_edges_cross, edge_gradient_weights_cross = build_cross_knn_edges(
        nodes_in=nodes_c,
        nodes_out=nodes_f,
        k=4,
    )

    idx_corr = torch.from_numpy(np.load("./data/darcy/idx_corr.npy").astype(np.int64)).to(device)
    if idx_corr.ndim == 1:
        idx_corr = idx_corr.unsqueeze(0)

    # scalar_feature = torch.ones(nodes_c.shape[0], nodes_c.shape[1], 1, dtype=nodes_c.dtype, device=device)
    x_cross = torch.from_numpy(features_coarse[None, :, :3]).to(device)

    ndim = 2
    k_max = 16
    modes = compute_Fourier_modes(ndim, [k_max, k_max], [Lx, Ly])
    modes = torch.tensor(modes, dtype=torch.float32, device=device)

    model = PCNO(ndim, modes, nmeasures=1,
                 layers=[128, 128, 128, 128, 128],
                 fc_dim=32,
                 inv_L_scale_hyper=[False, 0.5, 2.0],
                 in_dim=3,
                 out_dim=1,
                 act='gelu').to(device)

    model.eval()

    aux_cross = (
        node_mask_f,
        nodes_f,
        directed_edges_cross,
        edge_gradient_weights_cross,
        idx_corr,
    )

    interp_methods = ["spectral", "spectral_wls", "taylor"]

    with torch.no_grad():
        y_coarse = model(x_cross, aux_coarse)

    print("\nCoarse forward result:")
    print(f"  y_coarse shape: {tuple(y_coarse.shape)}")
    print(f"  y_coarse min/max: {float(y_coarse.min()):.6e}, {float(y_coarse.max()):.6e}")

    y_coarse_plot = y_coarse[0, :, 0].detach().cpu().numpy()

    cross_outputs = {}
    comparison_items = [
        {
            "nodes": nodes_coarse,
            "elems": elems_coarse,
            "values": features_coarse[:, -1],
            "title": "truth",
        }
    ]
    comparison_items.append(
        {
            "nodes": nodes_coarse,
            "elems": elems_coarse,
            "values": y_coarse_plot,
            "title": "coarse",
        }
    )

    post_interp = True
    for interp_method in interp_methods:
        with torch.no_grad():
            y_cross = model(
                x_cross,
                aux_coarse,
                aux_cross=aux_cross,
                interp_type=interp_method,
                post_interp=True,
            )

        cross_outputs[interp_method] = y_cross

        print(f"\nCross forward result ({interp_method}):")
        print(f"  y_cross shape: {tuple(y_cross.shape)}")
        print(f"  y_cross min/max: {float(y_cross.min()):.6e}, {float(y_cross.max()):.6e}")
        print(f"  y_cross mean/std: {float(y_cross.mean()):.6e}, {float(y_cross.std()):.6e}")

        y_plot = y_cross[0, :, 0].detach().cpu().numpy()
        comparison_items.append(
            {
                "nodes": nodes_fine,
                "elems": elems_fine,
                "values": y_plot,
                "title": interp_method,
            }
        )

    plot_outputs(
        items=comparison_items,
        save_path="./figs/x_shared_cb.png",
        title="PCNO coarse and cross outputs, shared colorbar",
    )

    plot_outputs_separate_colorbars(
        items=comparison_items,
        save_path="./figs/x_separate_cb.png",
        title="PCNO coarse and cross outputs, separate colorbars",
    )

    print("\nPairwise cross-output differences:")
    for i, method_i in enumerate(interp_methods):
        for method_j in interp_methods[i + 1:]:
            diff = cross_outputs[method_i] - cross_outputs[method_j]
            rel_l2 = torch.linalg.norm(diff) / torch.clamp(torch.linalg.norm(cross_outputs[method_i]), min=1e-12)
            max_err = torch.max(torch.abs(diff))
            print(f"  {method_i} vs {method_j}:")
            print(f"    relative L2: {float(rel_l2):.6e}")
            print(f"    max abs diff: {float(max_err):.6e}")

    print("\nidx_corr:")
    print(f"  shape: {tuple(idx_corr.shape)}")
    print(f"  exact corresponding nodes: {int((idx_corr >= 0).sum().item())}")

    print("\nDone.")
