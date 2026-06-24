import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_unnormalized_node_measures
from pcno.pcno import compute_Fourier_modes, compute_Fourier_bases, PCNO
from pcno.interpolation import interp


torch.set_printoptions(precision=8, sci_mode=True)
torch.manual_seed(42)
np.random.seed(42)


def plot_output_on_mesh(nodes, elems, values, save_path, title="PCNO cross output"):
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)
    values = np.asarray(values).reshape(-1)

    if elems.shape[1] == 4:
        triangles = elems[:, 1:4]
    elif elems.shape[1] == 3:
        triangles = elems
    else:
        raise ValueError(f"Expected elems shape [n_elems, 3] or [n_elems, 4], got {elems.shape}.")

    if values.shape[0] != nodes.shape[0]:
        raise ValueError(f"values must have length {nodes.shape[0]}, got {values.shape[0]}.")

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles.astype(np.int64))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    contour = ax.tricontourf(triangulation, values, levels=32)
    ax.triplot(triangulation, linewidth=0.25, alpha=0.35)
    # ax.scatter(nodes[:, 0], nodes[:, 1], s=3, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(contour, ax=ax, shrink=0.85)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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


def plot_reconstruction_comparison(nodes, elems, x_true, x_rec, save_path, title):
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)
    x_true = np.asarray(x_true).reshape(-1)
    x_rec = np.asarray(x_rec).reshape(-1)
    x_err = x_true - x_rec

    if elems.shape[1] == 4:
        triangles = elems[:, 1:4]
    elif elems.shape[1] == 3:
        triangles = elems
    else:
        raise ValueError(f"Expected elems shape [n_elems, 3] or [n_elems, 4], got {elems.shape}.")

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles.astype(np.int64))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Use the same colorbar range for original and reconstruction so their
    # magnitudes can be compared directly. Use a separate symmetric range for error.
    value_min = min(float(np.min(x_true)), float(np.min(x_rec)))
    value_max = max(float(np.max(x_true)), float(np.max(x_rec)))
    if np.isclose(value_min, value_max):
        value_min -= 1e-6
        value_max += 1e-6

    err_abs_max = float(np.max(np.abs(x_err)))
    err_abs_max = max(err_abs_max, 1e-12)

    value_levels = np.linspace(value_min, value_max, 33)
    err_levels = np.linspace(-err_abs_max, err_abs_max, 33)

    contour_true = axes[0].tricontourf(triangulation, x_true, levels=value_levels)
    axes[0].triplot(triangulation, linewidth=0.2, alpha=0.35)
    axes[0].set_title("Original")

    contour_rec = axes[1].tricontourf(triangulation, x_rec, levels=value_levels)
    axes[1].triplot(triangulation, linewidth=0.2, alpha=0.35)
    axes[1].set_title("Reconstruction")

    contour_err = axes[2].tricontourf(
        triangulation,
        x_err,
        levels=err_levels,
        cmap="coolwarm",
        vmin=-err_abs_max,
        vmax=err_abs_max,
    )
    axes[2].triplot(triangulation, linewidth=0.2, alpha=0.35)
    axes[2].set_title("Error")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(contour_true, ax=axes[:2], shrink=0.85, label="value")
    fig.colorbar(contour_err, ax=axes[2], shrink=0.85, label="error")

    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def test_spectral_interp_2d(nodes, elems, node_weights, modes, interp_method="spectral_wls"):
    """
    Fully runnable spectral_interp test independent of the PCNO model.

    It constructs two scalar fields on the given point-cloud mesh:
        1. a smooth Gaussian field;
        2. a discontinuous binary field generated by torch.where.

    Then it reconstructs each field by PCNO Fourier projection/reconstruction:
        coarse nodes -> Fourier coefficients -> same nodes.

    This tests spectral_interp itself, not PCNO.forward.
    """
    print(f"\nTesting {interp_method} without PCNO model:")
    if interp_method not in {"spectral", "spectral_wls"}:
        raise ValueError(f"interp_method must be 'spectral' or 'spectral_wls', got {interp_method}.")

    bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, modes)
    wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
    wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
    wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

    in_wbases = (wbases_c, wbases_s, wbases_0)
    out_bases = (bases_c, bases_s, bases_0)

    print(f"  node_weights sum: {node_weights.sum(dim=1).detach().cpu().numpy()}")

    x_coord = nodes[..., 0]
    y_coord = nodes[..., 1]

    x_min, x_max = torch.min(x_coord), torch.max(x_coord)
    y_min, y_max = torch.min(y_coord), torch.max(y_coord)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    domain_scale = torch.maximum(x_max - x_min, y_max - y_min)
    sigma = 0.15 * domain_scale

    gaussian = torch.exp(-((x_coord - cx) ** 2 + (y_coord - cy) ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian.unsqueeze(1)  # [B, 1, n_nodes]

    x_2d = torch.sin(2.0 * np.pi * (4.0 * x_coord + 3.0 * y_coord))
    binary = torch.where(
        x_2d > 0.5,
        torch.tensor(1.0, dtype=nodes.dtype, device=nodes.device),
        torch.tensor(-1.0, dtype=nodes.dtype, device=nodes.device),
    )
    binary = binary.unsqueeze(1)  # [B, 1, n_nodes]

    constant = torch.ones(
        nodes.shape[0],
        1,
        nodes.shape[1],
        dtype=nodes.dtype,
        device=nodes.device,
    )

    test_cases = [
        ("constant", constant, f"./figs/{interp_method}_const_2d.png"),
        ("gaussian", gaussian, f"./figs/{interp_method}_gauss_2d.png"),
        ("binary", binary, f"./figs/{interp_method}_binary_2d.png"),
    ]

    for name, x_true, fig_path in test_cases:
        x_rec = interp(
            x=x_true,
            interp_type=interp_method,
            nodes_in=nodes,
            nodes_out=nodes,
            in_bases=(bases_c, bases_s, bases_0),
            in_wbases=in_wbases,
            out_bases=out_bases,
            n_out=nodes.shape[1],
            eps=1e-8,
        )

        rel_l2 = torch.linalg.norm(x_true - x_rec) / torch.linalg.norm(x_true)
        max_err = torch.max(torch.abs(x_true - x_rec))

        print(f"  {name} field:")
        print(f"    x_true shape: {tuple(x_true.shape)}")
        print(f"    x_rec shape : {tuple(x_rec.shape)}")
        print(f"    relative L2 : {float(rel_l2):.6e}")
        print(f"    max error   : {float(max_err):.6e}")
        print(f"    rec min/max/mean: {float(x_rec.min()):.6e}, {float(x_rec.max()):.6e}, {float(x_rec.mean()):.6e}")

        plot_reconstruction_comparison(
            nodes=nodes[0].detach().cpu().numpy(),
            elems=elems,
            x_true=x_true[0, 0].detach().cpu().numpy(),
            x_rec=x_rec[0, 0].detach().cpu().numpy(),
            save_path=fig_path,
            title=f"{interp_method} test: {name}",
        )
        print(f"    figure saved to: {fig_path}")


def compute_1d_voronoi_weights(points, domain_start=0.0, domain_end=1.0):
    n_points = points.shape[0]
    if n_points == 1:
        return torch.ones_like(points) * (domain_end - domain_start)

    midpoints = 0.5 * (points[1:] + points[:-1])
    left_bounds = torch.cat([
        torch.tensor([domain_start], dtype=points.dtype, device=points.device),
        midpoints,
    ])
    right_bounds = torch.cat([
        midpoints,
        torch.tensor([domain_end], dtype=points.dtype, device=points.device),
    ])
    weights = right_bounds - left_bounds
    return weights


def build_real_fourier_basis_1d(nodes, k_max, L=1.0):
    batch_size, n_points, ndims = nodes.shape
    if ndims != 1:
        raise ValueError(f"build_real_fourier_basis_1d expects ndims=1, got {ndims}.")

    dtype = nodes.dtype
    device = nodes.device
    x_coord = nodes[..., 0]
    freqs = torch.arange(1, k_max + 1, dtype=dtype, device=device)
    angles = 2.0 * np.pi * x_coord.unsqueeze(-1) * freqs.view(1, 1, -1) / L

    ones = torch.ones(batch_size, n_points, 1, dtype=dtype, device=device)
    cos_basis = np.sqrt(2.0) * torch.cos(angles)
    sin_basis = np.sqrt(2.0) * torch.sin(angles)
    phi = torch.cat([ones, cos_basis, sin_basis], dim=-1)
    return phi


def plot_1d_basis_gram_matrices(point_sets, n_points, k_max, device, save_path):
    gram_results = {}

    for point_type, points in point_sets:
        nodes = points.view(1, n_points, 1).to(device)
        if point_type == "uniform":
            node_weights = torch.ones(1, n_points, 1, dtype=torch.float32, device=device) / n_points
        else:
            voronoi_weights = compute_1d_voronoi_weights(points, domain_start=0.0, domain_end=1.0)
            node_weights = voronoi_weights.view(1, n_points, 1).to(device)

        phi = build_real_fourier_basis_1d(nodes, k_max=k_max, L=1.0)
        weights = node_weights[..., 0]
        gram = torch.einsum("bnm,bn,bnl->bml", phi, weights, phi)[0]
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        gram_results[point_type] = (
            gram.detach().cpu().numpy(),
            (gram - eye).detach().cpu().numpy(),
        )

        off_diag = gram - torch.diag(torch.diag(gram))
        print(f"  {point_type} basis Gram:")
        print(f"    shape: {tuple(gram.shape)}")
        print(f"    max |G - I|: {float(torch.max(torch.abs(gram - eye))):.6e}")
        print(f"    max off-diagonal |G_ij|: {float(torch.max(torch.abs(off_diag))):.6e}")
        print(f"    diag min/max: {float(torch.min(torch.diag(gram))):.6e}, {float(torch.max(torch.diag(gram))):.6e}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    matrix_specs = [
        (axes[0, 0], gram_results["random"][0], "random: Gram G"),
        (axes[0, 1], gram_results["uniform"][0], "uniform: Gram G"),
        (axes[1, 0], gram_results["random"][1], "random: G - I"),
        (axes[1, 1], gram_results["uniform"][1], "uniform: G - I"),
    ]

    for ax, mat, title in matrix_specs:
        vmax = np.max(np.abs(mat))
        vmax = max(vmax, 1e-12)
        im = ax.imshow(mat, origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("basis index j")
        ax.set_ylabel("basis index i")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"1D weighted Fourier basis Gram matrices, k_max={k_max}")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  1D basis Gram matrix figure saved to: {save_path}")


def test_spectral_interp_1d(n_points=512, k_max=None, seed=42, device=None, interp_method="spectral"):
    print(f"\nTesting {interp_method} in 1D without PCNO model:")
    if interp_method not in {"spectral", "spectral_wls"}:
        raise ValueError(f"interp_method must be 'spectral' or 'spectral_wls', got {interp_method}.")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if k_max is None:
        k_max = n_points // 2

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    random_points = torch.rand(n_points, dtype=torch.float32, generator=generator)
    random_points, _ = torch.sort(random_points)

    uniform_points = torch.linspace(
        0.0,
        1.0,
        n_points + 1,
        dtype=torch.float32,
    )[:-1]

    point_sets = [
        ("random", random_points),
        ("uniform", uniform_points),
    ]

    # For visualizing delta_ij behavior, avoid using more real basis functions than points.
    # The real basis size is 1 + 2 * k_max, so use at most k_max <= (n_points - 1) // 2.
    gram_k_max = min(k_max, max(1, (n_points - 1) // 2))
    plot_1d_basis_gram_matrices(
        point_sets=point_sets,
        n_points=n_points,
        k_max=gram_k_max,
        device=device,
        save_path="./figs/gram.png",
    )

    modes = compute_Fourier_modes(1, [k_max], [1.0])
    modes = torch.tensor(modes, dtype=torch.float32, device=device)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)

    for col, (point_type, points) in enumerate(point_sets):
        nodes = points.view(1, n_points, 1).to(device)  # [B, n_nodes, ndims]
        if point_type == "uniform":
            node_weights = torch.ones(
                1,
                n_points,
                1,
                dtype=torch.float32,
                device=device,
            ) / n_points
        else:
            voronoi_weights = compute_1d_voronoi_weights(points, domain_start=0.0, domain_end=1.0)
            node_weights = voronoi_weights.view(1, n_points, 1).to(device)

        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, modes)
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        in_wbases = (wbases_c, wbases_s, wbases_0)
        out_bases = (bases_c, bases_s, bases_0)

        x_coord = nodes[..., 0]

        constant = torch.ones(
            1,
            1,
            n_points,
            dtype=torch.float32,
            device=device,
        )

        gaussian = torch.exp(
            -((x_coord - 0.5) ** 2) / (2.0 * 0.08 ** 2)
        ).unsqueeze(1)

        x_wave = torch.sin(2.0 * np.pi * 4.0 * x_coord)
        binary = torch.where(
            x_wave > 0.5,
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(-1.0, dtype=torch.float32, device=device),
        ).unsqueeze(1)

        test_cases = [
            ("constant", constant),
            ("gaussian", gaussian),
            ("binary", binary),
        ]

        print(f"\n  Point type: {point_type}")
        print(f"    n_points: {n_points}")
        print(f"    k_max: {k_max}")
        print(f"    node_weights sum: {node_weights.sum(dim=1).detach().cpu().numpy()}")

        x_plot = nodes[0, :, 0].detach().cpu().numpy()

        for row, (name, x_true) in enumerate(test_cases):
            x_rec = interp(
                x=x_true,
                interp_type=interp_method,
                nodes_in=nodes,
                nodes_out=nodes,
                in_bases=(bases_c, bases_s, bases_0),
                in_wbases=in_wbases,
                out_bases=out_bases,
                n_out=nodes.shape[1],
                eps=1e-8,
            )

            rel_l2 = torch.linalg.norm(x_true - x_rec) / torch.linalg.norm(x_true)
            max_err = torch.max(torch.abs(x_true - x_rec))

            print(f"    1D {name} field:")
            print(f"      x_true shape: {tuple(x_true.shape)}")
            print(f"      x_rec shape : {tuple(x_rec.shape)}")
            print(f"      relative L2 : {float(rel_l2):.6e}")
            print(f"      max error   : {float(max_err):.6e}")
            print(
                f"      rec min/max/mean: "
                f"{float(x_rec.min()):.6e}, "
                f"{float(x_rec.max()):.6e}, "
                f"{float(x_rec.mean()):.6e}"
            )

            ax = axes[row, col]
            ax.plot(
                x_plot,
                x_true[0, 0].detach().cpu().numpy(),
                label="original",
                linewidth=1.5,
            )
            ax.plot(
                x_plot,
                x_rec[0, 0].detach().cpu().numpy(),
                "--",
                label="reconstruction",
                linewidth=1.2,
            )
            ax.set_title(f"{point_type} points: {interp_method}, {name}")
            ax.set_xlabel("x")
            ax.set_ylabel("value")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.savefig(f"./figs/{interp_method}_1d.png", dpi=300)
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



def make_aux_from_mesh(nodes_np, elems_np, device):
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

    xy_min = np.min(nodes_np[:, :2], axis=0)
    xy_max = np.max(nodes_np[:, :2], axis=0)
    Ls = xy_max - xy_min + 0.1
    measure_dims = np.array([2], dtype=np.int64)

    node_measures, node_weights = compute_unnormalized_node_measures(
        nnodes,
        node_measures_raw,
        measure_dims=measure_dims,
        Ls=Ls,
    )

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
    return aux, node_rhos, Ls


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    nodes_coarse = np.load("./data/nodes_coarse.npy").astype(np.float32)
    elems_coarse = np.load("./data/elems_coarse.npy").astype(np.int64)
    nodes_fine = np.load("./data/nodes_fine.npy").astype(np.float32)
    elems_fine = np.load("./data/elems_fine.npy").astype(np.int64)

    print("Loaded meshes:")
    print(f"  coarse nodes: {nodes_coarse.shape}, coarse elems: {elems_coarse.shape}")
    print(f"  fine nodes  : {nodes_fine.shape}, fine elems  : {elems_fine.shape}")
    print(f"  coarse elems with dim marker: {add_elem_dim(elems_coarse).shape}")
    print(f"  fine elems with dim marker  : {add_elem_dim(elems_fine).shape}")

    aux_coarse, node_rhos_coarse, Ls = make_aux_from_mesh(nodes_coarse, elems_coarse, device)
    aux_fine, node_rhos_fine, _ = make_aux_from_mesh(nodes_fine, elems_fine, device)

    node_mask_c, nodes_c, node_weights_c, directed_edges_c, edge_gradient_weights_c = aux_coarse
    node_mask_f, nodes_f, node_weights_f, directed_edges_f, edge_gradient_weights_f = aux_fine

    directed_edges_cross, edge_gradient_weights_cross = build_cross_knn_edges(
        nodes_in=nodes_c,
        nodes_out=nodes_f,
        k=4,
    )

    idx_corr = torch.from_numpy(np.load("./data/idx_corr.npy").astype(np.int64)).to(device)
    if idx_corr.ndim == 1:
        idx_corr = idx_corr.unsqueeze(0)

    scalar_feature = torch.ones(nodes_c.shape[0], nodes_c.shape[1], 1, dtype=nodes_c.dtype, device=device)
    x_cross = torch.cat((scalar_feature, nodes_c, node_rhos_coarse), dim=-1)

    ndim = 2
    k_max = 16
    Lx, Ly = Ls  # 2.0, 2.0
    modes = compute_Fourier_modes(ndim, [k_max, k_max], [Lx, Ly])
    modes = torch.tensor(modes, dtype=torch.float32, device=device)

    # test_spectral_interp_1d(device=device, interp_method="spectral")
    # test_spectral_interp_2d(nodes_c, elems_coarse, node_weights_c, modes, interp_method="spectral")
    # test_spectral_interp_1d(device=device, interp_method="spectral_wls")
    # test_spectral_interp_2d(nodes_c, elems_coarse, node_weights_c, modes, interp_method="spectral_wls")

    model = PCNO(
        ndim,
        modes,
        nmeasures=1,
        layers=[32, 32, 32],
        fc_dim=32,
        inv_L_scale_hyper=[False, 0.5, 2.0],
        in_dim=4,
        out_dim=1,
        act='gelu',
    ).to(device)

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
            "values": y_coarse_plot,
            "title": "coarse",
        }
    ]

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
