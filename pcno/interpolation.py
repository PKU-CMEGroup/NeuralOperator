
import torch


def apply_idx_corresponding(x_out, x_in, idx_corresponding):
    """
    Overwrite output-node values by exact corresponding input-node values.

    Parameters:
        x_out : Tensor [batch_size, channels, n_out]
            Interpolated/resampled feature on output nodes.
        x_in : Tensor [batch_size, channels, n_in]
            Original feature on input nodes.
        idx_corresponding : Tensor or tuple/list
            If Tensor with shape [batch_size, n_out], then each entry stores the
            corresponding input index for that output node, and negative values mean
            no exact correspondence.

            If tuple/list (idx_out, idx_in), then idx_out and idx_in can have shape
            [m] or [batch_size, m], meaning x_out[..., idx_out] should be replaced by
            x_in[..., idx_in].

    Returns:
        x_out : Tensor [batch_size, channels, n_out]
            Output feature with exact corresponding nodes overwritten.
    """
    if idx_corresponding is None:
        return x_out

    batch_size, channels, n_out = x_out.shape
    _, _, n_in = x_in.shape
    x_out = x_out.clone()

    if isinstance(idx_corresponding, (tuple, list)):
        if len(idx_corresponding) != 2:
            raise ValueError("idx_corresponding as tuple/list must be (idx_out, idx_in).")
        idx_out, idx_in = idx_corresponding
        idx_out = idx_out.to(device=x_out.device, dtype=torch.long)
        idx_in = idx_in.to(device=x_out.device, dtype=torch.long)

        if idx_out.ndim == 1:
            valid = (idx_out >= 0) & (idx_out < n_out) & (idx_in >= 0) & (idx_in < n_in)
            if valid.any():
                x_out[:, :, idx_out[valid]] = x_in[:, :, idx_in[valid]]
            return x_out

        if idx_out.ndim == 2:
            batch_index = torch.arange(batch_size, device=x_out.device).unsqueeze(1)
            valid = (idx_out >= 0) & (idx_out < n_out) & (idx_in >= 0) & (idx_in < n_in)
            safe_out = torch.where(valid, idx_out, torch.zeros_like(idx_out))
            safe_in = torch.where(valid, idx_in, torch.zeros_like(idx_in))
            values = x_in.permute(0, 2, 1)[batch_index, safe_in]
            values = values * valid.unsqueeze(-1).to(values.dtype)
            x_out_perm = x_out.permute(0, 2, 1)
            x_out_perm[batch_index, safe_out] = torch.where(
                valid.unsqueeze(-1),
                values,
                x_out_perm[batch_index, safe_out],
            )
            return x_out_perm.permute(0, 2, 1)

        raise ValueError("idx_out and idx_in must have shape [m] or [batch_size, m].")

    idx_corresponding = idx_corresponding.to(device=x_out.device, dtype=torch.long)
    if idx_corresponding.ndim != 2:
        raise ValueError("idx_corresponding tensor must have shape [batch_size, n_out].")
    if idx_corresponding.shape[0] != batch_size or idx_corresponding.shape[1] != n_out:
        raise ValueError("idx_corresponding tensor must have shape [batch_size, n_out].")

    valid = (idx_corresponding >= 0) & (idx_corresponding < n_in)
    safe_idx = torch.where(valid, idx_corresponding, torch.zeros_like(idx_corresponding))
    batch_index = torch.arange(batch_size, device=x_out.device).unsqueeze(1)
    exact_values = x_in.permute(0, 2, 1)[batch_index, safe_idx]

    x_out_perm = x_out.permute(0, 2, 1)
    x_out_perm = torch.where(valid.unsqueeze(-1), exact_values, x_out_perm)
    return x_out_perm.permute(0, 2, 1)


def interp(
    x,
    interp_type,
    nodes_in,
    nodes_out,
    in_wbases=None,
    out_bases=None,
    in_bases=None,
    idx_corr=None,
    directed_edges=None,
    edge_weights=None,
    n_out=None,
    eps=1e-8,
):
    if interp_type == "taylor":
        x_out = taylor_interp(x=x, nodes_in=nodes_in, nodes_out=nodes_out, directed_edges=directed_edges, n_out=n_out, eps=eps)
    elif interp_type == "spectral":
        x_out = spectral_interp(x=x, in_wbases=in_wbases, out_bases=out_bases)
    elif interp_type == "spectral_wls":
        x_out = spectral_wls_interp(x=x, in_bases=in_bases, in_wbases=in_wbases, out_bases=out_bases, eps=1e-12)
    elif interp_type == "given_weight":
        x_out = given_weight_interp(x=x, directed_edges_cross=directed_edges, edge_weights=edge_weights, n_out=n_out, eps=eps)
    else:
        raise ValueError(f"Unknown resample_type: {interp_type}. Use 'taylor', 'spectral', 'spectral_wls' or 'given_weight'.")

    if idx_corr is not None:
        return apply_idx_corresponding(x_out, x, idx_corr)
    else:
        return x_out


def spectral_interp(x, in_wbases, out_bases):
    wbases_c, wbases_s, wbases_0 = in_wbases
    bases_c, bases_s, bases_0 = out_bases

    x_c_hat = torch.einsum("bix,bxkw->bikw", x, wbases_c)
    x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
    x_0_hat = torch.einsum("bix,bxkw->bikw", x, wbases_0)

    x_out = torch.einsum("bikw,bxkw->bix", x_0_hat, bases_0) + 2 * torch.einsum("bikw,bxkw->bix", x_c_hat, bases_c) - 2 * torch.einsum("bikw,bxkw->bix", x_s_hat, bases_s)

    return x_out


def _flatten_real_fourier_bases(bases):
    bases_c, bases_s, bases_0 = bases
    sqrt2 = 2.0 ** 0.5

    phi_0 = bases_0.reshape(bases_0.shape[0], bases_0.shape[1], -1)
    phi_c = (sqrt2 * bases_c).reshape(bases_c.shape[0], bases_c.shape[1], -1)
    phi_s = (sqrt2 * bases_s).reshape(bases_s.shape[0], bases_s.shape[1], -1)

    return torch.cat([phi_0, phi_c, phi_s], dim=-1)


def _flatten_weighted_real_fourier_bases(wbases):
    wbases_c, wbases_s, wbases_0 = wbases
    sqrt2 = 2.0 ** 0.5

    wphi_0 = wbases_0.reshape(wbases_0.shape[0], wbases_0.shape[1], -1)
    wphi_c = (sqrt2 * wbases_c).reshape(wbases_c.shape[0], wbases_c.shape[1], -1)
    wphi_s = (sqrt2 * wbases_s).reshape(wbases_s.shape[0], wbases_s.shape[1], -1)

    return torch.cat([wphi_0, wphi_c, wphi_s], dim=-1)


def spectral_wls_interp(x, in_bases, in_wbases, out_bases, eps=1e-12):
    """
    Weighted least-squares Fourier interpolation/reconstruction.

    This is a WLS-corrected version of spectral_interp. It does not assume that
    Fourier bases are discretely orthogonal on the input point cloud. Instead it
    solves
        coef = (Phi_in^T W Phi_in + eps I)^(-1) Phi_in^T W x,
    then evaluates
        x_out = Phi_out coef.

    Parameters:
        x : Tensor [batch_size, channels, n_in]
            Input feature on input nodes.
        in_bases : tuple or None
            (bases_c, bases_s, bases_0) on input nodes. If None, out_bases is
            used, which is correct only for same-grid reconstruction.
        in_wbases : tuple
            (wbases_c, wbases_s, wbases_0) on input nodes.
        out_bases : tuple
            (bases_c, bases_s, bases_0) on output nodes.
        eps : float
            Tikhonov regularization for the Gram matrix.

    Returns:
        x_out : Tensor [batch_size, channels, n_out]
            WLS Fourier reconstruction/interpolation on output nodes.
    """
    if in_bases is None:
        in_bases = out_bases

    phi_in = _flatten_real_fourier_bases(in_bases)
    wphi_in = _flatten_weighted_real_fourier_bases(in_wbases)
    phi_out = _flatten_real_fourier_bases(out_bases)

    batch_size, n_in, n_basis = phi_in.shape

    gram = torch.einsum("bnm,bnl->bml", phi_in, wphi_in)
    rhs = torch.einsum("bnm,bcn->bmc", wphi_in, x)

    eye = torch.eye(n_basis, dtype=x.dtype, device=x.device).unsqueeze(0)
    coef = torch.linalg.solve(gram + eps * eye, rhs)

    x_out = torch.einsum("bom,bmc->bco", phi_out, coef)
    return x_out


def given_weight_interp(
    x,
    directed_edges_cross,
    edge_weights,
    n_out=None,
):
    batch_size, channels, n_in = x.shape
    target, source = directed_edges_cross[..., 0], directed_edges_cross[..., 1]

    if n_out is None:
        valid_target = target[target >= 0]
        if valid_target.numel() == 0:
            raise ValueError("Cannot infer n_out because directed_edges_cross has no valid target indices.")
        n_out = int(valid_target.max().item()) + 1

    valid = (target >= 0) & (target < n_out) & (source >= 0) & (source < n_in)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_source = torch.where(valid, source, torch.zeros_like(source))
    safe_weight = torch.where(valid, edge_weights, torch.zeros_like(edge_weights))

    f = x.permute(0, 2, 1)  # [batch_size, n_in, channels]
    batch_index = torch.arange(batch_size, device=x.device).unsqueeze(1)
    source_feat = f[batch_index, safe_source]  # [batch_size, max_edges, channels]

    message = source_feat * safe_weight.unsqueeze(-1)

    x_out = torch.zeros(batch_size, n_out, channels, dtype=x.dtype, device=x.device)
    x_out.scatter_add_(
        dim=1,
        src=message,
        index=safe_target.unsqueeze(-1).repeat(1, 1, channels),
    )

    return x_out.permute(0, 2, 1)


def compute_taylor_weight(
    nodes_in,
    nodes_out,
    directed_edges_cross,
    n_out=None,
    eps=1e-8,
):
    """
    Compute first-order Taylor interpolation weights on cross edges.

    For each output node x, this finds coefficients c_i on neighboring input nodes x_i
    such that:
        sum_i c_i = 1,
        sum_i c_i * (x_i - x) = 0.
    Hence sum_i c_i f(x_i) = f(x) + O(h^2) for smooth scalar/vector features.

    Parameters:
        nodes_in : Tensor [batch_size, n_in, ndims]
            Input node coordinates.
        nodes_out : Tensor [batch_size, n_out, ndims]
            Output node coordinates.
        directed_edges_cross : LongTensor [batch_size, max_edges, 2]
            Each edge is (target_out, source_in).
        n_out : int or None
            Number of output nodes.
        eps : float
            Numerical stability constant.

    Returns:
        edge_weights_cross : Tensor [batch_size, max_edges]
            Taylor interpolation coefficient for each cross edge.
    """
    batch_size, n_in, ndims = nodes_in.shape
    max_edges = directed_edges_cross.shape[1]

    if n_out is None:
        n_out = nodes_out.shape[1]

    device = nodes_in.device
    dtype = nodes_in.dtype

    target, source = directed_edges_cross[..., 0], directed_edges_cross[..., 1]
    edge_weights_cross = torch.zeros(batch_size, max_edges, dtype=dtype, device=device)

    b_vec = torch.zeros(ndims + 1, 1, dtype=dtype, device=device)
    b_vec[0, 0] = 1.0

    for b in range(batch_size):
        target_batch = target[b]
        source_batch = source[b]

        valid_edge = (
            (target_batch >= 0)
            & (target_batch < n_out)
            & (source_batch >= 0)
            & (source_batch < n_in)
        )

        for t in torch.unique(target_batch[valid_edge]):
            t_int = int(t.item())
            edge_mask = valid_edge & (target_batch == t)
            edge_idx = torch.nonzero(edge_mask, as_tuple=False).squeeze(-1)
            src = source_batch[edge_idx]

            if src.numel() == 0:
                continue

            x0 = nodes_out[b, t_int]
            xi = nodes_in[b, src]
            di = xi - x0
            k = src.numel()

            if k < ndims + 1:
                dist = torch.linalg.norm(di, dim=-1)
                nearest_local = torch.argmin(dist)
                edge_weights_cross[b, edge_idx[nearest_local]] = 1.0
                continue

            ones = torch.ones(1, k, dtype=dtype, device=device)
            A = torch.cat([ones, di.T], dim=0)

            dist2 = torch.sum(di * di, dim=-1)
            local_weight = 1.0 / torch.clamp(dist2, min=eps)
            W = torch.diag(local_weight)

            M = A @ W @ A.T
            M = M + eps * torch.eye(ndims + 1, dtype=dtype, device=device)

            coeff = W @ A.T @ torch.linalg.solve(M, b_vec)
            coeff = coeff.squeeze(-1)  # [k]

            edge_weights_cross[b, edge_idx] = coeff

    return edge_weights_cross


def taylor_interp(
    x,
    nodes_in,
    nodes_out,
    directed_edges,
    n_out=None,
    eps=1e-8,
):
    if n_out is None:
        n_out = nodes_out.shape[1]

    edge_weights_cross = compute_taylor_weight(nodes_in=nodes_in, nodes_out=nodes_out, directed_edges_cross=directed_edges, n_out=n_out, eps=eps)
    x_out = given_weight_interp(x=x, directed_edges_cross=directed_edges, edge_weights=edge_weights_cross, n_out=n_out)
    
    return x_out
