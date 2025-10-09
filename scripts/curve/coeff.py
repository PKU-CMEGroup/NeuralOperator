import numpy as np
from typing import Callable, Literal, Optional, Dict
import numpy as np
import math
from tqdm import tqdm
import os

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

DomainType = Literal["box", "ball"]

def _volume(domain, domain_type: DomainType, n: int):
    if domain_type == "box":
        d = np.asarray(domain, float)
        if d.ndim != 2 or d.shape[1] != 2:
            raise ValueError("For 'box' domain_type, domain must be a 2D array/list with shape (n, 2) representing intervals.")
        return np.prod(d[:,1] - d[:,0])
    if domain_type == "ball":
        from math import pi, gamma
        R = float(domain)
        return (pi ** (n/2)) / gamma(n/2 + 1) * R**n
    raise ValueError("domain_type")

def _sample(domain, domain_type: DomainType, n: int, N: int, rng: np.random.Generator):
    if domain_type == "box":
        d = np.asarray(domain, float)
        return rng.random((N,n)) * (d[:,1]-d[:,0]) + d[:,0]
    if domain_type == "ball":
        R = float(domain)
        Z = rng.normal(size=(N,n))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        u = rng.random(N) ** (1.0 / n)
        return Z * (u[:,None] * R)
    raise ValueError("domain_type")

def project_irregular(
    k_func: Callable[[np.ndarray], np.ndarray],
    modes: np.ndarray,
    domain,
    domain_type: DomainType = "box",
    oversampling: float = 4.0,
    rng: Optional[np.random.Generator] = None,
    reg: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Real trigonometric (1, cos, sin) projection:
        f(x) ≈ c0 + Σ_k [ a_k cos(m_k · x) + b_k sin(m_k · x) ]
    Input modes are the frequency vectors m_k (only one copy per |m|; do NOT
    include the negative modes — the sine/cos basis already spans them).
    Returns both separated (c0, a, b) and a flattened coeffs array:
        coeffs = [c0, a_0, b_0, a_1, b_1, ...]
    Notes
    -----
    - Weighted Monte Carlo least squares (same weighting idea as before).
    - reg adds to diagonal of normal matrix.
    - Target k_func assumed (numerically) real; if complex, its real part is used.
    """
    modes = np.asarray(modes, float)
    M, n = modes.shape
    if rng is None:
        rng = np.random.default_rng()
    # Number of MC samples
    N = max(M + 5, int(np.ceil(oversampling * (M if M > 0 else 1))))
    X = _sample(domain, domain_type, n, N, rng)
    vol = _volume(domain, domain_type, n)

    # Evaluate function
    try:
        f = k_func(X)
    except Exception:
        f = np.array([k_func(x) for x in X])
    f = np.asarray(f, float)  # real projection

    theta = X @ modes.T  # (N,M)
    C = np.cos(theta)
    S = np.sin(theta)

    # Weight
    w = math.sqrt(vol / N)

    # Design matrix A_w (weighted)
    # Column order: [1, cos(m0·x), sin(m0·x), cos(m1·x), sin(m1·x), ...]
    ones = np.ones((N, 1))
    A_blocks = [ones]
    for j in range(M):
        A_blocks.append(C[:, j:j+1])
        A_blocks.append(S[:, j:j+1])
    A = np.hstack(A_blocks) * w  # (N, 1+2M)
    f_w = f * w

    D = 1 + 2 * M  # total columns
    # Normal equations
    # Use direct (regularized) least squares instead of normal equations
    if reg > 0:
        # Ridge (Tikhonov) regularization: solve min ||A x - f_w||^2 + reg ||x||^2
        A_aug = np.vstack([A, math.sqrt(reg) * np.eye(D)])
        f_aug = np.concatenate([f_w, np.zeros(D)])
        sol, *_ = np.linalg.lstsq(A_aug, f_aug, rcond=None)
    else:
        sol, *_ = np.linalg.lstsq(A, f_w, rcond=None)

    c0 = sol[0]
    a = sol[1::2][:M]  # cosine coeffs
    b = sol[2::2][:M]  # sine coeffs

    coeffs_flat = sol
    cond = np.linalg.cond(A)

    return dict(
        coeffs=coeffs_flat,
        c0=c0,
        coeffs_cos=a,
        coeffs_sin=b,
        sample_points=X,
        volume=vol,
        reg=reg,
        cond=cond,
        basis_type="trig_real"
    )

def reconstruct(modes: np.ndarray, coeffs, X: np.ndarray):
    """
    Real trig reconstruction:
        f(x) = c0 + Σ_k [ a_k cos(m_k·x) + b_k sin(m_k·x) ]
    coeffs can be:
      1) flattened array: [c0, a0, b0, a1, b1, ...]
      2) dict with keys: c0, coeffs_cos, coeffs_sin
    """
    modes = np.asarray(modes, float)
    X = np.asarray(X, float)
    M = modes.shape[0]

    if isinstance(coeffs, dict):
        c0 = float(coeffs["c0"])
        a = np.asarray(coeffs["coeffs_cos"], float)
        b = np.asarray(coeffs["coeffs_sin"], float)
    else:
        coeffs = np.asarray(coeffs, float)
        expected = 1 + 2 * M
        if coeffs.size != expected:
            raise ValueError(f"coeffs length {coeffs.size} != 1+2*M = {expected}")
        c0 = coeffs[0]
        a = coeffs[1::2][:M]
        b = coeffs[2::2][:M]

    if M == 0:
        return np.full(X.shape[0], c0, dtype=float)

    theta = X @ modes.T  # (N,M)
    return c0 + (np.cos(theta) * a + np.sin(theta) * b).sum(axis=1)


def estimate_L2_error(
    k_func: Callable[[np.ndarray], np.ndarray],
    modes: np.ndarray,
    coeffs,
    domain,
    domain_type: DomainType = "box",
    num_val: int = 20000,
    rng: Optional[np.random.Generator] = None
):
    modes = np.asarray(modes, float)
    n = modes.shape[1] if modes.size else (len(domain) if domain_type == "box" else None)
    if rng is None:
        rng = np.random.default_rng()
    Xv = _sample(domain, domain_type, n, num_val, rng)
    vol = _volume(domain, domain_type, n)

    try:
        fv = k_func(Xv)
    except Exception:
        fv = np.array([k_func(x) for x in Xv])
    fv = np.asarray(fv, float)

    fv_hat = reconstruct(modes, coeffs, Xv)
    diff2 = (fv - fv_hat) ** 2
    f2 = fv ** 2

    L2_err = math.sqrt(vol * diff2.mean())
    L2_norm = math.sqrt(vol * f2.mean()) + 1e-14
    return dict(rel_L2_error=L2_err / L2_norm)


def plot_result(modes, res, domain, k_func, nx=200, ny=200, slice_y=0.0, show=True):
    """
    可视化:
      1. 原函数
      2. 重构函数
      3. 差值
      4. y = slice_y 截面
      5. 频率平面上系数幅值 sqrt(a_k^2 + b_k^2)
    """
    import matplotlib.pyplot as plt
    (x0, x1), (y0, y1) = domain
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    try:
        f_true = k_func(XY).reshape(ny, nx).astype(float)
    except Exception:
        f_true = np.array([k_func(xy) for xy in XY]).reshape(ny, nx).astype(float)

    f_rec = reconstruct(modes, res, XY).reshape(ny, nx)
    f_err = f_rec - f_true

    vmax = np.max(np.abs(f_true))
    vrec = np.max(np.abs(f_rec))
    common_v = max(vmax, vrec)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1.4], hspace=0.25, wspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(f_true, extent=[x0, x1, y0, y1], origin='lower',
                     cmap='viridis', vmin=-common_v, vmax=common_v, aspect='auto')
    ax0.set_title("Original f")
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    plt.colorbar(im0, ax=ax0, shrink=0.8)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(f_rec, extent=[x0, x1, y0, y1], origin='lower',
                     cmap='viridis', vmin=-common_v, vmax=common_v, aspect='auto')
    ax1.set_title("Reconstructed f̂")
    ax1.set_xlabel("x")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(f_err, extent=[x0, x1, y0, y1], origin='lower',
                     cmap='coolwarm', aspect='auto')
    ax2.set_title("Difference f̂ - f")
    ax2.set_xlabel("x")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    j_mid = np.argmin(np.abs(ys - slice_y))
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(xs, f_true[j_mid], label=f"f (y≈{ys[j_mid]:.3g})")
    ax3.plot(xs, f_rec[j_mid], '--', label=f"f̂ (y≈{ys[j_mid]:.3g})")
    ax3.set_title("Slice")
    ax3.set_xlabel("x"); ax3.set_ylabel("value")
    ax3.legend(frameon=False)

    ax4 = fig.add_subplot(gs[1, 2])
    modes = np.asarray(modes, float)
    M, dim = modes.shape
    a = np.asarray(res["coeffs_cos"]) if "coeffs_cos" in res else np.asarray(res["coeffs"][1::2][:M])
    b = np.asarray(res["coeffs_sin"]) if "coeffs_sin" in res else np.asarray(res["coeffs"][2::2][:M])
    mag = np.sqrt(a**2 + b**2)
    if dim != 2:
        ax4.text(0.5, 0.5, f"Cannot scatter\nmodes (dim={dim} ≠ 2)", ha='center', va='center')
        ax4.set_axis_off()
    else:
        sc = ax4.scatter(modes[:, 0], modes[:, 1], c=mag, s=16,
                         edgecolors='k', linewidths=0.3)
        ax4.set_title(r"$\sqrt{a_k^2 + b_k^2}$")
        ax4.set_xlabel("m_x"); ax4.set_ylabel("m_y")
        plt.colorbar(sc, ax=ax4, shrink=0.75, label="amplitude")

    fig.suptitle("Projection / Reconstruction Summary (Real Trig Basis)", y=0.97, fontsize=15)
    if show:
        plt.show()
    return fig

def nonlinear_scale(modes, scale=1.0):
    norms = np.linalg.norm(modes, axis=1)
    max_norm = norms.max()
    scaled_norms = (norms / max_norm) ** scale
    modes_scaled = modes * scaled_norms[:, np.newaxis]
    return modes_scaled

if __name__ == "__main__":

    from quasi_sphere.modes_discrete import discrete_half_ball_modes
    from pcno.pcno import compute_Fourier_modes
    import numpy as np
#-------------------------------------------------------------------------
    n = 2
    k_layer = 16
    scale = 0
    discrete_type = "cube"  # "cube" or "half_ball"
    domain = [(-5, 5),(-5, 5)]
    L = 10
    def k_func(X):
        x, y = X[:,0], X[:,1]
        # return x
        # return np.sin(3*x) * np.cos(2*y)
        # return np.log(np.sqrt(x**2 + y**2) + 1e-8)  
        return 1/np.sqrt(x**2 + y**2 + 1e-2)
#-------------------------------------------------------------------------

    if discrete_type == "half_ball":
        modes = discrete_half_ball_modes(n, k_layer, scale=scale, min_dir_fraction=0) * k_layer * 2 * np.pi / L 
    elif discrete_type == "cube":
        modes = compute_Fourier_modes(n, [k_layer,k_layer], [L,L]).squeeze() 
        modes = nonlinear_scale(modes, scale=scale)
    
    res = project_irregular(k_func, modes, domain, oversampling=5, reg=1e-8)
    err = estimate_L2_error(k_func, modes, res["coeffs"], domain, num_val=1000)
    print("modes数:", len(modes), "\nerror:", err)

    plot_result(modes, res, domain, k_func)
