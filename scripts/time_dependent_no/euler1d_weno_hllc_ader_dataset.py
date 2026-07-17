"""
Generate a randomized 1D Euler dataset with
    WENO5 reconstruction + HLLC flux + ADER2/Lax-Wendroff predictor,
for Sod-like shock-tube initial data with
    left fixed inflow boundary and right reflective wall.

The saved dataset uses primitive variables [rho, u, p].

Important note
--------------
This is a finite-volume ADER2 implementation: WENO5 reconstructs spatial
interface states and a local Cauchy--Kowalewski / Lax--Wendroff predictor
advances those interface states to t^{n+1/2}. This is commonly used as a
second-order ADER-Hancock-type predictor. It is not a full high-order ADER-DG
local space-time predictor.

Usage
-----
Edit the CONFIG block below, then run:
python euler1d_weno_hllc_ader_dataset.py

Snapshots are saved at fixed times 0, T/n_steps, ..., T. The internal solver
may use smaller CFL-limited substeps between two saved snapshots.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Tuple
import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "time_dependent_no"


@dataclass
class CaseConfig:
    x_left: float
    x_right: float
    x_disc: float
    left_state: np.ndarray  # primitive [rho, u, p]
    right_state: np.ndarray  # primitive [rho, u, p]
    t_final: float


@dataclass(frozen=True)
class SampleConfig:
    domain_length_range: Tuple[float, float] = (0.85, 1.20)
    domain_left_range: Tuple[float, float] = (-0.10, 0.10)
    discontinuity_fraction_range: Tuple[float, float] = (0.15, 0.35)
    left_rho_range: Tuple[float, float] = (0.80, 1.40)
    left_u_range: Tuple[float, float] = (0.60, 2.20)
    left_p_range: Tuple[float, float] = (0.90, 3.00)
    right_rho_range: Tuple[float, float] = (0.08, 0.28)
    right_p_range: Tuple[float, float] = (0.06, 0.30)


@dataclass(frozen=True)
class DatasetConfig:
    out_path: str | Path = ARTIFACT_DIR / "euler1d_weno_hllc_ader_dataset.npz"
    n_cases: int = 1
    n_steps: int = 100
    nx: int = 256
    t_final: float = 0.5
    gamma: float = 1.4
    cfl: float = 0.25
    use_shock_flattening: bool = True
    use_hlle_on_troubled_faces: bool = True
    shock_sensor_threshold: float = 0.05
    shock_flatten_radius: int = 4
    seed: int = 1234
    num_workers: int = 1
    ng: int = 3
    rho_floor: float = 1e-12
    p_floor: float = 1e-12
    initialization_mode: str = "cell_center"
    storage_dtype: str = "float32"
    sample: SampleConfig = field(default_factory=SampleConfig)
    verbose: bool = True
    save_face_flux_integral: bool = False
    save_gif: bool = True
    gif_path: str | Path = ARTIFACT_DIR / "euler1d_weno_hllc_ader_sample0.gif"
    gif_case_id: int = 0
    gif_fps: int = 20
    gif_dpi: int = 120
    plot: bool = False
    plot_case_id: int = 0
    plot_time_ids: Tuple[int, ...] = (0, 25, 50, 75, 100)


# Edit this block to set the dataset-generation parameters.
CONFIG = DatasetConfig(
    out_path=ARTIFACT_DIR / "euler1d_weno_hllc_ader_dataset.npz",
    n_cases=1,
    n_steps=100,
    nx=2048,
    t_final=0.5,
    gamma=1.4,
    cfl=0.2,
    use_shock_flattening=True,
    use_hlle_on_troubled_faces=True,
    shock_sensor_threshold=0.05,
    shock_flatten_radius=4,
    seed=2401,
    num_workers=1,
    ng=3,
    rho_floor=1e-12,
    p_floor=1e-12,
    storage_dtype="float32",
    sample=SampleConfig(
        domain_length_range=(1.0, 1.0),
        domain_left_range=(0.10, 0.20),
        discontinuity_fraction_range=(0.15, 0.35),
        left_rho_range=(1, 1.50),
        left_u_range=(0.60, 2.40),
        left_p_range=(0.90, 4.00),
        right_rho_range=(0.1, 0.3),
        right_p_range=(0.1, 0.40),
    ),
    verbose=True,
    save_gif=True,
    gif_path=ARTIFACT_DIR / "euler1d_weno_hllc_ader_sample0.gif",
    gif_case_id=0,
    gif_fps=20,
    gif_dpi=120,
    plot=False,
    plot_case_id=0,
    plot_time_ids=(0, 25, 50, 75, 100),
)


# -----------------------------------------------------------------------------
# Euler variable conversions and fluxes
# -----------------------------------------------------------------------------


def primitive_to_conservative(V: np.ndarray, gamma: float) -> np.ndarray:
    """Convert primitive variables V=[rho,u,p] to conservative U=[rho,rho*u,E]."""
    V = np.asarray(V)
    rho = V[..., 0]
    u = V[..., 1]
    p = V[..., 2]
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E], axis=-1)


def conservative_to_primitive(
    U: np.ndarray,
    gamma: float,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
) -> np.ndarray:
    """Convert conservative variables U=[rho,rho*u,E] to primitive V=[rho,u,p]."""
    U = np.asarray(U)
    rho = np.maximum(U[..., 0], rho_floor)
    u = U[..., 1] / rho
    kinetic = 0.5 * rho * u * u
    p = (gamma - 1.0) * (U[..., 2] - kinetic)
    p = np.maximum(p, p_floor)
    return np.stack([rho, u, p], axis=-1)


def euler_flux_from_primitive(V: np.ndarray, gamma: float) -> np.ndarray:
    """Physical Euler flux F(U), with input primitive V=[rho,u,p]."""
    rho = V[..., 0]
    u = V[..., 1]
    p = V[..., 2]
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.stack([rho * u, rho * u * u + p, u * (E + p)], axis=-1)


def euler_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """Physical Euler flux F(U), with input conservative U=[rho,rho*u,E]."""
    return euler_flux_from_primitive(conservative_to_primitive(U, gamma), gamma)


def hllc_flux_from_primitive(
    VL: np.ndarray,
    VR: np.ndarray,
    gamma: float,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized HLLC numerical flux for 1D Euler.

    VL, VR have shape (..., 3), representing primitive left/right states at
    interfaces. The returned flux has shape (..., 3) in conservative variables.
    """
    VL = np.asarray(VL).copy()
    VR = np.asarray(VR).copy()

    VL[..., 0] = np.maximum(VL[..., 0], rho_floor)
    VR[..., 0] = np.maximum(VR[..., 0], rho_floor)
    VL[..., 2] = np.maximum(VL[..., 2], p_floor)
    VR[..., 2] = np.maximum(VR[..., 2], p_floor)

    rhoL, uL, pL = VL[..., 0], VL[..., 1], VL[..., 2]
    rhoR, uR, pR = VR[..., 0], VR[..., 1], VR[..., 2]

    UL = primitive_to_conservative(VL, gamma)
    UR = primitive_to_conservative(VR, gamma)
    FL = euler_flux_from_primitive(VL, gamma)
    FR = euler_flux_from_primitive(VR, gamma)

    EL = UL[..., 2]
    ER = UR[..., 2]
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    # Davis/Einfeldt-type outer wave speed estimates.
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    denom = np.where(
        np.abs(denom) < 1e-14, np.where(denom >= 0.0, 1e-14, -1e-14), denom
    )
    SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom

    def safe_nonzero(a: np.ndarray, eps_safe: float = 1e-14) -> np.ndarray:
        """Avoid division by zero while preserving the sign of wave-speed denominators."""
        return np.where(
            np.abs(a) < eps_safe, np.where(a >= 0.0, eps_safe, -eps_safe), a
        )

    # Toro's HLLC star-state formula. Denominator signs must be preserved.
    denom_star_L = safe_nonzero(SL - SM)
    denom_star_R = safe_nonzero(SR - SM)
    denom_energy_L = safe_nonzero(rhoL * (SL - uL))
    denom_energy_R = safe_nonzero(rhoR * (SR - uR))

    rho_star_L = rhoL * (SL - uL) / denom_star_L
    rho_star_R = rhoR * (SR - uR) / denom_star_R

    E_star_L = rho_star_L * (EL / rhoL + (SM - uL) * (SM + pL / denom_energy_L))
    E_star_R = rho_star_R * (ER / rhoR + (SM - uR) * (SM + pR / denom_energy_R))

    U_star_L = np.stack([rho_star_L, rho_star_L * SM, E_star_L], axis=-1)
    U_star_R = np.stack([rho_star_R, rho_star_R * SM, E_star_R], axis=-1)

    F_star_L = FL + SL[..., None] * (U_star_L - UL)
    F_star_R = FR + SR[..., None] * (U_star_R - UR)

    flux = np.empty_like(FL)
    mask_L = 0.0 <= SL
    mask_star_L = (SL <= 0.0) & (0.0 <= SM)
    mask_star_R = (SM <= 0.0) & (0.0 <= SR)
    mask_R = SR <= 0.0

    flux[mask_L] = FL[mask_L]
    flux[mask_star_L] = F_star_L[mask_star_L]
    flux[mask_star_R] = F_star_R[mask_star_R]
    flux[mask_R] = FR[mask_R]

    # Rare pathological states can leave a gap because of near-degenerate speeds.
    mask_unset = ~(mask_L | mask_star_L | mask_star_R | mask_R)
    if np.any(mask_unset):
        flux[mask_unset] = 0.5 * (FL[mask_unset] + FR[mask_unset])

    return flux


def hlle_flux_from_primitive(
    VL: np.ndarray,
    VR: np.ndarray,
    gamma: float,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
) -> np.ndarray:
    """Vectorized HLLE flux used as a more dissipative shock fallback."""
    VL = np.asarray(VL).copy()
    VR = np.asarray(VR).copy()

    VL[..., 0] = np.maximum(VL[..., 0], rho_floor)
    VR[..., 0] = np.maximum(VR[..., 0], rho_floor)
    VL[..., 2] = np.maximum(VL[..., 2], p_floor)
    VR[..., 2] = np.maximum(VR[..., 2], p_floor)

    rhoL, uL, pL = VL[..., 0], VL[..., 1], VL[..., 2]
    rhoR, uR, pR = VR[..., 0], VR[..., 1], VR[..., 2]

    UL = primitive_to_conservative(VL, gamma)
    UR = primitive_to_conservative(VR, gamma)
    FL = euler_flux_from_primitive(VL, gamma)
    FR = euler_flux_from_primitive(VR, gamma)

    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    denom = np.maximum(SR - SL, 1e-14)

    F_hlle = (
        SR[..., None] * FL - SL[..., None] * FR + (SL * SR)[..., None] * (UR - UL)
    ) / denom[..., None]
    return np.where(
        (SL >= 0.0)[..., None],
        FL,
        np.where((SR <= 0.0)[..., None], FR, F_hlle),
    )


# -----------------------------------------------------------------------------
# Boundary conditions and WENO5 reconstruction
# -----------------------------------------------------------------------------


def apply_boundary_conditions(
    U: np.ndarray,
    left_state: np.ndarray,
    gamma: float,
    ng: int,
) -> None:
    """
    Fill ghost cells in-place.

    Left boundary: fixed inflow primitive state = left_state.
    Right boundary: reflective wall, mirrored density/energy and reversed momentum.
    """
    U_left = primitive_to_conservative(np.asarray(left_state), gamma)
    U[:ng, :] = U_left

    # Right reflective wall. Mirror physical cells into ghost cells.
    # Ghost ng+nx+k mirrors inner cell ng+nx-1-k.
    nx = U.shape[0] - 2 * ng
    for k in range(ng):
        src = ng + nx - 1 - k
        dst = ng + nx + k
        U[dst, 0] = U[src, 0]
        U[dst, 1] = -U[src, 1]
        U[dst, 2] = U[src, 2]


def weno5_reconstruct_faces(
    V: np.ndarray,
    j_faces: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fifth-order finite-volume WENO-JS reconstruction of primitive variables.

    For each integer j in j_faces, reconstruct states at interface x_{j+1/2}:
        VL[j] = value from cell j side,
        VR[j] = value from cell j+1 side.

    V has shape [n_total_cells, 3]. Valid j requires j-2 >= 0 and j+3 < n_total.
    """
    jm2 = V[j_faces - 2]
    jm1 = V[j_faces - 1]
    j0 = V[j_faces]
    jp1 = V[j_faces + 1]
    jp2 = V[j_faces + 2]
    jp3 = V[j_faces + 3]

    # Left-biased value at x_{j+1/2}^-.
    q0 = (2.0 * jm2 - 7.0 * jm1 + 11.0 * j0) / 6.0
    q1 = (-jm1 + 5.0 * j0 + 2.0 * jp1) / 6.0
    q2 = (2.0 * j0 + 5.0 * jp1 - jp2) / 6.0

    beta0 = (13.0 / 12.0) * (jm2 - 2.0 * jm1 + j0) ** 2 + 0.25 * (
        jm2 - 4.0 * jm1 + 3.0 * j0
    ) ** 2
    beta1 = (13.0 / 12.0) * (jm1 - 2.0 * j0 + jp1) ** 2 + 0.25 * (jm1 - jp1) ** 2
    beta2 = (13.0 / 12.0) * (j0 - 2.0 * jp1 + jp2) ** 2 + 0.25 * (
        3.0 * j0 - 4.0 * jp1 + jp2
    ) ** 2

    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (eps + beta0) ** 2
    a1 = d1 / (eps + beta1) ** 2
    a2 = d2 / (eps + beta2) ** 2
    asum = a0 + a1 + a2
    w0 = a0 / asum
    w1 = a1 / asum
    w2 = a2 / asum
    VL = w0 * q0 + w1 * q1 + w2 * q2

    # Right-biased value at x_{j+1/2}^+; equivalent to applying the same
    # left reconstruction to the reversed array.
    q0r = (2.0 * jp3 - 7.0 * jp2 + 11.0 * jp1) / 6.0
    q1r = (-jp2 + 5.0 * jp1 + 2.0 * j0) / 6.0
    q2r = (2.0 * jp1 + 5.0 * j0 - jm1) / 6.0

    beta0r = (13.0 / 12.0) * (jp3 - 2.0 * jp2 + jp1) ** 2 + 0.25 * (
        jp3 - 4.0 * jp2 + 3.0 * jp1
    ) ** 2
    beta1r = (13.0 / 12.0) * (jp2 - 2.0 * jp1 + j0) ** 2 + 0.25 * (jp2 - j0) ** 2
    beta2r = (13.0 / 12.0) * (jp1 - 2.0 * j0 + jm1) ** 2 + 0.25 * (
        3.0 * jp1 - 4.0 * j0 + jm1
    ) ** 2

    a0r = d0 / (eps + beta0r) ** 2
    a1r = d1 / (eps + beta1r) ** 2
    a2r = d2 / (eps + beta2r) ** 2
    asumr = a0r + a1r + a2r
    w0r = a0r / asumr
    w1r = a1r / asumr
    w2r = a2r / asumr
    VR = w0r * q0r + w1r * q1r + w2r * q2r

    return VL, VR


def positivity_fallback_faces(
    VL: np.ndarray,
    VR: np.ndarray,
    V: np.ndarray,
    j_faces: np.ndarray,
    rho_floor: float = 1e-10,
    p_floor: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If WENO overshoots into negative density/pressure, replace that interface
    state by the neighboring cell-centered primitive state.
    """
    bad_L = (
        (~np.isfinite(VL).all(axis=-1))
        | (VL[:, 0] <= rho_floor)
        | (VL[:, 2] <= p_floor)
    )
    bad_R = (
        (~np.isfinite(VR).all(axis=-1))
        | (VR[:, 0] <= rho_floor)
        | (VR[:, 2] <= p_floor)
    )
    if np.any(bad_L):
        VL[bad_L] = V[j_faces[bad_L]]
    if np.any(bad_R):
        VR[bad_R] = V[j_faces[bad_R] + 1]
    VL[:, 0] = np.maximum(VL[:, 0], rho_floor)
    VR[:, 0] = np.maximum(VR[:, 0], rho_floor)
    VL[:, 2] = np.maximum(VL[:, 2], p_floor)
    VR[:, 2] = np.maximum(VR[:, 2], p_floor)
    return VL, VR


def reflect_primitive_state(V: np.ndarray) -> np.ndarray:
    """Mirror a primitive state across a stationary wall."""
    V_reflected = np.asarray(V).copy()
    V_reflected[..., 1] *= -1.0
    return V_reflected


def enforce_boundary_face_states(
    VL: np.ndarray,
    VR: np.ndarray,
    left_state: np.ndarray,
) -> None:
    """
    Enforce exact boundary states at the two domain faces.

    The right reflective wall must see mirrored primitive states at the boundary
    face; otherwise the HLLC flux can develop nonzero wall mass/energy flux
    after the ADER half-step predictor.
    """
    VL[0] = np.asarray(left_state, dtype=VL.dtype)
    VR[-1] = reflect_primitive_state(VL[-1])


def expand_face_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a boolean face mask by a fixed number of neighboring faces."""
    expanded = np.asarray(mask, dtype=bool).copy()
    for offset in range(1, max(0, int(radius)) + 1):
        expanded[offset:] |= mask[:-offset]
        expanded[:-offset] |= mask[offset:]
    return expanded


def shock_troubled_faces(
    V: np.ndarray,
    j_faces: np.ndarray,
    threshold: float,
    radius: int,
    p_floor: float,
) -> np.ndarray:
    """Detect faces near strong pressure jumps where WENO should be flattened."""
    p_left = V[j_faces, 2]
    p_right = V[j_faces + 1, 2]
    pressure_jump = np.abs(p_right - p_left) / np.maximum(p_left + p_right, p_floor)
    return expand_face_mask(pressure_jump > threshold, radius)


def set_first_order_faces(
    VL: np.ndarray,
    VR: np.ndarray,
    V: np.ndarray,
    j_faces: np.ndarray,
    face_mask: np.ndarray,
) -> None:
    """Replace selected reconstructed face states by neighboring cell averages."""
    if not np.any(face_mask):
        return
    VL[face_mask] = V[j_faces[face_mask]]
    VR[face_mask] = V[j_faces[face_mask] + 1]


def troubled_cells_from_faces(face_mask: np.ndarray, nx: int) -> np.ndarray:
    """Mark physical cells adjacent to troubled faces."""
    cells = np.zeros(nx, dtype=bool)
    if not np.any(face_mask):
        return cells
    face_ids = np.flatnonzero(face_mask)
    left_cells = face_ids - 1
    right_cells = face_ids
    left_cells = left_cells[(0 <= left_cells) & (left_cells < nx)]
    right_cells = right_cells[(0 <= right_cells) & (right_cells < nx)]
    cells[left_cells] = True
    cells[right_cells] = True
    return cells


# -----------------------------------------------------------------------------
# ADER2 predictor and finite-volume update
# -----------------------------------------------------------------------------


def primitive_A_times_grad(V: np.ndarray, dVdx: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute A(V) dV/dx for primitive Euler equations:
        rho_t + u rho_x + rho u_x = 0,
        u_t   + u u_x   + p_x/rho = 0,
        p_t   + u p_x   + gamma p u_x = 0.
    """
    rho = V[..., 0]
    u = V[..., 1]
    p = V[..., 2]
    rho_x = dVdx[..., 0]
    u_x = dVdx[..., 1]
    p_x = dVdx[..., 2]

    return np.stack(
        [
            u * rho_x + rho * u_x,
            u * u_x + p_x / np.maximum(rho, 1e-12),
            u * p_x + gamma * p * u_x,
        ],
        axis=-1,
    )


def step_weno_hllc_ader2(
    U: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    dt: float,
    gamma: float,
    ng: int,
    rho_floor: float = 1e-10,
    p_floor: float = 1e-10,
    use_shock_flattening: bool = True,
    use_hlle_on_troubled_faces: bool = True,
    shock_sensor_threshold: float = 0.05,
    shock_flatten_radius: int = 4,
    return_flux: bool = False,
) -> Tuple[np.ndarray, bool] | Tuple[np.ndarray, bool, np.ndarray]:
    """
    One finite-volume update using WENO5 + ADER2 predictor + HLLC flux.

    Return (U_new, success), optionally followed by the positive-x face flux.
    Ghost cells are filled inside this function.
    """
    U_work = U.copy()
    apply_boundary_conditions(U_work, left_state, gamma, ng)

    nx = U_work.shape[0] - 2 * ng
    j_faces = np.arange(ng - 1, ng + nx, dtype=np.int64)  # nx+1 update faces

    V = conservative_to_primitive(U_work, gamma, rho_floor, p_floor)
    VL, VR = weno5_reconstruct_faces(V, j_faces)
    VL, VR = positivity_fallback_faces(VL, VR, V, j_faces, rho_floor, p_floor)
    troubled_faces = np.zeros_like(j_faces, dtype=bool)
    if use_shock_flattening:
        troubled_faces = shock_troubled_faces(
            V,
            j_faces,
            shock_sensor_threshold,
            shock_flatten_radius,
            p_floor,
        )
        set_first_order_faces(VL, VR, V, j_faces, troubled_faces)
    enforce_boundary_face_states(VL, VR, left_state)

    # Cell-local WENO slopes for physical cells only:
    # slope_i = (right boundary state from cell i - left boundary state from cell i)/dx.
    # physical cell i corresponds to padded index ng+i.
    slopes = (VL[1:] - VR[:-1]) / dx  # shape [nx, 3]
    if use_shock_flattening:
        slopes[troubled_cells_from_faces(troubled_faces, nx)] = 0.0

    # For each face, assign the slope of its left and right neighboring cells.
    slopes_L = np.zeros_like(VL)
    slopes_R = np.zeros_like(VR)
    slopes_L[1:] = slopes  # left cell is physical for faces 1..nx
    slopes_R[:-1] = slopes  # right cell is physical for faces 0..nx-1

    # ADER2 / Cauchy--Kowalewski half-time predictor.
    VL_half = VL - 0.5 * dt * primitive_A_times_grad(VL, slopes_L, gamma)
    VR_half = VR - 0.5 * dt * primitive_A_times_grad(VR, slopes_R, gamma)
    if use_shock_flattening:
        set_first_order_faces(VL_half, VR_half, V, j_faces, troubled_faces)
    VL_half, VR_half = positivity_fallback_faces(
        VL_half, VR_half, V, j_faces, rho_floor, p_floor
    )
    enforce_boundary_face_states(VL_half, VR_half, left_state)

    F = hllc_flux_from_primitive(VL_half, VR_half, gamma, rho_floor, p_floor)
    hlle_faces = troubled_faces.copy()
    hlle_faces[0] = False
    hlle_faces[-1] = False
    if use_shock_flattening and use_hlle_on_troubled_faces and np.any(hlle_faces):
        F_hlle = hlle_flux_from_primitive(VL_half, VR_half, gamma, rho_floor, p_floor)
        F[hlle_faces] = F_hlle[hlle_faces]

    U_new = U_work.copy()
    U_new[ng : ng + nx] = U_work[ng : ng + nx] - (dt / dx) * (F[1:] - F[:-1])

    V_new = conservative_to_primitive(
        U_new[ng : ng + nx], gamma, rho_floor=-np.inf, p_floor=-np.inf
    )
    ok = (
        np.isfinite(U_new[ng : ng + nx]).all()
        and np.all(V_new[:, 0] > rho_floor)
        and np.all(V_new[:, 2] > p_floor)
    )
    if ok:
        apply_boundary_conditions(U_new, left_state, gamma, ng)
    if return_flux:
        return U_new, bool(ok), F
    return U_new, bool(ok)


def step_first_order_hllc(
    U: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    dt: float,
    gamma: float,
    ng: int,
    rho_floor: float = 1e-10,
    p_floor: float = 1e-10,
    return_flux: bool = False,
) -> Tuple[np.ndarray, bool] | Tuple[np.ndarray, bool, np.ndarray]:
    """Robust first-order fallback using piecewise-constant HLLC flux."""
    U_work = U.copy()
    apply_boundary_conditions(U_work, left_state, gamma, ng)
    nx = U_work.shape[0] - 2 * ng
    j_faces = np.arange(ng - 1, ng + nx, dtype=np.int64)
    V = conservative_to_primitive(U_work, gamma, rho_floor, p_floor)
    VL = V[j_faces]
    VR = V[j_faces + 1]
    F = hllc_flux_from_primitive(VL, VR, gamma, rho_floor, p_floor)
    U_new = U_work.copy()
    U_new[ng : ng + nx] = U_work[ng : ng + nx] - (dt / dx) * (F[1:] - F[:-1])
    V_new = conservative_to_primitive(
        U_new[ng : ng + nx], gamma, rho_floor=-np.inf, p_floor=-np.inf
    )
    ok = (
        np.isfinite(U_new[ng : ng + nx]).all()
        and np.all(V_new[:, 0] > rho_floor)
        and np.all(V_new[:, 2] > p_floor)
    )
    if ok:
        apply_boundary_conditions(U_new, left_state, gamma, ng)
    if return_flux:
        return U_new, bool(ok), F
    return U_new, bool(ok)


def stable_dt(U: np.ndarray, dx: float, cfl: float, gamma: float, ng: int) -> float:
    """CFL time step based on max(|u|+c) over physical cells."""
    nx = U.shape[0] - 2 * ng
    V = conservative_to_primitive(U[ng : ng + nx], gamma)
    rho = V[:, 0]
    u = V[:, 1]
    p = V[:, 2]
    c = np.sqrt(gamma * p / rho)
    max_speed = np.max(np.abs(u) + c)
    return cfl * dx / max(max_speed, 1e-14)


# -----------------------------------------------------------------------------
# Case sampling and time integration
# -----------------------------------------------------------------------------


def random_in_range(
    rng: np.random.Generator, value_range: Tuple[float, float]
) -> float:
    """Sample a scalar uniformly from a configured inclusive-low/exclusive-high range."""
    return float(rng.uniform(value_range[0], value_range[1]))


def sample_case(
    rng: np.random.Generator,
    sample_config: SampleConfig,
    t_final: float,
) -> CaseConfig:
    """
    Random Sod-like case with positive left velocity and random domain.

    The ranges are chosen so that a right-going shock usually reaches the right
    reflective wall within the saved time window.
    """
    length = random_in_range(rng, sample_config.domain_length_range)
    x_left = random_in_range(rng, sample_config.domain_left_range)
    x_right = x_left + length
    x_disc = (
        x_left
        + random_in_range(rng, sample_config.discontinuity_fraction_range) * length
    )

    # Sod-like but with positive inflow velocity and random amplitudes.
    rho_L = random_in_range(rng, sample_config.left_rho_range)
    u_L = random_in_range(rng, sample_config.left_u_range)
    p_L = random_in_range(rng, sample_config.left_p_range)

    rho_R = random_in_range(rng, sample_config.right_rho_range)
    u_R = 0.0
    p_R = random_in_range(rng, sample_config.right_p_range)

    return CaseConfig(
        x_left=float(x_left),
        x_right=float(x_right),
        x_disc=float(x_disc),
        left_state=np.array([rho_L, u_L, p_L], dtype=np.float64),
        right_state=np.array([rho_R, u_R, p_R], dtype=np.float64),
        t_final=float(t_final),
    )


def initialize_case(
    case: CaseConfig,
    nx: int,
    gamma: float,
    ng: int,
    initialization_mode: str = "cell_center",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Create grid and padded conservative state for one case."""
    x_edges = np.linspace(case.x_left, case.x_right, nx + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = (case.x_right - case.x_left) / nx

    if initialization_mode == "cell_center":
        V_phys = np.empty((nx, 3), dtype=np.float64)
        mask_left = x < case.x_disc
        V_phys[mask_left] = case.left_state
        V_phys[~mask_left] = case.right_state
        U_phys = primitive_to_conservative(V_phys, gamma)
    elif initialization_mode == "exact_cell_average":
        left_fraction = np.clip(
            (case.x_disc - x_edges[:-1]) / dx,
            0.0,
            1.0,
        )
        left_conservative = primitive_to_conservative(case.left_state, gamma)
        right_conservative = primitive_to_conservative(case.right_state, gamma)
        U_phys = (
            left_fraction[:, None] * left_conservative[None, :]
            + (1.0 - left_fraction[:, None]) * right_conservative[None, :]
        )
    else:
        raise ValueError(f"unsupported initialization_mode: {initialization_mode}")

    U = np.zeros((nx + 2 * ng, 3), dtype=np.float64)
    U[ng : ng + nx] = U_phys
    apply_boundary_conditions(U, case.left_state, gamma, ng)
    return x, U, dx


def integrate_case(
    case: CaseConfig,
    nx: int,
    n_steps: int,
    gamma: float,
    cfl: float,
    ng: int,
    rho_floor: float,
    p_floor: float,
    use_shock_flattening: bool,
    use_hlle_on_troubled_faces: bool,
    shock_sensor_threshold: float,
    shock_flatten_radius: int,
    return_face_flux_integral: bool = False,
    initialization_mode: str = "cell_center",
    storage_dtype: str = "float32",
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, float]
):
    """
    Integrate one case and return primitive snapshots.

    Returns
    -------
    x: [nx]
    t: [n_steps + 1]
    data: [n_steps + 1, nx, 3] primitive snapshots, including t=0 and t=T
    n_fallback: number of first-order fallback steps used
    face_flux_integral: optional [n_steps, nx + 1, 3] owner-oriented flux impulse
    closure_max_abs: optional maximum conservative closure error before serialization
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    data_dtype = np.dtype(storage_dtype)
    if data_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("storage_dtype must be float32 or float64")

    x, U, dx = initialize_case(
        case,
        nx,
        gamma,
        ng,
        initialization_mode=initialization_mode,
    )
    targets = np.linspace(0.0, case.t_final, n_steps + 1)
    data = np.empty((n_steps + 1, nx, 3), dtype=data_dtype)
    data[0] = conservative_to_primitive(U[ng : ng + nx], gamma).astype(data_dtype)
    face_flux_integral = (
        np.empty((n_steps, nx + 1, 3), dtype=np.float64)
        if return_face_flux_integral
        else None
    )
    closure_max_abs = 0.0

    t_now = 0.0
    fallback_count = 0

    for out_idx in range(1, n_steps + 1):
        target = float(targets[out_idx])
        interval_start = U[ng : ng + nx].copy()
        interval_flux_integral = (
            np.zeros((nx + 1, 3), dtype=np.float64)
            if return_face_flux_integral
            else None
        )
        # Advance exactly to the next output time.
        while t_now < target - 1e-14:
            dt = min(stable_dt(U, dx, cfl, gamma, ng), target - t_now)

            # Retry with smaller dt if WENO/ADER causes positivity failure.
            success = False
            dt_try = dt
            for _ in range(12):
                U_candidate, ok, step_flux = step_weno_hllc_ader2(
                    U,
                    case.left_state,
                    dx,
                    dt_try,
                    gamma,
                    ng,
                    rho_floor,
                    p_floor,
                    use_shock_flattening,
                    use_hlle_on_troubled_faces,
                    shock_sensor_threshold,
                    shock_flatten_radius,
                    return_flux=True,
                )
                if ok:
                    if interval_flux_integral is not None:
                        owner_oriented_flux = step_flux.copy()
                        owner_oriented_flux[0] *= -1.0
                        interval_flux_integral += dt_try * owner_oriented_flux
                    U = U_candidate
                    t_now += dt_try
                    success = True
                    break
                dt_try *= 0.5

            if not success:
                # Robust fallback: first-order HLLC with reduced step.
                # This should rarely be triggered; metadata records its frequency.
                dt_try = min(dt * 0.25, target - t_now)
                U_candidate, ok, step_flux = step_first_order_hllc(
                    U,
                    case.left_state,
                    dx,
                    dt_try,
                    gamma,
                    ng,
                    rho_floor,
                    p_floor,
                    return_flux=True,
                )
                if not ok:
                    raise RuntimeError(
                        "Positivity failure even after first-order fallback. "
                        "Try lowering cfl or reducing the sampled pressure/velocity ranges."
                    )
                if interval_flux_integral is not None:
                    owner_oriented_flux = step_flux.copy()
                    owner_oriented_flux[0] *= -1.0
                    interval_flux_integral += dt_try * owner_oriented_flux
                U = U_candidate
                t_now += dt_try
                fallback_count += 1

        if face_flux_integral is not None and interval_flux_integral is not None:
            face_flux_integral[out_idx - 1] = interval_flux_integral
            positive_x_integral = interval_flux_integral.copy()
            positive_x_integral[0] *= -1.0
            reconstructed = (
                interval_start
                - (positive_x_integral[1:] - positive_x_integral[:-1]) / dx
            )
            closure_max_abs = max(
                closure_max_abs,
                float(np.max(np.abs(U[ng : ng + nx] - reconstructed))),
            )

        data[out_idx] = conservative_to_primitive(U[ng : ng + nx], gamma).astype(
            data_dtype
        )

    result = (x.astype(np.float32), targets.astype(np.float32), data, fallback_count)
    if face_flux_integral is not None:
        return (*result, face_flux_integral.astype(np.float32), closure_max_abs)
    return result


def generate_dataset(
    out_path: str | Path,
    n_cases: int = 100,
    n_steps: int = 100,
    nx: int = 256,
    t_final: float = 0.5,
    gamma: float = 1.4,
    cfl: float = 0.35,
    use_shock_flattening: bool = True,
    use_hlle_on_troubled_faces: bool = True,
    shock_sensor_threshold: float = 0.05,
    shock_flatten_radius: int = 4,
    seed: int = 1234,
    ng: int = 3,
    rho_floor: float = 1e-10,
    p_floor: float = 1e-10,
    sample_config: SampleConfig = SampleConfig(),
    verbose: bool = True,
    save_face_flux_integral: bool = False,
    initialization_mode: str = "cell_center",
    num_workers: int = 1,
    storage_dtype: str = "float32",
) -> None:
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    if t_final <= 0.0:
        raise ValueError(f"t_final must be positive, got {t_final}.")
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dtype = np.dtype(storage_dtype)
    if data_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("storage_dtype must be float32 or float64")

    rng = np.random.default_rng(seed)
    cases = [sample_case(rng, sample_config, t_final) for _ in range(n_cases)]
    n_frames = n_steps + 1
    save_dt = t_final / n_steps

    data_all = np.empty((n_cases, n_frames, nx, 3), dtype=data_dtype)
    x_all = np.empty((n_cases, nx), dtype=np.float32)
    t_all = np.empty((n_cases, n_frames), dtype=np.float32)
    left_states = np.empty((n_cases, 3), dtype=np.float32)
    right_states = np.empty((n_cases, 3), dtype=np.float32)
    domains = np.empty((n_cases, 2), dtype=np.float32)
    x_disc = np.empty((n_cases,), dtype=np.float32)
    t_final_all = np.empty((n_cases,), dtype=np.float32)
    left_states_exact = np.empty((n_cases, 3), dtype=np.float64)
    right_states_exact = np.empty((n_cases, 3), dtype=np.float64)
    domains_exact = np.empty((n_cases, 2), dtype=np.float64)
    x_disc_exact = np.empty((n_cases,), dtype=np.float64)
    t_final_exact = np.empty((n_cases,), dtype=np.float64)
    fallback_counts = np.empty((n_cases,), dtype=np.int32)
    face_flux_integral_all = (
        np.empty((n_cases, n_steps, nx + 1, 3), dtype=np.float32)
        if save_face_flux_integral
        else None
    )
    face_flux_closure_max_abs = (
        np.empty((n_cases,), dtype=np.float64) if save_face_flux_integral else None
    )

    integrate = partial(
        integrate_case,
        nx=nx,
        n_steps=n_steps,
        gamma=gamma,
        cfl=cfl,
        ng=ng,
        rho_floor=rho_floor,
        p_floor=p_floor,
        use_shock_flattening=use_shock_flattening,
        use_hlle_on_troubled_faces=use_hlle_on_troubled_faces,
        shock_sensor_threshold=shock_sensor_threshold,
        shock_flatten_radius=shock_flatten_radius,
        return_face_flux_integral=save_face_flux_integral,
        initialization_mode=initialization_mode,
        storage_dtype=data_dtype.name,
    )
    if num_workers == 1:
        case_results = map(integrate, cases)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        case_results = executor.map(integrate, cases, chunksize=1)

    try:
        for k, (case, case_result) in enumerate(zip(cases, case_results)):
            x, t, data, n_fallback = case_result[:4]
            if (
                face_flux_integral_all is not None
                and face_flux_closure_max_abs is not None
            ):
                face_flux_integral_all[k] = case_result[4]
                face_flux_closure_max_abs[k] = case_result[5]
            data_all[k] = data
            x_all[k] = x
            t_all[k] = t
            left_states[k] = case.left_state.astype(np.float32)
            right_states[k] = case.right_state.astype(np.float32)
            domains[k] = np.array([case.x_left, case.x_right], dtype=np.float32)
            x_disc[k] = case.x_disc
            t_final_all[k] = case.t_final
            left_states_exact[k] = case.left_state
            right_states_exact[k] = case.right_state
            domains_exact[k] = (case.x_left, case.x_right)
            x_disc_exact[k] = case.x_disc
            t_final_exact[k] = case.t_final
            fallback_counts[k] = n_fallback

            if verbose and ((k + 1) % max(1, n_cases // 10) == 0 or k == 0):
                print(
                    f"generated {k + 1:4d}/{n_cases} cases | "
                    f"fallback steps in this case: {n_fallback}"
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    payload = dict(
        data=data_all,
        x=x_all,
        t=t_all,
        left_states=left_states,
        right_states=right_states,
        domains=domains,
        x_disc=x_disc,
        t_final=t_final_all,
        left_states_exact=left_states_exact,
        right_states_exact=right_states_exact,
        domains_exact=domains_exact,
        x_disc_exact=x_disc_exact,
        t_final_exact=t_final_exact,
        fallback_counts=fallback_counts,
        variable_names=np.array(["rho", "u", "p"]),
        gamma=np.array(gamma, dtype=np.float32),
        cfl=np.array(cfl, dtype=np.float32),
        use_shock_flattening=np.array(use_shock_flattening, dtype=np.bool_),
        use_hlle_on_troubled_faces=np.array(use_hlle_on_troubled_faces, dtype=np.bool_),
        shock_sensor_threshold=np.array(shock_sensor_threshold, dtype=np.float32),
        shock_flatten_radius=np.array(shock_flatten_radius, dtype=np.int32),
        save_dt=np.array(save_dt, dtype=np.float32),
        nx=np.array(nx, dtype=np.int32),
        n_cases=np.array(n_cases, dtype=np.int32),
        n_steps=np.array(n_steps, dtype=np.int32),
        n_frames=np.array(n_frames, dtype=np.int32),
        generation_workers=np.array(num_workers, dtype=np.int32),
        storage_dtype=np.array(data_dtype.name),
        method=np.array(
            "finite-volume WENO5 primitive reconstruction + HLLC flux + ADER2 predictor"
        ),
        boundary=np.array(
            "left fixed primitive inflow per case; right reflective wall"
        ),
        has_face_flux_integral=np.array(save_face_flux_integral, dtype=np.bool_),
        initialization_mode=np.array(initialization_mode),
    )
    if face_flux_integral_all is not None and face_flux_closure_max_abs is not None:
        payload.update(
            face_flux_integral=face_flux_integral_all,
            face_flux_orientation=np.array("owner-to-neighbor; outward at boundaries"),
            face_flux_integral_units=np.array("conservative flux times time"),
            face_flux_closure_max_abs=face_flux_closure_max_abs,
        )
    np.savez_compressed(out_path, **payload)
    if verbose:
        print(f"saved dataset to: {out_path}")
        print(f"data shape: {data_all.shape} = [case, time, x, variable]")
        print(f"total fallback steps: {int(np.sum(fallback_counts))}")
        if face_flux_closure_max_abs is not None:
            print(
                "max face-flux closure error before serialization: "
                f"{float(np.max(face_flux_closure_max_abs)):.3e}"
            )


def maybe_plot(npz_path: str, case_id: int = 0, time_ids=(0, 25, 50, 75, 100)) -> None:
    import os
    import tempfile

    mpl_cache_dir = os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib")
    )
    os.makedirs(mpl_cache_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    ds = np.load(npz_path, allow_pickle=True)
    data = ds["data"]
    x = ds["x"]
    n_steps = data.shape[1]
    time_ids = [min(int(i), n_steps - 1) for i in time_ids]

    for var_idx, name in enumerate(["rho", "u", "p"]):
        plt.figure(figsize=(8, 4))
        for ti in time_ids:
            plt.plot(x[case_id], data[case_id, ti, :, var_idx], label=f"t index {ti}")
        plt.xlabel("x")
        plt.ylabel(name)
        plt.title(f"case {case_id}: {name}")
        plt.legend()
        plt.tight_layout()
    plt.show()


def save_sample_gif(
    npz_path: str | Path,
    gif_path: str | Path,
    case_id: int = 0,
    fps: int = 20,
    dpi: int = 120,
    verbose: bool = True,
) -> None:
    """Save an animated GIF for one sample using primitive variables [rho, u, p]."""
    import os
    import tempfile

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    mpl_cache_dir = os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib")
    )
    os.makedirs(mpl_cache_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ds = np.load(npz_path, allow_pickle=True)
    data = ds["data"]
    x = ds["x"]
    t = ds["t"]
    variable_names = [str(name) for name in ds["variable_names"]]

    if not 0 <= case_id < data.shape[0]:
        raise ValueError(f"case_id must be in [0, {data.shape[0] - 1}], got {case_id}.")

    case_data = data[case_id]
    case_x = x[case_id]
    case_t = t[case_id]
    n_steps = case_data.shape[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    lines = []
    for var_idx, (ax, name) in enumerate(zip(axes, variable_names)):
        (line,) = ax.plot(case_x, case_data[0, :, var_idx], lw=2.0)
        y_min = float(np.min(case_data[:, :, var_idx]))
        y_max = float(np.max(case_data[:, :, var_idx]))
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(1.0, abs(y_max))
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        lines.append(line)

    axes[-1].set_xlabel("x")
    time_text = axes[0].text(
        0.02,
        0.90,
        "",
        transform=axes[0].transAxes,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.suptitle(f"1D Euler sample {case_id}: WENO5 + HLLC + ADER2")
    fig.tight_layout()

    def update(frame: int):
        for var_idx, line in enumerate(lines):
            line.set_ydata(case_data[frame, :, var_idx])
        time_text.set_text(f"frame {frame + 1}/{n_steps}, t = {case_t[frame]:.5f}")
        return (*lines, time_text)

    animation = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=1000.0 / max(fps, 1),
        blit=True,
    )
    animation.save(gif_path, writer=PillowWriter(fps=max(fps, 1)), dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"saved sample GIF to: {gif_path}")


def main(config: DatasetConfig = CONFIG) -> None:
    generate_dataset(
        out_path=config.out_path,
        n_cases=config.n_cases,
        n_steps=config.n_steps,
        nx=config.nx,
        t_final=config.t_final,
        gamma=config.gamma,
        cfl=config.cfl,
        use_shock_flattening=config.use_shock_flattening,
        use_hlle_on_troubled_faces=config.use_hlle_on_troubled_faces,
        shock_sensor_threshold=config.shock_sensor_threshold,
        shock_flatten_radius=config.shock_flatten_radius,
        seed=config.seed,
        ng=config.ng,
        rho_floor=config.rho_floor,
        p_floor=config.p_floor,
        sample_config=config.sample,
        verbose=config.verbose,
        save_face_flux_integral=config.save_face_flux_integral,
        initialization_mode=config.initialization_mode,
        num_workers=config.num_workers,
        storage_dtype=config.storage_dtype,
    )
    if config.save_gif:
        save_sample_gif(
            config.out_path,
            config.gif_path,
            case_id=config.gif_case_id,
            fps=config.gif_fps,
            dpi=config.gif_dpi,
            verbose=config.verbose,
        )
    if config.plot:
        maybe_plot(
            config.out_path,
            case_id=config.plot_case_id,
            time_ids=config.plot_time_ids,
        )


if __name__ == "__main__":
    main()
