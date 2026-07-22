"""Generate a periodic 1-D inviscid Burgers dataset with two moving shocks.

Equation
--------
    u_t + (u^2 / 2)_x = 0,   x in [0, L], periodic boundary condition.

Numerics
--------
* Finite-volume WENO5-Z reconstruction
* Exact Godunov flux for scalar Burgers
* SSP-RK3 time integration
* Reference-grid solve with conservative cell averaging when NREF > NNODES

Saved NPZ arrays
----------------
nodes : [NSAMPLES, NNODES]
    Coarse finite-volume cell centers. Every sample uses the same grid.
value : [NSAMPLES, NTIMES, NNODES, 1]
    Coarse finite-volume cell averages.
flux : [NSAMPLES, NTIMES-1, NNODES, 1]
    Time-averaged numerical flux through the RIGHT face of every coarse cell.
    For output interval n, it satisfies

        value[:, n+1, i, 0]
        = value[:, n, i, 0]
          - dt_out / dx * (flux[:, n, i, 0] - flux[:, n, i-1, 0])

    up to floating-point roundoff.

Additional arrays ``times`` and ``init_params`` are also saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# Reproducibility and output
RANDOM_SEED = 20260713
OUTPUT_FILE = Path("artifacts/burgers/burgers_two_shocks_weno5z.npz")
SAVE_COMPRESSED = True
SAVE_DTYPE = np.float64  # Change to np.float32 to reduce file size.

# Dataset dimensions
NSAMPLES = 1
NNODES = 1024  # Number of stored coarse finite-volume cells.
NTIMES = 101  # Includes t = 0 and t = T_FINAL.
T_FINAL = 0.50

# Periodic spatial domain [X_LEFT, X_RIGHT)
X_LEFT = 0.0
X_RIGHT = 1.0
LENGTH = X_RIGHT - X_LEFT

# Reference-grid resolution. Must be an integer multiple of NNODES; equality
# disables restriction for quick smoke datasets.
NREF = 1024

# Fourier initial condition:
# u_0(x) = c + A1 sin(2*pi*x/L + phi1)
#              + A2 sin(4*pi*x/L + phi2)
#              + A3 sin(6*pi*x/L + phi3)
# The k=2 mode is dominant and creates two main compression regions.
A2_MIN = 0.80
A2_MAX = 1.50
A1_REL_MAX = 0.20  # |A1| <= A1_REL_MAX * A2
A3_REL_MAX = 0.10  # |A3| <= A3_REL_MAX * A2
MEAN_SPEED_ABS_MIN = 0.20
MEAN_SPEED_ABS_MAX = 1.00

# Initial-condition screening. For smooth Burgers data,
# t_shock = -1 / min_x u_0'(x).
SHOCK_TIME_MIN = 0.05
SHOCK_TIME_MAX = 0.15
DERIVATIVE_CHECK_POINTS = 4096
SIGNIFICANT_MIN_RELATIVE = 0.40
MIN_COMPRESSION_SEPARATION = 0.15 * LENGTH
MAX_PARAMETER_ATTEMPTS = 100_000

# WENO5-Z + SSP-RK3 parameters
CFL = 0.35
MIN_INTERNAL_STEPS_PER_OUTPUT = 100
WENO_EPS = 1.0e-14
WENO_POWER = 2
PROGRESS_EVERY_OUTPUT = 10


# =============================================================================
# 2. FOURIER INITIAL-CONDITION SAMPLING
# =============================================================================

PARAMETER_NAMES = np.array(
    ["c", "A1", "A2", "A3", "phi1", "phi2", "phi3", "t_shock_est"]
)


def periodic_distance(x: float, y: float, length: float) -> float:
    """Shortest distance between x and y on a periodic interval."""
    d = abs(x - y)
    return min(d, length - d)


def evaluate_initial_derivative(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Evaluate the analytic derivative of one Fourier initial condition."""
    _, a1, a2, a3, phi1, phi2, phi3, _ = p
    xi = (x - X_LEFT) / LENGTH
    return (
        (2.0 * np.pi / LENGTH) * a1 * np.cos(2.0 * np.pi * xi + phi1)
        + (4.0 * np.pi / LENGTH) * a2 * np.cos(4.0 * np.pi * xi + phi2)
        + (6.0 * np.pi / LENGTH) * a3 * np.cos(6.0 * np.pi * xi + phi3)
    )


def candidate_has_two_compression_regions(
    derivative: np.ndarray, x_check: np.ndarray
) -> tuple[bool, float]:
    """Screen for exactly two significant negative derivative minima."""
    d_min = float(np.min(derivative))
    if not np.isfinite(d_min) or d_min >= 0.0:
        return False, np.inf

    t_shock = -1.0 / d_min
    if not (SHOCK_TIME_MIN <= t_shock <= SHOCK_TIME_MAX):
        return False, t_shock

    local_minimum = (derivative < np.roll(derivative, 1)) & (
        derivative <= np.roll(derivative, -1)
    )
    significant = local_minimum & (derivative <= SIGNIFICANT_MIN_RELATIVE * d_min)
    indices = np.flatnonzero(significant)

    if indices.size != 2:
        return False, t_shock

    separation = periodic_distance(
        float(x_check[indices[0]]), float(x_check[indices[1]]), LENGTH
    )
    if separation < MIN_COMPRESSION_SEPARATION:
        return False, t_shock

    return True, t_shock


def sample_initial_parameters(rng: np.random.Generator) -> np.ndarray:
    """Draw Fourier parameters and retain samples with two main compressions."""
    x_check = np.linspace(
        X_LEFT, X_RIGHT, DERIVATIVE_CHECK_POINTS, endpoint=False, dtype=np.float64
    )

    accepted: list[np.ndarray] = []
    attempts = 0

    while len(accepted) < NSAMPLES:
        if attempts >= MAX_PARAMETER_ATTEMPTS:
            raise RuntimeError(
                "Unable to obtain enough accepted initial conditions. "
                "Relax the screening thresholds or increase MAX_PARAMETER_ATTEMPTS."
            )
        attempts += 1

        a2 = rng.uniform(A2_MIN, A2_MAX)
        a1 = rng.uniform(-A1_REL_MAX * a2, A1_REL_MAX * a2)
        a3 = rng.uniform(-A3_REL_MAX * a2, A3_REL_MAX * a2)
        phi1, phi2, phi3 = rng.uniform(0.0, 2.0 * np.pi, size=3)

        speed_abs = rng.uniform(MEAN_SPEED_ABS_MIN, MEAN_SPEED_ABS_MAX)
        c = speed_abs if rng.random() < 0.5 else -speed_abs

        p = np.array([c, a1, a2, a3, phi1, phi2, phi3, np.nan])
        derivative = evaluate_initial_derivative(x_check, p)
        accepted_flag, t_shock = candidate_has_two_compression_regions(
            derivative, x_check
        )

        if accepted_flag:
            p[-1] = t_shock
            accepted.append(p)

    print(
        f"Accepted {NSAMPLES} initial conditions after {attempts} attempts "
        f"(acceptance rate {NSAMPLES / attempts:.1%})."
    )
    return np.stack(accepted, axis=0)


def fourier_cell_averages(
    cell_centers: np.ndarray, dx: float, parameters: np.ndarray
) -> np.ndarray:
    """Compute exact finite-volume cell averages of all Fourier samples."""
    c = parameters[:, 0:1]
    amplitudes = parameters[:, 1:4]
    phases = parameters[:, 4:7]
    modes = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    xi = (cell_centers - X_LEFT) / LENGTH
    angle = 2.0 * np.pi * modes[None, :, None] * xi[None, None, :] + phases[:, :, None]

    # np.sinc(z) = sin(pi*z)/(pi*z). For mode k, the exact cell-average
    # attenuation factor is sinc(k*dx/L).
    averaging_factor = np.sinc(modes * dx / LENGTH)
    weighted_amplitude = amplitudes * averaging_factor[None, :]

    return c + np.sum(weighted_amplitude[:, :, None] * np.sin(angle), axis=1)


# =============================================================================
# 3. WENO5-Z RECONSTRUCTION AND GODUNOV FLUX
# =============================================================================


def weno5z_interface_states(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct left/right states at every right cell interface.

    Parameters
    ----------
    u : ndarray, shape [batch, ncell]
        Finite-volume cell averages.

    Returns
    -------
    u_left, u_right : ndarray, shape [batch, ncell]
        At index i, states at the right face x_{i+1/2} of cell i.
    """
    im2 = np.roll(u, 2, axis=-1)
    im1 = np.roll(u, 1, axis=-1)
    ip1 = np.roll(u, -1, axis=-1)
    ip2 = np.roll(u, -2, axis=-1)
    ip3 = np.roll(u, -3, axis=-1)

    # Left state at i+1/2.
    q0 = (2.0 * im2 - 7.0 * im1 + 11.0 * u) / 6.0
    q1 = (-im1 + 5.0 * u + 2.0 * ip1) / 6.0
    q2 = (2.0 * u + 5.0 * ip1 - ip2) / 6.0

    b0 = (13.0 / 12.0) * (im2 - 2.0 * im1 + u) ** 2 + 0.25 * (
        im2 - 4.0 * im1 + 3.0 * u
    ) ** 2
    b1 = (13.0 / 12.0) * (im1 - 2.0 * u + ip1) ** 2 + 0.25 * (im1 - ip1) ** 2
    b2 = (13.0 / 12.0) * (u - 2.0 * ip1 + ip2) ** 2 + 0.25 * (
        3.0 * u - 4.0 * ip1 + ip2
    ) ** 2

    tau5 = np.abs(b0 - b2)
    a0 = 0.1 * (1.0 + (tau5 / (b0 + WENO_EPS)) ** WENO_POWER)
    a1 = 0.6 * (1.0 + (tau5 / (b1 + WENO_EPS)) ** WENO_POWER)
    a2 = 0.3 * (1.0 + (tau5 / (b2 + WENO_EPS)) ** WENO_POWER)
    a_sum = a0 + a1 + a2
    u_left = (a0 * q0 + a1 * q1 + a2 * q2) / a_sum

    # Right state at i+1/2, obtained by mirrored reconstruction.
    qr0 = (2.0 * ip3 - 7.0 * ip2 + 11.0 * ip1) / 6.0
    qr1 = (-ip2 + 5.0 * ip1 + 2.0 * u) / 6.0
    qr2 = (2.0 * ip1 + 5.0 * u - im1) / 6.0

    br0 = (13.0 / 12.0) * (ip3 - 2.0 * ip2 + ip1) ** 2 + 0.25 * (
        ip3 - 4.0 * ip2 + 3.0 * ip1
    ) ** 2
    br1 = (13.0 / 12.0) * (ip2 - 2.0 * ip1 + u) ** 2 + 0.25 * (ip2 - u) ** 2
    br2 = (13.0 / 12.0) * (ip1 - 2.0 * u + im1) ** 2 + 0.25 * (
        3.0 * ip1 - 4.0 * u + im1
    ) ** 2

    taur5 = np.abs(br0 - br2)
    ar0 = 0.1 * (1.0 + (taur5 / (br0 + WENO_EPS)) ** WENO_POWER)
    ar1 = 0.6 * (1.0 + (taur5 / (br1 + WENO_EPS)) ** WENO_POWER)
    ar2 = 0.3 * (1.0 + (taur5 / (br2 + WENO_EPS)) ** WENO_POWER)
    ar_sum = ar0 + ar1 + ar2
    u_right = (ar0 * qr0 + ar1 * qr1 + ar2 * qr2) / ar_sum

    return u_left, u_right


def burgers_godunov_flux(u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
    """Exact Godunov flux for f(u)=u^2/2."""
    f_left = 0.5 * u_left**2
    f_right = 0.5 * u_right**2

    rarefaction = u_left <= u_right
    rarefaction_flux = np.where(
        u_left >= 0.0,
        f_left,
        np.where(u_right <= 0.0, f_right, 0.0),
    )

    shock_speed = 0.5 * (u_left + u_right)
    shock_flux = np.where(shock_speed >= 0.0, f_left, f_right)
    return np.where(rarefaction, rarefaction_flux, shock_flux)


def numerical_flux(u: np.ndarray) -> np.ndarray:
    """Return right-interface numerical flux for every cell."""
    u_left, u_right = weno5z_interface_states(u)
    return burgers_godunov_flux(u_left, u_right)


def spatial_operator_from_flux(flux: np.ndarray, dx: float) -> np.ndarray:
    """Finite-volume residual using right-face flux indexing."""
    return -(flux - np.roll(flux, 1, axis=-1)) / dx


def ssprk3_step(u: np.ndarray, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance one SSP-RK3 step and return its conservative effective flux.

    The returned flux F_eff satisfies exactly

        u_new = u - dt/dx * (F_eff_i - F_eff_{i-1}).
    """
    f0 = numerical_flux(u)
    u1 = u + dt * spatial_operator_from_flux(f0, dx)

    f1 = numerical_flux(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * spatial_operator_from_flux(f1, dx))

    f2 = numerical_flux(u2)
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (
        u2 + dt * spatial_operator_from_flux(f2, dx)
    )

    # SSP-RK3 is equivalent to a conservative update with this weighted flux.
    effective_flux = (1.0 / 6.0) * f0 + (1.0 / 6.0) * f1 + (2.0 / 3.0) * f2
    return u_new, effective_flux


# =============================================================================
# 4. CONSERVATIVE RESTRICTION AND DATASET GENERATION
# =============================================================================


def restrict_cell_averages(u_fine: np.ndarray, ratio: int) -> np.ndarray:
    """Conservatively average fine finite-volume cells into coarse cells."""
    batch, n_fine = u_fine.shape
    if n_fine % ratio != 0:
        raise ValueError("Fine-grid size must be divisible by the restriction ratio.")
    return u_fine.reshape(batch, n_fine // ratio, ratio).mean(axis=-1)


def generate_dataset() -> dict[str, np.ndarray]:
    """Generate all samples simultaneously using a common stable time step."""
    if NREF % NNODES != 0:
        raise ValueError("NREF must be an integer multiple of NNODES.")
    if NTIMES < 2:
        raise ValueError("NTIMES must be at least 2.")
    if not (0.0 < CFL <= 1.0):
        raise ValueError("CFL must lie in (0, 1].")
    if MIN_INTERNAL_STEPS_PER_OUTPUT < 1:
        raise ValueError("MIN_INTERNAL_STEPS_PER_OUTPUT must be at least 1.")

    rng = np.random.default_rng(RANDOM_SEED)
    parameters = sample_initial_parameters(rng)

    dx_ref = LENGTH / NREF
    dx_out = LENGTH / NNODES
    ratio = NREF // NNODES

    x_ref = X_LEFT + (np.arange(NREF, dtype=np.float64) + 0.5) * dx_ref
    x_out = X_LEFT + (np.arange(NNODES, dtype=np.float64) + 0.5) * dx_out
    times = np.linspace(0.0, T_FINAL, NTIMES, dtype=np.float64)

    # Initial reference-grid finite-volume cell averages.
    u = fourier_cell_averages(x_ref, dx_ref, parameters).astype(np.float64)

    nodes = np.broadcast_to(x_out[None, :], (NSAMPLES, NNODES)).copy()
    value = np.empty((NSAMPLES, NTIMES, NNODES, 1), dtype=SAVE_DTYPE)
    flux = np.empty((NSAMPLES, NTIMES - 1, NNODES, 1), dtype=SAVE_DTYPE)

    value[:, 0, :, 0] = restrict_cell_averages(u, ratio).astype(SAVE_DTYPE)

    # Index of the fine-grid right face coinciding with each coarse right face.
    coarse_right_face_indices = (np.arange(NNODES, dtype=np.int64) + 1) * ratio - 1

    t = 0.0
    total_internal_steps = 0

    for n in range(NTIMES - 1):
        target_time = float(times[n + 1])
        interval_length = target_time - float(times[n])
        max_dt_for_min_steps = interval_length / MIN_INTERNAL_STEPS_PER_OUTPUT
        integrated_flux = np.zeros((NSAMPLES, NREF), dtype=np.float64)
        interval_internal_steps = 0

        while t < target_time - 10.0 * np.finfo(np.float64).eps:
            max_speed = float(np.max(np.abs(u)))
            if not np.isfinite(max_speed):
                raise FloatingPointError("Non-finite solution encountered.")

            if max_speed > 0.0:
                dt_cfl = CFL * dx_ref / max_speed
            else:
                dt_cfl = target_time - t

            dt = min(dt_cfl, max_dt_for_min_steps, target_time - t)
            u, effective_flux = ssprk3_step(u, dt, dx_ref)
            integrated_flux += dt * effective_flux
            t += dt
            total_internal_steps += 1
            interval_internal_steps += 1

        value[:, n + 1, :, 0] = restrict_cell_averages(u, ratio).astype(SAVE_DTYPE)

        # The time-averaged flux on each coarse right face gives an exact
        # conservative update over this complete output interval.
        coarse_interval_flux = (
            integrated_flux[:, coarse_right_face_indices] / interval_length
        )
        flux[:, n, :, 0] = coarse_interval_flux.astype(SAVE_DTYPE)

        if (n + 1) % PROGRESS_EVERY_OUTPUT == 0 or n == 0 or n + 1 == NTIMES - 1:
            print(
                f"Output {n + 1:4d}/{NTIMES - 1}: "
                f"t={target_time:.6f}, "
                f"interval steps={interval_internal_steps}, "
                f"total internal steps={total_internal_steps}"
            )

    return {
        "nodes": nodes.astype(SAVE_DTYPE),
        "value": value,
        "flux": flux,
        "times": times.astype(SAVE_DTYPE),
        "init_params": parameters.astype(SAVE_DTYPE),
        "param_names": PARAMETER_NAMES,
    }


def conservation_residual(dataset: dict[str, np.ndarray]) -> float:
    """Check the stored coarse-grid flux/value consistency relation."""
    value = dataset["value"][..., 0].astype(np.float64)
    flux = dataset["flux"][..., 0].astype(np.float64)
    times = dataset["times"].astype(np.float64)
    dx = LENGTH / NNODES

    dt = np.diff(times)[None, :, None]
    predicted = value[:, :-1, :] - (dt / dx) * (flux - np.roll(flux, 1, axis=-1))
    return float(np.max(np.abs(predicted - value[:, 1:, :])))


def save_dataset(dataset: dict[str, np.ndarray]) -> None:
    """Save arrays and a JSON configuration string to an NPZ file."""
    config = {
        "equation": "u_t + (u^2/2)_x = 0",
        "boundary": "periodic",
        "NSAMPLES": NSAMPLES,
        "NNODES": NNODES,
        "NTIMES": NTIMES,
        "T_FINAL": T_FINAL,
        "X_LEFT": X_LEFT,
        "X_RIGHT": X_RIGHT,
        "NREF": NREF,
        "CFL": CFL,
        "MIN_INTERNAL_STEPS_PER_OUTPUT": MIN_INTERNAL_STEPS_PER_OUTPUT,
        "method": "finite-volume WENO5-Z + exact Burgers Godunov + SSP-RK3",
        "flux_definition": (
            "flux[s,n,i,0] is the time-averaged numerical flux through "
            "the right face of coarse cell i over [times[n], times[n+1]]."
        ),
        "dtype": np.dtype(SAVE_DTYPE).name,
        "random_seed": RANDOM_SEED,
    }

    payload = dict(dataset)
    payload["config_json"] = np.array(json.dumps(config, indent=2))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_COMPRESSED:
        np.savez_compressed(OUTPUT_FILE, **payload)
    else:
        np.savez(OUTPUT_FILE, **payload)


def main() -> None:
    print("Generating periodic Burgers dataset...")
    print(
        f"samples={NSAMPLES}, stored grid={NNODES}, reference grid={NREF}, "
        f"times={NTIMES}, T={T_FINAL}"
    )

    dataset = generate_dataset()
    residual = conservation_residual(dataset)
    save_dataset(dataset)

    print("\nSaved dataset:", OUTPUT_FILE.resolve())
    print("nodes shape:", dataset["nodes"].shape)
    print("value shape:", dataset["value"].shape)
    print("flux shape: ", dataset["flux"].shape)
    print("times shape:", dataset["times"].shape)
    print("init_params shape:", dataset["init_params"].shape)
    print(f"Maximum stored conservation residual: {residual:.3e}")


if __name__ == "__main__":
    main()
