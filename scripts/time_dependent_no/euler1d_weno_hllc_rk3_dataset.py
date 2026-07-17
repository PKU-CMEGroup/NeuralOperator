"""
Generate a randomized 1D Euler dataset with
    WENO5 reconstruction + HLLC flux + SSP-RK3 time integration,
for Sod-like shock-tube initial data with
    left fixed inflow boundary and right reflective wall.

The saved dataset uses primitive variables [rho, u, p].

Usage
-----
Edit the CONFIG block below, then run:
python euler1d_weno_hllc_rk3_dataset.py

Snapshots are saved at fixed times 0, T/n_steps, ..., T. The internal solver
uses smaller CFL-limited substeps between two saved snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import numpy as np

from euler1d_weno_hllc_ader_dataset import (
    CaseConfig,
    SampleConfig,
    apply_boundary_conditions,
    conservative_to_primitive,
    enforce_boundary_face_states,
    hllc_flux_from_primitive,
    hlle_flux_from_primitive,
    initialize_case,
    positivity_fallback_faces,
    sample_case,
    set_first_order_faces,
    shock_troubled_faces,
    stable_dt,
    step_first_order_hllc,
    weno5_reconstruct_faces,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "time_dependent_no"


@dataclass(frozen=True)
class DatasetConfig:
    out_path: str | Path = ARTIFACT_DIR / "euler1d_weno_hllc_rk3_dataset.npz"
    n_cases: int = 1
    n_steps: int = 100
    nx: int = 2048
    t_final: float = 0.5
    gamma: float = 1.4
    cfl: float = 0.2
    use_shock_flattening: bool = True
    use_hlle_on_troubled_faces: bool = True
    shock_sensor_threshold: float = 0.08
    shock_flatten_radius: int = 3
    seed: int = 24011
    ng: int = 3
    rho_floor: float = 1e-12
    p_floor: float = 1e-12
    sample: SampleConfig = field(default_factory=SampleConfig)
    verbose: bool = True
    save_gif: bool = True
    gif_path: str | Path = ARTIFACT_DIR / "euler1d_weno_hllc_rk3_sample0.gif"
    gif_case_id: int = 0
    gif_fps: int = 20
    gif_dpi: int = 120
    plot: bool = False
    plot_case_id: int = 0
    plot_time_ids: Tuple[int, ...] = (0, 25, 50, 75, 100)


# Edit this block to set the dataset-generation parameters.
CONFIG = DatasetConfig(
    out_path=ARTIFACT_DIR / "euler1d_weno_hllc_rk3_dataset.npz",
    n_cases=1,
    n_steps=100,
    nx=2048,
    t_final=0.5,
    gamma=1.4,
    cfl=0.2,
    use_shock_flattening=True,
    use_hlle_on_troubled_faces=False,
    shock_sensor_threshold=0.08,
    shock_flatten_radius=3,
    seed=24011,
    ng=3,
    rho_floor=1e-12,
    p_floor=1e-12,
    sample=SampleConfig(
        domain_length_range=(1.0, 1.0),
        domain_left_range=(0.0, 0.0),
        discontinuity_fraction_range=(0.15, 0.35),
        left_rho_range=(1.0, 1.20),
        left_u_range=(0.60, 1.50),
        left_p_range=(0.90, 2.00),
        right_rho_range=(0.1, 0.2),
        right_p_range=(0.05, 0.1),
    ),
    verbose=True,
    save_gif=True,
    gif_path=ARTIFACT_DIR / "euler1d_weno_hllc_rk3_sample0.gif",
    gif_case_id=0,
    gif_fps=20,
    gif_dpi=120,
    plot=False,
    plot_case_id=0,
    plot_time_ids=(0, 25, 50, 75, 100),
)


# -----------------------------------------------------------------------------
# Semi-discrete WENO5 + HLLC operator and SSP-RK3 update
# -----------------------------------------------------------------------------

def physical_state_ok(
    U: np.ndarray,
    gamma: float,
    ng: int,
    rho_floor: float,
    p_floor: float,
) -> bool:
    """Check positivity and finiteness over physical cells."""
    nx = U.shape[0] - 2 * ng
    U_phys = U[ng:ng + nx]
    V_phys = conservative_to_primitive(U_phys, gamma, rho_floor=-np.inf, p_floor=-np.inf)
    return bool(
        np.isfinite(U_phys).all()
        and np.all(V_phys[:, 0] > rho_floor)
        and np.all(V_phys[:, 2] > p_floor)
    )


def compute_weno_hllc_rhs(
    U: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    gamma: float,
    ng: int,
    rho_floor: float,
    p_floor: float,
    use_shock_flattening: bool,
    use_hlle_on_troubled_faces: bool,
    shock_sensor_threshold: float,
    shock_flatten_radius: int,
) -> np.ndarray:
    """
    Compute dU/dt = -dF/dx for the semi-discrete finite-volume method.

    Reconstruction is WENO5 in primitive variables. The main numerical flux is
    HLLC, with optional local first-order flattening and HLLE flux near strong
    pressure jumps.
    """
    U_work = U.copy()
    apply_boundary_conditions(U_work, left_state, gamma, ng)

    nx = U_work.shape[0] - 2 * ng
    j_faces = np.arange(ng - 1, ng + nx, dtype=np.int64)
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
    F = hllc_flux_from_primitive(VL, VR, gamma, rho_floor, p_floor)

    hlle_faces = troubled_faces.copy()
    hlle_faces[0] = False
    hlle_faces[-1] = False
    if use_shock_flattening and use_hlle_on_troubled_faces and np.any(hlle_faces):
        F_hlle = hlle_flux_from_primitive(VL, VR, gamma, rho_floor, p_floor)
        F[hlle_faces] = F_hlle[hlle_faces]

    rhs = np.zeros_like(U)
    rhs[ng:ng + nx] = -(F[1:] - F[:-1]) / dx
    return rhs


def step_weno_hllc_rk3(
    U: np.ndarray,
    left_state: np.ndarray,
    dx: float,
    dt: float,
    gamma: float,
    ng: int,
    rho_floor: float,
    p_floor: float,
    use_shock_flattening: bool,
    use_hlle_on_troubled_faces: bool,
    shock_sensor_threshold: float,
    shock_flatten_radius: int,
) -> Tuple[np.ndarray, bool]:
    """One SSP-RK3 update for the WENO5 + HLLC semi-discrete operator."""
    rhs0 = compute_weno_hllc_rhs(
        U,
        left_state,
        dx,
        gamma,
        ng,
        rho_floor,
        p_floor,
        use_shock_flattening,
        use_hlle_on_troubled_faces,
        shock_sensor_threshold,
        shock_flatten_radius,
    )
    U1 = U + dt * rhs0
    apply_boundary_conditions(U1, left_state, gamma, ng)
    if not physical_state_ok(U1, gamma, ng, rho_floor, p_floor):
        return U1, False

    rhs1 = compute_weno_hllc_rhs(
        U1,
        left_state,
        dx,
        gamma,
        ng,
        rho_floor,
        p_floor,
        use_shock_flattening,
        use_hlle_on_troubled_faces,
        shock_sensor_threshold,
        shock_flatten_radius,
    )
    U2 = 0.75 * U + 0.25 * (U1 + dt * rhs1)
    apply_boundary_conditions(U2, left_state, gamma, ng)
    if not physical_state_ok(U2, gamma, ng, rho_floor, p_floor):
        return U2, False

    rhs2 = compute_weno_hllc_rhs(
        U2,
        left_state,
        dx,
        gamma,
        ng,
        rho_floor,
        p_floor,
        use_shock_flattening,
        use_hlle_on_troubled_faces,
        shock_sensor_threshold,
        shock_flatten_radius,
    )
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * rhs2)
    apply_boundary_conditions(U_new, left_state, gamma, ng)
    return U_new, physical_state_ok(U_new, gamma, ng, rho_floor, p_floor)


# -----------------------------------------------------------------------------
# Time integration and dataset output
# -----------------------------------------------------------------------------

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Integrate one case and return primitive snapshots.

    Returns
    -------
    x: [nx]
    t: [n_steps + 1]
    data: [n_steps + 1, nx, 3] primitive snapshots, including t=0 and t=T
    n_fallback: number of first-order fallback steps used
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")

    x, U, dx = initialize_case(case, nx, gamma, ng)
    targets = np.linspace(0.0, case.t_final, n_steps + 1)
    data = np.empty((n_steps + 1, nx, 3), dtype=np.float32)
    data[0] = conservative_to_primitive(U[ng:ng + nx], gamma).astype(np.float32)

    t_now = 0.0
    fallback_count = 0

    for out_idx in range(1, n_steps + 1):
        target = float(targets[out_idx])
        while t_now < target - 1e-14:
            dt = min(stable_dt(U, dx, cfl, gamma, ng), target - t_now)

            success = False
            dt_try = dt
            for _ in range(12):
                U_candidate, ok = step_weno_hllc_rk3(
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
                )
                if ok:
                    U = U_candidate
                    t_now += dt_try
                    success = True
                    break
                dt_try *= 0.5

            if not success:
                dt_try = min(dt * 0.25, target - t_now)
                U_candidate, ok = step_first_order_hllc(
                    U,
                    case.left_state,
                    dx,
                    dt_try,
                    gamma,
                    ng,
                    rho_floor,
                    p_floor,
                )
                if not ok:
                    raise RuntimeError(
                        "Positivity failure even after first-order fallback. "
                        "Try lowering cfl or reducing the sampled pressure/velocity ranges."
                    )
                U = U_candidate
                t_now += dt_try
                fallback_count += 1

        data[out_idx] = conservative_to_primitive(U[ng:ng + nx], gamma).astype(np.float32)

    return x.astype(np.float32), targets.astype(np.float32), data, fallback_count


def generate_dataset(
    out_path: str | Path,
    n_cases: int = 1,
    n_steps: int = 100,
    nx: int = 2048,
    t_final: float = 0.5,
    gamma: float = 1.4,
    cfl: float = 0.2,
    use_shock_flattening: bool = True,
    use_hlle_on_troubled_faces: bool = True,
    shock_sensor_threshold: float = 0.05,
    shock_flatten_radius: int = 4,
    seed: int = 2401,
    ng: int = 3,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
    sample_config: SampleConfig = SampleConfig(),
    verbose: bool = True,
) -> None:
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    if t_final <= 0.0:
        raise ValueError(f"t_final must be positive, got {t_final}.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_frames = n_steps + 1
    save_dt = t_final / n_steps

    data_all = np.empty((n_cases, n_frames, nx, 3), dtype=np.float32)
    x_all = np.empty((n_cases, nx), dtype=np.float32)
    t_all = np.empty((n_cases, n_frames), dtype=np.float32)
    left_states = np.empty((n_cases, 3), dtype=np.float32)
    right_states = np.empty((n_cases, 3), dtype=np.float32)
    domains = np.empty((n_cases, 2), dtype=np.float32)
    x_disc = np.empty((n_cases,), dtype=np.float32)
    t_final_all = np.empty((n_cases,), dtype=np.float32)
    fallback_counts = np.empty((n_cases,), dtype=np.int32)

    for k in range(n_cases):
        case = sample_case(rng, sample_config, t_final)
        x, t, data, n_fallback = integrate_case(
            case,
            nx,
            n_steps,
            gamma,
            cfl,
            ng,
            rho_floor,
            p_floor,
            use_shock_flattening,
            use_hlle_on_troubled_faces,
            shock_sensor_threshold,
            shock_flatten_radius,
        )
        data_all[k] = data
        x_all[k] = x
        t_all[k] = t
        left_states[k] = case.left_state.astype(np.float32)
        right_states[k] = case.right_state.astype(np.float32)
        domains[k] = np.array([case.x_left, case.x_right], dtype=np.float32)
        x_disc[k] = case.x_disc
        t_final_all[k] = case.t_final
        fallback_counts[k] = n_fallback

        if verbose and ((k + 1) % max(1, n_cases // 10) == 0 or k == 0):
            print(f"generated {k + 1:4d}/{n_cases} cases | fallback steps in this case: {n_fallback}")

    np.savez_compressed(
        out_path,
        data=data_all,
        x=x_all,
        t=t_all,
        left_states=left_states,
        right_states=right_states,
        domains=domains,
        x_disc=x_disc,
        t_final=t_final_all,
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
        method=np.array("finite-volume WENO5 primitive reconstruction + HLLC flux + SSP-RK3"),
        boundary=np.array("left fixed primitive inflow per case; right reflective wall"),
    )
    if verbose:
        print(f"saved dataset to: {out_path}")
        print(f"data shape: {data_all.shape} = [case, time, x, variable]")
        print(f"total fallback steps: {int(np.sum(fallback_counts))}")


def maybe_plot(npz_path: str, case_id: int = 0, time_ids=(0, 25, 50, 75, 100)) -> None:
    import os
    import tempfile

    mpl_cache_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
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

    mpl_cache_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
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
    fig.suptitle(f"1D Euler sample {case_id}: WENO5 + HLLC + RK3")
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
