#!/usr/bin/env python3
"""Generate an independent Pyro state reference for the D037 benchmark.

This adapter intentionally saves states only.  It does not expose Pyro's
accepted-substep face impulses and therefore cannot be used as the
conservative training-data source.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

SCHEMA = "shock_vortex_pyro_reference_v1"
PYRO_DISTRIBUTION = "pyro-hydro"
PYRO_VERSION = "4.5.0"
PYRO_WHEEL_SHA256 = "ff5116c797bcbbb517a8624131fe1e70caef29cce416b2997670c391fdbceda7"
PYRO_SOURCE = "https://github.com/python-hydro/pyro2"
BENCHMARK_SOURCE = "https://doi.org/10.1007/s10915-021-01743-1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=250)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--t-final", type=float, default=0.6)
    parser.add_argument("--output-dt", type=float, default=0.05)
    parser.add_argument("--cfl", type=float, default=0.4)
    return parser.parse_args(argv)


def _output_times(t_final: float, output_dt: float) -> tuple[float, ...]:
    if not np.isfinite(t_final) or not np.isfinite(output_dt):
        raise ValueError("t_final and output_dt must be finite")
    if t_final <= 0.0 or output_dt <= 0.0:
        raise ValueError("t_final and output_dt must be positive")
    final_decimal = Decimal(str(t_final))
    step_decimal = Decimal(str(output_dt))
    count = int(final_decimal // step_decimal)
    values = [Decimal(index) * step_decimal for index in range(count + 1)]
    if values[-1] != final_decimal:
        values.append(final_decimal)
    return tuple(float(value) for value in values)


def shock_vortex_primitive_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    """Evaluate the canonical published primitive initial condition."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    left = x < 0.5
    dx = x - 0.25
    dy = y - 0.5
    tau = np.sqrt(dx * dx + dy * dy) / 0.05
    envelope = np.exp(0.204 * (1.0 - tau * tau))
    thermal = 1.0 - (gamma - 1.0) * 0.3**2 * envelope**2 / (4.0 * 0.204 * gamma)
    vortex_rho = thermal ** (1.0 / (gamma - 1.0))
    vortex_pressure = vortex_rho**gamma
    upstream_u = 1.1 * np.sqrt(gamma)
    vortex_u = upstream_u + 0.3 * envelope * dy / 0.05
    vortex_v = -0.3 * envelope * dx / 0.05
    primitive = np.stack(
        (
            np.where(left, vortex_rho, 1.1691),
            np.where(left, vortex_u, 1.1133),
            np.where(left, vortex_v, 0.0),
            np.where(left, vortex_pressure, 1.245),
        ),
        axis=-1,
    )
    if not np.isfinite(primitive).all():
        raise ValueError("initial primitive state is non-finite")
    if np.any(primitive[..., 0] <= 0.0) or np.any(primitive[..., 3] <= 0.0):
        raise ValueError("initial primitive state is inadmissible")
    return primitive


def _initialize_pyro_state(my_data: Any, runtime_parameters: Any) -> None:
    gamma = float(runtime_parameters.get_param("eos.gamma"))
    grid = my_data.grid
    primitive = shock_vortex_primitive_numpy(grid.x2d, grid.y2d, gamma=gamma)
    rho, velocity_x, velocity_y, pressure = np.moveaxis(primitive, -1, 0)
    density = my_data.get_var("density")
    x_momentum = my_data.get_var("x-momentum")
    y_momentum = my_data.get_var("y-momentum")
    energy = my_data.get_var("energy")
    density[...] = rho
    x_momentum[...] = rho * velocity_x
    y_momentum[...] = rho * velocity_y
    energy[...] = pressure / (gamma - 1.0) + 0.5 * rho * (velocity_x**2 + velocity_y**2)


def _row_major_valid(array: Any) -> np.ndarray:
    """Convert Pyro's valid x-major ArrayIndexer view to row-major y,x."""

    valid = np.asarray(array.v(), dtype=np.float64)
    if valid.ndim != 2:
        raise ValueError("expected a two-dimensional Pyro valid-region array")
    return np.array(valid.T, copy=True)


def _extract_state_and_geometry(simulation: Any) -> tuple[np.ndarray, np.ndarray]:
    data = simulation.cc_data
    components = [
        _row_major_valid(data.get_var(name))
        for name in ("density", "x-momentum", "y-momentum", "energy")
    ]
    state = np.stack(components, axis=-1).reshape(-1, 4)
    x = _row_major_valid(data.grid.x2d)
    y = _row_major_valid(data.grid.y2d)
    centers = np.stack((x, y), axis=-1).reshape(-1, 2)
    return state, centers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primitive_from_conservative(state: np.ndarray, gamma: float) -> np.ndarray:
    rho = state[..., 0]
    velocity_x = state[..., 1] / rho
    velocity_y = state[..., 2] / rho
    pressure = (gamma - 1.0) * (
        state[..., 3] - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
    )
    return np.stack((rho, velocity_x, velocity_y, pressure), axis=-1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    if args.nx <= 0 or args.ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not 0.0 < args.cfl <= 0.5:
        raise ValueError("the independent-run CFL must lie in (0, 0.5]")
    times = _output_times(args.t_final, args.output_dt)
    installed_version = importlib_metadata.version(PYRO_DISTRIBUTION)
    if installed_version != PYRO_VERSION:
        raise RuntimeError(
            f"expected {PYRO_DISTRIBUTION} {PYRO_VERSION}, got {installed_version}"
        )

    from pyro.pyro_sim import Pyro

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    original_cwd = Path.cwd()
    source_path = Path(__file__).resolve()
    config = {
        "nx": args.nx,
        "ny": args.ny,
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "gamma": 1.4,
        "t_final": args.t_final,
        "output_times": list(times),
        "cfl": args.cfl,
        "solver": "compressible",
        "riemann": "HLLC",
        "limiter": 2,
        "use_flattening": 1,
        "artificial_viscosity": 0.1,
        "x_boundary": "outflow constant extrapolation",
        "y_boundary": "reflecting symmetry",
    }
    runtime_inputs = {
        "driver.tmax": args.t_final,
        "driver.max_steps": 100000,
        "driver.cfl": args.cfl,
        "driver.init_tstep_factor": 1.0,
        "driver.max_dt_change": 2.0,
        "driver.verbose": 0,
        "io.do_io": 0,
        "io.force_final_output": 0,
        "vis.dovis": 0,
        "mesh.nx": args.nx,
        "mesh.ny": args.ny,
        "mesh.xmin": 0.0,
        "mesh.xmax": 2.0,
        "mesh.ymin": 0.0,
        "mesh.ymax": 1.0,
        "mesh.xlboundary": "outflow",
        "mesh.xrboundary": "outflow",
        "mesh.ylboundary": "reflect",
        "mesh.yrboundary": "reflect",
        "eos.gamma": 1.4,
        "compressible.riemann": "HLLC",
        "compressible.use_flattening": 1,
        "compressible.limiter": 2,
        "compressible.cvisc": 0.1,
        "compressible.grav": 0.0,
        "compressible.small_dens": -1.0e200,
        "compressible.small_eint": -1.0e200,
        "sponge.do_sponge": 0,
        "particles.do_particles": 0,
    }
    try:
        os.chdir(output_dir)
        driver = Pyro("compressible")
        driver.add_problem("shock_vortex", _initialize_pyro_state)
        driver.initialize_problem("shock_vortex", inputs_dict=runtime_inputs)
        initial_state, centers = _extract_state_and_geometry(driver.sim)
        states = [initial_state]
        saved_times = [float(driver.sim.cc_data.t)]
        accepted_times: list[float] = []
        accepted_dt: list[float] = []
        started = time.perf_counter()
        for target_time in times[1:]:
            driver.sim.tmax = target_time
            while driver.sim.cc_data.t < target_time - 1.0e-14:
                before = float(driver.sim.cc_data.t)
                driver.single_step()
                after = float(driver.sim.cc_data.t)
                if not after > before:
                    raise RuntimeError("Pyro did not advance physical time")
                accepted_times.append(after)
                accepted_dt.append(after - before)
            if not np.isclose(
                driver.sim.cc_data.t, target_time, rtol=0.0, atol=2.0e-13
            ):
                raise RuntimeError("Pyro did not land on the requested saved time")
            state, current_centers = _extract_state_and_geometry(driver.sim)
            if not np.array_equal(current_centers, centers):
                raise RuntimeError("Pyro cell ordering changed during evolution")
            states.append(state)
            saved_times.append(float(driver.sim.cc_data.t))
        elapsed = time.perf_counter() - started
    finally:
        os.chdir(original_cwd)

    conservative = np.stack(states, axis=0)
    primitive = _primitive_from_conservative(conservative, gamma=1.4)
    cell_volume = np.full(args.nx * args.ny, 2.0 / (args.nx * args.ny))
    checks = {
        "all_saved_times_reached_exactly": bool(
            np.allclose(
                np.asarray(times),
                np.asarray(saved_times),
                rtol=0.0,
                atol=2.0e-13,
            )
        ),
        "raw_states_finite": bool(np.isfinite(conservative).all()),
        "raw_density_pressure_admissible": bool(
            np.isfinite(primitive).all()
            and np.all(primitive[..., 0] > 0.0)
            and np.all(primitive[..., 3] > 0.0)
        ),
        "cell_order_and_geometry_valid": bool(
            centers.shape == (args.nx * args.ny, 2)
            and np.all(cell_volume > 0.0)
            and np.isclose(np.sum(cell_volume), 2.0)
        ),
    }
    metadata = {
        "schema": SCHEMA,
        "benchmark_source": BENCHMARK_SOURCE,
        "independent_solver_source": PYRO_SOURCE,
        "pyro_distribution": PYRO_DISTRIBUTION,
        "pyro_version": installed_version,
        "pyro_official_wheel_sha256": PYRO_WHEEL_SHA256,
        "adapter_sha256": _sha256(source_path),
        "elapsed_seconds": elapsed,
        "future_reference_boundary_values": False,
        "clipping_or_positive_floors": False,
        "provides_reference_face_impulses": False,
    }
    artifact_path = output_dir / "reference.npz"
    np.savez_compressed(
        artifact_path,
        schema=np.asarray(SCHEMA),
        conservative_states=conservative,
        physical_times=np.asarray(saved_times),
        accepted_step_times=np.asarray(accepted_times),
        accepted_step_dt=np.asarray(accepted_dt),
        cell_centers=centers,
        cell_volume=cell_volume,
        coordinate_convention=np.asarray(
            "row-major cell averages; x increases right, y increases up"
        ),
        state_convention=np.asarray("[rho,rho*u,rho*v,total_energy]"),
        boundary_mode=np.asarray(
            "Pyro outflow constant extrapolation in x; reflecting symmetry in y"
        ),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        **metadata,
        "status": "passed" if all(checks.values()) else "failed",
        "contract_checks": checks,
        "reference_artifact": artifact_path.name,
        "reference_artifact_sha256": _sha256(artifact_path),
        "config": config,
        "arrays": {
            "conservative_states": list(conservative.shape),
            "cell_centers": list(centers.shape),
        },
        "accepted_steps": len(accepted_times),
        "minimum_accepted_dt": float(np.min(accepted_dt)),
        "maximum_accepted_dt": float(np.max(accepted_dt)),
        "minimum_density": float(np.min(primitive[..., 0])),
        "minimum_pressure": float(np.min(primitive[..., 3])),
        "claim_boundary": {
            "verified_if_passed": "one independent public-solver state trajectory under the declared Pyro contract",
            "unsupported": "reference-flux truth, conservative training labels, or benchmark closure without agreement against the converged primary reference",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError("independent Pyro artifact failed its contract checks")


if __name__ == "__main__":
    main()
