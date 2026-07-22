#!/usr/bin/env python3
"""Generate the frozen SharpClaw state comparison for the D037 benchmark.

This adapter deliberately exports cell states only.  PyClaw does not expose
the accepted-substep, oriented face impulses required by the finite-volume
training-data contract, so this artifact is an independent state-agreement
check and cannot serve as reference-flux truth.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "shock_vortex_sharpclaw_reference_v2"
CLAWPACK_VERSION = "5.9.0"
CLAWPACK_BUILD = "py311h3d4ca6a_1"
CLAWPACK_SOURCE = "https://github.com/clawpack/clawpack/releases/tag/v5.9.0"
BENCHMARK_SOURCE = "https://doi.org/10.1007/s10915-021-01743-1"

GAMMA = 1.4
X_MIN = 0.0
X_MAX = 2.0
Y_MIN = 0.0
Y_MAX = 1.0
SHOCK_X = 0.5
VORTEX_X = 0.25
VORTEX_Y = 0.5
VORTEX_EPSILON = 0.3
VORTEX_ALPHA = 0.204
VORTEX_RADIUS = 0.05
RIGHT_RHO = 1.1691
RIGHT_U = 1.1133
RIGHT_PRESSURE = 1.245

INITIAL_QUADRATURE_ORDER = 8
INITIAL_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-10
CFL_DESIRED = 0.35
CFL_MAX = 0.5
SHARPCLAW_NUM_GHOST = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=250)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--t-final", type=float, default=0.6)
    parser.add_argument("--output-dt", type=float, default=0.05)
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


def _validate_arguments(args: argparse.Namespace) -> tuple[float, ...]:
    if args.nx < 6 or args.ny < 6:
        raise ValueError("nx and ny must both be at least six for WENO5")
    return _output_times(float(args.t_final), float(args.output_dt))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned_clawpack(prefix: Path) -> dict[str, str]:
    """Validate the exact conda package record, including its build string."""

    conda_meta = prefix / "conda-meta"
    records: list[tuple[Path, dict[str, Any]]] = []
    if conda_meta.is_dir():
        for path in sorted(conda_meta.glob("clawpack-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("name") == "clawpack":
                records.append((path, payload))
    if len(records) != 1:
        raise RuntimeError(
            "expected exactly one active clawpack conda record under "
            f"{conda_meta}, found {len(records)}"
        )
    path, payload = records[0]
    version = str(payload.get("version", ""))
    build = str(payload.get("build", ""))
    if version != CLAWPACK_VERSION or build != CLAWPACK_BUILD:
        raise RuntimeError(
            "expected clawpack "
            f"{CLAWPACK_VERSION}/{CLAWPACK_BUILD}, got {version}/{build}"
        )
    return {
        "version": version,
        "build": build,
        "record_filename": path.name,
        "record_sha256": _sha256(path),
    }


def shock_vortex_primitive_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Evaluate the canonical published primitive initial condition."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    left = x < SHOCK_X
    dx = x - VORTEX_X
    dy = y - VORTEX_Y
    tau = np.sqrt(dx * dx + dy * dy) / VORTEX_RADIUS
    envelope = np.exp(VORTEX_ALPHA * (1.0 - tau * tau))
    thermal = 1.0 - (
        (gamma - 1.0) * VORTEX_EPSILON**2 * envelope**2 / (4.0 * VORTEX_ALPHA * gamma)
    )
    vortex_rho = thermal ** (1.0 / (gamma - 1.0))
    vortex_pressure = vortex_rho**gamma
    upstream_u = 1.1 * np.sqrt(gamma)
    vortex_u = upstream_u + VORTEX_EPSILON * envelope * dy / VORTEX_RADIUS
    vortex_v = -VORTEX_EPSILON * envelope * dx / VORTEX_RADIUS
    primitive = np.stack(
        (
            np.where(left, vortex_rho, RIGHT_RHO),
            np.where(left, vortex_u, RIGHT_U),
            np.where(left, vortex_v, 0.0),
            np.where(left, vortex_pressure, RIGHT_PRESSURE),
        ),
        axis=-1,
    )
    if not np.isfinite(primitive).all():
        raise ValueError("initial primitive state is non-finite")
    if np.any(primitive[..., 0] <= 0.0) or np.any(primitive[..., 3] <= 0.0):
        raise ValueError("initial primitive state is inadmissible")
    return primitive


def _primitive_to_conservative(primitive: np.ndarray, gamma: float) -> np.ndarray:
    rho, velocity_x, velocity_y, pressure = np.moveaxis(primitive, -1, 0)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return np.stack((rho, rho * velocity_x, rho * velocity_y, energy), axis=-1)


def _validate_pyclaw_x_boundary_call(
    state: Any,
    dimension: Any,
    time_value: float,
    qbc: np.ndarray,
    num_ghost: int,
) -> tuple[int, int]:
    """Fail closed unless a callback has the expected serial PyClaw 5.9 layout."""

    grid = getattr(state, "grid", None)
    dimensions = getattr(grid, "dimensions", None)
    num_cells = getattr(grid, "num_cells", None)
    if dimensions is None or len(dimensions) != 2 or dimension is not dimensions[0]:
        raise RuntimeError("linear extrapolation callback is valid only for x")
    if getattr(dimensions[0], "name", None) != "x":
        raise RuntimeError("first PyClaw dimension must be named x")
    if num_cells is None or len(num_cells) != 2:
        raise RuntimeError("expected a two-dimensional PyClaw grid")
    nx, ny = (int(value) for value in num_cells)
    if nx < 2 or ny < 1:
        raise RuntimeError("linear x extrapolation requires at least two x cells")
    if num_ghost != SHARPCLAW_NUM_GHOST:
        raise RuntimeError(
            f"expected {SHARPCLAW_NUM_GHOST} SharpClaw ghost cells, got {num_ghost}"
        )
    if np.asarray(qbc).shape != (
        4,
        nx + 2 * num_ghost,
        ny + 2 * num_ghost,
    ):
        raise RuntimeError("unexpected PyClaw qbc component or ghost-axis layout")
    if not np.issubdtype(np.asarray(qbc).dtype, np.floating):
        raise RuntimeError("PyClaw qbc must have a floating dtype")
    if not np.isfinite(time_value):
        raise RuntimeError("PyClaw boundary callback time must be finite")
    gamma = float(getattr(state, "problem_data", {}).get("gamma", np.nan))
    if not np.isclose(gamma, GAMMA, rtol=0.0, atol=0.0):
        raise RuntimeError(f"expected callback gamma {GAMMA}, got {gamma}")
    return nx, ny


def _fill_linear_primitive_x_boundary(
    state: Any,
    dimension: Any,
    time_value: float,
    qbc: np.ndarray,
    auxbc: np.ndarray | None,
    num_ghost: int,
    *,
    side: str,
) -> None:
    """Match the primary solver's linear primitive-variable x extrapolation."""

    del auxbc
    nx, ny = _validate_pyclaw_x_boundary_call(
        state, dimension, time_value, qbc, num_ghost
    )
    interior_y = slice(num_ghost, num_ghost + ny)
    if side == "lower":
        first_index = num_ghost
        second_index = num_ghost + 1
        target_indices = [num_ghost - offset for offset in range(1, num_ghost + 1)]
    elif side == "upper":
        first_index = num_ghost + nx - 1
        second_index = num_ghost + nx - 2
        target_indices = [
            num_ghost + nx - 1 + offset for offset in range(1, num_ghost + 1)
        ]
    else:
        raise ValueError(f"unknown x boundary side: {side}")

    first_conservative = np.moveaxis(qbc[:, first_index, interior_y], 0, -1)
    second_conservative = np.moveaxis(qbc[:, second_index, interior_y], 0, -1)
    first_primitive = _primitive_from_conservative(first_conservative, GAMMA)
    second_primitive = _primitive_from_conservative(second_conservative, GAMMA)
    if (
        not np.isfinite(first_primitive).all()
        or not np.isfinite(second_primitive).all()
    ):
        raise RuntimeError("non-finite interior primitive state at x boundary")

    for offset, target_index in enumerate(target_indices, start=1):
        ghost_primitive = (offset + 1) * first_primitive - offset * second_primitive
        ghost_conservative = _primitive_to_conservative(ghost_primitive, GAMMA)
        qbc[:, target_index, interior_y] = np.moveaxis(ghost_conservative, -1, 0)

        # PyClaw 5.9 subsequently applies the y-wall BC. Filling these corners
        # identically here makes the callback deterministic when tested alone.
        for y_offset in range(1, num_ghost + 1):
            bottom = np.array(qbc[:, target_index, num_ghost + y_offset - 1], copy=True)
            top = np.array(qbc[:, target_index, num_ghost + ny - y_offset], copy=True)
            bottom[2] *= -1.0
            top[2] *= -1.0
            qbc[:, target_index, num_ghost - y_offset] = bottom
            qbc[:, target_index, num_ghost + ny - 1 + y_offset] = top


def linear_primitive_x_lower_boundary(
    state: Any,
    dimension: Any,
    time_value: float,
    qbc: np.ndarray,
    auxbc: np.ndarray | None,
    num_ghost: int,
) -> None:
    _fill_linear_primitive_x_boundary(
        state,
        dimension,
        time_value,
        qbc,
        auxbc,
        num_ghost,
        side="lower",
    )


def linear_primitive_x_upper_boundary(
    state: Any,
    dimension: Any,
    time_value: float,
    qbc: np.ndarray,
    auxbc: np.ndarray | None,
    num_ghost: int,
) -> None:
    _fill_linear_primitive_x_boundary(
        state,
        dimension,
        time_value,
        qbc,
        auxbc,
        num_ghost,
        side="upper",
    )


def shock_vortex_initial_cell_averages_numpy(
    nx: int,
    ny: int,
    *,
    quadrature_order: int = INITIAL_QUADRATURE_ORDER,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Integrate conservative initial data over cells in row-major ``(y,x)`` order.

    Tensor Gauss--Legendre integration is split at the discontinuity ``x=0.5``.
    This mirrors the repaired primary initializer instead of center-sampling a
    shock-crossing cell.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    x_left = X_MIN + np.arange(nx, dtype=np.float64) * dx
    x_right = x_left + dx
    y_bottom = Y_MIN + np.arange(ny, dtype=np.float64) * dy
    y_mid = y_bottom + 0.5 * dy
    y_points = y_mid[:, None] + 0.5 * dy * nodes[None, :]
    y_weights = 0.5 * weights
    average = np.zeros((ny, nx, 4), dtype=np.float64)

    segments = (
        (x_left, np.minimum(x_right, SHOCK_X)),
        (np.maximum(x_left, SHOCK_X), x_right),
    )
    for segment_left, segment_right in segments:
        width = np.maximum(segment_right - segment_left, 0.0)
        midpoint = 0.5 * (segment_left + segment_right)
        x_points = midpoint[:, None] + 0.5 * width[:, None] * nodes[None, :]
        x_weights = 0.5 * width[:, None] * weights[None, :] / dx
        for x_index in range(quadrature_order):
            x = np.broadcast_to(x_points[:, x_index], (ny, nx))
            x_weight = x_weights[:, x_index][None, :, None]
            for y_index in range(quadrature_order):
                y = np.broadcast_to(y_points[:, y_index][:, None], (ny, nx))
                primitive = shock_vortex_primitive_numpy(x, y, gamma=gamma)
                conservative = _primitive_to_conservative(primitive, gamma)
                average += y_weights[y_index] * x_weight * conservative
    if not np.isfinite(average).all():
        raise ValueError("quadrature produced a non-finite initial cell average")
    return average


def _quadrature_certificate(
    candidate: np.ndarray,
    doubled_order: np.ndarray,
) -> dict[str, float]:
    if candidate.shape != doubled_order.shape:
        raise ValueError("quadrature candidates must have matching shapes")
    difference = np.asarray(candidate) - np.asarray(doubled_order)
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": float(
            np.linalg.norm(difference.ravel())
            / max(float(np.linalg.norm(np.asarray(doubled_order).ravel())), 1.0e-30)
        ),
    }


def _cell_geometry(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    x = X_MIN + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = Y_MIN + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    yy, xx = np.meshgrid(y, x, indexing="ij")
    centers = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    volume = np.full(nx * ny, dx * dy, dtype=np.float64)
    return centers, volume


def _assign_pyclaw_initial_state(state: Any, cell_average: np.ndarray) -> None:
    ny, nx, components = cell_average.shape
    expected = (components, nx, ny)
    if tuple(state.q.shape) != expected:
        raise ValueError(f"expected PyClaw q shape {expected}, got {state.q.shape}")
    state.q[...] = np.moveaxis(cell_average, -1, 0).transpose(0, 2, 1)


def _extract_row_major_state(state: Any, nx: int, ny: int) -> np.ndarray:
    conservative = np.asarray(state.q, dtype=np.float64)
    if conservative.shape != (4, nx, ny):
        raise ValueError(
            f"expected PyClaw q shape {(4, nx, ny)}, got {conservative.shape}"
        )
    row_major = np.moveaxis(conservative, 0, -1).transpose(1, 0, 2)
    return np.array(row_major.reshape(-1, 4), copy=True)


def _solution_state(solution: Any) -> Any:
    if hasattr(solution, "state"):
        return solution.state
    states = getattr(solution, "states", None)
    if states is None or len(states) != 1:
        raise ValueError("expected a single-state PyClaw solution frame")
    return states[0]


def _solution_time(solution: Any) -> float:
    if hasattr(solution, "t"):
        return float(solution.t)
    return float(_solution_state(solution).t)


def _assert_solver_contract(solver: Any) -> None:
    expected = {
        "kernel_language": "Fortran",
        "lim_type": 2,
        "weno_order": 5,
        "char_decomp": 0,
        "time_integrator": "SSP33",
        "cfl_desired": CFL_DESIRED,
        "cfl_max": CFL_MAX,
        "num_ghost": SHARPCLAW_NUM_GHOST,
    }
    for name, value in expected.items():
        actual = getattr(solver, name)
        if isinstance(value, float):
            matches = np.isclose(actual, value, rtol=0.0, atol=0.0)
        else:
            matches = actual == value
        if not matches:
            raise RuntimeError(
                f"SharpClaw contract mismatch for {name}: expected {value}, got {actual}"
            )


def _assert_boundary_registration(
    solver: Any,
    *,
    custom_boundary_value: Any,
    wall_boundary_value: Any,
) -> None:
    if list(solver.bc_lower) != [custom_boundary_value, wall_boundary_value]:
        raise RuntimeError("unexpected SharpClaw lower boundary registration")
    if list(solver.bc_upper) != [custom_boundary_value, wall_boundary_value]:
        raise RuntimeError("unexpected SharpClaw upper boundary registration")
    if solver.user_bc_lower is not linear_primitive_x_lower_boundary:
        raise RuntimeError("unexpected SharpClaw lower custom callback")
    if solver.user_bc_upper is not linear_primitive_x_upper_boundary:
        raise RuntimeError("unexpected SharpClaw upper custom callback")


def _primitive_from_conservative(state: np.ndarray, gamma: float) -> np.ndarray:
    rho = state[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity_x = state[..., 1] / rho
        velocity_y = state[..., 2] / rho
        pressure = (gamma - 1.0) * (
            state[..., 3]
            - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
        )
    return np.stack((rho, velocity_x, velocity_y, pressure), axis=-1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _artifact_arrays(
    conservative_states: np.ndarray,
    physical_times: np.ndarray,
    cell_centers: np.ndarray,
    cell_volume: np.ndarray,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(SCHEMA),
        "conservative_states": np.asarray(conservative_states, dtype=np.float64),
        "physical_times": np.asarray(physical_times, dtype=np.float64),
        "physical_delta_t": np.diff(np.asarray(physical_times, dtype=np.float64)),
        "cell_centers": np.asarray(cell_centers, dtype=np.float64),
        "cell_volume": np.asarray(cell_volume, dtype=np.float64),
        "coordinate_convention": np.asarray(
            "row-major cell averages; x increases right, y increases up"
        ),
        "state_convention": np.asarray("[rho,rho*u,rho*v,total_energy]"),
        "boundary_mode": np.asarray(
            "SharpClaw custom linear primitive-variable extrapolation in x; "
            "reflecting wall in y"
        ),
        "config_json": np.asarray(json.dumps(config, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    times = _validate_arguments(args)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    package = _assert_pinned_clawpack(Path(sys.prefix))

    # Runtime-only imports keep schema and initializer tests independent of Clawpack.
    from clawpack import pyclaw, riemann

    initial = shock_vortex_initial_cell_averages_numpy(args.nx, args.ny)
    certified_initial = shock_vortex_initial_cell_averages_numpy(
        args.nx,
        args.ny,
        quadrature_order=2 * INITIAL_QUADRATURE_ORDER,
    )
    quadrature = _quadrature_certificate(initial, certified_initial)

    solver = pyclaw.SharpClawSolver2D(riemann.euler_4wave_2D)
    solver.kernel_language = "Fortran"
    solver.lim_type = 2
    solver.weno_order = 5
    solver.char_decomp = 0
    solver.time_integrator = "SSP33"
    solver.cfl_desired = CFL_DESIRED
    solver.cfl_max = CFL_MAX
    solver.max_steps = 100000
    solver.user_bc_lower = linear_primitive_x_lower_boundary
    solver.user_bc_upper = linear_primitive_x_upper_boundary
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_upper[0] = pyclaw.BC.custom
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.wall
    _assert_solver_contract(solver)
    _assert_boundary_registration(
        solver,
        custom_boundary_value=pyclaw.BC.custom,
        wall_boundary_value=pyclaw.BC.wall,
    )

    x_dimension = pyclaw.Dimension(X_MIN, X_MAX, args.nx, name="x")
    y_dimension = pyclaw.Dimension(Y_MIN, Y_MAX, args.ny, name="y")
    domain = pyclaw.Domain([x_dimension, y_dimension])
    state = pyclaw.State(domain, 4)
    state.problem_data["gamma"] = GAMMA
    _assign_pyclaw_initial_state(state, initial)

    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.solver = solver
    controller.keep_copy = True
    controller.output_format = None
    controller.output_style = 2
    controller.out_times = np.asarray(times, dtype=np.float64)
    controller.tfinal = times[-1]

    started = time.perf_counter()
    controller_status = controller.run()
    elapsed = time.perf_counter() - started
    frames = list(controller.frames)
    saved_times = np.asarray([_solution_time(frame) for frame in frames])
    conservative = np.stack(
        [
            _extract_row_major_state(_solution_state(frame), args.nx, args.ny)
            for frame in frames
        ],
        axis=0,
    )
    centers, cell_volume = _cell_geometry(args.nx, args.ny)
    primitive = _primitive_from_conservative(conservative, GAMMA)
    initial_replay_relative = float(
        np.linalg.norm(conservative[0] - initial.reshape(-1, 4))
        / max(float(np.linalg.norm(initial.ravel())), 1.0e-30)
    )

    checks = {
        "pinned_clawpack_version_and_build": bool(
            package["version"] == CLAWPACK_VERSION
            and package["build"] == CLAWPACK_BUILD
        ),
        "solver_contract_exact": True,
        "all_saved_times_reached_exactly": bool(
            saved_times.shape == (len(times),)
            and np.allclose(
                saved_times,
                np.asarray(times),
                rtol=0.0,
                atol=2.0e-13,
            )
        ),
        "initial_cell_average_quadrature_certified": bool(
            quadrature["relative_l2"] <= INITIAL_QUADRATURE_RELATIVE_TOLERANCE
        ),
        "initial_state_loaded_without_reordering": bool(
            initial_replay_relative <= 1.0e-14
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
    source_path = Path(__file__).resolve()
    config = {
        "nx": args.nx,
        "ny": args.ny,
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "gamma": GAMMA,
        "t_final": args.t_final,
        "output_times": list(times),
        "solver": "pyclaw.SharpClawSolver2D",
        "riemann_solver": "riemann.euler_4wave_2D",
        "kernel_language": "Fortran",
        "lim_type": 2,
        "weno_order": 5,
        "char_decomp": 0,
        "time_integrator": "SSP33",
        "cfl_desired": CFL_DESIRED,
        "cfl_max": CFL_MAX,
        "num_ghost": SHARPCLAW_NUM_GHOST,
        "initial_quadrature_order": INITIAL_QUADRATURE_ORDER,
        "x_boundary": "custom linear extrapolation in primitive variables",
        "y_boundary": "reflecting wall",
    }
    metadata = {
        "schema": SCHEMA,
        "benchmark_source": BENCHMARK_SOURCE,
        "independent_solver_source": CLAWPACK_SOURCE,
        "clawpack_version": package["version"],
        "clawpack_build": package["build"],
        "clawpack_conda_record_filename": package["record_filename"],
        "clawpack_conda_record_sha256": package["record_sha256"],
        "adapter_sha256": _sha256(source_path),
        "elapsed_seconds": elapsed,
        "future_reference_boundary_values": False,
        "clipping_or_positive_floors": False,
        "provides_reference_face_impulses": False,
        "state_only_independent_comparison": True,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "reference.npz"
    np.savez_compressed(
        artifact_path,
        **_artifact_arrays(
            conservative,
            saved_times,
            centers,
            cell_volume,
            config,
            metadata,
        ),
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
        "controller_status": _json_safe(controller_status),
        "initial_quadrature_max_abs": quadrature["max_abs"],
        "initial_quadrature_relative_l2": quadrature["relative_l2"],
        "initial_replay_relative_l2": initial_replay_relative,
        "minimum_density": float(np.min(primitive[..., 0])),
        "minimum_pressure": float(np.min(primitive[..., 3])),
        "claim_boundary": {
            "verified_if_passed": "one pinned SharpClaw state trajectory under the matched linear-primitive x-boundary contract",
            "unsupported": "reference face-impulse truth, conservative training labels, or benchmark promotion before the frozen state-agreement gate passes",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError("SharpClaw artifact failed one or more contract checks")


if __name__ == "__main__":
    main()
