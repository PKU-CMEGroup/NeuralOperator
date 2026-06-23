"""Deterministic 2D Euler fixtures for schema and diagnostic smoke tests.

These helpers do not solve the Euler equations. They generate a small
CPG-style graph trajectory with positive primitive variables and a moving
shock-like pressure/density front. The fixture is useful before the real
benchmark data arrives because it exercises the same data contract and failure
diagnostics without introducing PyTorch or HDF5 dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from utility.time_dependent_no.euler2d import EulerNodeType


DegradationMode = Literal[
    "perfect",
    "lagged_shock",
    "smeared_shock",
    "boundary_leak",
    "positivity_failure",
]


@dataclass(frozen=True)
class SyntheticEuler2DConfig:
    """Configuration for the deterministic Euler diagnostic fixture."""

    nx: int = 16
    ny: int = 12
    num_steps: int = 8
    mach: float = 2.0
    shock_start: float = 0.32
    shock_speed: float = 0.035
    shock_width: float = 0.035
    gamma: float = 1.4

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def make_structured_grid_positions(config: SyntheticEuler2DConfig) -> np.ndarray:
    """Return row-major ``(N, 2)`` positions on ``[0, 1]^2``."""

    _validate_config(config)
    x = np.linspace(0.0, 1.0, config.nx, dtype=np.float64)
    y = np.linspace(0.0, 1.0, config.ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)


def make_structured_grid_edges(config: SyntheticEuler2DConfig) -> np.ndarray:
    """Return undirected-neighborhood edge pairs for the structured fixture."""

    _validate_config(config)
    edges: list[tuple[int, int]] = []
    for j in range(config.ny):
        for i in range(config.nx):
            node = _grid_index(i, j, config.nx)
            if i + 1 < config.nx:
                edges.append((node, _grid_index(i + 1, j, config.nx)))
            if j + 1 < config.ny:
                edges.append((node, _grid_index(i, j + 1, config.nx)))
    return np.asarray(edges, dtype=np.int64)


def make_structured_node_types(config: SyntheticEuler2DConfig) -> np.ndarray:
    """Return CPG-style node-type column for walls, inflow, and outflow."""

    _validate_config(config)
    node_type = np.full(
        config.nx * config.ny,
        int(EulerNodeType.NORMAL),
        dtype=np.int64,
    )
    for j in range(config.ny):
        for i in range(config.nx):
            node = _grid_index(i, j, config.nx)
            is_wall = j == 0 or j == config.ny - 1
            if is_wall:
                node_type[node] = int(EulerNodeType.WALL)
            elif i == 0:
                node_type[node] = int(EulerNodeType.INFLOW)
            elif i == config.nx - 1:
                node_type[node] = int(EulerNodeType.OUTFLOW)
    return node_type[:, None]


def make_synthetic_euler2d_primitives(
    config: SyntheticEuler2DConfig,
    *,
    shock_offset: float = 0.0,
    width_scale: float = 1.0,
) -> np.ndarray:
    """Generate positive primitive variables ``[rho, v1, v2, pres]``."""

    _validate_config(config)
    if width_scale <= 0.0:
        raise ValueError("width_scale must be positive")

    positions = make_structured_grid_positions(config)
    x = positions[:, 0][None, :]
    y = positions[:, 1][None, :]
    t = np.arange(config.num_steps, dtype=np.float64)[:, None]
    center = config.shock_start + config.shock_speed * t + shock_offset
    width = config.shock_width * width_scale
    transition = 0.5 * (1.0 + np.tanh((x - center) / width))
    phase = 2.0 * np.pi * y
    time_phase = 0.5 * np.pi * t / max(config.num_steps - 1, 1)

    rho = 1.0 + 0.75 * transition + 0.035 * np.sin(phase)
    v1 = 1.10 - 0.24 * transition + 0.020 * np.cos(phase)
    v2 = 0.030 * np.sin(2.0 * np.pi * x + time_phase)
    pressure = 1.0 + 1.15 * transition + 0.025 * np.cos(phase)
    return np.stack((rho, v1, v2, pressure), axis=-1)


def make_synthetic_cpg_trajectory(
    config: SyntheticEuler2DConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return a CPG-style trajectory mapping with static graph fields."""

    if config is None:
        config = SyntheticEuler2DConfig()
    primitive = make_synthetic_euler2d_primitives(config)
    num_nodes = config.nx * config.ny
    mach = np.full((num_nodes, 1), config.mach, dtype=np.float64)
    node_weights = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    return {
        "pos": make_structured_grid_positions(config),
        "edges": make_structured_grid_edges(config),
        "node_type": make_structured_node_types(config),
        "rho": primitive[..., 0:1],
        "v1": primitive[..., 1:2],
        "v2": primitive[..., 2:3],
        "pres": primitive[..., 3:4],
        "Mach": mach,
        "node_weights": node_weights,
    }


def make_degraded_synthetic_euler2d_prediction(
    config: SyntheticEuler2DConfig,
    mode: DegradationMode,
    *,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Return a controlled degraded prediction for diagnostic smoke tests."""

    if target is None:
        target = make_synthetic_euler2d_primitives(config)
    else:
        target = np.asarray(target, dtype=np.float64)
    if mode == "perfect":
        return target.copy()
    if mode == "lagged_shock":
        return make_synthetic_euler2d_primitives(config, shock_offset=-0.055)
    if mode == "smeared_shock":
        return make_synthetic_euler2d_primitives(config, width_scale=2.6)

    prediction = target.copy()
    if mode == "boundary_leak":
        node_type = make_structured_node_types(config)[:, 0]
        boundary = node_type != int(EulerNodeType.NORMAL)
        ramp = np.linspace(0.0, 1.0, config.num_steps, dtype=np.float64)[:, None]
        prediction[:, boundary, 0] += 0.06 * ramp
        prediction[:, boundary, 1] -= 0.04 * ramp
        prediction[:, boundary, 2] += 0.03 * ramp
        prediction[:, boundary, 3] += 0.10 * ramp
        return prediction
    if mode == "positivity_failure":
        pressure_nodes = np.argsort(target[-1, :, 3])[: max(1, config.nx // 4)]
        prediction[-1, pressure_nodes, 3] = -0.05
        prediction[-1, pressure_nodes[:1], 0] = -0.01
        return prediction
    raise ValueError(f"unknown degradation mode: {mode}")


def _validate_config(config: SyntheticEuler2DConfig) -> None:
    if config.nx < 3 or config.ny < 3:
        raise ValueError("nx and ny must both be at least 3")
    if config.num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if config.mach <= 0.0:
        raise ValueError("mach must be positive")
    if config.shock_width <= 0.0:
        raise ValueError("shock_width must be positive")
    if config.gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")


def _grid_index(i: int, j: int, nx: int) -> int:
    return j * nx + i

