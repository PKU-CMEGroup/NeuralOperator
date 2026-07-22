"""Conservative remapping and timed coarse-CFD rollout for shock--vortex."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    conservative_to_primitive,
    ssprk3_step,
    stable_time_step,
    state_is_admissible,
)


@dataclass(frozen=True)
class CoarseCFDRollout:
    """Native-grid states and solver-facing provenance from one coarse rollout."""

    config: ShockVortexFVConfig
    states: np.ndarray
    interval_boundary_exchange: np.ndarray
    core_seconds: float
    accepted_steps: int
    rejected_attempts: int
    face_reconstruction_fallbacks: int
    minimum_density: float
    minimum_pressure: float


def _validate_nested_grids(
    target_nx: int,
    target_ny: int,
    coarse_nx: int,
    coarse_ny: int,
) -> tuple[int, int]:
    values = (target_nx, target_ny, coarse_nx, coarse_ny)
    if any(
        isinstance(value, bool) or int(value) != value or value < 1 for value in values
    ):
        raise ValueError("grid dimensions must be positive integers")
    if target_nx % coarse_nx or target_ny % coarse_ny:
        raise ValueError("coarse grid must divide the target grid exactly")
    return target_nx // coarse_nx, target_ny // coarse_ny


def restrict_uniform_cell_averages(
    state: np.ndarray,
    *,
    target_nx: int,
    target_ny: int,
    coarse_nx: int,
    coarse_ny: int,
) -> np.ndarray:
    """Restrict flattened uniform-grid cell averages by exact block averaging."""

    ratio_x, ratio_y = _validate_nested_grids(
        target_nx, target_ny, coarse_nx, coarse_ny
    )
    values = np.asarray(state)
    if values.ndim < 2 or values.shape[-2] != target_nx * target_ny:
        raise ValueError("state does not have the declared flattened target grid")
    components = values.shape[-1]
    prefix = values.shape[:-2]
    blocked = values.reshape(
        *prefix,
        coarse_ny,
        ratio_y,
        coarse_nx,
        ratio_x,
        components,
    )
    restricted = blocked.mean(axis=(-4, -2))
    return restricted.reshape(*prefix, coarse_nx * coarse_ny, components)


def prolong_piecewise_constant(
    state: np.ndarray,
    *,
    target_nx: int,
    target_ny: int,
    coarse_nx: int,
    coarse_ny: int,
) -> np.ndarray:
    """Conservatively prolong coarse cell averages by piecewise-constant injection."""

    ratio_x, ratio_y = _validate_nested_grids(
        target_nx, target_ny, coarse_nx, coarse_ny
    )
    values = np.asarray(state)
    if values.ndim < 2 or values.shape[-2] != coarse_nx * coarse_ny:
        raise ValueError("state does not have the declared flattened coarse grid")
    components = values.shape[-1]
    prefix = values.shape[:-2]
    grid = values.reshape(*prefix, coarse_ny, coarse_nx, components)
    prolonged = np.repeat(np.repeat(grid, ratio_y, axis=-3), ratio_x, axis=-2)
    return prolonged.reshape(*prefix, target_nx * target_ny, components)


def remapping_oracle(
    target: np.ndarray,
    *,
    target_nx: int,
    target_ny: int,
    coarse_nx: int,
    coarse_ny: int,
) -> np.ndarray:
    """Return restrict-then-prolong truth for the remapping information floor."""

    restricted = restrict_uniform_cell_averages(
        target,
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=coarse_nx,
        coarse_ny=coarse_ny,
    )
    return prolong_piecewise_constant(
        restricted,
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=coarse_nx,
        coarse_ny=coarse_ny,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _boundary_exchange(
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
    dt: float,
    config: ShockVortexFVConfig,
) -> torch.Tensor:
    """Sum one accepted SSPRK face flux over the outward physical boundary."""

    exchange = float(dt) * (
        config.dy * (torch.sum(flux_x[:, -1], dim=0) - torch.sum(flux_x[:, 0], dim=0))
        + config.dx * (torch.sum(flux_y[-1], dim=0) - torch.sum(flux_y[0], dim=0))
    )
    return exchange.to(dtype=torch.float64)


def run_coarse_cfd_rollout(
    initial_state: np.ndarray,
    config: ShockVortexFVConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CoarseCFDRollout:
    """Advance one native coarse grid and time only evolution plus accounting.

    The timer starts after the initial state is on the requested device and ends
    after the last accepted update is synchronized.  It includes CFL reductions,
    WENO5--HLLC--SSPRK3 evolution, admissibility retries, output-state retention,
    and the solver's own boundary-exchange sums.  It excludes initialization,
    host transfer, remapping, metrics, and artifact I/O.
    """

    config.validated()
    if config.nx != config.coarse_nx or config.ny != config.coarse_ny:
        raise ValueError("coarse timing requires a native, unrestricted solver grid")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("coarse CFD dtype must be float32 or float64")
    resolved_device = torch.device(device)
    initial = np.asarray(initial_state)
    expected = (config.nx * config.ny, 4)
    if initial.shape != expected or not np.all(np.isfinite(initial)):
        raise ValueError("initial state has an invalid shape or nonfinite values")
    conservative = torch.as_tensor(
        initial.reshape(config.ny, config.nx, 4),
        dtype=dtype,
        device=resolved_device,
    ).clone()
    if not state_is_admissible(conservative, config):
        raise ValueError("coarse initial state is not admissible")

    states = [conservative.clone()]
    interval_exchanges: list[torch.Tensor] = []
    accepted_steps = 0
    rejected_attempts = 0
    face_fallbacks = 0
    current_time = 0.0
    _synchronize(resolved_device)
    started = perf_counter()
    for target_time in config.output_times[1:]:
        interval_exchange = torch.zeros(4, dtype=torch.float64, device=resolved_device)
        while current_time < target_time - 1.0e-14:
            trial_dt = min(
                stable_time_step(conservative, config),
                float(target_time - current_time),
            )
            attempt_fallbacks = 0
            for _ in range(config.max_step_retries):
                candidate, flux_x, flux_y, fallback_count, accepted = ssprk3_step(
                    conservative, trial_dt, config
                )
                attempt_fallbacks += fallback_count
                if accepted:
                    break
                rejected_attempts += 1
                trial_dt *= 0.5
            else:
                raise RuntimeError(
                    "coarse CFD step remained inadmissible after "
                    f"{config.max_step_retries} retries at t={current_time:.16g}"
                )
            if flux_x is None or flux_y is None:
                raise AssertionError("accepted coarse CFD step has no face flux")
            interval_exchange += _boundary_exchange(flux_x, flux_y, trial_dt, config)
            conservative = candidate
            current_time += trial_dt
            if abs(current_time - target_time) <= 5.0e-14:
                current_time = float(target_time)
            accepted_steps += 1
            face_fallbacks += attempt_fallbacks
        states.append(conservative.clone())
        interval_exchanges.append(interval_exchange)
    _synchronize(resolved_device)
    elapsed = perf_counter() - started

    stacked = torch.stack(states, dim=0)
    primitive = conservative_to_primitive(stacked, config.gamma)
    minimum_density = float(torch.min(primitive[..., 0]).item())
    minimum_pressure = float(torch.min(primitive[..., 3]).item())
    return CoarseCFDRollout(
        config=config,
        states=stacked.reshape(len(states), -1, 4).double().cpu().numpy(),
        interval_boundary_exchange=(
            torch.stack(interval_exchanges, dim=0).cpu().numpy()
        ),
        core_seconds=float(elapsed),
        accepted_steps=accepted_steps,
        rejected_attempts=rejected_attempts,
        face_reconstruction_fallbacks=face_fallbacks,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )
