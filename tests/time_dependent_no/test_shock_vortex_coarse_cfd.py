from __future__ import annotations

import numpy as np
import pytest
import torch

from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    prolong_piecewise_constant,
    remapping_oracle,
    restrict_uniform_cell_averages,
    run_coarse_cfd_rollout,
)
from utility.time_dependent_no.shock_vortex_fv import ShockVortexFVConfig


def test_nested_restriction_and_prolongation_preserve_uniform_cell_integrals() -> None:
    rng = np.random.default_rng(20260722)
    target = rng.normal(size=(3, 8 * 12, 4))
    restricted = restrict_uniform_cell_averages(
        target,
        target_nx=12,
        target_ny=8,
        coarse_nx=6,
        coarse_ny=4,
    )
    prolonged = prolong_piecewise_constant(
        restricted,
        target_nx=12,
        target_ny=8,
        coarse_nx=6,
        coarse_ny=4,
    )
    assert restricted.shape == (3, 24, 4)
    assert prolonged.shape == target.shape
    np.testing.assert_allclose(prolonged.mean(axis=1), target.mean(axis=1))
    np.testing.assert_allclose(
        remapping_oracle(
            prolonged,
            target_nx=12,
            target_ny=8,
            coarse_nx=6,
            coarse_ny=4,
        ),
        prolonged,
    )


def test_nested_remapping_rejects_nondivisible_or_mislabeled_grids() -> None:
    values = np.ones((8 * 12, 4))
    with pytest.raises(ValueError, match="divide"):
        restrict_uniform_cell_averages(
            values,
            target_nx=12,
            target_ny=8,
            coarse_nx=5,
            coarse_ny=4,
        )
    with pytest.raises(ValueError, match="flattened coarse grid"):
        prolong_piecewise_constant(
            values,
            target_nx=12,
            target_ny=8,
            coarse_nx=6,
            coarse_ny=4,
        )


def test_coarse_cfd_constant_state_is_admissible_and_closes_own_boundary() -> None:
    gamma = 1.4
    rho = 1.0
    velocity = 0.2
    pressure = 1.0
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
    initial = np.tile(
        np.asarray([rho, rho * velocity, 0.0, energy], dtype=np.float64),
        (8 * 6, 1),
    )
    config = ShockVortexFVConfig(
        nx=8,
        ny=6,
        coarse_nx=8,
        coarse_ny=6,
        t_final=0.001,
        output_times=(0.0, 0.0005, 0.001),
    ).validated()
    result = run_coarse_cfd_rollout(initial, config, device="cpu", dtype=torch.float64)
    assert result.states.shape == (3, 48, 4)
    assert result.interval_boundary_exchange.shape == (2, 4)
    assert result.accepted_steps >= 2
    assert result.rejected_attempts == 0
    assert result.core_seconds > 0.0
    assert result.minimum_density > 0.0
    assert result.minimum_pressure > 0.0
    np.testing.assert_allclose(
        result.states,
        np.broadcast_to(initial, result.states.shape),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    cell_volume = config.dx * config.dy
    integrated_change = cell_volume * np.sum(
        result.states[-1] - result.states[0], axis=0
    )
    np.testing.assert_allclose(
        integrated_change + np.sum(result.interval_boundary_exchange, axis=0),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
