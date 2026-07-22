import numpy as np

from scripts.burgers.generate_burgers_two_shocks import (
    restrict_cell_averages,
    ssprk3_step,
)


def test_ssprk3_preserves_constant_state_and_flux_closure():
    state = np.full((2, 16), 0.75, dtype=np.float64)
    dt = 0.01
    dx = 1.0 / state.shape[-1]

    updated, effective_flux = ssprk3_step(state, dt, dx)

    np.testing.assert_allclose(updated, state, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(effective_flux, 0.5 * state**2)
    reconstructed = state - (dt / dx) * (
        effective_flux - np.roll(effective_flux, 1, axis=-1)
    )
    np.testing.assert_allclose(updated, reconstructed, rtol=0.0, atol=1.0e-14)


def test_restrict_cell_averages_preserves_batch_means():
    fine = np.arange(32, dtype=np.float64).reshape(2, 16)

    coarse = restrict_cell_averages(fine, ratio=4)

    assert coarse.shape == (2, 4)
    np.testing.assert_allclose(coarse.mean(axis=-1), fine.mean(axis=-1))
