from __future__ import annotations

import math

import numpy as np
import pytest

from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig,
    KolmogorovReferenceStepper,
    resize_dealiased_vorticity,
)


def _random_canonical(
    stepper: KolmogorovReferenceStepper, seed: int = 20260826
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return stepper.canonicalize(rng.normal(size=stepper.state_shape))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resolution", 15, "resolution"),
        ("forcing_wavenumber", 7, "retained band"),
        ("viscosity", 0.0, "viscosity"),
        ("linear_drag", -0.1, "linear_drag"),
        ("cfl", 1.1, "cfl"),
        ("max_substeps", 0, "max_substeps"),
    ],
)
def test_config_rejects_invalid_contract_fields(
    field: str, value: float, message: str
) -> None:
    kwargs = {"resolution": 18, field: value}
    with pytest.raises(ValueError, match=message):
        KolmogorovReferenceConfig(**kwargs).validated()


def test_canonicalization_is_mean_zero_dealiased_and_idempotent() -> None:
    config = KolmogorovReferenceConfig(resolution=18)
    stepper = KolmogorovReferenceStepper(config)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=stepper.state_shape).astype(np.float32)
    canonical = stepper.canonicalize(raw)

    assert canonical.dtype == np.float64
    assert canonical.flags.c_contiguous
    assert abs(float(np.mean(canonical))) < 1.0e-14
    np.testing.assert_allclose(
        stepper.canonicalize(canonical), canonical, rtol=0.0, atol=2.0e-15
    )

    modes = np.fft.fftfreq(config.resolution) * config.resolution
    mode_x, mode_y = np.meshgrid(modes, modes, indexing="ij")
    retained = (np.abs(mode_x) <= config.resolution // 3) & (
        np.abs(mode_y) <= config.resolution // 3
    )
    spectrum = np.fft.fft2(canonical)
    assert float(np.max(np.abs(spectrum[~retained]))) < 2.0e-13


def test_taylor_green_vorticity_has_the_exact_linear_decay() -> None:
    config = KolmogorovReferenceConfig(
        resolution=24,
        viscosity=0.1,
        linear_drag=0.05,
        forcing_amplitude=0.0,
        macro_dt=0.03,
        dt_max=1.0e-3,
    )
    stepper = KolmogorovReferenceStepper(config)
    grid = np.arange(config.resolution, dtype=np.float64) * (
        config.domain_length / config.resolution
    )
    x, y = np.meshgrid(grid, grid, indexing="ij")
    initial = stepper.canonicalize(2.0 * np.sin(x) * np.sin(y))
    result = stepper.advance_canonical(initial)
    expected = initial * math.exp(
        -(2.0 * config.viscosity + config.linear_drag) * config.macro_dt
    )

    np.testing.assert_allclose(result.state, expected, rtol=2.0e-13, atol=2.0e-13)
    assert result.diagnostics.substeps == 30
    assert result.diagnostics.input_projection_relative_l2 < 1.0e-14


def test_configured_laminar_kolmogorov_state_is_steady() -> None:
    config = KolmogorovReferenceConfig(
        resolution=24,
        viscosity=0.05,
        linear_drag=0.1,
        forcing_amplitude=0.2,
        forcing_wavenumber=3,
        macro_dt=0.04,
        dt_max=1.0e-3,
    )
    stepper = KolmogorovReferenceStepper(config)
    steady = stepper.laminar_vorticity()
    result = stepper.advance_canonical(steady)

    np.testing.assert_allclose(result.state, steady, rtol=0.0, atol=3.0e-13)
    diagnostics = stepper.diagnostics_canonical(result.state)
    assert abs(diagnostics.mean_vorticity) < 1.0e-14
    assert diagnostics.kinetic_energy > 0.0
    assert diagnostics.enstrophy > 0.0
    assert diagnostics.palinstrophy > 0.0
    _, shell_energy = stepper.kinetic_energy_spectrum_canonical(result.state)
    np.testing.assert_allclose(
        np.sum(shell_energy), diagnostics.kinetic_energy, rtol=2.0e-15, atol=1.0e-15
    )


def test_dealiased_spectral_resize_roundtrip_preserves_source_polynomial() -> None:
    source = KolmogorovReferenceStepper(
        KolmogorovReferenceConfig(resolution=18, forcing_wavenumber=2)
    )
    state = _random_canonical(source)
    upsampled = resize_dealiased_vorticity(state, 36)
    restored = resize_dealiased_vorticity(upsampled, 18)

    assert upsampled.shape == (36, 36)
    np.testing.assert_allclose(restored, state, rtol=0.0, atol=3.0e-15)
    with pytest.raises(ValueError, match="float64"):
        resize_dealiased_vorticity(state.astype(np.float32), 36)
    with pytest.raises(ValueError, match="target resolution"):
        resize_dealiased_vorticity(state, 17)


def test_canonical_restart_is_deterministic_and_closes_repeated_rollout() -> None:
    config = KolmogorovReferenceConfig(
        resolution=18,
        viscosity=0.02,
        linear_drag=0.1,
        forcing_amplitude=0.2,
        forcing_wavenumber=2,
        macro_dt=0.01,
        dt_max=2.0e-3,
    )
    stepper = KolmogorovReferenceStepper(config)
    initial = _random_canonical(stepper)

    first_a = stepper.advance_canonical(initial)
    first_b = stepper.advance_canonical(initial.copy())
    assert np.array_equal(first_a.state, first_b.state)
    assert first_a.diagnostics == first_b.diagnostics

    second = stepper.advance_canonical(first_a.state)
    rollout, records = stepper.rollout_canonical(initial, 2)
    assert np.array_equal(rollout[0], initial)
    assert np.array_equal(rollout[1], first_a.state)
    assert np.array_equal(rollout[2], second.state)
    assert records == (first_a.diagnostics, second.diagnostics)


def test_projection_is_explicit_and_canonical_path_fails_closed() -> None:
    stepper = KolmogorovReferenceStepper(
        KolmogorovReferenceConfig(
            resolution=18,
            macro_dt=0.005,
            dt_max=1.0e-3,
        )
    )
    rng = np.random.default_rng(9)
    raw = rng.normal(size=stepper.state_shape)

    with pytest.raises(ValueError, match="canonical"):
        stepper.advance_canonical(raw)
    with pytest.raises(ValueError, match="float64"):
        stepper.advance_canonical(stepper.canonicalize(raw).astype(np.float32))
    with pytest.raises(ValueError, match="shape"):
        stepper.advance_projected(raw[:-1])
    nonfinite = raw.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        stepper.advance_projected(nonfinite)

    projected = stepper.advance_projected(raw)
    assert projected.diagnostics.input_projection_relative_l2 > 0.1
    assert stepper.projection_relative_l2(projected.state) < 1.0e-14


def test_substep_limit_fails_closed() -> None:
    stepper = KolmogorovReferenceStepper(
        KolmogorovReferenceConfig(
            resolution=18,
            forcing_wavenumber=2,
            macro_dt=0.01,
            dt_max=1.0e-3,
            max_substeps=1,
        )
    )
    initial = _random_canonical(stepper)
    with pytest.raises(RuntimeError, match="max_substeps"):
        stepper.advance_canonical(initial)
