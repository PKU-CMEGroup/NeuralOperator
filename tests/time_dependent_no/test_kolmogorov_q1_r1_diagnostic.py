from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no.run_m1_kolmogorov_q1_r1_diagnostic import (
    _quick_contract,
    autocorrelation_fft,
    integrated_autocorrelation_time,
    pooled_effective_sample_size,
    split_rhat,
    summarize_spatial_rows,
)


def test_chain_diagnostics_distinguish_white_noise_from_persistent_series() -> None:
    rng = np.random.default_rng(20260826)
    white = rng.normal(size=1024)
    persistent = np.empty(1024, dtype=np.float64)
    persistent[0] = white[0]
    for index in range(1, persistent.size):
        persistent[index] = 0.9 * persistent[index - 1] + white[index]

    white_tau = integrated_autocorrelation_time(white)
    persistent_tau = integrated_autocorrelation_time(persistent)
    assert 1.0 <= white_tau < 2.0
    assert persistent_tau > 5.0 * white_tau
    correlation = autocorrelation_fft(persistent)
    assert correlation[0] == 1.0
    assert correlation[1] > 0.8


def test_split_rhat_and_pooled_ess_detect_between_chain_shift() -> None:
    rng = np.random.default_rng(17)
    chains = rng.normal(size=(4, 512))
    shifted = chains.copy()
    shifted[-1] += 2.0

    assert split_rhat(chains) < 1.02
    assert split_rhat(shifted) > 1.1
    effective, taus = pooled_effective_sample_size(chains)
    assert effective > 1000.0
    assert len(taus) == 4


def test_chain_diagnostics_fail_closed_on_invalid_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        autocorrelation_fft(np.ones((2, 3)))
    with pytest.raises(ValueError, match="even"):
        split_rhat(np.ones((4, 5)))
    with pytest.raises(ValueError, match="finite"):
        split_rhat(np.full((4, 8), np.nan))


def test_spatial_screen_requires_absolute_and_contraction_gates() -> None:
    _, settings = _quick_contract()
    low, candidate, high = settings.spatial_resolutions
    rows = []
    for case_index, family in enumerate(("clean", "displaced")):
        for pair, errors in (
            (f"{low}_to_{candidate}", (0.02, 0.08)),
            (f"{candidate}_to_{high}", (0.005, 0.02)),
        ):
            for horizon, error in (
                (1, errors[0]),
                (settings.spatial_horizon, errors[1]),
            ):
                rows.append(
                    {
                        "case": f"{family}_{case_index}",
                        "family": family,
                        "pair": pair,
                        "horizon": horizon,
                        "state_relative_l2": error,
                        "energy_relative_difference": error,
                        "enstrophy_relative_difference": error,
                        "palinstrophy_relative_difference": error,
                        "spectrum_total_variation": error,
                        "finite": True,
                    }
                )

    passed = summarize_spatial_rows(rows, settings)
    assert passed["absolute_pass"]
    assert passed["contraction_pass"]
    assert passed["screen_without_repeatability"]

    for row in rows:
        if row["pair"] == f"{candidate}_to_{high}" and row["horizon"] == 1:
            row["state_relative_l2"] = 0.03
    failed = summarize_spatial_rows(rows, settings)
    assert not failed["absolute_pass"]
    assert not failed["screen_without_repeatability"]
