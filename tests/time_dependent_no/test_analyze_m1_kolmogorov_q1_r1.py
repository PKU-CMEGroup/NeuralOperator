from __future__ import annotations

import numpy as np
import pytest

from scripts.time_dependent_no.analyze_m1_kolmogorov_q1_r1 import (
    _horizon_values,
    candidate_table,
    metric_endpoint_table,
)


def _metric(pass_value: bool, offset: float) -> dict[str, object]:
    return {
        "rows": [
            {"half_change": 0.02 + offset},
            {"half_change": 0.03 + offset},
        ],
        "split_rhat": 1.01 + offset,
        "pooled_effective_sample_size": 150.0 - offset,
        "half_change_pass": pass_value,
        "shared_drift": {"pass": True},
        "rhat_pass": pass_value,
        "ess_pass": pass_value,
    }


def test_candidate_table_preserves_registered_gate_values() -> None:
    diagnostic = {
        "candidates": [
            {
                "burnin_calls": 512,
                "observation_start_call": 513,
                "observation_end_call": 1024,
                "energy": _metric(True, 0.0),
                "enstrophy": _metric(False, 0.1),
                "pass": False,
            }
        ]
    }
    row = candidate_table(diagnostic)[0]
    assert row["energy_maximum_half_change"] == 0.03
    assert row["enstrophy_maximum_half_change"] == 0.13
    assert row["energy_rhat_pass"]
    assert not row["enstrophy_ess_pass"]
    assert not row["candidate_pass"]


def _spatial_rows() -> list[dict[str, object]]:
    rows = []
    for pair_index, pair in enumerate(("64_to_128", "128_to_256"), start=1):
        for horizon in (1, 2, 3):
            for case_index in range(2):
                value = 0.01 * pair_index * horizon + 0.001 * case_index
                rows.append(
                    {
                        "pair": pair,
                        "horizon": horizon,
                        "state_relative_l2": value,
                        "energy_relative_difference": value / 10.0,
                        "enstrophy_relative_difference": value / 5.0,
                        "palinstrophy_relative_difference": value / 2.0,
                        "spectrum_total_variation": value / 4.0,
                    }
                )
    return rows


def test_horizon_values_are_sorted_and_aggregate_each_case() -> None:
    horizons, medians, minima, maxima = _horizon_values(
        _spatial_rows(), "64_to_128", "state_relative_l2"
    )
    np.testing.assert_array_equal(horizons, np.asarray([1, 2, 3]))
    np.testing.assert_allclose(medians, np.asarray([0.0105, 0.0205, 0.0305]))
    np.testing.assert_allclose(minima, np.asarray([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(maxima, np.asarray([0.011, 0.021, 0.031]))


def test_metric_endpoint_table_reports_median_and_maximum() -> None:
    table = metric_endpoint_table(_spatial_rows(), "128_to_256", 3)
    by_metric = {row["metric"]: row for row in table}
    assert by_metric["energy_relative_difference"]["median"] == pytest.approx(0.00605)
    assert by_metric["energy_relative_difference"]["maximum"] == pytest.approx(0.0061)
    assert by_metric["palinstrophy_relative_difference"]["median"] == pytest.approx(
        0.03025
    )
