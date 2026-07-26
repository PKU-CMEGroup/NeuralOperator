from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.benchmark_pcno_shock_vortex_coarse_cfd import (
    load_pcno_point,
    paired_case_bootstrap,
    select_full_grids,
)
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


def test_pilot_selection_retains_cost_and_error_matches_without_a_sweep() -> None:
    rows = [
        {
            "grid": "25x10",
            "cells": 250,
            "median_core_seconds": 0.1,
            "mean_horizon_error": 0.04,
        },
        {
            "grid": "50x20",
            "cells": 1000,
            "median_core_seconds": 0.4,
            "mean_horizon_error": 0.02,
        },
        {
            "grid": "125x50",
            "cells": 6250,
            "median_core_seconds": 1.5,
            "mean_horizon_error": 0.008,
        },
    ]
    selected = select_full_grids(rows, pcno_seconds=0.45, pcno_error=0.009, maximum=2)
    assert selected["pilot_cost_match_grid"] == "50x20"
    assert selected["pilot_error_match_grid"] == "125x50"
    assert selected["selected_grids"] == ["50x20", "125x50"]


def test_pcno_point_keeps_pilot_and_full_cohort_errors_distinct(tmp_path: Path) -> None:
    call_metrics = tmp_path / "call_metrics.csv"
    call_metrics.write_text(
        "variant,call,trajectory,scaled_relative_l2_physical_volume\n"
        "pcno_baseline,3,pilot_a,1.0\n"
        "pcno_baseline,3,pilot_b,3.0\n"
        "pcno_baseline,3,heldout,10.0\n",
        encoding="utf-8",
    )
    trajectory_metrics = tmp_path / "trajectory_metrics.csv"
    trajectory_metrics.write_text(
        "variant,trajectory,total_forward_seconds\n"
        "pcno_baseline,pilot_a,0.1\n"
        "pcno_baseline,pilot_b,0.2\n"
        "pcno_baseline,heldout,0.3\n",
        encoding="utf-8",
    )
    contract = {
        "source": {
            "aggregates": {
                "pcno_baseline": {
                    "common_endpoint_mean_relative_l2": 14.0 / 3.0,
                }
            },
            "cost": {
                "throughput_samples_per_second": 120.0,
                "throughput_batch_size": 4,
            },
        },
        "num_steps": 3,
        "keys": ["pilot_a", "pilot_b", "heldout"],
        "pilot_keys": ["pilot_a", "pilot_b"],
        "source_files": {
            "call_metrics.csv": call_metrics,
            "trajectory_metrics.csv": trajectory_metrics,
        },
        "timing": {"cost": {"contract": {"boundary": "device_resident"}}},
        "pcno_horizon_seconds": 0.25,
        "pcno_horizon_p95_envelope_seconds": 0.3,
        "pcno_timing_p95_median_ratio": 1.2,
    }

    point = load_pcno_point(contract)

    assert point["mean_horizon_error"] == pytest.approx(14.0 / 3.0)
    assert point["pilot_mean_horizon_error"] == pytest.approx(2.0)


def test_paired_case_bootstrap_reports_only_case_resampling() -> None:
    summary = paired_case_bootstrap(
        np.asarray([0.01, 0.02, 0.03, 0.04]), repetitions=2000, seed=7
    )
    assert summary["mean_cfd_minus_pcno"] == pytest.approx(0.025)
    assert summary["pcno_lower_error_fraction"] == 1.0
    assert 0.0 < summary["ci95_low"] <= summary["ci95_high"]
