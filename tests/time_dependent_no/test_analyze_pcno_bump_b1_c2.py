from __future__ import annotations

import math

import pytest

from scripts.time_dependent_no import analyze_pcno_bump_b1_c2 as analysis


def test_summary_stats_use_sample_standard_deviation() -> None:
    result = analysis._summary_stats([1.0, 2.0, 3.0])

    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["sample_std"] == 1.0
    assert result["coefficient_of_variation"] == 0.5


def test_log_log_slope_recovers_power_law() -> None:
    counts = [8, 16, 32, 64]
    values = [count**-2 for count in counts]

    assert analysis._log_log_slope(counts, values) == pytest.approx(
        -2.0 * math.log10(2.0)
    )


def test_paired_bootstrap_is_deterministic_and_keeps_pairs() -> None:
    first = analysis._paired_bootstrap(
        [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], draws=200, seed=7
    )
    second = analysis._paired_bootstrap(
        [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], draws=200, seed=7
    )

    assert first == second
    assert first["pcno_over_pcfno"] == 0.5
    assert first["ratio_ci025"] == pytest.approx(0.5)
    assert first["ratio_ci975"] == pytest.approx(0.5)
    assert first["pcno_lower_case_count"] == 3


def test_generalization_ratio_summary_uses_each_complete_identity() -> None:
    rows = []
    for seed in analysis.SEEDS:
        for count in analysis.TRAJECTORY_COUNTS:
            for architecture in analysis.ARCHITECTURES:
                for step, ratio in ((512, 1.2), (256, 1.0)):
                    rows.append(
                        {
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            "optimizer_step": step,
                            "fixed_validation_over_seen_ratio": ratio,
                        }
                    )
    summaries = analysis.generalization_ratio_summary_rows(rows)
    assert len(summaries) == (
        len(analysis.SEEDS)
        * len(analysis.TRAJECTORY_COUNTS)
        * len(analysis.ARCHITECTURES)
    )
    assert summaries[0]["first_ratio"] == 1.0
    assert summaries[0]["last_ratio"] == 1.2
    assert summaries[0]["relative_range"] == pytest.approx(0.2 / 1.1)


def test_validate_cells_requires_exact_three_seed_matrix() -> None:
    common = ["c"]
    outside_by_seed = {str(seed): [f"s{seed}"] for seed in analysis.SEEDS}
    cells = []
    for seed in analysis.SEEDS:
        for count in analysis.TRAJECTORY_COUNTS:
            for architecture in analysis.ARCHITECTURES:
                for role in analysis.CHECKPOINT_ROLES:
                    key = f"s{seed}"
                    cells.append(
                        {
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            "checkpoint_role": role,
                            "audit_cohort": {"outside_selection_keys": [key]},
                            "outside_selection_rollout": {
                                "trajectories": [{"trajectory": key}]
                            },
                            "common_outside_selection_rollout": {"keys": common},
                        }
                    )
    summary = {
        "cells": cells,
        "common_outside_selection_keys": common,
        "outside_selection_keys_by_seed": outside_by_seed,
    }

    indexed = analysis.validate_cells(summary)

    assert len(indexed) == 72
    summary["cells"] = cells[:-1]
    with pytest.raises(ValueError, match="identity matrix"):
        analysis.validate_cells(summary)
