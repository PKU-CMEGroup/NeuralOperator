from __future__ import annotations

import pytest

from scripts.time_dependent_no import analyze_pcno_bump_b1_c3 as analysis


def _cell_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for precision_index, precision in enumerate(analysis.PRECISIONS):
        for seed_index, seed in enumerate(analysis.SEEDS):
            for count in analysis.COUNTS:
                for architecture in analysis.ARCHITECTURES:
                    architecture_scale = 0.5 if architecture == "pcno" else 1.0
                    value = architecture_scale * (1.0 + 0.1 * seed_index)
                    value *= 1.0 + 0.01 * precision_index
                    rows.append(
                        {
                            "precision": precision,
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            "optimizer_step": 64 * count,
                            **{metric: value for metric in analysis.ERROR_METRICS},
                        }
                    )
    return rows


def test_architecture_ratios_preserve_paired_seed_count_and_precision() -> None:
    ratios = analysis._architecture_ratios(_cell_rows(), analysis.ERROR_METRICS)

    assert len(ratios) == (
        len(analysis.PRECISIONS)
        * len(analysis.SEEDS)
        * len(analysis.COUNTS)
        * len(analysis.ERROR_METRICS)
    )
    assert {row["pcno_over_pcfno"] for row in ratios} == {0.5}
    assert all(row["pcno_lower"] for row in ratios)


def test_precision_ratios_compare_identical_checkpoint_cells() -> None:
    ratios = analysis._precision_ratios(_cell_rows())

    assert len(ratios) == (
        len(analysis.SEEDS)
        * len(analysis.COUNTS)
        * len(analysis.ARCHITECTURES)
        * len(analysis.ERROR_METRICS)
    )
    assert all(row["fp32_over_bf16"] == pytest.approx(1.01) for row in ratios)


def test_scaling_summary_treats_a_flat_curve_as_nonincreasing() -> None:
    summary = analysis._scaling_summary(
        _cell_rows(), ("outside_rollout_h79_relative_l2",)
    )

    assert len(summary) == len(analysis.PRECISIONS) * len(analysis.ARCHITECTURES)
    assert all(row["monotone_nonincreasing"] for row in summary)
    assert all(row["n128_over_n8"] == pytest.approx(1.0) for row in summary)
    assert all(row["n256_over_n128"] == pytest.approx(1.0) for row in summary)


def test_paired_bootstrap_is_deterministic_and_keeps_case_pairing() -> None:
    first = analysis._bootstrap_ratio(
        [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], draws=200, seed=11
    )
    second = analysis._bootstrap_ratio(
        [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], draws=200, seed=11
    )

    assert first == second
    assert first[0] == pytest.approx(0.5)
    assert first[1] == pytest.approx(0.5)


def test_bootstrap_rejects_unpaired_or_too_small_input() -> None:
    with pytest.raises(ValueError, match="equal nonempty"):
        analysis._bootstrap_ratio([1.0], [1.0, 2.0], draws=100, seed=0)
    with pytest.raises(ValueError, match="equal nonempty"):
        analysis._bootstrap_ratio([], [], draws=100, seed=0)


def test_cell_row_separates_one_step_generalization_and_rollout_amplification() -> None:
    cell = {
        "seed": analysis.SEEDS[0],
        "trajectory_count": analysis.COUNTS[0],
        "architecture": "pcno",
        "optimizer_step": 64 * analysis.COUNTS[0],
        "checkpoint_sha256": "0" * 64,
        "checkpoint_training_metrics": {
            "online_train_one_step_relative_l2": 0.05,
            "stored_fixed_validation_one_step_relative_l2": 0.2,
            "fixed_seen_train_one_step_relative_l2": 0.1,
            "fixed_validation_one_step_relative_l2": 0.2,
        },
        "selection_rollout": {
            "mean_full_horizon_relative_l2": 0.3,
            "mean_endpoint_relative_l2": {"79": 0.4},
        },
        "outside_selection_rollout": {
            "mean_full_horizon_relative_l2": 0.5,
            "mean_endpoint_relative_l2": {"79": 0.6},
            "physical_admissibility_rate": 1.0,
        },
        "common_outside_selection_rollout": {
            "mean_full_horizon_relative_l2": 0.7,
            "mean_endpoint_relative_l2": {"79": 0.8},
        },
        "outside_selection_h79_structure_means": {},
    }

    row = analysis._cell_row("bf16", cell)

    assert row["fixed_validation_over_seen"] == pytest.approx(2.0)
    assert row["outside_h79_over_fixed_validation"] == pytest.approx(3.0)


def test_transition_rows_require_a_permanent_not_transient_crossing() -> None:
    rows = []
    for seed in analysis.SEEDS:
        for count in analysis.COUNTS:
            for step, rollout_ratio in ((256, 1.2), (512, 0.8), (768, 0.7)):
                rows.append(
                    {
                        "seed": str(seed),
                        "trajectory_count": str(count),
                        "optimizer_step": str(step),
                        "pcno_over_pcfno_rollout_h79_relative_l2": str(
                            rollout_ratio
                        ),
                        "pcno_over_pcfno_fixed_validation_one_step_relative_l2": (
                            "0.75"
                        ),
                    }
                )

    result = analysis._transition_rows(rows)

    assert len(result) == len(analysis.SEEDS) * len(analysis.COUNTS)
    assert all(row["permanent_pcno_lower_step"] == 512 for row in result)
    assert result[0]["permanent_pcno_lower_scheduler_fraction"] == pytest.approx(
        512 / analysis.B1_C2_TOTAL_STEPS
    )
    assert result[0]["permanent_pcno_lower_presentations_per_trajectory"] == 64

    coordinate_summary = analysis._transition_coordinate_summary(result)
    by_coordinate = {row["coordinate"]: row for row in coordinate_summary}
    assert by_coordinate["permanent_pcno_lower_step"][
        "maximum_over_minimum"
    ] == pytest.approx(1.0)
    assert by_coordinate[
        "permanent_pcno_lower_presentations_per_trajectory"
    ]["maximum_over_minimum"] == pytest.approx(
        max(analysis.COUNTS) / min(analysis.COUNTS)
    )
