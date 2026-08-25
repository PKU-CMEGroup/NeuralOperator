from __future__ import annotations

import pytest

from scripts.time_dependent_no import analyze_pcno_bump_b1_c3_r1 as analysis


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for precision_index, precision in enumerate(analysis.PRECISIONS):
        for seed_index, seed in enumerate(analysis.SEEDS):
            for count in analysis.COUNTS:
                for architecture in analysis.ARCHITECTURES:
                    scale = 0.5 if architecture == "pcno" else 1.0
                    value = scale * (1.0 + 0.1 * seed_index)
                    value *= 1.0 + 0.01 * precision_index
                    rows.append(
                        {
                            "precision": precision,
                            "seed": seed,
                            "trajectory_count": count,
                            "architecture": architecture,
                            **{metric: value for metric in analysis.prior.ERROR_METRICS},
                        }
                    )
    return rows


def test_three_seed_architecture_ratios_keep_pairing() -> None:
    ratios = analysis._architecture_ratios(_rows(), analysis.prior.ERROR_METRICS)
    summary = analysis._ratio_summary(ratios)

    assert len(ratios) == (
        len(analysis.PRECISIONS)
        * len(analysis.SEEDS)
        * len(analysis.COUNTS)
        * len(analysis.prior.ERROR_METRICS)
    )
    assert {row["pcno_over_pcfno"] for row in ratios} == {0.5}
    assert all(row["seed_count"] == 3 for row in summary)
    assert all(row["pcno_lower_seed_count"] == 3 for row in summary)
    assert all(row["direction_consistent"] for row in summary)


def test_three_seed_precision_ratios_match_cells() -> None:
    ratios = analysis._precision_ratios(_rows())

    assert len(ratios) == (
        len(analysis.SEEDS)
        * len(analysis.COUNTS)
        * len(analysis.ARCHITECTURES)
        * len(analysis.prior.ERROR_METRICS)
    )
    assert all(row["fp32_over_bf16"] == pytest.approx(1.01) for row in ratios)


def test_r1_metric_value_separates_training_validation_and_rollout() -> None:
    row = {
        "learning_rate": 1.0e-3,
        "train": {
            "relative_l2": 0.1,
            "comparable_seen": {"relative_l2": 0.2},
        },
        "validation": {"relative_l2": 0.3},
        "rollout": {"mean_endpoint_relative_l2": {"79": 0.4}},
    }

    assert analysis._r1_metric_value(row, "learning_rate") == 1.0e-3
    assert (
        analysis._r1_metric_value(row, "online_train_one_step_relative_l2")
        == 0.1
    )
    assert (
        analysis._r1_metric_value(row, "fixed_seen_train_one_step_relative_l2")
        == 0.2
    )
    assert (
        analysis._r1_metric_value(row, "fixed_validation_one_step_relative_l2")
        == 0.3
    )
    assert analysis._r1_metric_value(row, "rollout_h79_relative_l2") == 0.4


def test_r1_metric_value_preserves_missing_sparse_observations() -> None:
    row = {
        "learning_rate": 1.0e-3,
        "train": {"relative_l2": 0.1, "comparable_seen": {}},
        "validation": {"relative_l2": 0.2},
        "rollout": {},
    }

    assert (
        analysis._r1_metric_value(row, "fixed_seen_train_one_step_relative_l2")
        is None
    )
    assert analysis._r1_metric_value(row, "rollout_h79_relative_l2") is None

    row["train"]["comparable_seen"] = None
    row["rollout"] = None
    assert (
        analysis._r1_metric_value(row, "fixed_seen_train_one_step_relative_l2")
        is None
    )
    assert analysis._r1_metric_value(row, "rollout_h79_relative_l2") is None


def test_optional_float_preserves_sparse_training_surface_cells() -> None:
    assert analysis._optional_float(None) is None
    assert analysis._optional_float("") is None
    assert analysis._optional_float("0.125") == pytest.approx(0.125)


def test_source_comparison_statuses_are_explicit() -> None:
    old = {
        "source_snapshot": {
            "files": {
                "same.py": {"sha256": "a"},
                "changed.py": {"sha256": "b"},
                "old.py": {"sha256": "c"},
            }
        }
    }
    new_files = {
        "same.py": {"sha256": "a"},
        "changed.py": {"sha256": "d"},
        "new.py": {"sha256": "e"},
    }

    # Exercise the status expression without constructing the filesystem-backed
    # run inventory used by the integration path.
    rows = []
    for name in sorted(set(old["source_snapshot"]["files"]) | set(new_files)):
        old_hash = old["source_snapshot"]["files"].get(name, {}).get("sha256")
        new_hash = new_files.get(name, {}).get("sha256")
        rows.append(
            "same"
            if old_hash == new_hash
            else "historical_only"
            if new_hash is None
            else "replay_only"
            if old_hash is None
            else "changed"
        )

    assert rows == ["changed", "replay_only", "historical_only", "same"]
