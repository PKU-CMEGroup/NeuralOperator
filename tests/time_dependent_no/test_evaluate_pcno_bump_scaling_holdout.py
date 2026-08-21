from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.time_dependent_no import evaluate_pcno_bump_scaling_holdout as holdout
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    EXPECTED_SPLIT_PARTITION_DIGEST,
    SPLIT_SCHEMA,
    build_parser,
    outside_selection_keys,
    schedule_decision,
)


def _manifest() -> dict[str, object]:
    return {
        "schema": SPLIT_SCHEMA,
        "state_arrays_opened": False,
        "historical_test_population_opened": False,
        "partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
        "split": {"open_validation_keys": [str(index) for index in range(44)]},
    }


def _cell_split() -> dict[str, object]:
    return {
        "val_keys": [str(index) for index in range(44)],
        "rollout_keys": [str(index) for index in range(0, 32, 2)],
    }


def test_outside_selection_cohort_is_exact_complement() -> None:
    validation, selection, outside = outside_selection_keys(
        _manifest(), [_cell_split() for _ in range(4)]
    )

    assert len(validation) == 44
    assert len(selection) == 16
    assert len(outside) == 28
    assert set(selection).isdisjoint(outside)
    assert [key for key in validation if key not in set(selection)] == outside


def test_outside_selection_rejects_cell_cohort_drift() -> None:
    cells = [_cell_split() for _ in range(4)]
    cells[-1] = dict(cells[-1])
    cells[-1]["rollout_keys"] = [str(index) for index in range(16)]

    with pytest.raises(ValueError, match="do not share"):
        outside_selection_keys(_manifest(), cells)


def _audit_cell(full: float, h79: float, *, complete: bool = True) -> dict[str, object]:
    return {
        "outside_selection_rollout": {
            "completion_rate": 1.0 if complete else 0.9,
            "hard_failure_count": 0 if complete else 1,
            "mean_full_horizon_relative_l2": full,
            "mean_endpoint_relative_l2": {"79": h79},
        }
    }


def test_schedule_decision_uses_rollout_not_admissibility() -> None:
    cells = {
        ("pcno", "prefix_tail"): _audit_cell(0.10, 0.15),
        ("pcfno", "prefix_tail"): _audit_cell(0.09, 0.14),
        ("pcno", "stretched"): _audit_cell(0.06, 0.09),
        ("pcfno", "stretched"): _audit_cell(0.08, 0.12),
    }

    decision = schedule_decision(cells)

    assert decision["winner"] == "stretched"
    assert decision["authorizes_seed0_ladder_schedule"] is True
    assert decision["paired_stretched_over_prefix_tail_ratios"]["pcno"][
        "all_call_mean_stretched_over_prefix_tail"
    ] == pytest.approx(0.6)


def test_numerical_failure_precedes_lower_error() -> None:
    cells = {
        ("pcno", "prefix_tail"): _audit_cell(0.10, 0.15),
        ("pcfno", "prefix_tail"): _audit_cell(0.09, 0.14),
        ("pcno", "stretched"): _audit_cell(0.01, 0.02, complete=False),
        ("pcfno", "stretched"): _audit_cell(0.01, 0.02),
    }

    decision = schedule_decision(cells)

    assert decision["winner"] == "prefix_tail"


def test_parser_exposes_no_historical_test_input() -> None:
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    destinations = {action.dest for action in parser._actions}
    assert "test" not in destinations
    assert "test_root" not in destinations
    assert {"gate_root", "data_dir", "split_manifest", "output_dir"} <= destinations


def test_evaluate_cell_rejects_checkpoint_boundary_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = {
        "data_manifest_digest": "manifest",
        "source_snapshot": {"source_set_digest": "source"},
        "val_keys": ["trajectory"],
        "boundary_contract": {"policy_digests": {"trajectory": "stale"}},
    }
    monkeypatch.setattr(holdout, "load_bump_checkpoint", lambda _: checkpoint)
    descriptor = {
        "checkpoint": tmp_path / "best.pt",
        "split": {"val_keys": ["trajectory"]},
    }
    store = type("Store", (), {"manifest_digest": "manifest"})()

    with pytest.raises(ValueError, match="boundary-policy digest changed"):
        holdout._evaluate_cell(
            descriptor,
            store=store,
            outside_keys=["trajectory"],
            policies={},
            policy_metadata={"trajectory": {"policy_digest": "current"}},
            expected_checkpoint_source_set_digest="source",
            device=holdout.torch.device("cpu"),
            amp="none",
            shock_quantile=0.9,
        )
