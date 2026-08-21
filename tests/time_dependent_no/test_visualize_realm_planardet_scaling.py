from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no.visualize_realm_planardet_scaling import (
    BASE_VISUALIZER_PATH,
    VISUALIZATION_SOURCE_PATHS,
    VISUALIZER_PATH,
    _frozen_initial_persistence_rows,
    _model_records,
    _residual_npe,
    _rollout_curve_rows,
    _sampler_accounting,
    _truth_input_residuals,
    _validate_fields,
    build_parser,
)


def test_truth_input_residuals_are_one_step_not_accumulated() -> None:
    truth = np.asarray([0.0, 2.0, 5.0, 9.0], dtype=np.float32)[:, None, None]
    teacher = np.asarray([0.0, 2.5, 4.0, 11.0], dtype=np.float32)[:, None, None]

    true_residual, predicted_residual, prediction_error = _truth_input_residuals(
        truth, teacher
    )

    np.testing.assert_array_equal(true_residual[:, 0, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(predicted_residual[:, 0, 0], [2.5, 2.0, 6.0])
    np.testing.assert_array_equal(prediction_error[:, 0, 0], [0.5, -1.0, 2.0])


def test_truth_input_residuals_reject_shape_drift() -> None:
    with pytest.raises(ValueError, match="must match"):
        _truth_input_residuals(np.zeros((2, 1)), np.zeros((3, 1)))


def test_residual_npe_uses_per_frame_squared_relative_error() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    error = np.asarray([[1.0, 0.0], [0.0, 2.0]])

    np.testing.assert_allclose(_residual_npe(truth, error), [1.0 / 5.0, 4.0 / 25.0])


def test_model_records_have_fixed_architecture_exposure_order(tmp_path: Path) -> None:
    args = argparse.Namespace(
        d092_training_dir=tmp_path / "d092_train",
        d092_evaluation_dir=tmp_path / "d092_eval",
        d093_training_root=tmp_path / "d093_train",
        d093_evaluation_root=tmp_path / "d093_eval",
    )

    records = _model_records(args)

    assert [(record.architecture, record.exposure) for record in records] == [
        ("pcno", 7),
        ("pcfno", 7),
        ("ffno", 7),
        ("pcno", 3),
        ("pcfno", 3),
        ("ffno", 3),
    ]
    assert [record.training_complete for record in records] == [
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_sampler_accounting_distinguishes_duplicate_terms_from_updates() -> None:
    n3 = _sampler_accounting(3, 850)
    n7 = _sampler_accounting(7, 850)

    assert n3["optimizer_steps"] == n7["optimizer_steps"] == 850
    assert n3["raw_micro_presentations"] == n7["raw_micro_presentations"] == 5950
    assert n3["duplicate_condition_window_terms_per_step"] == 4
    assert n7["duplicate_condition_window_terms_per_step"] == 0
    assert n3["scheduled_window_positions_per_step"] == 1
    assert n7["scheduled_window_positions_per_step"] == 1
    assert n3["two_call_window_cycles_per_active_trajectory"] == pytest.approx(
        360 / 48
    )
    assert n7["two_call_window_cycles_per_active_trajectory"] == pytest.approx(
        360 / 48
    )

    with pytest.raises(ValueError, match="three or seven"):
        _sampler_accounting(5, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        _sampler_accounting(3, -1)


def test_rollout_rows_keep_teacher_and_free_sequences_separate(tmp_path: Path) -> None:
    args = argparse.Namespace(
        d092_training_dir=tmp_path / "d092_train",
        d092_evaluation_dir=tmp_path / "d092_eval",
        d093_training_root=tmp_path / "d093_train",
        d093_evaluation_root=tmp_path / "d093_eval",
    )
    records = _model_records(args)[:1]
    results = {
        records[0].key: {
            "views": {
                "ordered_teacher_forced": {
                    "summary": {"npe_total_by_call": [1.0, 2.0]}
                },
                "free_recurrence": {
                    "summary": {"npe_total_by_call": [3.0, 4.0]}
                },
            }
        }
    }

    rows = _rollout_curve_rows(records, results)

    assert [row["teacher_npe"] for row in rows] == [1.0, 2.0]
    assert [row["free_npe"] for row in rows] == [3.0, 4.0]
    assert rows[-1]["teacher_cumulative_npe"] == pytest.approx(3.0)
    assert rows[-1]["free_cumulative_npe"] == pytest.approx(7.0)


def test_frozen_initial_persistence_never_refreshes_from_truth() -> None:
    sequence = torch.stack(
        (
            torch.zeros((13, 1, 1)),
            torch.ones((13, 1, 1)),
            torch.full((13, 1, 1), 2.0),
        )
    )

    rows = _frozen_initial_persistence_rows(sequence, [0.0, 0.1, 0.2])

    assert [row["frozen_u0_npe"] for row in rows] == pytest.approx([5.0, 20.0])
    assert rows[-1]["frozen_u0_cumulative_npe"] == pytest.approx(25.0)
    assert rows[-1]["physical_time_s"] == pytest.approx(0.2)
    assert rows[-1]["elapsed_time_s"] == pytest.approx(0.2)
    assert rows[-1]["chem_npe"] == pytest.approx(4.0)


def test_frozen_initial_persistence_rejects_time_or_state_drift() -> None:
    sequence = torch.zeros((2, 13, 1, 1))
    with pytest.raises(ValueError, match="match time"):
        _frozen_initial_persistence_rows(sequence, [0.0])
    nonfinite = sequence.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _frozen_initial_persistence_rows(nonfinite, [0.0, 1.0])


def test_animation_fields_and_parser_are_fail_closed() -> None:
    assert _validate_fields(("H2O", "T", "pMax")) == ("H2O", "T", "pMax")
    with pytest.raises(ValueError):
        _validate_fields(("T", "T"))
    with pytest.raises(ValueError):
        _validate_fields(("not-a-field",))

    destinations = {action.dest for action in build_parser()._actions}
    assert "test" not in destinations
    assert "test_root" not in destinations
    assert "skip_animations" in destinations


def test_visualization_source_inventory_includes_both_visualizers() -> None:
    assert VISUALIZER_PATH in VISUALIZATION_SOURCE_PATHS
    assert BASE_VISUALIZER_PATH in VISUALIZATION_SOURCE_PATHS
    assert len(VISUALIZATION_SOURCE_PATHS) == len(set(VISUALIZATION_SOURCE_PATHS))
