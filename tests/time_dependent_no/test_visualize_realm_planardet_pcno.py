from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.time_dependent_no import visualize_realm_planardet_pcno as visualizer
from utility.time_dependent_no.realm_planardet import sha256_file


def _summary(npe: list[float], correlation: list[float]) -> dict[str, object]:
    groups = {
        name: [value / 5.0 for value in npe] for name in ("chem", "T", "rho", "u", "p")
    }
    return {
        "npe_total_by_call": npe,
        "decoded_correlation_by_call": correlation,
        "pMax_decoded_correlation_by_call": correlation,
        "npe_group_by_call": groups,
        "normalized_mse_by_call_channel": [
            [value / 13.0 for _ in range(13)] for value in npe
        ],
        "realm_npe_mean": float(np.mean(npe)),
    }


def _result() -> dict[str, object]:
    teacher = _summary([0.1] * 49, [0.95] * 49)
    free = _summary(
        [0.1 + 0.05 * index for index in range(49)],
        [0.95 - 0.01 * index for index in range(49)],
    )
    return {
        "controls": {
            "persistence": {"realm_npe_mean": 1.0},
            "linear_normalized": {"realm_npe_mean": 2.0},
        },
        "views": {
            "ordered_teacher_forced": {"summary": teacher},
            "free_recurrence": {"summary": free},
        },
    }


def test_error_rows_keep_call_time_group_and_gap_contract() -> None:
    times = np.linspace(0.0, 4.9, 50)
    rows = visualizer._error_rows(_result(), times)
    assert len(rows) == 49
    assert rows[0]["call"] == 1
    assert rows[0]["physical_time_s"] == pytest.approx(0.1)
    assert rows[0]["free_minus_teacher_npe"] == pytest.approx(0.0)
    assert rows[2]["free_over_teacher_npe"] == pytest.approx(2.0)
    assert rows[-1]["teacher_p_npe"] == pytest.approx(0.02)


def test_event_helpers_find_first_false_without_treating_later_recovery_as_prefix() -> (
    None
):
    events = [
        {"admissible": True},
        {"admissible": False},
        {"admissible": True},
    ]
    assert visualizer._first_event(events, "admissible", expected=False) == 2
    assert visualizer._first_event(events, "admissible", expected=True) == 1
    assert visualizer._first_threshold([1.0, 1.9, 2.1], 2.0, below=False) == 3
    assert visualizer._first_threshold([0.9, 0.7], 0.8, below=True) == 2


def test_orientation_restores_increasing_physical_axes() -> None:
    values = np.arange(6).reshape(2, 3)
    x = np.asarray([3.0, 2.0, 1.0])
    y = np.asarray([2.0, 1.0])
    oriented = visualizer._oriented(values, x, y)
    assert oriented.tolist() == [[5, 4, 3], [2, 1, 0]]


def test_final_manifest_verifier_hashes_exact_inventory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    payload = {
        "schema": "schema",
        "files": {"a.txt": sha256_file(tmp_path / "a.txt")},
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    verified = visualizer._verify_final_manifest(
        tmp_path,
        payload,
        schema="schema",
        expected_files=frozenset({"a.txt"}),
    )
    assert verified == payload["files"]
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        visualizer._verify_final_manifest(
            tmp_path,
            payload,
            schema="schema",
            expected_files=frozenset({"a.txt"}),
        )


def test_fixed_scale_gif_renders_every_released_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(visualizer, "VALIDATION_HORIZON", 2)
    truth = np.zeros((3, 4, 5), dtype=np.float32)
    teacher = truth.copy()
    free = truth.copy()
    teacher[1:] = 1.0
    free[1:] = 2.0
    arrays = {"truth": truth, "teacher": teacher, "free": free}
    summary_teacher = {
        "npe_total_by_call": [0.1, 0.2],
    }
    summary_free = {
        "npe_total_by_call": [0.1, 0.4],
    }
    result = {
        "views": {
            "ordered_teacher_forced": {"summary": summary_teacher},
            "free_recurrence": {"summary": summary_free},
        }
    }
    events = [
        {"admissible": True, "pMax_nondecreasing": True},
        {"admissible": False, "pMax_nondecreasing": False},
    ]
    path = tmp_path / "tiny.gif"
    record = visualizer.render_animation(
        "pMax",
        arrays,
        {
            "label": "pMax [MPa]",
            "state_min": 0.0,
            "state_max": 2.0,
            "error_abs_q99_5": 2.0,
        },
        result,
        events,
        events,
        np.asarray([0.0, 1.0, 2.0]),
        x=np.linspace(0.0, 1.0, 5),
        y=np.linspace(0.0, 1.0, 4),
        output_path=path,
        fps=2,
        dpi=50,
    )
    assert path.is_file()
    assert record["rendered_frame_count"] == 3
    assert record["all_released_frames_rendered"]
    assert record["sha256"] == sha256_file(path)


def test_one_step_residual_views_use_truth_input_not_accumulated_rollout() -> None:
    truth = np.stack(
        [
            np.zeros((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
            np.full((4, 5), 2.0, dtype=np.float32),
        ]
    )
    teacher = np.stack(
        [
            np.zeros((4, 5), dtype=np.float32),
            np.full((4, 5), 2.0, dtype=np.float32),
            np.full((4, 5), 2.5, dtype=np.float32),
        ]
    )
    free = np.stack(
        [
            np.zeros((4, 5), dtype=np.float32),
            np.full((4, 5), 100.0, dtype=np.float32),
            np.full((4, 5), 200.0, dtype=np.float32),
        ]
    )
    true_residual, predicted_residual, prediction_error = (
        visualizer._one_step_residual_views(
            {"truth": truth, "teacher": teacher, "free": free}
        )
    )
    assert np.array_equal(true_residual, np.ones((2, 4, 5), dtype=np.float32))
    assert np.array_equal(
        predicted_residual,
        np.stack(
            [
                np.full((4, 5), 2.0, dtype=np.float32),
                np.full((4, 5), 1.5, dtype=np.float32),
            ]
        ),
    )
    assert np.array_equal(
        prediction_error,
        np.stack(
            [
                np.ones((4, 5), dtype=np.float32),
                np.full((4, 5), 0.5, dtype=np.float32),
            ]
        ),
    )
    assert np.array_equal(prediction_error, teacher[1:] - truth[1:])
    assert np.array_equal(prediction_error, predicted_residual - true_residual)
    assert not np.array_equal(predicted_residual, teacher[1:] - teacher[:-1])
    assert not np.array_equal(predicted_residual, free[1:] - free[:-1])


def test_one_step_residual_gif_renders_every_truth_input_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(visualizer, "VALIDATION_HORIZON", 2)
    truth = np.zeros((3, 4, 5), dtype=np.float32)
    teacher = truth.copy()
    teacher[1] = 1.0
    teacher[2] = -2.0
    free = np.full_like(truth, 100.0)
    arrays = {"truth": truth, "teacher": teacher, "free": free}
    path = tmp_path / "one_step_residual.gif"
    record = visualizer.render_one_step_residual_animation(
        "T",
        arrays,
        {
            "label": "T [released units]",
            "state_min": 0.0,
            "state_max": 2.0,
            "error_abs_q99_5": 2.0,
        },
        np.asarray([0.0, 1.0, 2.0]),
        x=np.linspace(0.0, 1.0, 5),
        y=np.linspace(0.0, 1.0, 4),
        output_path=path,
        fps=2,
        dpi=50,
    )
    assert path.is_file()
    assert record["true_residual_definition"] == (
        "decoded_reference[n+1]-decoded_reference[n]"
    )
    assert record["predicted_residual_definition"] == (
        "decoded_truth_input_prediction[n+1]-decoded_reference[n]"
    )
    assert record["prediction_error_definition"] == (
        "decoded_truth_input_prediction[n+1]-decoded_reference[n+1]"
    )
    assert record["prediction_error_equals"] == ("predicted_residual-true_residual")
    assert record["accumulated_rollout_error_visualized"] is False
    assert record["residual_space"] == "decoded_released_units"
    assert record["network_raw_normalized_residual_visualized"] is False
    assert record["rendered_frame_count"] == 2
    assert record["all_truth_input_pairs_rendered"]
    assert record["sha256"] == sha256_file(path)


def test_loss_rows_align_training_and_validation_at_each_checkpoint() -> None:
    history = {
        "rows": [
            {
                "completed_step": 50,
                "phase": "one_call",
                "calls": 1,
                "interval_mean_train_grouped_loss": 0.25,
                "learning_rate_used": 1.0e-3,
                "strict_improvement": False,
                "validation": {
                    "realm_npe_mean": 0.5,
                    "decoded_correlation_case_first": 0.8,
                },
            }
        ]
    }
    rows = visualizer._loss_rows(history)
    assert rows == [
        {
            "optimizer_step": 50,
            "phase": "one_call",
            "calls": 1,
            "interval_train_grouped_loss": 0.25,
            "truth_input_validation_grouped_npe": 0.5,
            "validation_minus_train": 0.25,
            "validation_over_train": 2.0,
            "learning_rate_used": 1.0e-3,
            "decoded_validation_correlation": 0.8,
            "strict_validation_improvement": False,
        }
    ]


def test_visualizer_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        visualizer.main(["--help"])
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_analysis_summary_is_json_serializable() -> None:
    payload = {
        "schema": visualizer.VISUALIZATION_SCHEMA,
        "fields": visualizer._validate_fields(["pMax", "T", "H2O"]),
    }
    json.dumps(payload, allow_nan=False)
