import pytest

from scripts.time_dependent_no import analyze_euler1d_stride_attribution as analysis


def test_first_event_respects_direction_and_frame_order():
    rows = [
        {"target_frame": 3, "value": "0.3"},
        {"target_frame": 1, "value": "0.1"},
        {"target_frame": 2, "value": "0.2"},
    ]

    assert analysis._first_event(rows, "value", 0.15, relation="above") == 2
    assert analysis._first_event(rows, "value", 0.15, relation="below") == 1
    assert analysis._first_event(rows, "value", 2.0, relation="above") is None


def test_batch_presentation_audit_separates_small_metric_drift_from_event_shift():
    full = [
        {
            "case_id": "7",
            "stride": "1",
            "frame": "8",
            "fixed_scale_conservative_relative_l2": "0.1",
        }
    ]
    batch1 = [
        {
            "case_id": "7",
            "model": "s1",
            "mode": "autoregressive",
            "target_frame": "8",
            "proposal_valid": "True",
            "state_cons_scaled_rel_l2": "0.101",
        },
        {
            "case_id": "7",
            "model": "s1",
            "mode": "autoregressive",
            "target_frame": "10",
            "proposal_valid": "False",
            "state_cons_scaled_rel_l2": "0.2",
        },
    ]
    terminations = [
        {
            "case_id": "7",
            "stride": "1",
            "termination_frame": "9",
        }
    ]

    result = analysis.batch_presentation_audit(full, batch1, terminations)

    assert result["common_metric_rows"] == 1
    assert result["absolute_difference_max"] == pytest.approx(0.001)
    assert result["failure_frame_mismatches"] == [
        {
            "case_id": 7,
            "stride": 1,
            "full_batch_failure_frame": 9,
            "batch1_failure_frame": 10,
        }
    ]
