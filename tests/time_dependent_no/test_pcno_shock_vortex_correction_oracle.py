from __future__ import annotations

import pytest

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_correction_oracle import (
    ANTI_SMEARING_METRICS,
    metric_acceptance,
    metric_report_passed,
    promotion_decision,
)


def _metrics(value: float) -> dict[str, float]:
    return {name: value for name in ANTI_SMEARING_METRICS}


def _passing_row() -> dict[str, object]:
    return {
        "status": "complete",
        "state_error_reduction": 0.2,
        "smooth_highpass_error_reduction": 0.3,
        "state_error_nonworse": True,
        "smooth_highpass_error_nonworse": True,
        "raw_admissible": True,
        "conservation_pass": True,
        "support_cap_pass": True,
        "update_cap_pass": True,
        "anti_smearing_pass": True,
    }


def test_metric_acceptance_enforces_five_percent_without_zero_division() -> None:
    baseline = _metrics(2.0)
    accepted = metric_acceptance(baseline, _metrics(2.1))
    rejected = metric_acceptance(baseline, _metrics(2.10001))

    assert metric_report_passed(accepted)
    assert not metric_report_passed(rejected)
    zero = metric_acceptance(_metrics(0.0), _metrics(1.0e-6))
    assert not metric_report_passed(zero)


def test_promotion_requires_every_predeclared_gate() -> None:
    rows = [_passing_row() for _ in range(4)]
    decision = promotion_decision(rows, expected_rows=4)
    assert decision["passed"] is True
    assert decision["decision"].startswith("authorize_one")

    rows[0] = {**rows[0], "smooth_highpass_error_nonworse": False}
    rows[1] = {**rows[1], "smooth_highpass_error_nonworse": False}
    rejected = promotion_decision(rows, expected_rows=4)
    assert rejected["passed"] is False
    assert rejected["joint_state_and_smooth_nonworse_fraction"] == pytest.approx(0.5)


def test_promotion_counts_missing_or_zero_correction_rows_honestly() -> None:
    rows = [_passing_row(), {"status": "baseline_incomplete_before_endpoint"}]
    decision = promotion_decision(rows, expected_rows=2)
    assert decision["passed"] is False
    assert decision["gates"]["all_rows_available"] is False

    zero = {**_passing_row(), "state_error_reduction": 0.0}
    zero_decision = promotion_decision([zero], expected_rows=1)
    assert zero_decision["passed"] is False
    assert zero_decision["median_state_error_reduction"] == 0.0
