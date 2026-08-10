from __future__ import annotations

import math

import torch

from utility.time_dependent_no.pcno_inadmissibility import (
    inadmissibility_episodes,
    stage_transition_label,
    state_diagnostics,
    trajectory_event_summary,
    weighted_relative_l2,
    weighted_rms,
)


def _state() -> torch.Tensor:
    return torch.tensor(
        [[[1.0, 0.0, 0.0, 2.5], [2.0, 0.0, 0.0, 5.0], [1.0, 1.0, 0.0, 3.0]]]
    )


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(1, 3, 1), torch.ones(1, 3, 1)


def test_weighted_norms_use_proxy_mass_and_component_scale() -> None:
    target = _state()
    prediction = target.clone()
    prediction[0, 0, 0] += 2.0
    weights, mask = _geometry()
    scale = torch.tensor([[[2.0, 1.0, 1.0, 1.0]]])

    rms = weighted_rms(
        prediction - target,
        weights,
        mask,
        component_scale=scale,
    )
    relative = weighted_relative_l2(prediction, target, weights, mask, scale)

    assert math.isclose(rms, math.sqrt(1.0 / 12.0), rel_tol=1.0e-12)
    expected_denominator = (
        (1.0 / 2.0) ** 2
        + (2.0 / 2.0) ** 2
        + (1.0 / 2.0) ** 2
        + 2.5**2
        + 5.0**2
        + 1.0**2
        + 3.0**2
    )
    assert math.isclose(relative, math.sqrt(1.0 / expected_denominator), rel_tol=1e-12)


def test_state_diagnostics_keeps_finite_negative_internal_energy_distinct() -> None:
    target = _state()
    state = target.clone()
    state[0, 1] = torch.tensor([1.0, 2.0, 0.0, 1.0])
    weights, mask = _geometry()
    metrics, invalid = state_diagnostics(
        state,
        target=target,
        node_weights=weights,
        node_mask=mask,
        state_mean=torch.zeros(1, 1, 4),
        state_scale=torch.ones(1, 1, 4),
        reference_max_abs=5.0,
        gamma=1.4,
    )

    assert metrics["failure_cause"] == "nonpositive_internal_energy"
    assert metrics["finite_component_fraction"] == 1.0
    assert metrics["invalid_node_count"] == 1
    assert metrics["nonpositive_density_node_count"] == 0
    assert metrics["nonpositive_internal_energy_node_count"] == 1
    assert math.isclose(metrics["min_internal_energy"], -1.0)
    assert invalid.tolist() == [[False, True, False]]


def test_state_diagnostics_labels_nonfinite_state_without_fabricated_error() -> None:
    target = _state()
    state = target.clone()
    state[0, 2, 3] = float("nan")
    weights, mask = _geometry()
    metrics, invalid = state_diagnostics(
        state,
        target=target,
        node_weights=weights,
        node_mask=mask,
        state_mean=torch.zeros(1, 1, 4),
        state_scale=torch.ones(1, 1, 4),
        reference_max_abs=5.0,
        gamma=1.4,
    )

    assert metrics["failure_cause"] == "nonfinite_state"
    assert metrics["proxy_scaled_relative_l2"] is None
    assert metrics["proxy_physical_error_rms"] is None
    assert metrics["finite_component_fraction"] == 11.0 / 12.0
    assert invalid.tolist() == [[False, False, True]]


def test_stage_transition_labels_attribute_recovery_to_the_observed_stage() -> None:
    assert (
        stage_transition_label("nonpositive_pressure", "admissible", stage="model")
        == "model_recovery"
    )
    assert (
        stage_transition_label("admissible", "nonpositive_pressure", stage="model")
        == "model_introduced_inadmissibility"
    )
    assert (
        stage_transition_label(
            "nonpositive_pressure",
            "nonpositive_pressure",
            stage="output_projection",
        )
        == "output_projection_persistent_inadmissibility"
    )


def test_inadmissibility_episodes_include_transient_and_terminal_intervals() -> None:
    episodes = inadmissibility_episodes([False, True, True, False, True])

    assert episodes == [
        {
            "start_call": 2,
            "end_call": 3,
            "duration_calls": 2,
            "recovered": True,
            "recovery_call": 4,
        },
        {
            "start_call": 5,
            "end_call": 5,
            "duration_calls": 1,
            "recovered": False,
            "recovery_call": None,
        },
    ]


def _call_row(call: int, failure: str, ratio: float, error: float) -> dict:
    return {
        "call": call,
        "deployed_failure_cause": failure,
        "deployed_max_abs_to_reference_max_ratio": ratio,
        "deployed_proxy_scaled_relative_l2": error,
        "input_projection_transition": "input_projection_admissible",
        "model_transition": "model_admissible",
        "output_projection_transition": "output_projection_admissible",
    }


def test_trajectory_summary_does_not_equate_recovered_invalidity_with_blowup() -> None:
    rows = [
        _call_row(1, "admissible", 1.0, 0.1),
        _call_row(2, "nonpositive_internal_energy", 1.1, 0.2),
        _call_row(3, "admissible", 1.2, 0.3),
    ]
    rows[2]["model_transition"] = "model_recovery"

    summary = trajectory_event_summary(rows)

    assert summary["ever_inadmissible"] is True
    assert summary["first_inadmissible_call"] == 2
    assert summary["recovered_episode_count"] == 1
    assert summary["terminal_admissible"] is True
    assert summary["registered_blowup"] is False
    assert summary["model_recovery_calls"] == 1


def test_trajectory_summary_records_diagnostic_blowup_lag() -> None:
    rows = [
        _call_row(1, "nonpositive_pressure", 1.0, 0.1),
        _call_row(2, "nonpositive_pressure", 101.0, 0.2),
    ]

    summary = trajectory_event_summary(rows)

    assert summary["registered_blowup"] is True
    assert summary["first_100x_reference_amplitude_call"] == 2
    assert summary["lag_first_inadmissible_to_blowup"] == 1
