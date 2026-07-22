from __future__ import annotations

import numpy as np

from scripts.time_dependent_no.probe_cpg_boundary_sensitivity import (
    counterfactual_masks,
    lagged_boundary_state,
    positive_scale_boundary_state,
    sensitivity_metrics,
)


def test_counterfactual_masks_separate_support_and_endpoint_cone():
    injection = np.array([True, True, True, False, False])
    dependency = np.array([False, True, True, True, False])
    cone = np.array([True, False, True, True, False])

    masks = counterfactual_masks(injection, dependency, cone)

    np.testing.assert_array_equal(
        masks["support_injected"], np.array([False, True, True, False, False])
    )
    np.testing.assert_array_equal(
        masks["endpoint_cone_injected"], np.array([True, False, True, False, False])
    )
    np.testing.assert_array_equal(
        masks["support_outside_endpoint_cone"],
        np.array([False, True, False, False, False]),
    )


def test_lagged_boundary_replaces_only_selected_nodes():
    official = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0
    current = official + 100.0
    mask = np.array([False, True, False, True])

    result = lagged_boundary_state(official, current, mask)

    np.testing.assert_array_equal(result[mask], current[mask])
    np.testing.assert_array_equal(result[~mask], official[~mask])


def test_positive_scale_probe_preserves_velocity_and_positivity():
    state = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0]], dtype=np.float64)
    mask = np.array([True, False])

    result = positive_scale_boundary_state(state, mask, amplitude=0.01)

    np.testing.assert_allclose(result[0, 0], 1.01)
    np.testing.assert_allclose(result[0, 3], 4.0 * 1.014)
    np.testing.assert_array_equal(result[:, 1:3], state[:, 1:3])
    np.testing.assert_array_equal(result[1], state[1])


def test_sensitivity_metrics_report_target_and_normal_response():
    baseline = np.ones((3, 4), dtype=np.float64)
    candidate = baseline.copy()
    candidate[1, 3] = 3.0
    truth = baseline.copy()
    official_input = baseline.copy()
    candidate_input = baseline.copy()
    candidate_input[0, 0] = 2.0

    metrics = sensitivity_metrics(
        baseline_output=baseline,
        candidate_output=candidate,
        target_truth=truth,
        target_node=1,
        normal_mask=np.array([False, True, True]),
        official_input=official_input,
        candidate_input=candidate_input,
        changed_mask=np.array([True, False, False]),
    )

    assert metrics["changed_node_count"] == 1
    assert metrics["target_output_delta_l2"] == 2.0
    assert metrics["candidate_target_error_l2"] == 2.0
    assert metrics["target_error_l2_change"] == 2.0
    assert metrics["target_output_delta_pres"] == 2.0
    assert metrics["input_delta_rmse_rho"] == 1.0
