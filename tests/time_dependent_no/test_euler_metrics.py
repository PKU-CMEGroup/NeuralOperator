import numpy as np

from utility.time_dependent_no.euler2d_metrics import (
    boundary_leakage_metrics,
    conservation_drift,
    conservation_totals,
    near_shock_error,
    node_variation_score,
    positivity_metrics,
    rollout_relative_l2_by_time,
    rollout_rmse_by_time,
    shock_centroid_distance,
    shock_indicator,
    shock_region_metrics,
    summarize_euler2d_rollout,
    valid_prediction_time,
)


def _primitive_sequence():
    sequence = np.ones((3, 5, 4), dtype=np.float64)
    sequence[..., 1] = 0.2
    sequence[..., 2] = 0.0
    sequence[..., 3] = np.array([1.0, 1.0, 2.0, 2.0, 2.0])[None, :]
    return sequence


def _edges():
    return np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)


def test_rollout_errors_and_valid_prediction_time_are_time_resolved():
    truth = _primitive_sequence()
    prediction = truth.copy()
    prediction[1, :, 3] += 0.1
    prediction[2, :, 3] += 0.2

    rmse = rollout_rmse_by_time(prediction, truth)
    rel = rollout_relative_l2_by_time(prediction, truth)
    vpt = valid_prediction_time(rel, threshold=0.05, dt=0.25)

    assert rmse.shape == (3,)
    assert rel.shape == (3,)
    assert rmse[0] == 0.0
    assert rmse[2] > rmse[1] > 0.0
    assert vpt == 0.25


def test_positivity_metrics_detect_density_and_pressure_failures():
    primitive = _primitive_sequence()
    primitive[1, 0, 0] = 0.0
    primitive[2, 1, 3] = -0.5

    metrics = positivity_metrics(primitive)

    assert not metrics["all_positive"]
    assert metrics["num_nonpositive_density"] == 1
    assert metrics["num_nonpositive_pressure"] == 1
    assert metrics["min_pressure"] == -0.5


def test_conservation_totals_and_drift_use_euler_energy():
    sequence = _primitive_sequence()
    totals = conservation_totals(sequence[0])
    drift_zero = conservation_drift(sequence)
    changed = sequence.copy()
    changed[2, :, 3] += 0.5
    drift_changed = conservation_drift(changed)

    assert totals.shape == (4,)
    np.testing.assert_allclose(drift_zero["relative_drift"], 0.0)
    assert drift_changed["max_relative_by_variable"]["energy"] > 0.0


def test_node_variation_and_shock_indicator_find_pressure_jump():
    primitive = _primitive_sequence()[0]

    scores = node_variation_score(primitive[..., 3], _edges())
    mask = shock_indicator(primitive, _edges(), quantile=0.75)
    metrics = shock_region_metrics(primitive, _edges(), quantile=0.75)

    assert scores[1] > 0.0
    assert scores[2] > 0.0
    assert mask[1]
    assert mask[2]
    assert metrics["shock_fraction"] > 0.0
    assert metrics["max_variation_score"] == scores.max()


def test_near_shock_error_separates_shock_and_smooth_nodes():
    truth = _primitive_sequence()[0]
    prediction = truth.copy()
    shock_mask = np.array([False, True, True, False, False])
    prediction[shock_mask, 3] += 1.0

    errors = near_shock_error(prediction, truth, shock_mask)

    assert errors["shock_rmse"] > 0.0
    assert errors["smooth_rmse"] == 0.0
    assert errors["shock_relative_l2"] > errors["smooth_relative_l2"]


def test_shock_centroid_distance_tracks_shifted_jump():
    truth = _primitive_sequence()[0]
    prediction = truth.copy()
    prediction[:, 3] = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    positions = np.stack((np.arange(5, dtype=np.float64), np.zeros(5)), axis=-1)

    distance = shock_centroid_distance(prediction, truth, positions, _edges())

    assert distance > 0.0


def test_boundary_leakage_uses_non_normal_nodes_only():
    truth = _primitive_sequence()[0]
    prediction = truth.copy()
    node_type = np.array([0, 1, 2, 3, 0])
    prediction[1:4, 0] += 0.5

    metrics = boundary_leakage_metrics(prediction, truth, node_type)

    assert metrics["num_boundary_nodes"] == 3
    assert metrics["boundary_rmse"] > 0.0
    assert metrics["boundary_max_abs"] == 0.5


def test_summarize_euler2d_rollout_returns_scalar_diagnostics():
    truth = _primitive_sequence()
    prediction = truth.copy()
    prediction[2, 1:3, 3] += 0.2
    node_type = np.array([0, 1, 2, 3, 0])
    positions = np.stack((np.arange(5, dtype=np.float64), np.zeros(5)), axis=-1)

    summary = summarize_euler2d_rollout(
        prediction,
        truth,
        node_type=node_type,
        edges=_edges(),
        positions=positions,
    )

    assert summary["mean_rmse"] > 0.0
    assert summary["all_density_pressure_positive"]
    assert summary["num_boundary_nodes"] == 3
    assert "max_relative_energy_drift" in summary
    assert "target_final_shock_fraction" in summary

