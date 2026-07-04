import numpy as np

from utility.time_dependent_no.euler2d_metrics import (
    boundary_leakage_metrics,
    conservation_drift,
    conservation_totals,
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    front_region_masks,
    local_shift_alignment_metrics,
    median_edge_length,
    near_shock_error,
    node_variation_score,
    positivity_metrics,
    rollout_relative_l2_by_time,
    rollout_rmse_by_time,
    shift_grid,
    shock_centroid_distance,
    shock_front_masks,
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


def test_front_overlap_distance_and_regions_track_shifted_pressure_jump():
    truth = np.ones((1, 6, 4), dtype=np.float64)
    prediction = truth.copy()
    truth[0, :, 3] = np.array([1.0, 1.0, 4.0, 4.0, 4.0, 4.0])
    prediction[0, :, 3] = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
    normal = np.array([True, True, True, True, True, False])
    positions = np.stack((np.arange(6, dtype=np.float64), np.zeros(6)), axis=-1)

    fronts = shock_front_masks(
        prediction,
        truth,
        edges,
        quantile=0.8,
        node_mask=normal,
    )
    overlap = front_overlap_metrics(fronts["prediction_mask"], fronts["target_mask"])
    distances = front_distance_metrics(
        fronts["prediction_mask"], fronts["target_mask"], positions
    )
    centroid = front_centroid_distance(
        fronts["prediction_mask"], fronts["target_mask"], positions
    )
    regions = front_region_masks(
        fronts["prediction_mask"], fronts["target_mask"], node_mask=normal
    )

    np.testing.assert_allclose(overlap["iou"], np.array([1.0 / 3.0]))
    np.testing.assert_allclose(overlap["f1"], np.array([0.5]))
    np.testing.assert_allclose(distances["symmetric_chamfer_mean"], np.array([0.5]))
    np.testing.assert_allclose(centroid, np.array([1.0]))
    assert np.count_nonzero(regions["front_overlap"]) == 1
    assert np.count_nonzero(regions["predicted_front_only"]) == 1
    assert np.count_nonzero(regions["target_front_only"]) == 1
    assert not np.any(regions["smooth"][..., ~normal])


def test_local_shift_alignment_reduces_displaced_front_error():
    positions = np.stack((np.arange(6, dtype=np.float64), np.zeros(6)), axis=-1)
    truth = np.ones((1, 6, 4), dtype=np.float64)
    prediction = truth.copy()
    truth[0, :, 3] = np.array([1.0, 1.0, 4.0, 4.0, 4.0, 4.0])
    prediction[0, :, 3] = np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
    region = np.ones((1, 6), dtype=bool)
    shifts = shift_grid(1.0, grid_size=3, dim=2)

    metrics = local_shift_alignment_metrics(
        prediction,
        truth,
        positions,
        region,
        shifts,
        scalar_index=3,
    )

    assert metrics["best_shift_rmse"][0] < metrics["baseline_rmse"][0]
    assert metrics["relative_rmse_reduction"][0] > 0.0
    np.testing.assert_allclose(metrics["best_shift"][0, 0], -1.0)


def test_median_edge_length_uses_physical_positions():
    positions = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [6.0, 3.0]], dtype=np.float64
    )
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)

    assert median_edge_length(positions, edges) == 3.0
