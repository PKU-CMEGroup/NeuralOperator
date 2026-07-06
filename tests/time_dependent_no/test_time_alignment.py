import numpy as np

from utility.time_dependent_no.time_alignment import compute_time_alignment, fit_best_time_curve


def _truth_sequence(num_steps: int) -> np.ndarray:
    steps = np.arange(num_steps, dtype=np.float64)
    nodes = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    values = steps[:, None, None] + 0.01 * nodes[None, :, None]
    return values


def test_time_alignment_recovers_fixed_delay():
    truth = _truth_sequence(12)
    pred_steps = np.arange(2, 10)
    prediction = truth[pred_steps - 2]
    target = truth

    alignment = compute_time_alignment(prediction, target, variables=[0])

    np.testing.assert_array_equal(alignment.best_target_indices, pred_steps - 2)
    fit = fit_best_time_curve(pred_steps, alignment.best_target_indices)
    assert np.isclose(fit["slope"], 1.0)
    assert np.isclose(fit["intercept"], -2.0)
    assert np.isclose(fit["median_lag"], -2.0)


def test_time_alignment_recovers_speed_mismatch():
    truth = _truth_sequence(20)
    pred_steps = np.arange(2, 18, 2)
    prediction = truth[pred_steps // 2]

    alignment = compute_time_alignment(prediction, truth, variables=[0])

    np.testing.assert_array_equal(alignment.best_target_indices, pred_steps // 2)
    fit = fit_best_time_curve(pred_steps, alignment.best_target_indices)
    assert np.isclose(fit["slope"], 0.5)
    assert np.isclose(fit["intercept"], 0.0)
