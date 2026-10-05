from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_local_defects as runner


class AffineMap(torch.nn.Module):
    def __init__(self, multiplier, offset, batch_offset=0.0):
        super().__init__()
        self.multiplier = multiplier
        self.offset = offset
        self.batch_offset = batch_offset
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().cpu().numpy().copy())
        nxt = self.multiplier * value + self.offset
        if len(value) > 1:
            nxt = nxt + self.batch_offset
        return dict(next_state=nxt, raw_next=nxt + 100)


def truth_path():
    initial = np.array([[1, 2], [3, 4]], dtype=np.float32)
    return np.stack([initial * np.float32(0.5**n) for n in range(5)])


def test_fixed_scale_is_spatial_mean_square_not_state_relative_error():
    assert runner.mse(np.array([[1, 3], [-1, -3]]), 2) == pytest.approx(1.25)
    assert runner.mse(np.ones((2, 2)), 2) == runner.mse(np.ones((4, 4)), 2)
    assert runner.mse(np.array([2.0, 6.0]), 4) == runner.mse(np.array([1.0, 3.0]), 2)


def test_exact_error_decomposition_retains_cancellation():
    clean_next = np.array([[1., 2.], [3., 4.]])
    flow_target = clean_next + np.array([[1., -2.], [3., -4.]])
    row = runner.error_terms(clean_next, flow_target, clean_next, scale=2)
    assert row["local_mse"] == pytest.approx(1.875)
    assert row["propagated_mse"] == pytest.approx(1.875)
    assert row["cross_term"] == pytest.approx(-3.75)
    assert row["next_error_mse"] == 0
    assert row["next_error_mse"] == pytest.approx(
        row["local_mse"] + row["propagated_mse"] + row["cross_term"])


def test_rollout_uses_its_own_deployed_next_state_and_only_initial_truth():
    truth = truth_path()
    altered = truth.copy()
    altered[1:] = -1000
    model = AffineMap(2, 1)
    states = runner.rollout(model, altered, "cpu")
    assert states.shape == truth.shape
    for n, received in enumerate(model.inputs):
        np.testing.assert_array_equal(received[0], states[n])
        np.testing.assert_array_equal(states[n + 1], 2 * states[n] + 1)
    np.testing.assert_array_equal(states, runner.rollout(AffineMap(2, 1), truth, "cpu"))
    np.testing.assert_array_equal(states[0], truth[0])


def test_common_inputs_keep_model_and_rollout_source_axes_distinct():
    truth = truth_path()
    models = [AffineMap(0.5, 1), AffineMap(1, 2)]
    reached = np.stack([runner.rollout(model, truth, "cpu") for model in models])
    labels = 0.5 * reached[:, :-1].astype(np.float64)
    for model in models:
        model.inputs.clear()
    clean, local, accumulated, propagated, cross = runner.score_paths(
        models, truth, reached, labels, scale=2, device="cpu")
    assert clean.shape == (2, 4)
    assert local.shape == (2, 2, 4)
    assert accumulated.shape == (2, 5)
    assert propagated.shape == cross.shape == (2, 4)
    for n in range(4):
        for model in models:
            np.testing.assert_array_equal(model.inputs[n],
                                          np.stack([truth[n], reached[0, n], reached[1, n]]))
        for m, (multiplier, offset) in enumerate(((0.5, 1), (1, 2))):
            expected_clean = multiplier * truth[n].astype(np.float64) + offset - truth[n + 1]
            assert clean[m, n] == pytest.approx(np.mean(expected_clean**2) / 4)
            for source in range(2):
                expected_defect = (multiplier - 0.5) * reached[source, n].astype(np.float64) + offset
                assert local[m, source, n] == pytest.approx(np.mean(expected_defect**2) / 4)
    assert local[1, 0, 3] != local[1, 1, 3]
    np.testing.assert_array_equal(accumulated[:, 0], 0)
    np.testing.assert_allclose(accumulated[:, 1:],
                               local[np.arange(2), np.arange(2)] + propagated + cross)


def test_own_defect_uses_stored_autonomous_step_even_if_batch_evaluation_differs():
    truth = truth_path()
    models = [AffineMap(0.5, 1, batch_offset=0.25), AffineMap(1, 2, batch_offset=0.5)]
    reached = np.stack([runner.rollout(model, truth, "cpu") for model in models])
    labels = 0.5 * reached[:, :-1].astype(np.float64)
    _, local, accumulated, propagated, cross = runner.score_paths(
        models, truth, reached, labels, scale=2, device="cpu")
    for source in range(2):
        exact_own = np.mean((reached[source, 1:].astype(np.float64) - labels[source])**2,
                            axis=(-2, -1)) / 4
        np.testing.assert_allclose(local[source, source], exact_own)
    np.testing.assert_allclose(local[0, 1], 1.25**2 / 4)
    np.testing.assert_allclose(accumulated[:, 1:],
                               local[np.arange(2), np.arange(2)] + propagated + cross)


def test_rollout_error_is_subtracted_in_float64_for_the_same_error_identity():
    truth = truth_path() * np.float32(0.1)
    models = [AffineMap(0.5, 1), AffineMap(1, 2)]
    reached = np.stack([runner.rollout(model, truth, "cpu") for model in models])
    labels = 0.5 * reached[:, :-1].astype(np.float64)
    _, local, accumulated, propagated, cross = runner.score_paths(
        models, truth, reached, labels, scale=2, device="cpu")
    expected = np.mean((reached.astype(np.float64) - truth.astype(np.float64)[None])**2,
                       axis=(-2, -1)) / 4
    np.testing.assert_allclose(accumulated, expected, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(accumulated[:, 1:],
                               local[np.arange(2), np.arange(2)] + propagated + cross,
                               rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_normalization_rejects_undefined_scale(scale):
    with pytest.raises(ValueError):
        runner.mse(np.ones((2, 2)), scale)


def test_animation_distinguishes_accumulated_own_and_common_input_defects():
    from scripts.time_dependent_no.plot_kolmogorov_local_defects import animation_fields

    truth = truth_path().astype(np.float64)
    reached = np.stack([truth + 2, truth - 3])
    labels = np.stack([truth[:-1] + 5, truth[:-1] - 7])
    targets = np.stack([truth[1:], *labels])
    predictions = np.stack([targets + 4, targets - 6])
    views = animation_fields(truth, reached, labels, scale=2, predictions=predictions)
    np.testing.assert_array_equal(views["rollout"][3], 1)
    np.testing.assert_array_equal(views["rollout"][4], -1.5)
    np.testing.assert_array_equal(views["rollout"][5], -2.5)
    for n in range(4):
        np.testing.assert_array_equal(views["own"][0][n], (reached[0, n + 1] - labels[0, n]) / 2)
        np.testing.assert_array_equal(views["own"][1][n], (reached[1, n + 1] - labels[1, n]) / 2)
    for name in ("truth", "pretrained", "continued"):
        np.testing.assert_array_equal(views[name][0], 2)
        np.testing.assert_array_equal(views[name][1], -3)
        np.testing.assert_array_equal(views[name][2], -5)
    assert len(views["rollout"][0]) == 5 and len(views["own"][0]) == 4


def test_animation_orients_xy_fields_with_y_upwards_and_refuses_clipping():
    import base64
    import io
    import matplotlib
    from PIL import Image
    from scripts.time_dependent_no.plot_kolmogorov_local_defects import display_xy, _field_png

    field = np.array([[0., 1.], [2., 3.]])
    np.testing.assert_array_equal(display_xy(field), [[1, 3], [0, 2]])
    with pytest.raises(ValueError, match="clip"):
        _field_png(field, limit=2)
    encoded = _field_png(field, limit=3)
    pixels = np.array(Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1]))).convert("RGBA"))
    expected = matplotlib.colormaps["RdBu_r"]((display_xy(field) / 3 + 1) / 2, bytes=True)
    np.testing.assert_array_equal(pixels, expected)
