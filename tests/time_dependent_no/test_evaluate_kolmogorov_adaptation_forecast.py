import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_adaptation_forecast as experiment


def test_affine_candidate_forecast_matches_independent_composition():
    matrices = torch.tensor([[[.8, 1.5], [0., .7]], [[.9, 0.], [.8, .6]],
                             [[.7, -.5], [.3, .8]]], dtype=torch.float64)
    offsets = torch.tensor([[.02, -.01], [-.01, .03], [.04, .01]], dtype=torch.float64)
    baseline = torch.tensor([[.3, -.2], [.2, -.1], [.1, .1], [.2, .2]], dtype=torch.float64)
    candidate = lambda n, x: matrices[n] @ x + offsets[n]
    result = experiment.forecast_on_baseline(candidate, baseline, baseline)
    actual = [baseline[0]]
    for n in range(3):
        actual.append(matrices[n] @ actual[-1] + offsets[n])
    torch.testing.assert_close(baseline + result["displacement"], torch.stack(actual),
                               rtol=1e-13, atol=1e-13)
    assert result["curvature_estimate"].abs().max() < 1e-13
    assert not torch.allclose(result["displacement"], result["identity_displacement"])


def test_nonlinear_probes_do_not_change_linear_forecast():
    baseline = torch.zeros(4, 1, dtype=torch.float64)
    result = experiment.forecast_on_baseline(lambda n, x: x+x.square()+.1, baseline, baseline)
    torch.testing.assert_close(result["displacement"],
                               torch.tensor([[0.], [.1], [.2], [.3]], dtype=torch.float64))
    torch.testing.assert_close(result["curvature_estimate"],
                               torch.tensor([[0.], [0.], [.01], [.05]], dtype=torch.float64))
    for key in ("plus_remainder", "minus_remainder"):
        torch.testing.assert_close(result[key], torch.tensor([[0.], [.01], [.04]], dtype=torch.float64))
    assert all(not v.requires_grad and v.grad_fn is None for v in result.values())


def test_metric_pools_all_predicted_states_and_excludes_initial_condition():
    baseline = torch.ones(3, 1, dtype=torch.float64)
    result = dict(displacement=torch.tensor([[0.], [3.], [4.]], dtype=torch.float64),
                  curvature_estimate=torch.zeros_like(baseline),
                  linear_response=torch.ones(2, 1), plus_remainder=torch.zeros(2, 1),
                  minus_remainder=torch.zeros(2, 1), ad_no_grad_offset=torch.zeros(2, 1))
    row = experiment.case_metrics(baseline, baseline, result, 1.)
    assert row["predicted_sse"] == 25.
    assert row["target_sse"] == 2.
    pooled = experiment.summarize([row])
    assert pooled["predicted_relative_l2"] == pytest.approx(np.sqrt(12.5))


def test_large_remainder_or_propagated_curvature_disqualifies_range():
    row = dict(predicted_sse=.0025, base_sse=.01, curvature_sse=0., target_sse=1.,
               max_remainder_over_linear=.05, max_ad_offset_scaled=0.)
    result = experiment.summarize([row])
    assert result["locally_qualified"]
    assert result["range_relative_l2"] == pytest.approx([.045, .055])
    assert not experiment.summarize([dict(row, max_remainder_over_linear=.11)])["locally_qualified"]
    assert not experiment.summarize([dict(row, curvature_sse=.0001)])["locally_qualified"]


def test_snapshot_binding_rejects_changed_reference_and_missing_steps(tmp_path):
    reference = np.ones((9, 2, 2), dtype=np.float32)
    path = tmp_path / "snapshot.npz"
    np.savez(path, step=np.arange(9), reference=reference, state=reference)
    digest = experiment.evaluation.sha256(path)
    np.testing.assert_array_equal(experiment.load_baseline(path, digest, reference), reference)
    with pytest.raises(ValueError, match="reference"):
        experiment.load_baseline(path, digest, reference+1)
    with pytest.raises(ValueError, match="hash"):
        experiment.load_baseline(path, "0"*64, reference)
    np.savez(path, step=np.arange(8), reference=reference[:8], state=reference[:8])
    with pytest.raises(ValueError, match="steps"):
        experiment.load_baseline(path, experiment.evaluation.sha256(path), reference)
