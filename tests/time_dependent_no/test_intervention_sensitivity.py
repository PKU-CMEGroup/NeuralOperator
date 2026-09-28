import pytest
import torch

from utility.time_dependent_no.intervention_sensitivity import (
    blend_sensitivity, local_blend_remainder,
)
from utility.time_dependent_no import intervention_sensitivity as intervention


def test_time_ordered_first_variation_matches_parameter_autograd():
    matrices = torch.tensor([[[.8, 1.5], [0., .7]], [[.9, 0.], [.8, .6]],
                             [[.7, -.5], [.3, .8]]], dtype=torch.float64)
    alternative = matrices + torch.tensor([[.1, .2], [-.1, .05]], dtype=torch.float64)
    offset = torch.tensor([.12, -.07], dtype=torch.float64)
    initial = torch.tensor([.3, -.2], dtype=torch.float64)
    baseline = [initial]
    for matrix in matrices:
        baseline.append(matrix @ baseline[-1])
    baseline = torch.stack(baseline)
    calls = []

    def first(n, x):
        torch.testing.assert_close(x, baseline[n], atol=0, rtol=0)
        calls.append(("base", n))
        return matrices[n] @ x

    def other(n, x):
        torch.testing.assert_close(x, baseline[n], atol=0, rtol=0)
        calls.append(("other", n))
        return alternative[n] @ x + offset

    result = blend_sensitivity(first, other, baseline)

    def rollout(alpha):
        values = [initial]
        for matrix, changed in zip(matrices, alternative):
            x = values[-1]
            values.append((1-alpha)*(matrix@x) + alpha*(changed@x+offset))
        return torch.stack(values)

    expected = torch.autograd.functional.jacobian(rollout, torch.tensor(0., dtype=torch.float64))
    torch.testing.assert_close(result["sensitivity"], expected, atol=1e-13, rtol=1e-13)
    assert len(calls) == 6
    assert not torch.allclose(result["sensitivity"], result["identity_response_sensitivity"])
    assert torch.count_nonzero(result["baseline_replay_residual"]) == 0


def test_replay_discrepancy_is_not_injected_as_intervention():
    baseline = torch.tensor([[1.], [1.2], [1.3]], dtype=torch.float64)
    result = blend_sensitivity(lambda n, x: x, lambda n, x: x, baseline)
    assert torch.count_nonzero(result["sensitivity"]) == 0
    torch.testing.assert_close(result["baseline_replay_residual"], baseline[:-1]-baseline[1:])


def test_sparse_remainder_retains_state_curvature_and_map_change():
    x = torch.tensor([.2, -.3], dtype=torch.float64)
    z = torch.tensor([.4, .1], dtype=torch.float64)
    base = lambda n, value: value + value.square()
    other = lambda n, value: 2*value
    dz = (1+2*x)*z + other(0, x)-base(0, x)
    result = local_blend_remainder(base, other, 2, x, z, dz, [.1, .05])
    for i, a in enumerate(result["coefficients"]):
        for j, sign in enumerate((1, -1)):
            expected = a*a*(z.square()+(1-2*x)*z)-sign*a**3*z.square()
            torch.testing.assert_close(result["remainder"][i, j], expected, atol=1e-14, rtol=1e-12)


def test_no_grad_preserves_parameters_and_does_not_accumulate_time_graph():
    layer = torch.nn.Linear(2, 2).double()
    for p in layer.parameters():
        p.grad = torch.full_like(p, 7.)
    before = [(p.detach().clone(), p.grad.clone()) for p in layer.parameters()]
    values = [torch.ones(2, dtype=torch.float64)]
    with torch.no_grad():
        for _ in range(3):
            values.append(layer(values[-1]))
        result = blend_sensitivity(lambda n, x: layer(x), lambda n, x: .5*x, torch.stack(values))
    assert all(not v.requires_grad and v.grad_fn is None for v in result.values())
    for p, (weight, grad) in zip(layer.parameters(), before):
        torch.testing.assert_close(p, weight, atol=0, rtol=0)
        torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)


@pytest.mark.parametrize("coefficients", [[], [0.], [-1.], [float("nan")]])
def test_invalid_coefficients_rejected(coefficients):
    x = torch.ones(2)
    with pytest.raises(ValueError, match="coefficients"):
        local_blend_remainder(lambda n, x: x, lambda n, x: x, 0, x, x, x, coefficients)


def test_prefix_displacement_uses_time_ordered_response_without_new_forcing():
    # Wrong temporal index or injecting baseline replay error changes these values.
    matrices = torch.tensor([[[1., 2.], [0., 1.]], [[1., 0.], [3., 1.]]], dtype=torch.float64)
    baseline = torch.zeros(3, 2, dtype=torch.float64)
    initial = torch.tensor([.1, .2], dtype=torch.float64)
    result = intervention.initial_displacement_response(
        lambda n, x: matrices[n-8]@x + .01, baseline, initial, start_index=8)
    torch.testing.assert_close(result["displacement"],
        torch.tensor([[.1, .2], [.5, .2], [.5, 1.7]], dtype=torch.float64))
    torch.testing.assert_close(result["baseline_replay_residual"], torch.full((2, 2), .01, dtype=torch.float64))
    assert result["plus_remainder"].abs().max() < 1e-14
    assert result["curvature_estimate"].abs().max() < 1e-14


def test_curvature_probes_do_not_feed_nonlinear_outputs_back_into_forecast():
    # With base zero and F(x)=x+x^2, J=1: z stays .1 and q grows by .01.
    # An accidental autonomous nonlinear recurrence changes both sequences.
    baseline = torch.zeros(4, 1, dtype=torch.float64)
    result = intervention.initial_displacement_response(
        lambda n, x: x+x.square(), baseline, torch.tensor([.1], dtype=torch.float64))
    torch.testing.assert_close(result["displacement"], torch.full((4, 1), .1, dtype=torch.float64))
    torch.testing.assert_close(result["curvature_estimate"],
                               torch.tensor([[0.], [.01], [.02], [.03]], dtype=torch.float64))
    torch.testing.assert_close(result["plus_remainder"], torch.full((3, 1), .01, dtype=torch.float64))
    torch.testing.assert_close(result["minus_remainder"], torch.full((3, 1), .01, dtype=torch.float64))
    assert all(not v.requires_grad and v.grad_fn is None for v in result.values())


def test_prefix_zero_displacement_has_zero_response():
    result = intervention.initial_displacement_response(
        lambda n, x: x.square(), torch.ones(3, 2), torch.zeros(2))
    assert torch.count_nonzero(result["displacement"]) == 0
    assert torch.count_nonzero(result["curvature_estimate"]) == 0


def test_switch_uses_eight_prefix_calls_and_resets_between_cases():
    from scripts.time_dependent_no import evaluate_kolmogorov_prefix_switch as experiment
    def increment(amount):
        return lambda x: dict(raw_next=x+amount, next_state=x+amount)
    reference = torch.zeros(129, 8, 8).numpy()
    for _ in range(2):
        result, snapshots = experiment.switch_rollout(
            increment(1.), increment(10.), reference, 1., torch.device("cpu"))
        assert result["status"] == "completed"
        assert snapshots["state"][list(snapshots["step"]).index(8), 0, 0] == 8.
        assert snapshots["state"][-1, 0, 0] == 1208.


def test_forecast_selection_requires_useful_range_and_small_remainder():
    from scripts.time_dependent_no import evaluate_kolmogorov_prefix_switch as experiment
    result = experiment.forecast_decision(0.0025, 0.01, 0., 1., .05)
    assert result["selected"]
    assert result["predicted_relative_l2"] == pytest.approx(.05)
    assert result["range_relative_l2"] == pytest.approx([.045, .055])
    assert not experiment.forecast_decision(0.0025, 0.01, 0., 1., .11)["selected"]
    assert not experiment.forecast_decision(0.0025, 0.01, .001, 1., .05)["selected"]
