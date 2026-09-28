from __future__ import annotations

import pytest
import torch

from utility.time_dependent_no.forced_tangent import (
    decompose_forecast_error,
    finite_amplitude_response,
    forced_tangent_forecast,
)
from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO,
    canonicalize_vorticity,
)


def test_ordered_nonnormal_affine_forecast_matches_independent_rollout():
    matrices = torch.tensor(
        [
            [[0.8, 2.0], [0.0, 0.7]],
            [[0.9, 0.0], [1.5, 0.8]],
            [[0.7, -1.0], [0.0, 0.9]],
        ],
        dtype=torch.float64,
    )
    offsets = torch.tensor([[0.1, -0.2], [0.3, 0.1], [-0.2, 0.4]], dtype=torch.float64)
    reference = torch.tensor(
        [[0.2, -0.4], [0.3, -0.1], [0.1, 0.2], [-0.2, 0.3]],
        dtype=torch.float64,
    )
    calls = []

    def step(index, state):
        calls.append((index, state.detach().clone()))
        return matrices[index] @ state + offsets[index]

    result = forced_tangent_forecast(step, reference)
    actual = [reference[0]]
    for matrix, offset in zip(matrices, offsets, strict=True):
        actual.append(matrix @ actual[-1] + offset)
    torch.testing.assert_close(
        result["predicted_error"],
        torch.stack(actual) - reference,
        rtol=1e-13,
        atol=1e-13,
    )
    expected_forcing = (
        torch.einsum("tij,tj->ti", matrices, reference[:-1]) + offsets - reference[1:]
    )
    torch.testing.assert_close(result["clean_forcing"], expected_forcing)
    torch.testing.assert_close(
        result["identity_response_error"][1:], torch.cumsum(expected_forcing, dim=0)
    )
    assert not torch.allclose(
        result["predicted_error"], result["identity_response_error"]
    )
    assert not torch.allclose(matrices[1] @ matrices[0], matrices[0] @ matrices[1])
    assert len(calls) == 3
    for index, state in calls:
        torch.testing.assert_close(state, reference[index], rtol=0, atol=0)


def test_nonlinear_forecast_queries_clean_states_only_and_does_not_use_remainders():
    reference = torch.zeros(4, 1, dtype=torch.float64)
    calls = []

    def step(index, state):
        # Any exact rollout or displaced finite-secant construction would fail.
        torch.testing.assert_close(state, reference[index], rtol=0, atol=0)
        calls.append(index)
        return 0.2 + state + state.square()

    result = forced_tangent_forecast(step, reference)
    expected = torch.tensor([[0.0], [0.2], [0.4], [0.6]], dtype=torch.float64)
    torch.testing.assert_close(result["predicted_error"], expected)
    actual = torch.zeros(1, dtype=torch.float64)
    for _ in range(3):
        actual = 0.2 + actual + actual.square()
    assert actual.item() > result["predicted_error"][-1].item()
    assert calls == [0, 1, 2]


def test_outer_no_grad_keeps_jvp_and_preserves_parameters_inputs_and_gradients():
    layer = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.3, 0.4], [-0.2, 0.8]], dtype=torch.float64))
        layer.bias.fill_(0.1)
    for parameter in layer.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    parameters = [
        (p.detach().clone(), p.grad.clone(), p.requires_grad)
        for p in layer.parameters()
    ]
    reference = torch.tensor(
        [[0.2, -0.4], [0.1, 0.2], [0.3, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    before = reference.detach().clone()
    regular = forced_tangent_forecast(lambda n, x: layer(x), reference)
    with torch.no_grad():
        actual = forced_tangent_forecast(lambda n, x: layer(x), reference)
    for name, value in actual.items():
        torch.testing.assert_close(value, regular[name], rtol=0, atol=0)
        assert value.device == reference.device and value.dtype == reference.dtype
        assert not value.requires_grad and value.grad_fn is None
    torch.testing.assert_close(reference, before, rtol=0, atol=0)
    assert reference.grad is None
    for parameter, (value, gradient, requires_grad) in zip(
        layer.parameters(), parameters, strict=True
    ):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        assert parameter.requires_grad == requires_grad


def test_signed_quadratic_probe_separates_linear_odd_and_even_response():
    state = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.3, 0.6], dtype=torch.float64, requires_grad=True)
    calls = []

    def step(index, value):
        calls.append(index)
        return value + value.square() + 0.05 * index

    with torch.no_grad():
        result = finite_amplitude_response(step, 2, state, direction, [0.25, 0.5, 1.0])
    amplitude = result["multipliers"][:, None]
    expected_linear = amplitude * (1 + 2 * state) * direction
    expected_even = amplitude.square() * direction.square()
    for name in ("linear_response", "odd_response"):
        torch.testing.assert_close(
            result[name], expected_linear, rtol=1e-13, atol=1e-13
        )
    for name in ("even_response", "plus_remainder", "minus_remainder"):
        torch.testing.assert_close(result[name], expected_even, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(result["plus_displacement"], amplitude * direction)
    torch.testing.assert_close(result["minus_displacement"], -amplitude * direction)
    assert calls == [2] * 7
    assert all(
        not value.requires_grad and value.grad_fn is None for value in result.values()
    )
    assert state.grad is None and direction.grad is None


def test_actual_periodic_pcno_jvp_matches_centered_difference_with_shared_restriction():
    with torch.random.fork_rng():
        torch.manual_seed(43)
        model = PeriodicVorticityPCNO(
            16, train_scale=2.0, modes=2, width=4, depth=1, fc_dim=8
        ).double().eval()
        state = canonicalize_vorticity(torch.randn(1, 16, 16, dtype=torch.float64))
        direction = canonicalize_vorticity(torch.randn_like(state))
    parameters = [p.detach().clone() for p in model.parameters()]
    def step(n, x):
        return model(x)["next_state"]

    result = finite_amplitude_response(step, 0, state, direction, [1e-4, 5e-5])
    difference = result["odd_response"][1] / result["multipliers"][1]
    torch.testing.assert_close(result["jvp"], difference, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(
        canonicalize_vorticity(result["jvp"]), result["jvp"], rtol=1e-12, atol=1e-12
    )
    for parameter, before in zip(model.parameters(), parameters, strict=True):
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize(
    "reference",
    [
        torch.zeros(3),
        torch.zeros(1, 2),
        torch.zeros(2, 0),
        torch.zeros(2, 1, dtype=torch.int64),
        torch.zeros(2, 1, dtype=torch.float16),
        torch.tensor([[0.0], [float("nan")]]),
    ],
)
def test_invalid_reference_rejected_before_calls(reference):
    def forbidden(n, x):
        raise AssertionError("invalid reference must not reach the model")

    with pytest.raises(ValueError):
        forced_tangent_forecast(forbidden, reference)


@pytest.mark.parametrize(
    "step",
    [
        lambda n, x: x.reshape(-1)[:1],
        lambda n, x: x.double(),
        lambda n, x: torch.full_like(x, float("inf")),
    ],
)
def test_invalid_transition_output_rejected(step):
    with pytest.raises(ValueError):
        forced_tangent_forecast(step, torch.zeros(2, 2))


def test_nonfinite_tangent_and_inference_mode_fail_explicitly():
    with pytest.raises(ArithmeticError, match="tangent"):
        forced_tangent_forecast(lambda n, x: x.sqrt(), torch.zeros(2, 1))
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        forced_tangent_forecast(lambda n, x: x, torch.zeros(2, 1))


@pytest.mark.parametrize("multipliers", [[], [0], [-1], [float("nan")], [[1]]])
def test_invalid_probe_multipliers_rejected(multipliers):
    with pytest.raises(ValueError, match="multipliers"):
        finite_amplitude_response(
            lambda n, x: x, 0, torch.zeros(2), torch.ones(2), multipliers
        )


def test_zero_or_unresolved_probe_direction_rejected():
    with pytest.raises(ValueError, match="zero direction"):
        finite_amplitude_response(
            lambda n, x: x, 0, torch.zeros(2), torch.zeros(2), [1]
        )
    with pytest.raises(ValueError, match="unresolved"):
        finite_amplitude_response(
            lambda n, x: x, 0, torch.full((2,), 1e8), torch.ones(2), [1]
        )


def test_posthoc_ordered_affine_discrepancy_matches_independent_evolution():
    matrices = torch.tensor(
        [[[0.8, 2.0], [0.0, 0.7]], [[0.9, 0.0], [1.5, 0.8]]],
        dtype=torch.float64,
    )
    reference = torch.tensor(
        [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]], dtype=torch.float64
    )

    def step(n, x):
        return matrices[n] @ x + 0.1

    forecast = forced_tangent_forecast(step, reference)
    actual = [reference[0] + torch.tensor([0.1, -0.2], dtype=torch.float64)]
    for matrix in matrices:
        actual.append(matrix @ actual[-1] + 0.1)
    errors = torch.stack(actual) - reference
    for n, matrix in enumerate(matrices):
        result = decompose_forecast_error(step, n, reference, forecast, errors)
        expected = matrix @ (forecast["predicted_error"][n] - errors[n])
        torch.testing.assert_close(result["discrepancy_next"], expected)
        torch.testing.assert_close(result["propagated_discrepancy"], expected)
        torch.testing.assert_close(
            result["actual_remainder"], torch.zeros_like(expected), atol=1e-14, rtol=0
        )
        assert result["closure_relative_l2"].item() < 1e-13


def test_posthoc_actual_remainder_uses_observed_direction_and_retains_cancellation():
    reference = torch.zeros(2, 2, dtype=torch.float64)
    e = torch.tensor([0.2, -0.3], dtype=torch.float64)
    z = e + torch.tensor([0.04, 0.045], dtype=torch.float64)
    forecast = {
        "clean_forcing": reference[:1],
        "predicted_error": torch.stack((z, z)),
    }
    errors = torch.stack((e, e + e.square()))
    result = decompose_forecast_error(
        lambda n, x: x + x.square(), 0, reference, forecast, errors
    )
    torch.testing.assert_close(result["actual_remainder"], e.square())
    assert not torch.allclose(result["actual_remainder"], z.square())
    torch.testing.assert_close(
        result["propagated_discrepancy"],
        torch.tensor([0.04, 0.045], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["discrepancy_next"],
        torch.tensor([0.0, -0.045], dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert result["closure_relative_l2"].item() < 1e-13


def test_posthoc_clean_execution_offset_has_positive_sign():
    reference = torch.zeros(3, 2, dtype=torch.float64)
    offset = torch.tensor([0.02, -0.03], dtype=torch.float64)
    matrix = torch.tensor([[1.1, 0.4], [-0.2, 0.8]], dtype=torch.float64)

    def step(n, x):
        return matrix @ x + 0.1 + (offset if torch.is_grad_enabled() else 0.0)

    forecast = forced_tangent_forecast(step, reference)
    actual = [reference[0]]
    for _ in range(2):
        actual.append(matrix @ actual[-1] + 0.1)
    errors = torch.stack(actual) - reference
    for n in range(2):
        result = decompose_forecast_error(step, n, reference, forecast, errors)
        torch.testing.assert_close(result["clean_kernel_offset"], offset)
        torch.testing.assert_close(
            result["discrepancy_next"], result["propagated_discrepancy"] + offset
        )
        assert result["closure_relative_l2"].item() < 1e-13


def test_posthoc_closure_retains_stored_update_residual_without_rewriting_forecast():
    reference = torch.zeros(2, 2, 2, dtype=torch.float32)
    forecast = forced_tangent_forecast(lambda n, x: 1.5 * x + 0.1, reference)
    errors = forecast["predicted_error"].clone()
    perturbation = torch.tensor([[0.01, -0.02], [0.03, -0.04]])
    forecast["predicted_error"][1] += perturbation
    saved = {key: value.clone() for key, value in forecast.items()}
    result = decompose_forecast_error(
        lambda n, x: 1.5 * x + 0.1, 0, reference, forecast, errors
    )
    expected = saved["predicted_error"][1].double() - errors[1].double()
    torch.testing.assert_close(result["closure_residual"], expected, rtol=0, atol=0)
    assert result["closure_relative_l2"].item() == pytest.approx(1.0)
    for key in forecast:
        torch.testing.assert_close(forecast[key], saved[key], rtol=0, atol=0)
    assert all(value.dtype == torch.float64 for value in result.values())
    assert all(
        not value.requires_grad and value.grad_fn is None for value in result.values()
    )


def test_posthoc_zero_current_discrepancy_can_develop_nonlinear_error():
    reference = torch.zeros(2, 2, 2, dtype=torch.float64)
    e = torch.tensor([[0.2, -0.1], [0.4, -0.3]], dtype=torch.float64)
    forecast = {
        "clean_forcing": reference[:1],
        "predicted_error": torch.stack((e, e)),
    }
    errors = torch.stack((e, e + e.square()))
    result = decompose_forecast_error(
        lambda n, x: x + x.square(), 0, reference, forecast, errors
    )
    torch.testing.assert_close(result["discrepancy_now"], torch.zeros_like(e))
    torch.testing.assert_close(result["propagated_discrepancy"], torch.zeros_like(e))
    torch.testing.assert_close(result["discrepancy_next"], -e.square())
    assert result["closure_relative_l2"].item() < 1e-13
