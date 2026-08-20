from __future__ import annotations

import inspect

import pytest
import torch

from scripts.time_dependent_no.smoke_realm_ignithit_ffno import main as smoke_main
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    grouped_next_state_mse,
    normalize_realm_coordinates,
    parameter_count_within_reported_tolerance,
    trainable_parameter_count,
)


def _small_config() -> RealmFFNOConfig:
    return RealmFFNOConfig(
        state_channels=12,
        coordinate_channels=2,
        output_channels=12,
        width=8,
        layers=2,
        modes_y=3,
        modes_x=3,
        feedforward_factor=2,
        feedforward_layers=2,
        head_width=8,
    )


def test_ffno_m_parameter_count_matches_frozen_architecture() -> None:
    model = RealmFFNO2d()
    count = trainable_parameter_count(model)
    assert count == 8_936_460
    assert parameter_count_within_reported_tolerance(count)


def test_small_ffno_forward_backward_is_finite_and_shape_exact() -> None:
    torch.manual_seed(7)
    model = RealmFFNO2d(_small_config())
    state = torch.randn(2, 12, 8, 8, requires_grad=True)
    coordinates = torch.randn(1, 2, 8, 8)
    truth = torch.randn_like(state)

    prediction = model(state, coordinates)
    loss, grouped = grouped_next_state_mse(prediction, truth)
    loss.backward()

    assert prediction.shape == state.shape
    assert torch.isfinite(prediction).all()
    assert set(grouped) == {"chem", "T", "rho", "u"}
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_bfloat16_autocast_keeps_fft_in_float32_and_backpropagates() -> None:
    torch.manual_seed(11)
    model = RealmFFNO2d(_small_config())
    state = torch.randn(1, 12, 8, 8, requires_grad=True)
    coordinates = torch.randn(1, 2, 8, 8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = model(state, coordinates)
        loss = prediction.square().mean()
    loss.backward()

    assert prediction.dtype == torch.bfloat16
    assert torch.isfinite(prediction).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_coordinate_normalization_preserves_relative_scale_and_batch_broadcast() -> (
    None
):
    y = torch.tensor([4.0, 2.0]).view(1, 1, 2, 1).expand(1, 1, 2, 3)
    x = torch.tensor([13.0, 11.0, 9.0]).view(1, 1, 1, 3).expand(1, 1, 2, 3)
    coordinates = torch.cat((y, x), dim=1)
    normalized = normalize_realm_coordinates(coordinates)
    assert normalized[:, 0].min() == 0.0
    assert normalized[:, 0].max() == 1.0
    assert normalized[:, 1].min() == 0.0
    assert normalized[:, 1].max() == 2.0

    model = RealmFFNO2d(_small_config())
    state = torch.zeros(3, 12, 2, 3)
    with pytest.raises(ValueError, match="modes exceed"):
        model(state, normalized)


def test_grouped_loss_is_sum_of_present_group_means_and_retains_gradient() -> None:
    truth = torch.zeros(2, 12, 4, 4)
    prediction = torch.ones_like(truth, requires_grad=True)
    loss, grouped = grouped_next_state_mse(prediction, truth)
    assert loss.item() == pytest.approx(4.0)
    assert {name: value.item() for name, value in grouped.items()} == {
        "chem": 1.0,
        "T": 1.0,
        "rho": 1.0,
        "u": 1.0,
    }
    loss.backward()
    assert prediction.grad is not None


def test_ffno_forward_surface_has_no_truth_argument_and_rejects_contract_drift() -> (
    None
):
    assert tuple(inspect.signature(RealmFFNO2d.forward).parameters) == (
        "self",
        "state",
        "static_coordinates",
    )
    model = RealmFFNO2d(_small_config())
    state = torch.zeros(1, 12, 8, 8)
    with pytest.raises(ValueError, match="coordinate case count"):
        model(state, torch.zeros(2, 2, 8, 8))
    with pytest.raises(ValueError, match="share dtype"):
        model(state, torch.zeros(1, 2, 8, 8, dtype=torch.float64))


def test_config_and_coordinate_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        RealmFFNOConfig(width=0)
    with pytest.raises(ValueError, match="positive finite range"):
        normalize_realm_coordinates(torch.ones(1, 2, 4, 4))
    with pytest.raises(ValueError, match="positive integer"):
        parameter_count_within_reported_tolerance(0)


def test_smoke_cli_help_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        smoke_main(["--help"])
    assert exit_info.value.code == 0
    rendered = capsys.readouterr().out
    assert "--manifest" in rendered
    assert "--data-root" in rendered
    assert "--normalizer-arrays" in rendered
    assert "--report-dir" in rendered
