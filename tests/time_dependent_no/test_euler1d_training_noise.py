import importlib.util
from pathlib import Path

import torch

from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
)
from utility.time_dependent_no.euler1d_targets import (
    LimitedConservativeResidualTargetAdapter,
)


def _load_ladder_module():
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root / "scripts" / "time_dependent_no" / "train_euler1d_target_ladder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_euler1d_target_ladder", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _batch():
    x = torch.linspace(0.0, 1.0, 6).unsqueeze(0)
    rho = 0.8 + 0.2 * x
    velocity = -0.1 + 0.4 * x
    pressure = 0.6 + 0.3 * x
    current = torch.stack((rho, velocity, pressure), dim=-1)
    target = current + torch.tensor([0.02, -0.01, 0.03])
    return make_euler1d_batch(
        current,
        x,
        torch.tensor([0.05]),
        target_primitive=target,
        left_boundary_primitive=current[:, 0],
    )


def test_apply_primitive_input_noise_zero_std_returns_original_batch():
    ladder = _load_ladder_module()
    batch = _batch()

    noisy = ladder.apply_primitive_input_noise(batch, 0.0)

    assert noisy is batch


def test_apply_primitive_input_noise_preserves_clean_training_contract():
    ladder = _load_ladder_module()
    batch = _batch()
    torch.manual_seed(1234)

    noisy = ladder.apply_primitive_input_noise(batch, 0.05)

    assert noisy is not batch
    assert torch.all(noisy.current_primitive[..., 0] > 0.0)
    assert torch.all(noisy.current_primitive[..., 2] > 0.0)
    assert not torch.allclose(noisy.current_primitive, batch.current_primitive)
    torch.testing.assert_close(noisy.target_primitive, batch.target_primitive)
    torch.testing.assert_close(noisy.dt, batch.dt)
    torch.testing.assert_close(
        noisy.geometry.cell_volume,
        batch.geometry.cell_volume,
    )
    torch.testing.assert_close(
        noisy.left_boundary_primitive,
        batch.left_boundary_primitive,
    )


def test_proposed_safety_metrics_report_pre_limiter_state():
    ladder = _load_ladder_module()
    batch = _batch()
    adapter = LimitedConservativeResidualTargetAdapter(safety=1.0)
    unsafe_delta = torch.zeros_like(batch.current_conservative)
    unsafe_delta[..., 0] = -2.0 * batch.current_conservative[..., 0]
    unsafe_delta[..., 2] = -2.0 * batch.current_conservative[..., 2]

    prediction = adapter(unsafe_delta, batch)
    proposed = ladder.proposed_conservative(prediction)
    proposed_primitive = conservative_to_primitive(proposed, gamma=batch.gamma)
    limited_primitive = conservative_to_primitive(
        prediction.conservative,
        gamma=batch.gamma,
    )
    accumulator = ladder.new_conservative_safety_accumulator()
    ladder.update_conservative_safety_accumulator(
        accumulator,
        proposed,
        gamma=batch.gamma,
    )

    metrics = ladder.proposed_safety_metrics(accumulator)

    torch.testing.assert_close(proposed, prediction.aux["proposed_conservative"])
    assert torch.any(proposed_primitive[..., 0] <= 0.0)
    assert torch.any(proposed_primitive[..., 2] <= 0.0)
    assert torch.all(limited_primitive[..., 0] > 0.0)
    assert torch.all(limited_primitive[..., 2] > 0.0)
    assert metrics["num_nonpositive_proposed_density"] > 0
    assert metrics["num_nonpositive_proposed_pressure"] > 0
    assert (
        metrics["num_nonpositive_raw_density"]
        == metrics["num_nonpositive_proposed_density"]
    )
    assert (
        metrics["num_nonpositive_raw_pressure"]
        == metrics["num_nonpositive_proposed_pressure"]
    )
