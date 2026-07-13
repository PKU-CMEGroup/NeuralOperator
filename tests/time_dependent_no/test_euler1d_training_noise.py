import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from utility.time_dependent_no.euler1d_models import CPGNetEuler1D
from utility.time_dependent_no.euler1d_data import Euler1DNPZ
from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
)
from utility.time_dependent_no.euler1d_targets import (
    CPGNetInterfaceTargetAdapter,
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
        right_initial_primitive=current[:, -1],
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
    torch.testing.assert_close(
        noisy.right_initial_primitive,
        batch.right_initial_primitive,
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


def test_cpgnet_unrolled_training_backpropagates_through_recurrent_states():
    ladder = _load_ladder_module()
    batch = _batch()
    target_sequence = torch.stack(
        (
            batch.target_primitive,
            batch.target_primitive + torch.tensor([0.01, -0.01, 0.01]),
        ),
        dim=1,
    )
    dt_sequence = torch.full((1, 2), 0.05)
    model = CPGNetEuler1D(
        hidden_dim=8,
        message_passing_steps=1,
        mlp_layers=2,
        edge_hidden_dim=4,
        edge_encoder_steps=0,
    )
    adapter = CPGNetInterfaceTargetAdapter()
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    parameter = next(model.parameters())
    before = parameter.detach().clone()

    metrics = ladder.train_unrolled_epoch(
        model,
        adapter,
        [(batch, target_sequence, dt_sequence)],
        normalizer,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert not torch.equal(before, parameter.detach())


def test_cpgnet_rejects_non_solver_target_contract():
    ladder = _load_ladder_module()

    with pytest.raises(ValueError, match="cpg_interface"):
        ladder.requested_experiment_pairs("cpgnet", "limited_residual")

    assert ladder.requested_experiment_pairs("cpgnet", "all") == [
        ("cpgnet", "cpg_interface")
    ]


def test_raw_cpgnet_rollout_stops_at_first_nonpositive_cell_state():
    ladder = _load_ladder_module()
    num_cells = 5
    data = np.zeros((1, 3, num_cells, 3), dtype=np.float32)
    data[..., 0] = 1.0
    data[..., 2] = 1.0
    source = Euler1DNPZ(
        data=data,
        x=np.linspace(0.0, 1.0, num_cells, dtype=np.float32)[None],
        t=np.array([[0.0, 0.05, 0.1]], dtype=np.float32),
        left_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        right_states=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        gamma=1.4,
        metadata={},
    )

    class UnsafePositiveInterfaces(torch.nn.Module):
        def forward(self, batch):
            num_faces = batch.geometry.face_owner.shape[1]
            interface = torch.ones(
                batch.current_primitive.shape[0],
                num_faces,
                2,
                3,
                device=batch.current_primitive.device,
            )
            interface[..., 1] = 0.0
            interface[:, 2, :, 1] = 100.0
            return interface

    row = ladder.rollout_case(
        UnsafePositiveInterfaces(),
        CPGNetInterfaceTargetAdapter(),
        source,
        case_id=0,
        steps=2,
        step_stride=1,
        device=torch.device("cpu"),
        final_frame=2,
    )

    assert row["num_steps"] == 0
    assert row["finite"] is True
    assert row["completed_horizon"] is False
    assert row["termination_reason"] == "nonpositive_raw_state"
    assert row["first_invalid_step"] == 1
    assert row["raw_min_density"] < 0.0
    assert row["num_nonpositive_raw_density"] > 0


def test_rollout_checkpoint_selection_uses_fit_only_after_validity():
    ladder = _load_ladder_module()
    complete = {
        "finite": True,
        "admissible": True,
        "completed_horizon": True,
        "rollout_relative_l2_final": 0.4,
    }
    incomplete_good_fit = {
        "finite": True,
        "admissible": False,
        "completed_horizon": False,
        "rollout_relative_l2_final": 0.01,
    }
    incomplete_bad_fit = dict(incomplete_good_fit)
    nonfinite = dict(incomplete_good_fit, finite=False)

    complete_score = ladder.rollout_selection_score(complete, 10.0)
    good_fit_score = ladder.rollout_selection_score(incomplete_good_fit, 0.01)
    bad_fit_score = ladder.rollout_selection_score(incomplete_bad_fit, 0.1)
    nonfinite_score = ladder.rollout_selection_score(nonfinite, 0.001)

    assert complete_score < good_fit_score < bad_fit_score < nonfinite_score
