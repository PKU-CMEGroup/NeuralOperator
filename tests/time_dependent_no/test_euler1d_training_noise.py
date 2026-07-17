import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utility.time_dependent_no.euler1d_models import CPGNetEuler1D
from utility.time_dependent_no.euler1d_data import Euler1DNPZ
from utility.time_dependent_no.euler1d import (
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler1d_targets import (
    CPGNetInterfaceTargetAdapter,
    ConservativeResidualTargetAdapter,
    FluxTargetAdapter,
    LimitedConservativeResidualTargetAdapter,
    ProjectedConservativeResidualTargetAdapter,
    canonicalize_owner_oriented_face_flux,
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
    sys.modules[spec.name] = module
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


class _ScalarResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, batch):
        return self.weight * torch.ones_like(batch.current_primitive)


def _fixed_conservative_normalizer(ladder):
    return ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )


def test_deferred_one_step_metrics_preserve_updates_and_metrics():
    ladder = _load_ladder_module()
    batch = _batch()
    regular = _ScalarResidual()
    deferred = _ScalarResidual()
    deferred.load_state_dict(regular.state_dict())
    regular_optimizer = torch.optim.SGD(regular.parameters(), lr=1.0e-3)
    deferred_optimizer = torch.optim.SGD(deferred.parameters(), lr=1.0e-3)
    normalizer = _fixed_conservative_normalizer(ladder)
    adapter = ConservativeResidualTargetAdapter()

    regular_metrics = ladder.train_one_epoch(
        regular,
        adapter,
        [batch, batch],
        normalizer,
        regular_optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        target_supervision="state",
    )
    deferred_metrics = ladder.train_one_epoch(
        deferred,
        adapter,
        [batch, batch],
        normalizer,
        deferred_optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        target_supervision="state",
        defer_metric_sync=True,
    )

    torch.testing.assert_close(deferred.weight, regular.weight, rtol=0.0, atol=0.0)
    for key in ("loss", "state_loss", "relative_l2"):
        assert deferred_metrics[key] == pytest.approx(regular_metrics[key], rel=1.0e-7)


def test_deferred_metrics_reject_nonfinite_loss():
    ladder = _load_ladder_module()
    model = _ScalarResidual()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    with pytest.raises(RuntimeError, match="non-finite training loss"):
        ladder.train_one_epoch(
            model,
            ConservativeResidualTargetAdapter(),
            [_batch()],
            _fixed_conservative_normalizer(ladder),
            optimizer,
            torch.device("cpu"),
            grad_clip=1.0,
            loss_coordinates="conservative",
            target_supervision="state",
            defer_metric_sync=True,
        )


def test_deferred_unrolled_metrics_preserve_updates_and_metrics():
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
    regular = _ScalarResidual()
    deferred = _ScalarResidual()
    deferred.load_state_dict(regular.state_dict())
    regular_optimizer = torch.optim.SGD(regular.parameters(), lr=1.0e-3)
    deferred_optimizer = torch.optim.SGD(deferred.parameters(), lr=1.0e-3)
    normalizer = _fixed_conservative_normalizer(ladder)
    adapter = ConservativeResidualTargetAdapter()
    loader = [(batch, target_sequence, dt_sequence)] * 2

    regular_metrics = ladder.train_unrolled_epoch(
        regular,
        adapter,
        loader,
        normalizer,
        regular_optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
    )
    deferred_metrics = ladder.train_unrolled_epoch(
        deferred,
        adapter,
        loader,
        normalizer,
        deferred_optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        defer_metric_sync=True,
    )

    torch.testing.assert_close(deferred.weight, regular.weight, rtol=0.0, atol=0.0)
    for key in ("loss", "state_loss", "admissibility_loss", "relative_l2"):
        assert deferred_metrics[key] == pytest.approx(regular_metrics[key], rel=1.0e-7)


def test_split_seed_is_independent_of_training_seed():
    ladder = _load_ladder_module()
    source = SimpleNamespace(num_cases=32)
    common = {
        "train_cases": 8,
        "val_cases": 4,
        "test_cases": 4,
        "split_seed": 17,
    }

    split_a = ladder.split_cases(source, SimpleNamespace(seed=1, **common))
    split_b = ladder.split_cases(source, SimpleNamespace(seed=2, **common))

    for indices_a, indices_b in zip(split_a, split_b, strict=True):
        np.testing.assert_array_equal(indices_a, indices_b)


def test_continuation_checkpoint_loads_compatible_smaller_stride_weights(
    tmp_path: Path,
):
    ladder = _load_ladder_module()
    args = SimpleNamespace(
        fno_width=8,
        fno_modes=4,
        fno_layers=2,
        fno_fc_dim=8,
        fno_pad_ratio=0.0,
        input_coordinates="conservative",
        input_normalization="fixed_physical",
        loss_coordinates="conservative",
        loss_normalization="fixed_physical",
        recurrent_coordinates="conservative",
        positive_transform="none",
        target_supervision="state",
        flux_gauge_mode="raw",
        interface_flux_mode="rusanov",
        input_noise_std=0.0,
        initial_frame_weight=1.0,
        boundary_exchange_loss_weight=0.0,
        unroll_admissibility_weight=0.0,
        step_stride=2,
    )
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )
    source_model = ladder.build_model("fno", "residual", args, normalizer)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.125)
    cases = np.array([1, 4], dtype=np.int64)
    source_args = vars(args).copy()
    for legacy_field in (
        "interface_flux_mode",
        "initial_frame_weight",
        "boundary_exchange_loss_weight",
    ):
        source_args.pop(legacy_field)
    source_args.update(step_stride=1, seed=7, split_seed=11)
    checkpoint_path = tmp_path / "stride1.pt"
    torch.save(
        {
            "model": "fno",
            "target": "residual",
            "args": source_args,
            "model_state_dict": source_model.state_dict(),
            "train_cases": cases,
            "val_cases": cases,
            "test_cases": cases,
            "best_epoch": 59,
        },
        checkpoint_path,
    )
    target_model = ladder.build_model("fno", "residual", args, normalizer)

    metadata = ladder.load_continuation_checkpoint(
        target_model,
        checkpoint_path,
        model_name="fno",
        target_name="residual",
        args=args,
        train_cases=cases,
        val_cases=cases,
        test_cases=cases,
    )

    for name, value in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], value)
    assert metadata["source_step_stride"] == 1
    assert metadata["source_best_epoch"] == 59
    assert metadata["optimizer_state_loaded"] is False

    source_args["step_stride"] = 2
    torch.save(
        {
            "model": "fno",
            "target": "residual",
            "args": source_args,
            "model_state_dict": source_model.state_dict(),
            "train_cases": cases,
            "val_cases": cases,
            "test_cases": cases,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="smaller positive step_stride"):
        ladder.load_continuation_checkpoint(
            target_model,
            checkpoint_path,
            model_name="fno",
            target_name="residual",
            args=args,
            train_cases=cases,
            val_cases=cases,
            test_cases=cases,
        )


def test_initial_frame_sampler_is_default_off_and_only_upweights_frame_zero():
    ladder = _load_ladder_module()
    items = [(2, 0), (2, 1), (5, 0), (5, 3)]
    generator = torch.Generator().manual_seed(11)

    assert (
        ladder.make_initial_frame_sampler(
            items,
            initial_frame_weight=1.0,
            generator=generator,
        )
        is None
    )
    sampler = ladder.make_initial_frame_sampler(
        items,
        initial_frame_weight=15.0,
        generator=generator,
    )

    assert sampler is not None
    torch.testing.assert_close(
        sampler.weights,
        torch.tensor([15.0, 1.0, 15.0, 1.0], dtype=torch.float64),
    )
    assert sampler.num_samples == len(items)
    assert sampler.replacement is True


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


def test_additive_cpg_input_noise_matches_release_contract():
    ladder = _load_ladder_module()
    batch = _batch()
    torch.manual_seed(4321)
    expected_noise = torch.randn_like(batch.current_primitive) * 0.02
    torch.manual_seed(4321)

    noisy = ladder.apply_primitive_input_noise(batch, 0.02, mode="additive")

    torch.testing.assert_close(
        noisy.current_primitive, batch.current_primitive + expected_noise
    )
    torch.testing.assert_close(noisy.target_primitive, batch.target_primitive)
    torch.testing.assert_close(
        noisy.left_boundary_primitive, batch.left_boundary_primitive
    )


def test_fixed_physical_conservative_normalizer_uses_energy_reference_scale():
    ladder = _load_ladder_module()
    primitive = _batch().current_primitive.numpy()
    source = Euler1DNPZ(
        data=np.stack((primitive, primitive), axis=1),
        x=np.linspace(0.0, 1.0, primitive.shape[1], dtype=np.float32)[None],
        t=np.array([[0.0, 0.1]], dtype=np.float32),
        left_states=primitive[:, 0],
        right_states=primitive[:, -1],
        gamma=1.4,
        metadata={},
    )

    normalizer = ladder.PrimitiveNormalizer.from_source_inputs(
        source,
        np.array([0]),
        coordinates="conservative",
        normalization="fixed_physical",
    )

    torch.testing.assert_close(normalizer.mean, torch.zeros(1, 1, 3))
    torch.testing.assert_close(
        normalizer.std,
        torch.tensor([[[1.0, 1.0, 2.5]]]),
    )
    assert normalizer.coordinates == "conservative"
    assert normalizer.normalization == "fixed_physical"


def test_state_pair_for_loss_selects_conservative_prediction_and_target():
    ladder = _load_ladder_module()
    batch = _batch()
    prediction = ConservativeResidualTargetAdapter()(
        torch.zeros_like(batch.current_conservative), batch
    )

    predicted, target = ladder.state_pair_for_loss(
        prediction,
        batch,
        "conservative",
    )

    torch.testing.assert_close(predicted, batch.current_conservative)
    torch.testing.assert_close(
        target,
        primitive_to_conservative(batch.target_primitive, gamma=batch.gamma),
    )


def test_face_flux_normalizer_aggregates_macro_step_impulses():
    ladder = _load_ladder_module()
    primitive = _batch().current_primitive.numpy()
    face_flux_integral = np.zeros((1, 2, 7, 3), dtype=np.float32)
    face_flux_integral[:, 0] = 0.1
    face_flux_integral[:, 1] = 0.3
    source = Euler1DNPZ(
        data=np.stack((primitive, primitive, primitive), axis=1),
        x=np.linspace(0.0, 1.0, primitive.shape[1], dtype=np.float32)[None],
        t=np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
        left_states=primitive[:, 0],
        right_states=primitive[:, -1],
        gamma=1.4,
        metadata={},
        face_flux_integral=face_flux_integral,
    )

    normalizer = ladder.PrimitiveNormalizer.from_source_face_flux(
        source,
        np.array([0]),
        step_stride=2,
        normalization="empirical",
    )

    torch.testing.assert_close(normalizer.mean, torch.full((1, 1, 3), 2.0))
    torch.testing.assert_close(normalizer.std, torch.full((1, 1, 3), 1.0e-6))
    assert normalizer.coordinates == "face_flux"


def test_boundary_exchange_normalizer_uses_macro_impulse_rms():
    ladder = _load_ladder_module()
    primitive = _batch().current_primitive.numpy()
    face_flux_integral = np.zeros((1, 2, 7, 3), dtype=np.float32)
    face_flux_integral[:, 0] = 0.1
    face_flux_integral[:, 1] = 0.3
    source = Euler1DNPZ(
        data=np.stack((primitive, primitive, primitive), axis=1),
        x=np.linspace(0.0, 1.0, primitive.shape[1], dtype=np.float32)[None],
        t=np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
        left_states=primitive[:, 0],
        right_states=primitive[:, -1],
        gamma=1.4,
        metadata={},
        face_flux_integral=face_flux_integral,
    )

    normalizer = ladder.PrimitiveNormalizer.from_source_boundary_exchange(
        source,
        np.array([0]),
        step_stride=2,
    )

    torch.testing.assert_close(normalizer.mean, torch.zeros(1, 1, 3))
    torch.testing.assert_close(normalizer.std, torch.full((1, 1, 3), 0.8))
    assert normalizer.coordinates == "boundary_exchange"


def test_projected_residual_loss_supervises_solver_boundary_impulse():
    ladder = _load_ladder_module()
    base = _batch()
    target_exchange = ladder.state_pair_boundary_exchange_target(
        base.current_primitive,
        base.target_primitive,
        base,
    )
    target_face_flux = torch.zeros(1, 7, 3)
    target_face_flux[:, 0] = -target_exchange / base.dt[:, None]
    batch = make_euler1d_batch(
        base.current_primitive,
        base.geometry.cell_centers.squeeze(-1),
        base.dt,
        target_primitive=base.target_primitive,
        target_face_flux=target_face_flux,
        left_boundary_primitive=base.left_boundary_primitive,
        right_initial_primitive=base.right_initial_primitive,
    )
    error = torch.tensor([[0.1, -0.2, 0.3]])
    raw = torch.zeros(1, 7, 3)
    raw[:, -1] = target_exchange + error
    prediction = ProjectedConservativeResidualTargetAdapter()(raw, batch)
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
    )
    boundary_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="boundary_exchange",
    )

    total, state_loss, flux_loss, boundary_loss = ladder.supervised_loss(
        raw,
        prediction,
        batch,
        normalizer,
        loss_coordinates="conservative",
        target_supervision="state",
        flux_normalizer=None,
        flux_loss_weight=1.0,
        boundary_exchange_normalizer=boundary_normalizer,
        boundary_exchange_loss_weight=0.25,
    )

    torch.testing.assert_close(
        ladder.solver_boundary_exchange_target(batch),
        target_exchange,
    )
    assert flux_loss is None
    assert boundary_loss is not None
    torch.testing.assert_close(boundary_loss, error.square().mean())
    torch.testing.assert_close(total, state_loss + 0.25 * boundary_loss)


def test_supervised_loss_supports_direct_and_joint_face_flux_losses():
    ladder = _load_ladder_module()
    base = _batch()
    target_face_flux = torch.ones(1, 7, 3)
    batch = make_euler1d_batch(
        base.current_primitive,
        base.geometry.cell_centers.squeeze(-1),
        base.dt,
        target_primitive=base.target_primitive,
        target_face_flux=target_face_flux,
        left_boundary_primitive=base.left_boundary_primitive,
        right_initial_primitive=base.right_initial_primitive,
    )
    raw = target_face_flux + torch.tensor([1.0, 2.0, 4.0])
    prediction = FluxTargetAdapter()(raw, batch)
    state_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
    )
    flux_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="face_flux",
    )

    direct, state_loss, flux_loss, boundary_loss = ladder.supervised_loss(
        raw,
        prediction,
        batch,
        state_normalizer,
        loss_coordinates="primitive",
        target_supervision="direct_flux",
        flux_normalizer=flux_normalizer,
        flux_loss_weight=0.25,
    )
    joint, joint_state_loss, joint_flux_loss, joint_boundary_loss = (
        ladder.supervised_loss(
            raw,
            prediction,
            batch,
            state_normalizer,
            loss_coordinates="primitive",
            target_supervision="joint",
            flux_normalizer=flux_normalizer,
            flux_loss_weight=0.25,
        )
    )

    assert flux_loss is not None
    assert joint_flux_loss is not None
    assert boundary_loss is None
    assert joint_boundary_loss is None
    torch.testing.assert_close(direct, flux_loss)
    torch.testing.assert_close(joint_state_loss, state_loss)
    torch.testing.assert_close(joint_flux_loss, flux_loss)
    torch.testing.assert_close(joint, state_loss + 0.25 * flux_loss)


def test_face_flux_error_decomposition_uses_owner_oriented_null_mode():
    ladder = _load_ladder_module()
    base = _batch()
    target_face_flux = torch.zeros(1, 7, 3)
    batch = make_euler1d_batch(
        base.current_primitive,
        base.geometry.cell_centers.squeeze(-1),
        base.dt,
        target_primitive=base.target_primitive,
        target_face_flux=target_face_flux,
        left_boundary_primitive=base.left_boundary_primitive,
        right_initial_primitive=base.right_initial_primitive,
    )
    amplitude = torch.tensor([[[0.5, -1.0, 2.0]]])
    raw = batch.geometry.face_normal * amplitude
    flux_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="face_flux",
    )

    full_mse, active_mse, gauge_mse = ladder.normalized_face_flux_error_decomposition(
        raw,
        batch,
        flux_normalizer,
    )
    prediction = FluxTargetAdapter()(raw, batch)

    torch.testing.assert_close(
        active_mse, torch.zeros_like(active_mse), atol=1.0e-7, rtol=0.0
    )
    torch.testing.assert_close(full_mse, gauge_mse)
    torch.testing.assert_close(
        prediction.conservative,
        batch.current_conservative,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_canonical_face_flux_projection_preserves_update_and_removes_null_mode():
    batch = _batch()
    torch.manual_seed(7)
    face_flux = 0.1 * torch.randn(1, 7, 3)
    gauge = batch.geometry.face_normal * torch.tensor([[[0.2, -0.1, 0.3]]])

    canonical = canonicalize_owner_oriented_face_flux(
        face_flux + gauge,
        batch.geometry.face_normal,
    )
    canonical_without_gauge = canonicalize_owner_oriented_face_flux(
        face_flux,
        batch.geometry.face_normal,
    )
    raw_prediction = FluxTargetAdapter()(face_flux + gauge, batch)
    canonical_prediction = FluxTargetAdapter(flux_gauge_mode="canonical")(
        face_flux + gauge,
        batch,
    )

    torch.testing.assert_close(canonical, canonical_without_gauge)
    torch.testing.assert_close(
        (canonical * batch.geometry.face_normal).sum(dim=1),
        torch.zeros(1, 3),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        canonical_prediction.conservative,
        raw_prediction.conservative,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_canonical_flux_supervision_ignores_solver_flux_gauge():
    ladder = _load_ladder_module()
    base = _batch()
    target_face_flux = base.geometry.face_normal * torch.tensor([[[0.3, -0.2, 0.1]]])
    batch = make_euler1d_batch(
        base.current_primitive,
        base.geometry.cell_centers.squeeze(-1),
        base.dt,
        target_primitive=base.target_primitive,
        target_face_flux=target_face_flux,
        left_boundary_primitive=base.left_boundary_primitive,
        right_initial_primitive=base.right_initial_primitive,
    )
    raw = base.geometry.face_normal * torch.tensor([[[-0.4, 0.5, -0.6]]])
    prediction = FluxTargetAdapter(flux_gauge_mode="canonical")(raw, batch)
    state_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
    )
    flux_normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="face_flux",
    )

    loss, _, flux_loss, boundary_loss = ladder.supervised_loss(
        raw,
        prediction,
        batch,
        state_normalizer,
        loss_coordinates="primitive",
        target_supervision="direct_flux",
        flux_normalizer=flux_normalizer,
        flux_loss_weight=1.0,
    )

    assert flux_loss is not None
    assert boundary_loss is None
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        prediction.conservative,
        batch.current_conservative,
        atol=1.0e-6,
        rtol=0.0,
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


def test_cpgnet_unrolled_training_backpropagates_through_recurrent_states(
    monkeypatch,
):
    ladder = _load_ladder_module()
    monkeypatch.setattr(
        ladder,
        "conservative_admissibility_barrier",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("zero-weight barrier must not be evaluated")
        ),
    )
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


def test_conservative_admissibility_barrier_is_smooth_and_state_sensitive():
    ladder = _load_ladder_module()
    conservative = torch.tensor(
        [[[0.2, 0.0, 0.5], [0.05, 0.0, 0.05]]],
        requires_grad=True,
    )

    safe = ladder.conservative_admissibility_barrier(
        conservative[:, :1],
        gamma=1.4,
        density_margin=0.1,
        pressure_margin=0.1,
    )
    risky = ladder.conservative_admissibility_barrier(
        conservative[:, 1:],
        gamma=1.4,
        density_margin=0.1,
        pressure_margin=0.1,
    )
    risky.backward()

    assert risky > safe
    assert conservative.grad is not None
    assert torch.isfinite(conservative.grad).all()


def test_flux_head_unrolled_training_preserves_differentiable_conservative_state():
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

    class LearnableFaceFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.zeros(1, 7, 3))
            self.seen_conservative = []

        def forward(self, step_batch):
            self.seen_conservative.append(step_batch.current_conservative_state)
            return self.raw.expand(step_batch.current_primitive.shape[0], -1, -1)

    model = LearnableFaceFlux()
    adapter = FluxTargetAdapter()
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    before = model.raw.detach().clone()

    metrics = ladder.train_unrolled_epoch(
        model,
        adapter,
        [(batch, target_sequence, dt_sequence)],
        normalizer,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        admissibility_weight=0.1,
        density_margin=0.1,
        pressure_margin=0.05,
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["state_loss"]))
    assert torch.isfinite(torch.tensor(metrics["admissibility_loss"]))
    assert len(model.seen_conservative) == 2
    assert all(state is not None for state in model.seen_conservative)
    assert model.seen_conservative[1].grad_fn is not None
    assert model.raw.grad is not None
    assert torch.isfinite(model.raw.grad).all()
    assert not torch.equal(before, model.raw.detach())


def test_unrolled_training_detaches_model_generated_burn_in():
    ladder = _load_ladder_module()
    batch = _batch()
    target_sequence = torch.stack(
        (
            batch.target_primitive,
            batch.target_primitive + torch.tensor([0.01, -0.01, 0.01]),
            batch.target_primitive + torch.tensor([0.02, -0.01, 0.02]),
        ),
        dim=1,
    )
    dt_sequence = torch.full((1, 3), 0.05)

    class BurnInProbeFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.zeros(1, 7, 3))
            self.grad_enabled = []
            self.seen_conservative = []

        def forward(self, step_batch):
            self.grad_enabled.append(torch.is_grad_enabled())
            self.seen_conservative.append(step_batch.current_conservative_state)
            return self.raw.expand(step_batch.current_primitive.shape[0], -1, -1)

    model = BurnInProbeFlux()
    adapter = FluxTargetAdapter()
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    metrics = ladder.train_unrolled_epoch(
        model,
        adapter,
        [(batch, target_sequence, dt_sequence)],
        normalizer,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        burn_in_steps=1,
    )

    assert model.grad_enabled == [False, True, True]
    assert model.seen_conservative[1].grad_fn is None
    assert model.seen_conservative[2].grad_fn is not None
    assert model.raw.grad is not None
    assert torch.isfinite(model.raw.grad).all()
    assert torch.isfinite(torch.tensor(metrics["burn_in_relative_l2"]))
    assert metrics["burn_in_nonpositive_sample_fraction"] == 0.0
    assert metrics["burn_in_min_density"] > 0.0
    assert metrics["burn_in_min_pressure"] > 0.0


def test_unrolled_training_teacher_offset_skips_generated_prefix():
    ladder = _load_ladder_module()
    batch = _batch()
    target_sequence = torch.stack(
        (
            batch.target_primitive,
            batch.target_primitive + torch.tensor([0.01, -0.01, 0.01]),
            batch.target_primitive + torch.tensor([0.02, -0.01, 0.02]),
        ),
        dim=1,
    )
    dt_sequence = torch.full((1, 3), 0.05)

    class TeacherOffsetProbeFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.zeros(1, 7, 3))
            self.seen_conservative = []

        def forward(self, step_batch):
            self.seen_conservative.append(step_batch.current_conservative_state)
            return self.raw.expand(step_batch.current_primitive.shape[0], -1, -1)

    model = TeacherOffsetProbeFlux()
    normalizer = ladder.PrimitiveNormalizer(
        mean=torch.zeros(1, 1, 3),
        std=torch.ones(1, 1, 3),
        coordinates="conservative",
        normalization="fixed_physical",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    metrics = ladder.train_unrolled_epoch(
        model,
        FluxTargetAdapter(),
        [(batch, target_sequence, dt_sequence)],
        normalizer,
        optimizer,
        torch.device("cpu"),
        grad_clip=1.0,
        loss_coordinates="conservative",
        burn_in_steps=1,
        burn_in_mode="teacher",
    )

    assert len(model.seen_conservative) == 2
    torch.testing.assert_close(
        model.seen_conservative[0],
        primitive_to_conservative(target_sequence[:, 0]),
    )
    assert model.seen_conservative[0].grad_fn is None
    assert model.seen_conservative[1].grad_fn is not None
    assert metrics["burn_in_relative_l2"] == 0.0
    assert metrics["burn_in_nonpositive_sample_fraction"] == 0.0
    assert metrics["burn_in_min_density"] > 0.0
    assert metrics["burn_in_min_pressure"] > 0.0

    with pytest.raises(ValueError, match="requires burn_in_steps"):
        ladder.train_unrolled_epoch(
            model,
            FluxTargetAdapter(),
            [(batch, target_sequence, dt_sequence)],
            normalizer,
            optimizer,
            torch.device("cpu"),
            grad_clip=1.0,
            burn_in_steps=0,
            burn_in_mode="teacher",
        )


def test_top2_pressure_front_metric_separates_rank_swap_from_displacement():
    ladder = _load_ladder_module()
    x = np.linspace(0.0, 1.0, 41)

    def state(first_strength, first_x, second_strength, second_x):
        pressure = np.ones_like(x)
        pressure += first_strength * (x >= first_x)
        pressure += second_strength * (x >= second_x)
        primitive = np.zeros((1, x.size, 3), dtype=np.float64)
        primitive[..., 0] = 1.0
        primitive[..., 2] = pressure
        return primitive

    truth = state(1.0, 0.3, 2.0, 0.7)
    rank_swap = state(2.0, 0.3, 1.0, 0.7)
    displaced = state(1.0, 0.5, 2.0, 0.7)

    truth_primary = ladder.pressure_front_position_np(truth, x)
    swap_primary = ladder.pressure_front_position_np(rank_swap, x)
    swap_metrics = ladder.pressure_front_top2_metrics_np(
        rank_swap,
        truth,
        x,
        min_separation_cells=2,
    )
    displaced_metrics = ladder.pressure_front_top2_metrics_np(
        displaced,
        truth,
        x,
        min_separation_cells=2,
    )

    assert abs(float(swap_primary[0] - truth_primary[0])) > 0.3
    assert swap_metrics["position_assignment_mae"][0] == pytest.approx(0.0)
    assert swap_metrics["primary_to_truth_top2_mae"][0] == pytest.approx(0.0)
    assert swap_metrics["strength_relative_l1"][0] > 0.5
    assert displaced_metrics["position_assignment_mae"][0] > 0.05


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


def test_raw_flux_rollout_stops_at_first_nonpositive_cell_state():
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

    class UnsafeFlux(torch.nn.Module):
        def forward(self, batch):
            raw = torch.zeros(
                batch.current_primitive.shape[0],
                batch.geometry.face_owner.shape[1],
                3,
                device=batch.current_primitive.device,
            )
            raw[:, 1, 0] = 100.0
            return raw

    row = ladder.rollout_case(
        UnsafeFlux(),
        FluxTargetAdapter(),
        source,
        case_id=0,
        steps=2,
        step_stride=1,
        device=torch.device("cpu"),
        final_frame=2,
        recurrent_coordinates="conservative",
    )

    assert row["num_steps"] == 0
    assert row["finite"] is True
    assert row["completed_horizon"] is False
    assert row["termination_reason"] == "nonpositive_raw_state"
    assert row["first_invalid_step"] == 1
    assert row["raw_min_density"] < 0.0
    assert row["num_nonpositive_raw_density"] > 0


def test_raw_residual_rollout_can_recur_in_conservative_coordinates():
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

    class ZeroResidual(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros_like(batch.current_conservative)

    row = ladder.rollout_case(
        ZeroResidual(),
        ConservativeResidualTargetAdapter(),
        source,
        case_id=0,
        steps=2,
        step_stride=1,
        device=torch.device("cpu"),
        final_frame=2,
        recurrent_coordinates="conservative",
    )

    assert row["completed_horizon"] is True
    assert row["recurrent_coordinates"] == "conservative"
    assert row["min_density"] == pytest.approx(1.0)
    assert row["min_pressure"] == pytest.approx(1.0)


def test_rollout_checkpoint_selection_uses_survival_before_fit_after_validity():
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
        "survival_fraction_mean": 0.25,
        "rollout_relative_l2_final": 0.01,
    }
    incomplete_bad_fit = dict(incomplete_good_fit)
    incomplete_longer = dict(incomplete_good_fit, survival_fraction_mean=0.75)
    nonfinite = dict(incomplete_good_fit, finite=False)

    complete_score = ladder.rollout_selection_score(complete, 10.0)
    good_fit_score = ladder.rollout_selection_score(incomplete_good_fit, 0.01)
    bad_fit_score = ladder.rollout_selection_score(incomplete_bad_fit, 0.1)
    longer_score = ladder.rollout_selection_score(incomplete_longer, 10.0)
    nonfinite_score = ladder.rollout_selection_score(nonfinite, 0.001)

    assert complete_score < longer_score < good_fit_score < bad_fit_score
    assert bad_fit_score < nonfinite_score


def test_rollout_summary_reports_survival_and_effective_cfl():
    ladder = _load_ladder_module()
    rows = [
        {
            "finite": True,
            "admissible": True,
            "completed_horizon": True,
            "num_steps": 4,
            "final_frame": 4,
            "survival_fraction": 1.0,
            "initial_effective_cfl": 3.0,
            "truth_effective_cfl_max": 5.0,
        },
        {
            "finite": True,
            "admissible": False,
            "completed_horizon": False,
            "num_steps": 2,
            "final_frame": 2,
            "survival_fraction": 0.5,
            "initial_effective_cfl": 7.0,
            "truth_effective_cfl_max": 11.0,
        },
    ]
    metric_defaults = {
        key: 0.0
        for key in (
            "rollout_relative_l2_mean",
            "rollout_relative_l2_final",
            "min_density",
            "min_pressure",
            "proposed_min_density",
            "proposed_min_pressure",
            "raw_min_density",
            "raw_min_pressure",
            "max_abs_primitive",
            "shock_position_mae",
            "shock_top2_position_mae",
            "shock_top2_strength_relative_l1",
            "shock_primary_to_truth_top2_mae",
            "conservative_total_error_final",
            "limiter_theta_mean",
            "limiter_theta_min",
            "limiter_activation_fraction",
            "flux_correction_abs_over_bound_mean",
            "flux_correction_abs_over_bound_max",
            "flux_correction_saturation_fraction",
        )
    }
    for row in rows:
        row.update(metric_defaults)

    summary = ladder.summarize_rollouts(rows)

    assert summary["num_completed_cases"] == 1
    assert summary["completion_fraction"] == pytest.approx(0.5)
    assert summary["survival_fraction_mean"] == pytest.approx(0.75)
    assert summary["survival_fraction_min"] == pytest.approx(0.5)
    assert summary["initial_effective_cfl_median"] == pytest.approx(5.0)
    assert summary["truth_effective_cfl_max_max"] == pytest.approx(11.0)
