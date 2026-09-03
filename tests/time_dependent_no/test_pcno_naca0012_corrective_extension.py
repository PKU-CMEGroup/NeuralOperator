import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from utility.time_dependent_no.pcno_naca0012 import (
    NACANormalization,
    NACAPCNOResidual,
    predict_normalized_residual,
)
from utility.time_dependent_no.pcno_naca0012_corrective_extension import (
    EXTENSION_EXPERIMENT_ID,
    EXTENSION_LEARNED_ARMS,
    REFINER_ALPHAS_CUMPROD,
    REFINER_BETAS,
    ExponentialMovingAverage,
    FourStepVPredictionScheduler,
    NACAPCNORefiner,
    RefinerNoiseTape,
    VerifiedExtensionContract,
    build_refiner_input,
    curriculum_exposure_probability,
    group_examples_by_prefix_depth,
    load_extension_contract,
    make_detached_prefix_presentation,
    make_paired_displaced_presentation,
    make_refiner_training_presentation,
    maximum_available_prefix_depth,
    paired_bank_sign_for_epoch,
    paired_clean_displaced_objective,
    realize_prefix_depths,
    refined_recurrent_step,
    refiner_parameter_difference_from_baseline,
    sample_curriculum_requested_prefix_depth,
    sample_mp_pde_prefix_depth,
    sample_refiner_timesteps,
    validate_extension_math_contract,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPOSITORY
    / "docs"
    / "time_dependent_no"
    / "B3B4_NACA_CORRECTIVE_EXTENSION_CONTRACT.json"
)
PREREGISTRATION_PATH = (
    REPOSITORY
    / "docs"
    / "time_dependent_no"
    / "B3B4_NACA_CORRECTIVE_EXTENSION_PREREGISTRATION.md"
)


def _normalization() -> NACANormalization:
    return NACANormalization(
        state_mean=np.zeros(5, dtype=np.float64),
        state_scale=np.asarray([2.0, 3.0, 4.0, 5.0, 6.0]),
        residual_scale=np.asarray([0.5, 1.0, 2.0, 4.0, 8.0]),
        state_rms=np.ones(5, dtype=np.float64),
    )


def _geometry(batch: int, nodes: int) -> dict[str, torch.Tensor]:
    return {"static_features": torch.zeros(batch, nodes, 6)}


class _ToyResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, features, geometry_batch, *, fourier_tensors=None):
        del geometry_batch, fourier_tensors
        return self.weight * features[..., 11:16]


class _CountingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def forward(self, features, geometry_batch, *, fourier_tensors=None):
        del geometry_batch, fourier_tensors
        timestep = int(torch.argmax(features[0, 0, 21:25]).item())
        self.calls.append(timestep)
        return torch.zeros_like(features[..., 16:21])


def test_extension_contract_and_preregistration_binding_are_exact() -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    validate_extension_math_contract(payload)
    assert payload["experiment_id"] == EXTENSION_EXPERIMENT_ID
    assert tuple(payload["arms"]) == EXTENSION_LEARNED_ARMS
    assert (
        payload["preregistration_sha256"]
        == hashlib.sha256(PREREGISTRATION_PATH.read_bytes()).hexdigest()
    )
    verified = load_extension_contract(CONTRACT_PATH, PREREGISTRATION_PATH)
    assert verified.payload() == payload
    assert tuple(verified.payload()["arms"]) == EXTENSION_LEARNED_ARMS
    validate_extension_math_contract(verified.payload())
    with pytest.raises(TypeError, match="use load_extension_contract"):
        VerifiedExtensionContract(payload, _token=object())

    sources = payload["literature_source_binding"]
    assert sources == {
        "mp_pde_commit": "8415c5f6b6044749582bd9ebff7ecd8045ad81b2",
        "mp_pde_files": {
            "experiments/train.py": {
                "bytes": 13153,
                "sha256": "58bd330dcb5c38dbe95126e31ebe3a71caba36d72aed3281665bf8a0b1c4f1cb",
            },
            "experiments/train_helper.py": {
                "bytes": 8475,
                "sha256": "bc18ff949dbcfc25191ecd3751e272e82b743ca42ecf3ebbc1467775e10f69a9",
            },
            "common/utils.py": {
                "bytes": 11440,
                "sha256": "f2d039d65e901bdb38a65231e5a2ae9e7615a2a919c699f014dae8920992b204",
            },
        },
        "pdearena_commit": "78a8b03d50115d8bb24ce9f04efa5b920fcd4369",
        "pdearena_files": {
            "pdearena/models/pderefiner.py": {
                "bytes": 17201,
                "sha256": "f7f79d53b6bedcb4dc903133fe3e1e7f22509513fdb1ebfb14da2384b0c31131",
            },
            "setup.py": {
                "bytes": 1704,
                "sha256": "41b6db4883c33eeaf63c7606da01fc957605db1121ba51ae66007538b0b4547b",
            },
        },
        "diffusers_version": "0.17.1",
        "diffusers_scheduler_file": {
            "path": "src/diffusers/schedulers/scheduling_ddpm.py",
            "bytes": 21904,
            "sha256": "222de208eb6acf688f23a5dfe53edbf5bc162766d80bde9758ade3a05250c3b6",
        },
        "ddpm_variance_type": "fixed_small",
    }

    tampered = json.loads(json.dumps(payload))
    tampered["arms"]["PCNO_PDEREFINER_K3_VPRED"]["trained_betas"][0] *= 2
    with pytest.raises(ValueError, match="trained_betas differs"):
        validate_extension_math_contract(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["arms"]["PAIRED_RECOVERY"]["clean_loss_weight"] = 0.4
    with pytest.raises(ValueError, match="canonical payload differs"):
        validate_extension_math_contract(tampered)


def test_pushforward_schedules_and_train_boundary_caps_are_frozen() -> None:
    assert [
        maximum_available_prefix_depth(center) for center in (956, 957, 958, 959)
    ] == [0, 1, 2, 3]
    assert curriculum_exposure_probability(1) == pytest.approx(0.05)
    assert curriculum_exposure_probability(10) == pytest.approx(0.5)
    assert curriculum_exposure_probability(100) == pytest.approx(0.5)
    generator_a = torch.Generator().manual_seed(19)
    generator_b = torch.Generator().manual_seed(19)
    depths_a = [sample_mp_pde_prefix_depth(e, generator_a) for e in range(1, 20)]
    depths_b = [sample_mp_pde_prefix_depth(e, generator_b) for e in range(1, 20)]
    assert depths_a == depths_b
    assert depths_a[0] == 0
    assert set(depths_a).issubset({0, 1})
    assert 1 in depths_a
    curriculum_a = torch.Generator().manual_seed(23)
    curriculum_b = torch.Generator().manual_seed(23)
    sampled_a = [
        sample_curriculum_requested_prefix_depth(10, curriculum_a) for _ in range(60)
    ]
    sampled_b = [
        sample_curriculum_requested_prefix_depth(10, curriculum_b) for _ in range(60)
    ]
    assert sampled_a == sampled_b
    assert set(sampled_a).issubset({0, 1, 2, 3})
    assert {0, 1, 2, 3}.issubset(set(sampled_a))
    assert realize_prefix_depths(3, [956, 957, 958, 959, 1000]) == (0, 1, 2, 3, 3)
    assert realize_prefix_depths(1, [956, 957, 1000]) == (0, 1, 1)
    assert group_examples_by_prefix_depth((0, 1, 2, 3, 3)) == {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3, 4),
    }


def test_arbitrary_depth_prefix_is_aligned_and_prefix_gradients_are_stopped() -> None:
    normalization = _normalization()
    model = _ToyResidual()
    clean = torch.stack(
        [torch.full((2, 3, 5), float(index)) for index in range(1, 6)], dim=1
    )
    presentation = make_detached_prefix_presentation(
        model, clean, 2, _geometry(2, 3), normalization
    )
    state_scale = torch.tensor(normalization.state_scale, dtype=torch.float32)
    residual_scale = torch.tensor(normalization.residual_scale, dtype=torch.float32)
    generated_3 = (
        clean[:, 1]
        + model.weight.detach() * (clean[:, 1] / state_scale) * residual_scale
    )
    generated_4 = (
        generated_3
        + model.weight.detach() * (generated_3 / state_scale) * residual_scale
    )
    torch.testing.assert_close(presentation.previous, generated_3)
    torch.testing.assert_close(presentation.current, generated_4)
    torch.testing.assert_close(presentation.aligned_clean_current, clean[:, 3])
    torch.testing.assert_close(
        presentation.target_normalized_residual,
        (clean[:, 4] - generated_4) / residual_scale,
    )
    assert presentation.current.requires_grad is False
    assert presentation.prefix_model_calls == 2

    prediction = predict_normalized_residual(
        model,  # type: ignore[arg-type]
        presentation.previous,
        presentation.current,
        _geometry(2, 3),
        normalization,
    )
    torch.sum((prediction - presentation.target_normalized_residual) ** 2).backward()
    observed = model.weight.grad.detach().clone()
    reference = _ToyResidual()
    reference.weight.data.copy_(model.weight.detach())
    reference_prediction = predict_normalized_residual(
        reference,  # type: ignore[arg-type]
        presentation.previous.detach(),
        presentation.current.detach(),
        _geometry(2, 3),
        normalization,
    )
    torch.sum(
        (reference_prediction - presentation.target_normalized_residual.detach()) ** 2
    ).backward()
    torch.testing.assert_close(observed, reference.weight.grad)


def test_paired_recovery_and_relabeling_share_inputs_and_change_only_target() -> None:
    normalization = _normalization()
    previous = torch.full((2, 4, 5), 1.0)
    current = torch.full((2, 4, 5), 2.0)
    clean = torch.full((2, 4, 5), 3.0)
    solver = torch.full((2, 4, 5), 4.0)
    recovery = make_paired_displaced_presentation(
        previous, current, clean, solver, normalization, target_kind="recovery"
    )
    relabel = make_paired_displaced_presentation(
        previous,
        current,
        clean,
        solver,
        normalization,
        target_kind="dynamics_relabel",
    )
    assert recovery.previous.data_ptr() == relabel.previous.data_ptr()
    assert recovery.current.data_ptr() == relabel.current.data_ptr()
    residual_scale = torch.tensor(normalization.residual_scale, dtype=torch.float32)
    torch.testing.assert_close(
        recovery.target_normalized_residual, (clean - current) / residual_scale
    )
    torch.testing.assert_close(
        relabel.target_normalized_residual, (solver - current) / residual_scale
    )
    signs = [paired_bank_sign_for_epoch(epoch) for epoch in range(1, 101)]
    assert signs.count(-1) == 50
    assert signs.count(1) == 50
    clean_loss = torch.tensor(2.0, requires_grad=True)
    displaced_loss = torch.tensor(6.0, requires_grad=True)
    objective = paired_clean_displaced_objective(clean_loss, displaced_loss)
    assert float(objective.detach()) == pytest.approx(4.0)
    objective.backward()
    assert float(clean_loss.grad) == pytest.approx(0.5)
    assert float(displaced_loss.grad) == pytest.approx(0.5)


def test_ema_initialization_update_and_checkpoint_round_trip() -> None:
    online = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        online.weight.fill_(2.0)
        online.bias.fill_(1.0)
    ema = ExponentialMovingAverage(online, decay=0.75)
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())
    stochastic = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(p=0.5))
    stochastic_ema = ExponentialMovingAverage(stochastic)
    stochastic_ema.train()
    assert stochastic_ema.training
    assert not stochastic_ema.model.training
    assert all(not child.training for child in stochastic_ema.model.modules())
    with torch.no_grad():
        online.weight.fill_(6.0)
        online.bias.fill_(5.0)
    ema.update(online)
    torch.testing.assert_close(ema.model.weight, torch.full_like(ema.model.weight, 3.0))
    torch.testing.assert_close(ema.model.bias, torch.full_like(ema.model.bias, 2.0))
    assert int(ema.num_updates.item()) == 1
    restored = ExponentialMovingAverage(online, decay=0.2)
    restored.load_state_dict(ema.state_dict())
    assert restored.decay == pytest.approx(0.75)
    assert torch.equal(restored.model.weight, ema.model.weight)
    assert torch.equal(restored.model.bias, ema.model.bias)
    assert torch.equal(restored.num_updates, ema.num_updates)


def test_v_prediction_scheduler_closes_and_uses_exact_reverse_formula() -> None:
    scheduler = FourStepVPredictionScheduler()
    assert scheduler.betas == REFINER_BETAS
    assert scheduler.alphas_cumprod == pytest.approx(REFINER_ALPHAS_CUMPROD, abs=2e-16)
    clean = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(2))
    timesteps = torch.arange(4, dtype=torch.int64)
    presentation = make_refiner_training_presentation(
        clean, noise, timesteps, scheduler
    )
    reconstructed = scheduler.reconstruct_clean(
        presentation.noised_candidate, presentation.velocity_target, timesteps
    )
    torch.testing.assert_close(reconstructed, clean, rtol=2e-6, atol=2e-6)

    sample = torch.full((1, 2, 5), 0.75)
    velocity = torch.full_like(sample, -0.25)
    reverse_noise = torch.full_like(sample, 0.5)
    # Frozen scalar outputs from the pinned Diffusers v0.17.1 fixed-small
    # equations, including the terminal beta=1 step at t=3.
    reference_outputs = {
        0: 0.7631313809828704,
        1: 0.8032204450902395,
        2: 0.8553888697500227,
        3: 0.43418951657017946,
    }
    for timestep, scalar in reference_outputs.items():
        observed = scheduler.step(
            velocity,
            timestep,
            sample,
            noise=None if timestep == 0 else reverse_noise,
        )
        torch.testing.assert_close(
            observed,
            torch.full_like(sample, scalar),
            rtol=1e-6,
            atol=1e-6,
        )
    with pytest.raises(ValueError, match="takes no noise"):
        scheduler.step(velocity, 0, sample, noise=reverse_noise)

    generator_a = torch.Generator().manual_seed(41)
    generator_b = torch.Generator().manual_seed(41)
    sampled_a = sample_refiner_timesteps(100, generator=generator_a, device="cpu")
    sampled_b = sample_refiner_timesteps(100, generator=generator_b, device="cpu")
    assert torch.equal(sampled_a, sampled_b)
    assert set(sampled_a.tolist()) == {0, 1, 2, 3}


def test_refiner_input_parameter_cost_and_four_call_final_only_recurrence() -> None:
    normalization = _normalization()
    previous = torch.full((2, 3, 5), 1.0)
    current = torch.full((2, 3, 5), 2.0)
    candidate = torch.full((2, 3, 5), 0.25)
    timesteps = torch.tensor([0, 3], dtype=torch.int64)
    geometry = _geometry(2, 3)
    geometry["static_features"] = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    features = build_refiner_input(
        previous, current, candidate, timesteps, geometry, normalization
    )
    assert features.shape == (2, 3, 25)
    assert torch.equal(features[..., 0:6], geometry["static_features"])
    state_scale = torch.tensor(normalization.state_scale, dtype=torch.float32)
    torch.testing.assert_close(features[..., 6:11], previous / state_scale)
    torch.testing.assert_close(features[..., 11:16], current / state_scale)
    assert torch.equal(features[..., 16:21], candidate)
    assert torch.equal(features[0, 0, 21:25], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(features[1, 0, 21:25], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    baseline = NACAPCNOResidual(fourier_lengths=(2.0, 3.0))
    refiner = NACAPCNORefiner(fourier_lengths=(2.0, 3.0))
    assert refiner_parameter_difference_from_baseline(refiner, baseline) == 1152
    baseline_shapes = {
        name: tuple(parameter.shape) for name, parameter in baseline.named_parameters()
    }
    refiner_shapes = {
        name: tuple(parameter.shape) for name, parameter in refiner.named_parameters()
    }
    assert set(refiner_shapes) == set(baseline_shapes)
    assert {
        name
        for name in baseline_shapes
        if baseline_shapes[name] != refiner_shapes[name]
    } == {"backbone.fc0.weight"}
    assert torch.count_nonzero(refiner.backbone.fc2.weight) == 0
    assert torch.count_nonzero(refiner.backbone.fc2.bias) == 0
    with pytest.raises(ValueError, match="requires exact zero initialization"):
        NACAPCNORefiner(fourier_lengths=(2.0, 3.0), zero_initialize=False)

    model = _CountingVelocity()
    tape = RefinerNoiseTape(
        initial=torch.full_like(current, 0.1),
        reverse_t3=torch.full_like(current, 0.2),
        reverse_t2=torch.full_like(current, 0.3),
        reverse_t1=torch.full_like(current, 0.4),
    )
    step = refined_recurrent_step(
        model,
        previous,
        current,
        _geometry(2, 3),
        normalization,
        FourStepVPredictionScheduler(),
        tape,
    )
    assert model.calls == [3, 2, 1, 0]
    assert step.model_calls == 4
    assert len(step.intermediate_candidates) == 5
    residual_scale = torch.tensor(normalization.residual_scale, dtype=torch.float32)
    torch.testing.assert_close(
        step.next_state, current + step.final_normalized_residual * residual_scale
    )
    assert torch.equal(step.recurrent_previous, current)
    replay_model = _CountingVelocity()
    replay = refined_recurrent_step(
        replay_model,
        previous,
        current,
        _geometry(2, 3),
        normalization,
        FourStepVPredictionScheduler(),
        tape,
    )
    assert torch.equal(replay.final_normalized_residual, step.final_normalized_residual)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            replay.intermediate_candidates, step.intermediate_candidates, strict=True
        )
    )
