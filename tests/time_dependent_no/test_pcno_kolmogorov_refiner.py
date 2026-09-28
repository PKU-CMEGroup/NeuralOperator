import math

import pytest
import torch

from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO, canonicalize_vorticity
from utility.time_dependent_no.pcno_kolmogorov_refiner import (
    PeriodicVorticityRefiner, RefinerSchedule, noise_tape,
)


def tiny_model():
    return PeriodicVorticityRefiner(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)


def test_schedule_recovers_exact_clean_sample_at_every_training_level():
    schedule = RefinerSchedule()
    assert schedule.betas[0] == pytest.approx(4e-7)
    assert schedule.alphas_cumprod[-1] == 0
    generator = torch.Generator().manual_seed(41)
    clean, noise = [torch.randn(4, 16, 16, dtype=torch.float64, generator=generator) for _ in range(2)]
    levels = torch.arange(4)
    sample, velocity = schedule.presentation(clean, noise, levels)
    torch.testing.assert_close(schedule.reconstruct(sample, velocity, levels), clean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sample[3], noise[3], rtol=0, atol=0)
    torch.testing.assert_close(velocity[3], -clean[3], rtol=0, atol=0)


def test_float32_final_reverse_step_uses_the_training_noise_scale():
    schedule = RefinerSchedule()
    generator = torch.Generator().manual_seed(15)
    clean = .01*torch.randn(2, 16, 16, generator=generator)
    noise = torch.randn(clean.shape, generator=generator)
    sample, velocity = schedule.presentation(clean, noise, torch.zeros(2, dtype=torch.int64))
    recovered = schedule.step(sample, velocity, 0, None)
    torch.testing.assert_close(recovered, clean, atol=1e-7, rtol=1e-6)


def test_reverse_step_matches_gaussian_posterior_and_oracle_finishes_exactly():
    schedule = RefinerSchedule()
    generator = torch.Generator().manual_seed(72)
    clean = torch.randn(2, 16, 16, dtype=torch.float64, generator=generator)
    sample = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    for level in (3, 2, 1, 0):
        a = schedule.alphas_cumprod[level]
        previous_a = schedule.alphas_cumprod[level-1] if level else 1.
        # Independent oracle in the conditional Gaussian forward process.
        velocity = (math.sqrt(a)*sample-clean)/math.sqrt(1-a)
        noise = torch.randn(sample.shape, dtype=sample.dtype, generator=generator) if level else None
        actual = schedule.step(sample, velocity, level, noise)
        beta = schedule.betas[level]
        expected = math.sqrt(previous_a)*beta/(1-a)*clean
        expected += math.sqrt(1-beta)*(1-previous_a)/(1-a)*sample
        if level:
            expected += math.sqrt((1-previous_a)*beta/(1-a))*noise
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        sample = actual
    torch.testing.assert_close(sample, clean, rtol=2e-10, atol=2e-10)


def test_parent_transplant_preserves_level3_increment_and_training_reaches_new_features():
    torch.manual_seed(13)
    parent = PeriodicVorticityPCNO(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)
    model = tiny_model()
    model.initialize_from_parent(parent)
    current = canonicalize_vorticity(torch.randn(4, 16, 16))
    candidate = torch.randn_like(current)
    levels = torch.full((4,), 3, dtype=torch.int64)
    velocity = model(current, candidate, levels)
    torch.testing.assert_close(current-model.train_scale*velocity, parent(current)["raw_next"], atol=1e-6, rtol=1e-6)
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in parent.parameters()) == 5*4
    mixed = torch.arange(4)
    model(current, candidate, mixed).square().mean().backward()
    assert torch.isfinite(model.pcno.fc0.weight.grad).all()
    assert model.pcno.fc0.weight.grad[:, 2:].abs().sum() > 0
    model.double()
    assert model._fourier_cache is None
    assert model(current.double(), candidate.double(), mixed).dtype == torch.float64


def test_complete_transition_uses_four_calls_shared_tape_and_only_final_restriction():
    torch.manual_seed(34)
    model = tiny_model()
    current = torch.randn(1, 16, 16)
    generator = torch.Generator().manual_seed(19)
    tape = noise_tape(current, generator)
    seen = []
    hook = model.register_forward_pre_hook(lambda _m, args: seen.append(int(args[2][0])))
    first = model.transition(current, tape)
    hook.remove()
    second = model.transition(current, tape)
    assert seen == [3, 2, 1, 0]
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    torch.testing.assert_close(first["next_state"], canonicalize_vorticity(first["raw_next"]))
    # Fresh randomness changes a realized map; reusing it does not consume RNG.
    different = model.transition(current, noise_tape(current, generator))
    assert not torch.equal(first["raw_next"], different["raw_next"])
    with pytest.raises(ValueError, match="four finite"):
        model.transition(current, tape[:3])


def test_rejects_invalid_levels_and_reverse_noise():
    model = tiny_model()
    state = torch.zeros(1, 16, 16)
    with pytest.raises(ValueError, match="levels"):
        model(state, state, torch.tensor([4]))
    with pytest.raises(ValueError, match="deterministic"):
        model.schedule.step(state, state, 0, state)
    with pytest.raises(ValueError, match="aligned noise"):
        model.schedule.step(state, state, 3, None)
