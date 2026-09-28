import math

import pytest
import torch

from utility.time_dependent_no.pcno_kolmogorov import (
    PeriodicVorticityPCNO, canonicalize_vorticity,
)


def module():
    from utility.time_dependent_no import pcno_kolmogorov_acdm
    return pcno_kolmogorov_acdm


def test_diffusion_noise_decodes_joint_clean_fields_and_last_step_has_no_noise():
    schedule = module().ACDMSchedule()
    clean = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2)/8
    epsilon = torch.flip(clean, (-1,))-.7
    levels = torch.tensor([0, 19])
    sample = schedule.add_noise(clean, epsilon, levels)
    torch.testing.assert_close(schedule.reconstruct(sample, epsilon, levels), clean,
                               atol=1e-12, rtol=1e-12)
    # The lowest forward variance is .0025; the posterior returns x0 exactly
    # with an oracle epsilon and has no additional random contribution.
    first = math.sqrt(.9975)*clean + .05*epsilon
    torch.testing.assert_close(schedule.step(first, epsilon, 0, None), clean,
                               atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        schedule.step(first, epsilon, 0, torch.zeros_like(first))


def test_reverse_step_matches_independent_posterior_formula():
    schedule = module().ACDMSchedule()
    betas = [.0025+(.5-.0025)*i/19 for i in range(20)]
    level = 9
    abar = math.prod(1-b for b in betas[:level+1])
    prev = math.prod(1-b for b in betas[:level])
    sample = torch.tensor([[[.4, -.2]]], dtype=torch.float64)
    epsilon, noise = sample+.3, sample-.7
    clean = (sample-math.sqrt(1-abar)*epsilon)/math.sqrt(abar)
    expected = (math.sqrt(prev)*betas[level]/(1-abar)*clean
        + math.sqrt(1-betas[level])*(1-prev)/(1-abar)*sample
        + math.sqrt(betas[level]*(1-prev)/(1-abar))*noise)
    torch.testing.assert_close(schedule.step(sample, epsilon, level, noise), expected,
                               atol=1e-12, rtol=1e-12)


def test_deployment_refreshes_condition_and_recovers_oracle_successor():
    acdm = module()
    torch.set_num_threads(2)
    network = acdm.PeriodicVorticityACDM(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)
    current = canonicalize_vorticity(torch.randn(2, 16, 16))
    target = canonicalize_vorticity(torch.randn_like(current))
    tape = acdm.noise_tape(current, torch.Generator().manual_seed(10))
    original = current.clone()
    levels_seen = []

    def oracle(joint, levels):
        signal, sigma = network.schedule.factors(levels, joint)
        expected_condition = signal[:, 0]*(current/2)+sigma[:, 0]*tape[1]
        torch.testing.assert_close(joint[:, 0], expected_condition, atol=0, rtol=0)
        levels_seen.append(int(levels[0]))
        # Deliberately wrong conditioning prediction must never be fed back.
        return torch.stack((torch.full_like(current, 1e3),
            (joint[:, 1]-signal[:, 0]*(target/2))/sigma[:, 0]), 1)

    network.forward = oracle
    output = network.transition(current, tape)
    torch.testing.assert_close(output["next_state"], target, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(current, original, atol=0, rtol=0)
    assert levels_seen == list(range(19, -1, -1))


def test_shared_tape_replay_and_final_restriction():
    acdm = module()
    torch.set_num_threads(2)
    network = acdm.PeriodicVorticityACDM(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8).eval()
    current = torch.randn(2, 16, 16)
    tape = acdm.noise_tape(current, torch.Generator().manual_seed(11))
    with torch.no_grad():
        first, second = network.transition(current, tape), network.transition(current, tape)
    torch.testing.assert_close(first["raw_next"], second["raw_next"], atol=0, rtol=0)
    torch.testing.assert_close(first["next_state"], canonicalize_vorticity(first["raw_next"]), atol=0, rtol=0)
    torch.testing.assert_close(first["next_state"]+first["projection_residual"], first["raw_next"])
    with pytest.raises(ValueError):
        network.transition(current, tape[:-1])


def test_parent_body_transfer_has_zero_joint_noise_output_and_trains():
    acdm = module()
    torch.set_num_threads(2)
    config = dict(resolution=16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)
    parent = PeriodicVorticityPCNO(**config)
    network = acdm.PeriodicVorticityACDM(**config)
    network.initialize_from_parent(parent)
    joint, levels = torch.randn(2, 2, 16, 16), torch.tensor([0, 19])
    output = network(joint, levels)
    torch.testing.assert_close(output, torch.zeros_like(joint), atol=0, rtol=0)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    (output-1).square().mean().backward()
    optimizer.step()
    assert float(network(joint, levels).detach().abs().sum()) > 0
