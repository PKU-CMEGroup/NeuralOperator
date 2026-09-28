import json

import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_dissipative as fit
from scripts.time_dependent_no import evaluate_kolmogorov_dissipative as evaluate
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet
from utility.time_dependent_no.kolmogorov_dissipative import (
    CONTRACTION, RMS_BOUND, SHELL_INNER, SHELL_OUTER, SHELL_WEIGHT,
    EnvelopeProjection, shell_samples, shell_error,
)
from utility.time_dependent_no.pcno_kolmogorov import canonicalize_vorticity


class ScalarMap(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(factor))
        self.register_buffer('train_scale', torch.tensor(2.))

    def forward(self, x):
        y = self.factor * x
        return {'next_state': y, 'raw_next': y}


def test_shell_law_radius_subspace_and_replay():
    torch.set_num_threads(2)
    before = torch.get_rng_state().clone()
    x = shell_samples((2048, 16, 16), torch.Generator().manual_seed(1703))
    assert torch.equal(before, torch.get_rng_state())
    radius = x.double().square().mean((-1, -2)).sqrt()
    assert float(radius.min()) >= SHELL_INNER * (1 - 1e-6)
    assert float(radius.max()) <= SHELL_OUTER * (1 + 1e-6)
    # Uniform radius differs sharply from volume-uniform sampling in high dimension.
    scaled = (radius - SHELL_INNER) / (SHELL_OUTER - SHELL_INNER)
    assert abs(float(scaled.mean()) - .5) < .025
    assert abs(float(scaled.var()) - 1 / 12) < .01
    torch.testing.assert_close(x, canonicalize_vorticity(x), atol=7e-5, rtol=2e-6)
    torch.testing.assert_close(x, shell_samples(x.shape, torch.Generator().manual_seed(1703)), rtol=0, atol=0)
    assert float(shell_error(CONTRACTION * x, x)) == 0


def test_projection_preserves_inside_and_contracts_to_ball():
    torch.manual_seed(6)
    x = canonicalize_vorticity(torch.randn(2, 16, 16))
    x[1] *= 100
    wrapper = EnvelopeProjection(ScalarMap(1.), record_calls=True)
    y = wrapper(x)
    torch.testing.assert_close(y['next_state'][0], x[0], atol=0, rtol=0)
    assert float(y['next_state'][1].detach().double().square().mean().sqrt()) == pytest.approx(RMS_BOUND, rel=1e-7)
    torch.testing.assert_close(y['raw_next'], x)
    torch.testing.assert_close(y['next_state'], canonicalize_vorticity(y['next_state']), atol=2e-5, rtol=2e-6)
    target = x / 100
    assert torch.all((y['next_state'] - target).square().sum((-2, -1)) <= (x - target).square().sum((-2, -1)))
    assert wrapper.calls[0]['factors'][0] == 1
    assert wrapper.calls[0]['factors'][1] < 1


def test_projection_does_not_hide_nonfinite_proposal():
    with pytest.raises(FloatingPointError, match='nonfinite'):
        EnvelopeProjection(ScalarMap(float('inf')))(torch.ones(1, 8, 8))


def test_shell_gradient_and_clean_exposure_match_declared_objective():
    a, b = ScalarMap(.9), ScalarMap(.9)
    batches = [(torch.ones(2, 4, 4), torch.full((2, 4, 4), .8)),
               (torch.full((2, 4, 4), 2.), torch.full((2, 4, 4), 1.7))]
    shell = torch.full((2, 4, 4), 80.)
    opt_a, opt_b = [torch.optim.SGD(m.parameters(), lr=.03) for m in (a, b)]
    objective = sum(.5 * ((b(x)['next_state'] - y) / 2).square().mean() for x, y in batches)
    objective = objective + SHELL_WEIGHT * ((b.factor - .5) ** 2)
    objective.backward()
    opt_b.step()
    row = fit.update(a, opt_a, batches, shell, 'cpu')
    torch.testing.assert_close(a.factor, b.factor, rtol=0, atol=0)
    assert row['loss'] == pytest.approx(float(objective.detach()))


def test_small_fit_reload_and_terminal_rejection(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'FIT_UPDATES', 3)
    output = tmp_path / 'fit'
    result = fit.run(packet, output, 'fit', 'cpu')
    assert result['status'] == 'completed' and result['updates_completed'] == 3
    manifest, captured, parent, _, train_hash = fit.base.load_inputs(packet)
    model, _, _ = evaluate.load_shell(captured, parent, manifest, train_hash, output, 'cpu')
    observed = fit.shell_probe(model, 16, 'cpu')
    assert observed == result['shell_after']
    assert not any(p.requires_grad for p in model.parameters())
    result['updates_completed'] = 2
    (output / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='identity/terminal/scope'):
        evaluate.load_shell(captured, parent, manifest, train_hash, output, 'cpu')
