import copy

import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_spectral_control as runner
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO
from utility.time_dependent_no.pcno_kolmogorov_spectral_control import (
    BoundedSpectralConv, MatrixNormCap, restrict_channel_maps,
)


def test_norm_cap_preserves_small_maps_and_bounds_large_maps_with_gradients():
    w = torch.diag(torch.tensor([3., .5])).requires_grad_()
    effective = MatrixNormCap()(w)
    assert float(torch.linalg.matrix_norm(effective.detach(), ord=2)) == pytest.approx(1.)
    effective.square().sum().backward()
    assert torch.isfinite(w.grad).all()
    small = torch.eye(3) * .4
    torch.testing.assert_close(MatrixNormCap()(small), small, atol=0, rtol=0)


def test_complex_mode_caps_and_identity_when_inactive():
    torch.set_num_threads(2)
    model = PeriodicVorticityPCNO(resolution=16, train_scale=2., modes=1, width=3, depth=2, fc_dim=4)
    # Make every cap inactive, so an independent ordinary PCNO must be recovered.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name:
                p.mul_(.01)
    ordinary = copy.deepcopy(model)
    restrict_channel_maps(model)
    x = torch.randn(2, 16, 16)
    torch.testing.assert_close(model(x)['next_state'], ordinary(x)['next_state'], rtol=2e-6, atol=2e-6)
    layer = model.pcno.sp_convs[0]
    assert isinstance(layer, BoundedSpectralConv)
    with torch.no_grad():
        layer.weights_c.mul_(1e5)
        layer.weights_s.mul_(1e5)
    weight, zero = layer.effective_weights()
    assert float(torch.linalg.matrix_norm(weight.detach().permute(2, 3, 0, 1), ord=2).max()) <= 1 + 2e-6
    assert float(torch.linalg.matrix_norm(zero.detach().permute(2, 3, 0, 1), ord=2).max()) <= 1 + 2e-6
    model(x)['next_state'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_small_fit_has_exact_reload_and_same_clean_indices(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'FIT_UPDATES', 3)
    output = tmp_path / 'fit'
    result = runner.fit(packet, output, 'fit', 'cpu')
    assert result['status'] == 'completed'
    manifest, captured, parent, _, train_hash = runner.base.load_inputs(packet)
    model, _, _ = runner.load_fitted(captured, parent, manifest, train_hash, output, 'cpu')
    checkpoint = torch.load(output / 'terminal.pt', weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint['model'][key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in model.parameters())
    import json
    rows = [json.loads(line) for line in (output / 'updates.jsonl').read_text().splitlines()]
    samplers = [runner.base.EpochSampler(8, 2, seed) for seed in (17, 1701)]
    for row in rows:
        assert row['first_indices'] == samplers[0].next_indices().tolist()
        assert row['second_indices'] == samplers[1].next_indices().tolist()
