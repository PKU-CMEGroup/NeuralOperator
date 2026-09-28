import copy
import io
import json

import numpy as np
import pytest
import torch

from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO
from utility.time_dependent_no.pcno_kolmogorov_acdm import PeriodicVorticityACDM


def module():
    from scripts.time_dependent_no import fit_kolmogorov_acdm
    return fit_kolmogorov_acdm


def model():
    return PeriodicVorticityACDM(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)


def test_presentation_noises_both_fields_without_changing_physical_target():
    fit = module()
    network = model()
    current, target = torch.randn(8, 16, 16), torch.randn(8, 16, 16)
    originals = current.clone(), target.clone()
    noised, noise, levels = fit.presentation(network, current, target, torch.Generator().manual_seed(11))
    clean = network.schedule.reconstruct(noised, noise, levels)
    torch.testing.assert_close(clean[:, 0], current/2, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(clean[:, 1], target/2, atol=2e-6, rtol=2e-6)
    assert not torch.equal(noised[:, 0], current/2)
    assert not torch.equal(noise[:, 0], noise[:, 1])
    torch.testing.assert_close(current, originals[0], atol=0, rtol=0)
    torch.testing.assert_close(target, originals[1], atol=0, rtol=0)


def test_checkpoint_replays_next_randomized_update_and_pair_streams(tmp_path):
    fit = module()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=fit.LEARNING_RATE)
    rng = torch.Generator().manual_seed(1720)
    samplers = [fit.base.EpochSampler(16, 8, s) for s in (17, 1701)]
    batches = [(torch.randn(8, 16, 16), torch.randn(8, 16, 16)) for _ in range(2)]
    fit.update(network, optimizer, batches, rng, "cpu")
    for sampler in samplers:
        sampler.next_indices()
    path = tmp_path/"checkpoint.pt"
    digest = fit.checkpoint(path, network, optimizer, samplers, rng, {"fixture": True}, 1)
    assert fit.base.sha256(path) == digest
    saved = torch.load(path, weights_only=False)
    restored = model()
    restored.load_state_dict(saved["model"])
    optim = torch.optim.Adam(restored.parameters(), lr=fit.LEARNING_RATE)
    optim.load_state_dict(saved["optimizer"])
    restored_rng = torch.Generator().set_state(saved["noise_rng"])
    assert fit.update(network, optimizer, batches, rng, "cpu") == fit.update(restored, optim, batches, restored_rng, "cpu")
    for a, b in zip(network.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for sampler, state in zip(samplers, saved["samplers"], strict=True):
        resumed = fit.base.EpochSampler(16, 8, 999)
        resumed.load_state_dict(state)
        np.testing.assert_array_equal(sampler.next_indices(), resumed.next_indices())


def test_acquisition_replays_tapes_and_reports_realizations_separately(monkeypatch):
    fit = module()
    torch.set_num_threads(2)
    network = model().requires_grad_(False)
    train = np.random.default_rng(8).normal(size=(2, 9, 16, 16)).astype(np.float32)
    monkeypatch.setattr(fit, "PROBE_STEPS", (0, 7))
    first = fit.acquisition_probe(network, train, "cpu")
    second = fit.acquisition_probe(network, train, "cpu")
    first.pop("seconds")
    second.pop("seconds")
    assert first == second
    assert first["pairs"] == 4 and first["recurrent_calls"] == 0
    assert first["calls_per_transition"] == 20
    assert len(first["deployed_relative_l2"]) == 3
    assert len(first["per_pair"]) == 12
    assert np.isfinite(first["joint_noise_mse"]).all()


@pytest.mark.parametrize("phase", ("resource", "fit"))
def test_small_run_binds_terminal_and_never_evaluates_autonomous_states(tmp_path, monkeypatch, phase):
    fit = module()
    torch.set_num_threads(2)
    config = dict(resolution=16, modes=2, width=4, depth=1, fc_dim=8)
    parent = PeriodicVorticityPCNO(**config, train_scale=2.)
    train = np.random.default_rng(11).normal(size=(2, 9, 16, 16)).astype(np.float32)
    source = tmp_path/"input"
    source.mkdir()
    np.save(source/"train.npy", train)
    (source/"checkpoint.pt").write_bytes(b"fixture")
    manifest = dict(artifacts={n: fit.base.sha256(source/n) for n in ("train.npy", "checkpoint.pt")})
    fit.base.write_json(source/"manifest.json", manifest)
    teacher = io.BytesIO()
    np.savez(teacher, sentinel_input=train[:, 0])
    metadata = dict(model_config=config, checkpoint_identity={"train_scale_float64": 2.})
    digest = fit.base.sha256(source/"manifest.json")
    # Substitute authentication of the large private input, preserving the real
    # network, training, sampler, checkpoint and acquisition implementation.
    monkeypatch.setattr(fit.base, "load_inputs", lambda _p: (manifest, {"teacher.npz": teacher.getvalue()}, metadata, train, digest))
    monkeypatch.setattr(fit.base, "load_model", lambda *_a: copy.deepcopy(parent))
    monkeypatch.setattr(fit.base, "terminal_replay", lambda *_a: {"passed": True, "synthetic": True})
    monkeypatch.setattr(fit, "FIT_UPDATES", 16)
    monkeypatch.setattr(fit, "PROBE_INTERVAL", 8)
    monkeypatch.setattr(fit, "PROBE_STEPS", (0, 7))
    result = fit.run(source, tmp_path/phase, phase, "cpu")
    assert result["status"] == "completed" and result["updates_completed"] == 16
    assert result["protected_access"] is result["development_access"] is False
    assert result["autonomous_steps"] == 0
    assert json.loads((tmp_path/phase/"result.json").read_text()) == result
    for name, digest in result["artifacts"].items():
        assert fit.base.sha256(tmp_path/phase/name) == digest
    if phase == "fit":
        saved = torch.load(tmp_path/phase/"terminal.pt", weights_only=False)
        assert saved["identity"] == result["identity"] and saved["update"] == 16
        assert [p["update"] for p in result["probes"]] == [8, 16]
        assert not (tmp_path/phase/"latest.pt").exists()
