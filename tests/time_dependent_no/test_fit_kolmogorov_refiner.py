import copy
import io
import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_refiner as fit
from utility.time_dependent_no.pcno_kolmogorov_refiner import PeriodicVorticityRefiner


def model():
    return PeriodicVorticityRefiner(16, train_scale=2., modes=2, width=4, depth=1, fc_dim=8)


def test_training_presentation_uses_clean_condition_and_archived_successor():
    network = model()
    current, target = torch.randn(8, 16, 16), torch.randn(8, 16, 16)
    original = current.clone()
    generator = torch.Generator().manual_seed(11)
    candidate, velocity, levels = fit.presentation(network, current, target, generator)
    recovered = network.schedule.reconstruct(candidate, velocity, levels)
    torch.testing.assert_close(recovered, (target-current)/network.train_scale, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(current, original, rtol=0, atol=0)
    assert len(levels.unique()) > 1


def test_ema_and_saved_optimizer_rng_reproduce_next_stochastic_update(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    network = model()
    averaged = copy.deepcopy(network).requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=fit.LEARNING_RATE)
    generator = torch.Generator().manual_seed(1719)
    samplers = [fit.base.EpochSampler(16, 8, s) for s in (17, 1701)]
    batches = [(torch.randn(8, 16, 16), torch.randn(8, 16, 16)) for _ in range(2)]
    old_ema = {n: p.detach().clone() for n, p in averaged.named_parameters()}
    fit.update(network, averaged, optimizer, batches, generator, "cpu")
    for name, value in network.named_parameters():
        expected = fit.EMA_DECAY*old_ema[name]+(1-fit.EMA_DECAY)*value
        torch.testing.assert_close(dict(averaged.named_parameters())[name], expected)
    for sampler in samplers:
        sampler.next_indices()
    path = tmp_path/"checkpoint.pt"
    digest = fit.checkpoint(path, network, averaged, optimizer, samplers, generator, {"fixture": True}, 1)
    assert digest == fit.base.sha256(path)
    saved = torch.load(path, weights_only=False)
    restored, restored_ema = model(), model().requires_grad_(False)
    restored.load_state_dict(saved["model"])
    restored_ema.load_state_dict(saved["ema"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=fit.LEARNING_RATE)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_rng = torch.Generator()
    restored_rng.set_state(saved["noise_rng"])
    left = fit.update(network, averaged, optimizer, batches, generator, "cpu")
    right = fit.update(restored, restored_ema, restored_optimizer, batches, restored_rng, "cpu")
    assert left == right
    for a, b in zip(averaged.parameters(), restored_ema.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for original, state in zip(samplers, saved["samplers"], strict=True):
        resumed = fit.base.EpochSampler(16, 8, 999)
        resumed.load_state_dict(state)
        np.testing.assert_array_equal(original.next_indices(), resumed.next_indices())


def test_acquisition_uses_only_clean_inputs_and_repeats_noise_law(monkeypatch):
    torch.set_num_threads(2)
    network = model().requires_grad_(False)
    train = np.random.default_rng(8).normal(size=(2, 257, 16, 16)).astype(np.float32)
    monkeypatch.setattr(fit, "PROBE_STEPS", (0, 7))
    first = fit.acquisition_probe(network, train, "cpu")
    second = fit.acquisition_probe(network, train, "cpu")
    assert first.pop("seconds") > 0
    assert second.pop("seconds") > 0
    assert first == second
    assert first["pairs"] == 4 and first["recurrent_calls"] == 0
    assert {r["input_step"] for r in first["per_pair"]} == {0, 7}
    assert len(first["per_pair"]) == 4*3
    assert np.isfinite(first["level_velocity_mse"]).all()


@pytest.mark.parametrize("phase", ("resource", "fit"))
def test_small_run_closes_artifacts_and_retains_continuation_state(tmp_path, monkeypatch, phase):
    torch.set_num_threads(2)
    from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO
    config = dict(resolution=16, modes=2, width=4, depth=1, fc_dim=8)
    network = PeriodicVorticityPCNO(**config, train_scale=2.)
    train = np.random.default_rng(11).normal(size=(2, 9, 16, 16)).astype(np.float32)
    source = tmp_path/"input"
    source.mkdir()
    np.save(source/"train.npy", train)
    manifest = dict(artifacts={"train.npy": fit.base.sha256(source/"train.npy"),
                               "checkpoint.pt": "fixture"})
    # This synthetic loader substitutes only authentication, not fitting/probes.
    (source/"checkpoint.pt").write_bytes(b"fixture")
    manifest["artifacts"]["checkpoint.pt"] = fit.base.sha256(source/"checkpoint.pt")
    fit.base.write_json(source/"manifest.json", manifest)
    teacher = io.BytesIO()
    np.savez(teacher, sentinel_input=train[:, 0])
    captured = {"teacher.npz": teacher.getvalue()}
    metadata = dict(model_config=config, checkpoint_identity={"train_scale_float64": 2.})
    digest = fit.base.sha256(source/"manifest.json")
    monkeypatch.setattr(fit.base, "load_inputs", lambda _p: (manifest, captured.copy(), metadata, train, digest))
    monkeypatch.setattr(fit.base, "load_model", lambda *_args: copy.deepcopy(network))
    monkeypatch.setattr(fit.base, "terminal_replay", lambda *_args: {"passed": True, "synthetic": True})
    monkeypatch.setattr(fit, "FIT_UPDATES", 16)
    monkeypatch.setattr(fit, "PROBE_INTERVAL", 8)
    monkeypatch.setattr(fit, "PROBE_STEPS", (0, 7))
    output = tmp_path/phase
    result = fit.run(source, output, phase, "cpu")
    assert result["status"] == "completed" and result["updates_completed"] == 16
    assert result["stopped_before_development_and_autonomous_evaluation"] is True
    assert result["transplant_rms_scaled"] < 1e-6
    assert json.loads((output/"result.json").read_text()) == result
    for name, expected in result["artifacts"].items():
        assert fit.base.sha256(output/name) == expected
    if phase == "fit":
        saved = torch.load(output/"terminal.pt", weights_only=False)
        assert saved["identity"] == result["identity"] and saved["update"] == 16
        assert {"model", "ema", "optimizer", "noise_rng", "samplers"} <= saved.keys()
        assert not (output/"latest.pt").exists()
        assert [p["update"] for p in result["probes"]] == [8, 16]
    else:
        assert not (output/"terminal.pt").exists()
