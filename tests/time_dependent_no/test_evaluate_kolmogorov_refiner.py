import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_refiner as run


class FourNoiseSampler:
    def transition(self, value, tape):
        assert len(tape) == 4
        noise = sum((i+1)*v for i, v in enumerate(tape))
        nxt = value+.01*noise
        return dict(raw_next=nxt+10, next_state=nxt)


def test_refiner_adapter_shares_all_four_noises_and_refreshes_each_step(monkeypatch):
    common = run.common
    values = np.stack([np.full((16, 16), v, np.float32) for v in (1., 1.2, .8)])
    _, samples = common.shared_predictions(FourNoiseSampler(), values, 101, "cpu", run.noise_tape)
    np.testing.assert_allclose(samples[1]-samples[0], .2, atol=2e-7)
    np.testing.assert_allclose(samples[2]-samples[0], -.2, atol=2e-7)
    repeat = common.shared_predictions(FourNoiseSampler(), values, 101, "cpu", run.noise_tape)[1]
    np.testing.assert_array_equal(samples, repeat)
    monkeypatch.setattr(common.base, "HORIZONS", (2, 4))
    monkeypatch.setattr(common.base, "SNAPSHOTS", set(range(5)))
    truth = np.ones((5, 16, 16), np.float32)
    adapter = common.SampledTransition(FourNoiseSampler(), 101, run.noise_tape)
    case, snapshots = common.base.rollout_case(adapter, truth, 1., "cpu")
    assert case["status"] == "completed" and adapter.calls == 4
    # Raw outputs are deliberately far away; only restricted states recur.
    assert snapshots["state"].max() < 2
    increments = np.diff(snapshots["state"], axis=0)
    assert not np.array_equal(increments[0], increments[1])


def test_actual_refiner_adapter_uses_four_network_calls_and_exact_tape():
    model = run.PeriodicVorticityRefiner(resolution=16, width=4, depth=1, modes=2,
                                       fc_dim=8, train_scale=1.).eval()
    values = torch.ones(1, 16, 16)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    actual = run.common.SampledTransition(model, 101, run.noise_tape)(values)
    assert len(calls) == 4
    hook.remove()
    with torch.no_grad():
        expected = model.transition(values, run.noise_tape(values, torch.Generator().manual_seed(101)))
    torch.testing.assert_close(actual["next_state"], expected["next_state"], rtol=0, atol=0)


@pytest.fixture
def saved_fit(tmp_path):
    config = dict(resolution=16, width=4, depth=1, modes=2, fc_dim=8)
    model = run.PeriodicVorticityRefiner(**config, train_scale=2.)
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    raw = {k: v.clone() for k, v in ema.items()}
    raw["pcno.fc0.bias"] += 1
    manifest = dict(artifacts={"checkpoint.pt": "parent"})
    identity = dict(updates=16384, initial_update=8192, recipe=run.fit.RECIPE,
        deployment=run.DEPLOYMENT, input_manifest_sha256="manifest", model_config=config,
        train_scale=2., parent_checkpoint_sha256="parent")
    torch.save(dict(identity=identity, update=16384, model=raw, ema=ema), tmp_path/"terminal.pt")
    result = dict(status="completed", updates_completed=16384, identity=identity,
        input_manifest=manifest, sources={n: run.common.base.sha256(run.common.base.ROOT/n)
                                         for n in run.fit.SOURCE_PATHS},
        artifacts={"terminal.pt": run.common.base.sha256(tmp_path/"terminal.pt")})
    (tmp_path/"result.json").write_text(json.dumps(result))
    return tmp_path, manifest, dict(model_config=config), ema, result


def test_loader_selects_ema_not_raw_and_freezes_parameters(saved_fit):
    path, manifest, parent, expected, _ = saved_fit
    model, _ = run.load_model(path, manifest, parent, "manifest", "cpu")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert not model.training and not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("corruption", ("deployment", "updates", "artifact"))
def test_loader_rejects_changed_map_or_checkpoint(saved_fit, corruption):
    path, manifest, parent, _, result = saved_fit
    if corruption == "deployment":
        result["identity"]["deployment"] = "raw"
    elif corruption == "updates":
        result["updates_completed"] = 8192
    else:
        result["artifacts"]["terminal.pt"] = "changed"
    (path/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        run.load_model(path, manifest, parent, "manifest", "cpu")


def test_internal_nonfinite_refinement_is_censored_but_other_errors_propagate(monkeypatch):
    monkeypatch.setattr(run.common.base, "HORIZONS", (2, 4))
    class InvalidSampler:
        message = "candidate must be a finite aligned field"
        def transition(self, value, tape):
            raise ValueError(self.message)
    model = InvalidSampler()
    adapter = run.common.SampledTransition(model, 101, run.noise_tape)
    case, _ = run.common.base.rollout_case(adapter, np.ones((5, 16, 16), np.float32), 1., "cpu")
    assert case["status"] == "nonfinite_prediction" and case["failed_at_step"] == 1
    assert all(not horizon["complete"] for horizon in case["horizons"])
    model.message = "unrelated alignment error"
    with pytest.raises(ValueError, match="unrelated alignment"):
        adapter(torch.ones(1, 16, 16))
