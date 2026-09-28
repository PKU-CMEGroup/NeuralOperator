from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_tangent_forecast as runner
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO


@pytest.fixture
def packet(tmp_path):
    torch.manual_seed(19)
    directory = tmp_path / "input"
    directory.mkdir()
    config = dict(resolution=16, width=2, depth=1, modes=1, fc_dim=0)
    model = PeriodicVorticityPCNO(**config, train_scale=1.0).eval()
    identity = dict(model_config=config, train_scale_float64=1.0,
                    train_scale_model_float32=1.0)
    torch.save(dict(identity=identity, update=49152, schedule_position=49152,
                    model=model.state_dict()), directory / "checkpoint.pt")
    fit = dict(model_config=config, checkpoint_identity=identity, updates_completed=49152,
               terminal_checkpoint=dict(file="terminal_049152.pt", update=49152,
                                        sha256=runner.sha256(directory / "checkpoint.pt")))
    runner.write_json(directory / "fit_result.json", fit)
    x = torch.arange(16) * (2 * torch.pi / 16)
    state = torch.sin(x)[:, None].expand(16, 16).clone()
    np.savez(directory / "reference.npz", states=state.repeat(5, 1, 1).numpy())
    with torch.no_grad():
        prediction = model(state[None])
    np.savez(directory / "teacher.npz", sentinel_global_indices=np.array([0]),
             sentinel_input=state[None].numpy(),
             sentinel_raw=prediction["raw_next"].numpy(),
             sentinel_next=prediction["next_state"].numpy())
    manifest = dict(schema_version=1, role="train", seed=2026090611, start_step=0,
                    horizon=4, model_name="clean32", parent_bindings={},
                    artifacts={f: runner.sha256(directory / f) for f in runner.INPUT_FILES},
                    sources={f: runner.sha256(runner.ROOT / f) for f in runner.MODEL_SOURCES})
    runner.write_json(directory / "manifest.json", manifest)
    return directory


def test_hash_failure_precedes_array_or_checkpoint_decode(packet, monkeypatch):
    with (packet / "reference.npz").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("decoded before hashes"))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded before hashes"))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runner.load_inputs(packet)


@pytest.mark.parametrize("key,value", [("role", "development"), ("seed", 2026090621),
                                      ("start_step", 1), ("horizon", 33)])
def test_scope_rejected_before_decode(packet, monkeypatch, key, value):
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest[key] = value
    runner.write_json(packet / "manifest.json", manifest)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("scope opened arrays"))
    with pytest.raises(ValueError, match="input scope"):
        runner.load_inputs(packet)


def test_checkpoint_schedule_and_strict_state_dictionary(packet):
    _, captured, fit, _, _ = runner.load_inputs(packet)
    checkpoint = torch.load(packet / "checkpoint.pt", weights_only=False)
    checkpoint["schedule_position"] = 0
    import io
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    captured["checkpoint.pt"] = buffer.getvalue()
    with pytest.raises(ValueError, match="identity/schedule"):
        runner.load_model(captured, fit, "cpu")
    checkpoint["schedule_position"] = 49152
    checkpoint["model"]["unexpected"] = torch.ones(1)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    captured["checkpoint.pt"] = buffer.getvalue()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        runner.load_model(captured, fit, "cpu")


def test_clean8_historical_terminal_schema(packet):
    fit = json.loads((packet / "fit_result.json").read_text())
    terminal = fit.pop("terminal_checkpoint")
    terminal.pop("update")
    fit.update(accepted_checkpoint="terminal_049152.pt", terminal_checkpoints={"49152": terminal})
    runner.write_json(packet / "fit_result.json", fit)
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["model_name"] = "clean8"
    manifest["artifacts"]["fit_result.json"] = runner.sha256(packet / "fit_result.json")
    runner.write_json(packet / "manifest.json", manifest)
    loaded, _, _, reference, _ = runner.load_inputs(packet)
    assert loaded["model_name"] == "clean8"
    assert len(reference) == 5


def test_development_sentinel_rejected_before_forward():
    import io
    buffer = io.BytesIO()
    np.savez(buffer, sentinel_global_indices=np.array([32 * 512]))
    with pytest.raises(ValueError, match="training-only"):
        runner.terminal_replay(lambda _: pytest.fail("opened development sentinel"),
                               buffer.getvalue(), "cpu", 32)


def test_actual_small_pcno_checkpoint_replay_and_jvp(packet, tmp_path):
    result = runner.run(packet, tmp_path / "output")
    assert result["status"] == "completed", result.get("error")
    assert result["jvp_validation"]["passed"]
    assert result["observed_steps_retained"] == 4
    assert result["per_step"][1]["forecast_mismatch_rms_scaled"] < 1e-6
    assert result["phases"][0]["phase"] == "forecast_frozen"
    assert result["phases"][1]["phase"] == "rollout_completed"
    assert len(result["physical"]) == 5
    assert result["probes"] == []


@pytest.mark.parametrize("fail_rollout", [False, True])
def test_forecast_freezes_before_any_rollout_and_failure_retains_prefix(
        packet, tmp_path, monkeypatch, fail_rollout):
    output = tmp_path / "output"
    finished = False
    observed_calls = 0

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("train_scale", torch.tensor(1.0))

        def forward(self, state):
            nonlocal observed_calls
            if finished:
                assert (output / "forecast_freeze.json").exists()
                freeze = json.loads((output / "forecast_freeze.json").read_text())
                assert freeze["sha256"] == runner.sha256(output / "forecast.npz")
                observed_calls += 1
            value = 0.9 * state + 0.1
            if fail_rollout and observed_calls == 2:
                value = value * float("nan")
            return {"raw_next": value, "next_state": value}

    original = runner.forced_tangent_forecast

    def forecast(step, reference):
        nonlocal finished
        result = original(step, reference)
        finished = True
        return result

    monkeypatch.setattr(runner, "load_model", lambda *args: Affine())
    monkeypatch.setattr(runner, "terminal_replay", lambda *args: {})
    monkeypatch.setattr(runner, "forced_tangent_forecast", forecast)
    result = runner.run(packet, output)
    assert result["status"] == ("incomplete" if fail_rollout else "completed")
    with np.load(output / "observed.npz") as saved:
        assert len(saved["states"]) == (3 if fail_rollout else 5)
    if fail_rollout:
        assert result["phase"] == "nonlinear_rollout"
        assert result["error"]["type"] == "FloatingPointError"
        assert "skill_vs_identity_excluding_step_1" not in result
    else:
        assert result["skill_vs_identity_excluding_step_1"] == pytest.approx(1.0, abs=1e-10)
        assert observed_calls == 4
    with pytest.raises(FileExistsError):
        runner.run(packet, output)


def test_skill_undefined_when_identity_is_exact():
    errors = torch.zeros(4, 2, 2)
    forecast = dict(predicted_error=errors.clone(), identity_response_error=errors.clone(),
                    clean_forcing=errors[:-1].clone())
    rows, skill = runner.compare_forecast(forecast, errors, 1.0)
    assert skill is None
    assert rows[1]["forcing_propagated_cosine"] is None


def test_transition_directions_share_absolute_amplitudes_and_recenter_base(monkeypatch):
    state = torch.zeros(16, 16)
    direction = torch.linspace(-1, 1, 256).reshape(16, 16)
    original = runner.finite_amplitude_response

    def biased_primal(*args):
        result = original(*args)
        offset = torch.full_like(state, 1e-4)
        result["base_value"] += offset
        for key in ("even_response", "plus_remainder", "minus_remainder"):
            result[key] -= offset
        return result

    monkeypatch.setattr(runner, "finite_amplitude_response", biased_primal)
    step = lambda n, x: 1.25 * x + 0.1
    native_amplitudes = tuple(runner.rms(direction * size) / 2.0 for size in (0.01, 0.05))
    probes = [runner.transition_probe(step, 4, state, direction * size, 2.0, native_amplitudes)
              for size in (0.01, 0.05)]
    for probe, rows, native in probes:
        assert [row["amplitude_train_scale_rms"] for row in rows[:7]] == list(runner.TRANSITION_AMPLITUDES)
        assert native in [row["amplitude_train_scale_rms"] for row in rows]
        assert [row["amplitude_train_scale_rms"] for row in rows[-2:]] == list(native_amplitudes)
        assert probe["even_response"].dtype == torch.float64
        for i, amplitude in enumerate(runner.TRANSITION_AMPLITUDES):
            assert runner.rms(probe["plus_displacement"][i]) / 2.0 == pytest.approx(amplitude, rel=2e-6)
            assert rows[i]["plus_gain"] == pytest.approx(1.25, abs=3e-5)
        assert float(probe["even_response"].abs().max()) < torch.finfo(torch.float32).eps
    torch.testing.assert_close(probes[0][0]["plus_displacement"], probes[1][0]["plus_displacement"])


def test_parent_hashes_checked_before_array_decoding(packet, tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    result = runner.run(packet, parent)
    assert result["status"] == "completed"
    manifest, _, _, reference, manifest_hash = runner.load_inputs(packet)
    result_hash = runner.sha256(parent / "result.json")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("parent arrays decoded before hashes"))
    with pytest.raises(ValueError, match="parent result hash"):
        runner.load_parent(parent, "0" * 64, manifest, manifest_hash, reference)
    with (parent / "forecast.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="parent artifact hash"):
        runner.load_parent(parent, result_hash, manifest, manifest_hash, reference)


def test_transition_phase_preserves_parent_and_records_decomposition(packet, tmp_path):
    with np.load(packet / "reference.npz") as saved:
        first = saved["states"][0].copy()
    np.savez(packet / "reference.npz", states=np.repeat(first[None], 33, axis=0))
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["horizon"] = 32
    manifest["artifacts"]["reference.npz"] = runner.sha256(packet / "reference.npz")
    runner.write_json(packet / "manifest.json", manifest)
    parent = tmp_path / "parent"
    result = runner.run(packet, parent)
    assert result["status"] == "completed", result.get("error")
    before = {path.name: runner.sha256(path) for path in parent.iterdir()}
    result = runner.run_response_transition(packet, tmp_path / "transition", parent, before["result.json"])
    assert result["status"] == "completed", result.get("error")
    assert result["parent_unchanged"]
    assert result["role"] == "posthoc_train_only"
    assert len(result["responses"]) == 6
    for a, b in zip(result["responses"][::2], result["responses"][1::2]):
        assert [r["amplitude_train_scale_rms"] for r in a["rows"]] == [r["amplitude_train_scale_rms"] for r in b["rows"]]
    assert len(result["decompositions"]) == 3
    assert {path.name: runner.sha256(path) for path in parent.iterdir()} == before
    for row in result["decompositions"]:
        assert sum(row["projection_fraction_onto_next_discrepancy"].values()) == pytest.approx(1, abs=1e-10)
    with pytest.raises(ValueError, match="separate"):
        runner.run_response_transition(packet, parent / "nested", parent, before["result.json"])
