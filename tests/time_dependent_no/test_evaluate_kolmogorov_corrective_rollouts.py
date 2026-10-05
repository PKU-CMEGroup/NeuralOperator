from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_corrective_rollouts as runner
from tests.time_dependent_no.test_evaluate_kolmogorov_local_defects import AffineMap, truth_path
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet as packet
from tests.time_dependent_no.test_run_kolmogorov_displacement_comparison import prepared as prepared


class HalfFlow:
    def canonicalize(self, x):
        return np.asarray(x, dtype=np.float64)

    def advance_canonical(self, x):
        return SimpleNamespace(state=.5 * x)


def test_own_and_probe_defects_use_same_input_and_true_next_state(monkeypatch):
    monkeypatch.setattr(runner.diagnostic, "physical_diagnostics", lambda x, _: (np.zeros((len(x), 4)), ("a", "b", "c", "d")))
    model, probe = AffineMap(.5, 1), AffineMap(.5, .25)
    truth = truth_path()
    score, fields, checks, _ = runner.evaluate_case(model, probe, truth, 2, "cpu", HalfFlow(), HalfFlow())
    np.testing.assert_allclose(score["own_defect_mse"], .25)
    np.testing.assert_allclose(score["probe_defect_mse"], .25**2 / 4)
    for n, seen in enumerate(probe.inputs):
        np.testing.assert_array_equal(seen[0], fields["reached"][n])
    np.testing.assert_array_equal(fields["solver_next"], fields["reached"][:-1] * .5)
    np.testing.assert_allclose(score["rollout_mse"][1:], score["own_defect_mse"] + score["propagated_mse"] + score["cross_term"])
    assert checks["error_identity_max_residual"] < 1e-12
    # Probe predictions never enter the model trajectory.
    other = runner.evaluate_case(AffineMap(.5, 1), AffineMap(8, -17), truth, 2, "cpu", HalfFlow(), HalfFlow())[1]
    np.testing.assert_array_equal(other["reached"], fields["reached"])


def test_corrective_terminal_loads_only_matching_completed_recipe(prepared, tmp_path):
    _, _, manifest, captured, original, train, train_hash, _, arrays, data_hash = prepared
    out = tmp_path / "native_one_recovery"
    runner.comparison.train_condition(train, captured, original, train_hash, data_hash, arrays,
                                     "native_one", "recovery", out, "cpu")
    model, identity = runner.load_terminal(out, original, manifest, train_hash, "cpu", prepared_hash=data_hash)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert identity["checkpoint_sha256"] == runner.training.sha256(out / "terminal.pt")
    with pytest.raises(ValueError, match="protocol"):
        runner.load_terminal(out, original, manifest, train_hash, "cpu", prepared_hash="wrong")
    result = json.loads((out / "result.json").read_text())
    result["status"] = "running"
    runner.comparison.write_json(out / "result.json", result)
    with pytest.raises(ValueError, match="update mismatch"):
        runner.load_terminal(out, original, manifest, train_hash, "cpu", prepared_hash=data_hash)


def test_probe_rejects_reused_weights_identity(packet, tmp_path):
    manifest, captured, original, _, train_hash = runner.training.load_inputs(packet)
    out = tmp_path / "probe"
    out.mkdir()
    identity = dict(input_manifest_sha256=train_hash, model_config=original["model_config"],
                    run_id="CM_NEXT_KF_CLEAN_PROBE_20261001A", original_weights_loaded=True,
                    recipe=dict(initialization_seed=29), train_array_sha256=manifest["artifacts"]["train.npy"])
    runner.comparison.write_json(out / "result.json", dict(status="completed", updates_completed=49152, identity=identity))
    with pytest.raises(ValueError, match="independently initialized"):
        runner.load_terminal(out, original, manifest, train_hash, "cpu", probe=True)


def test_watch_never_uses_intermediate_weights(tmp_path):
    root = tmp_path / "run"
    (root / "native_one_recovery").mkdir(parents=True)
    runner.comparison.write_json(root / "queue.json", dict(status="training"))
    runner.comparison.write_json(root / "native_one_recovery/result.json", dict(status="running", updates_completed=1024))
    (root / "native_one_recovery/latest.pt").write_bytes(b"not a terminal")
    with pytest.raises(RuntimeError, match="not ready"):
        runner.await_terminal(root, "native_one_recovery", False)
    runner.comparison.write_json(root / "queue.json", dict(status="failed"))
    with pytest.raises(RuntimeError, match="queue failed"):
        runner.await_terminal(root, "native_one_recovery", True)


def test_plot_table_keeps_population_and_time_aggregation_explicit(tmp_path, monkeypatch):
    from scripts.time_dependent_no import plot_kolmogorov_corrective_rollouts as plotter
    root = tmp_path / "evaluation"
    target = root / "pretrained"
    target.mkdir(parents=True)
    h = 32
    metrics = {k: np.ones((2, h)) for k in ("clean_mse", "own_defect_mse", "probe_defect_mse", "propagated_mse", "cross_term")}
    metrics["rollout_mse"] = np.stack([np.r_[0., np.ones(h)], np.r_[0., np.full(h, 4.)]])
    metrics["physical_values"] = np.ones((2, h+1, 4))
    metrics["physical_values"][1, :, 2] = 4
    np.savez(target / "metrics.npz", **metrics)
    runner.comparison.write_json(target / "result.json", dict(status="completed", artifacts={"metrics.npz": plotter.digest(target / "metrics.npz")}))
    np.savez(root / "ground_truth.npz", probe_clean_mse=np.ones((2, h)) * .0001, physical_truth=np.ones((2, h+1, 4)))
    runner.comparison.write_json(root / "result.json", dict(completed_methods=["pretrained"], horizon=h,
                                macro_dt=.05, populations=["train", "validation"], ground_truth_sha256=plotter.digest(root / "ground_truth.npz")))
    original_save = plotter.save

    def check_limits(fig, output, stem):
        if stem in ("rollout_errors", "enstrophy"):
            for axis in fig.axes:
                lower, upper = axis.get_ylim()
                for line in axis.lines:
                    values = np.asarray(line.get_ydata())
                    assert np.all((values >= lower) & (values <= upper))
        original_save(fig, output, stem)

    monkeypatch.setattr(plotter, "save", check_limits)
    plotter.plot(root, root / "figures")
    summary = json.loads((root / "figures/summary.json").read_text())
    assert summary["rows"][0]["rollout_rms_H32"] == 1
    assert summary["rows"][1]["rollout_rms_H32"] == 2
    assert summary["probe_clean_rms"] == [.01, .01]
    assert (root / "figures/probe_errors.pdf").stat().st_size > 1000
    assert "pretrained" in (root / "figures/index.html").read_text()
