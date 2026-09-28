from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_unroll as runner
from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluate
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet as packet
from tests.time_dependent_no.test_evaluate_kolmogorov_recovery import evaluation as evaluation


@pytest.fixture
def configured(packet, monkeypatch):
    monkeypatch.setattr(runner, "DEPTH", 2)
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "FIT_UPDATES", 3)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    return packet


def test_sequence_windows_include_last_successor_without_crossing_paths():
    train = np.arange(2 * 7, dtype=np.float32).reshape(2, 7, 1, 1)
    batch = runner.sequence_batch(train, np.array([0, 2, 3, 5]))
    np.testing.assert_array_equal(batch[:, :, 0, 0],
                                 [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6],
                                  [7, 8, 9, 10, 11], [9, 10, 11, 12, 13]])
    for indices in ([-1], [6], [0.5], []):
        with pytest.raises(ValueError, match="indices"):
            runner.sequence_batch(train, np.asarray(indices))


def test_identical_forward_loss_but_analytically_distinct_temporal_gradients():
    class ScalarMap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
            self.register_buffer("train_scale", torch.tensor(2., dtype=torch.float64))
            self.inputs = []

        def forward(self, x):
            self.inputs.append((x.detach().clone(), x.requires_grad))
            return {"next_state": self.weight * x}

    sequence = torch.tensor([1., .9, .8, .7, .6], dtype=torch.float64).reshape(1, 5, 1, 1)
    rows, models = {}, {}
    for arm in runner.ARMS:
        model = models[arm] = ScalarMap()
        rows[arm] = runner.update(model, torch.optim.SGD(model.parameters(), lr=0),
                                  sequence, arm, "cpu")
        n = np.arange(1, 5)
        predicted = .7 ** n
        derivative = (.7 ** (n - 1)) * (n if arm == "full" else 1)
        expected = 2 / (4 * 2**2) * np.sum((predicted - sequence.numpy().ravel()[1:]) * derivative)
        assert float(model.weight.grad) == pytest.approx(expected, abs=1e-14)
        assert [flag for _, flag in model.inputs] == [False] + [arm == "full"] * 3
    assert rows["detached"]["step_mse_scaled"] == rows["full"]["step_mse_scaled"]
    for (left, _), (right, _) in zip(models["detached"].inputs, models["full"].inputs):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert rows["detached"]["gradient_l2"] != rows["full"]["gradient_l2"]


def test_matched_fits_restart_parent_and_preserve_sampler_and_objective(configured, tmp_path, monkeypatch):
    original = runner.update
    starts = []

    def observe(model, optimizer, *args):
        if not optimizer.state:
            starts.append({key: value.clone() for key, value in model.state_dict().items()})
        return original(model, optimizer, *args)

    monkeypatch.setattr(runner, "update", observe)
    resource = runner.run(configured, tmp_path / "resource", "full", "resource", "cpu")
    results = [runner.run(configured, tmp_path / arm, arm, "fit", "cpu") for arm in runner.ARMS]
    assert resource["updates_completed"] == 2
    assert not (tmp_path / "resource" / "terminal.pt").exists()
    assert len(starts) == 3
    assert all(torch.equal(starts[0][k], s[k]) for s in starts[1:] for k in starts[0])
    logs = [[json.loads(line) for line in (tmp_path / arm / "updates.jsonl").read_text().splitlines()]
            for arm in runner.ARMS]
    assert [[r["sequence_indices"] for r in log] for log in logs][0] == [r["sequence_indices"] for r in logs[1]]
    assert logs[0][0]["step_mse_scaled"] == logs[1][0]["step_mse_scaled"]
    assert all(r["updates_completed"] == 3 and r["identity"]["depth"] == 2 for r in results)
    assert all(r["identity"]["loss_weights"] == [.5, .5] for r in results)


@pytest.mark.parametrize("arm", runner.ARMS)
def test_existing_assay_loads_only_exact_completed_unroll_recipe(configured, evaluation, tmp_path, arm):
    fit = tmp_path / "fit"
    result = runner.run(configured, fit, arm, "fit", "cpu")
    assay = evaluate.run(configured, evaluation, tmp_path / "assay", "assay", "cpu", fit)
    assert assay["status"] == "completed" and assay["recurrent_model_calls"] == 0
    assert assay["model"]["identity"] == result["identity"]
    original = json.loads((fit / "result.json").read_text())
    for field, value in (("depth", 99), ("temporal_gradients", arm != "full"),
                         ("loss_weights", [1., 0.]), ("sampler_seed", 900)):
        modified = json.loads(json.dumps(original))
        modified["identity"][field] = value
        runner.base.write_json(fit / "result.json", modified)
        with pytest.raises(ValueError, match="fit identity"):
            evaluate.run(configured, evaluation, tmp_path / field, "assay", "cpu", fit)


def test_failure_records_completed_update_and_leaves_input_immutable(configured, tmp_path, monkeypatch):
    before = {p.name: runner.base.sha256(p) for p in configured.iterdir()}
    original, calls = runner.update, 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failed update")
        return original(*args)

    monkeypatch.setattr(runner, "update", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.run(configured, output, "full", "fit", "cpu")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed" and result["updates_completed"] == 1
    assert not (output / "terminal.pt").exists()
    assert before == {p.name: runner.base.sha256(p) for p in configured.iterdir()}
    with pytest.raises(FileExistsError):
        runner.run(configured, output, "full", "fit", "cpu")


def test_invalid_arm_rejected_before_training_packet_access(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.base, "load_inputs", lambda _: pytest.fail("opened input"))
    with pytest.raises(ValueError, match="arm"):
        runner.run(tmp_path / "input", tmp_path / "out", "unknown", "fit", "cpu")
