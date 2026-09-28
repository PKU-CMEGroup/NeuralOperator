from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_targets as runner
from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation_runner
from tests.time_dependent_no.test_generate_kolmogorov_target_bank import (
    packet as packet, evaluation as evaluation, target_packets as target_packets,
)


@pytest.fixture
def fitted_inputs(target_packets, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FIT_UPDATES", 3)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    train, dev = target_packets
    bank = tmp_path / "bank"
    result = runner.bank.run(train, bank, "train", "cpu")
    return train, dev, bank, result


def test_both_arms_see_identical_inputs_and_only_displaced_targets_differ(fitted_inputs):
    train_path, _, bank_path, result = fitted_inputs
    _, arrays, _ = runner.bank.load_bank(bank_path, "train", result["training_manifest_sha256"],
                                        result["parent_checkpoint_sha256"])
    train = np.load(train_path / "train.npy")
    anchors = np.array([r["anchor"] for r in result["rows"]])
    first, second = np.array([0, 7]), np.array([1, len(anchors)-2])
    rec = runner.training_batches(train, arrays, anchors, first, second, "recovery")
    dyn = runner.training_batches(train, arrays, anchors, first, second, "dynamics")
    for branch in range(2):
        assert torch.equal(rec[branch][0], dyn[branch][0])
    assert torch.equal(rec[0][1], dyn[0][1])
    assert not torch.equal(rec[1][1], dyn[1][1])
    np.testing.assert_array_equal(rec[1][1].numpy(), arrays["clean_targets"][anchors[second]])
    np.testing.assert_array_equal(dyn[1][1].numpy(), arrays["dynamics_targets"][second].astype(np.float32))


def test_parent_fresh_optimizer_sample_tapes_and_evaluator_binding(fitted_inputs, tmp_path, monkeypatch):
    train, dev, bank, _ = fitted_inputs
    starts, original = [], runner.base.update

    def observe(model, optimizer, batches, device):
        if not optimizer.state:
            starts.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return original(model, optimizer, batches, device)

    monkeypatch.setattr(runner.base, "update", observe)
    resource = runner.run(train, bank, tmp_path / "resource", "recovery", "resource", "cpu")
    rec = runner.run(train, bank, tmp_path / "rec", "recovery", "fit", "cpu")
    dyn = runner.run(train, bank, tmp_path / "dyn", "dynamics", "fit", "cpu")
    assert resource["updates_completed"] == 2 and not (tmp_path / "resource" / "terminal.pt").exists()
    assert len(starts) == 3
    assert all(torch.equal(starts[0][k], start[k]) for start in starts[1:] for k in starts[0])
    tapes = [[json.loads(s) for s in (tmp_path / arm / "updates.jsonl").read_text().splitlines()]
             for arm in ("rec", "dyn")]
    assert [[(r["first_indices"], r["second_indices"]) for r in tape] for tape in tapes][0] == [
        (r["first_indices"], r["second_indices"]) for r in tapes[1]]
    assert rec["identity"]["bank_manifest_sha256"] == dyn["identity"]["bank_manifest_sha256"]
    manifest, captured, parent, _, digest = runner.base.load_inputs(train)
    model, identity, hashes = evaluation_runner.evaluation_model(
        captured, parent, manifest, digest, tmp_path / "dyn", "cpu")
    assert not model.training and identity["identity"]["arm"] == "dynamics"
    assert hashes["bank_result.json"] == rec["identity"]["bank_manifest_sha256"]
    with (tmp_path / "dyn" / "bank_result.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="bank receipt hash"):
        evaluation_runner.evaluation_model(captured, parent, manifest, digest, tmp_path / "dyn", "cpu")


def test_target_assay_independent_defect_and_no_composition(fitted_inputs, tmp_path, monkeypatch):
    train, dev, bank, _ = fitted_inputs
    fitted = tmp_path / "fit"
    runner.run(train, bank, fitted, "dynamics", "fit", "cpu")
    monkeypatch.setattr(evaluation_runner, "rollout_case", lambda *a: pytest.fail("target assay composed"))
    output = tmp_path / "assay"
    result = evaluation_runner.run(train, dev, output, "target_assay", "cpu", fitted,
                                   target_bank=bank, bank_role="train")
    assert result["status"] == "completed" and result["recurrent_model_calls"] == 0
    with np.load(output / "target_predictions.npz") as saved:
        prediction = saved["displaced_prediction"].astype(np.float64)
        centers = saved["center_prediction"].astype(np.float64)
    actual = np.load(bank / "dynamics_targets.npy")
    solver_clean = np.load(bank / "solver_clean.npy")
    for i, row in enumerate(result["target_responses"]):
        assert row["dynamics_defect_rms_scaled"] == pytest.approx(runner.bank.rms(prediction[i]-actual[i]) / 2.)
        a = row["anchor"]
        assert row["response_defect_rms_scaled"] == pytest.approx(
            runner.bank.rms(prediction[i] - centers[a] - actual[i] + solver_clean[a]) / 2.)
    with pytest.raises(ValueError, match="requires"):
        evaluation_runner.run("missing", "missing", "unused", "target_assay", "cpu")


def test_partial_fit_failure_has_no_terminal_and_keeps_bank(fitted_inputs, tmp_path, monkeypatch):
    train, _, bank, result = fitted_inputs
    hashes = {k: runner.base.sha256(bank / k) for k in result["artifacts"]}
    with pytest.raises(ValueError, match="outside"):
        runner.run(train, bank, bank / "bad", "dynamics", "fit", "cpu")
    original, calls = runner.base.update, []

    def fail(model, optimizer, batches, device):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected target update failure")
        return original(model, optimizer, batches, device)

    monkeypatch.setattr(runner.base, "update", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(train, bank, output, "dynamics", "fit", "cpu")
    saved = json.loads((output / "result.json").read_text())
    assert saved["status"] == "failed" and saved["updates_completed"] == 1
    assert not (output / "terminal.pt").exists()
    assert hashes == {k: runner.base.sha256(bank / k) for k in hashes}
