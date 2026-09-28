from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_response_preserving as runner
from scripts.time_dependent_no import fit_kolmogorov_targets as target_fit
from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation_runner
from tests.time_dependent_no.test_fit_kolmogorov_targets import (
    packet as packet, evaluation as evaluation, target_packets as target_packets,
    fitted_inputs as fitted_inputs,
)


class Affine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.bias = torch.nn.Parameter(torch.tensor(.5))
        self.register_buffer("train_scale", torch.tensor(1.))

    def forward(self, x):
        return {"next_state": self.weight*x + self.bias}


def test_response_gradient_reaches_both_inputs_and_does_not_penalize_offset():
    model = Affine()
    u, x = torch.tensor([1., 2.]), torch.tensor([2., 4.])
    clean = [(u, model(u)["next_state"].detach())]*2
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    row = runner.update(model, optimizer, clean, (u, x, x-u), "cpu", 1.)
    assert row["response_mse_scaled"] == pytest.approx(2.5)
    assert model.weight.item() == pytest.approx(1.5)
    assert model.bias.item() == pytest.approx(.5)


def test_zero_penalty_matches_existing_clean_update():
    candidate = Affine()
    control = copy.deepcopy(candidate)
    batches = [(torch.tensor([1., 2.]), torch.tensor([0., 1.])),
               (torch.tensor([-1., 3.]), torch.tensor([1., 2.]))]
    left = torch.optim.Adam(candidate.parameters(), lr=1e-4)
    right = torch.optim.Adam(control.parameters(), lr=1e-4)
    runner.update(candidate, left, batches, None, "cpu", 0.)
    runner.base.update(control, right, batches, "cpu")
    for a, b in zip(candidate.parameters(), control.parameters()):
        assert torch.equal(a, b)


@pytest.fixture
def response_inputs(fitted_inputs, tmp_path, monkeypatch):
    train, dev, bank, _ = fitted_inputs
    fitted = tmp_path / "teacher"
    query = tmp_path / "teacher_query"
    target_fit.run(train, bank, fitted, "dynamics", "fit", "cpu")
    evaluation_runner.run(train, dev, query, "target_assay", "cpu", fitted,
                          target_bank=bank, bank_role="train")
    monkeypatch.setattr(runner, "FIT_UPDATES", 3)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    return train, dev, bank, fitted, query


def test_paired_start_clean_tapes_teacher_targets_and_checkpoint_roundtrip(response_inputs, tmp_path):
    train, dev, bank, teacher, query = response_inputs
    original = {str(p): runner.base.sha256(p) for p in (teacher / "terminal.pt", query / "target_predictions.npz")}
    results, logs = [], []
    for arm in runner.ARMS:
        output = tmp_path / arm
        results.append(runner.run(train, bank, teacher, query, output, arm, "fit", "cpu"))
        logs.append([json.loads(s) for s in (output / "updates.jsonl").read_text().splitlines()])
        manifest, captured, parent, _, digest = runner.base.load_inputs(train)
        model, identity, _ = evaluation_runner.evaluation_model(captured, parent, manifest, digest, output, "cpu")
        assert not model.training and identity["identity"]["arm"] == arm
        assert identity["identity"]["initial_checkpoint_sha256"] == original[str(teacher / "terminal.pt")]
    assert [[(r["first_indices"], r["second_indices"], r["response_indices"]) for r in log]
            for log in logs][0] == [(r["first_indices"], r["second_indices"], r["response_indices"])
                                   for r in logs[1]]
    assert all(r["teacher_replay"]["relative_l2"] < 1e-5 for r in results)
    assert all(row["response_mse_scaled"] == 0 for row in logs[0])
    assert logs[1][0]["response_mse_scaled"] < 1e-10
    assert logs[1][-1]["response_mse_scaled"] > 0
    assert original == {p: runner.base.sha256(p) for p in original}
    monkeypatch_fail_rollout = lambda *a: pytest.fail("assay must not compose a new path")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation_runner, "rollout_case", monkeypatch_fail_rollout)
        result = evaluation_runner.run(train, dev, tmp_path / "assay", "target_assay", "cpu",
                                       tmp_path / runner.ARMS[1], target_bank=bank, bank_role="train")
    assert result["recurrent_model_calls"] == 0


def test_resource_does_not_write_terminal_and_wrong_teacher_role_is_rejected(response_inputs, tmp_path):
    train, _, bank, teacher, query = response_inputs
    out = tmp_path / "resource"
    runner.run(train, bank, teacher, query, out, runner.ARMS[1], "resource", "cpu")
    assert not (out / "terminal.pt").exists()
    data = json.loads((query / "result.json").read_text())
    data["bank_role"] = "probe"
    (query / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="training"):
        runner.run(train, bank, teacher, query, tmp_path / "bad", runner.ARMS[1], "fit", "cpu")
    assert not (tmp_path / "bad/terminal.pt").exists()


def test_copied_teacher_receipt_is_required_for_evaluation(response_inputs, tmp_path):
    train, _, bank, teacher, query = response_inputs
    output = tmp_path / "candidate"
    runner.run(train, bank, teacher, query, output, runner.ARMS[1], "fit", "cpu")
    with (output / "teacher_fit_result.json").open("a") as stream:
        stream.write(" ")
    manifest, captured, parent, _, digest = runner.base.load_inputs(train)
    with pytest.raises(ValueError, match="receipt"):
        evaluation_runner.evaluation_model(captured, parent, manifest, digest, output, "cpu")


def test_replication_changes_sample_order_within_matched_pairs_and_reloads(response_inputs, tmp_path):
    train, _, bank, teacher, query = response_inputs
    tapes = {}
    initial = None
    for seed, arm in ((17, "clean_only"), (18, "clean_only"), (18, "preserve_response")):
        output = tmp_path / f"seed{seed}_{arm}"
        result = runner.run(train, bank, teacher, query, output, arm, "fit", "cpu", seed=seed)
        rows = [json.loads(s) for s in (output / "updates.jsonl").read_text().splitlines()]
        tapes[seed, arm] = [(r["first_indices"], r["second_indices"], r["response_indices"]) for r in rows]
        manifest, captured, parent, _, digest = runner.base.load_inputs(train)
        _, loaded, _ = evaluation_runner.evaluation_model(captured, parent, manifest, digest, output, "cpu")
        assert loaded["identity"] == result["identity"]
        if initial is None:
            initial = loaded["identity"]["initial_checkpoint_sha256"]
        assert loaded["identity"]["initial_checkpoint_sha256"] == initial
    assert tapes[18, "clean_only"] == tapes[18, "preserve_response"]
    assert tapes[17, "clean_only"] != tapes[18, "clean_only"]
    result["identity"]["sampler_seeds"][1] += 1
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="identity"):
        evaluation_runner.evaluation_model(captured, parent, manifest, digest, output, "cpu")
