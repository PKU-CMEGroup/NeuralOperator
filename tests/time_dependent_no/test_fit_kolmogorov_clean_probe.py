from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_clean_probe as runner
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO, canonicalize_vorticity


@pytest.fixture
def packet(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    shape = (2, 5, 16, 16)
    monkeypatch.setattr(runner.training, "TRAIN_SHAPE", shape)
    monkeypatch.setattr(runner.training, "TRAIN_SEEDS", [11, 12])
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "FIT_UPDATES", 5)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    monkeypatch.setattr(runner, "CHECKPOINT_INTERVAL", 2)
    directory = tmp_path / "input"
    directory.mkdir()
    config = dict(resolution=16, width=2, depth=1, modes=1, fc_dim=0)
    model = PeriodicVorticityPCNO(**config, train_scale=2.0).eval()
    identity = dict(model_config=config, train_scale_float64=2.0, train_scale_model_float32=2.0)
    torch.save(dict(identity=identity, model=model.state_dict(),
                    optimizer={"must_not_be_restored": True}), directory / "checkpoint.pt")
    runner.training.write_json(directory / "fit_result.json", dict(
        model_config=config, checkpoint_identity=identity, updates_completed=49152,
        terminal_checkpoint=dict(file="terminal_049152.pt", update=49152,
                                 sha256=runner.training.sha256(directory / "checkpoint.pt"))))
    states = canonicalize_vorticity(torch.randn(10, 16, 16)).reshape(shape).numpy()
    np.save(directory / "train.npy", states)
    indices = np.array([0, 1, 4, 7])
    paths, steps = np.divmod(indices, 4)
    with torch.no_grad():
        prediction = model(torch.from_numpy(states[paths, steps]))
    np.savez(directory / "teacher.npz", sentinel_global_indices=indices,
             sentinel_input=states[paths, steps], sentinel_target=states[paths, steps + 1],
             sentinel_raw=prediction["raw_next"].numpy(), sentinel_next=prediction["next_state"].numpy())
    runner.training.write_json(directory / "manifest.json", dict(
        schema_version=1, role="fixed_data_train", train_seeds=[11, 12],
        shape=list(shape), dtype="float32",
        artifacts={name: runner.training.sha256(directory / name) for name in runner.training.INPUT_FILES},
        sources={name: runner.training.sha256(runner.training.ROOT / name)
                 for name in runner.training.MODEL_SOURCES}))
    return directory


def test_batch_samples_exact_clean_next_states_without_replacement(packet):
    _, _, _, train, _ = runner.training.load_inputs(packet)
    sampler = runner.clean.EpochSampler(8, 2, runner.SEED)
    visited = []
    for _ in range(4):
        selected = sampler.next_indices()
        inputs, targets = runner.training_batch(train, selected)
        paths, steps = np.divmod(selected, 4)
        np.testing.assert_array_equal(inputs.numpy(), train[paths, steps])
        np.testing.assert_array_equal(targets.numpy(), train[paths, steps + 1])
        visited.extend(selected)
    assert sorted(visited) == list(range(8))


def test_fresh_initialization_resource_restart_and_fixed_recipe(packet, tmp_path, monkeypatch):
    original_update = runner.update
    initial_hashes, optimizer_sizes = [], []

    def observe(model, optimizer, batch, device, position):
        if position == 1:
            initial_hashes.append(runner.state_fingerprint(model))
            optimizer_sizes.append(len(optimizer.state))
            assert torch.count_nonzero(model.pcno.fc2.weight) == 0
            assert torch.count_nonzero(model.pcno.fc2.bias) == 0
        return original_update(model, optimizer, batch, device, position)

    monkeypatch.setattr(runner, "update", observe)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("fresh probe must not load parent weights"))
    resource = runner.run(packet, tmp_path / "resource", "resource", "cpu")
    fitted = runner.run(packet, tmp_path / "fit", "fit", "cpu")
    assert initial_hashes == [fitted["identity"]["initialization_state_sha256"]] * 2
    assert optimizer_sizes == [0, 0]
    assert not list((tmp_path / "resource").glob("*.pt"))
    assert resource["updates_completed"] == 2 and fitted["updates_completed"] == 5
    assert not fitted["original_weights_loaded"]
    assert not fitted["development_access"] and not fitted["rollout_access"]
    assert fitted["identity"]["recipe"]["initialization_seed"] == 29
    assert fitted["identity"]["train_scale_model_float32"] == 2.0
    assert fitted["artifacts"]["terminal.pt"] == runner.training.sha256(tmp_path / "fit" / "terminal.pt")
    rows = [json.loads(line) for line in (tmp_path / "fit" / "updates.jsonl").read_text().splitlines()]
    assert [row["learning_rate"] for row in rows] == [runner.clean.clean_lr(i) for i in range(1, 6)]
    config = fitted["identity"]["model_config"]
    monkeypatch.setattr(runner, "SEED", 17)
    assert runner.state_fingerprint(runner.make_model(config, 2.0, "cpu")) != initial_hashes[0]


def test_original_schedule_and_single_branch_objective():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7))
            self.register_buffer("train_scale", torch.tensor(2.0))

        def forward(self, x):
            return {"next_state": self.weight * x}

    model, expected = Toy(), Toy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.7)
    batch = torch.tensor([1., 3.]), torch.tensor([0.5, -1.])
    lr = runner.clean.clean_lr(32769)
    independent_optimizer = torch.optim.Adam(expected.parameters(), lr=lr)
    objective = ((expected(batch[0])["next_state"] - batch[1]) / 2).square().mean()
    objective.backward()
    independent_optimizer.step()
    row = runner.update(model, optimizer, batch, "cpu", 32769)
    torch.testing.assert_close(model.weight, expected.weight, rtol=0, atol=0)
    assert row["loss_normalized_mse"] == pytest.approx(float(objective.detach()))
    assert row["learning_rate"] == 1e-4
    assert runner.clean.clean_lr(512) == 1e-3
    assert runner.clean.clean_lr(32768) == runner.clean.clean_lr(49152) == 1e-4


def test_atomic_checkpoint_resume_reproduces_uninterrupted_fit(packet, tmp_path, monkeypatch):
    original_update = runner.update
    fresh = runner.run(packet, tmp_path / "fresh", "fit", "cpu")

    def fail_fourth(model, optimizer, batch, device, position):
        if position == 4:
            raise RuntimeError("synthetic interruption after unsaved update")
        return original_update(model, optimizer, batch, device, position)

    monkeypatch.setattr(runner, "update", fail_fourth)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(packet, tmp_path / "resumed", "fit", "cpu")
    latest = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)
    assert latest["update"] == latest["schedule_position"] == 2
    assert all(name in latest for name in ("model", "optimizer", "sampler", "rng"))
    monkeypatch.setattr(runner, "update", original_update)
    resumed = runner.run(packet, tmp_path / "resumed", "fit", "cpu", resume=True)
    assert resumed["resumed_from_update"] == 2
    assert resumed["identity"] == fresh["identity"]
    one = torch.load(tmp_path / "fresh" / "terminal.pt", weights_only=False)
    two = torch.load(tmp_path / "resumed" / "terminal.pt", weights_only=False)
    assert all(torch.equal(value, two["model"][key]) for key, value in one["model"].items())
    assert torch.equal(one["sampler"]["generator"], two["sampler"]["generator"])
    assert torch.equal(one["rng"]["torch_cpu"], two["rng"]["torch_cpu"])
    rows = [json.loads(line) for line in (tmp_path / "resumed" / "updates.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == list(range(1, 6))
    restored = runner.make_model(two["identity"]["model_config"], 2.0, "cpu")
    restored.load_state_dict(two["model"])
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in two["model"].items())


def test_scope_and_output_preservation(packet, tmp_path):
    with pytest.raises(ValueError, match="immutable input"):
        runner.run(packet, packet / "nested", "fit", "cpu")
    assert not (packet / "nested").exists()
    output = tmp_path / "fit"
    runner.run(packet, output, "fit", "cpu")
    before = (output / "result.json").read_bytes()
    with pytest.raises(ValueError, match="completed fit"):
        runner.run(packet, output, "fit", "cpu", resume=True)
    assert (output / "result.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        runner.run(packet, output, "fit", "cpu")
