from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as runner
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO, canonicalize_vorticity


@pytest.fixture
def packet(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    shape = (2, 5, 16, 16)
    monkeypatch.setattr(runner, "TRAIN_SHAPE", shape)
    monkeypatch.setattr(runner, "TRAIN_SEEDS", [2026090611, 2026090612])
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "FIT_UPDATES", 3)
    monkeypatch.setattr(runner, "RESOURCE_UPDATES", 2)
    directory = tmp_path / "input"
    directory.mkdir()
    config = dict(resolution=16, width=2, depth=1, modes=1, fc_dim=0)
    model = PeriodicVorticityPCNO(**config, train_scale=2.0).eval()
    identity = dict(model_config=config, train_scale_float64=2.0, train_scale_model_float32=2.0)
    torch.save(dict(identity=identity, update=49152, schedule_position=49152,
                    model=model.state_dict(), optimizer={"must_not_be_restored": True}),
               directory / "checkpoint.pt")
    runner.write_json(directory / "fit_result.json", dict(
        model_config=config, checkpoint_identity=identity, updates_completed=49152,
        terminal_checkpoint=dict(file="terminal_049152.pt", update=49152,
                                 sha256=runner.sha256(directory / "checkpoint.pt"))))
    states = canonicalize_vorticity(torch.randn(10, 16, 16)).reshape(shape).numpy()
    np.save(directory / "train.npy", states)
    indices = np.array([0, 1, 4, 7])
    paths, steps = np.divmod(indices, 4)
    sentinels = torch.from_numpy(states[paths, steps])
    with torch.no_grad():
        predicted = model(sentinels)
    np.savez(directory / "teacher.npz", sentinel_global_indices=indices,
             sentinel_input=sentinels.numpy(), sentinel_target=states[paths, steps + 1],
             sentinel_raw=predicted["raw_next"].numpy(),
             sentinel_next=predicted["next_state"].numpy())
    runner.write_json(directory / "manifest.json", dict(
        schema_version=1, role="fixed_data_train", train_seeds=runner.TRAIN_SEEDS,
        shape=list(shape), dtype="float32",
        artifacts={name: runner.sha256(directory / name) for name in runner.INPUT_FILES},
        sources={name: runner.sha256(runner.ROOT / name) for name in runner.MODEL_SOURCES}))
    return directory


def test_gaussian_ensemble_scale_subspace_and_independent_generator():
    torch.set_num_threads(2)
    global_before = torch.get_rng_state().clone()
    noise = runner.gaussian_noise((2048, 16, 16), 2.5, torch.Generator().manual_seed(1702))
    assert torch.equal(torch.get_rng_state(), global_before)
    torch.testing.assert_close(noise, canonicalize_vorticity(noise), atol=3e-8, rtol=1e-6)
    rms = noise.double().square().mean((-2, -1)).sqrt() / 2.5
    assert abs(float(rms.square().mean()) / runner.SIGMA**2 - 1) < 0.02
    assert float(rms.std()) > 0.0003  # Per-field normalization would erase this variance.
    torch.manual_seed(900)
    repeated = runner.gaussian_noise(noise.shape, 2.5, torch.Generator().manual_seed(1702))
    torch.testing.assert_close(repeated, noise, rtol=0, atol=0)


def test_matched_sample_tapes_exact_targets_and_control(packet):
    _, _, _, train, _ = runner.load_inputs(packet)
    samplers = [[runner.EpochSampler(8, 2, seed) for seed in (17, 1701)] for _ in range(2)]
    noise_generators = [torch.Generator().manual_seed(1702) for _ in range(2)]
    visited = [[], []]
    for _ in range(4):
        arm_indices = [[sampler.next_indices() for sampler in pair] for pair in samplers]
        for branch in range(2):
            np.testing.assert_array_equal(arm_indices[0][branch], arm_indices[1][branch])
            visited[branch].extend(arm_indices[0][branch].tolist())
        clean, clean_rms = runner.training_batches(train, *arm_indices[0], "clean_continuation", 2, noise_generators[0])
        recovery, recovery_rms = runner.training_batches(train, *arm_indices[1], "recovery", 2, noise_generators[1])
        assert clean_rms == 0 and recovery_rms > 0
        for branch, indices in enumerate(arm_indices[0]):
            paths, steps = np.divmod(indices, 4)
            np.testing.assert_array_equal(clean[branch][0].numpy(), train[paths, steps])
            np.testing.assert_array_equal(clean[branch][1].numpy(), train[paths, steps + 1])
            assert torch.equal(clean[branch][1], recovery[branch][1])
        assert torch.equal(clean[0][0], recovery[0][0])
        assert not torch.equal(clean[1][0], recovery[1][0])
    assert all(sorted(indices) == list(range(8)) for indices in visited)


def test_two_branch_update_matches_weighted_objective():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7))
            self.register_buffer("train_scale", torch.tensor(2.0))

        def forward(self, x):
            return {"next_state": self.weight * x}

    model, expected = Toy(), Toy()
    batches = [(torch.tensor([1., 2.]), torch.tensor([0.5, 1.])),
               (torch.tensor([3., 4.]), torch.tensor([-1., 2.]))]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    independent_optimizer = torch.optim.Adam(expected.parameters(), lr=1e-4)
    objective = sum(0.5 * ((expected(x)["next_state"] - y) / 2).square().mean() for x, y in batches)
    objective.backward()
    independent_optimizer.step()
    row = runner.update(model, optimizer, batches, "cpu")
    torch.testing.assert_close(model.weight, expected.weight, rtol=0, atol=0)
    assert row["loss"] == pytest.approx(float(objective.detach()))


def test_scope_and_hash_checks_precede_array_decode(packet, monkeypatch):
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["role"] = "development"
    runner.write_json(manifest_path, manifest)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("opened arrays before scope/hash check"))
    with pytest.raises(ValueError, match="training population"):
        runner.load_inputs(packet)
    manifest["role"] = "fixed_data_train"
    runner.write_json(manifest_path, manifest)
    with (packet / "train.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runner.load_inputs(packet)


@pytest.mark.parametrize("field,message", [("sentinel_input", "archived training states"),
                                          ("sentinel_target", "archived training successors")])
def test_teacher_must_replay_actual_training_pairs(packet, field, message):
    teacher_path = packet / "teacher.npz"
    with np.load(teacher_path) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays[field][0, 0, 0] += 1
    np.savez(teacher_path, **arrays)
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["artifacts"]["teacher.npz"] = runner.sha256(teacher_path)
    runner.write_json(packet / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        runner.load_inputs(packet)


def test_resource_and_fit_restart_same_parent_without_optimizer_state(packet, tmp_path, monkeypatch):
    original_update = runner.update
    starts, optimizer_sizes = [], []

    def observe(model, optimizer, batches, device):
        if not optimizer.state:
            starts.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            optimizer_sizes.append(len(optimizer.state))
        return original_update(model, optimizer, batches, device)

    monkeypatch.setattr(runner, "update", observe)
    resource = runner.run(packet, tmp_path / "resource", "recovery", "resource", "cpu")
    fit = runner.run(packet, tmp_path / "fit", "recovery", "fit", "cpu")
    repeat = runner.run(packet, tmp_path / "repeat", "recovery", "fit", "cpu")
    runner.run(packet, tmp_path / "clean", "clean_continuation", "fit", "cpu")
    assert resource["updates_completed"] == 2 and fit["updates_completed"] == 3
    assert not (tmp_path / "resource" / "terminal.pt").exists()
    assert optimizer_sizes == [0, 0, 0, 0]
    assert len(starts) == 4
    for key in starts[0]:
        assert all(torch.equal(starts[0][key], start[key]) for start in starts[1:])
    one = torch.load(tmp_path / "fit" / "terminal.pt", weights_only=False)
    two = torch.load(tmp_path / "repeat" / "terminal.pt", weights_only=False)
    assert one["identity"] == two["identity"] == fit["identity"] == repeat["identity"]
    assert all(torch.equal(value, two["model"][key]) for key, value in one["model"].items())
    assert any(not torch.equal(value, starts[0][key]) for key, value in one["model"].items())
    assert all(error <= 1e-6 for errors in fit["parent_replay"]["errors"].values() for error in errors)
    assert fit["artifacts"]["terminal.pt"] == runner.sha256(tmp_path / "fit" / "terminal.pt")


def test_failure_reports_last_completed_update_and_never_writes_input(packet, tmp_path, monkeypatch):
    original_update = runner.update
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second-update failure")
        return original_update(*args)

    monkeypatch.setattr(runner, "update", fail_second)
    with pytest.raises(ValueError, match="outside the immutable input"):
        runner.run(packet, packet / "nested", "recovery", "fit", "cpu")
    assert not (packet / "nested").exists()
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="second-update failure"):
        runner.run(packet, output, "recovery", "fit", "cpu")
    saved = json.loads((output / "result.json").read_text())
    assert saved["status"] == "failed" and saved["updates_completed"] == 1
    assert "second-update failure" in saved["error"]
    assert not (output / "terminal.pt").exists()
