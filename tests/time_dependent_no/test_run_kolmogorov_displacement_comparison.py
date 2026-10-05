from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import run_kolmogorov_displacement_comparison as runner
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet as packet


@pytest.fixture
def prepared(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LENGTHS", (1, 2))
    monkeypatch.setattr(runner, "ROWS_PER_PATH", 2)
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "FIT_UPDATES", 3)
    monkeypatch.setattr(runner, "CHECKPOINT_INTERVAL", 2)
    monkeypatch.setattr(runner, "REFERENCE", replace(runner.REFERENCE, resolution=16, macro_dt=1e-5, dt_max=1e-5))
    solver = runner.KolmogorovReferenceStepper(runner.REFERENCE)
    train = np.load(packet / "train.npy")
    for p in range(len(train)):
        train[p] = solver.rollout_canonical(solver.canonicalize(train[p, 0]), 4)[0]
    np.save(packet / "train.npy", train)
    manifest = json.loads((packet / "manifest.json").read_text())
    original = json.loads((packet / "fit_result.json").read_text())
    captured = {k: (packet / k).read_bytes() for k in runner.base.INPUT_FILES if k != "train.npy"}
    model = runner.base.load_model(captured, original, "cpu")
    with np.load(packet / "teacher.npz") as saved:
        teacher = {k: saved[k].copy() for k in saved.files}
    p, t = np.divmod(teacher["sentinel_global_indices"], 4)
    teacher["sentinel_input"], teacher["sentinel_target"] = train[p, t], train[p, t+1]
    with torch.no_grad():
        predicted = model(torch.from_numpy(train[p, t]))
    teacher["sentinel_raw"], teacher["sentinel_next"] = predicted["raw_next"].numpy(), predicted["next_state"].numpy()
    np.savez(packet / "teacher.npz", **teacher)
    manifest["artifacts"] = {k: runner.base.sha256(packet / k) for k in runner.base.INPUT_FILES}
    runner.write_json(packet / "manifest.json", manifest)
    manifest, captured, original, train, digest = runner.base.load_inputs(packet)
    output = tmp_path / "displacements"
    runner.prepare_inputs(train, model, manifest, digest, output)
    result, arrays, data_hash = runner.load_prepared(output, train, manifest, digest)
    return packet, output, manifest, captured, original, train, digest, result, arrays, data_hash


def test_native_predictions_preserve_magnitudes_and_correct_endpoint():
    class Multiply(torch.nn.Module):
        def forward(self, x):
            return {"next_state": 1.5 * x}

    train = np.stack([np.full((3, 3), j + 1, dtype=np.float32) for j in range(10)])[None]
    rows = [dict(path_index=0, input_step=8, mixed_length=4),
            dict(path_index=0, input_step=5, mixed_length=1)]
    one = runner.native_inputs(Multiply(), train, rows, False, "cpu")
    mixed = runner.native_inputs(Multiply(), train, rows, True, "cpu")
    np.testing.assert_array_equal(one[0], 1.5 * train[0, 7])
    np.testing.assert_array_equal(mixed[0], 1.5**4 * train[0, 4])
    np.testing.assert_array_equal(mixed[1], one[1])
    assert runner.rms(mixed[0] - train[0, 8]) != pytest.approx(runner.rms(one[0] - train[0, 8]))


def test_sampling_balances_lengths_and_covers_full_training_interval():
    rows = runner.sample_rows(32, 512)
    assert len(rows) == 1024 and rows == runner.sample_rows(32, 512)
    for path in range(32):
        selected = [r for r in rows if r["path_index"] == path]
        assert {k: sum(r["mixed_length"] == k for r in selected) for k in runner.LENGTHS} == {k: 8 for k in runner.LENGTHS}
        assert len({r["input_step"] for r in selected}) == 32
        assert selected[0]["input_step"] < 24 and selected[-1]["input_step"] >= 496
        assert all(0 <= r["input_step"] - r["mixed_length"] and r["input_step"] + 1 <= 512 for r in selected)


def test_gaussian_ensemble_scale_without_normalizing_each_sample():
    solver = runner.KolmogorovReferenceStepper(replace(runner.REFERENCE, resolution=16))
    rng = np.random.default_rng(3)
    values = np.stack([runner.gaussian_input(np.zeros((16, 16), dtype=np.float32), .3, rng, solver) for _ in range(512)])
    norms = np.sqrt(np.mean(values.astype(float)**2, axis=(1, 2)))
    assert np.mean(norms**2) == pytest.approx(.3**2, rel=.025)
    assert np.std(norms) > .01
    assert np.abs(values.mean(axis=(1, 2))).max() < 1e-7
    np.testing.assert_allclose(solver.canonicalize(values[0]), values[0], rtol=0, atol=3e-8)


def test_real_solver_labels_and_correct_recovery_targets(prepared):
    _, _, _, captured, original, train, _, result, arrays, _ = prepared
    model = runner.base.load_model(captured, original, "cpu")
    solver = runner.KolmogorovReferenceStepper(runner.REFERENCE)
    for i, row in enumerate(result["rows"]):
        p, t = row["path_index"], row["input_step"]
        np.testing.assert_array_equal(arrays["clean_targets"][i], train[p, t+1])
        for condition in runner.CONDITIONS:
            expected = solver.advance_projected(arrays[condition + "_inputs"][i]).state
            np.testing.assert_array_equal(arrays[condition + "_dynamics"][i], expected)
        native = runner.native_inputs(model, train, [row], True, "cpu")[0]
        np.testing.assert_array_equal(arrays["native_mixed_inputs"][i], native)
    assert result["gaussian_expected_rms_scaled"] == pytest.approx(np.sqrt(np.mean(np.square(result["displacement_rms_scaled"]["native_mixed"]))))
    assert not arrays["centers"].flags.writeable


@pytest.mark.parametrize("condition", runner.CONDITIONS)
def test_target_pair_inputs_and_sample_tapes_are_identical(prepared, tmp_path, condition):
    _, _, _, captured, original, train, digest, _, arrays, data_hash = prepared
    first, second = np.array([0, 5]), np.array([0, 3])
    rec = runner.training_batches(train, arrays, first, second, condition, "recovery")
    dyn = runner.training_batches(train, arrays, first, second, condition, "dynamics")
    assert torch.equal(rec[0][0], dyn[0][0]) and torch.equal(rec[0][1], dyn[0][1])
    assert torch.equal(rec[1][0], dyn[1][0])
    np.testing.assert_array_equal(rec[1][1].numpy(), arrays["clean_targets"][second])
    assert not torch.equal(rec[1][1], dyn[1][1])
    tapes = []
    for target in runner.TARGETS:
        out = tmp_path / target
        result = runner.train_condition(train, captured, original, digest, data_hash, arrays, condition, target, out, "cpu")
        assert result["status"] == "completed" and result["terminal_reload_verified"]
        latest = torch.load(out / "latest.pt", weights_only=False)
        assert latest["update"] == 2 and len(latest["samplers"]) == 2
        rows = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
        tapes.append([(r["clean_indices"], r["displaced_indices"]) for r in rows])
    assert tapes[0] == tapes[1]


def test_corrupted_training_inputs_are_rejected_before_decoding(prepared, monkeypatch):
    _, output, manifest, _, _, train, digest, _, _, _ = prepared
    with (output / "centers.npy").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("decoded unauthenticated arrays"))
    with pytest.raises(ValueError, match="artifact hash"):
        runner.load_prepared(output, train, manifest, digest)


def test_queue_waits_for_verified_probe_and_stops_on_failure(tmp_path):
    probe = tmp_path / "probe"
    probe.mkdir()
    assert runner.probe_state(probe) is False
    runner.write_json(probe / "result.json", {"status": "running"})
    assert runner.probe_state(probe) is False
    runner.write_json(probe / "result.json", {"status": "failed"})
    with pytest.raises(RuntimeError, match="probe failed"):
        runner.probe_state(probe)
    identity = dict(run_id="CM_NEXT_KF_CLEAN_PROBE_20261001A", original_weights_loaded=False)
    torch.save(dict(identity=identity, update=49152), probe / "terminal.pt")
    receipt = dict(status="completed", phase="fit", updates_completed=49152, identity=identity,
                   artifacts={"terminal.pt": runner.base.sha256(probe / "terminal.pt")})
    runner.write_json(probe / "result.json", receipt)
    assert runner.probe_state(probe)
    with (probe / "terminal.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="terminal identity"):
        runner.probe_state(probe)


def test_failed_dependency_does_not_start_data_preparation(tmp_path, monkeypatch):
    probe = tmp_path / "probe"
    probe.mkdir()
    runner.write_json(probe / "result.json", {"status": "failed"})
    monkeypatch.setattr(runner.base, "load_inputs", lambda *a: pytest.fail("started before probe"))
    with pytest.raises(RuntimeError, match="probe failed"):
        runner.run(tmp_path / "input", tmp_path / "queue", probe, "cpu")
    assert json.loads((tmp_path / "queue" / "queue.json").read_text())["status"] == "failed"


def test_numerical_qualification_rejects_unresolved_labels():
    with pytest.raises(ValueError, match="temporal"):
        runner.check_metrics(dict(temporal_rms_scaled=.1))
    with pytest.raises(ValueError, match="projection"):
        runner.check_metrics(dict(projection_rms_scaled=float("nan")))


def test_complete_serial_queue_trains_six_from_same_checkpoint(prepared, tmp_path, monkeypatch):
    packet, _, _, _, _, _, _, _, _, _ = prepared
    initial_states = []
    original_update = runner.base.update

    def observed_update(model, optimizer, batches, device):
        if not optimizer.state:
            initial_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return original_update(model, optimizer, batches, device)

    monkeypatch.setattr(runner.base, "update", observed_update)
    output = tmp_path / "complete_queue"
    result = runner.run(packet, output, None, "cpu")
    expected_names = [c + "_" + t for c in runner.CONDITIONS for t in runner.TARGETS]
    assert result["status"] == "completed"
    assert [r["name"] for r in result["training_runs"]] == expected_names
    assert len(initial_states) == 6
    assert all(torch.equal(initial_states[0][k], state[k]) for state in initial_states[1:] for k in state)
    tapes = []
    for name in expected_names:
        rows = [json.loads(s) for s in (output / name / "updates.jsonl").read_text().splitlines()]
        tapes.append([(r["clean_indices"], r["displaced_indices"]) for r in rows])
    assert all(tape == tapes[0] for tape in tapes)
