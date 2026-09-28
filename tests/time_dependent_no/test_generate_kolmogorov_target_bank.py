from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import generate_kolmogorov_target_bank as bank
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet as packet
from tests.time_dependent_no.test_evaluate_kolmogorov_recovery import evaluation as evaluation


@pytest.fixture
def target_packets(packet, evaluation, monkeypatch):
    monkeypatch.setattr(bank, "TRAIN_STEPS", (1, 3))
    monkeypatch.setattr(bank, "PROBE_STEPS", (2,))
    monkeypatch.setattr(bank, "REFERENCE", replace(bank.REFERENCE, resolution=16,
                                                  macro_dt=1e-5, dt_max=1e-5))
    solver = bank.KolmogorovReferenceStepper(bank.REFERENCE)
    values = np.load(packet / "train.npy")
    for p in range(len(values)):
        values[p] = solver.rollout_canonical(solver.canonicalize(values[p, 0]), 4)[0]
    np.save(packet / "train.npy", values)
    with np.load(packet / "teacher.npz") as saved:
        teacher = {k: saved[k].copy() for k in saved.files}
    indices = teacher["sentinel_global_indices"]
    paths, steps = np.divmod(indices, 4)
    teacher["sentinel_input"], teacher["sentinel_target"] = values[paths, steps], values[paths, steps+1]
    parent = torch.load(packet / "checkpoint.pt", weights_only=False)
    from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO
    model = PeriodicVorticityPCNO(**parent["identity"]["model_config"], train_scale=2).eval()
    model.load_state_dict(parent["model"])
    with torch.no_grad():
        prediction = model(torch.from_numpy(teacher["sentinel_input"]))
    teacher["sentinel_raw"], teacher["sentinel_next"] = [prediction[k].numpy() for k in ("raw_next", "next_state")]
    np.savez(packet / "teacher.npz", **teacher)
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["artifacts"] = {k: bank.base.sha256(packet / k) for k in manifest["artifacts"]}
    bank.base.write_json(packet / "manifest.json", manifest)
    # This synthetic development path is generated independently of its labels.
    dev = solver.rollout_canonical(solver.canonicalize(values[0, 0] * .9), 4)[0][None].astype(np.float32)
    np.save(evaluation / "development.npy", dev)
    manifest = json.loads((evaluation / "manifest.json").read_text())
    manifest["artifacts"]["development.npy"] = bank.base.sha256(evaluation / "development.npy")
    bank.base.write_json(evaluation / "manifest.json", manifest)
    return packet, evaluation


def test_inputs_are_fixed_amplitude_antithetic_and_do_not_change_center():
    stepper = bank.KolmogorovReferenceStepper(replace(bank.REFERENCE, resolution=16))
    rng = np.random.default_rng(3)
    center = stepper.canonicalize(rng.normal(size=(16, 16))).astype(np.float32)
    original = center.copy()
    native = rng.normal(size=(16, 16))
    rows = list(bank.input_rows(center, native, 2., np.random.default_rng(4), stepper))
    assert len(rows) == 8
    np.testing.assert_array_equal(center, original)
    for j in range(0, len(rows), 2):
        p, m = rows[j:j+2]
        assert p[0]["sign"] == 1 and m[0]["sign"] == -1
        assert p[0]["input_rms_scaled"] == pytest.approx(p[0]["amplitude"], rel=1e-6)
        np.testing.assert_allclose((p[1].astype(float) + m[1]) / 2, center, rtol=0, atol=1e-7)
        bank.check_metrics(p[0])


def test_real_solver_bank_targets_scope_and_unused_probes(target_packets, tmp_path):
    train_path, dev_path = target_packets
    out, probes = tmp_path / "bank", tmp_path / "probes"
    result = bank.run(train_path, out, "train", "cpu")
    probe_result = bank.run(train_path, probes, "probe", "cpu", dev_path)
    assert result["qualified"] and len(result["refinements"]) == 4
    assert probe_result["qualified"] and not probe_result["refinements"]
    training_hash = bank.base.sha256(train_path / "manifest.json")
    parent_hash = result["parent_checkpoint_sha256"]
    _, arrays, _ = bank.load_bank(out, "train", training_hash, parent_hash)
    _, unused, _ = bank.load_bank(probes, "probe", training_hash, parent_hash)
    assert set(bank.TRAIN_STEPS).isdisjoint(bank.PROBE_STEPS)
    truth = np.load(train_path / "train.npy")
    solver = bank.KolmogorovReferenceStepper(bank.REFERENCE)
    for i, a in enumerate(result["anchors"]):
        np.testing.assert_array_equal(arrays["clean_targets"][i], truth[a["path_index"], a["input_step"]+1])
    for i in (0, 9, len(arrays["inputs"])-1):
        expected = solver.advance_projected(arrays["inputs"][i]).state
        np.testing.assert_array_equal(arrays["dynamics_targets"][i], expected)
    assert not arrays["inputs"].flags.writeable
    assert len(unused["inputs"]) == len(probe_result["rows"])
    with pytest.raises(ValueError, match="role/population"):
        bank.load_bank(probes, "train", training_hash, parent_hash)


def test_refinement_detects_response_change_and_unresolved_projection():
    x = np.linspace(-1., 1., 16*16).reshape(16, 16)
    solver = bank.KolmogorovReferenceStepper(replace(bank.REFERENCE, resolution=16))
    x = solver.canonicalize(x)
    inputs = np.stack([x, x + .01*x, x - .01*x])
    fine = np.stack([bank.resize_dealiased_vorticity(a, 32) for a in inputs])
    states = dict(A=inputs, B=inputs.copy(), C=fine, D=fine.copy())
    assert max(bank.refinement_metrics(states, inputs)[0].values()) < 1e-12
    states["B"][1] += .1*x
    with pytest.raises(ValueError, match="qualification failed"):
        bank.check_metrics(bank.refinement_metrics(states, inputs)[0])
    with pytest.raises(ValueError, match="projection"):
        bank.check_metrics(dict(projection_over_displacement=.1))


def test_scope_precedes_access_and_failed_bank_cannot_be_loaded(target_packets, tmp_path, monkeypatch):
    train, dev = target_packets
    with pytest.raises(ValueError, match="must not receive"):
        bank.run("missing", "unused", "train", "cpu", dev)
    with pytest.raises(ValueError, match="outside"):
        bank.run(train, train / "nested", "train", "cpu")
    monkeypatch.setattr(bank, "MAX_SECONDS", -1)
    output = tmp_path / "failed"
    with pytest.raises(TimeoutError):
        bank.run(train, output, "train", "cpu")
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["status"] == "failed" and not receipt["qualified"]
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("decoded rejected bank"))
    with pytest.raises(ValueError, match="qualification"):
        bank.load_bank(output, "train", receipt["training_manifest_sha256"], receipt["parent_checkpoint_sha256"])
