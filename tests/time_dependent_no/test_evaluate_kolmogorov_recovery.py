from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as runner
from tests.time_dependent_no.test_fit_kolmogorov_recovery import packet as packet


@pytest.fixture
def evaluation(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DEVELOPMENT_SEEDS", [2026090621])
    monkeypatch.setattr(runner, "INPUT_STEPS", (1, 3))
    monkeypatch.setattr(runner, "TIME_BANDS", ((0, 2), (2, 4)))
    monkeypatch.setattr(runner, "HORIZONS", (2, 4))
    monkeypatch.setattr(runner, "SNAPSHOTS", {0, 1, 3, 4})
    directory = tmp_path / "evaluation"
    directory.mkdir()
    train = np.load(packet / "train.npy")
    development = train[:1] * np.float32(0.9)
    np.save(directory / "development.npy", development)
    arrays = dict(train_seeds=np.array(runner.training.TRAIN_SEEDS),
                  development_seeds=np.array(runner.DEVELOPMENT_SEEDS),
                  input_steps=np.array(runner.INPUT_STEPS))
    for role, values in (("train", train), ("development", development)):
        center = values[:, runner.INPUT_STEPS]
        states = center + np.float32(0.03)
        arrays[f"{role}_states"] = states
        arrays[f"{role}_errors"] = states - center
    np.savez(directory / "donors.npz", **arrays)
    parent = json.loads((packet / "manifest.json").read_text())
    runner.write_json(directory / "manifest.json", dict(
        schema_version=1, role="open_development",
        train_seeds=runner.training.TRAIN_SEEDS,
        development_seeds=runner.DEVELOPMENT_SEEDS, input_steps=list(runner.INPUT_STEPS),
        parent_checkpoint_sha256=parent["artifacts"]["checkpoint.pt"],
        artifacts={name: runner.sha256(directory / name)
                   for name in ("development.npy", "donors.npz")}))
    return directory


class FixedMap(torch.nn.Module):
    def __init__(self, multiplier=1.1, nonfinite_at=None):
        super().__init__()
        self.multiplier = multiplier
        self.nonfinite_at = nonfinite_at
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        nxt = self.multiplier * value
        if self.calls == self.nonfinite_at:
            nxt = nxt * float("nan")
        return dict(next_state=nxt, raw_next=nxt + 10)


def test_finite_antithetic_identity_with_odd_even_and_signed_alignment():
    center = np.array([[1., 2.], [3., 4.]])
    noise = np.array([[.2, -.1], [.1, -.3]])
    step = lambda x: 0.5 * x + 0.2 * x**2
    target = center * .7
    row = runner.pair_metrics(step(center), step(center + noise), step(center - noise),
                              target, center + noise, center - noise, center, 2)
    assert abs(row["antithetic_closure_scaled"]) < 1e-15
    assert abs(row["positive_closure_scaled"]) < 1e-15
    assert row["even_rms_scaled"] > 0
    assert row["paired_mse_scaled"] == pytest.approx(
        row["bias_plus_even_rms_scaled"]**2 + row["odd_rms_scaled"]**2)
    assert row["positive_mse_scaled"] == pytest.approx(
        row["paired_mse_scaled"] + row["signed_alignment_scaled"])


def test_teacher_evaluates_every_exact_pair_and_saves_time_resolved_sse():
    truth = np.stack([np.full((16, 16), 2.**i, np.float32) for i in range(5)])[None]
    model = FixedMap(multiplier=2.)
    arrays, _ = runner.teacher_assay(model, {"train": truth}, "cpu")
    np.testing.assert_array_equal(arrays["train_sse"], 0)
    np.testing.assert_allclose(arrays["train_raw_sse"], 16 * 16 * 100)
    np.testing.assert_allclose(arrays["train_restriction_sse"], 16 * 16 * 100)
    assert arrays["train_sse"].shape == (1, 4)
    assert model.calls == 1


@pytest.mark.parametrize("kind", ["role", "seeds", "hash", "index"])
def test_evaluation_boundaries_before_model_load(packet, evaluation, monkeypatch, kind):
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if kind == "role":
        manifest["role"] = "sealed_test"
    elif kind == "seeds":
        manifest["development_seeds"] = [2026090999]
    elif kind == "hash":
        with (evaluation / "development.npy").open("ab") as stream:
            stream.write(b"changed")
    else:
        manifest["input_steps"] = [0, 3]
    runner.write_json(manifest_path, manifest)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("decoded before scope/hash"))
    parent = json.loads((packet / "manifest.json").read_text())
    with pytest.raises(ValueError):
        runner.load_evaluation(evaluation, parent)


def test_unexpected_donor_arrays_are_rejected(packet, evaluation):
    with np.load(evaluation / "donors.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["unregistered_population"] = np.ones((1, 16, 16), np.float32)
    np.savez(evaluation / "donors.npz", **arrays)
    manifest = json.loads((evaluation / "manifest.json").read_text())
    manifest["artifacts"]["donors.npz"] = runner.sha256(evaluation / "donors.npz")
    runner.write_json(evaluation / "manifest.json", manifest)
    parent = json.loads((packet / "manifest.json").read_text())
    with pytest.raises(ValueError, match="unexpected donor array set"):
        runner.load_evaluation(evaluation, parent)


def test_assay_never_composes_and_raw_vectors_recompute(packet, evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "rollout_case", lambda *a, **kw: pytest.fail("assay composed a path"))
    output = tmp_path / "assay"
    result = runner.run(packet, evaluation, output, "assay", "cpu")
    assert result["status"] == "completed" and result["recurrent_model_calls"] == 0
    assert len(result["responses"]) == 3 * 2 * 2
    assert {r["role"] for r in result["responses"]} == {"train", "development"}
    with np.load(output / "responses.npz") as saved:
        for i, row in enumerate(result["responses"]):
            independent = runner.pair_metrics(*(saved[key][i] for key in
                ("clean", "positive", "negative", "target", "positive_input", "negative_input", "center")), 2.)
            assert independent == {key: row[key] for key in independent}
    assert not list(output.glob("rollout_*"))


def test_gaussian_inputs_identical_across_models_and_native_positive_is_exact(packet, evaluation):
    parent = json.loads((packet / "manifest.json").read_text())
    _, dev, donors, _ = runner.load_evaluation(evaluation, parent)
    train = np.load(packet / "train.npy")
    populations = dict(train=train, development=dev)
    first, rows = runner.response_assay(FixedMap(1.1), populations, donors, 2., "cpu")
    second, _ = runner.response_assay(FixedMap(.9), populations, donors, 2., "cpu")
    np.testing.assert_array_equal(first["positive_input"], second["positive_input"])
    np.testing.assert_array_equal(first["negative_input"], second["negative_input"])
    for index, row in enumerate(rows):
        if row["direction"] == "native_parent":
            seeds = runner.training.TRAIN_SEEDS if row["role"] == "train" else runner.DEVELOPMENT_SEEDS
            actual = donors[f"{row['role']}_states"][seeds.index(row["seed"]),
                                                      runner.INPUT_STEPS.index(row["input_step"])]
            np.testing.assert_array_equal(first["positive_input"][index], actual)


def test_rollout_uses_restricted_state_and_retains_requested_snapshots(evaluation):
    reference = np.ones((5, 16, 16), np.float32)
    model = FixedMap(1.1)
    case, saved = runner.rollout_case(model, reference, 1., "cpu")
    assert case["status"] == "completed" and model.calls == 4
    np.testing.assert_array_equal(saved["step"], [0, 1, 3, 4])
    expected = reference[0]
    for row in case["rows"]:
        expected = np.float32(1.1) * expected
        assert row["sse"] == pytest.approx(runner.squared(expected.astype(np.float64) - 1))
    np.testing.assert_array_equal(saved["state"][-1], expected)
    assert case["horizons"][-1]["complete"]


@pytest.mark.parametrize("failure", ["amplitude", "nonfinite"])
def test_censoring_never_drops_failed_paths_or_reports_guard_horizon(evaluation, failure):
    model = FixedMap(2e6 if failure == "amplitude" else 1.1,
                     nonfinite_at=2 if failure == "nonfinite" else None)
    truth = np.ones((5, 16, 16), np.float32)
    case, _ = runner.rollout_case(model, truth, 1., "cpu")
    assert case["status"] == ("amplitude_limit" if failure == "amplitude" else "nonfinite_prediction")
    assert all(not h["complete"] and h["relative_l2"] is None for h in case["horizons"])
    good, _ = runner.rollout_case(FixedMap(1.), truth, 1., "cpu")
    pooled = runner.pooled_horizons([dict(case, role="train"), dict(good, role="train")])
    selected = [row for row in pooled if row["role"] == "train"]
    assert all(row["paths"] == 2 and row["complete_paths"] == 1
               and row["relative_l2"] is None for row in selected)


@pytest.mark.parametrize("corruption", ["resource", "learning_rate", "parent"])
def test_new_terminal_identity_and_resource_rejected(packet, evaluation, tmp_path, corruption):
    fit_dir = tmp_path / "fit"
    runner.training.run(packet, fit_dir, "recovery", "fit", "cpu")
    manifest, captured, fit, _, digest = runner.training.load_inputs(packet)
    model, identity, _ = runner.evaluation_model(captured, fit, manifest, digest, fit_dir, "cpu")
    assert identity["identity"]["arm"] == "recovery"
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    record = json.loads((fit_dir / "result.json").read_text())
    if corruption == "resource":
        record["phase"] = "resource"
    elif corruption == "learning_rate":
        record["identity"]["learning_rate"] *= 2
    else:
        record["identity"]["parent_checkpoint_sha256"] = "0" * 64
    runner.write_json(fit_dir / "result.json", record)
    with pytest.raises(ValueError, match="identity/schedule/parent"):
        runner.evaluation_model(captured, fit, manifest, digest, fit_dir, "cpu")


def test_rollout_phase_does_not_run_assay_and_output_is_immutable(packet, evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "teacher_assay", lambda *a, **kw: pytest.fail("rollout queried teacher"))
    monkeypatch.setattr(runner, "response_assay", lambda *a, **kw: pytest.fail("rollout queried probes"))
    output = tmp_path / "rollout"
    result = runner.run(packet, evaluation, output, "rollout", "cpu")
    assert result["status"] == "completed" and len(result["cases"]) == 3
    assert len(result["artifacts"]) == 3
    with pytest.raises(FileExistsError):
        runner.run(packet, evaluation, output, "rollout", "cpu")


def test_failure_receipt_survives_assay_exception(packet, evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "teacher_assay", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected")))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(packet, evaluation, output, "assay", "cpu")
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed" and "injected" in result["error"]


def test_reached_probes_match_amplitudes_and_preserve_exact_native_endpoints():
    rng = np.random.default_rng(7)
    center = rng.normal(size=(16, 16)).astype(np.float32)
    parent = (center + rng.normal(0, .2, center.shape)).astype(np.float32)
    recovery = (center + rng.normal(0, .04, center.shape)).astype(np.float32)
    probes, skipped = runner.reached_probes(center, parent, recovery,
                                           rng.normal(size=center.shape), 2.)
    assert len(probes) == 15 and not skipped
    for row, plus, minus in probes:
        assert plus.dtype == minus.dtype == np.float32
        actual = np.sqrt(np.mean((plus.astype(np.float64) - center)**2)) / 2
        assert actual == pytest.approx(row["requested_rms_scaled"], rel=2e-6)
        np.testing.assert_allclose((plus.astype(np.float64) + minus) / 2,
                                   center, atol=2e-7, rtol=0)
        if row["amplitude"] == "native_" + row["direction"]:
            np.testing.assert_array_equal(plus, parent if row["direction"] == "parent" else recovery)
    probes, skipped = runner.reached_probes(center, center, recovery,
                                           np.zeros_like(center), 2.)
    assert skipped == ["parent", "gaussian"] and len(probes) == 5
    zero = next(p for p in probes if p[0]["amplitude"] == "native_parent")
    np.testing.assert_array_equal(zero[1], center)
    np.testing.assert_array_equal(zero[2], center)


def test_reached_response_shared_inputs_and_saved_vector_closure(evaluation, packet, monkeypatch):
    monkeypatch.setattr(runner, "REACHED_STEPS", (1, 2, 3))
    train = np.load(packet / "train.npy")
    dev = np.load(evaluation / "development.npy")
    populations = dict(train=train, development=dev)
    donors = {kind: {role: values + np.float32(delta) for role, values in populations.items()}
              for kind, delta in (("parent", .13), ("recovery", .03))}
    first, rows, skipped = runner.reached_response_assay(FixedMap(1.1), populations, donors, 2., "cpu")
    second, others, _ = runner.reached_response_assay(FixedMap(.9), populations, donors, 2., "cpu")
    assert len(rows) == 3 * 3 * 15 and not skipped
    assert [r["input_sha256"] for r in rows] == [r["input_sha256"] for r in others]
    for key in ("positive_input", "negative_input", "center", "target"):
        np.testing.assert_array_equal(first[key], second[key])
    for i, row_index in enumerate(first["row_indices"]):
        row = rows[row_index]
        metric = runner.pair_metrics(*(first[key][i] for key in
            ("clean", "positive", "negative", "target", "positive_input", "negative_input", "center")), 2.)
        assert metric == {key: row[key] for key in metric}
    assert all(abs(r["positive_closure_scaled"]) < 1e-13 for r in rows)
    assert any(r["positive_response_gain"] > 1 for r in rows)


def test_dense_replay_and_bound_reached_phase(packet, evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPLAY_HORIZON", 4)
    monkeypatch.setattr(runner, "REACHED_STEPS", (1, 2, 3))
    fit = tmp_path / "fit_recovery"
    runner.training.run(packet, fit, "recovery", "fit", "cpu")
    for kind, fitted in (("parent", None), ("recovery", fit)):
        prior = tmp_path / ("prior_" + kind)
        runner.run(packet, evaluation, prior, "rollout", "cpu", fitted)
        replay = tmp_path / ("replay_" + kind)
        result = runner.run(packet, evaluation, replay, "replay", "cpu", fitted,
                            prior_rollout=prior)
        assert len(result["replay_checks"]) == 3
        assert all(r["bitwise_equal"] for r in result["replay_checks"])
        with np.load(replay / "dense.npz") as data:
            assert data["train"].shape == (2, 5, 16, 16)
            assert data["development"].shape == (1, 5, 16, 16)
    # The response phase consumes the declared frozen replays; it cannot compose.
    monkeypatch.setattr(runner, "replay_reached_states",
                        lambda *a, **kw: pytest.fail("response replayed a trajectory"))
    result = runner.run(packet, evaluation, tmp_path / "reached", "reached_response", "cpu",
                        donor_replays=tmp_path)
    assert result["status"] == "completed" and result["recurrent_model_calls"] == 0
    assert len(result["responses"]) == 3 * 3 * 15
    assert len(result["prior_artifacts"]) == 4
    # A scope substitution must be rejected before any donor arrays are decoded.
    receipt = tmp_path / "replay_parent" / "result.json"
    altered = json.loads(receipt.read_text())
    altered["evaluation_manifest_sha256"] = "0" * 64
    runner.write_json(receipt, altered)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("decoded before scope check"))
    with pytest.raises(ValueError, match="scope/phase"):
        runner.load_reached_donors(tmp_path, result["training_manifest_sha256"],
            result["evaluation_manifest_sha256"], result["model"]["checkpoint_sha256"],
            result["sources"])


def test_dense_replay_rejects_changed_historical_state(evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPLAY_HORIZON", 4)
    monkeypatch.setattr(runner.training, "TRAIN_SEEDS", [1])
    model = FixedMap(1.1)
    reference = np.ones((5, 16, 16), np.float32)
    _, snapshots = runner.rollout_case(model, reference, 1., "cpu")
    snapshots["state"][1] += np.float32(.01)
    name = "rollout_train_1.npz"
    np.savez(tmp_path / name, **snapshots)
    prior = dict(artifacts={name: runner.sha256(tmp_path / name)})
    with pytest.raises(ValueError, match="differs from prior"):
        runner.replay_reached_states(FixedMap(1.1), {"train": reference[None]},
                                    1., "cpu", prior, tmp_path)


@pytest.mark.parametrize("phase,kwargs", [
    ("replay", {}), ("reached_response", {}),
    ("rollout", {"prior_rollout": "unexpected"}),
    ("assay", {"donor_replays": "unexpected"}),
])
def test_new_phase_arguments_fail_before_input_access(phase, kwargs):
    with pytest.raises(ValueError, match="requires"):
        runner.run("missing", "missing", "unused", phase, "cpu", **kwargs)
