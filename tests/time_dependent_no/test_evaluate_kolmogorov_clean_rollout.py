from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_clean_rollout as p
from tests.time_dependent_no.test_fit_kolmogorov_clean import population as population


class FixedMap(torch.nn.Module):
    def __init__(self, multiplier=1.1, bad_at=None):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.multiplier = multiplier
        self.bad_at = bad_at
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        nxt = self.multiplier * state
        if self.calls == self.bad_at:
            nxt = nxt * float("nan")
        return {"next_state": nxt, "raw_next": nxt + 100.0}


def trajectory():
    x = np.arange(16) * 2 * np.pi / 16
    wave = np.cos(x[:, None] + 2 * x[None, :])
    return np.stack([wave * 0.95**t for t in range(5)]).astype(np.float32)


def test_recurrence_uses_restricted_prediction_not_raw_or_truth():
    truth = trajectory()
    before = truth.copy()
    model = FixedMap()
    case, snapshots = p.rollout_one(model, truth, 1.0, 2, perf_counter() + 60)
    assert case["status"] == "completed" and case["completed_steps"] == 4
    assert model.calls == 4 and not model.training
    np.testing.assert_array_equal(truth, before)
    expected = truth[0].copy()
    for step, row in enumerate(case["rows"], 1):
        expected = np.float32(1.1) * expected
        difference = expected.astype(np.float64) - truth[step].astype(np.float64)
        assert row["rollout_sse"] == pytest.approx(float(np.sum(difference**2)))
        assert sum(row["spectral_error_sse"]) == pytest.approx(
            row["rollout_sse"], rel=1e-12
        )
        assert row["restriction_sse"] == pytest.approx(16 * 16 * 100**2, rel=1e-7)
    np.testing.assert_allclose(snapshots["next_state"][-1], expected, atol=1e-7)
    assert list(snapshots["step"]) == [0, 1, 4]
    teacher = np.array(
        [
            np.sum((1.1 * truth[i].astype(np.float64) - truth[i + 1]) ** 2)
            for i in range(4)
        ]
    )
    values = p.summarize(case, teacher, 1.0, 256, (1, 4))
    assert values[-1]["rollout_relative_l2"] > values[-1]["teacher_relative_l2"]


def test_physical_diagnostics_and_spectral_parseval():
    state = trajectory()[0].astype(np.float64)
    values = p.field_diagnostics(state)
    assert values["mean_vorticity"] == pytest.approx(0, abs=1e-8)
    assert values["kinetic_energy"] == pytest.approx(1 / 20, rel=1e-7)
    assert values["enstrophy"] == pytest.approx(1 / 4, rel=1e-7)
    assert values["palinstrophy"] == pytest.approx(5 / 4, rel=1e-7)
    assert sum(p.spectral_error_sse(state, 2)) == pytest.approx(np.sum(state**2))
    assert p.spectral_error_sse(state, 2)[0] > 0.999 * np.sum(state**2)


@pytest.mark.parametrize("kind", ["nonfinite", "amplitude", "budget"])
def test_failure_is_censored_without_reset_or_survivor_only_summary(kind):
    model = FixedMap(
        multiplier=2e7 if kind == "amplitude" else 1.1,
        bad_at=2 if kind == "nonfinite" else None,
    )
    deadline = perf_counter() + (60 if kind != "budget" else -1)
    case, _ = p.rollout_one(model, trajectory(), 1.0, 2, deadline)
    expected = {
        "nonfinite": ("nonfinite_prediction", 1, 2),
        "amplitude": ("amplitude_limit", 1, 1),
        "budget": ("incomplete_budget", 0, 1),
    }[kind]
    assert (case["status"], case["completed_steps"], case["failed_at_step"]) == expected
    summaries = p.summarize(case, np.ones(4), 1.0, 256, (1, 4))
    assert summaries[-1]["complete"] is False
    assert summaries[-1]["rollout_relative_l2"] is None
    assert summaries[0]["complete"] is (kind == "nonfinite")


@pytest.fixture
def fitted(population, tmp_path):
    parent, source, _, _ = population
    packet = tmp_path / "clean"
    result = p.clean.run_clean(parent, source, packet, "cpu", unit_fixture=True)
    assert result["status"] == "completed"
    return parent, source, packet, p.REPO_ROOT


def test_tiny_checkpoint_replay_and_all_role_rollouts(fitted, tmp_path):
    parent, parent_source, packet, source = fitted
    before = {f.name: p.clean._hash(f) for f in packet.iterdir()}
    output = tmp_path / "rollouts"
    result = p.run(
        parent, parent_source, packet, source, output, "cpu", unit_fixture=True
    )
    assert result["status"] == "completed" and result["all_rollouts_completed"]
    assert len(result["cases"]) == 12
    assert [c["role"] for c in result["cases"]] == ["train"] * 8 + ["development"] * 4
    assert result["failure_counts"] == {"train": 0, "development": 0}
    assert result["checkpoint_replay"]["passed"] and result["parent_evidence_stable"]
    assert result["optimization_steps"] == result["solver_calls"] == 0
    assert not result["protected_access"] and not result["geometry_fitted"]
    assert before == {f.name: p.clean._hash(f) for f in packet.iterdir()}
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert set(manifest["artifacts"]) == {f.name for f in output.iterdir()} - {
        "artifact_manifest.json"
    }
    assert all(
        p.clean._hash(output / name) == value
        for name, value in manifest["artifacts"].items()
    )
    with pytest.raises(FileExistsError):
        p.run(parent, parent_source, packet, source, output, "cpu", unit_fixture=True)


def test_tampered_checkpoint_rejected_before_loading(fitted, tmp_path, monkeypatch):
    _, _, packet, source = fitted
    result = p.clean._read(packet / "result.json")
    terminal = packet / f"terminal_{result['updates_completed']:06d}.pt"
    terminal.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("unverified pickle opened")
    )
    with pytest.raises(ValueError, match="hash"):
        p.validate_clean(packet, source, unit_fixture=True)


def test_scientific_interface_rejects_fixture_and_cpu(fitted, tmp_path):
    parent, parent_source, packet, source = fitted
    with pytest.raises(ValueError, match="pinned"):
        p.validate_clean(packet, source)
    with pytest.raises(ValueError, match="CUDA"):
        p.run(parent, parent_source, packet, source, tmp_path / "no", "cpu")
    assert not (tmp_path / "no").exists()
