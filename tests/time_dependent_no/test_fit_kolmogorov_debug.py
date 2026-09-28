from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_debug as fit
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO


def _rehash_parent(parent: Path) -> None:
    manifest = fit._read(parent / "artifact_manifest.json")
    manifest["artifacts"] = {
        path.name: fit._hash(path)
        for path in parent.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    fit._write(parent / "artifact_manifest.json", manifest)


@pytest.fixture
def parent_fixture(tmp_path):
    parent, source = tmp_path / "parent", tmp_path / "source"
    parent.mkdir()
    for name in fit.PARENT_SOURCES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fit.REPO_ROOT / name, target)
    sources = {name: fit._hash(source / name) for name in fit.PARENT_SOURCES}
    protocol = {
        "resolution": 16,
        "steps": 32,
        "seeds": [2026090603, 2026090604],
        "anchor_steps": [0, 16, 32],
        "late_horizon": 2,
    }
    x, y = np.meshgrid(
        np.arange(16) * 2 * np.pi / 16, np.arange(16) * 2 * np.pi / 16, indexing="ij"
    )
    # Synthetic fixture only: no PDE solver or existing trajectory is read.
    states = np.stack(
        [
            (1 + 0.003 * i) * np.cos(x + y - 0.01 * i) + 0.1 * np.sin(2 * x - y)
            for i in range(33)
        ]
    )
    cases = []
    for seed in protocol["seeds"]:
        blocks = []
        for first, last in ((0, 0), (1, 32)):
            path = parent / f"trajectory_{seed}_{first:05d}_{last:05d}.npz"
            np.savez_compressed(
                path, steps=np.arange(first, last + 1), states=states[first : last + 1]
            )
            blocks.append(
                {
                    "file": path.name,
                    "sha256": fit._hash(path),
                    "first_step": first,
                    "last_step": last,
                    "states": last - first + 1,
                }
            )
        queries = []
        for anchor in protocol["anchor_steps"]:
            for probe, sign in (
                ("clean", 0),
                ("low_retained", -1),
                ("low_retained", 1),
                ("high_retained", -1),
                ("high_retained", 1),
            ):
                queries.append(
                    {
                        "anchor_step": anchor,
                        "probe": probe,
                        "sign": sign,
                        "spatial_relative_l2": 0.0,
                        "spatial_response_over_displacement_rms": 0.0,
                        "temporal_response_over_displacement_rms": 0.0,
                        "fine_discarded_state_relative_l2": 0.0,
                    }
                )
        cases.append(
            {
                "seed": seed,
                "status": "completed",
                "last_retained_step": 32,
                "config": {"resolution": 16},
                "blocks": blocks,
                "queries": queries,
                "late_rows": [
                    {"step": i, "restricted_state_relative_l2": 0.0} for i in (1, 2)
                ],
            }
        )
    counts = (6, 24, 24, 30, 2)
    gates = {
        name: {
            "pass": True,
            "complete": True,
            "limit": limit,
            "maximum": 0.0,
            "sample_count": count,
            "expected_sample_count": count,
        }
        for (name, limit), count in zip(fit.GATE_LIMITS.items(), counts, strict=True)
    }
    run_id = fit.PARENT_RUN_ID + "__UNIT_FIXTURE"
    result = {
        "run_id": run_id,
        "status": "completed",
        "source_stable": True,
        "engineering_gates_pass": True,
        "sources_before": sources,
        "sources_after": sources,
        "scope": "unit_test_only",
        "protocol": protocol,
        "cases": cases,
        "numeric_gates": gates,
    }
    fit._write(parent / "result.json", result)
    fit._write(
        parent / "launch.json",
        {
            "run_id": run_id,
            "protocol": protocol,
            "sources": sources,
            "gate_limits": fit.GATE_LIMITS,
        },
    )
    fit._write(
        parent / "artifact_manifest.json",
        {"run_id": run_id, "source_stable": True, "sources": sources, "artifacts": {}},
    )
    _rehash_parent(parent)
    return parent, source, states


def test_fixed_real_parent_pin_and_fixture_cannot_open_real_fit(
    parent_fixture, tmp_path
):
    assert (
        fit.EXPECTED_PARENT_MANIFEST_SHA256
        == "ebbfa9bf70eb4df892ce9d14e785d1c2f4e54e6c87e06da7d661424f39e8a939"
    )
    parent, source, _ = parent_fixture
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="pinned completed-packet audit"):
        fit.run_debug_fit(parent, source, output, "cpu")
    assert not output.exists()


def test_only_selected_seed_arrays_are_opened(parent_fixture, monkeypatch):
    parent, source, states = parent_fixture
    verified = fit.validate_parent(parent, source, unit_fixture=True)
    original = np.load
    opened = []

    def record_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fit.np, "load", record_load)
    actual = fit._load_first_pairs(parent, verified)
    np.testing.assert_array_equal(actual, states)
    assert len(opened) == 2
    assert all("2026090603" in name for name in opened)


@pytest.mark.parametrize(
    "field", ("status", "scope", "engineering_gates_pass", "numeric_gates")
)
def test_incomplete_wrong_role_or_failing_parent_rejected_before_output(
    parent_fixture, tmp_path, field
):
    parent, source, _ = parent_fixture
    result = fit._read(parent / "result.json")
    if field == "numeric_gates":
        result[field]["clean_spatial_relative_l2"]["pass"] = False
    else:
        result[field] = {
            "status": "incomplete_budget",
            "scope": "confirmation",
            "engineering_gates_pass": False,
        }[field]
    fit._write(parent / "result.json", result)
    _rehash_parent(parent)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        fit.run_debug_fit(parent, source, output, "cpu", unit_fixture=True)
    assert not output.exists()


@pytest.mark.parametrize("corrupt", ("artifact", "source"))
def test_corrupt_parent_bytes_rejected_before_arrays(
    parent_fixture, monkeypatch, corrupt
):
    parent, source, _ = parent_fixture
    path = (
        parent / "result.json"
        if corrupt == "artifact"
        else source / fit.PARENT_SOURCES[0]
    )
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(
        fit.np,
        "load",
        lambda *args, **kwargs: pytest.fail("array opened before integrity gate"),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        fit.validate_parent(parent, source, unit_fixture=True)


def test_tiny_debug_packet_metrics_checkpoint_scale_and_replay(
    parent_fixture, tmp_path
):
    parent, source, states = parent_fixture
    output = tmp_path / "debug_fit"
    result = fit.run_debug_fit(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "completed"
    assert result["scope"] == "synthetic_unit_fixture"
    assert (
        result["online_solver_calls"] == 0 and result["generalization_claim"] is False
    )
    assert result["prospective_evidence"] is False and result["heldout_access"] is False
    assert result["sources_before"] == result["sources_after"]
    assert result["fitted_input_rms_float64"] == pytest.approx(
        np.sqrt(np.mean(states[:32] ** 2))
    )
    assert result["fitted_input_rms_float64"] != pytest.approx(
        np.sqrt(np.mean(states**2))
    )
    logs = [
        json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()
    ]
    assert [row["first_pair"] for row in logs] == [0, 8, 16, 24]
    assert all(row["batch_size"] == 8 for row in logs)
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        arrays = dict(data)
    for name, array in arrays.items():
        assert fit._array_hash(array) == result["array_hashes"][name]
    launch = fit._read(output / "launch.json")
    for name, digest in launch["selected_array_hashes"].items():
        assert digest == result["array_hashes"][name]
    np.testing.assert_array_equal(arrays["initial_teacher_raw"], arrays["inputs"])
    np.testing.assert_array_equal(arrays["inputs"], states[:32].astype(np.float32))
    np.testing.assert_array_equal(arrays["targets"], states[1:].astype(np.float32))
    np.testing.assert_array_equal(
        arrays["normalized_inputs"],
        arrays["inputs"] / np.float32(result["actual_model_scale_float32"]),
    )
    assert (
        fit._metrics(arrays["teacher_next"], arrays["targets"])
        == result["teacher_metrics"]
    )
    assert (
        fit._metrics(arrays["rollout_next"], arrays["targets"])
        == result["rollout_metrics"]
    )
    assert arrays["rollout_next"].shape == (32, 16, 16)
    checkpoint = torch.load(output / "terminal_checkpoint.pt", weights_only=True)
    assert checkpoint["completed_updates"] == 4
    model = PeriodicVorticityPCNO(
        16,
        train_scale=result["actual_model_scale_float32"],
        modes=2,
        width=4,
        depth=1,
        fc_dim=8,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        first = model(torch.from_numpy(arrays["inputs"][:1]))["next_state"].numpy()[0]
    np.testing.assert_allclose(first, arrays["rollout_next"][0], rtol=1e-6, atol=1e-7)
    manifest = fit._read(output / "artifact_manifest.json")
    for name, digest in manifest["artifacts"].items():
        assert fit._hash(output / name) == digest
    assert "terminal_checkpoint.pt" in manifest["artifacts"]
    with pytest.raises(FileExistsError):
        fit.run_debug_fit(parent, source, output, "cpu", unit_fixture=True)


@pytest.mark.parametrize("nonfinite", ("last_prediction", "final_metric"))
def test_nonfinite_final_evaluation_writes_failed_packet(
    parent_fixture, tmp_path, monkeypatch, nonfinite
):
    parent, source, _ = parent_fixture
    if nonfinite == "last_prediction":
        original = PeriodicVorticityPCNO.forward
        calls = 0

        def corrupt_last(self, state):
            nonlocal calls
            prediction = original(self, state)
            calls += 1
            # Four initial teacher batches, four updates, four final teacher
            # batches and 32 recurrence steps: fail only on the final output.
            if calls == 44:
                prediction["raw_next"] = torch.full_like(state, float("nan"))
            return prediction

        monkeypatch.setattr(PeriodicVorticityPCNO, "forward", corrupt_last)
    else:
        monkeypatch.setattr(fit, "_metrics", lambda *args: {"mse": float("inf")})
    output = tmp_path / "failed_debug"
    result = fit.run_debug_fit(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "failed"
    assert "teacher_metrics" not in result
    assert result["sources_before"] == result["sources_after"]
    assert fit._read(output / "result.json") == result
    manifest = fit._read(output / "artifact_manifest.json")
    for name, digest in manifest["artifacts"].items():
        assert fit._hash(output / name) == digest
    if nonfinite == "last_prediction":
        assert calls == 44
        assert not (output / "predictions.npz").exists()
