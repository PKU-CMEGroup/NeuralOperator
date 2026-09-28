import hashlib
import json

import pytest
import torch

from scripts.time_dependent_no import smoke_pcno_kolmogorov as smoke
from scripts.time_dependent_no.smoke_pcno_kolmogorov import (
    SmokeCase,
    run_smoke,
    synthetic_pair,
)


def test_synthetic_pair_is_explicit_non_pde_target():
    state, target = synthetic_pair(16, 2, torch.device("cpu"))
    assert state.shape == target.shape == (2, 16, 16)
    y = torch.arange(16) * (2 * torch.pi / 16)
    torch.testing.assert_close(
        target, 0.97 * state + 0.03 * torch.cos(4 * y), atol=5e-7, rtol=1e-5
    )
    assert not torch.equal(state[0], state[1])


def test_tiny_fixture_packet_and_nonoverwrite(tmp_path):
    output = tmp_path / "smoke"
    cases = (SmokeCase(16, 4, 2, 1, 4, 2, 3, 2),)
    result = run_smoke(output, "cpu", cases=cases)
    assert result["status"] == "completed"
    assert result["run_id"].endswith("__UNIT_FIXTURE")
    assert result["solver_calls"] == 0 and not result["scientific_training"]
    assert result["sources_before"] == result["sources_after"]
    row = result["cases"][0]
    assert len(row["losses"]) == 3
    assert len(row["inference_seconds"]) == 2
    assert row["median_update_seconds"] > 0
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert set(manifest["artifacts"]) == {
        "launch.json",
        "updates.jsonl",
        "case_records.json",
        "result.json",
    }
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_smoke(output, "cpu", cases=cases)


def test_invalid_device_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="device"):
        run_smoke(tmp_path / "unused", "cdua")
    assert not (tmp_path / "unused").exists()


def test_overflow_is_recorded_as_failed_packet(tmp_path, monkeypatch):
    original_pair = synthetic_pair

    def overflowing_pair(n, batch_size, device):
        state, target = original_pair(n, batch_size, device)
        return state, torch.full_like(target, 1e30)

    monkeypatch.setattr(smoke, "synthetic_pair", overflowing_pair)
    output = tmp_path / "overflow"
    result = run_smoke(output, "cpu", cases=(SmokeCase(16, 4, 2, 1, 4, 2, 3),))
    assert result["status"] == "failed"
    assert (
        result["cases"][0]["error"]
        == "synthetic initial loss must be finite and positive"
    )
    assert result["tiny_fit_90_percent_reduction"] is None
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert (
        hashlib.sha256((output / "result.json").read_bytes()).hexdigest()
        == manifest["artifacts"]["result.json"]
    )
