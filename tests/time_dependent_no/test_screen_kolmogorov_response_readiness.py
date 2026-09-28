from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.time_dependent_no import screen_kolmogorov_response_readiness as screen
from scripts.time_dependent_no.screen_kolmogorov_response_readiness import (
    PROTOCOL,
    RUN_ID,
    _case,
    run_screen,
)


@pytest.fixture
def tiny_protocol():
    return replace(
        PROTOCOL,
        resolution=16,
        viscosities=(0.01,),
        seeds=(2026090601,),
        macro_dt=0.004,
        dt_max=0.002,
        high_mode=(4, 3),
        fixture_label="unit_test_only",
    )


def test_frozen_successor_preserves_physics_and_changes_grid_identity():
    assert RUN_ID == "CM_NEXT_KF_R0_20260906B"
    assert PROTOCOL.resolution == 128
    assert PROTOCOL.viscosities == (0.01, 0.005)
    assert PROTOCOL.seeds == (2026090601, 2026090602)
    assert PROTOCOL.macro_dt == 0.05
    assert PROTOCOL.dt_max == 0.002
    assert PROTOCOL.anchor_steps == 4
    assert PROTOCOL.high_mode == (18, 17)
    assert PROTOCOL.fixture_label == "preliminary_grid_refinement"


def test_tiny_case_repeat_restart_signed_probes_and_resolution(tiny_protocol):
    result = _case(tiny_protocol, 0.01, tiny_protocol.seeds[0])
    again = _case(tiny_protocol, 0.01, tiny_protocol.seeds[0])
    assert result["status"] == "completed"
    assert result["repeat_exact"] and result["restart_exact"]
    assert result["raw_input_rejected"]
    assert result["raw_input_projection_relative_l2"] > 0.005
    assert result["canonicalized_raw_projection_relative_l2"] < 1.0e-12
    assert result["anchor_sha256"] == again["anchor_sha256"]
    rows = result["probe_rows"]
    assert [(row["probe"], row["sign"]) for row in rows] == [
        ("clean", 0),
        ("low_retained", -1),
        ("low_retained", 1),
        ("high_retained", -1),
        ("high_retained", 1),
    ]
    assert rows[0]["trusted_response_gain"] is None
    assert rows[0]["trusted_response_over_anchor_rms"] == 0.0
    response_keys = (
        "half_dt_response_gain",
        "fine_restricted_response_gain",
        "base_vs_half_response_over_displacement_rms",
        "coarse_half_vs_fine_restricted_response_over_displacement_rms",
    )
    for key in response_keys:
        assert rows[0][key] is None
    for row, repeated in zip(rows, again["probe_rows"], strict=True):
        assert row["coarse_next_sha256"] == repeated["coarse_next_sha256"]
        assert row["input_lift_roundtrip_relative_l2"] < 1.0e-12
        assert row["query_projection_relative_l2"] < 1.0e-12
        assert row["output_projection_relative_l2"] < 1.0e-12
        for key in (
            "base_vs_half_dt_relative_l2",
            "coarse_half_vs_restricted_fine_half_relative_l2",
            "fine_discarded_relative_l2",
        ):
            assert np.isfinite(row[key]) and row[key] >= 0.0
        if row["sign"]:
            assert row["displacement_over_anchor_rms"] == pytest.approx(0.01)
            assert np.isfinite(row["trusted_response_gain"])
            for key in response_keys:
                assert np.isfinite(row[key]) and row[key] >= 0.0
        for record in row["calls"].values():
            assert record["substeps"] >= 2
            assert record["seconds"] >= 0.0


def test_packet_hashes_scope_and_no_clobber(tmp_path, tiny_protocol):
    output = tmp_path / "new_packet"
    result = run_screen(output, protocol=tiny_protocol)
    assert result["status"] == "completed"
    assert result["run_id"] == RUN_ID + "__UNIT_FIXTURE"
    assert result["source_stable"]
    assert result["parent_screen"] == {
        "run_id": "CM_NEXT_KF_R0_20260906A",
        "artifact_manifest_sha256": "d2101f544b9e2d55e52165a9a1d98d24fbc02ba481ea7dff71f23429e286a464",
        "source_archive_sha256": "44b2b94ab922bb743ca44539a4007bd77fcfb37c8f86a4129d90f19cbcec9e33",
        "relation": "same physical initial recipe; anchor regenerated on successor grid",
    }
    for key in (
        "qualification_claim",
        "stationarity_claim",
        "population_qualification_claim",
        "coarse_state_closure_claim",
        "primary_pde_selection",
        "existing_scientific_data_access",
        "checkpoint_access",
        "model_access",
        "remote_execution",
    ):
        assert result[key] is False
    assert result["generated_solver_states"] is True
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_screen(output, protocol=tiny_protocol)
    assert (output / "result.json").read_bytes() == before


def test_response_convergence_subtracts_matching_clean_reference(
    tiny_protocol, monkeypatch
):
    def linear_advance(stepper, state):
        if stepper.config.resolution == 2 * tiny_protocol.resolution:
            gain = 1.3
        elif stepper.config.dt_max == tiny_protocol.dt_max / 2.0:
            gain = 1.2
        else:
            gain = 1.1
        return gain * state, {"seconds": 0.0}

    monkeypatch.setattr(screen, "_advance", linear_advance)
    result = _case(tiny_protocol, 0.01, tiny_protocol.seeds[0])
    for row in result["probe_rows"][1:]:
        assert row["trusted_response_gain"] == pytest.approx(1.1)
        assert row["half_dt_response_gain"] == pytest.approx(1.2)
        assert row["fine_restricted_response_gain"] == pytest.approx(1.3)
        assert row["base_vs_half_response_over_displacement_rms"] == pytest.approx(0.1)
        assert row[
            "coarse_half_vs_fine_restricted_response_over_displacement_rms"
        ] == pytest.approx(0.1)


def test_solver_failure_is_recorded_not_promoted(tmp_path, tiny_protocol, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(
        "scripts.time_dependent_no.screen_kolmogorov_response_readiness._case", fail
    )
    result = run_screen(tmp_path / "failed_packet", protocol=tiny_protocol)
    assert result["status"] == "incomplete"
    assert result["cases"][0]["error_type"] == "RuntimeError"
    assert result["cases"][0]["error"] == "synthetic solver failure"
    assert result["qualification_claim"] is False


def test_source_drift_invalidates_packet(tmp_path, tiny_protocol, monkeypatch):
    original_hash = screen._sha256
    pilot_path = screen.REPO_ROOT / screen.SOURCE_PATHS[0]
    pilot_hash_calls = 0

    def drifting_hash(path):
        nonlocal pilot_hash_calls
        if path == pilot_path:
            pilot_hash_calls += 1
            if pilot_hash_calls > 1:
                return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(screen, "_sha256", drifting_hash)
    result = run_screen(tmp_path / "source_changed", protocol=tiny_protocol)
    assert result["cases"][0]["status"] == "completed"
    assert result["source_stable"] is False
    assert result["status"] == "invalid_source"
    assert (tmp_path / "source_changed" / "artifact_manifest.json").is_file()


@pytest.mark.parametrize(
    ("status", "source_stable"),
    [("incomplete", True), ("invalid_source", False), ("completed", False)],
)
def test_cli_exits_nonzero_for_incomplete_or_source_unstable(
    status, source_stable, monkeypatch
):
    monkeypatch.setattr("sys.argv", ["screen", "--output", "unused"])
    monkeypatch.setattr(
        screen,
        "run_screen",
        lambda output: {
            "run_id": RUN_ID + "__UNIT_FIXTURE",
            "status": status,
            "source_stable": source_stable,
            "seconds": 0.0,
        },
    )
    with pytest.raises(SystemExit, match="1"):
        screen.main()
