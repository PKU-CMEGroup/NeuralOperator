from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import screen_kolmogorov_trajectory_readiness as screen


@pytest.fixture
def tiny_protocol():
    return replace(
        screen.PROTOCOL,
        resolution=16,
        seeds=(2026090603,),
        steps=4,
        anchor_steps=(0, 2, 4),
        late_anchor=2,
        late_horizon=2,
        block_steps=2,
        macro_dt=0.004,
        dt_max=0.002,
        high_mode=(4, 3),
        wall_seconds=60,
        fixture_label="unit_test_only",
    )


def test_fixed_long_protocol_and_tolerances():
    assert screen.RUN_ID == "CM_NEXT_KF_LONG_20260906A"
    p = screen.PROTOCOL
    assert p.resolution == 128 and p.seeds == (2026090603, 2026090604)
    assert p.steps == 512 and p.macro_dt == 0.05 and p.dt_max == 0.002
    assert p.anchor_steps == (0, 64, 256, 512) and p.late_anchor == 256
    assert p.late_horizon == 8 and p.block_steps == 32 and p.wall_seconds == 5400
    assert p.high_mode == (18, 17)
    assert screen.GATE_LIMITS == {
        "clean_spatial_relative_l2": 1e-3,
        "spatial_response_over_displacement_rms": 1e-2,
        "temporal_response_over_displacement_rms": 1e-3,
        "fine_discarded_state_relative_l2": 1e-3,
        "late_endpoint_relative_l2": 2e-3,
    }


def _verify_packet(output):
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert set(manifest["artifacts"]) == {p.name for p in output.iterdir()} - {
        "artifact_manifest.json"
    }
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_tiny_complete_bank_blocks_and_hashes(tmp_path, tiny_protocol):
    output = tmp_path / "complete"
    result = screen.run_screen(output, protocol=tiny_protocol)
    assert result["status"] == "completed" and result["source_stable"]
    assert result["run_id"].endswith("__UNIT_FIXTURE")
    case = result["cases"][0]
    assert case["last_retained_step"] == 4
    assert [r["anchor_step"] for r in case["queries"]] == [0] * 5 + [2] * 5 + [4] * 5
    assert [r["step"] for r in case["late_rows"]] == [1, 2]
    indices, states = [], []
    for block in case["blocks"]:
        with np.load(output / block["file"], allow_pickle=False) as data:
            assert data["states"].dtype == np.float64
            indices.extend(data["steps"].tolist())
            states.extend(data["states"])
    assert indices == list(range(5))
    retained = dict(zip(indices, states, strict=True))
    for query in case["queries"]:
        if query["probe"] == "clean":
            assert query["query_sha256"] == screen._state_hash(
                retained[query["anchor_step"]]
            )
    half_config = replace(
        screen.KolmogorovReferenceConfig(**case["config"]),
        dt_max=tiny_protocol.dt_max / 2,
        cfl=case["config"]["cfl"] / 2,
    )
    assert case["comparison_configs"]["half_dt"]["cfl"] == 0.2
    assert case["comparison_configs"]["fine_half_dt"]["cfl"] == 0.2
    late_initial = retained[tiny_protocol.late_anchor]
    expected_coarse = (
        screen.KolmogorovReferenceStepper(half_config)
        .advance_canonical(late_initial)
        .state
    )
    expected_fine = (
        screen.KolmogorovReferenceStepper(
            replace(half_config, resolution=2 * tiny_protocol.resolution)
        )
        .advance_canonical(
            screen.resize_dealiased_vorticity(
                late_initial, 2 * tiny_protocol.resolution
            )
        )
        .state
    )
    assert case["late_rows"][0]["coarse_sha256"] == screen._state_hash(expected_coarse)
    assert case["late_rows"][0]["fine_sha256"] == screen._state_hash(expected_fine)
    checkpoint = json.loads(
        (output / f"checkpoint_{tiny_protocol.seeds[0]}.json").read_text()
    )
    assert checkpoint["last_step"] == 4
    assert checkpoint["state_sha256"] == screen._state_hash(states[-1])
    assert all(
        g["complete"] and g["pass"] is not None
        for g in result["numeric_gates"].values()
    )
    assert result["numeric_gates"]["late_endpoint_relative_l2"]["sample_count"] == 1
    for row in case["queries"]:
        for name in (
            "fine_discarded_response_over_displacement_rms",
            "full_fine_response_gain",
            "full_fine_response_mismatch_over_displacement_rms",
        ):
            assert np.isfinite(row[name]) if row["sign"] else row[name] is None
    for name in (
        "qualification_claim",
        "stationarity_claim",
        "manifold_claim",
        "population_qualification_claim",
        "primary_pde_selection",
    ):
        assert result[name] is False
    _verify_packet(output)
    with pytest.raises(FileExistsError):
        screen.run_screen(output, protocol=tiny_protocol)


def test_budget_flushes_partial_block_and_keeps_incomplete_packet(
    tmp_path, tiny_protocol, monkeypatch
):
    advance = screen._advance
    calls = 0

    def stop_after_first_transition(*args):
        nonlocal calls
        calls += 1
        if calls == 17:
            raise screen.BudgetExceeded("synthetic budget exit")
        return advance(*args)

    monkeypatch.setattr(screen, "_advance", stop_after_first_transition)
    output = tmp_path / "budget"
    result = screen.run_screen(output, protocol=tiny_protocol)
    assert result["status"] == "incomplete_budget"
    assert result["engineering_gates_pass"] is None
    assert result["cases"][0]["last_retained_step"] == 1
    assert all(g["pass"] is None for g in result["numeric_gates"].values())
    assert result["cases"][0]["blocks"][-1]["last_step"] == 1
    _verify_packet(output)


def test_disk_preflight_fails_before_directory_creation(
    tmp_path, tiny_protocol, monkeypatch
):
    monkeypatch.setattr(
        screen.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=512 * 1024**2 - 1),
    )
    output = tmp_path / "no_space"
    with pytest.raises(RuntimeError, match="512 MiB"):
        screen.run_screen(output, protocol=tiny_protocol)
    assert not output.exists()


def test_late_composition_does_not_reset_fine_state(tiny_protocol, monkeypatch):
    config = screen.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.004, dt_max=0.001
    )
    coarse = screen.KolmogorovReferenceStepper(config)
    fine = screen.KolmogorovReferenceStepper(replace(config, resolution=32))
    initial = screen._initial(coarse, tiny_protocol.seeds[0])
    x, y = screen._grid(32)
    outside_coarse_band = np.cos(7 * x + y)
    fine_inputs = []

    def injected_advance(stepper, state, deadline):
        if stepper.config.resolution == 32:
            fine_inputs.append(state.copy())
            return state + outside_coarse_band, {"seconds": 0.0}
        return state.copy(), {"seconds": 0.0}

    monkeypatch.setattr(screen, "_advance", injected_advance)
    rows = list(screen._late_rows(coarse, fine, initial, tiny_protocol, float("inf")))
    np.testing.assert_allclose(
        fine_inputs[1] - fine_inputs[0], outside_coarse_band, atol=1e-12
    )
    assert (
        rows[-1]["fine_discarded_state_relative_l2"]
        > rows[0]["fine_discarded_state_relative_l2"]
    )


def test_deadline_is_visible_inside_solver_substeps():
    stepper = screen.BudgetedReferenceStepper(
        screen.KolmogorovReferenceConfig(resolution=16), deadline=-1.0
    )
    with pytest.raises(screen.BudgetExceeded, match="RK4 substeps"):
        stepper.advance_canonical(stepper.laminar_vorticity())


def test_nonexpired_deadline_wrapper_is_bitwise_identical(tiny_protocol):
    config = screen.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.004, dt_max=0.002
    )
    original = screen.KolmogorovReferenceStepper(config)
    bounded = screen.BudgetedReferenceStepper(config, deadline=float("inf"))
    state = screen._initial(original, tiny_protocol.seeds[0])
    expected = original.advance_canonical(state)
    actual = bounded.advance_canonical(state)
    assert np.array_equal(actual.state, expected.state)
    assert actual.diagnostics == expected.diagnostics


def test_temporal_refinement_reduces_actual_cfl_limited_substeps():
    config = screen.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.004, dt_max=0.002, cfl=0.4
    )
    base = screen.KolmogorovReferenceStepper(config)
    refined = screen.KolmogorovReferenceStepper(
        replace(config, dt_max=config.dt_max / 2, cfl=config.cfl / 2)
    )
    state = base.canonicalize(50 * base.laminar_vorticity())
    coarse_step = base.advance_canonical(state)
    refined_step = refined.advance_canonical(state)
    assert coarse_step.diagnostics.maximum_substep < config.dt_max
    assert (
        refined_step.diagnostics.maximum_substep
        < 0.51 * coarse_step.diagnostics.maximum_substep
    )
    assert refined_step.diagnostics.substeps > coarse_step.diagnostics.substeps


def test_missing_query_counts_cannot_pass_numeric_gates(tiny_protocol):
    records = [{"status": "completed", "queries": [], "late_rows": []}]
    gates = screen.numeric_gates(records, tiny_protocol)
    assert all(not gate["complete"] and gate["pass"] is None for gate in gates.values())
    assert gates["clean_spatial_relative_l2"]["expected_sample_count"] == 3
    assert (
        gates["spatial_response_over_displacement_rms"]["expected_sample_count"] == 12
    )
    assert gates["fine_discarded_state_relative_l2"]["expected_sample_count"] == 15


def test_discarded_fine_response_is_not_hidden_by_restriction(
    tiny_protocol, monkeypatch
):
    config = screen.KolmogorovReferenceConfig(resolution=16, viscosity=0.01)
    coarse = screen.KolmogorovReferenceStepper(config)
    fine = screen.KolmogorovReferenceStepper(replace(config, resolution=32))
    initial = screen._initial(coarse, tiny_protocol.seeds[0])

    def lift_response(stepper, state, deadline):
        if stepper.config.resolution == 32:
            x, y = screen._grid(32)
            response = 0.1 * np.mean(state * np.cos(x + y)) * np.cos(7 * x + y)
            return state + response, {"seconds": 0.0}
        return state.copy(), {"seconds": 0.0}

    monkeypatch.setattr(screen, "_advance", lift_response)
    rows = list(
        screen._query_rows(coarse, coarse, fine, initial, tiny_protocol, float("inf"))
    )
    for row in rows[1:3]:
        assert row["spatial_response_over_displacement_rms"] < 1e-11
        assert row["fine_discarded_response_over_displacement_rms"] == pytest.approx(
            0.05
        )
        assert row[
            "full_fine_response_mismatch_over_displacement_rms"
        ] == pytest.approx(0.05)
        assert row["full_fine_response_gain"] == pytest.approx(np.sqrt(1 + 0.05**2))
