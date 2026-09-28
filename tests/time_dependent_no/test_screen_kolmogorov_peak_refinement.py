from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import screen_kolmogorov_peak_refinement as screen


def _rehash(parent):
    manifest = json.loads((parent / "artifact_manifest.json").read_text())
    manifest["artifacts"] = {
        p.name: screen._sha256(p)
        for p in parent.iterdir()
        if p.name != "artifact_manifest.json"
    }
    screen._json(parent / "artifact_manifest.json", manifest)


@pytest.fixture
def parent_fixture(tmp_path):
    """Synthetic selection metadata only; never a scientific parent or array."""
    parent = tmp_path / "parent"
    parent.mkdir()
    sources = dict(screen.FROZEN_SOURCES)
    cases = []
    for seed, role in screen.PARENT_ROLES:
        peak_step = 2
        state_hash = hashlib.sha256(str(seed).encode()).hexdigest()
        case = {
            "seed": seed,
            "role": role,
            "status": "completed",
            "maximum_palinstrophy": {
                "step": peak_step,
                "value": 3.0,
                "state_sha256": state_hash,
                "complete_trajectory": True,
            },
            "trajectory_diagnostics": [
                {"step": step, "structure": {"palinstrophy": value}}
                for step, value in enumerate((1.0, 2.0, 3.0, 2.0, 1.0))
            ],
            "queries": [
                {
                    "anchor_step": peak_step,
                    "probe": "clean",
                    "sign": 0,
                    "query_sha256": state_hash,
                    "fine_discarded_state_relative_l2": 0.009
                    if seed == 2026090617
                    else 0.008
                    if seed == 2026090621
                    else 0.002,
                }
            ],
        }
        screen._json(parent / f"case_{seed}.json", case)
        cases.append(case)
    identity = screen.PARENT_RUN_ID + "__UNIT_FIXTURE"
    common = {
        "run_id": identity,
        "protocol": {"resolution": 16, "steps": 4},
        "seed_roles": [
            {"seed": seed, "role": role} for seed, role in screen.PARENT_ROLES
        ],
    }
    screen._json(parent / "launch.json", {**common, "sources": sources})
    screen._json(
        parent / "result.json",
        {
            **common,
            "scope": "unit_test_only",
            "status": "completed",
            "engineering_gates_pass": False,
            "source_stable": True,
            "sources_before": sources,
            "sources_after": sources,
            "cases": cases,
        },
    )
    screen._json(
        parent / "artifact_manifest.json",
        {
            "run_id": identity,
            "sources": sources,
            "source_stable": True,
            "artifacts": {},
        },
    )
    _rehash(parent)
    return parent


def _verify_packet(output):
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert set(manifest["artifacts"]) == {p.name for p in output.iterdir()} - {
        "artifact_manifest.json"
    }
    for name, digest in manifest["artifacts"].items():
        assert screen._sha256(output / name) == digest
    return manifest


def test_fixed_contract_and_metadata_only_parent_selection(parent_fixture, monkeypatch):
    assert screen.RUN_ID == "CM_NEXT_KF_PEAK_20260906A"
    assert screen.PROTOCOL.resolution == 256 and screen.PROTOCOL.steps == 64
    assert screen.PROTOCOL.anchor_steps == (16, 24, 64)
    assert (
        screen.PROTOCOL.continuation_steps == 8 and screen.PROTOCOL.wall_seconds == 5400
    )
    assert screen.MINIMUM_FREE_BYTES == 512 * 1024**2
    assert screen.GATE_LIMITS == {
        "clean_spatial_relative_l2": 1e-3,
        "fine_discarded_state_relative_l2": 1e-3,
        "continuation_endpoint_relative_l2": 2e-3,
    }
    monkeypatch.setattr(
        screen.np, "load", lambda *a, **k: pytest.fail("parent array opened")
    )
    evidence = screen.validate_parent(parent_fixture, unit_fixture=True)
    assert [(s["seed"], s["parent_role"]) for s in evidence["selected"]] == list(
        screen.SELECTED_ROLES
    )
    assert len(evidence["evidence_hashes"]) == 15


def test_fixture_cannot_enter_real_identity_and_corrupt_parent_is_rejected(
    parent_fixture, tmp_path
):
    with pytest.raises(ValueError, match="pinned population"):
        screen.run_screen(parent_fixture, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
    path = parent_fixture / "case_2026090617.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        screen.validate_parent(parent_fixture, unit_fixture=True)


def test_changed_role_winner_is_not_silently_accepted(parent_fixture):
    result = json.loads((parent_fixture / "result.json").read_text())
    case = result["cases"][0]
    case["queries"][0]["fine_discarded_state_relative_l2"] = 0.1
    screen._json(parent_fixture / f"case_{case['seed']}.json", case)
    screen._json(parent_fixture / "result.json", result)
    _rehash(parent_fixture)
    with pytest.raises(ValueError, match="frozen two seeds"):
        screen.validate_parent(parent_fixture, unit_fixture=True)


def test_native_queries_full_fine_continuation_and_recomputed_gates(
    parent_fixture, tmp_path
):
    output = tmp_path / "screen"
    result = screen.run_screen(parent_fixture, output, unit_fixture=True)
    assert (
        result["run_id"].endswith("__UNIT_FIXTURE")
        and result["scope"] == "unit_test_only"
    )
    assert result["status"] == "completed" and result["parent_evidence_stable"]
    assert result["sources_before"] == result["sources_after"]
    assert (
        result["parent_array_access"] is False
        and result["population_qualification"] is False
    )
    values = {name: [] for name in screen.GATE_LIMITS}
    for case in result["cases"]:
        retained = {}
        for block in case["blocks"]:
            with np.load(output / block["file"], allow_pickle=False) as data:
                for step, state in zip(data["steps"], data["states"], strict=True):
                    assert int(step) not in retained
                    retained[int(step)] = state
        assert sorted(retained) == list(range(5))
        config = screen.KolmogorovReferenceConfig(**case["config"])
        base = screen.BudgetedReferenceStepper(config, float("inf"))
        half = screen.BudgetedReferenceStepper(
            replace(config, dt_max=0.001, cfl=0.2), float("inf")
        )
        fine = screen.BudgetedReferenceStepper(
            replace(half.config, resolution=32), float("inf")
        )
        np.testing.assert_array_equal(retained[0], screen._initial(base, case["seed"]))
        assert case["comparison_configs"]["half"]["dt_max"] == 0.001
        assert case["comparison_configs"]["half"]["cfl"] == 0.2
        assert case["comparison_configs"]["fine"]["cfl"] == 0.2
        peak = int(
            np.argmax(
                [d["structure"]["palinstrophy"] for d in case["trajectory_diagnostics"]]
            )
        )
        assert case["maximum_palinstrophy"]["step"] == peak
        assert case["expected_anchor_steps"] == sorted({1, 2, 4, peak})
        for row in case["queries"]:
            assert row["status"] == "completed"
            with np.load(output / row["file"], allow_pickle=False) as data:
                arrays = dict(data)
            assert set(arrays) == {"input", "base_next", "half_next", "fine_next"}
            for name, array in arrays.items():
                assert array.dtype == np.float64 and row["array_hashes"][
                    name
                ] == screen._state_hash(array)
            np.testing.assert_array_equal(arrays["input"], retained[row["anchor_step"]])
            np.testing.assert_array_equal(
                arrays["base_next"], base.advance_canonical(arrays["input"]).state
            )
            np.testing.assert_array_equal(
                arrays["half_next"], half.advance_canonical(arrays["input"]).state
            )
            np.testing.assert_array_equal(
                arrays["fine_next"],
                fine.advance_canonical(
                    screen.resize_dealiased_vorticity(arrays["input"], 32)
                ).state,
            )
            restricted = screen.resize_dealiased_vorticity(arrays["fine_next"], 16)
            spatial = np.linalg.norm(arrays["half_next"] - restricted) / np.linalg.norm(
                restricted
            )
            discarded = np.linalg.norm(
                arrays["fine_next"] - screen.resize_dealiased_vorticity(restricted, 32)
            ) / np.linalg.norm(arrays["fine_next"])
            assert row["spatial_relative_l2"] == pytest.approx(spatial)
            assert row["fine_discarded_state_relative_l2"] == pytest.approx(discarded)
            values["clean_spatial_relative_l2"].append(row["spatial_relative_l2"])
            values["fine_discarded_state_relative_l2"].append(
                row["fine_discarded_state_relative_l2"]
            )
        if case["seed"] != screen.CONTINUATION_SEED:
            assert case["continuation_rows"] == []
            continue
        previous_coarse = retained[peak]
        previous_fine = screen.resize_dealiased_vorticity(previous_coarse, 32)
        assert case["continuation_anchor_step"] == peak
        for row in case["continuation_rows"]:
            with np.load(output / row["file"], allow_pickle=False) as data:
                arrays = dict(data)
            np.testing.assert_array_equal(arrays["coarse_input"], previous_coarse)
            np.testing.assert_array_equal(arrays["fine_input"], previous_fine)
            previous_coarse = half.advance_canonical(previous_coarse).state
            previous_fine = fine.advance_canonical(previous_fine).state
            np.testing.assert_array_equal(arrays["coarse_next"], previous_coarse)
            np.testing.assert_array_equal(arrays["fine_next"], previous_fine)
        values["continuation_endpoint_relative_l2"].append(
            case["continuation_rows"][-1]["restricted_state_relative_l2"]
        )
    for name, samples in values.items():
        gate = result["numeric_gates"][name]
        assert gate["maximum"] == max(samples) and gate["complete"]
        assert gate["sample_count"] == gate["expected_sample_count"] == len(samples)
        assert gate["pass"] == all(value <= gate["limit"] for value in samples)
    _verify_packet(output)
    with pytest.raises(FileExistsError):
        screen.run_screen(parent_fixture, output, unit_fixture=True)


@pytest.mark.parametrize("peak", (0, 2))
def test_peak_tie_or_fixed_anchor_deduplication(
    parent_fixture, tmp_path, monkeypatch, peak
):
    original = screen.BudgetedReferenceStepper

    class FixtureDiagnostics(original):
        def diagnostics_canonical(self, state):
            index = getattr(self, "fixture_index", 0)
            self.fixture_index = index + 1
            value = 1.0 if peak == 0 else (1.0, 2.0, 5.0, 4.0, 3.0)[index]
            return replace(super().diagnostics_canonical(state), palinstrophy=value)

    monkeypatch.setattr(screen, "BudgetedReferenceStepper", FixtureDiagnostics)
    result = screen.run_screen(parent_fixture, tmp_path / "ties", unit_fixture=True)
    for case in result["cases"]:
        assert case["maximum_palinstrophy"]["step"] == peak
        assert sorted(row["anchor_step"] for row in case["queries"]) == sorted(
            {1, 2, 4, peak}
        )


def test_budget_retains_partial_query_and_unstarted_seed(
    parent_fixture, tmp_path, monkeypatch
):
    advance, now = screen._advance, screen.perf_counter
    expired = False

    def expire_at_fine(stepper, state, deadline):
        nonlocal expired
        if stepper.config.resolution == 32:
            expired = True
            raise screen.BudgetExceeded("synthetic fine-query timeout")
        return advance(stepper, state, deadline)

    monkeypatch.setattr(screen, "_advance", expire_at_fine)
    monkeypatch.setattr(
        screen, "perf_counter", lambda: now() + (1000 if expired else 0)
    )
    output = tmp_path / "budget"
    result = screen.run_screen(parent_fixture, output, unit_fixture=True)
    assert (
        result["status"] == "incomplete_budget"
        and result["engineering_gates_pass"] is None
    )
    first, second = result["cases"]
    assert first["last_retained_step"] == 1
    assert first["queries"][0]["status"] == "incomplete_budget"
    with np.load(output / first["queries"][0]["file"], allow_pickle=False) as data:
        assert set(data.files) == {"input", "base_next", "half_next"}
    assert second["started_generation"] is False and second["blocks"] == []
    assert (
        result["numeric_gates"]["clean_spatial_relative_l2"]["expected_sample_count"]
        is None
    )
    _verify_packet(output)


def test_numerical_failure_is_not_infrastructure_failure(
    parent_fixture, tmp_path, monkeypatch
):
    original = screen._clean_query

    def high_error(*args):
        original(*args)
        args[1]["queries"][-1]["spatial_relative_l2"] = 0.1

    monkeypatch.setattr(screen, "_clean_query", high_error)
    result = screen.run_screen(
        parent_fixture, tmp_path / "numeric_fail", unit_fixture=True
    )
    assert result["status"] == "completed" and result["engineering_gates_pass"] is False
    assert result["numeric_gates"]["clean_spatial_relative_l2"]["pass"] is False


def test_disk_preflight_and_failed_cli_are_fail_closed(
    parent_fixture, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        screen.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=screen.MINIMUM_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="512 MiB"):
        screen.run_screen(parent_fixture, tmp_path / "forbidden", unit_fixture=True)
    assert not (tmp_path / "forbidden").exists()
    monkeypatch.setattr(
        screen,
        "run_screen",
        lambda *a: {"status": "completed", "engineering_gates_pass": False},
    )
    monkeypatch.setattr(
        "sys.argv", ["screen", "--parent", "unused", "--output", "unused"]
    )
    with pytest.raises(SystemExit) as error:
        screen.main()
    assert error.value.code == 1
