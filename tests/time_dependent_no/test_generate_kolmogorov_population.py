from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import generate_kolmogorov_population as p


def _manifest(root, sources):
    value = {
        "run_id": p.PARENT_RUN_ID + "__UNIT_FIXTURE",
        "sources": sources,
        "source_stable": True,
        "artifacts": {
            f.name: p.long._sha256(f)
            for f in root.iterdir()
            if f.name != "artifact_manifest.json"
        },
    }
    p.long._json(root / "artifact_manifest.json", value)


def _verify(root):
    value = json.loads((root / "artifact_manifest.json").read_text())
    assert set(value["artifacts"]) == {f.name for f in root.iterdir()} - {
        "artifact_manifest.json"
    }
    for name, digest in value["artifacts"].items():
        assert p.long._sha256(root / name) == digest


@pytest.fixture(scope="module")
def parent_template(tmp_path_factory):
    """Tiny generated B-shaped parent; no historical scientific arrays accessed."""
    tmp_path = tmp_path_factory.mktemp("parent_template")
    parent = tmp_path / "B"
    parent.mkdir()
    source = tmp_path / "B_launch/source"
    source.mkdir(parents=True)
    sources = {}
    for name in p.SOURCE_PATHS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic archived source: " + name)
        sources[name] = p.long._sha256(path)
    protocol = replace(
        p.PROTOCOL,
        resolution=16,
        steps=8,
        anchor_steps=(0, 4, 6, 8),
        late_anchor=6,
        late_horizon=2,
        block_steps=2,
        macro_dt=0.004,
        high_mode=(4, 3),
        wall_seconds=60,
        fixture_label="unit_test_only",
    )
    cases = []
    for index, (seed, role) in enumerate(p.SEED_ROLES):
        if index < 6:
            partial_protocol = (
                replace(protocol, steps=4, anchor_steps=(0, 4))
                if index == 5
                else protocol
            )
            case = p._run_case(
                parent, partial_protocol, seed, role, p.perf_counter() + 60
            )
            assert case["status"] == "completed"
            if index == 5:
                case.update(
                    status="incomplete_budget",
                    error="synthetic B interruption",
                    seconds=123456.0,
                    expected_anchor_steps=None,
                )
                case["maximum_palinstrophy"]["complete_trajectory"] = False
                row = next(q for q in case["queries"] if q["anchor_step"] == 4)
                with np.load(parent / row["file"], allow_pickle=False) as data:
                    arrays = {
                        name: data[name].copy()
                        for name in ("input", "base_next", "half_next")
                    }
                p._save_arrays(parent, row, arrays)
                row["status"] = "incomplete_budget"
                del row["calls"]["fine_next"]
                for key in (
                    "temporal_relative_l2",
                    "spatial_relative_l2",
                    "fine_discarded_state_relative_l2",
                ):
                    row.pop(key)
                # Drop any completed extra peak query from this deliberately
                # stopped fixture; its prefix peak remains in the diagnostics.
                for extra in list(case["queries"]):
                    if extra["anchor_step"] not in (0, 4):
                        (parent / extra["file"]).unlink()
                        case["queries"].remove(extra)
                p.long._json(parent / f"case_{seed}.json", case)
        else:
            case = p._run_case(parent, protocol, seed, role, p.perf_counter() - 1)
        cases.append(case)
    identity = p.PARENT_RUN_ID + "__UNIT_FIXTURE"
    launch = {
        "run_id": identity,
        "sources": sources,
        "protocol": asdict(protocol),
        "seed_roles": [{"seed": s, "role": r} for s, r in p.SEED_ROLES],
    }
    p.long._json(parent / "launch.json", launch)
    result = {
        **launch,
        "status": "incomplete_budget",
        "engineering_gates_pass": None,
        "source_stable": True,
        "parent_evidence_stable": True,
        "sources_before": sources,
        "sources_after": sources,
        "cases": cases,
        "numeric_gates": p._numeric_gates(cases, protocol),
        "runtime": {"system": "FixtureOrigin", "numpy": np.__version__},
    }
    p.long._json(parent / "result.json", result)
    _manifest(parent, sources)
    audit = {
        "audit_status": "PASS",
        "population_verdict": "INCOMPLETE_BUDGET_NOT_QUALIFIED",
        "packet_manifest_sha256": p.long._sha256(parent / "artifact_manifest.json"),
        "result_sha256": p.long._sha256(parent / "result.json"),
    }
    p.long._json(source.parent / "independent_population_audit.json", audit)
    return parent, source, protocol


@pytest.fixture
def parent_fixture(parent_template, tmp_path):
    parent, source, protocol = parent_template
    shutil.copytree(parent, tmp_path / "B")
    shutil.copytree(source.parent, tmp_path / "B_launch")
    return tmp_path / "B", tmp_path / "B_launch/source", protocol


def test_contract_and_parent_binding(parent_fixture):
    parent, source, _ = parent_fixture
    assert (
        p.RUN_ID == "CM_NEXT_KF_POP_20260907C"
        and p.PARENT_RUN_ID == "CM_NEXT_KF_POP_20260906B"
    )
    assert p.PROTOCOL.resolution == 256 and p.PROTOCOL.steps == 512
    assert p.PROTOCOL.anchor_steps == (0, 16, 24, 64, 256, 512)
    assert p.PROTOCOL.wall_seconds == 28800 and p.MINIMUM_FREE_BYTES == 6 * 1024**3
    assert p.REPLAY_LIMIT == 1e-10 and p.LATE_SEEDS == (2026090617, 2026090621)
    assert len(p.SEED_ROLES) == 12 and len(p.SOURCE_PATHS) == 12
    assert p.GATE_LIMITS == {
        "clean_spatial_relative_l2": 1e-3,
        "fine_discarded_state_relative_l2": 1e-3,
        "continuation_endpoint_relative_l2": 2e-3,
    }
    evidence, result = p.validate_parent(parent, source, unit_fixture=True)
    assert (
        len(evidence["source_hashes"]) == 12 and result["status"] == "incomplete_budget"
    )
    with pytest.raises(ValueError, match="pinned"):
        p.validate_parent(parent, source)


@pytest.mark.parametrize("failure", ("payload", "source", "audit", "role"))
def test_parent_corruption_is_pre_output(parent_fixture, tmp_path, failure):
    parent, source, _ = parent_fixture
    if failure == "source":
        (source / p.SOURCE_PATHS[0]).write_text("corrupt")
    elif failure == "audit":
        (source.parent / "independent_population_audit.json").write_text("{}")
    else:
        path = parent / (
            "case_2026090616.json" if failure == "payload" else "result.json"
        )
        value = json.loads(path.read_text())
        if failure == "payload":
            value["error"] = "changed"
        else:
            value["cases"][0]["role"] = "development"
        p.long._json(path, value)
    output = tmp_path / "forbidden"
    with pytest.raises((ValueError, KeyError)):
        p.run_population(parent, source, output, unit_fixture=True)
    assert not output.exists()


def test_recovery_bytewise_inheritance_replay_resume_and_full_coverage(
    parent_fixture, tmp_path
):
    parent, source, protocol = parent_fixture
    parent_hashes = {f.name: p.long._sha256(f) for f in parent.iterdir()}
    prior = json.loads((parent / "result.json").read_text())
    prefix_progress = (parent / "progress_2026090616.jsonl").read_bytes()
    output = tmp_path / "C"
    result = p.run_population(parent, source, output, unit_fixture=True)
    assert result["status"] == "completed" and result["run_id"].endswith(
        "__UNIT_FIXTURE"
    )
    assert result["runtime_replay"]["status"] == "completed"
    assert len(result["runtime_replay"]["calls"]) == 14
    for row in result["runtime_replay"]["calls"]:
        assert row["status"] == "completed" and row["relative_l2"] <= p.REPLAY_LIMIT
        if row["repetition"]:
            assert row["same_process_bitwise"]
        with np.load(output / row["file"], allow_pickle=False) as data:
            assert set(data.files) == {"input", "reference", "output"}
    for old, new in zip(prior["cases"][:5], result["cases"][:5], strict=True):
        assert old == new
        names = [
            f"case_{old['seed']}.json",
            f"progress_{old['seed']}.jsonl",
            f"checkpoint_{old['seed']}.json",
        ] + [row["file"] for key in ("blocks", "queries") for row in old[key]]
        for name in names:
            assert p.long._sha256(output / name) == parent_hashes[name]
    resumed = result["cases"][5]
    assert resumed["last_retained_step"] == 8 and "error" not in resumed
    assert resumed["seconds"] != 123456 and result["seconds"] < 123456
    assert result["origin"]["parent_resume_error"] == "synthetic B interruption"
    assert result["origin"]["parent_case_seconds"]["2026090616"] == 123456
    assert (
        (output / "progress_2026090616.jsonl").read_bytes().startswith(prefix_progress)
    )
    for block in prior["cases"][5]["blocks"]:
        assert p.long._sha256(output / block["file"]) == parent_hashes[block["file"]]
    q0 = next(q for q in prior["cases"][5]["queries"] if q["anchor_step"] == 0)
    assert p.long._sha256(output / q0["file"]) == parent_hashes[q0["file"]]
    fixed = [q for q in resumed["queries"] if q["anchor_step"] == 4]
    assert len(fixed) == 1 and fixed[0]["status"] == "completed"
    with np.load(output / fixed[0]["file"], allow_pickle=False) as data:
        assert set(data.files) == {"input", "base_next", "half_next", "fine_next"}
    for case in result["cases"]:
        states = p._retained_states(output, case)
        assert sorted(states) == list(range(9))
        assert [row["step"] for row in case["trajectory_diagnostics"]] == list(range(9))
        peak = int(
            np.argmax(
                [d["structure"]["palinstrophy"] for d in case["trajectory_diagnostics"]]
            )
        )
        assert case["maximum_palinstrophy"]["step"] == peak
        assert case["expected_anchor_steps"] == sorted({0, 4, 6, 8, peak})
        for anchor in case["expected_continuation_anchors"]:
            coarse = states[anchor]
            fine = p.long.resize_dealiased_vorticity(coarse, 32)
            for row in [
                r for r in case["continuation_rows"] if r["anchor_step"] == anchor
            ]:
                with np.load(output / row["file"], allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["coarse_input"], coarse)
                    np.testing.assert_array_equal(data["fine_input"], fine)
                    coarse, fine = data["coarse_next"].copy(), data["fine_next"].copy()
    assert result["numeric_gates"] == p._numeric_gates(result["cases"], protocol)
    assert all(g["complete"] for g in result["numeric_gates"].values())
    assert parent_hashes == {f.name: p.long._sha256(f) for f in parent.iterdir()}
    _verify(output)
    with pytest.raises(FileExistsError):
        p.run_population(parent, source, output, unit_fixture=True)


@pytest.mark.parametrize("mismatch", ("answer", "repeat"))
def test_replay_failure_stops_generation_and_retains_answers(
    parent_fixture, tmp_path, monkeypatch, mismatch
):
    parent, source, _ = parent_fixture
    original = p._advance
    calls = 0

    def changed(*args):
        nonlocal calls
        values, timing = original(*args)
        calls += 1
        if mismatch == "answer" or calls == 2:
            values = values * (1.01 if mismatch == "answer" else (1 + 1e-12))
        return values, timing

    monkeypatch.setattr(p, "_advance", changed)
    output = tmp_path / "replay_failed"
    result = p.run_population(parent, source, output, unit_fixture=True)
    assert result["status"] == "replay_failed" and not result["generation_started"]
    assert result["cases"] == [] and result["engineering_gates_pass"] is None
    assert result["runtime_replay"]["calls"] and not list(output.glob("trajectory*"))
    assert result["runtime_replay"]["finished_utc"]
    _verify(output)


def test_budget_begins_before_parent_validation(parent_fixture, tmp_path, monkeypatch):
    parent, source, _ = parent_fixture
    elapsed = p.perf_counter()
    original = p.validate_parent

    def validated(*args, **kwargs):
        nonlocal elapsed
        answer = original(*args, **kwargs)
        elapsed += 61
        return answer

    monkeypatch.setattr(p, "perf_counter", lambda: elapsed)
    monkeypatch.setattr(p, "validate_parent", validated)
    result = p.run_population(parent, source, tmp_path / "budget", unit_fixture=True)
    assert (
        result["status"] == "incomplete_budget"
        and result["runtime_replay"]["calls"] == []
    )
    assert not result["generation_started"]
    assert not result["parent_state_arrays_decoded"]
    assert not result["generated_solver_states"]


def test_resume_budget_retains_prefix_and_terminal_status(
    parent_fixture, tmp_path, monkeypatch
):
    parent, source, _ = parent_fixture
    elapsed = p.perf_counter()
    original = p._runtime_replay

    def replay(*args, **kwargs):
        nonlocal elapsed
        answer = original(*args, **kwargs)
        elapsed += 61
        return answer

    monkeypatch.setattr(p, "perf_counter", lambda: elapsed)
    monkeypatch.setattr(p, "_runtime_replay", replay)
    output = tmp_path / "resume_budget"
    result = p.run_population(parent, source, output, unit_fixture=True)
    assert result["status"] == "incomplete_budget"
    assert result["cases"][5]["last_retained_step"] == 4
    assert [case["status"] for case in result["cases"]] == ["completed"] * 5 + [
        "incomplete_budget"
    ] * 7
    assert all(g["pass"] is None for g in result["numeric_gates"].values())
    _verify(output)


def test_resume_partial_query_timeout_keeps_new_runtime_partial_arrays(
    parent_fixture, tmp_path, monkeypatch
):
    parent, source, _ = parent_fixture
    original = p.peak._advance
    calls = 0

    def timeout_on_fine(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise p.long.BudgetExceeded("fixture fine query interrupted")
        return original(*args)

    monkeypatch.setattr(p.peak, "_advance", timeout_on_fine)
    output = tmp_path / "partial_query"
    result = p.run_population(parent, source, output, unit_fixture=True)
    case = result["cases"][5]
    assert result["status"] == "incomplete_budget"
    assert case["status"] == "incomplete_budget" and case["last_retained_step"] == 4
    query = next(row for row in case["queries"] if row["anchor_step"] == 4)
    assert query["status"] == "incomplete_budget"
    with np.load(output / query["file"], allow_pickle=False) as data:
        assert set(data.files) == {"input", "base_next", "half_next"}
    assert result["origin"]["parent_resume_error"] != case["error"]
    assert all(g["pass"] is None for g in result["numeric_gates"].values())
    _verify(output)


def test_live_source_drift_is_invalid_packet(parent_fixture, tmp_path, monkeypatch):
    parent, source, _ = parent_fixture
    original = p._sources
    calls = 0

    def drifted():
        nonlocal calls
        calls += 1
        hashes = original()
        if calls == 2:
            hashes[p.SOURCE_PATHS[0]] = "0" * 64
        return hashes

    monkeypatch.setattr(p, "_sources", drifted)
    result = p.run_population(parent, source, tmp_path / "drift", unit_fixture=True)
    assert result["status"] == "invalid_source" and not result["source_stable"]
    assert result["engineering_gates_pass"] is None


def test_disk_preflight_and_cli_failure(parent_fixture, tmp_path, monkeypatch):
    parent, source, _ = parent_fixture
    monkeypatch.setattr(
        p.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=p.MINIMUM_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="6 GiB"):
        p.run_population(parent, source, tmp_path / "disk", unit_fixture=True)
    assert not (tmp_path / "disk").exists()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "population",
            "--parent",
            "B",
            "--parent-source",
            "archive/source",
            "--output",
            "C",
        ],
    )
    monkeypatch.setattr(
        p,
        "run_population",
        lambda *args: {"status": "replay_failed", "engineering_gates_pass": None},
    )
    with pytest.raises(SystemExit) as error:
        p.main()
    assert error.value.code == 1
