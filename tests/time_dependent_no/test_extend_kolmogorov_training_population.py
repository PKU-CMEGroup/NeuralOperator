from __future__ import annotations

import copy
from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest

from scripts.time_dependent_no import extend_kolmogorov_training_population as p
from tests.time_dependent_no.test_fit_kolmogorov_clean import _seal
from tests.time_dependent_no.test_fit_kolmogorov_clean import population as population


@pytest.fixture
def replay_parent(population):
    parent, source, _, sources = population
    protocol = replace(
        p.PROTOCOL,
        resolution=16,
        steps=4,
        anchor_steps=(0, 1, 4),
        late_anchor=2,
        late_horizon=2,
        block_steps=2,
    )
    case = p.parent_code._run_case(
        parent, protocol, 2026090611, "train", perf_counter() + 60
    )
    assert case["status"] == "completed"
    result = p.clean._read(parent / "result.json")
    result["cases"][0] = case
    p.clean._write(parent / "result.json", result)
    _seal(parent, sources)
    return parent, source, sources


def test_fixed_disjoint_index_and_physics():
    rows = p.population_index(p.NEW_SEEDS)
    assert len(rows) == 36
    assert [r["role"] for r in rows] == ["train"] * 32 + ["development"] * 4
    assert [r["seed"] for r in rows[-4:]] == list(range(2026090621, 2026090625))
    assert all(r["packet"] == "parent_C" for r in rows[:8] + rows[-4:])
    assert all(r["packet"] == "this_packet" for r in rows[8:32])
    assert p.PROTOCOL.resolution == 256 and p.PROTOCOL.steps == 512
    assert p.PROTOCOL.macro_dt == 0.05 and p.PROTOCOL.wall_seconds == 8 * 3600
    assert p.CONTINUATION_SEEDS == (2026090801, 2026090802)
    for seeds in ((2026090621,), (2026090611,), (2026090801, 2026090801)):
        with pytest.raises(ValueError, match="disjoint"):
            p.population_index(seeds)


def test_tiny_expansion_retains_parent_and_full_query_evidence(replay_parent, tmp_path):
    parent, source, _ = replay_parent
    before = {f.name: p.clean._hash(f) for f in parent.iterdir()}
    output = tmp_path / "expanded"
    result = p.run(parent, source, output, unit_fixture=True)
    assert result["status"] == "completed"
    assert result["run_id"].endswith("__UNIT_FIXTURE")
    assert result["runtime_replay"]["status"] == "completed"
    assert len(result["runtime_replay"]["calls"]) == 6
    assert result["retained_new_training_transitions"] == 8
    assert len(result["population_index"]) == 14
    assert not result["protected_access"] and not result["model_evaluated"]
    assert result["optimization_steps"] == result["new_development_trajectories"] == 0
    assert result["source_stable"] and result["parent_evidence_stable"]
    assert before == {f.name: p.clean._hash(f) for f in parent.iterdir()}
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert set(manifest["artifacts"]) == {f.name for f in output.iterdir()} - {
        "artifact_manifest.json"
    }
    assert all(p.clean._hash(output / n) == h for n, h in manifest["artifacts"].items())
    for case in result["cases"]:
        assert case["role"] == "train"
        assert set(p.parent_code._retained_states(output, case)) == set(range(5))
        assert all(
            r["status"] == "completed"
            for r in case["queries"] + case["continuation_rows"]
        )
        for row in case["queries"] + case["continuation_rows"]:
            assert p.clean._hash(output / row["file"]) == row["sha256"]
    assert all(g["complete"] for g in result["numeric_gates"].values())
    assert (
        result["numeric_gates"]["continuation_endpoint_relative_l2"]["sample_count"]
        >= 2
    )
    protocol = p.parent_code.long.TrajectoryProtocol(**result["protocol"])
    for kind in ("missing", "duplicate", "bad_value", "incomplete", "wrong_role"):
        cases = copy.deepcopy(result["cases"])
        if kind == "missing":
            cases[0]["queries"].pop()
        elif kind == "duplicate":
            cases[0]["continuation_rows"].append(cases[0]["continuation_rows"][0])
        elif kind == "bad_value":
            cases[0]["queries"][0]["spatial_relative_l2"] = 1.0
        elif kind == "incomplete":
            cases[0]["status"] = "incomplete_budget"
        else:
            cases[0]["role"] = "development"
        gates = p.numerical_gates(cases, protocol, p.CONTINUATION_SEEDS)
        assert not all(g["pass"] is True for g in gates.values())
    with pytest.raises(FileExistsError):
        p.run(parent, source, output, unit_fixture=True)


def test_failed_replay_generates_no_new_paths(replay_parent, tmp_path, monkeypatch):
    parent, source, _ = replay_parent
    original = p.parent_code._advance

    def changed_answer(*args):
        answer, timing = original(*args)
        return answer * 1.01, timing

    monkeypatch.setattr(p.parent_code, "_advance", changed_answer)
    result = p.run(parent, source, tmp_path / "failed_replay", unit_fixture=True)
    assert result["status"] == "replay_failed"
    assert result["cases"] == [] and result["engineering_gates_pass"] is None
    assert result["retained_new_training_transitions"] == 0
    assert all(g["pass"] is None for g in result["numeric_gates"].values())


def test_real_identity_rejects_fixture_before_output(replay_parent, tmp_path):
    parent, source, _ = replay_parent
    output = tmp_path / "not_scientific"
    with pytest.raises(ValueError, match="pinned"):
        p.run(parent, source, output)
    assert not output.exists()


def test_no_empty_success_or_nonfinite_gate():
    gates = p.numerical_gates([], p.PROTOCOL, p.CONTINUATION_SEEDS)
    assert all(g["pass"] is None for g in gates.values())
    assert len(p.SOURCE_PATHS) == len(set(p.SOURCE_PATHS))
    assert np.isfinite(p.REPLAY_LIMIT) and p.REPLAY_LIMIT > 0
