from __future__ import annotations

import copy
import io
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import screen_kolmogorov_common_solver as screen
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovStep,
    KolmogorovStepDiagnostics,
)


def _grid(n):
    x = np.arange(n, dtype=np.float64) * (2 * np.pi / n)
    return np.meshgrid(x, x, indexing="ij")


def _fake_map(state):
    _, y = _grid(len(state))
    return np.ascontiguousarray(0.9 * state + 0.03 * np.cos(4 * y))


def _read(path):
    return json.loads(path.read_text())


def _refresh(fixture):
    """Rebind deliberately edited synthetic metadata; never touch real packets."""
    screen._json(
        fixture.population / "artifact_manifest.json",
        {
            "run_id": "CM_NEXT_KF_POP_20260907C",
            "source_stable": True,
            "artifacts": {
                p.name: screen._sha256(p)
                for p in fixture.population.iterdir()
                if p.name != "artifact_manifest.json"
            },
        },
    )
    fixture.result["input_evidence"]["parent"]["manifest_sha256"] = screen._sha256(
        fixture.population / "artifact_manifest.json"
    )
    for record in fixture.result["banks"] + fixture.result["cases"]:
        path = fixture.common / record["file"]
        if path.is_file():
            record["sha256"] = screen._sha256(path)
    screen._json(fixture.common / "result.json", fixture.result)
    screen._json(
        fixture.common / "artifact_manifest.json",
        {
            "run_id": fixture.result["run_id"],
            "sources": {},
            "artifacts": {
                p.name: screen._sha256(p)
                for p in fixture.common.iterdir()
                if p.name != "artifact_manifest.json"
            },
        },
    )


@pytest.fixture
def parent_fixture(tmp_path):
    """Two N16 training paths with C's step/donor metadata; no real artifacts."""
    common, population = tmp_path / "common", tmp_path / "population"
    common.mkdir()
    population.mkdir()
    config = screen.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.05
    )
    canonicalizer = screen.KolmogorovReferenceStepper(config)
    x, y = _grid(16)
    directions = (np.sqrt(2) * np.cos(x), np.sqrt(2) * np.sin(2 * y))
    amplitude = 0.05
    result = {
        "run_id": screen.COMMON_RUN_ID + "__UNIT_FIXTURE",
        "status": "completed",
        "steps": [1, 8, 32, 64],
        "solver_calls": 0,
        "model_forward_calls": 386,
        "source_stable": True,
        "input_evidence_stable": True,
        "calibration": {
            "validation_pairs": 0,
            "training_pairs": 16384,
            "physical_rms": amplitude,
        },
        "input_evidence": {"parent": {}},
        "train_scale": 2.0,
        "model_config": {"resolution": 16, "modes": 3},
        "banks": [],
        "cases": [],
    }
    for ordinal, seed in enumerate(screen.SEEDS):
        references = [
            canonicalizer.canonicalize(
                (1 + 0.1 * ordinal + 0.07 * anchor) * np.cos(x + y)
                + 0.3 * np.sin(2 * y)
            )
            for anchor in range(4)
        ]
        successors = [_fake_map(u) for u in references]
        bank = {
            "reference_input": np.asarray(references, dtype=np.float32),
            "reference_next": np.asarray(successors, dtype=np.float32),
        }
        probes, queries = [], []
        for anchor, step in enumerate(result["steps"]):
            u = bank["reference_input"][anchor]
            for donor, direction in zip(screen.MODELS, directions, strict=True):
                for view in ("natural",) if step == 1 else ("natural", "matched_rms"):
                    probe = (
                        u.copy()
                        if step == 1
                        else (u + amplitude * direction).astype(np.float32)
                    )
                    queries.append(
                        {
                            "query_index": len(probes),
                            "anchor": anchor,
                            "output_step": step,
                            "input_step": step - 1,
                            "donor": donor,
                            "view": view,
                            "input_sha256": screen.array_hash(probe),
                        }
                    )
                    probes.append(probe)
        bank["probe_input"] = np.stack(probes)
        bank_name = f"bank_{seed}.npz"
        np.savez_compressed(common / bank_name, **bank)
        result["banks"].append(
            {
                "seed": seed,
                "role": "train",
                "file": bank_name,
                "queries": queries,
                "array_hashes": {k: screen.array_hash(v) for k, v in bank.items()},
            }
        )
        for model, gain in zip(screen.MODELS, (1.15, 0.8), strict=True):
            predictions = {}
            for prefix, values in (
                ("clean", bank["reference_input"]),
                ("probe", bank["probe_input"]),
            ):
                raw = np.stack(
                    [
                        gain * u
                        + 0.03 * np.cos(4 * y)
                        + 0.02 * np.mean(u * np.cos(x)) * np.cos(7 * x)
                        for u in values
                    ]
                ).astype(np.float32)
                predictions[prefix + "_raw"] = raw
                predictions[prefix + "_next"] = np.stack(
                    [canonicalizer.canonicalize(u.astype(np.float64)) for u in raw]
                ).astype(np.float32)
            name = f"prediction_{model}_{seed}.npz"
            np.savez_compressed(common / name, **predictions)
            result["cases"].append(
                {
                    "seed": seed,
                    "recipient": model,
                    "role": "train",
                    "file": name,
                    "rows": [
                        {**q, "output_kind": kind}
                        for q in queries
                        for kind in ("raw", "next")
                    ],
                }
            )
        blocks = []
        for anchor, step in ((1, 8), (2, 32)):
            name = f"block_{seed}_{step}.npz"
            np.savez_compressed(
                population / name,
                steps=np.asarray([step - 1, step], dtype=np.int64),
                states=np.stack([references[anchor], successors[anchor]]),
            )
            blocks.append(
                {
                    "first_step": step - 1,
                    "last_step": step,
                    "file": name,
                    "sha256": screen._sha256(population / name),
                }
            )
        screen._json(
            population / f"case_{seed}.json",
            {
                "seed": seed,
                "role": "train",
                "status": "completed",
                "config": asdict(config),
                "blocks": blocks,
            },
        )
    # A selection bug must fail on this absent, explicitly synthetic validation file.
    result["banks"].append(
        {
            "seed": 2026090621,
            "role": "development",
            "file": "synthetic_validation_must_not_open.npz",
        }
    )
    fixture = SimpleNamespace(
        common=common,
        population=population,
        result=result,
        config=config,
        output=tmp_path / "output",
    )
    _refresh(fixture)
    return fixture


@pytest.fixture
def fake_stepper(monkeypatch):
    calls = []

    class FakeStepper:
        def __init__(self, config, deadline):
            self.config = config

        def advance_canonical(self, state):
            calls.append((self.config, state.copy()))
            return KolmogorovStep(
                _fake_map(state), KolmogorovStepDiagnostics(1, 0.05, 0.05, 0.0)
            )

    monkeypatch.setattr(screen, "BudgetedReferenceStepper", FakeStepper)
    return calls


def _load(fixture):
    bindings = {}
    data, scale, modes = screen.load_inputs(
        fixture.common,
        fixture.population,
        bindings,
        float("inf"),
        {},
        unit_fixture=True,
    )
    return data, scale, modes, bindings


def _run(fixture, phase="assay"):
    return screen.run(
        fixture.common,
        fixture.population,
        fixture.output,
        phase=phase,
        unit_fixture=True,
    )


def _packet(output, record):
    assert _read(output / "result.json") == record
    manifest = _read(output / "artifact_manifest.json")
    assert (
        manifest["status"] == record["status"]
        and manifest["run_id"] == record["run_id"]
    )
    assert set(manifest["artifacts"]) == {p.name for p in output.iterdir()} - {
        "artifact_manifest.json"
    }
    assert all(
        screen._sha256(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )


def test_fixed_refinement_contract_and_train_mapping(parent_fixture, monkeypatch):
    original_load = np.load
    captured = []

    def bytes_only(source, **kwargs):
        assert isinstance(source, io.BytesIO) and kwargs["allow_pickle"] is False
        captured.append(source)
        return original_load(source, **kwargs)

    monkeypatch.setattr(screen.np, "load", bytes_only)
    selected, scale, modes, bindings = _load(parent_fixture)
    assert [(a["seed"], a["output_step"]) for a in selected] == [
        (seed, step) for seed in (2026090611, 2026090612) for step in (8, 32)
    ]
    assert scale == 2.0 and modes == 3 and captured
    assert all(
        "2026090621" not in name and "validation" not in name for _, name in bindings
    )
    for anchor in selected:
        assert [q["donor"] for q in anchor["queries"]] == ["clean8", "clean32"]
        assert all(
            q["input_step"] == anchor["output_step"] - 1 for q in anchor["queries"]
        )
        assert anchor["raw_inputs"].dtype == np.float32
        assert anchor["reference_input64"].dtype == np.float64
    configs = screen.numerical_configs(parent_fixture.config)
    assert [(c.resolution, c.dt_max, c.cfl) for c in configs.values()] == [
        (16, 0.002, 0.4),
        (16, 0.001, 0.2),
        (32, 0.001, 0.2),
        (32, 0.0005, 0.1),
    ]
    assert all(c.macro_dt == 0.05 and c.viscosity == 0.01 for c in configs.values())
    assert screen.WALL_SECONDS == 5400


@pytest.mark.parametrize(
    "change",
    [
        "bank_role",
        "duplicate_bank",
        "recipient_role",
        "query_time",
        "duplicate_donor",
        "recipient_pairing",
        "population_role",
        "population_recipe",
        "array_hash",
        "query_hash",
    ],
)
def test_mismatched_selection_and_pairing_rejected(parent_fixture, change):
    f = parent_fixture
    bank = f.result["banks"][0]
    query = next(
        q
        for q in bank["queries"]
        if q["output_step"] == 8 and q["view"] == "matched_rms"
    )
    if change == "bank_role":
        bank["role"] = "development"
    elif change == "duplicate_bank":
        f.result["banks"].append(copy.deepcopy(bank))
    elif change == "recipient_role":
        f.result["cases"][0]["role"] = "development"
    elif change == "query_time":
        query["input_step"] = 8
    elif change == "duplicate_donor":
        next(
            q
            for q in bank["queries"]
            if q["output_step"] == 8
            and q["donor"] == "clean32"
            and q["view"] == "matched_rms"
        )["donor"] = "clean8"
    elif change == "recipient_pairing":
        next(
            r
            for r in f.result["cases"][0]["rows"]
            if r["query_index"] == query["query_index"]
        )["input_step"] = 6
    elif change in ("population_role", "population_recipe"):
        path = f.population / f"case_{screen.SEEDS[0]}.json"
        row = _read(path)
        if change == "population_role":
            row["role"] = "development"
        else:
            row["config"]["cfl"] = 0.3
        screen._json(path, row)
    elif change == "array_hash":
        bank["array_hashes"]["reference_input"] = "wrong"
    else:
        query["input_sha256"] = "wrong"
    _refresh(f)
    with pytest.raises(ValueError):
        _load(f)


@pytest.mark.parametrize("kind", ["bank", "prediction", "states", "steps"])
def test_wrong_array_dtype_rejected(parent_fixture, kind):
    f = parent_fixture
    if kind == "bank":
        path, field, dtype = (
            f.common / f.result["banks"][0]["file"],
            "reference_input",
            np.float64,
        )
    elif kind == "prediction":
        path, field, dtype = (
            f.common / f.result["cases"][0]["file"],
            "clean_raw",
            np.float64,
        )
    else:
        path = f.population / f"block_{screen.SEEDS[0]}_8.npz"
        field, dtype = (
            ("states", np.float32) if kind == "states" else ("steps", np.float64)
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays[field] = arrays[field].astype(dtype)
    np.savez_compressed(path, **arrays)
    if kind in ("states", "steps"):
        case_path = f.population / f"case_{screen.SEEDS[0]}.json"
        case = _read(case_path)
        case["blocks"][0]["sha256"] = screen._sha256(path)
        screen._json(case_path, case)
    _refresh(f)
    with pytest.raises(ValueError, match="dtype"):
        _load(f)


def test_byte_capture_rejects_changed_parent_and_path_escape(parent_fixture, tmp_path):
    f = parent_fixture
    bank = f.common / f.result["banks"][0]["file"]
    bank.write_bytes(bank.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load(f)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="inside"):
        screen.bound_bytes(
            f.common,
            "../outside.npz",
            screen._sha256(outside),
            {},
            "common",
            float("inf"),
        )


def test_validate_packet_is_solver_free_and_preserves_parent_and_source(
    parent_fixture, monkeypatch
):
    f = parent_fixture
    paths = [p for root in (f.common, f.population) for p in root.iterdir()]
    before = {p: screen._sha256(p) for p in paths}
    monkeypatch.setattr(
        screen,
        "BudgetedReferenceStepper",
        lambda *args: pytest.fail("solver constructed"),
    )
    result = _run(f, "validate")
    assert result["status"] == "completed" and result["qualification_passed"] is None
    assert result["solver_call_attempts"] == result["solver_calls_completed"] == 0
    assert result["expected_metric_rows"] == 0 and len(result["anchors"]) == 4
    assert result["inputs_stable"] and result["source_stable"]
    assert result["sources_before"] == result["sources_after"]
    assert {p: screen._sha256(p) for p in paths} == before
    for anchor in result["anchors"]:
        with np.load(f.output / anchor["file"], allow_pickle=False) as arrays:
            assert (
                arrays["raw_inputs"].dtype
                == arrays["reference_next32"].dtype
                == np.float32
            )
            assert (
                arrays["canonical_inputs"].dtype
                == arrays["lifted_inputs"].dtype
                == np.float64
            )
            assert arrays["lifted_inputs"].shape == (3, 32, 32)
            for name in arrays.files:
                assert screen.array_hash(arrays[name]) == anchor["array_hashes"][name]
            assert not np.array_equal(
                arrays["raw_inputs"].astype(np.float64), arrays["canonical_inputs"]
            )
    _packet(f.output, result)


def test_full_assay_shares_clean_solves_and_retains_all_53_calls_128_rows(
    parent_fixture, fake_stepper
):
    f = parent_fixture
    before = {
        p: screen._sha256(p)
        for root in (f.common, f.population)
        for p in root.iterdir()
    }
    result = _run(f)
    assert result["status"] == "completed" and result["qualification_passed"] is True
    assert (
        result["solver_call_attempts"]
        == result["solver_calls_completed"]
        == len(fake_stepper)
        == 53
    )
    assert sum(len(a["metrics"]) for a in result["anchors"]) == 128
    assert [len(a["calls"]) for a in result["anchors"]] == [14, 13, 13, 13]
    assert result["anchors"][0]["repeat_exact"] is True
    assert (
        result["model_calls"]
        == result["checkpoint_loads"]
        == result["optimization_steps"]
        == 0
    )
    assert all(
        result[k] is False
        for k in (
            "validation_arrays_read",
            "protected_access",
            "new_rollouts",
            "geometry_fitted",
            "corrections_added",
        )
    )
    for anchor in result["anchors"]:
        assert anchor["native_replay_relative_l2"] == 0
        assert len(anchor["numeric_rows"]) == 2 and anchor["qualified"]
        assert {
            (r["donor"], r["recipient"], r["output_kind"], r["reference_level"])
            for r in anchor["metrics"]
        } == {
            (d, m, k, level)
            for d in screen.MODELS
            for m in screen.MODELS
            for k in ("raw", "next")
            for level in screen.LEVELS
        }
        with np.load(f.output / anchor["file"], allow_pickle=False) as arrays:
            for call in anchor["calls"]:
                name = (
                    "native_replay"
                    if call["label"] == "native_fp64_replay"
                    else call["label"]
                )
                assert screen.array_hash(arrays[name]) == call["output_sha256"]
                np.testing.assert_array_equal(
                    arrays[name], _fake_map(fake_stepper.pop(0)[1])
                )
    assert {p: screen._sha256(p) for p in before} == before
    assert (
        result["sources_before"] == result["sources_after"] and result["inputs_stable"]
    )
    _packet(f.output, result)


def test_metrics_use_same_map_clean_endpoint_and_keep_raw_next_attribution(
    parent_fixture,
):
    data, scale, modes, _ = _load(parent_fixture)
    anchor = data[0]
    canonical, lifted, geometry = screen.prepare_anchor(anchor)
    outputs = {
        level: np.stack(
            [_fake_map(v) for v in (canonical if level in ("A", "B") else lifted)]
        )
        for level in screen.LEVELS
    }
    numeric, baseline = screen.measure_anchor(anchor, outputs, geometry, scale, modes)
    changed = copy.deepcopy(anchor)
    changed["reference_next32"] = changed["reference_next32"] + 1000
    numeric_changed, metrics_changed = screen.measure_anchor(
        changed, outputs, geometry, scale, modes
    )
    assert metrics_changed == baseline
    assert (
        numeric_changed[0]["archived_clean_successor_difference_scaled_rms"]["D"] > 100
    )
    assert numeric[0]["archived_clean_successor_difference_scaled_rms"]["D"] < 1e-5
    raw = [
        r
        for r in baseline
        if r["donor"] == "clean8"
        and r["recipient"] == "clean8"
        and r["reference_level"] == "D"
    ]
    assert (
        len(raw) == 2
        and raw[0]["output_kind"] == "raw"
        and raw[1]["output_kind"] == "next"
    )
    assert (
        raw[0]["fourier_band_scaled_energy"]["response_defect"][2]
        > raw[1]["fourier_band_scaled_energy"]["response_defect"][2]
    )
    assert all(r["recovery_closure_scaled_rms"] < 1e-12 for r in baseline)


def test_projection_vector_gate_and_collapsed_direction(parent_fixture):
    data, *_ = _load(parent_fixture)
    anchor = data[0]
    _, _, geometry = screen.prepare_anchor(anchor)
    assert screen.qualification_checks(geometry, include_solver=False)
    geometry[0]["projected_direction_change"] = (
        2 * screen.LIMITS["projected_direction_change"]
    )
    assert not screen.qualification_checks(geometry, include_solver=False)
    collapsed = copy.deepcopy(anchor)
    collapsed["raw_inputs"] = np.zeros((3, 16, 16), dtype=np.float32)
    collapsed["raw_inputs"][1:] = 0.05
    canonical, lifted, geometry = screen.prepare_anchor(collapsed)
    assert not np.any(canonical) and not np.any(lifted)
    assert all(not g["direction_resolved"] for g in geometry)
    assert not screen.qualification_checks(geometry, include_solver=False)
    assert not screen.qualification_checks([], include_solver=True)


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0, 1.01e-3])
def test_response_gates_reject_unresolved_or_above_limit_values(bad):
    row = {key: 0.0 for key in screen.LIMITS}
    row["direction_resolved"] = True
    row["coarse_temporal_response_over_displacement"] = bad
    assert not screen.qualification_checks([row], include_solver=True)


def test_refinement_sensitivity_is_separate_from_absolute_gates(parent_fixture):
    data, scale, modes, _ = _load(parent_fixture)
    anchor = data[0]
    canonical, lifted, geometry = screen.prepare_anchor(anchor)
    outputs = {}
    coefficients = {"A": 1.0, "B": 1.0002, "C": 1.0004, "D": 1.0006}
    for level, gain in coefficients.items():
        values = canonical if level in ("A", "B") else lifted
        outputs[level] = gain * values
    for model, gain in (("clean8", 1.00065), ("clean32", 1.0007)):
        anchor["predictions"][model] = {
            kind: gain * canonical for kind in ("raw", "next")
        }
    numeric, _ = screen.measure_anchor(anchor, outputs, geometry, scale, modes)
    for row in numeric:
        row["native_replay_relative_l2"] = 0.0
        assert row[
            "response_refinement_sensitivity_over_displacement"
        ] == pytest.approx(0.0006, rel=1e-5)
        assert row["refinement_resolution"]["raw"]["indicator_not_error_bound"] is True
        assert not any(
            row["refinement_resolution"]["raw"][
                "response_defect_exceeds_indicator"
            ].values()
        )
        assert (
            row["refinement_resolution"]["raw"]["recipient_gap_exceeds_twice_indicator"]
            is False
        )
    assert screen.qualification_checks(numeric, include_solver=True)


@pytest.mark.parametrize(
    "stage,completed",
    [
        ("source hashing", 0),
        ("array decoding completion", 0),
        ("input validation completion", 0),
        ("anchor preparation completion", 0),
        ("input qualification", 0),
        ("stepper construction", 0),
        ("counted solver invocation", 0),
        ("native replay comparison", 1),
        ("solver answer serialization completion", 2),
        ("repeat comparison", 14),
        ("response metrics", 14),
        ("work completion", 53),
    ],
)
def test_deadline_stages_retain_honest_partial_accounting(
    parent_fixture, fake_stepper, monkeypatch, stage, completed
):
    original = screen.check_deadline

    def expired(deadline, current):
        if current == stage:
            raise screen.BudgetExceeded("synthetic timeout at " + stage)
        original(deadline, current)

    monkeypatch.setattr(screen, "check_deadline", expired)
    result = _run(parent_fixture)
    assert (
        result["status"] == "incomplete_budget"
        and result["qualification_passed"] is None
    )
    assert result["solver_calls_completed"] == len(fake_stepper) == completed
    assert result["solver_call_attempts"] == completed
    assert stage in result["error"]
    _packet(parent_fixture.output, result)


@pytest.mark.parametrize("error", [screen.BudgetExceeded, ArithmeticError])
def test_failed_solver_call_is_attempted_but_not_completed(
    parent_fixture, fake_stepper, monkeypatch, error
):
    original = screen.BudgetedReferenceStepper.advance_canonical

    def fail_third(self, state):
        if len(fake_stepper) == 2:
            raise error("synthetic third-call failure")
        return original(self, state)

    monkeypatch.setattr(
        screen.BudgetedReferenceStepper, "advance_canonical", fail_third
    )
    result = _run(parent_fixture)
    assert result["solver_call_attempts"] == 3 and result["solver_calls_completed"] == 2
    assert result["status"] == (
        "incomplete_budget" if error is screen.BudgetExceeded else "failed"
    )
    assert result["qualification_passed"] is None
    _packet(parent_fixture.output, result)


@pytest.mark.parametrize("change", ["parent", "source"])
def test_post_capture_provenance_change_invalidates_completion(
    parent_fixture, fake_stepper, monkeypatch, change
):
    f = parent_fixture
    if change == "parent":
        original = screen.BudgetedReferenceStepper.advance_canonical

        def changed(self, state):
            if not fake_stepper:
                path = f.population / f"case_{screen.SEEDS[0]}.json"
                path.write_bytes(path.read_bytes() + b" ")
            return original(self, state)

        monkeypatch.setattr(
            screen.BudgetedReferenceStepper, "advance_canonical", changed
        )
    else:
        original, calls = screen._sha256, []
        target = screen.REPO_ROOT / screen.SOURCE_PATHS[0]

        def changed(path):
            if Path(path) == target:
                calls.append(path)
                if len(calls) > 1:
                    return "synthetic changed source digest"
            return original(path)

        monkeypatch.setattr(screen, "_sha256", changed)
    result = _run(f)
    assert (
        result["status"] == "invalid_provenance"
        and result["qualification_passed"] is None
    )
    assert result["solver_calls_completed"] == 53
    assert result["inputs_stable" if change == "parent" else "source_stable"] is False
    _packet(f.output, result)


def test_overlap_overwrite_and_fixture_cli_are_rejected(parent_fixture, monkeypatch):
    f = parent_fixture
    for output in (f.common, f.common / "nested", f.common.parent):
        with pytest.raises(ValueError, match="overlaps"):
            screen.run(
                f.common, f.population, output, phase="validate", unit_fixture=True
            )
    f.output.mkdir()
    sentinel = f.output / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(FileExistsError):
        _run(f, "validate")
    assert sentinel.read_text() == "existing output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "screen",
            "--common",
            str(f.common),
            "--population",
            str(f.population),
            "--output",
            str(f.output),
            "--phase",
            "validate",
            "--unit-fixture",
        ],
    )
    with pytest.raises(SystemExit) as error:
        screen.main()
    assert error.value.code == 2


def test_fixture_cannot_impersonate_scientific_parent(parent_fixture):
    f = parent_fixture
    result = screen.run(f.common, f.population, f.output, phase="validate")
    assert result["status"] == "failed" and result["solver_call_attempts"] == 0
    assert "hash mismatch" in result["error"]
    _packet(f.output, result)


@pytest.mark.parametrize("factor", [0.0, 0.5])
def test_collapsed_or_rescaled_bank_direction_is_rejected(parent_fixture, factor):
    f = parent_fixture
    bank = f.result["banks"][0]
    query = next(
        q
        for q in bank["queries"]
        if q["output_step"] == 8 and q["view"] == "matched_rms"
    )
    path = f.common / bank["file"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    u = arrays["reference_input"][query["anchor"]]
    j = query["query_index"]
    arrays["probe_input"][j] = u + factor * (arrays["probe_input"][j] - u)
    query["input_sha256"] = screen.array_hash(arrays["probe_input"][j])
    for case in f.result["cases"]:
        if case["seed"] == bank["seed"]:
            for row in case["rows"]:
                if row["query_index"] == j:
                    row.update(query)
    bank["array_hashes"] = {k: screen.array_hash(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays)
    _refresh(f)
    with pytest.raises(ValueError, match="collapsed or changed scale"):
        _load(f)


@pytest.mark.parametrize("failed_call", [1, 14])
def test_native_replay_or_repeat_failure_stops_later_solves(
    parent_fixture, fake_stepper, monkeypatch, failed_call
):
    original = screen.BudgetedReferenceStepper.advance_canonical

    def wrong(self, state):
        answer = original(self, state)
        if len(fake_stepper) == failed_call:
            return KolmogorovStep(answer.state + 1e-4, answer.diagnostics)
        return answer

    monkeypatch.setattr(screen.BudgetedReferenceStepper, "advance_canonical", wrong)
    result = _run(parent_fixture)
    assert result["status"] == "failed" and result["qualification_passed"] is None
    assert (
        result["solver_call_attempts"]
        == result["solver_calls_completed"]
        == failed_call
    )
    assert len(result["anchors"]) == 1 and result["anchors"][0]["metrics"] == []
    _packet(parent_fixture.output, result)


def test_completed_negative_qualification_is_distinct_from_incomplete_work(
    parent_fixture, fake_stepper, monkeypatch
):
    original = screen.BudgetedReferenceStepper.advance_canonical

    def divergent_fine(self, state):
        answer = original(self, state)
        if self.config.resolution == 32 and self.config.cfl == 0.1:
            return KolmogorovStep(answer.state * 1.1, answer.diagnostics)
        return answer

    monkeypatch.setattr(
        screen.BudgetedReferenceStepper, "advance_canonical", divergent_fine
    )
    result = _run(parent_fixture)
    assert result["status"] == "completed" and result["qualification_passed"] is False
    assert result["solver_calls_completed"] == 53
    assert sum(len(row["metrics"]) for row in result["anchors"]) == 128
    assert all(not row["qualified"] for row in result["anchors"])
    _packet(parent_fixture.output, result)


def test_real_low_resolution_solver_is_finite_canonical_and_repeatable():
    config = screen.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.05
    )
    stepper = screen.BudgetedReferenceStepper(config, float("inf"))
    x, y = _grid(16)
    state = stepper.canonicalize(np.cos(x + y) + 0.1 * np.sin(2 * y))
    first = stepper.advance_canonical(state)
    second = stepper.advance_canonical(state.copy())
    np.testing.assert_array_equal(first.state, second.state)
    assert np.isfinite(first.state).all() and first.state.dtype == np.float64
    assert abs(first.state.mean()) < 1e-12 and not np.array_equal(first.state, state)
    assert first.diagnostics.substeps > 1
