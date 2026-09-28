from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.time_dependent_no import generate_kolmogorov_paired_bank as bank
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovStep,
    KolmogorovStepDiagnostics,
)

core = bank.core


def _grid(n):
    x = np.arange(n, dtype=np.float64) * (2 * np.pi / n)
    return np.meshgrid(x, x, indexing="ij")


def _map(state):
    _, y = _grid(len(state))
    return np.ascontiguousarray(0.9 * state + 0.03 * np.cos(4 * y))


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(fixture, mutation=None):
    """Hash only synthetic files and bind their four immutable-schema parents."""
    hashes = {}
    for role, root in fixture.roots.items():
        result = copy.deepcopy(fixture.results[role])
        if role == "extension":
            result["parent_manifest_sha256"] = hashes["population"]
        elif role == "common":
            result["input_evidence"]["parent"]["manifest_sha256"] = hashes["population"]
        elif role == "qualification":
            result["input_files"] = {
                key + "/artifact_manifest.json": hashes[key]
                for key in ("population", "common")
            }
        if mutation and mutation[0] == role:
            target = result
            for key in mutation[1][:-1]:
                target = target[key]
            target[mutation[1][-1]] = mutation[2]
        core._json(root / "result.json", result)
        manifest = {
            "run_id": result["run_id"],
            "source_stable": True,
            "sources": {},
            "artifacts": {
                path.name: core._sha256(path)
                for path in sorted(root.iterdir())
                if path.name != "artifact_manifest.json"
            },
        }
        # The frozen population writer does not emit manifest.status.
        if role != "population":
            manifest["status"] = "completed"
        if role == "extension":
            manifest["parent_manifest_sha256"] = hashes["population"]
        core._json(root / "artifact_manifest.json", manifest)
        hashes[role] = core._sha256(root / "artifact_manifest.json")


@pytest.fixture
def parents(tmp_path, monkeypatch):
    roots = {role: tmp_path / role for role in bank.PINS}
    for root in roots.values():
        root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "synthetic_source.py").write_bytes(b"# Synthetic source binding only.\n")
    monkeypatch.setattr(bank, "REPO_ROOT", source)
    monkeypatch.setattr(bank, "SOURCE_PATHS", ("synthetic_source.py",))
    config = core.KolmogorovReferenceConfig(
        resolution=16, viscosity=0.01, macro_dt=0.05
    )
    x, y = _grid(16)
    results = {
        role: {
            "run_id": pin[0] + "__UNIT_FIXTURE",
            "status": "completed",
            "source_stable": True,
        }
        for role, pin in bank.PINS.items()
    }
    results["extension"].update(
        engineering_gates_pass=True,
        population_index=[
            {
                "seed": seed,
                "role": "train",
                "packet": packet,
                "case_file": f"case_{seed}.json",
            }
            for seed, packet in ((2026090611, "parent_C"), (2026090801, "this_packet"))
        ],
    )
    results["common"].update(
        input_evidence={"parent": {}},
        input_evidence_stable=True,
        calibration={
            "validation_pairs": 0,
            "training_pairs": 16384,
            "physical_rms": 0.05,
        },
        model_config={"resolution": 16},
        train_scale=2.0,
    )
    results["qualification"].update(
        qualification_passed=True, inputs_stable=True, solver_calls_completed=53
    )
    for ordinal, (role, seed) in enumerate(
        (("population", 2026090611), ("extension", 2026090801))
    ):
        state = core.KolmogorovReferenceStepper(config).canonicalize(
            (1 + ordinal * 0.1) * np.cos(x + y) + 0.2 * np.sin(2 * y)
        )
        states = [state]
        for _ in range(3):
            states.append(_map(states[-1]))
        name = f"block_{seed}.npz"
        np.savez_compressed(
            roots[role] / name,
            steps=np.arange(1, 5, dtype=np.int64),
            states=np.stack(states),
        )
        core._json(
            roots[role] / f"case_{seed}.json",
            {
                "seed": seed,
                "role": "train",
                "status": "completed",
                "config": asdict(config),
                "blocks": [
                    {
                        "first_step": 1,
                        "last_step": 4,
                        "file": name,
                        "sha256": core._sha256(roots[role] / name),
                    }
                ],
            },
        )
        (roots[role] / "validation_do_not_read.npz").write_bytes(
            b"Synthetic forbidden validation sentinel."
        )
    fixture = SimpleNamespace(
        roots=roots,
        source=source,
        results=results,
        config=config,
        calls=[],
        fail_at=None,
        hook=None,
        gain_by_level=False,
    )
    _seal(fixture)

    class FakeStepper:
        def __init__(self, numerical_config, deadline):
            self.config = numerical_config
            self.deadline = deadline

        def advance_canonical(self, value):
            fixture.calls.append((self.config, value.copy(), self.deadline))
            if fixture.fail_at == len(fixture.calls):
                raise RuntimeError("synthetic solver failure")
            answer = _map(value)
            if fixture.gain_by_level and self.config.dt_max < config.dt_max:
                answer = answer + 0.05 * value
            if fixture.hook:
                answer = fixture.hook(len(fixture.calls), answer)
            return KolmogorovStep(
                answer, KolmogorovStepDiagnostics(1000, 0.00005, 0.00005, 0.0)
            )

    monkeypatch.setattr(core, "BudgetedReferenceStepper", FakeStepper)
    return fixture


def _run(fixture, output, phase="generate", **kwargs):
    return bank.run(
        *(fixture.roots[role] for role in bank.PINS),
        output,
        phase=phase,
        unit_fixture=True,
        **kwargs,
    )


def _rewrite_case(fixture, mutate, *, rehash_blocks=True):
    root = fixture.roots["population"]
    path = root / "case_2026090611.json"
    case = _read(path)
    if rehash_blocks:
        for block in case["blocks"]:
            block["sha256"] = core._sha256(root / block["file"])
    mutate(case)
    core._json(path, case)
    _seal(fixture)


def _assert_packet(output, result):
    assert _read(output / "result.json") == result
    manifest = _read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert set(manifest["artifacts"]) == {p.name for p in output.iterdir()} - {
        "artifact_manifest.json"
    }
    for name, digest in manifest["artifacts"].items():
        assert core._sha256(output / name) == digest
    for row in result["anchors"]:
        assert row["sha256"] == manifest["artifacts"][row["file"]]
        with np.load(output / row["file"], allow_pickle=False) as saved:
            assert row["array_hashes"] == {
                name: core.array_hash(saved[name]) for name in saved.files
            }


def test_registered_selection_and_exact_call_accounting():
    old, added, steps, sentinel_steps = bank.selection()
    assert old == tuple(range(2026090611, 2026090619))
    assert added == tuple(range(2026090801, 2026090825))
    assert steps == tuple(range(15, 496, 32))
    assert sentinel_steps == (15, 239, 495)
    assert bank.SENTINEL_SEEDS == (2026090611, 2026090612, 2026090801, 2026090802)
    assert bank.expected_calls() == 1536 + 108 + 12 + 1 == 1657
    assert bank.expected_calls(unit_fixture=True) == 12 + 18 + 2 + 1 == 33


@pytest.mark.parametrize("n,rank", [(16, 120), (24, 288), (256, 29240)])
def test_gaussian_dimension_exact_seed_scaling_and_pairing(n, rank):
    config = core.KolmogorovReferenceConfig(resolution=n)
    x, y = _grid(n)
    reference = core.KolmogorovReferenceStepper(config).canonicalize(np.cos(x + y))
    sigma, seed, step = 0.05, 2026090611, 15
    raw, canonical, lifted, geometry, noise = bank.prepare_inputs(
        reference, config, sigma, seed, step
    )
    again = bank.prepare_inputs(reference, config, sigma, seed, step)
    for left, right in zip((raw, canonical, lifted), again[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    assert geometry == again[3] and noise == again[4]
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([2026091101, seed, step]))
    )
    eta = core.KolmogorovReferenceStepper(config).canonicalize(
        rng.standard_normal((n, n))
    ) * (sigma * n / np.sqrt(rank))
    assert noise["retained_real_dimension"] == rank
    assert noise["intended_sha256"] == core.array_hash(eta)
    assert noise["seed_sequence"] == [2026091101, seed, step]
    assert noise["expected_rms"] == sigma
    assert raw.dtype == np.float32 and canonical.dtype == lifted.dtype == np.float64
    np.testing.assert_array_equal(raw[0], reference.astype(np.float32))
    for j, sign in ((1, 1), (2, -1)):
        np.testing.assert_array_equal(
            raw[j], (raw[0].astype(np.float64) + sign * eta).astype(np.float32)
        )
        assert geometry[j - 1]["sign"] == sign
        assert (
            geometry[j - 1]["rounding_over_intended_displacement"] < bank.ROUNDING_LIMIT
        )
        np.testing.assert_allclose(
            core.resize_dealiased_vorticity(lifted[j], n), canonical[j], atol=2e-15
        )
    assert core.qualification_checks(geometry, include_solver=False)
    assert noise["antithetic_residual_over_expected_rms"] < 1e-5


def test_gaussian_ensemble_scale_not_per_draw_normalized():
    config = core.KolmogorovReferenceConfig(resolution=16)
    reference = np.zeros((16, 16), dtype=np.float64)
    sigma = 0.05
    sizes = [
        bank.prepare_inputs(reference, config, sigma, 2026090611, step)[3][0][
            "intended_displacement_rms"
        ]
        for step in range(192)
    ]
    assert np.mean(np.square(sizes)) / sigma**2 == pytest.approx(1.0, abs=0.035)
    assert np.std(sizes) / np.mean(sizes) > 0.04
    assert len(set(sizes)) == len(sizes)
    a = bank.prepare_inputs(reference, config, sigma, 2026090611, 1)[4]
    b = bank.prepare_inputs(reference, config, sigma, 2026090801, 1)[4]
    assert a["intended_sha256"] != b["intended_sha256"]


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_gaussian_scale_rejected(sigma):
    with pytest.raises(ValueError, match="Gaussian scale"):
        bank.prepare_inputs(
            np.zeros((16, 16)),
            core.KolmogorovReferenceConfig(resolution=16),
            sigma,
            1,
            1,
        )


def test_full_fixture_same_a_labels_clean_reuse_and_source_input_nonmutation(
    parents, tmp_path, monkeypatch
):
    paths = [
        p for root in (*parents.roots.values(), parents.source) for p in root.iterdir()
    ]
    before = {p: core._sha256(p) for p in paths}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert not path.name.startswith("validation_"), "validation file was accessed"
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        result = _run(parents, tmp_path / "generated")
    assert result["status"] == "completed" and result["qualification_passed"] is True
    assert result["expected_anchors"] == len(result["anchors"]) == 4
    assert result["expected_displaced_rows"] == 8
    assert result["expected_sentinels"] == 2
    assert (
        result["solver_call_attempts"]
        == result["solver_calls_completed"]
        == len(parents.calls)
        == 33
    )
    assert [len(row["calls"]) for row in result["anchors"]] == [14, 3, 13, 3]
    assert [
        (r["seed"], r["input_step"], r["output_step"], r["role"])
        for r in result["anchors"]
    ] == [(s, t, t + 1, "train") for s in (2026090611, 2026090801) for t in (1, 3)]
    assert (
        result["source_stable"]
        and result["inputs_stable"]
        and result["inputs_validated"]
    )
    assert len(result["input_files"]) == 12 and all(result["input_checks"].values())
    assert not any("validation" in name for name in result["input_files"])
    assert all(
        result[key] == 0
        for key in ("model_calls", "checkpoint_loads", "optimization_steps")
    )
    assert all(
        result[key] is False
        for key in ("validation_arrays_read", "protected_access", "new_rollouts")
    )
    cursor = 0
    for index, row in enumerate(result["anchors"]):
        labels = (["native_replay"] if row["sentinel"] else []) + [
            f"A_{j}" for j in range(3)
        ]
        if row["sentinel"]:
            labels += [f"{level}_{j}" for level in ("B", "C", "D") for j in range(3)]
        if index == 0:
            labels += ["repeat_A_clean"]
        assert [call["label"] for call in row["calls"]] == labels
        assert row["training_columns"] == {
            "clean_input": "raw_inputs[0]",
            "recovery_target": "A_0",
            "signed_rows": [
                {"sign": sign, "input": f"raw_inputs[{j}]", "dynamics_target": f"A_{j}"}
                for j, sign in ((1, 1), (2, -1))
            ],
        }
        with np.load(
            tmp_path / "generated" / row["file"], allow_pickle=False
        ) as arrays:
            assert set(arrays.files) == {"raw_inputs", *labels}
            for call in row["calls"]:
                config, state, _ = parents.calls[cursor]
                cursor += 1
                assert call["input_sha256"] == core.array_hash(state)
                assert call["output_sha256"] == core.array_hash(arrays[call["label"]])
                assert call["config"] == asdict(config)
                assert arrays[call["label"]].dtype == np.float64
                if call["label"].startswith("A_"):
                    j = int(call["label"][-1])
                    expected_input = core.KolmogorovReferenceStepper(
                        parents.config
                    ).canonicalize(arrays["raw_inputs"][j].astype(np.float64))
                    np.testing.assert_array_equal(state, expected_input)
                    np.testing.assert_array_equal(
                        arrays[call["label"]], _map(expected_input)
                    )
            if index == 0:
                assert row["repeat_exact"] is True
                np.testing.assert_array_equal(arrays["repeat_A_clean"], arrays["A_0"])
        assert len(row["numeric_rows"]) == (2 if row["sentinel"] else 0)
    assert before == {p: core._sha256(p) for p in paths}
    _assert_packet(tmp_path / "generated", result)


def test_refinement_lifts_same_input_halves_both_controls_and_never_relabels(
    parents, tmp_path
):
    parents.gain_by_level = True
    result = _run(parents, tmp_path / "refinement")
    assert result["status"] == "completed" and result["qualification_passed"] is False
    first = result["anchors"][0]
    by_label = dict(
        zip((c["label"] for c in first["calls"]), parents.calls[:14], strict=True)
    )
    for j in range(3):
        a_config, a_input, _ = by_label[f"A_{j}"]
        b_config, b_input, _ = by_label[f"B_{j}"]
        c_config, c_input, _ = by_label[f"C_{j}"]
        d_config, d_input, _ = by_label[f"D_{j}"]
        np.testing.assert_array_equal(a_input, b_input)
        np.testing.assert_array_equal(c_input, d_input)
        np.testing.assert_array_equal(
            c_input, core.resize_dealiased_vorticity(a_input, 32)
        )
        assert a_config.resolution == b_config.resolution == 16
        assert c_config.resolution == d_config.resolution == 32
        assert b_config.dt_max == c_config.dt_max == a_config.dt_max / 2
        assert b_config.cfl == c_config.cfl == a_config.cfl / 2
        assert (
            d_config.dt_max == a_config.dt_max / 4 and d_config.cfl == a_config.cfl / 4
        )
    with np.load(tmp_path / "refinement" / first["file"], allow_pickle=False) as arrays:
        assert not np.array_equal(
            arrays["A_1"], core.resize_dealiased_vorticity(arrays["D_1"], 16)
        )
        assert first["training_columns"]["signed_rows"][0]["dynamics_target"] == "A_1"
    _assert_packet(tmp_path / "refinement", result)


def test_refinement_retains_discarded_fine_response_and_state_tail():
    x, y = _grid(16)
    coarse = np.stack(
        [np.cos(x), np.cos(x) + 0.05 * np.sin(y), np.cos(x) - 0.05 * np.sin(y)]
    )
    fine = np.stack([core.resize_dealiased_vorticity(value, 32) for value in coarse])
    xf, _ = _grid(32)
    d = fine.copy()
    d[1] += 0.01 * np.cos(7 * xf)
    geometry = [
        {"input_displacement_rms": core.rms(coarse[j] - coarse[0])} for j in (1, 2)
    ]
    rows = bank.refinement_rows(
        {"A": coarse, "B": coarse, "C": fine, "D": d}, geometry, 0.0
    )
    assert rows[0]["discarded_fine_response_over_displacement"] == pytest.approx(0.2)
    assert rows[0]["discarded_fine_state_relative_l2"] > 0.005
    assert rows[0]["fine_temporal_response_over_displacement"] < 1e-12
    assert rows[1]["discarded_fine_response_over_displacement"] < 1e-12


def test_validate_constructs_complete_bank_with_zero_solver_calls(parents, tmp_path):
    result = _run(parents, tmp_path / "validated", phase="validate")
    assert result["status"] == "completed" and result["qualification_passed"] is None
    assert (
        result["solver_calls_completed"]
        == result["solver_call_attempts"]
        == result["expected_solver_calls"]
        == 0
    )
    assert parents.calls == [] and len(result["anchors"]) == 4
    for row in result["anchors"]:
        assert row["calls"] == row["numeric_rows"] == []
        assert "training_columns" not in row
        with np.load(
            tmp_path / "validated" / row["file"], allow_pickle=False
        ) as arrays:
            assert arrays.files == ["raw_inputs"]
    _assert_packet(tmp_path / "validated", result)


@pytest.mark.parametrize(
    "role,path,value",
    [
        ("population", ("run_id",), "wrong"),
        ("population", ("status",), "failed"),
        ("population", ("source_stable",), False),
        ("extension", ("parent_manifest_sha256",), "wrong"),
        ("extension", ("engineering_gates_pass",), False),
        ("extension", ("population_index", 0, "role"), "development"),
        ("common", ("input_evidence_stable",), False),
        ("common", ("calibration", "validation_pairs"), 1),
        ("common", ("calibration", "training_pairs"), 8),
        ("common", ("calibration", "physical_rms"), 0.0),
        ("common", ("train_scale",), -1.0),
        ("common", ("model_config", "resolution"), 24),
        ("qualification", ("qualification_passed",), False),
        ("qualification", ("inputs_stable",), False),
        ("qualification", ("solver_calls_completed",), 52),
        ("qualification", ("input_files", "common/artifact_manifest.json"), "wrong"),
    ],
)
def test_parent_identity_roles_calibration_and_qualification_rejected(
    parents, tmp_path, role, path, value
):
    _seal(parents, (role, path, value))
    result = _run(parents, tmp_path / "rejected", phase="validate")
    assert result["status"] == "failed" and result["qualification_passed"] is None
    assert parents.calls == [] and result["anchors"] == []
    _assert_packet(tmp_path / "rejected", result)


@pytest.mark.parametrize(
    "change",
    [
        "role",
        "seed",
        "config",
        "block_hash",
        "ambiguous_block",
        "float32",
        "step_order",
        "step_dtype",
        "noncanonical",
        "nonfinite",
    ],
)
def test_training_case_and_native_block_rejection(parents, tmp_path, change):
    root = parents.roots["population"]
    if change in ("float32", "step_order", "step_dtype", "noncanonical", "nonfinite"):
        path = root / "block_2026090611.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if change == "float32":
            arrays["states"] = arrays["states"].astype(np.float32)
        elif change == "step_order":
            arrays["steps"] = arrays["steps"][::-1]
        elif change == "step_dtype":
            arrays["steps"] = arrays["steps"].astype(np.int32)
        elif change == "noncanonical":
            arrays["states"] += 1.0
        else:
            arrays["states"][0, 0, 0] = np.nan
        np.savez_compressed(path, **arrays)

    def mutate(case):
        if change in ("role", "seed"):
            case[change] = "development" if change == "role" else 2026090621
        elif change == "config":
            case["config"]["viscosity"] = 0.02
        elif change == "block_hash":
            case["blocks"][0]["sha256"] = "0" * 64
        elif change == "ambiguous_block":
            case["blocks"] *= 2

    _rewrite_case(parents, mutate)
    result = _run(parents, tmp_path / "rejected", phase="validate")
    assert (
        result["status"] == "failed" and parents.calls == [] and result["anchors"] == []
    )
    _assert_packet(tmp_path / "rejected", result)


@pytest.mark.parametrize(
    "role,name",
    [
        ("common", "result.json"),
        ("population", "case_2026090611.json"),
        ("population", "block_2026090611.npz"),
    ],
)
def test_changed_captured_parent_bytes_rejected(parents, tmp_path, role, name):
    path = parents.roots[role] / name
    path.write_bytes(path.read_bytes() + b" ")
    result = _run(parents, tmp_path / "rejected", phase="validate")
    assert result["status"] == "failed" and "hash mismatch" in result["error"]
    assert parents.calls == []


@pytest.mark.parametrize(
    "which",
    ["input", "input_child", "input_parent", "source", "source_parent", "existing"],
)
def test_output_exclusivity_and_symmetric_isolation(parents, tmp_path, which):
    choices = {
        "input": parents.roots["population"],
        "input_child": parents.roots["population"] / "new",
        "input_parent": parents.roots["population"].parent,
        "source": parents.source,
        "source_parent": parents.source.parent,
        "existing": tmp_path / "existing",
    }
    output = choices[which]
    if which == "existing":
        output.mkdir()
        (output / "keep.txt").write_bytes(b"keep")
    with pytest.raises((ValueError, FileExistsError)):
        _run(parents, output)
    assert parents.calls == []
    if which == "existing":
        assert (output / "keep.txt").read_bytes() == b"keep"


@pytest.mark.parametrize(
    "failure,completed",
    [("solver", 2), ("native", 1), ("clean_label", 4), ("repeat", 14), ("rounding", 0)],
)
def test_failed_draws_and_partial_solver_answers_are_retained(
    parents, tmp_path, failure, completed
):
    if failure == "solver":
        parents.fail_at = 3
    elif failure == "rounding":
        _seal(parents, ("common", ("calibration", "physical_rms"), 1e-20))
    else:
        trigger = {"native": 1, "clean_label": 2, "repeat": 14}[failure]
        parents.hook = lambda number, answer: (
            answer * 1.01 if number == trigger else answer
        )
    result = _run(parents, tmp_path / "failed")
    assert result["status"] == "failed" and result["qualification_passed"] is None
    assert result["solver_calls_completed"] == completed
    assert result["solver_call_attempts"] == completed + (failure == "solver")
    assert len(result["anchors"]) == 1
    row = result["anchors"][0]
    with np.load(tmp_path / "failed" / row["file"], allow_pickle=False) as arrays:
        assert "raw_inputs" in arrays and len(arrays.files) == completed + 1
        assert all(
            call["label"] in arrays
            for call in row["calls"]
            if call["status"] == "completed"
        )
    _assert_packet(tmp_path / "failed", result)


@pytest.mark.parametrize(
    "stage,completed",
    [
        ("source hashing", 0),
        ("solver invocation", 0),
        ("saved solver answer", 1),
        ("completed anchor", 14),
        ("work completion", 33),
    ],
)
def test_stage_deadline_preserves_accounted_answers(
    parents, tmp_path, monkeypatch, stage, completed
):
    original = core.check_deadline

    def check(deadline, current):
        if current == stage:
            raise core.BudgetExceeded("synthetic deadline at " + stage)
        original(deadline, current)

    monkeypatch.setattr(core, "check_deadline", check)
    result = _run(parents, tmp_path / "deadline")
    assert (
        result["status"] == "incomplete_budget"
        and result["qualification_passed"] is None
    )
    assert (
        result["solver_calls_completed"] == result["solver_call_attempts"] == completed
    )
    if stage == "solver invocation":
        assert result["anchors"][0]["calls"][0]["status"] == "prepared"
    _assert_packet(tmp_path / "deadline", result)


@pytest.mark.parametrize("binding", ["source", "input"])
def test_final_provenance_rehash_invalidates_otherwise_complete_bank(
    parents, tmp_path, binding
):
    path = (
        parents.source / "synthetic_source.py"
        if binding == "source"
        else parents.roots["common"] / "result.json"
    )

    def mutate(number, answer):
        if number == 1:
            path.write_bytes(path.read_bytes() + b" ")
        return answer

    parents.hook = mutate
    result = _run(parents, tmp_path / "changed")
    assert (
        result["status"] == "invalid_provenance"
        and result["qualification_passed"] is None
    )
    assert result["solver_calls_completed"] == 33
    assert result["source_stable"] is (binding != "source")
    assert result["inputs_stable"] is (binding != "input")
    _assert_packet(tmp_path / "changed", result)


def test_cli_has_no_fixture_selector(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["generate_kolmogorov_paired_bank", "--help"])
    with pytest.raises(SystemExit) as error:
        bank.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--unit-fixture" not in help_text and "unit_fixture" not in help_text
    assert "validate,generate" in help_text
