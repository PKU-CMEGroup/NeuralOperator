from __future__ import annotations

import hashlib
import json
import sys
import zipfile
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_coverage as p
from tests.time_dependent_no.test_fit_kolmogorov_clean import (
    population as population,  # noqa: PLC0414
)


def seal_added(added, result):
    p.clean._write(added / "result.json", result)
    for case in result["cases"]:
        p.clean._write(added / f"case_{case['seed']}.json", case)
    p.clean._write(
        added / "artifact_manifest.json",
        {
            "run_id": result["run_id"],
            "status": "completed",
            "source_stable": True,
            "parent_manifest_sha256": result["parent_manifest_sha256"],
            "sources": result["sources_before"],
            "artifacts": {
                f.name: p.clean._hash(f)
                for f in added.iterdir()
                if f.name != "artifact_manifest.json"
            },
        },
    )


@pytest.fixture
def coverage_inputs(population, tmp_path):
    """Synthetic evidence only; no reference solver or scientific arrays."""
    parent, parent_source, old_states, _ = population
    parent_result, evidence = p.clean.validate_parent(
        parent, parent_source, unit_fixture=True
    )
    added = tmp_path / "added"
    added.mkdir()
    protocol = replace(
        p.extension.PROTOCOL,
        resolution=16,
        steps=4,
        anchor_steps=(0, 1, 4),
        late_anchor=2,
        late_horizon=2,
        block_steps=2,
        wall_seconds=60,
        seeds=p.extension.NEW_SEEDS[:2],
        fixture_label="unit_test_only",
    )
    cases, added_states = [], []
    for offset, seed in enumerate(protocol.seeds):
        states = old_states[offset] * (3.0 + offset)
        added_states.append(states)
        filename = f"trajectory_{seed}.npz"
        np.savez(added / filename, steps=np.arange(5), states=states)
        cases.append(
            {
                "seed": seed,
                "role": "train",
                "status": "completed",
                "last_retained_step": 4,
                "maximum_palinstrophy": {"step": 1, "complete_trajectory": True},
                "config": parent_result["cases"][0]["config"],
                "blocks": [{"file": filename}],
                "trajectory_diagnostics": [
                    {"step": i, "state_sha256": p.clean._state_hash(s)}
                    for i, s in enumerate(states)
                ],
                "expected_anchor_steps": [0, 1, 4],
                "expected_continuation_anchors": [1, 2],
                "queries": [
                    {
                        "anchor_step": a,
                        "status": "completed",
                        "spatial_relative_l2": 0.0,
                        "fine_discarded_state_relative_l2": 0.0,
                    }
                    for a in (0, 1, 4)
                ],
                "continuation_rows": [
                    {
                        "anchor_step": a,
                        "step": t,
                        "status": "completed",
                        "restricted_state_relative_l2": 0.0,
                    }
                    for a in (1, 2)
                    for t in (1, 2)
                ],
            }
        )
    sources = {
        name: p.clean._hash(p.REPO_ROOT / name) for name in p.extension.SOURCE_PATHS
    }
    result = {
        "run_id": p.extension.RUN_ID + "__UNIT_FIXTURE",
        "status": "completed",
        "engineering_gates_pass": True,
        "source_stable": True,
        "parent_evidence_stable": True,
        "sources_before": sources,
        "sources_after": sources,
        "parent_evidence": evidence,
        "parent_manifest_sha256": evidence["manifest_sha256"],
        "initial_recipe": parent_result["initial_recipe"],
        "protocol": json.loads(json.dumps(asdict(protocol))),
        "population_index": p.extension.population_index(protocol.seeds),
        "new_development_trajectories": 0,
        "protected_access": False,
        "model_evaluated": False,
        "optimization_steps": 0,
        "retained_new_training_transitions": 8,
        "continuation_seeds": list(p.extension.CONTINUATION_SEEDS),
        "cases": cases,
        "numeric_gates": p.extension.numerical_gates(
            cases, protocol, p.extension.CONTINUATION_SEEDS
        ),
        "runtime_replay": {
            "status": "completed",
            "calls": [
                {"status": "completed", "repeat_equal": True, "relative_l2": 0.0}
                for _ in range(6)
            ],
        },
    }
    p.clean._write(added / "launch.json", result)
    seal_added(added, result)
    expected = np.concatenate((old_states[:8], np.stack(added_states), old_states[8:]))
    return parent, parent_source, added, expected, result


@pytest.fixture
def baseline_packet(coverage_inputs, tmp_path, monkeypatch):
    parent, source, _, _, _ = coverage_inputs
    original = p.clean.clean_decision

    def force_twelve(previous, current, competence, *, extended=False):
        decision = original(previous, current, competence, extended=extended)
        if not extended:
            decision.update(extend=True, accepted_terminal=False)
        return decision

    with monkeypatch.context() as patch:
        patch.setattr(p.clean, "clean_decision", force_twelve)
        packet = tmp_path / "baseline"
        result = p.clean.run_clean(parent, source, packet, "cpu", unit_fixture=True)
    assert result["status"] == "completed" and result["updates_completed"] == 12
    return packet


def run_fixture(inputs, baseline_packet, output, phase="fit"):
    parent, source, added, _, _ = inputs
    return p.run(
        parent,
        source,
        added,
        p.REPO_ROOT,
        baseline_packet,
        p.REPO_ROOT,
        output,
        phase,
        "cpu" if phase == "fit" else None,
        unit_fixture=True,
    )


def test_original_bytes_scale_and_disjoint_transition_mapping(coverage_inputs):
    parent, source, added, states, _ = coverage_inputs
    receipt = {}
    pop = p.load_population(
        parent, source, added, p.REPO_ROOT, unit_fixture=True, receipt=receipt
    )
    np.testing.assert_array_equal(pop.states, states.astype(np.float32))
    expected_scale = np.sqrt(np.mean(states[:8, :4] ** 2))
    assert pop.train_scale == pytest.approx(expected_scale, abs=1e-14)
    assert pop.metadata["descriptive_union_input_rms_float64"] == pytest.approx(
        np.sqrt(np.mean(states[:10, :4] ** 2))
    )
    assert pop.metadata["descriptive_union_input_rms_float64"] > 1.5 * pop.train_scale
    assert pop.train_scale < np.sqrt(np.mean(states**2)) / 10
    assert pop.metadata["transition_counts"] == {"train": 40, "development": 16}
    assert (
        pop.metadata["state_store_sha256"]
        == hashlib.sha256(states.astype(np.float32).tobytes()).hexdigest()
    )
    assert [r["seed"] for r in pop.index[-4:]] == list(range(2026090621, 2026090625))
    for role, selected, trajectories, times in (
        ("train", [0, 31, 32, 39], [0, 7, 8, 9], [0, 3, 0, 3]),
        ("development", [0, 3, 4, 15], [10, 10, 11, 13], [0, 3, 0, 3]),
    ):
        x, y = pop.batch(selected, role)
        np.testing.assert_array_equal(x, states[trajectories, times].astype(np.float32))
        np.testing.assert_array_equal(
            y, states[trajectories, np.array(times) + 1].astype(np.float32)
        )
    for bad in ([-1], [40], [0.1], [], [[0]], [True]):
        with pytest.raises(ValueError):
            pop.batch(bad)
    with pytest.raises(ValueError):
        pop.batch([16], "development")
    assert receipt["original"]["training_input_state_count"] == 32
    assert not pop.metadata["development_used_for_scaling"]
    assert not pop.metadata["added_training_used_for_scaling"]
    assert not pop.metadata["terminal_train_targets_used_for_scaling"]


@pytest.mark.parametrize(
    "kind",
    [
        "artifact",
        "extra_file",
        "source",
        "role",
        "parent_binding",
        "gate",
        "missing_path",
        "missing_state",
        "repeated_state",
        "bad_hash",
        "noncanonical",
        "nonfinite",
        "physics",
        "replay",
    ],
)
def test_fail_closed_added_inputs(coverage_inputs, kind):
    parent, source, added, _, result = coverage_inputs
    case = result["cases"][0]
    if kind == "artifact":
        with (added / case["blocks"][0]["file"]).open("ab") as stream:
            stream.write(b"corrupt")
    elif kind == "extra_file":
        (added / "unexpected.txt").write_text("not registered")
    elif kind == "source":
        result["sources_before"][p.extension.SOURCE_PATHS[0]] = "0" * 64
    elif kind == "role":
        result["population_index"][-1]["role"] = "train"
    elif kind == "parent_binding":
        result["parent_manifest_sha256"] = "0" * 64
    elif kind == "gate":
        result["numeric_gates"]["clean_spatial_relative_l2"]["pass"] = False
    elif kind == "missing_path":
        result["cases"].pop()
    elif kind == "physics":
        case["config"] = {**case["config"], "viscosity": 0.1}
    elif kind == "replay":
        result["runtime_replay"]["calls"][0]["repeat_equal"] = False
    else:
        block = added / case["blocks"][0]["file"]
        with np.load(block, allow_pickle=False) as data:
            steps, states = data["steps"].copy(), data["states"].copy()
        if kind == "missing_state":
            steps, states = steps[:-1], states[:-1]
        elif kind == "repeated_state":
            steps[-1] = steps[-2]
        elif kind == "bad_hash":
            states[0] *= 1.01
        else:
            states[0] += 3.0 if kind == "noncanonical" else np.nan
            case["trajectory_diagnostics"][0]["state_sha256"] = p.clean._state_hash(
                states[0]
            )
        np.savez(block, steps=steps, states=states)
    if kind not in ("artifact", "extra_file"):
        seal_added(added, result)
    with pytest.raises((ValueError, KeyError)):
        p.load_population(parent, source, added, p.REPO_ROOT, unit_fixture=True)


def teacher_rows(count=14, steps=4):
    rows = {
        "trajectory_index": np.repeat(np.arange(count), steps),
        "input_step": np.tile(np.arange(steps), count),
    }
    for j, name in enumerate(p.clean.PAIR_COLUMNS, 1):
        rows[name] = j + np.arange(count * steps, dtype=np.float64) ** 2 / 13
    return rows


def test_metric_parity_with_original_evaluator_and_old_added_groups():
    rows = teacher_rows()
    index = p.extension.population_index(p.extension.NEW_SEEDS[:2])
    summary = p.summarize_teacher(
        rows, train_scale=2.0, node_count=256, steps=4, bands=(0, 1, 2, 4), index=index
    )
    selected = (rows["trajectory_index"] < 8) | (rows["trajectory_index"] >= 10)
    old_rows = {name: value[selected].copy() for name, value in rows.items()}
    old_rows["trajectory_index"][old_rows["trajectory_index"] >= 10] -= 2
    old = p.clean.summarize_teacher(
        old_rows, train_scale=2.0, node_count=256, steps=4, bands=(0, 1, 2, 4)
    )
    assert summary["original_train"] == old["train"]
    assert summary["development"] == old["development"]
    assert (
        summary["trajectories"][:8] + summary["trajectories"][-4:]
        == old["trajectories"]
    )
    for role, mask in (
        ("train", rows["trajectory_index"] < 10),
        (
            "added_train",
            (rows["trajectory_index"] >= 8) & (rows["trajectory_index"] < 10),
        ),
    ):
        assert summary[role]["relative_l2"] == pytest.approx(
            np.sqrt(rows["learned_sse"][mask].sum() / rows["target_sse"][mask].sum())
        )
        assert summary[role]["normalized_mse"] == pytest.approx(
            rows["learned_sse"][mask].sum() / (mask.sum() * 256 * 4)
        )
    assert p.clean.check_clean_readiness(summary) == p.clean.check_clean_readiness(old)
    assert len(summary["trajectories"]) == 14


def test_full_32_4_index_budget_and_zero_denominators():
    index = p.extension.population_index(p.extension.NEW_SEEDS)
    rows = teacher_rows(36, 512)
    for name in p.clean.PAIR_COLUMNS:
        rows[name].fill(0)
    summary = p.summarize_teacher(
        rows,
        train_scale=4.0,
        node_count=256**2,
        steps=512,
        bands=(0, 64, 256, 512),
        index=index,
    )
    assert summary["train"]["pair_count"] == 16384
    assert summary["original_train"]["pair_count"] == 4096
    assert summary["added_train"]["pair_count"] == 12288
    assert summary["development"]["pair_count"] == 2048
    assert all(
        row["relative_l2"] is None and row["normalized_mse"] == 0
        for row in summary["trajectories"]
    )
    assert p.clean.check_clean_readiness(summary)["competence_pass"] is None
    assert p.TOTAL_UPDATES * 8 // 16384 == 24
    assert p.TOTAL_UPDATES * 8 // 4096 == 96


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing",
        "index",
        "nan",
        "negative",
        "shape",
        "bands",
        "scale",
        "role",
    ],
)
def test_teacher_rejects_incomplete_or_invalid_metrics(kind):
    rows, bands, scale = teacher_rows(), (0, 1, 2, 4), 2.0
    index = p.extension.population_index(p.extension.NEW_SEEDS[:2])
    if kind == "duplicate":
        rows["input_step"][-1] = 2
    elif kind == "missing":
        rows = {k: v[:-1] for k, v in rows.items()}
    elif kind == "index":
        rows["trajectory_index"][-1] = 14
    elif kind in ("nan", "negative"):
        rows["learned_sse"][0] = np.nan if kind == "nan" else -1
    elif kind == "shape":
        rows["raw_sse"] = rows["raw_sse"][:, None]
    elif kind == "bands":
        bands = (0, 2, 1, 4)
    elif kind == "scale":
        scale = 0.0
    else:
        index[-1]["role"] = "train"
    with pytest.raises(ValueError):
        p.summarize_teacher(
            rows, train_scale=scale, node_count=256, steps=4, bands=bands, index=index
        )


def test_validate_phase_creates_no_model(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not construct a model or call the solver")

    monkeypatch.setattr(p.clean, "PeriodicVorticityPCNO", forbidden)
    monkeypatch.setattr(p.extension.parent_code, "_advance", forbidden)
    result = run_fixture(
        coverage_inputs, baseline_packet, tmp_path / "validated", "validate"
    )
    assert result["status"] == "completed"
    assert result["source_stable"] and result["input_evidence_stable"]
    assert (
        result["updates_completed"]
        == result["teacher_forward_calls"]
        == result["solver_calls"]
        == 0
    )
    assert not result["model_created"] and not result["scientific_training"]
    assert result["terminal_checkpoint"] is None


def test_full_tiny_fit_exact_budget_sampler_ground_truth_and_replay(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch
):
    parent, _, added, states, _ = coverage_inputs
    before = {
        str(root): {f.name: p.clean._hash(f) for f in root.iterdir()}
        for root in (parent, added, baseline_packet)
    }

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "no adaptive extension, solver query or rollout is allowed"
        )

    monkeypatch.setattr(p.clean, "clean_decision", forbidden)
    monkeypatch.setattr(p.extension.parent_code, "_advance", forbidden)
    monkeypatch.setattr(p.baseline, "rollout_one", forbidden)
    output = tmp_path / "new_fit"
    result = run_fixture(coverage_inputs, baseline_packet, output)
    assert result["status"] == "completed", result.get("error")
    assert result["updates_completed"] == 12
    assert result["source_stable"] and result["input_evidence_stable"]
    assert (
        result["run_id"].endswith("__UNIT_FIXTURE")
        and not result["scientific_training"]
    )
    assert (
        not result["rollouts_evaluated"] and not result["development_used_for_updates"]
    )
    assert not result["protected_access"] and result["solver_calls"] == 0
    updates = [
        json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()
    ]
    sampler = p.clean.EpochSampler(40)
    for row, update in zip(updates, range(1, 13), strict=True):
        assert row["update"] == update
        assert row["train_transition_indices"] == sampler.next_indices().tolist()
        assert row["learning_rate"] == p.clean.clean_lr(update, unit_fixture=True)
        assert np.isfinite(row["loss_normalized_mse"]) and np.isfinite(
            row["unclipped_gradient_norm"]
        )
    for epoch in (0, 1):
        values = [
            i
            for row in updates[epoch * 5 : (epoch + 1) * 5]
            for i in row["train_transition_indices"]
        ]
        assert sorted(values) == list(range(40))
    assert [e["update"] for e in result["evaluations"]] == list(range(0, 13, 2))
    expected32 = states.astype(np.float32)
    for e in result["evaluations"]:
        with np.load(output / e["file"], allow_pickle=False) as saved:
            assert saved["learned_sse"].shape == (56,)
            sentinels = saved["sentinel_global_indices"]
            np.testing.assert_array_equal(sentinels, [0, 31, 32, 39, 40, 55])
            np.testing.assert_array_equal(
                saved["sentinel_input"], expected32[sentinels // 4, sentinels % 4]
            )
            np.testing.assert_array_equal(
                saved["sentinel_target"], expected32[sentinels // 4, sentinels % 4 + 1]
            )
            for name, prediction in (
                ("learned_sse", "sentinel_next"),
                ("raw_sse", "sentinel_raw"),
            ):
                errors = saved[prediction].astype(np.float64) - saved[
                    "sentinel_target"
                ].astype(np.float64)
                np.testing.assert_allclose(
                    saved[name][sentinels],
                    np.sum(errors**2, axis=(1, 2)),
                    rtol=1e-12,
                    atol=1e-12,
                )
            for name, prediction in (
                ("raw_output_sse", "sentinel_raw"),
                ("next_output_sse", "sentinel_next"),
            ):
                assert saved[name].shape == (56,)
                assert np.isfinite(saved[name]).all() and np.all(saved[name] >= 0)
                np.testing.assert_allclose(
                    saved[name][sentinels],
                    np.sum(saved[prediction].astype(np.float64) ** 2, axis=(1, 2)),
                    rtol=1e-12,
                    atol=1e-12,
                )
            recomputed = p.summarize_teacher(
                saved,
                train_scale=result["checkpoint_identity"]["train_scale_model_float32"],
                node_count=256,
                steps=4,
                bands=(0, 1, 2, 4),
                index=result["population_index"],
            )
            assert recomputed == e["summary"]
    terminal = result["terminal_checkpoint"]
    assert terminal["update"] == 12 and terminal["replay"]["passed"]
    assert terminal["sha256"] == p.clean._hash(output / terminal["file"])
    assert len(list(output.glob("terminal_*.pt"))) == 1
    assert not list(output.glob("*best*"))
    comparison = result["matched_comparison"]
    assert comparison["baseline_updates"] == comparison["new_updates"] == 12
    assert comparison["baseline_training_passes"] == 3
    assert comparison["new_training_passes"] == 2.4
    assert result["improvement_window_updates"] == [8, 12]
    for group, improvement in result["relative_improvement_last_two_intervals"].items():
        assert improvement == pytest.approx(
            1
            - result["evaluations"][-1]["summary"][group]["relative_l2"]
            / result["evaluations"][-3]["summary"][group]["relative_l2"]
        )
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert set(manifest["artifacts"]) == {f.name for f in output.iterdir()} - {
        "artifact_manifest.json"
    }
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )
    assert len(manifest["input_manifests"]) == 3
    assert before == {
        str(root): {f.name: p.clean._hash(f) for f in root.iterdir()}
        for root in (parent, added, baseline_packet)
    }
    model = p.clean.PeriodicVorticityPCNO(
        **result["model_config"], train_scale=result["data"]["train_input_rms_float64"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    restored = p.clean.EpochSampler(40, seed=999)
    assert (
        p.clean.load_clean_checkpoint(
            output / terminal["file"],
            model,
            optimizer,
            restored,
            identity=result["checkpoint_identity"],
        )
        == 12
    )
    # Fresh model replay and enough draws to cross multiple reshuffle boundaries.
    assert p.clean._terminal_replay(model, output / result["evaluations"][-1]["file"])[
        "passed"
    ]
    for _ in range(12):
        assert restored.next_indices().tolist() == sampler.next_indices().tolist()
    with pytest.raises(FileExistsError):
        run_fixture(coverage_inputs, baseline_packet, output)


@pytest.mark.parametrize(
    "kind",
    [
        "normalizer",
        "baseline_bytes",
        "extension_bytes",
        "loss",
        "evaluation",
        "budget",
        "low_disk",
        "mutate_parent",
    ],
)
def test_honest_failure_packets(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch, kind
):
    parent, _, added, _, _ = coverage_inputs
    if kind == "normalizer":
        original_load = p.load_population

        def wrong_scale(*args, **kwargs):
            pop = original_load(*args, **kwargs)
            pop.train_scale *= 1.1
            return pop

        monkeypatch.setattr(p, "load_population", wrong_scale)
    elif kind == "baseline_bytes":
        with (baseline_packet / "terminal_000012.pt").open("ab") as stream:
            stream.write(b"drift")
    elif kind == "extension_bytes":
        with (added / "result.json").open("a") as stream:
            stream.write(" ")
    elif kind == "loss":
        forward = p.clean.PeriodicVorticityPCNO.forward

        def bad_forward(self, x):
            value = forward(self, x)
            if self.training:
                value["next_state"] = value["next_state"] * float("nan")
            return value

        monkeypatch.setattr(p.clean.PeriodicVorticityPCNO, "forward", bad_forward)
    elif kind in ("evaluation", "budget"):

        def fail(*args, **kwargs):
            if kind == "budget":
                raise TimeoutError("synthetic budget")
            raise RuntimeError("synthetic nonfinite teacher")

        monkeypatch.setattr(p, "teacher_evaluation", fail)
    elif kind == "low_disk":
        monkeypatch.setattr(
            p.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})()
        )
    else:
        original = p.teacher_evaluation

        def mutate(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[3] == 12:
                with (parent / "result.json").open("a") as stream:
                    stream.write(" ")
            return value

        monkeypatch.setattr(p, "teacher_evaluation", mutate)
    output = tmp_path / ("failure_" + kind)
    result = run_fixture(coverage_inputs, baseline_packet, output)
    assert result["status"] == (
        "incomplete_budget"
        if kind == "budget"
        else "invalid_provenance"
        if kind == "mutate_parent"
        else "invalid_input"
        if kind in ("normalizer", "baseline_bytes", "extension_bytes")
        else "failed"
    )
    if kind != "mutate_parent":
        assert (
            result["updates_completed"] == 0 and result["terminal_checkpoint"] is None
        )
    if kind == "extension_bytes":
        assert result["input_evidence_checks"] == {
            "baseline": True,
            "original": True,
            "extension": None,
        }
        assert result["input_evidence_stable"] is None
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


def test_scientific_pins_and_cli_reject_fixture_and_tuning(
    coverage_inputs, monkeypatch, tmp_path
):
    parent, source, added, _, _ = coverage_inputs
    parent_result, evidence = p.clean.validate_parent(parent, source, unit_fixture=True)
    with pytest.raises(ValueError, match="pinned"):
        p.validate_extension(added, p.REPO_ROOT, parent_result, evidence)
    assert p.TOTAL_UPDATES == 49152 and p.WALL_SECONDS == 18000
    assert len(p.SOURCE_PATHS) == len(set(p.SOURCE_PATHS))
    assert all((p.REPO_ROOT / name).is_file() for name in p.SOURCE_PATHS)
    assert p.clean.clean_lr(p.TOTAL_UPDATES) == 1e-4
    with pytest.raises(ValueError):
        p.clean.clean_lr(p.TOTAL_UPDATES + 1)
    for flags in (
        ("--help",),
        ("--unit-fixture",),
        ("--seed", "29"),
        ("--device", "cpu"),
    ):
        monkeypatch.setattr(sys, "argv", ["coverage", *flags])
        with pytest.raises(SystemExit) as exit_info:
            p.main()
        assert exit_info.value.code == (0 if flags == ("--help",) else 2)
    with pytest.raises(ValueError):
        p.run(
            parent,
            source,
            added,
            p.REPO_ROOT,
            tmp_path / "unused",
            p.REPO_ROOT,
            tmp_path / "no_cpu_science",
            "fit",
            "cpu",
        )
    assert not (tmp_path / "no_cpu_science").exists()


def test_teacher_deadline_is_checked_before_forward(coverage_inputs, tmp_path):
    parent, source, added, _, _ = coverage_inputs
    pop = p.load_population(parent, source, added, p.REPO_ROOT, unit_fixture=True)
    model = p.clean.PeriodicVorticityPCNO(
        resolution=16, width=8, depth=2, modes=2, fc_dim=16, train_scale=pop.train_scale
    )
    receipt = {"teacher_forward_calls": 0}
    with pytest.raises(TimeoutError):
        p.teacher_evaluation(
            model, pop, tmp_path, 0, (0, 1, 2, 4), perf_counter() - 1, receipt
        )
    assert receipt["teacher_forward_calls"] == 0


def test_teacher_output_norms_match_every_pair_and_sentinel(
    coverage_inputs, tmp_path, monkeypatch
):
    parent, source, added, _, _ = coverage_inputs
    pop = p.load_population(parent, source, added, p.REPO_ROOT, unit_fixture=True)
    model = p.clean.PeriodicVorticityPCNO(
        resolution=16, width=8, depth=2, modes=2, fc_dim=16, train_scale=pop.train_scale
    )
    monkeypatch.setattr(
        model,
        "forward",
        lambda inputs: {"raw_next": inputs + 1.0, "next_state": inputs * 2.0},
    )
    receipt = {"teacher_forward_calls": 0}
    evaluation = p.teacher_evaluation(
        model, pop, tmp_path, 0, (0, 1, 2, 4), perf_counter() + 300, receipt
    )
    inputs = pop.states[:, :-1].reshape(56, 16, 16)
    with np.load(tmp_path / evaluation["file"], allow_pickle=False) as saved:
        sentinels = saved["sentinel_global_indices"]
        np.testing.assert_array_equal(sentinels, [0, 31, 32, 39, 40, 55])
        for name, field, expected in (
            ("raw_output_sse", "sentinel_raw", inputs + np.float32(1.0)),
            ("next_output_sse", "sentinel_next", inputs * np.float32(2.0)),
        ):
            assert saved[name].shape == (56,) and saved[name].dtype == np.float64
            assert np.isfinite(saved[name]).all() and np.all(saved[name] >= 0)
            np.testing.assert_array_equal(saved[field], expected[sentinels])
            np.testing.assert_allclose(
                saved[name],
                np.sum(expected.astype(np.float64) ** 2, axis=(1, 2)),
                rtol=1e-12,
                atol=1e-12,
            )
    assert receipt["teacher_forward_calls"] == 7


def test_nonfinite_backward_stops_before_optimizer_update(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch
):
    forward = p.clean.PeriodicVorticityPCNO.forward
    backward = torch.Tensor.backward
    losses, gradient_calls, optimizer_calls = [], [], []

    def bad_gradient(gradient):
        assert torch.isfinite(gradient).all()
        gradient_calls.append(True)
        return torch.full_like(gradient, float("nan"))

    def finite_forward_bad_backward(self, inputs):
        predicted = forward(self, inputs)
        if self.training:
            assert torch.isfinite(predicted["next_state"]).all()
            predicted["next_state"].register_hook(bad_gradient)
        return predicted

    def checked_backward(loss, *args, **kwargs):
        losses.append(float(loss.detach()))
        assert np.isfinite(losses[-1])
        return backward(loss, *args, **kwargs)

    def forbidden_step(*args, **kwargs):
        optimizer_calls.append(True)
        raise AssertionError("nonfinite gradients must not reach Adam.step")

    monkeypatch.setattr(
        p.clean.PeriodicVorticityPCNO, "forward", finite_forward_bad_backward
    )
    monkeypatch.setattr(torch.Tensor, "backward", checked_backward)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_step)
    output = tmp_path / "nonfinite_gradient"
    result = run_fixture(coverage_inputs, baseline_packet, output)
    assert len(losses) == len(gradient_calls) == 1 and not optimizer_calls
    assert result["status"] == "failed" and result["stage"] == "training"
    assert result["error_type"] == "RuntimeError" and "non-finite" in result["error"]
    assert result["updates_completed"] == 0 and result["terminal_checkpoint"] is None
    assert result["latest_checkpoint"]["update"] == 0
    assert result["source_stable"] and result["input_evidence_stable"]
    assert (output / "updates.jsonl").read_text() == ""
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == "failed"
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


@pytest.mark.parametrize("role", ["original", "extension", "baseline"])
@pytest.mark.parametrize("relationship", ["same", "nested", "ancestor"])
def test_output_isolation_precedes_directory_creation(
    coverage_inputs, tmp_path, role, relationship
):
    parent, _, added, _, _ = coverage_inputs
    old_clean = tmp_path / "untouched_baseline"
    old_clean.mkdir()
    roots = {"original": parent, "extension": added, "baseline": old_clean}
    packet = roots[role]
    output = {
        "same": packet,
        "nested": packet / "new" / "nested",
        "ancestor": packet.parent,
    }[relationship]
    before = {
        str(root): {f.name: p.clean._hash(f) for f in root.iterdir()}
        for root in roots.values()
    }
    with pytest.raises(ValueError, match="disjoint"):
        run_fixture(coverage_inputs, old_clean, output, "validate")
    assert before == {
        str(root): {f.name: p.clean._hash(f) for f in root.iterdir()}
        for root in roots.values()
    }
    assert not (packet / "new").exists()


@pytest.mark.parametrize("mutate", [False, True])
def test_loading_failure_preserves_all_validated_evidence(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch, mutate
):
    parent = coverage_inputs[0]
    original_empty = np.empty

    def fail_union(shape, *args, **kwargs):
        if shape == (14, 5, 16, 16):
            if mutate:
                with (parent / "result.json").open("a") as stream:
                    stream.write(" ")
            raise MemoryError("synthetic union allocation failure")
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", fail_union)
    output = tmp_path / "loading_failure"
    result = run_fixture(coverage_inputs, baseline_packet, output, "validate")
    assert result["status"] == ("invalid_provenance" if mutate else "invalid_input")
    assert result["error_type"] == "MemoryError"
    assert result["input_evidence_checks"] == {
        "baseline": True,
        "original": not mutate,
        "extension": True,
    }
    assert result["input_evidence_stable"] is not mutate
    assert not result["model_created"] and result["updates_completed"] == 0
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert len(manifest["input_manifests"]) == 3
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


@pytest.mark.parametrize("error_type", [zipfile.BadZipFile, EOFError, zlib.error])
@pytest.mark.parametrize("mutate", [False, True])
def test_npz_decoder_failure_preserves_all_validated_evidence(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch, error_type, mutate
):
    parent, _, added, _, extension_result = coverage_inputs
    block = added / extension_result["cases"][0]["blocks"][0]["file"]
    original_load = np.load

    def fail_decode(path, *args, **kwargs):
        if Path(path) == block:
            if mutate:
                with (parent / "result.json").open("a") as stream:
                    stream.write(" ")
            raise error_type("synthetic NPZ decoder failure")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", fail_decode)
    output = tmp_path / "decoder_failure"
    result = run_fixture(coverage_inputs, baseline_packet, output, "validate")
    assert result["status"] == ("invalid_provenance" if mutate else "invalid_input")
    assert result["error_type"] == error_type.__name__
    assert result["input_evidence_checks"] == {
        "baseline": True,
        "original": not mutate,
        "extension": True,
    }
    assert result["input_evidence_stable"] is not mutate
    assert not result["model_created"] and result["updates_completed"] == 0
    assert p.clean._read(output / "result.json") == result
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert len(manifest["input_manifests"]) == 3
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


def test_aliased_input_paths_do_not_validate_unchecked_roles(baseline_packet, tmp_path):
    output = tmp_path / "aliased_inputs"
    result = p.run(
        baseline_packet,
        p.REPO_ROOT,
        baseline_packet,
        p.REPO_ROOT,
        baseline_packet,
        p.REPO_ROOT,
        output,
        "validate",
        unit_fixture=True,
    )
    assert result["status"] == "invalid_input" and result["source_stable"]
    assert result["input_evidence_checks"] == {
        "baseline": True,
        "original": None,
        "extension": None,
    }
    assert result["input_evidence_stable"] is None
    assert not result["model_created"] and result["updates_completed"] == 0
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == "invalid_input"
    assert manifest["input_manifests"] == [
        result["baseline_evidence"]["manifest_sha256"]
    ]
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


@pytest.mark.parametrize("kind", ["initial_hash", "final_hash", "launch_write"])
def test_source_and_setup_errors_produce_honest_failure_packets(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch, kind
):
    source_path = p.REPO_ROOT / p.SOURCE_PATHS[0]
    original_hash, original_write = p.clean._hash, p.clean._write
    calls = 0

    def fail_hash(path):
        nonlocal calls
        if path == source_path:
            calls += 1
            if (kind, calls) in (("initial_hash", 1), ("final_hash", 2)):
                raise OSError("synthetic unreadable source")
        return original_hash(path)

    def fail_write(path, value):
        if kind == "launch_write" and path.name == "launch.json":
            raise OSError("synthetic launch write failure")
        return original_write(path, value)

    monkeypatch.setattr(p.clean, "_hash", fail_hash)
    monkeypatch.setattr(p.clean, "_write", fail_write)
    output = tmp_path / kind
    result = run_fixture(coverage_inputs, baseline_packet, output, "validate")
    assert result["status"] == (
        "invalid_input" if kind == "launch_write" else "invalid_provenance"
    )
    assert result["source_stable"] is (kind == "launch_write")
    if kind == "final_hash":
        assert result["sources_after"][p.SOURCE_PATHS[0]] is None
        assert p.SOURCE_PATHS[0] in result["source_check_errors"]
        assert result["input_evidence_stable"] is True
    else:
        assert result["input_evidence_stable"] is None
        assert result["input_evidence_checks"] == dict.fromkeys(
            ("baseline", "original", "extension")
        )
    assert not result["model_created"] and result["updates_completed"] == 0
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert all(
        original_hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )


@pytest.mark.parametrize("kind", ["last_batch", "serialization"])
def test_teacher_detects_deadline_crossed_inside_last_operation(
    coverage_inputs, tmp_path, monkeypatch, kind
):
    parent, source, added, _, _ = coverage_inputs
    pop = p.load_population(parent, source, added, p.REPO_ROOT, unit_fixture=True)
    model = p.clean.PeriodicVorticityPCNO(
        resolution=16, width=8, depth=2, modes=2, fc_dim=16, train_scale=pop.train_scale
    )
    clock = [0.0]
    receipt = {"teacher_forward_calls": 0}
    forward, save = model.forward, np.savez_compressed

    def finish_last_batch(inputs):
        result = forward(inputs)
        if receipt["teacher_forward_calls"] == 6:
            clock[0] = 301.0
        return result

    def finish_serialization(*args, **kwargs):
        save(*args, **kwargs)
        clock[0] = 301.0

    monkeypatch.setattr(p, "perf_counter", lambda: clock[0])
    if kind == "last_batch":
        monkeypatch.setattr(model, "forward", finish_last_batch)
    else:
        monkeypatch.setattr(np, "savez_compressed", finish_serialization)
    with pytest.raises(TimeoutError, match="teacher"):
        p.teacher_evaluation(model, pop, tmp_path, 0, (0, 1, 2, 4), 300.0, receipt)
    assert receipt["teacher_forward_calls"] == 7
    assert (tmp_path / "teacher_000000.npz").exists() is (kind == "serialization")


@pytest.mark.parametrize(
    "kind", ["last_update", "latest_save", "terminal_save", "terminal_load", "replay"]
)
def test_deadline_crossing_cannot_be_reported_as_completed(
    coverage_inputs, baseline_packet, tmp_path, monkeypatch, kind
):
    clock = [0.0]
    calls = 0
    original_step = torch.optim.Adam.step
    original_save = p.clean.save_clean_checkpoint
    original_load = p.clean.load_clean_checkpoint
    original_replay = p.clean._terminal_replay

    def finish_step(*args, **kwargs):
        nonlocal calls
        value = original_step(*args, **kwargs)
        calls += 1
        if kind == "last_update" and calls == 12:
            clock[0] = 301.0
        return value

    def finish_save(path, *args, **kwargs):
        value = original_save(path, *args, **kwargs)
        if kwargs["update"] == 12 and (
            (kind == "latest_save" and path.name == "latest_checkpoint.pt")
            or (kind == "terminal_save" and path.name == "terminal_000012.pt")
        ):
            clock[0] = 301.0
        return value

    def finish_load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        if kind == "terminal_load":
            clock[0] = 301.0
        return value

    def finish_replay(*args, **kwargs):
        value = original_replay(*args, **kwargs)
        if kind == "replay":
            clock[0] = 301.0
        return value

    monkeypatch.setattr(p, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.optim.Adam, "step", finish_step)
    monkeypatch.setattr(p.clean, "save_clean_checkpoint", finish_save)
    monkeypatch.setattr(p.clean, "load_clean_checkpoint", finish_load)
    monkeypatch.setattr(p.clean, "_terminal_replay", finish_replay)
    output = tmp_path / kind
    result = run_fixture(coverage_inputs, baseline_packet, output)
    assert result["status"] == "incomplete_budget"
    assert result["updates_completed"] == 12
    assert result["terminal_checkpoint"] is None
    assert result["source_stable"] and result["input_evidence_stable"]
    assert result["work_seconds"] == 301.0
    assert result["provenance_check_seconds"] == 0.0
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == "incomplete_budget"
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )
