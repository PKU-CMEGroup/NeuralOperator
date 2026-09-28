from __future__ import annotations

import json
import random
import sys

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_clean as p


def _seal(parent, sources):
    p._write(
        parent / "artifact_manifest.json",
        {
            "run_id": p.PARENT_RUN_ID + "__UNIT_FIXTURE",
            "source_stable": True,
            "sources": sources,
            "artifacts": {
                f.name: p._hash(f)
                for f in parent.iterdir()
                if f.name != "artifact_manifest.json"
            },
        },
    )


@pytest.fixture
def population(tmp_path):
    parent, sources_root = tmp_path / "C", tmp_path / "source"
    parent.mkdir()
    sources = {}
    for name in p.PARENT_SOURCE_PATHS:
        path = sources_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source binding " + name)
        sources[name] = p._hash(path)
    x, y = np.meshgrid(
        np.arange(16) * 2 * np.pi / 16, np.arange(16) * 2 * np.pi / 16, indexing="ij"
    )
    full = []
    cases = []
    for i, (seed, role) in enumerate(p.SEED_ROLES):
        states = np.stack(
            [
                ((1.00000009 + 0.03 * i + 0.2 * t) if t < 4 and i < 8 else 30.0 + i + t)
                * np.cos(x + y + 0.1 * i)
                + 0.17 * np.sin(2 * x - y)
                for t in range(5)
            ]
        ).astype(np.float64)
        full.append(states)
        name = f"trajectory_{seed}.npz"
        np.savez(parent / name, steps=np.arange(5), states=states)
        case = {
            "seed": seed,
            "role": role,
            "status": "completed",
            "last_retained_step": 4,
            "maximum_palinstrophy": {"complete_trajectory": True},
            "config": {
                "resolution": 16,
                "viscosity": 0.01,
                "linear_drag": 0.1,
                "forcing_amplitude": 1.0,
                "forcing_wavenumber": 4,
                "macro_dt": 0.05,
                "domain_length": 2 * np.pi,
            },
            "blocks": [{"file": name}],
            "trajectory_diagnostics": [
                {"step": t, "state_sha256": p._state_hash(s)}
                for t, s in enumerate(states)
            ],
        }
        cases.append(case)
        p._write(parent / f"case_{seed}.json", case)
    result = {
        "run_id": p.PARENT_RUN_ID + "__UNIT_FIXTURE",
        "status": "completed",
        "source_stable": True,
        "parent_evidence_stable": True,
        "engineering_gates_pass": True,
        "sources_before": sources,
        "sources_after": sources,
        "protocol": {"resolution": 16, "steps": 4, "macro_dt": 0.05},
        "initial_recipe": {"zero_mean_velocity": True},
        "seed_roles": [{"seed": s, "role": r} for s, r in p.SEED_ROLES],
        "retained_transitions_by_role": {"train": 32, "development": 16},
        "cases": cases,
        "numeric_gates": {
            name: {
                "complete": True,
                "pass": True,
                "limit": limit,
                "maximum": 0.0,
                "sample_count": 12,
                "expected_sample_count": 12,
            }
            for name, limit in p.GATE_LIMITS.items()
        },
    }
    p._write(parent / "result.json", result)
    p._write(
        parent / "launch.json",
        {
            "run_id": result["run_id"],
            "sources": sources,
            "protocol": result["protocol"],
        },
    )
    _seal(parent, sources)
    return parent, sources_root, np.stack(full), sources


def test_strict_train_only_float64_scale_and_transition_mapping(population):
    parent, source, states, _ = population
    data = p.load_population(parent, source, unit_fixture=True)
    expected = np.sqrt(np.mean(states[:8, :4] ** 2))
    assert data.train_scale == pytest.approx(expected, abs=1e-14)
    assert (
        abs(
            data.train_scale
            - float(np.sqrt(np.mean(states[:8, :4].astype(np.float32) ** 2)))
        )
        > 1e-9
    )
    assert data.train_scale < np.sqrt(np.mean(states**2)) / 10
    assert data.states.dtype == np.float32 and data.states.shape == (12, 5, 16, 16)
    assert data.states.nbytes == 12 * 5 * 16 * 16 * 4
    a, b = data.batch([0, 3, 4, 31])
    np.testing.assert_array_equal(
        a.numpy(), states[[0, 0, 1, 7], [0, 3, 0, 3]].astype(np.float32)
    )
    np.testing.assert_array_equal(
        b.numpy(), states[[0, 0, 1, 7], [1, 4, 1, 4]].astype(np.float32)
    )
    a, b = data.batch([0, 15], "development")
    np.testing.assert_array_equal(a.numpy(), states[[8, 11], [0, 3]].astype(np.float32))
    np.testing.assert_array_equal(b.numpy(), states[[8, 11], [1, 4]].astype(np.float32))
    for bad in ([-1], [32], [0.0], [True], []):
        with pytest.raises(ValueError):
            data.batch(bad)
    assert not data.metadata["development_used_for_scaling"]
    with pytest.raises(ValueError, match="pinned"):
        p.validate_parent(parent, source)


@pytest.mark.parametrize(
    "corruption", ("payload", "source", "role", "gate", "index", "noncanonical")
)
def test_parent_rejection_and_honest_failure_manifest(population, tmp_path, corruption):
    parent, source, states, sources = population
    if corruption == "source":
        (source / p.PARENT_SOURCE_PATHS[0]).write_text("tamper")
    elif corruption == "payload":
        (parent / "result.json").write_text("tamper")
    else:
        result = p._read(parent / "result.json")
        if corruption == "role":
            result["seed_roles"][0]["role"] = "development"
        elif corruption == "gate":
            result["numeric_gates"]["clean_spatial_relative_l2"]["pass"] = False
        else:
            case = result["cases"][0]
            indices = np.arange(5)
            if corruption == "index":
                indices[2] = 1
            else:
                states[0, 2] += 0.1
                case["trajectory_diagnostics"][2]["state_sha256"] = p._state_hash(
                    states[0, 2]
                )
                p._write(parent / f"case_{case['seed']}.json", case)
            np.savez(
                parent / case["blocks"][0]["file"], steps=indices, states=states[0]
            )
        p._write(parent / "result.json", result)
        _seal(parent, sources)
    output = tmp_path / "failed"
    result = p.run_resource(parent, source, output, "cpu", unit_fixture=True)
    launch = p._read(output / "launch.json")
    assert launch["model_config"] == result["model_config"]
    assert launch["recipe"] == result["recipe"]
    assert result["status"] == "invalid_parent"
    assert not (output / "updates.jsonl").exists()
    manifest = p._read(output / "artifact_manifest.json")
    assert all(
        p._hash(output / name) == value for name, value in manifest["artifacts"].items()
    )


def test_tiny_resource_complete_no_checkpoint_no_scientific_claim(population, tmp_path):
    parent, source, _, _ = population
    output = tmp_path / "resource"
    result = p.run_resource(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "completed" and result["run_id"].endswith(
        "__UNIT_FIXTURE"
    )
    assert result["source_stable"] and result["parent_evidence_stable"]
    assert result["zero_final_head"] and len(result["updates"]) == 5
    expected = torch.randperm(32, generator=torch.Generator().manual_seed(17)).tolist()[
        :10
    ]
    assert [
        i for row in result["updates"] for i in row["train_transition_indices"]
    ] == expected
    assert all(
        np.isfinite(row["loss_normalized_mse"])
        and np.isfinite(row["unclipped_gradient_norm"])
        for row in result["updates"]
    )
    assert result["train_scale_model_float32"] == float(
        np.float32(result["data"]["train_input_rms_float64"])
    )
    assert result["timing_warmup_updates_excluded"] == 3
    assert result["state_dict_tensor_bytes"] > result["parameter_count"] * 4
    assert set(result["inference"]) == {"2", "1"}
    assert all(len(value["timings"]) == 10 for value in result["inference"].values())
    assert {f.name for f in output.iterdir()} == {
        "launch.json",
        "updates.jsonl",
        "result.json",
        "artifact_manifest.json",
    }
    assert not result["scientific_training"] and not result["checkpoint_saved"]
    assert not result["development_model_evaluation"]
    with pytest.raises(FileExistsError):
        p.run_resource(parent, source, output, "cpu", unit_fixture=True)


def test_nonfinite_loss_stops_without_optimizer_step(population, tmp_path, monkeypatch):
    parent, source, _, _ = population
    original = p.PeriodicVorticityPCNO.forward

    def bad(self, inputs):
        value = original(self, inputs)
        value["next_state"] = value["next_state"] * float("nan")
        return value

    monkeypatch.setattr(p.PeriodicVorticityPCNO, "forward", bad)
    result = p.run_resource(
        parent, source, tmp_path / "nonfinite", "cpu", unit_fixture=True
    )
    assert result["status"] == "failed" and result["updates"] == []
    assert result["error"] == "nonfinite resource loss"


def test_source_drift_and_cli_failure(population, tmp_path, monkeypatch):
    parent, source, _, _ = population
    original = p._sources
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        values = original()
        if calls > 1:
            values[p.SOURCE_PATHS[0]] = "0" * 64
        return values

    monkeypatch.setattr(p, "_sources", changed)
    result = p.run_resource(
        parent, source, tmp_path / "drift", "cpu", unit_fixture=True
    )
    assert result["status"] == "invalid_source"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resource",
            "--parent",
            "C",
            "--parent-source",
            "source",
            "--output",
            "fresh",
            "--device",
            "cuda",
        ],
    )
    monkeypatch.setattr(p, "run_resource", lambda *args: {"status": "failed"})
    with pytest.raises(SystemExit) as error:
        p.main()
    assert error.value.code == 1


def test_epoch_sampler_exact_coverage_and_resumed_boundary():
    sampler = p.EpochSampler(32, batch_size=8, seed=17)
    twin = p.EpochSampler(32, batch_size=8, seed=17)
    epochs = []
    for _ in range(3):
        batches = [sampler.next_indices() for _ in range(4)]
        assert all(batch.dtype == np.int64 and batch.shape == (8,) for batch in batches)
        order = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(order), np.arange(32))
        np.testing.assert_array_equal(
            order, np.concatenate([twin.next_indices() for _ in range(4)])
        )
        epochs.append(order)
    assert not np.array_equal(epochs[0], epochs[1])
    # Save at an exact epoch boundary, then also inside the next epoch.
    for _ in range(2):
        state = sampler.state_dict()
        resumed = p.EpochSampler(32, batch_size=8, seed=123)
        resumed.load_state_dict(state)
        for _ in range(5):
            np.testing.assert_array_equal(
                sampler.next_indices(), resumed.next_indices()
            )
    full = p.EpochSampler(4096, batch_size=8, seed=17)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([full.next_indices() for _ in range(512)])),
        np.arange(4096),
    )


def test_clean_schedule_warmup_cosine_and_single_extension():
    assert p.clean_lr(1) == pytest.approx(1e-3 / 512)
    assert p.clean_lr(512) == pytest.approx(1e-3)
    assert p.clean_lr(513) == pytest.approx(
        1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi / (32768 - 512)))
    )
    assert p.clean_lr(16640) == pytest.approx(0.00055)
    assert p.clean_lr(32768) == pytest.approx(1e-4)
    assert p.clean_lr(32769) == p.clean_lr(49152) == 1e-4
    assert p.clean_lr(1, unit_fixture=True) == pytest.approx(0.0005)
    assert p.clean_lr(2, unit_fixture=True) == pytest.approx(0.001)
    assert p.clean_lr(8, unit_fixture=True) == pytest.approx(0.0001)
    assert p.clean_lr(9, unit_fixture=True) == p.clean_lr(12, unit_fixture=True) == 1e-4


@pytest.mark.parametrize("corruption", ("duplicate", "cursor", "size"))
def test_epoch_sampler_rejects_invalid_continuation(corruption):
    sampler = p.EpochSampler(32)
    state = sampler.state_dict()
    if corruption == "duplicate":
        state["order"][-1] = state["order"][0]
    elif corruption == "cursor":
        state["cursor"] = 3
    else:
        state["size"] = 64
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


def _teacher_rows():
    # Complete twelve-trajectory, four-transition synthetic metric bank.
    return {
        "trajectory_index": np.repeat(np.arange(12), 4),
        "input_step": np.tile(np.arange(4), 12),
        "learned_sse": np.tile([1.0, 0.0, 1.0, 0.0], 12),
        "raw_sse": np.tile([4.0, 0.0, 4.0, 0.0], 12),
        "target_sse": np.tile([1.0, 100.0, 1.0, 100.0], 12),
        "input_sse": np.full(48, 16.0),
        "persistence_sse": np.full(48, 4.0),
        "projection_sse": np.ones(48),
    }


def _teacher_summary(rows):
    return p.summarize_teacher(
        rows, train_scale=2.0, node_count=2, steps=4, bands=(0, 1, 2, 4)
    )


def test_teacher_metrics_pool_squares_and_report_each_development_band():
    rows = _teacher_rows()
    original = {key: value.copy() for key, value in rows.items()}
    summary = _teacher_summary(rows)
    for role in ("train", "development"):
        metric = summary[role]
        assert metric["relative_l2"] == pytest.approx(np.sqrt(2 / 202))
        assert metric["relative_l2"] != pytest.approx(0.5)  # Mean per-pair ratio.
        assert metric["normalized_mse"] == pytest.approx(0.5 / (2 * 2**2))
        assert metric["persistence_relative_l2"] == pytest.approx(np.sqrt(16 / 202))
        assert metric["zero_relative_l2"] == 1.0
        assert metric["learned_over_persistence"] == pytest.approx(np.sqrt(2 / 16))
        assert metric["raw_relative_l2"] == pytest.approx(np.sqrt(8 / 202))
    assert [(row["seed"], row["role"]) for row in summary["trajectories"]] == list(
        p.SEED_ROLES
    )
    development = [
        row for row in summary["trajectories"] if row["role"] == "development"
    ]
    assert len(development) == 4
    for row in development:
        assert [(b["start"], b["stop"]) for b in row["bands"]] == [
            (0, 1),
            (1, 2),
            (2, 4),
        ]
        assert [b["learned_over_persistence"] for b in row["bands"]] == pytest.approx(
            [0.5, 0.0, np.sqrt(1 / 8)]
        )
    for key in rows:
        np.testing.assert_array_equal(rows[key], original[key])


def test_real_development_band_boundaries_are_half_open():
    steps = np.tile(np.arange(512), 12)
    rows = {
        "trajectory_index": np.repeat(np.arange(12), 512),
        "input_step": steps,
        **{name: np.ones(12 * 512) for name in p.PAIR_COLUMNS},
    }
    rows["learned_sse"] = np.isin(steps, [63, 64, 255, 256, 511]).astype(np.float64)
    summary = p.summarize_teacher(
        rows, train_scale=1.0, node_count=1, steps=512, bands=(0, 64, 256, 512)
    )
    for trajectory in summary["trajectories"][8:]:
        assert [band["pair_count"] for band in trajectory["bands"]] == [64, 192, 256]
        assert [
            band["squared_norm_sums"]["learned_sse"] for band in trajectory["bands"]
        ] == [1, 2, 2]
        assert [
            band["learned_over_persistence"] for band in trajectory["bands"]
        ] == pytest.approx(np.sqrt([1 / 64, 2 / 192, 2 / 256]))


def test_readiness_checks_all_twelve_bands_and_zero_denominators():
    rows = _teacher_rows()
    rows["target_sse"][:] = 1e6
    rows["learned_sse"][:] = 0.25 * rows["persistence_sse"]
    ready = p.check_clean_readiness(_teacher_summary(rows))
    assert ready["global_pass"] is True and ready["band_pass"] is True
    assert ready["competence_pass"] is True
    last_band = (rows["trajectory_index"] == 11) & (rows["input_step"] >= 2)
    rows["learned_sse"][last_band] = 0.26 * rows["persistence_sse"][last_band]
    unready = p.check_clean_readiness(_teacher_summary(rows))
    assert unready["global_pass"] is True and unready["band_pass"] is False
    assert unready["competence_pass"] is False
    rows["learned_sse"][last_band] = 0.0
    rows["persistence_sse"][last_band] = 0.0
    summary = _teacher_summary(rows)
    assert summary["trajectories"][-1]["bands"][-1]["learned_over_persistence"] is None
    assert p.check_clean_readiness(summary)["competence_pass"] is not True
    rows["target_sse"][:] = 0.0
    summary = _teacher_summary(rows)
    assert summary["development"]["relative_l2"] is None
    assert summary["development"]["zero_relative_l2"] is None
    assert p.check_clean_readiness(summary)["global_pass"] is not True


@pytest.mark.parametrize(
    "field",
    (
        "learned_sse",
        "raw_sse",
        "target_sse",
        "input_sse",
        "persistence_sse",
        "projection_sse",
    ),
)
@pytest.mark.parametrize("value", (np.nan, np.inf, -1.0))
def test_teacher_metrics_reject_invalid_squared_errors(field, value):
    rows = _teacher_rows()
    rows[field][0] = value
    with pytest.raises(ValueError):
        _teacher_summary(rows)


@pytest.mark.parametrize(
    "corruption", ("missing", "duplicate", "invalid_trajectory", "invalid_step")
)
def test_teacher_metrics_require_complete_unique_transition_coverage(corruption):
    rows = _teacher_rows()
    if corruption == "missing":
        rows = {name: values[:-1] for name, values in rows.items()}
    elif corruption == "duplicate":
        for values in rows.values():
            values[-1] = values[-2]
    elif corruption == "invalid_trajectory":
        rows["trajectory_index"][-1] = 12
    else:
        rows["input_step"][-1] = 4
    with pytest.raises(ValueError):
        _teacher_summary(rows)


@pytest.mark.parametrize(
    "current,improving",
    ((0.95, False), (np.nextafter(0.95, 0.0), True), (np.nextafter(0.95, 1.0), False)),
)
def test_clean_extension_strict_five_percent_and_no_second_extension(
    current, improving
):
    decision = p.clean_decision(1.0, current, True)
    assert decision["still_improving"] is improving
    assert decision["extend"] is improving
    assert decision["accepted_terminal"] is (not improving)
    final = p.clean_decision(1.0, current, True, extended=True)
    assert final["extend"] is False
    assert final["still_improving"] is improving
    assert final["accepted_terminal"] is (not improving)
    assert (
        p.clean_decision(1.0, current, False, extended=True)["accepted_terminal"]
        is False
    )


@pytest.mark.parametrize(
    "previous,current",
    ((0.0, 0.0), (None, 0.1), (1.0, None), (np.nan, 0.1), (1.0, np.inf)),
)
def test_clean_decision_undefined_or_nonfinite_never_promotes(previous, current):
    decision = p.clean_decision(previous, current, True)
    assert decision["accepted_terminal"] is False
    assert decision["extend"] is False
    assert decision["relative_improvement"] is None


def test_clean_checkpoint_restores_sampler_optimizer_and_all_cpu_rng(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = p.EpochSampler(32, batch_size=8, seed=17)
    inputs = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64
    identity = {"run_id": "CHECKPOINT__UNIT_FIXTURE", "parent": "synthetic only"}

    def update(net, opt, order):
        indices = order.next_indices()
        random_value = (
            random.random() + float(np.random.random()) + float(torch.rand(()))
        )
        opt.zero_grad(set_to_none=True)
        loss = ((net(inputs[indices]) - random_value) ** 2).mean()
        loss.backward()
        opt.step()
        return indices, float(loss.detach()), random_value

    for _ in range(4):
        update(model, optimizer, sampler)
    checkpoint = tmp_path / "resume.pt"
    digest = p.save_clean_checkpoint(
        checkpoint, model, optimizer, sampler, update=4, identity=identity
    )
    assert digest == p._hash(checkpoint)
    expected = update(model, optimizer, sampler)
    restored = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1)
    )
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.2)
    restored_sampler = p.EpochSampler(32, batch_size=8, seed=999)
    assert (
        p.load_clean_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            restored_sampler,
            identity=identity,
        )
        == 4
    )
    actual = update(restored, restored_optimizer, restored_sampler)
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]
    for before, after in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    first, second = optimizer.state_dict(), restored_optimizer.state_dict()
    assert first["param_groups"] == second["param_groups"]
    for index, state in first["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, second["state"][index][name], rtol=0, atol=0
            )
    with pytest.raises(ValueError):
        p.load_clean_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            restored_sampler,
            identity={"run_id": "wrong identity"},
        )


@pytest.mark.parametrize("force_extension", (False, True))
def test_full_tiny_clean_terminal_evidence_and_teacher_replay(
    population, tmp_path, monkeypatch, force_extension
):
    parent, source, states, _ = population
    before = p._hash(parent / "artifact_manifest.json")
    real_decision = p.clean_decision

    def fixture_branch(previous, current, competence, *, extended=False):
        decision = real_decision(previous, current, competence, extended=extended)
        if not extended:
            # Exercise both bounded orchestration branches, not a scientific gate.
            decision.update(
                extend=force_extension,
                accepted_terminal=False,
                verdict="synthetic fixture branch override only",
            )
        return decision

    monkeypatch.setattr(p, "clean_decision", fixture_branch)
    output = tmp_path / "full_clean"
    result = p.run_clean(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "completed"
    assert result["run_id"] == "CM_NEXT_KF_CLEAN_20260907A__UNIT_FIXTURE"
    assert result["source_stable"] and result["parent_evidence_stable"]
    assert p._hash(parent / "artifact_manifest.json") == before
    assert result["data"]["evidence"]["manifest_sha256"] == before
    assert result["data"]["train_input_rms_float64"] == pytest.approx(
        np.sqrt(np.mean(states[:8, :4] ** 2)), abs=1e-14
    )
    updates = [
        json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()
    ]
    terminal = 12 if force_extension else 8
    assert [row["update"] for row in updates] == list(range(1, terminal + 1))
    for first in range(0, terminal, 4):
        indices = np.concatenate(
            [row["train_transition_indices"] for row in updates[first : first + 4]]
        )
        np.testing.assert_array_equal(np.sort(indices), np.arange(32))
    expected_sampler = p.EpochSampler(32, batch_size=8, seed=17)
    for row in updates:
        np.testing.assert_array_equal(
            row["train_transition_indices"], expected_sampler.next_indices()
        )
        assert row["learning_rate"] == pytest.approx(
            p.clean_lr(row["update"], unit_fixture=True)
        )
        assert np.isfinite(row["loss_normalized_mse"])
        assert np.isfinite(row["unclipped_gradient_norm"])
    evaluations = [
        json.loads(line)
        for line in (output / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [row["update"] for row in evaluations] == list(range(0, terminal + 1, 2))
    assert result["evaluations"] == evaluations
    state32 = states.astype(np.float32)
    for evaluation in evaluations:
        path = output / f"teacher_{evaluation['update']:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            assert data["learned_sse"].shape == (48,)
            for role, selector in (
                ("train", data["trajectory_index"] < 8),
                ("development", data["trajectory_index"] >= 8),
            ):
                learned = np.sum(data["learned_sse"][selector], dtype=np.float64)
                target = np.sum(data["target_sse"][selector], dtype=np.float64)
                count = int(np.count_nonzero(selector))
                summary = evaluation["summary"][role]
                assert summary["relative_l2"] == pytest.approx(
                    np.sqrt(learned / target), rel=1e-12
                )
                assert summary["normalized_mse"] == pytest.approx(
                    learned
                    / (
                        count
                        * 16**2
                        * float(np.float32(result["data"]["train_input_rms_float64"]))
                        ** 2
                    ),
                    rel=1e-12,
                )
            ids = data["sentinel_global_indices"]
            np.testing.assert_array_equal(ids, [0, 31, 32, 47])
            np.testing.assert_array_equal(
                data["sentinel_input"], state32[ids // 4, ids % 4]
            )
            np.testing.assert_array_equal(
                data["sentinel_target"], state32[ids // 4, ids % 4 + 1]
            )
            for key, prediction in (
                ("learned_sse", "sentinel_next"),
                ("raw_sse", "sentinel_raw"),
            ):
                error = data[prediction].astype(np.float64) - data[
                    "sentinel_target"
                ].astype(np.float64)
                np.testing.assert_allclose(
                    data[key][ids],
                    np.sum(error**2, axis=(1, 2)),
                    rtol=1e-12,
                    atol=1e-12,
                )
            residual = data["sentinel_raw"].astype(np.float64) - data[
                "sentinel_next"
            ].astype(np.float64)
            np.testing.assert_allclose(
                data["projection_sse"][ids],
                np.sum(residual**2, axis=(1, 2)),
                rtol=1e-12,
                atol=1e-12,
            )
    manifest = p._read(output / "artifact_manifest.json")
    assert manifest["sources"] == result["sources_before"] == result["sources_after"]
    assert {path.name for path in output.iterdir()} == set(manifest["artifacts"]) | {
        "artifact_manifest.json"
    }
    assert all(
        p._hash(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )
    expected_terminals = {8, 12} if force_extension else {8}
    assert {
        int(update) for update in result["terminal_checkpoints"]
    } == expected_terminals
    for update, checkpoint in result["terminal_checkpoints"].items():
        assert checkpoint["sha256"] == p._hash(output / checkpoint["file"])
        assert checkpoint["file"] == f"terminal_{int(update):06d}.pt"
    assert not any("rollout" in path.name for path in output.iterdir())
    assert result["decision"]["extend"] is False
    final_checkpoint = output / f"terminal_{terminal:06d}.pt"
    model = p.PeriodicVorticityPCNO(
        **result["model_config"], train_scale=result["data"]["train_input_rms_float64"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = p.EpochSampler(32, batch_size=8, seed=999)
    assert (
        p.load_clean_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            sampler,
            identity=result["checkpoint_identity"],
        )
        == terminal
    )
    model.eval()
    with np.load(output / f"teacher_{terminal:06d}.npz", allow_pickle=False) as data:
        with torch.no_grad():
            replay = model(torch.from_numpy(data["sentinel_input"].copy()))
        np.testing.assert_allclose(
            replay["raw_next"].numpy(), data["sentinel_raw"], rtol=1e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            replay["next_state"].numpy(), data["sentinel_next"], rtol=1e-6, atol=2e-6
        )
    with pytest.raises(FileExistsError):
        p.run_clean(parent, source, output, "cpu", unit_fixture=True)


@pytest.mark.parametrize("corruption", ("loss", "gradient", "evaluation"))
def test_full_clean_nonfinite_failure_has_no_optimizer_step_or_promotion(
    population, tmp_path, monkeypatch, corruption
):
    parent, source, _, _ = population
    original = p.PeriodicVorticityPCNO.forward
    steps = 0
    original_step = torch.optim.Adam.step

    def counted_step(self, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(self, *args, **kwargs)

    def bad(self, inputs):
        result = original(self, inputs)
        if (corruption == "evaluation" and not self.training) or (
            self.training and corruption == "loss"
        ):
            result["next_state"] = result["next_state"] * float("nan")
        elif self.training and corruption == "gradient":
            result["next_state"].register_hook(lambda value: value * float("nan"))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    monkeypatch.setattr(p.PeriodicVorticityPCNO, "forward", bad)
    output = tmp_path / "nonfinite_clean"
    result = p.run_clean(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "failed" and result["updates_completed"] == 0
    assert steps == 0 and result["accepted_checkpoint"] is None
    assert result["terminal_checkpoints"] == {}
    assert result["source_stable"] and result["parent_evidence_stable"]
    manifest = p._read(output / "artifact_manifest.json")
    assert all(
        p._hash(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )
    assert "NaN" not in (output / "result.json").read_text()


@pytest.mark.parametrize("corruption", ("source", "parent"))
def test_full_clean_detects_end_of_run_binding_mutation(
    population, tmp_path, monkeypatch, corruption
):
    parent, source, _, _ = population
    real_decision = p.clean_decision

    def no_extension(previous, current, competence, *, extended=False):
        result = real_decision(previous, current, competence, extended=extended)
        result.update(extend=False, accepted_terminal=False)
        return result

    monkeypatch.setattr(p, "clean_decision", no_extension)
    if corruption == "source":
        original = p._sources
        calls = 0

        def changed_sources():
            nonlocal calls
            calls += 1
            values = original()
            if calls > 1:
                values[p.SOURCE_PATHS[0]] = "0" * 64
            return values

        monkeypatch.setattr(p, "_sources", changed_sources)
    else:
        original = p._teacher_evaluation

        def changed_parent(model, data, output, update, bands, **kwargs):
            result = original(model, data, output, update, bands, **kwargs)
            if update == 2:
                path = parent / "result.json"
                path.write_text(path.read_text() + "\n")
            return result

        monkeypatch.setattr(p, "_teacher_evaluation", changed_parent)
    output = tmp_path / "mutated_clean"
    result = p.run_clean(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "invalid_" + corruption
    assert result["accepted_checkpoint"] is None
    assert result["updates_completed"] == 8
    manifest = p._read(output / "artifact_manifest.json")
    assert all(
        p._hash(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )


def test_clean_cli_is_explicit_and_reports_failed_run(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean",
            "--phase",
            "clean",
            "--parent",
            "C",
            "--parent-source",
            "source",
            "--output",
            "fresh",
            "--device",
            "cuda",
        ],
    )
    calls = []

    def failed(*args):
        calls.append(args)
        return {"status": "failed"}

    monkeypatch.setattr(p, "run_clean", failed)
    monkeypatch.setattr(
        p, "run_resource", lambda *args: pytest.fail("wrong default resource route")
    )
    with pytest.raises(SystemExit) as error:
        p.main()
    assert error.value.code == 1 and len(calls) == 1
    assert [str(value) for value in calls[0]] == ["C", "source", "fresh", "cuda"]


def test_full_clean_parent_preflight_failure_records_no_training(population, tmp_path):
    parent, source, _, _ = population
    (parent / "result.json").write_text("tampered fixture manifest member")
    output = tmp_path / "invalid_clean_parent"
    result = p.run_clean(parent, source, output, "cpu", unit_fixture=True)
    assert result["status"] == "invalid_parent"
    assert result["scientific_training"] is False
    assert result["updates_completed"] == 0 and result["teacher_forward_calls"] == 0
    assert result["evaluations"] == [] and result["accepted_checkpoint"] is None
    assert result["terminal_checkpoints"] == {}
    assert not (output / "updates.jsonl").exists()
    assert not (output / "latest_checkpoint.pt").exists()
    manifest = p._read(output / "artifact_manifest.json")
    assert all(
        p._hash(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )
