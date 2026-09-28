from __future__ import annotations

import copy
import io
import json
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import fit_kolmogorov_paired as paired
from scripts.time_dependent_no.fit_kolmogorov_clean import EpochSampler


class TinyModel(torch.nn.Module):
    """Synthetic FP32 transition with an intentionally nonzero trained head."""

    def __init__(self, scale=2.5):
        super().__init__()
        self.head = torch.nn.Linear(1, 1)
        self.register_buffer("train_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("marker", torch.tensor(0.125, dtype=torch.float32))
        with torch.no_grad():
            self.head.weight.fill_(0.6)
            self.head.bias.fill_(-0.15)

    def forward(self, inputs):
        next_state = self.head(inputs.unsqueeze(-1)).squeeze(-1)
        return {"next_state": next_state, "raw_next": next_state + self.marker}


def _receipt():
    return {
        "checkpoint_load_attempts": 0,
        "checkpoint_loads": 0,
        "model_forward_calls": 0,
        "optimizer_step_attempts": 0,
        "updates_completed": 0,
    }


def _bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _parent_checkpoint():
    model = TinyModel()
    identity = {
        "run_id": "SYNTHETIC_CLEAN32__UNIT_FIXTURE",
        "train_scale_model_float32": float(model.train_scale),
        "source_sha256": "synthetic-not-a-scientific-checkpoint",
    }
    value = {
        "identity": identity,
        "update": 12288,
        "schedule_position": 12288,
        "model": copy.deepcopy(model.state_dict()),
        "model_training": False,
        # These intentionally cannot initialize an optimizer, sampler or RNG.
        "optimizer": {"must_not_restore": True},
        "sampler": {"must_not_restore": True},
        "rng": {"must_not_restore": True},
    }
    parent = {
        "checkpoint_identity": copy.deepcopy(identity),
        "updates_completed": 12288,
    }
    return value, parent


@pytest.fixture
def data():
    states = np.arange(2 * 5 * 16 * 16, dtype=np.float32).reshape(2, 5, 16, 16) / 1024
    anchors = np.asarray([[0, 1], [0, 3], [1, 1], [1, 3]], dtype=np.int64)
    clean = states[anchors[:, 0], anchors[:, 1]].copy()
    inputs = np.stack(
        [clean, clean + np.float32(0.125), clean - np.float32(0.125)], axis=1
    )
    targets = np.stack(
        [clean + np.float32(0.25), clean + np.float32(0.75), clean - np.float32(0.5)],
        axis=1,
    )
    return paired.PairedTrainingData(
        states, inputs, targets, anchors, 2.5, {"fixture": "synthetic arrays only"}
    )


def test_clean_transition_mapping_and_cpu_fp32_identity(data):
    assert data.clean_size == 8 and data.bank_size == 8
    ids = np.asarray([7, 0, 3, 3], dtype=np.int64)
    before = data.states.copy()
    inputs, targets = data.clean_batch(ids)
    for value in (inputs, targets):
        assert value.device.type == "cpu" and value.dtype == torch.float32
    np.testing.assert_array_equal(inputs.numpy(), data.states[ids // 4, ids % 4])
    np.testing.assert_array_equal(targets.numpy(), data.states[ids // 4, ids % 4 + 1])
    inputs[0].fill_(999)
    np.testing.assert_array_equal(data.states, before)


@pytest.mark.parametrize("arm", ["clean_continuation", "recovery", "dynamics"])
def test_signed_column_identity_preserves_duplicate_control_rows(data, arm):
    assert paired.ARM_NAMES == ("clean_continuation", "recovery", "dynamics")
    ids = np.asarray([0, 1, 7, 2, 2], dtype=np.int64)
    anchors, signs = ids // 2, ids % 2 + 1
    before = (data.bank_inputs.copy(), data.bank_targets.copy())
    inputs, targets = data.bank_batch(ids, arm)
    input_slots = np.zeros_like(signs) if arm == "clean_continuation" else signs
    target_slots = signs if arm == "dynamics" else np.zeros_like(signs)
    np.testing.assert_array_equal(
        inputs.numpy(), data.bank_inputs[anchors, input_slots]
    )
    np.testing.assert_array_equal(
        targets.numpy(), data.bank_targets[anchors, target_slots]
    )
    assert len(inputs) == len(targets) == len(ids)
    assert inputs.dtype == targets.dtype == torch.float32
    assert inputs.device.type == targets.device.type == "cpu"
    if arm == "clean_continuation":
        torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
    else:
        assert not torch.equal(inputs[0], inputs[1])
    if arm != "dynamics":
        torch.testing.assert_close(targets[0], targets[1], rtol=0, atol=0)
    else:
        assert not torch.equal(targets[0], targets[1])
    inputs[0].fill_(999)
    targets[0].fill_(999)
    np.testing.assert_array_equal(data.bank_inputs, before[0])
    np.testing.assert_array_equal(data.bank_targets, before[1])


@pytest.mark.parametrize("ids", [[], [-1], [8], [0.0], [True], [[0]]])
def test_invalid_transition_and_signed_ids_rejected(data, ids):
    with pytest.raises((ValueError, TypeError, IndexError)):
        data.clean_batch(ids)
    with pytest.raises((ValueError, TypeError, IndexError)):
        data.bank_batch(ids, "recovery")


def test_unknown_arm_rejected(data):
    with pytest.raises(ValueError):
        data.bank_batch([0], "best_available_target")


def test_full_budget_independent_sampler_tapes_match_across_three_arms():
    baseline = None
    for _ in paired.ARM_NAMES:
        clean = EpochSampler(16384, batch_size=8, seed=17)
        signed = EpochSampler(1024, batch_size=8, seed=1701)
        tapes = tuple(
            np.stack([sampler.next_indices() for _ in range(4096)])
            for sampler in (clean, signed)
        )
        for tape, size, repeats in zip(tapes, (16384, 1024), (2, 32), strict=True):
            assert tape.shape == (4096, 8)
            np.testing.assert_array_equal(
                np.bincount(tape.ravel(), minlength=size), np.full(size, repeats)
            )
            for epoch in tape.reshape(repeats, size):
                np.testing.assert_array_equal(np.sort(epoch), np.arange(size))
        if baseline is not None:
            for actual, expected in zip(tapes, baseline, strict=True):
                np.testing.assert_array_equal(actual, expected)
        baseline = tapes
        assert (
            clean.state_dict()["epoch"] == 1 and clean.state_dict()["cursor"] == 16384
        )
        assert (
            signed.state_dict()["epoch"] == 31 and signed.state_dict()["cursor"] == 1024
        )
        assert not np.array_equal(tapes[0][:128] % 1024, tapes[1][:128])


def test_model_only_checkpoint_load_preserves_head_buffers_and_rng():
    value, parent = _parent_checkpoint()
    captured = _bytes(value)
    model = TinyModel(scale=99.0)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    model.train()
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    receipt = _receipt()
    paired.load_parent_model(model, captured, parent, receipt)
    assert receipt["checkpoint_load_attempts"] == receipt["checkpoint_loads"] == 1
    assert model.training
    for key, expected in value["model"].items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert torch.count_nonzero(model.head.weight) and torch.count_nonzero(
        model.head.bias
    )
    assert (
        float(model.train_scale)
        == parent["checkpoint_identity"]["train_scale_model_float32"]
    )
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    optimizer = paired.fresh_optimizer(model)
    assert type(optimizer) is torch.optim.Adam and len(optimizer.state) == 0
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-4 and group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0.0
    assert receipt["optimizer_step_attempts"] == receipt["updates_completed"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "identity",
        "update",
        "schedule",
        "missing_weight",
        "extra_weight",
        "shape",
        "normalizer",
        "nonfinite_normalizer",
    ],
)
def test_checkpoint_identity_schedule_strict_model_and_normalizer_rejection(corruption):
    value, parent = _parent_checkpoint()
    if corruption == "identity":
        value["identity"]["run_id"] = "wrong"
    elif corruption == "update":
        value["update"] += 1
        value["schedule_position"] += 1
    elif corruption == "schedule":
        value["schedule_position"] -= 1
    elif corruption == "missing_weight":
        del value["model"]["head.weight"]
    elif corruption == "extra_weight":
        value["model"]["unknown.weight"] = torch.ones(1)
    elif corruption == "shape":
        value["model"]["head.weight"] = torch.ones(2, 1)
    else:
        value["model"]["train_scale"] = torch.tensor(
            float("nan") if corruption == "nonfinite_normalizer" else 9.0
        )
    receipt = _receipt()
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        paired.load_parent_model(TinyModel(), _bytes(value), parent, receipt)
    assert receipt["checkpoint_load_attempts"] == 1 and receipt["checkpoint_loads"] == 0


def _batches():
    inputs = torch.linspace(-0.7, 1.3, 32, dtype=torch.float32).reshape(2, 4, 4)
    return (inputs, 0.4 * inputs + 0.2), (inputs + 0.35, 0.8 * inputs - 0.1)


def test_sequential_half_losses_match_joint_gradients_and_adam_updates():
    sequential, joint = TinyModel(), TinyModel()
    optimizer = paired.fresh_optimizer(sequential)
    oracle_optimizer = torch.optim.Adam(
        joint.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    receipt = _receipt()
    clean_batch, bank_batch = _batches()
    before = tuple(
        value.clone() for batch in (clean_batch, bank_batch) for value in batch
    )
    for update in range(1, 4):
        actual = paired.paired_update(
            sequential,
            optimizer,
            clean_batch,
            bank_batch,
            torch.device("cpu"),
            float("inf"),
            receipt,
        )
        oracle_optimizer.zero_grad(set_to_none=True)
        losses = [
            torch.mean((joint(inputs)["next_state"] - targets) ** 2)
            / joint.train_scale.square()
            for inputs, targets in (clean_batch, bank_batch)
        ]
        expected_loss = 0.5 * losses[0] + 0.5 * losses[1]
        expected_loss.backward()
        expected_gradient_l2 = torch.sqrt(
            sum(
                torch.sum(parameter.grad.double().square())
                for parameter in joint.parameters()
            )
        ).item()
        assert actual["clean_mse_scaled"] == pytest.approx(losses[0].item(), rel=2e-6)
        assert actual["bank_mse_scaled"] == pytest.approx(losses[1].item(), rel=2e-6)
        assert actual["loss"] == pytest.approx(expected_loss.item(), rel=2e-6)
        assert actual["gradient_l2"] == pytest.approx(expected_gradient_l2, rel=2e-6)
        oracle_optimizer.step()
        for left, right in zip(
            sequential.parameters(), joint.parameters(), strict=True
        ):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-6, atol=1e-8)
        for index, state in optimizer.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value,
                    oracle_optimizer.state_dict()["state"][index][key],
                    rtol=2e-6,
                    atol=1e-8,
                )
        assert receipt["model_forward_calls"] == 2 * update
        assert (
            receipt["optimizer_step_attempts"] == receipt["updates_completed"] == update
        )
    for actual, expected in zip(
        (value for batch in (clean_batch, bank_batch) for value in batch),
        before,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_backward_order_and_state_scale_without_clipping(monkeypatch):
    events = []

    class Traced(TinyModel):
        def forward(self, inputs):
            assert inputs.dtype == torch.float32 and inputs.device.type == "cpu"
            events.append("forward")
            return super().forward(inputs)

    model = Traced(scale=4.0)
    optimizer = paired.fresh_optimizer(model)
    original_backward, original_step = torch.Tensor.backward, optimizer.step

    def backward(loss, *args, **kwargs):
        events.append("backward")
        return original_backward(loss, *args, **kwargs)

    def step(*args, **kwargs):
        events.append("step")
        return original_step(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("AMP or clipping is not part of the paired update")

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    monkeypatch.setattr(optimizer, "step", step)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", forbidden)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", forbidden)
    monkeypatch.setattr(torch, "autocast", forbidden)
    clean_batch, bank_batch = _batches()
    with torch.no_grad():
        expected = [
            torch.mean(
                (TinyModel(scale=4.0)(inputs)["next_state"] - targets) ** 2
            ).item()
            / 16
            for inputs, targets in (clean_batch, bank_batch)
        ]
    result = paired.paired_update(
        model,
        optimizer,
        clean_batch,
        bank_batch,
        torch.device("cpu"),
        float("inf"),
        _receipt(),
    )
    assert events == ["forward", "backward", "forward", "backward", "step"]
    assert result["clean_mse_scaled"] == pytest.approx(expected[0], rel=2e-6)
    assert result["bank_mse_scaled"] == pytest.approx(expected[1], rel=2e-6)


@pytest.mark.parametrize("branch,part", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nonfinite_batch_rejected_before_optimizer_step(branch, part):
    batches = [list(batch) for batch in _batches()]
    batches[branch][part] = batches[branch][part].clone()
    batches[branch][part][0, 0, 0] = float("nan")
    model = TinyModel()
    before = copy.deepcopy(model.state_dict())
    optimizer, receipt = paired.fresh_optimizer(model), _receipt()
    with pytest.raises((ValueError, RuntimeError, ArithmeticError)):
        paired.paired_update(
            model,
            optimizer,
            batches[0],
            batches[1],
            torch.device("cpu"),
            float("inf"),
            receipt,
        )
    assert receipt["optimizer_step_attempts"] == receipt["updates_completed"] == 0
    assert len(optimizer.state) == 0
    for key, expected in before.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["loss", "gradient", "deadline"])
def test_second_branch_failure_never_steps_partial_first_branch_gradient(
    monkeypatch, failure
):
    clock = [0.0]
    monkeypatch.setattr(paired.core, "perf_counter", lambda: clock[0])

    class Failing(TinyModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            output = super().forward(inputs)["next_state"]
            if self.calls == 2:
                if failure == "loss":
                    output = output * float("nan")
                elif failure == "gradient":
                    output.register_hook(
                        lambda gradient: torch.full_like(gradient, float("nan"))
                    )
                else:
                    clock[0] = 101.0
            return {"next_state": output}

    model = Failing()
    before = copy.deepcopy(model.state_dict())
    optimizer, receipt = paired.fresh_optimizer(model), _receipt()

    def forbidden_step(*args, **kwargs):
        raise AssertionError("partial two-branch update reached optimizer")

    monkeypatch.setattr(optimizer, "step", forbidden_step)
    clean_batch, bank_batch = _batches()
    with pytest.raises((ValueError, RuntimeError, ArithmeticError, TimeoutError)):
        paired.paired_update(
            model,
            optimizer,
            clean_batch,
            bank_batch,
            torch.device("cpu"),
            100.0,
            receipt,
        )
    assert model.calls == receipt["model_forward_calls"] == 2
    assert receipt["optimizer_step_attempts"] == receipt["updates_completed"] == 0
    assert len(optimizer.state) == 0
    for key, expected in before.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


def test_expired_update_deadline_makes_no_forward_or_step(monkeypatch):
    monkeypatch.setattr(paired.core, "perf_counter", lambda: 2.0)
    model, receipt = TinyModel(), _receipt()
    optimizer = paired.fresh_optimizer(model)
    clean_batch, bank_batch = _batches()
    with pytest.raises((RuntimeError, TimeoutError)):
        paired.paired_update(
            model, optimizer, clean_batch, bank_batch, torch.device("cpu"), 1.0, receipt
        )
    assert (
        receipt["model_forward_calls"]
        == receipt["optimizer_step_attempts"]
        == receipt["updates_completed"]
        == 0
    )


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seal_disk(fixture, mutation=None):
    """Rebind only this test's synthetic metadata and in-memory checkpoint."""
    hashes = {}
    for role, root in fixture.roots.items():
        result = copy.deepcopy(fixture.results[role])
        if role == "extension":
            result["parent_manifest_sha256"] = hashes["population"]
        elif role == "bank":
            result["input_files"] = {
                r + "/artifact_manifest.json": hashes[r]
                for r in ("population", "extension")
            }
            for row in result["anchors"]:
                row["sha256"] = paired.clean._hash(root / row["file"])
        elif role == "parent":
            identity = {
                "run_id": result["run_id"],
                "input_manifests": [hashes["population"], hashes["extension"]],
                "population_index": fixture.index,
                "sources": fixture.closed_sources,
                "model_config": result["model_config"],
                "train_scale_float64": 2.50000003,
                "train_scale_model_float32": 2.5,
            }
            payload = {
                **fixture.checkpoint,
                "identity": identity,
                "update": 12,
                "schedule_position": 12,
            }
            (root / "terminal_000012.pt").write_bytes(_bytes(payload))
            result["checkpoint_identity"] = identity
            result["terminal_checkpoint"] = {
                "file": "terminal_000012.pt",
                "update": 12,
                "sha256": paired.clean._hash(root / "terminal_000012.pt"),
                "replay": {"passed": True, "file": "teacher_archive.npz"},
            }
        if mutation and mutation[0] == role:
            target = result
            for key in mutation[1][:-1]:
                target = target[key]
            target[mutation[1][-1]] = mutation[2]
        paired.core._json(root / "result.json", result)
        manifest = {
            "run_id": result["run_id"],
            "source_stable": True,
            "sources": fixture.closed_sources,
            "artifacts": {
                path.name: paired.clean._hash(path)
                for path in root.iterdir()
                if path.name != "artifact_manifest.json"
            },
        }
        if role != "population":
            manifest["status"] = "completed"
        if role == "extension":
            manifest["parent_manifest_sha256"] = hashes["population"]
        paired.core._json(root / "artifact_manifest.json", manifest)
        hashes[role] = paired.clean._hash(root / "artifact_manifest.json")


@pytest.fixture
def disk_fixture(data, tmp_path, monkeypatch):
    roots = {role: tmp_path / role for role in paired.PINS}
    for root in roots.values():
        root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture_source.py").write_bytes(b"# Synthetic source binding.\n")
    monkeypatch.setattr(paired, "REPO_ROOT", source)
    monkeypatch.setattr(paired, "SOURCE_PATHS", ("fixture_source.py",))
    seeds = (2026090611, 2026090801)
    index = [
        {
            "seed": seed,
            "role": "train",
            "packet": packet,
            "case_file": f"case_{seed}.json",
        }
        for seed, packet in zip(seeds, ("parent_C", "this_packet"), strict=True)
    ]
    config = asdict(
        paired.core.KolmogorovReferenceConfig(
            resolution=16, viscosity=0.01, macro_dt=0.05
        )
    )
    results = {
        role: {
            "run_id": pin[0] + "__UNIT_FIXTURE",
            "status": "completed",
            "source_stable": True,
        }
        for role, pin in paired.PINS.items()
    }
    results["population"]["engineering_gates_pass"] = True
    results["extension"].update(engineering_gates_pass=True, population_index=index)
    native = data.states.astype(np.float64) + 1e-8
    for i, (role, seed) in enumerate(
        zip(("population", "extension"), seeds, strict=True)
    ):
        root = roots[role]
        name = f"trajectory_{seed}.npz"
        np.savez_compressed(
            root / name, steps=np.arange(5, dtype=np.int64), states=native[i]
        )
        paired.core._json(
            root / f"case_{seed}.json",
            {
                "seed": seed,
                "role": "train",
                "status": "completed",
                "config": config,
                "blocks": [
                    {
                        "file": name,
                        "first_step": 0,
                        "last_step": 4,
                        "sha256": paired.clean._hash(root / name),
                    }
                ],
            },
        )
        for name in ("validation_2026090621.npz", "protected_do_not_read.npz"):
            (root / name).write_bytes(
                b"Synthetic access-boundary sentinel, not an array."
            )
    arrays, rows = [], []
    for a, (trajectory, step) in enumerate(data.anchor_indices):
        raw = data.bank_inputs[a].copy()
        raw[0] = native[trajectory, step].astype(np.float32)
        values = {
            "raw_inputs": raw,
            **{
                f"A_{j}": data.bank_targets[a, j].astype(np.float64) + 1e-8
                for j in range(3)
            },
        }
        # These validly stored diagnostic members must never be decoded by fit.
        values.update(
            {
                f"{level}_{j}": np.zeros((n, n), dtype=np.float64)
                for level, n in (("B", 16), ("C", 32), ("D", 32))
                for j in range(3)
            }
        )
        name = f"bank_{seeds[trajectory]}_{step:03d}.npz"
        np.savez_compressed(roots["bank"] / name, **values)
        rows.append(
            {
                "seed": seeds[trajectory],
                "role": "train",
                "input_step": int(step),
                "output_step": int(step + 1),
                "status": "completed",
                "file": name,
                "array_hashes": {
                    key: paired.core.array_hash(value) for key, value in values.items()
                },
            }
        )
        arrays.append(values)
    results["bank"].update(
        phase="generate",
        qualification_passed=True,
        inputs_stable=True,
        solver_calls_completed=33,
        solver_call_attempts=33,
        seeds=list(seeds),
        input_steps=[1, 3],
        signs=[1, -1],
        expected_displaced_rows=8,
        validation_arrays_read=False,
        protected_access=False,
        model_calls=0,
        checkpoint_loads=0,
        optimization_steps=0,
        train_scale=2.5,
        config=config,
        anchors=rows,
        target_map="Synthetic A-only target fixture",
    )
    results["parent"].update(
        phase="fit",
        updates_completed=12,
        input_evidence_stable=True,
        population_index=index,
        model_config={
            "resolution": 16,
            "width": 4,
            "depth": 1,
            "modes": 2,
            "fc_dim": 8,
        },
    )
    (roots["parent"] / "teacher_archive.npz").write_bytes(
        b"Synthetic mixed-role archive: never read."
    )
    checkpoint, _ = _parent_checkpoint()
    fixture = SimpleNamespace(
        roots=roots,
        source=source,
        results=results,
        index=index,
        native=native,
        bank_arrays=arrays,
        checkpoint=checkpoint,
        models=[],
        decoded=[],
        opened=[],
        closed_sources={"closed_fixture.py": "synthetic-closed-source"},
        monkeypatch=monkeypatch,
    )
    _seal_disk(fixture)

    def construct(**kwargs):
        model = TinyModel(scale=kwargs["train_scale"])
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        fixture.models.append(model)
        return model

    monkeypatch.setattr(paired.clean, "PeriodicVorticityPCNO", construct)
    return fixture


@contextmanager
def _read_guards(fixture):
    original_open, original_column = Path.open, np.lib.npyio.NpzFile.__getitem__

    def open_path(path, *args, **kwargs):
        assert not path.name.startswith(("validation_", "protected_", "teacher_")), (
            "forbidden population/archive read"
        )
        fixture.opened.append(path)
        return original_open(path, *args, **kwargs)

    def column(archive, key):
        assert not str(key).startswith(("B_", "C_", "D_")), (
            "refined diagnostic decoded as training data"
        )
        fixture.decoded.append(key)
        return original_column(archive, key)

    with fixture.monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_path)
        patch.setattr(np.lib.npyio.NpzFile, "__getitem__", column)
        yield


def _load_disk(fixture):
    bindings = {}
    with _read_guards(fixture):
        manifests, results = paired.load_metadata(
            fixture.roots, bindings, float("inf"), {}, unit_fixture=True
        )
        data = paired.load_training_data(
            fixture.roots, manifests, results, bindings, float("inf"), unit_fixture=True
        )
    return data, bindings


def _run_disk(fixture, output, phase="fit", arm="recovery", **kwargs):
    with _read_guards(fixture):
        return paired.run(
            *(fixture.roots[role] for role in paired.PINS),
            output,
            phase=phase,
            arm=arm,
            device_name="cpu",
            unit_fixture=True,
            **kwargs,
        )


def _assert_packet(output, result):
    assert _read(output / "result.json") == result
    manifest = _read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert set(manifest["artifacts"]) == {path.name for path in output.iterdir()} - {
        "artifact_manifest.json"
    }
    for name, digest in manifest["artifacts"].items():
        assert paired.clean._hash(output / name) == digest


def test_synthetic_loader_reads_complete_train_store_and_only_raw_a_columns(
    disk_fixture,
):
    data, bindings = _load_disk(disk_fixture)
    np.testing.assert_array_equal(data.states, disk_fixture.native.astype(np.float32))
    np.testing.assert_array_equal(
        data.bank_inputs, np.stack([a["raw_inputs"] for a in disk_fixture.bank_arrays])
    )
    np.testing.assert_array_equal(
        data.bank_targets,
        np.stack(
            [
                np.stack([a[f"A_{j}"] for j in range(3)])
                for a in disk_fixture.bank_arrays
            ]
        ).astype(np.float32),
    )
    np.testing.assert_array_equal(data.anchor_indices, [[0, 1], [0, 3], [1, 1], [1, 3]])
    assert data.clean_size == data.bank_size == 8 and data.train_scale == 2.5
    assert data.train_scale != pytest.approx(np.sqrt(np.mean(data.states**2)))
    assert set(disk_fixture.decoded) == {
        "states",
        "steps",
        "raw_inputs",
        "A_0",
        "A_1",
        "A_2",
    }
    assert len(bindings) == 16 and all("validation" not in name for _, name in bindings)
    assert disk_fixture.models == []
    for key, value in (
        ("state_store_sha256", data.states),
        ("bank_input_sha256", data.bank_inputs),
        ("bank_target_fp32_sha256", data.bank_targets),
        ("anchor_indices_sha256", data.anchor_indices),
    ):
        assert data.metadata[key] == paired.core.array_hash(value)


def test_validate_hashes_checkpoint_without_construction_or_deserialization(
    disk_fixture, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "validate cannot construct a model or deserialize checkpoint"
        )

    monkeypatch.setattr(paired.clean, "PeriodicVorticityPCNO", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    result = _run_disk(disk_fixture, tmp_path / "validated", phase="validate")
    assert result["status"] == "completed", result.get("error")
    assert (
        result["inputs_validated"]
        and result["inputs_stable"]
        and result["source_stable"]
    )
    for key in (
        "model_construction_attempts",
        "checkpoint_load_attempts",
        "checkpoint_loads",
        "model_forward_calls",
        "optimizer_step_attempts",
        "updates_completed",
        "expected_updates",
    ):
        assert result[key] == 0
    assert not result["model_created"] and result["terminal_checkpoint"] is None
    assert "parent/terminal_000012.pt" in result["input_files"]
    assert _read(tmp_path / "validated" / "history.json") == []
    _assert_packet(tmp_path / "validated", result)


def test_resource_updates_are_fresh_and_leave_no_checkpoint(disk_fixture, tmp_path):
    result = _run_disk(disk_fixture, tmp_path / "resource", phase="resource")
    assert result["status"] == "completed", result.get("error")
    assert result["updates_completed"] == result["expected_updates"] == 2
    assert result["model_forward_calls"] == 4 and result["checkpoint_loads"] == 1
    assert (
        result["optimizer_step_attempts"] == 2 and result["terminal_checkpoint"] is None
    )
    assert not list((tmp_path / "resource").glob("*.pt"))
    assert not (tmp_path / "resource" / "terminal_replay.npz").exists()
    assert len(_read(tmp_path / "resource" / "history.json")) == 2
    _assert_packet(tmp_path / "resource", result)


@pytest.mark.parametrize("arm", ["clean_continuation", "recovery", "dynamics"])
def test_full_fit_same_parent_target_views_tapes_and_train_only_replay(
    disk_fixture, tmp_path, arm
):
    paths = [
        path
        for root in (*disk_fixture.roots.values(), disk_fixture.source)
        for path in root.iterdir()
    ]
    before = {path: paired.clean._hash(path) for path in paths}
    data, _ = _load_disk(disk_fixture)
    output = tmp_path / arm
    result = _run_disk(disk_fixture, output, arm=arm)
    assert result["status"] == "completed", result.get("error")
    assert (
        result["updates_completed"]
        == result["expected_updates"]
        == result["optimizer_step_attempts"]
        == 4
    )
    assert (
        result["model_forward_calls"] == 10
        and result["checkpoint_loads"] == result["checkpoint_load_attempts"] == 2
    )
    assert result["model_construction_attempts"] == len(disk_fixture.models) == 1
    assert all(
        result[key] is False
        for key in ("validation_arrays_read", "protected_access", "new_rollouts")
    )
    assert (
        result["solver_calls"] == 0
        and result["inputs_stable"]
        and result["source_stable"]
    )
    history = _read(output / "history.json")
    assert [row["update"] for row in history] == [1, 2, 3, 4]
    with np.load(output / "sampling.npz", allow_pickle=False) as saved:
        tapes = {key: saved[key] for key in saved.files}
    expected_samplers = {
        "clean": EpochSampler(8, 8, 17),
        "bank": EpochSampler(8, 8, 1701),
    }
    oracle = TinyModel()
    optimizer = torch.optim.Adam(
        oracle.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    for update, row in enumerate(history):
        for name, sampler in expected_samplers.items():
            np.testing.assert_array_equal(tapes[name][update], sampler.next_indices())
        optimizer.zero_grad(set_to_none=True)
        losses = [
            torch.mean(((oracle(x)["next_state"] - y) / oracle.train_scale) ** 2)
            for x, y in (
                data.clean_batch(tapes["clean"][update]),
                data.bank_batch(tapes["bank"][update], arm),
            )
        ]
        loss = 0.5 * (losses[0] + losses[1])
        assert row["loss"] == pytest.approx(loss.item(), rel=2e-6)
        loss.backward()
        optimizer.step()
    terminal = result["terminal_checkpoint"]
    assert terminal["update"] == 4 and terminal["replay"]["passed"] is True
    checkpoint = torch.load(
        io.BytesIO((output / terminal["file"]).read_bytes()),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["schema"] == "kolmogorov_paired_adaptation_v1"
    assert (
        checkpoint["parent_update"] == 12
        and checkpoint["identity"] == result["checkpoint_identity"]
    )
    for key, expected in oracle.state_dict().items():
        torch.testing.assert_close(
            checkpoint["model"][key], expected, rtol=2e-6, atol=1e-7
        )
    for name, expected_sampler in expected_samplers.items():
        restored = EpochSampler(8, 8, 999)
        restored.load_state_dict(checkpoint["samplers"][name])
        np.testing.assert_array_equal(
            restored.next_indices(), expected_sampler.next_indices()
        )
    with np.load(output / "terminal_replay.npz", allow_pickle=False) as replay:
        inputs, _ = data.clean_batch(np.asarray([0, 7]))
        np.testing.assert_array_equal(replay["inputs"], inputs.numpy())
        with torch.no_grad():
            predicted = disk_fixture.models[0](inputs)
        for key in ("raw_next", "next_state"):
            np.testing.assert_array_equal(replay[key], predicted[key].numpy())
    assert before == {path: paired.clean._hash(path) for path in paths}
    _assert_packet(output, result)


@pytest.mark.parametrize(
    "role,path,value",
    [
        ("bank", ("phase",), "validate"),
        ("bank", ("qualification_passed",), False),
        ("bank", ("solver_call_attempts",), 32),
        ("bank", ("validation_arrays_read",), True),
        ("extension", ("population_index", 0, "role"), "development"),
        ("parent", ("updates_completed",), 13),
        ("parent", ("input_evidence_stable",), False),
    ],
)
def test_metadata_failures_precede_model_access(
    disk_fixture, tmp_path, role, path, value
):
    _seal_disk(disk_fixture, (role, path, value))
    result = _run_disk(disk_fixture, tmp_path / "invalid_metadata")
    assert result["status"] == "failed" and result["stage"] == "metadata"
    assert not result["model_created"] and result["checkpoint_load_attempts"] == 0
    assert result["model_forward_calls"] == result["updates_completed"] == 0
    _assert_packet(tmp_path / "invalid_metadata", result)


@pytest.mark.parametrize(
    "corruption",
    [
        "role",
        "missing_first",
        "missing_last",
        "block_bytes",
        "checkpoint_bytes",
        "bank_dtype",
        "bank_clean_input",
    ],
)
def test_loader_and_checkpoint_binding_rejection_before_deserialization(
    disk_fixture, tmp_path, corruption
):
    root = disk_fixture.roots["population"]
    case_path = root / "case_2026090611.json"
    case = _read(case_path)
    if corruption == "role":
        case["role"] = "development"
        paired.core._json(case_path, case)
    elif corruption in ("missing_first", "missing_last"):
        first, last = (1, 4) if corruption == "missing_first" else (0, 3)
        block_path = root / case["blocks"][0]["file"]
        np.savez_compressed(
            block_path,
            steps=np.arange(first, last + 1, dtype=np.int64),
            states=disk_fixture.native[0, first : last + 1],
        )
        case["blocks"][0].update(
            first_step=first, last_step=last, sha256=paired.clean._hash(block_path)
        )
        paired.core._json(case_path, case)
    elif corruption in ("bank_dtype", "bank_clean_input"):
        arrays = disk_fixture.bank_arrays[0]
        if corruption == "bank_dtype":
            arrays["A_1"] = arrays["A_1"].astype(np.float32)
        else:
            arrays["raw_inputs"][0] += 0.25
        row = disk_fixture.results["bank"]["anchors"][0]
        row["array_hashes"] = {
            key: paired.core.array_hash(value) for key, value in arrays.items()
        }
        np.savez_compressed(disk_fixture.roots["bank"] / row["file"], **arrays)
    _seal_disk(disk_fixture)
    if corruption in ("block_bytes", "checkpoint_bytes"):
        path = (
            root / case["blocks"][0]["file"]
            if corruption == "block_bytes"
            else disk_fixture.roots["parent"] / "terminal_000012.pt"
        )
        path.write_bytes(path.read_bytes() + b" ")
    result = _run_disk(disk_fixture, tmp_path / "invalid_inputs")
    assert result["status"] == "failed" and result["stage"] == "training_data"
    assert (
        result["checkpoint_load_attempts"] == result["model_construction_attempts"] == 0
    )
    assert result["model_forward_calls"] == result["updates_completed"] == 0
    _assert_packet(tmp_path / "invalid_inputs", result)


@pytest.mark.parametrize("binding", ["source", "input"])
def test_late_binding_drift_invalidates_complete_updates(
    disk_fixture, tmp_path, monkeypatch, binding
):
    original = paired.paired_update
    path = (
        disk_fixture.source / "fixture_source.py"
        if binding == "source"
        else disk_fixture.roots["bank"] / "result.json"
    )

    def update(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[-1]["updates_completed"] == 4:
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(paired, "paired_update", update)
    result = _run_disk(disk_fixture, tmp_path / "changed")
    assert result["status"] == "invalid_provenance" and result["updates_completed"] == 4
    assert result["source_stable"] is (binding != "source")
    assert result["inputs_stable"] is (binding != "input")
    assert (
        result["terminal_checkpoint"] is None
        and result["terminal_candidate"]["update"] == 4
    )
    assert (tmp_path / "changed" / "terminal_000004.pt").is_file()
    _assert_packet(tmp_path / "changed", result)


def test_optimizer_completion_deadline_retains_attempted_ids_and_honest_counts(
    disk_fixture, tmp_path, monkeypatch
):
    original = paired.core.check_deadline

    def deadline(limit, stage):
        if stage == "optimizer completion":
            raise paired.core.BudgetExceeded("synthetic post-step deadline")
        original(limit, stage)

    monkeypatch.setattr(paired.core, "check_deadline", deadline)
    result = _run_disk(disk_fixture, tmp_path / "deadline")
    assert result["status"] == "incomplete_budget" and result["stage"] == "updates"
    assert result["optimizer_step_attempts"] == result["updates_completed"] == 1
    assert result["model_forward_calls"] == 2 and result["terminal_checkpoint"] is None
    assert _read(tmp_path / "deadline" / "history.json") == []
    with np.load(
        tmp_path / "deadline" / "sampling.npz", allow_pickle=False
    ) as sampling:
        assert sampling["clean"].shape == sampling["bank"].shape == (1, 8)
    _assert_packet(tmp_path / "deadline", result)


@pytest.mark.parametrize("role", ["population", "extension", "bank", "parent"])
@pytest.mark.parametrize("relationship", ["same", "nested", "ancestor"])
def test_output_isolation_is_symmetric_and_precedes_creation(
    disk_fixture, role, relationship
):
    root = disk_fixture.roots[role]
    output = {"same": root, "nested": root / "new", "ancestor": root.parent}[
        relationship
    ]
    with pytest.raises(ValueError, match="overlap"):
        _run_disk(disk_fixture, output)
    assert disk_fixture.models == []
    if relationship == "nested":
        assert not output.exists()


def test_existing_output_and_aliased_input_roles_are_rejected(disk_fixture, tmp_path):
    output = tmp_path / "keep"
    output.mkdir()
    (output / "keep.txt").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        _run_disk(disk_fixture, output)
    assert (output / "keep.txt").read_bytes() == b"keep"
    disk_fixture.roots["extension"] = disk_fixture.roots["population"]
    with pytest.raises(ValueError, match="alias"):
        _run_disk(disk_fixture, tmp_path / "aliased")
    assert not (tmp_path / "aliased").exists()


def test_fixture_packets_cannot_enter_production_pins(disk_fixture, tmp_path):
    with _read_guards(disk_fixture):
        result = paired.run(
            *(disk_fixture.roots[role] for role in paired.PINS),
            tmp_path / "production_rejected",
            phase="validate",
            arm="recovery",
            device_name="cpu",
        )
    assert result["status"] == "failed" and "hash mismatch" in result["error"]
    assert result["checkpoint_load_attempts"] == result["model_forward_calls"] == 0


def test_sync_failure_after_adam_counts_attempt_not_completed_update(monkeypatch):
    model = TinyModel()
    before = copy.deepcopy(model.state_dict())
    optimizer, receipt = paired.fresh_optimizer(model), _receipt()

    def failed_sync(*args, **kwargs):
        raise RuntimeError("synthetic asynchronous optimizer failure")

    monkeypatch.setattr(paired.clean, "_sync", failed_sync)
    clean_batch, bank_batch = _batches()
    with pytest.raises(RuntimeError, match="asynchronous optimizer"):
        paired.paired_update(
            model,
            optimizer,
            clean_batch,
            bank_batch,
            torch.device("cpu"),
            float("inf"),
            receipt,
        )
    assert receipt["optimizer_step_attempts"] == 1 and receipt["updates_completed"] == 0
    assert receipt["model_forward_calls"] == 2 and len(optimizer.state) == 2
    assert any(
        not torch.equal(model.state_dict()[key], value) for key, value in before.items()
    )


def test_terminal_deadline_precedes_checkpoint_write(
    disk_fixture, tmp_path, monkeypatch
):
    original = paired.core.check_deadline

    def deadline(limit, stage):
        if stage == "terminal serialization start":
            raise paired.core.BudgetExceeded("synthetic pre-save deadline")
        original(limit, stage)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint save started after deadline")

    monkeypatch.setattr(paired.core, "check_deadline", deadline)
    monkeypatch.setattr(paired, "save_terminal", forbidden)
    result = _run_disk(disk_fixture, tmp_path / "pre_save")
    assert result["status"] == "incomplete_budget" and result["updates_completed"] == 4
    assert result["terminal_candidate"] is result["terminal_checkpoint"] is None
    assert not list((tmp_path / "pre_save").glob("*.pt"))
    _assert_packet(tmp_path / "pre_save", result)


def test_failed_replay_retains_candidate_without_promoting_terminal(
    disk_fixture, tmp_path, monkeypatch
):
    def failed_replay(*args, **kwargs):
        raise ValueError("synthetic terminal replay failure")

    monkeypatch.setattr(paired, "replay_terminal", failed_replay)
    result = _run_disk(disk_fixture, tmp_path / "failed_replay")
    assert result["status"] == "failed" and result["updates_completed"] == 4
    assert (
        result["terminal_checkpoint"] is None
        and result["terminal_candidate"]["update"] == 4
    )
    candidate = tmp_path / "failed_replay" / result["terminal_candidate"]["file"]
    assert paired.clean._hash(candidate) == result["terminal_candidate"]["sha256"]
    assert result["inputs_stable"] and result["source_stable"]
    _assert_packet(tmp_path / "failed_replay", result)


@pytest.mark.parametrize("receipt_name", ["history.json", "sampling.npz"])
def test_ancillary_serialization_failure_preserves_failure_and_provenance(
    disk_fixture, tmp_path, monkeypatch, receipt_name
):
    original_json, original_npz = paired.core._json, np.savez_compressed

    def write_json(path, *args, **kwargs):
        if Path(path).name == receipt_name:
            raise OSError("synthetic history write failure")
        return original_json(path, *args, **kwargs)

    def write_npz(path, *args, **kwargs):
        if Path(path).name == receipt_name:
            raise OSError("synthetic sampler write failure")
        return original_npz(path, *args, **kwargs)

    monkeypatch.setattr(paired.core, "_json", write_json)
    monkeypatch.setattr(np, "savez_compressed", write_npz)
    result = _run_disk(disk_fixture, tmp_path / "failed_receipt")
    assert result["status"] == "failed" and result["updates_completed"] == 4
    assert result["training_record_error"]["type"] == "OSError"
    assert (
        result["terminal_checkpoint"] is None
        and result["terminal_candidate"]["replay"]["passed"] is True
    )
    assert result["source_stable"] and result["inputs_stable"]
    _assert_packet(tmp_path / "failed_receipt", result)


@pytest.mark.parametrize(
    "kind,accepted",
    [
        ("subthreshold", True),
        ("superthreshold", False),
        ("zero_exact", True),
        ("zero_changed", False),
        ("per_sentinel", False),
    ],
)
def test_replay_uses_per_sentinel_relative_tolerance_and_zero_convention(
    data, tmp_path, kind, accepted
):
    class ReplayModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            value = torch.ones_like(inputs)
            if kind.startswith("zero"):
                value.zero_()
            if kind == "per_sentinel":
                value[0].fill_(1e4)
                value[1].fill_(1e-4)
            if self.calls == 2:
                if kind == "subthreshold":
                    value += 0.9e-6
                elif kind == "superthreshold":
                    value += 1.1e-6
                elif kind == "zero_changed":
                    value += 1e-12
                elif kind == "per_sentinel":
                    value[1] += 2e-10
            return {"raw_next": value, "next_state": value}

    model = ReplayModel()
    identity = {"parent_update": 12, "fixture": "synthetic replay only"}
    output = tmp_path / "replay"
    output.mkdir()
    digest = paired.save_terminal(
        output / "terminal.pt",
        model,
        paired.fresh_optimizer(model),
        EpochSampler(8),
        EpochSampler(8, seed=1701),
        identity,
        4,
    )
    terminal, receipt = (
        {"file": "terminal.pt", "sha256": digest, "update": 4},
        _receipt(),
    )
    if accepted:
        result = paired.replay_terminal(
            model, data, output, terminal, identity, float("inf"), receipt
        )
        assert result["passed"] and result["relative_rms_limit"] == 1e-6
        assert all(
            0 <= error <= 1e-6
            for errors in result["errors"].values()
            for error in errors
        )
        assert (output / result["file"]).is_file()
    else:
        with pytest.raises(ValueError, match="same-device replay"):
            paired.replay_terminal(
                model, data, output, terminal, identity, float("inf"), receipt
            )
        assert not (output / "terminal_replay.npz").exists()
    assert (
        receipt["model_forward_calls"] == 2
        and receipt["checkpoint_load_attempts"] == receipt["checkpoint_loads"] == 1
    )


def test_replay_requires_exact_serialized_model_state(data, tmp_path):
    model = TinyModel()
    identity = {"parent_update": 12, "fixture": "synthetic replay only"}
    output = tmp_path / "changed_model"
    output.mkdir()
    digest = paired.save_terminal(
        output / "terminal.pt",
        model,
        paired.fresh_optimizer(model),
        EpochSampler(8),
        EpochSampler(8, seed=1701),
        identity,
        4,
    )
    with torch.no_grad():
        model.head.bias.add_(0.001)
    receipt = _receipt()
    with pytest.raises(ValueError, match="serialized model state"):
        paired.replay_terminal(
            model,
            data,
            output,
            {"file": "terminal.pt", "sha256": digest, "update": 4},
            identity,
            float("inf"),
            receipt,
        )
    assert receipt["model_forward_calls"] == receipt["checkpoint_load_attempts"] == 1
    assert receipt["checkpoint_loads"] == 0


@pytest.mark.parametrize("arm", ["clean_continuation", "recovery", "dynamics"])
def test_small_real_pcno_fit_preserves_parent_buffers_and_replays_shared_restriction(
    disk_fixture, tmp_path, monkeypatch, arm
):
    from utility.time_dependent_no.pcno_kolmogorov import (
        PeriodicVorticityPCNO,
        canonicalize_vorticity,
    )

    config = disk_fixture.results["parent"]["model_config"]
    assert config == {"resolution": 16, "width": 4, "depth": 1, "modes": 2, "fc_dim": 8}
    torch.manual_seed(117)
    parent_model = PeriodicVorticityPCNO(**config, train_scale=2.5)
    with torch.no_grad():
        parent_model.pcno.fc2.weight.fill_(0.035)
        parent_model.pcno.fc2.bias.fill_(0.07)
    initial = copy.deepcopy(parent_model.state_dict())
    disk_fixture.checkpoint["model"] = initial
    _seal_disk(disk_fixture)
    monkeypatch.setattr(paired.clean, "PeriodicVorticityPCNO", PeriodicVorticityPCNO)
    original_load = paired.load_parent_model
    initialized = []

    def load_model(model, *args, **kwargs):
        original_load(model, *args, **kwargs)
        assert type(model) is PeriodicVorticityPCNO
        for key, expected in initial.items():
            torch.testing.assert_close(
                model.state_dict()[key], expected, rtol=0, atol=0
            )
        assert torch.count_nonzero(model.pcno.fc2.weight) > 0
        assert torch.count_nonzero(model.pcno.fc2.bias) > 0
        initialized.append(True)

    monkeypatch.setattr(paired, "load_parent_model", load_model)
    output = tmp_path / ("real_pcno_" + arm)
    result = _run_disk(disk_fixture, output, arm=arm)
    assert result["status"] == "completed", result.get("error")
    assert initialized == [True]
    assert result["updates_completed"] == result["optimizer_step_attempts"] == 4
    assert result["model_forward_calls"] == 10 and result["checkpoint_loads"] == 2
    assert result["inputs_stable"] and result["source_stable"]
    assert not result["validation_arrays_read"] and result["solver_calls"] == 0
    terminal = result["terminal_checkpoint"]
    assert terminal["replay"]["passed"] is True
    checkpoint = torch.load(
        io.BytesIO((output / terminal["file"]).read_bytes()),
        map_location="cpu",
        weights_only=False,
    )
    for name, expected in parent_model.named_buffers():
        torch.testing.assert_close(checkpoint["model"][name], expected, rtol=0, atol=0)
    restored = PeriodicVorticityPCNO(**config, train_scale=2.5)
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored.eval()
    with np.load(output / "terminal_replay.npz", allow_pickle=False) as replay:
        inputs = torch.from_numpy(replay["inputs"])
        with torch.no_grad():
            outputs = restored(inputs)
        for key in ("raw_next", "next_state"):
            assert replay[key].dtype == np.float32 and np.isfinite(replay[key]).all()
            torch.testing.assert_close(
                outputs[key], torch.from_numpy(replay[key]), rtol=2e-6, atol=2e-6
            )
        torch.testing.assert_close(
            torch.from_numpy(replay["next_state"]),
            canonicalize_vorticity(torch.from_numpy(replay["raw_next"])),
            rtol=2e-6,
            atol=2e-6,
        )
        assert not np.allclose(replay["raw_next"], replay["next_state"])
    _assert_packet(output, result)
