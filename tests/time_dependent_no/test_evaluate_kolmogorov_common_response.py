from __future__ import annotations

import copy
import hashlib
import io
from time import perf_counter

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_common_response as p

REAL_LOAD_MODEL = p.load_model


class AffineMap(torch.nn.Module):
    def __init__(self, gain, bias):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(gain))
        self.bias = bias
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().numpy().copy())
        return {
            "raw_next": self.scale * x + self.bias + 0.125,
            "next_state": self.scale * x + self.bias,
        }


def synthetic_case():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(65, 16, 16)).astype(np.float32)
    donors = {}
    models = {"clean8": AffineMap(1.1, 0.2), "clean32": AffineMap(1.7, -0.1)}
    for name, model in models.items():
        x = reference[np.asarray(p.STEPS) - 1].copy()
        x[1:] += rng.normal(scale=0.03, size=x[1:].shape).astype(np.float32)
        with torch.no_grad():
            result = model(torch.from_numpy(x))
        donors[name] = {
            "previous_input": x,
            "raw_output": result["raw_next"].numpy(),
            "next_state": result["next_state"].numpy(),
            "truth": reference[list(p.STEPS)],
        }
        model.inputs.clear()
    return reference, donors, models


def test_pins_and_source_closure():
    assert p.STEPS == (1, 8, 32, 64)
    assert p.MODELS == ("clean8", "clean32")
    assert len(p.SOURCE_PATHS) == len(set(p.SOURCE_PATHS))
    assert set(p.expanded.SOURCE_PATHS) < set(p.SOURCE_PATHS)
    assert p.REPLAY_LIMIT == 1e-6
    assert (
        p.ROLLOUT_PINS["clean32"][1]
        == "63ed02808bd7ce68126e685c7e51e9ce18156aae606ce1c695bfacd57883cd9f"
    )


def test_common_bank_preserves_natural_inputs_and_matches_rms_without_mutation():
    reference, donors, _ = synthetic_case()
    before = copy.deepcopy((reference, donors))
    bank = p.make_bank(reference, donors, 0.02)
    assert len(bank["queries"]) == 14
    assert bank["skipped_matched_rms"] == [
        {"step": 1, "donor": name, "reason": "zero_direction"} for name in p.MODELS
    ]
    for query, x in zip(bank["queries"], bank["arrays"]["probe_input"], strict=True):
        anchor = query["anchor"]
        assert query["input_step"] == query["output_step"] - 1
        assert query["input_sha256"] == p.array_hash(x)
        u = reference[query["input_step"]]
        if query["view"] == "natural":
            np.testing.assert_array_equal(
                x, donors[query["donor"]]["previous_input"][anchor]
            )
        else:
            delta = x.astype(np.float64) - u
            direction = (
                donors[query["donor"]]["previous_input"][anchor].astype(np.float64) - u
            )
            assert np.sqrt(np.mean(delta**2)) == pytest.approx(0.02, rel=1e-5)
            assert (
                np.vdot(delta, direction)
                / (np.linalg.norm(delta) * np.linalg.norm(direction))
                > 1 - 1e-10
            )
    np.testing.assert_array_equal(reference, before[0])
    for donor in p.MODELS:
        for key in donors[donor]:
            np.testing.assert_array_equal(donors[donor][key], before[1][donor][key])


@pytest.mark.parametrize("rms", [0, -1, np.nan, np.inf, 1e-15])
def test_bad_or_unrepresentable_scale_fails(rms):
    reference, donors, _ = synthetic_case()
    with pytest.raises(ValueError):
        p.make_bank(reference, donors, rms)


def write_teacher(path, *, validation_sse=1e10, perturb=None):
    arrays = {
        "trajectory_index": np.repeat(np.arange(36), 512),
        "input_step": np.tile(np.arange(512), 36),
        "learned_sse": np.concatenate(
            (np.full(32 * 512, 4.0), np.full(4 * 512, validation_sse))
        ),
    }
    if perturb is not None:
        perturb(arrays)
    np.savez(path, **arrays)


def test_calibration_uses_all_training_pairs_and_no_validation(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    write_teacher(a)
    write_teacher(b, validation_sse=1e30)
    index = p.fit.extension.population_index(p.fit.extension.NEW_SEEDS)
    first, second = [
        p.calibrate(path, index, steps=512, nodes=256**2, train_scale=4.0)
        for path in (a, b)
    ]
    assert first["physical_rms"] == second["physical_rms"] == 2 / 256
    assert first["rms_over_train_scale"] == 2 / 1024
    assert first["training_pairs"] == 16384 and first["validation_pairs"] == 0
    altered = copy.deepcopy(index)
    altered[8]["role"] = "development"
    with pytest.raises(ValueError, match="32/4 population"):
        p.calibrate(a, altered, steps=512, nodes=256**2, train_scale=4.0)


@pytest.mark.parametrize(
    "kind", ["order", "time", "dtype", "negative", "nonfinite", "zero"]
)
def test_calibration_rejects_invalid_teacher(tmp_path, kind):
    def perturb(arrays):
        if kind == "order":
            arrays["trajectory_index"][0] = 35
        elif kind == "time":
            arrays["input_step"][0] = 511
        elif kind == "dtype":
            arrays["learned_sse"] = arrays["learned_sse"].astype(np.float32)
        elif kind == "negative":
            arrays["learned_sse"][0] = -1
        elif kind == "nonfinite":
            arrays["learned_sse"][0] = np.nan
        else:
            arrays["learned_sse"][:] = 0

    path = tmp_path / "teacher.npz"
    write_teacher(path, perturb=perturb)
    with pytest.raises(ValueError):
        p.calibrate(
            path,
            p.fit.extension.population_index(p.fit.extension.NEW_SEEDS),
            steps=512,
            nodes=256**2,
            train_scale=4.0,
        )


def test_affine_cross_model_assay_identifies_gain_and_preserves_shared_inputs():
    reference, donors, models = synthetic_case()
    bank = p.make_bank(reference, donors, 0.02)
    for name, model in models.items():
        output, rows, replays, calls = p.evaluate_bank(
            model, bank, donors[name], name, 2.0, 2, perf_counter() + 60
        )
        assert len(rows) == 28 and calls == 16 and len(replays) == 4
        assert len(model.inputs) == 16
        assert output["probe_next"].shape == (14, 16, 16)
        for row in rows:
            if row["output_step"] == 1:
                assert row["learned_secant_gain"] is None
                assert row["learned_response_scaled_rms"] == 0
            else:
                assert row["learned_secant_gain"] == pytest.approx(
                    float(model.scale.detach()), rel=1e-5
                )
            assert row["energy_identity_relative"] < 1e-12
            assert row["restriction_scaled_rms"] == pytest.approx(0.0625, rel=1e-5)
            assert (
                "displaced_defect_scaled_rms" not in row
                and "tube_escape_slope" not in row
            )
            assert row["path_recovery_error_scaled_rms"] ** 2 == pytest.approx(
                sum(row["fourier_band_scaled_energy"]["path_recovery_error"]), rel=1e-12
            )
    for a, b in zip(models["clean8"].inputs, models["clean32"].inputs, strict=True):
        np.testing.assert_array_equal(a, b)


def test_signed_alignment_changes_path_error_with_same_response_gain():
    u = np.zeros((16, 16))
    x = np.ones_like(u)
    metrics = [p.measure(u, x, u + bias, x + bias, u, 1.0, 2) for bias in (2, -2)]
    assert [m["learned_secant_gain"] for m in metrics] == [1, 1]
    assert [m["defect_response_cosine"] for m in metrics] == [1, -1]
    assert [m["path_error_excess_slope"] for m in metrics] == [1, -1]
    assert [m["path_recovery_error_scaled_rms"] for m in metrics] == [3, 1]


@pytest.mark.parametrize(
    "corruption", ["missing", "truth", "time", "dtype", "nonfinite", "initial"]
)
def test_snapshot_validation_rejects_misalignment(tmp_path, corruption):
    reference, donors, _ = synthetic_case()
    arrays = {"step": np.array(p.STEPS), **donors["clean8"]}
    if corruption == "missing":
        arrays["step"][1] = 9
    elif corruption == "truth":
        arrays["truth"][1] += 1
    elif corruption == "time":
        arrays["step"][1] = 1
    elif corruption == "dtype":
        arrays["previous_input"] = arrays["previous_input"].astype(np.float64)
    elif corruption == "nonfinite":
        arrays["raw_output"][1, 0, 0] = np.nan
    else:
        arrays["previous_input"][0] += 1
    path = tmp_path / "snap.npz"
    np.savez(path, **arrays)
    case = {
        "snapshot_file": path.name,
        "snapshot_sha256": p.clean._hash(path),
        "completed_steps": 64,
    }
    with pytest.raises(ValueError):
        p.read_snapshots(tmp_path, case, reference)


def test_snapshot_validation_binds_exact_reference_and_hash(tmp_path):
    reference, donors, _ = synthetic_case()
    path = tmp_path / "snap.npz"
    np.savez(path, step=np.array(p.STEPS), **donors["clean8"])
    case = {
        "snapshot_file": path.name,
        "snapshot_sha256": p.clean._hash(path),
        "completed_steps": 64,
    }
    data = p.read_snapshots(tmp_path, case, reference)
    np.testing.assert_array_equal(
        data["previous_input"], donors["clean8"]["previous_input"]
    )
    case["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        p.read_snapshots(tmp_path, case, reference)


def test_donor_replay_mismatch_and_deadline_fail():
    reference, donors, models = synthetic_case()
    bank = p.make_bank(reference, donors, 0.02)
    with pytest.raises(TimeoutError):
        p.evaluate_bank(models["clean8"], bank, donors["clean8"], "clean8", 2.0, 2, 0)
    donors["clean8"]["raw_output"][0] += 1
    with pytest.raises(RuntimeError, match="snapshot replay"):
        p.evaluate_bank(
            models["clean8"],
            bank,
            donors["clean8"],
            "clean8",
            2.0,
            2,
            perf_counter() + 60,
        )


def test_checkpoint_hash_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "terminal.pt"
    path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("unverified pickle decoded")
    )
    with pytest.raises(ValueError, match="captured"):
        p.load_model(
            tmp_path,
            {},
            {"checkpoint": path.name, "checkpoint_sha256": "0" * 64},
            "cpu",
        )


def test_checkpoint_deserializes_the_same_captured_bytes(tmp_path, monkeypatch):
    path = tmp_path / "terminal.pt"
    original = b"captured checkpoint bytes"
    path.write_bytes(original)

    def decode(stream, **kwargs):
        assert isinstance(stream, io.BytesIO) and stream.getvalue() == original
        path.write_bytes(b"changed after capture")
        return {"identity": {"bad": True}, "update": 49152, "schedule_position": 49152}

    monkeypatch.setattr(torch, "load", decode)
    with pytest.raises(ValueError, match="identity"):
        p.load_model(
            tmp_path,
            {"checkpoint_identity": {}},
            {
                "checkpoint": path.name,
                "checkpoint_sha256": hashlib.sha256(original).hexdigest(),
            },
            "cpu",
        )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    reference, donors, models = synthetic_case()
    cases = [
        {
            "seed": seed,
            "role": role,
            "bank": p.make_bank(reference, donors, 0.02),
            "donors": donors,
        }
        for seed, role in p.clean.SEED_ROLES
    ]
    paths = {
        key: tmp_path / key
        for key in (
            "parent",
            "parent_source",
            "clean",
            "clean_source",
            "coverage",
            "coverage_source",
            "old_rollout",
            "new_rollout",
        )
    }

    def load(paths, record, bindings, deadline):
        for key in ("parent", *p.MODELS, "old_rollout", "new_rollout"):
            bindings[key] = (
                tmp_path / key,
                tmp_path / "source",
                {"teacher_file": "teacher.npz", "manifest_sha256": key},
            )
        record.update(
            train_scale=2.0,
            model_config={"modes": 2},
            calibration={"physical_rms": 0.02},
        )
        return cases, {name: {} for name in p.MODELS}

    monkeypatch.setattr(p, "load_inputs", load)
    monkeypatch.setattr(p.fit, "evidence_stable", lambda *a: True)

    def load_model(packet, *args, record, deadline):
        record["model_construction_attempts"] += 1
        record["models_constructed"] += 1
        record["model_created"] = True
        return models[packet.name]

    for name, model in models.items():
        packet = tmp_path / name
        packet.mkdir()
        sentinel_input = torch.zeros((1, 16, 16))
        with torch.no_grad():
            sentinel = model(sentinel_input)
        np.savez(
            packet / "teacher.npz",
            sentinel_input=sentinel_input.numpy(),
            sentinel_raw=sentinel["raw_next"].numpy(),
            sentinel_next=sentinel["next_state"].numpy(),
        )
        model.inputs.clear()

    monkeypatch.setattr(p, "load_model", load_model)
    return paths, cases


def test_validate_creates_bank_without_model_or_inference(
    prepared, tmp_path, monkeypatch
):
    paths, _ = prepared
    monkeypatch.setattr(
        p, "load_model", lambda *a: pytest.fail("validate constructed model")
    )
    result = p.run(paths, tmp_path / "validation", phase="validate")
    assert result["status"] == "completed" and len(result["banks"]) == 12
    assert result["cases"] == [] and result["model_created"] is False
    assert result["model_forward_calls"] == 0 and result["expected_metric_rows"] == 672
    assert result["input_evidence_stable"] is True


def test_complete_assay_array_packet_and_accounting(prepared, tmp_path):
    paths, _ = prepared
    output = tmp_path / "assay"
    result = p.run(
        paths, output, phase="assay", device="cuda"
    )  # Loader is a CPU synthetic map.
    assert result["status"] == "completed"
    assert result["completed_recipients"] == list(p.MODELS)
    assert len(result["cases"]) == 24 and result["model_forward_calls"] == 386
    assert result["model_construction_attempts"] == result["models_constructed"] == 2
    assert sum(len(c["rows"]) for c in result["cases"]) == 672
    assert result["solver_calls"] == result["optimization_steps"] == 0
    manifest = p.clean._read(output / "artifact_manifest.json")
    assert set(manifest["artifacts"]) == {f.name for f in output.iterdir()} - {
        "artifact_manifest.json"
    }
    assert all(
        p.clean._hash(output / name) == sha
        for name, sha in manifest["artifacts"].items()
    )
    for case in result["cases"]:
        assert len(case["natural_replays"]) == 4
        assert case["forward_calls"] == 16
        bank = next(b for b in result["banks"] if b["seed"] == case["seed"])
        with (
            np.load(output / bank["file"], allow_pickle=False) as inputs,
            np.load(output / case["file"], allow_pickle=False) as outputs,
        ):
            scale = result["train_scale"]
            for row in case["rows"]:
                j, a, kind = row["query_index"], row["anchor"], row["output_kind"]
                u = inputs["reference_input"][a].astype(np.float64)
                truth = inputs["reference_next"][a].astype(np.float64)
                x = inputs["probe_input"][j].astype(np.float64)
                clean = outputs["clean_" + kind][a].astype(np.float64)
                displaced = outputs["probe_" + kind][j].astype(np.float64)
                eta, forcing, response = (
                    (x - u) / scale,
                    (clean - truth) / scale,
                    (displaced - clean) / scale,
                )

                def rms(value):
                    return float(np.sqrt(np.mean(value**2)))

                eta_norm, forcing_norm, response_norm = map(
                    rms, (eta, forcing, response)
                )
                path_norm = rms((displaced - truth) / scale)
                cross = float(np.mean(forcing * response))
                assert row["input_displacement_scaled_rms"] == pytest.approx(
                    eta_norm, rel=1e-12
                )
                assert row["clean_defect_scaled_rms"] == pytest.approx(
                    forcing_norm, rel=1e-12
                )
                assert row["learned_response_scaled_rms"] == pytest.approx(
                    response_norm, rel=1e-12
                )
                assert row["path_recovery_error_scaled_rms"] == pytest.approx(
                    path_norm, rel=1e-12
                )
                assert row["defect_response_scaled_inner"] == pytest.approx(
                    cross, rel=1e-12, abs=1e-15
                )
                if eta_norm:
                    assert row["learned_secant_gain"] == pytest.approx(
                        response_norm / eta_norm, rel=1e-12
                    )
                    assert row["path_error_excess_slope"] == pytest.approx(
                        (path_norm - forcing_norm) / eta_norm, rel=1e-12, abs=1e-12
                    )
                else:
                    assert row["learned_secant_gain"] is None
                if forcing_norm * response_norm:
                    assert row["defect_response_cosine"] == pytest.approx(
                        cross / (forcing_norm * response_norm), rel=1e-12, abs=1e-15
                    )
                else:
                    assert row["defect_response_cosine"] is None


@pytest.mark.parametrize(
    "failure", ["decode", "model", "partial", "mutation", "deadline", "missing_binding"]
)
def test_failures_leave_non_success_packets(prepared, tmp_path, monkeypatch, failure):
    paths, cases = prepared
    if failure == "decode":
        monkeypatch.setattr(
            p, "load_inputs", lambda *a: (_ for _ in ()).throw(EOFError("decoder"))
        )
    elif failure == "model":
        monkeypatch.setattr(
            p,
            "load_model",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model")),
        )
    elif failure == "partial":
        cases.pop()
    elif failure == "mutation":
        monkeypatch.setattr(p.fit, "evidence_stable", lambda *a: False)
    elif failure == "deadline":
        monkeypatch.setattr(p, "WALL_SECONDS", -1)
    else:
        load = p.load_inputs

        def missing(paths, record, bindings, deadline):
            result = load(paths, record, bindings, deadline)
            bindings.pop("old_rollout")
            return result

        monkeypatch.setattr(p, "load_inputs", missing)
    output = tmp_path / "failure"
    result = p.run(paths, output, phase="assay", device="cuda")
    assert result["status"] != "completed"
    assert p.clean._read(output / "result.json")["status"] == result["status"]
    assert (
        p.clean._read(output / "artifact_manifest.json")["status"] == result["status"]
    )


@pytest.mark.parametrize("relationship", ["equal", "inside", "ancestor"])
def test_output_overlap_fails_before_writes(tmp_path, relationship):
    source = tmp_path / "source"
    source.mkdir()
    output = (
        source
        if relationship == "equal"
        else source / "child"
        if relationship == "inside"
        else tmp_path
    )
    with pytest.raises(ValueError, match="disjoint"):
        p.run({"parent_source": source}, output, phase="validate")
    assert list(source.iterdir()) == []


def test_existing_output_refused(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        p.run({"parent": tmp_path / "input"}, output, phase="validate")


def test_float32_rounding_can_preserve_rms_but_destroy_direction():
    epsilon = 2.0**-23
    reference = np.ones((65, 16, 16), dtype=np.float32)
    x = reference[np.asarray(p.STEPS) - 1].copy()
    delta = np.tile([epsilon, 2 * epsilon], 128).reshape(16, 16)
    x[1:] += delta.astype(np.float32)
    desired = epsilon / np.sqrt(2)
    intended = delta * desired / np.sqrt(np.mean(delta**2))
    realized = (reference[0].astype(np.float64) + intended).astype(
        np.float32
    ) - reference[0]
    assert np.sqrt(np.mean(realized.astype(np.float64) ** 2)) == pytest.approx(desired)
    assert np.vdot(realized, delta) / (
        np.linalg.norm(realized) * np.linalg.norm(delta)
    ) == pytest.approx(2 / np.sqrt(5))
    donors = {name: {"previous_input": x.copy()} for name in p.MODELS}
    with pytest.raises(ValueError, match="unresolved"):
        p.make_bank(reference, donors, desired)


def test_loading_past_deadline_prevents_sentinel_inference(
    prepared, tmp_path, monkeypatch
):
    paths, _ = prepared
    clock = [0.0]
    original = p.load_model
    sentinel_calls = []

    def slow_load(*args, **kwargs):
        model = original(*args, **kwargs)
        clock[0] = p.WALL_SECONDS + 1
        return model

    monkeypatch.setattr(p, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(p.fit, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(p, "load_model", slow_load)
    monkeypatch.setattr(p, "terminal_replay", lambda *args: sentinel_calls.append(True))
    result = p.run(paths, tmp_path / "expired_load", phase="assay", device="cuda")
    assert result["status"] == "incomplete_budget"
    assert sentinel_calls == []
    assert result["model_forward_calls"] == 0


def test_failed_own_replay_retains_already_executed_forward_count(prepared, tmp_path):
    paths, cases = prepared
    cases[0]["donors"]["clean8"]["raw_output"][0] += 1
    result = p.run(paths, tmp_path / "failed_replay", phase="assay", device="cuda")
    assert result["status"] == "failed" and result["cases"] == []
    assert result["model_forward_calls"] == 5  # One sentinel batch, four clean queries.


def test_failed_sentinel_retains_its_actual_model_invocation(
    prepared, tmp_path, monkeypatch
):
    paths, _ = prepared

    def failed_replay(model, path, deadline):
        with torch.no_grad():
            model(torch.zeros((1, 16, 16)))
        raise RuntimeError("sentinel mismatch after inference")

    monkeypatch.setattr(p, "terminal_replay", failed_replay)
    result = p.run(paths, tmp_path / "failed_sentinel", phase="assay", device="cuda")
    assert result["status"] == "failed" and result["cases"] == []
    assert result["model_forward_calls"] == 1


def test_state_loading_failure_reports_completed_construction(
    prepared, tmp_path, monkeypatch
):
    paths, _ = prepared
    original_inputs = p.load_inputs
    identity = {"train_scale_float64": 2.0, "train_scale_model_float32": 2.0}
    config = {"resolution": 16, "width": 4, "depth": 1, "modes": 2, "fc_dim": 4}
    packet = tmp_path / "checkpoint_input"
    packet.mkdir()
    checkpoint = packet / "terminal.pt"
    torch.save(
        {
            "identity": identity,
            "update": 49152,
            "schedule_position": 49152,
            "model": {},
        },
        checkpoint,
    )

    def inputs(paths, record, bindings, *args):
        cases, previous = original_inputs(paths, record, bindings, *args)
        bindings["clean8"] = (
            packet,
            packet,
            {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": p.clean._hash(checkpoint),
                "teacher_file": "unused.npz",
                "manifest_sha256": "fixture",
            },
        )
        previous["clean8"] = {"checkpoint_identity": identity, "model_config": config}
        return cases, previous

    # Bypass the fixture's fake loader, but exercise the real constructor and load_state_dict.
    monkeypatch.setattr(p, "load_inputs", inputs)
    monkeypatch.setattr(p, "load_model", REAL_LOAD_MODEL)
    result = p.run(
        paths, tmp_path / "failed_state_loading", phase="assay", device="cuda"
    )
    assert result["status"] == "failed" and result["model_forward_calls"] == 0
    assert result["model_created"] is True
    assert result["model_construction_attempts"] == result["models_constructed"] == 1


def test_unsigned_descending_snapshot_times_are_rejected(tmp_path):
    reference, donors, _ = synthetic_case()
    arrays = {key: values[::-1] for key, values in donors["clean8"].items()}
    path = tmp_path / "descending.npz"
    np.savez(path, step=np.asarray(p.STEPS[::-1], dtype=np.uint64), **arrays)
    case = {
        "snapshot_file": path.name,
        "snapshot_sha256": p.clean._hash(path),
        "completed_steps": 64,
    }
    with pytest.raises(ValueError, match="ordered"):
        p.read_snapshots(tmp_path, case, reference)


def test_load_inputs_stops_between_packet_validations(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(p.fit, "perf_counter", lambda: clock[0])
    evidence = {"manifest_sha256": "validated_old"}

    def old_validation(*args):
        clock[0] = 2.0
        return {}, evidence

    monkeypatch.setattr(p.baseline, "validate_clean", old_validation)
    monkeypatch.setattr(
        p.expanded,
        "validate_coverage",
        lambda *a: pytest.fail("continued after deadline"),
    )
    bindings = {}
    with pytest.raises(TimeoutError):
        p.load_inputs({"clean": tmp_path, "clean_source": tmp_path}, {}, bindings, 1.0)
    assert bindings["clean8"][2] == evidence


def test_bank_serialization_deadline_prevents_next_bank(
    prepared, tmp_path, monkeypatch
):
    paths, _ = prepared
    clock = [0.0]
    original = p.save_arrays
    writes = []

    def save(path, arrays):
        result = original(path, arrays)
        writes.append(path.name)
        clock[0] = p.WALL_SECONDS + 1
        return result

    monkeypatch.setattr(p, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(p.fit, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(p, "save_arrays", save)
    result = p.run(paths, tmp_path / "expired_serialization", phase="validate")
    assert result["status"] == "incomplete_budget"
    assert len(writes) == len(result["banks"]) == 1
    assert result["model_created"] is False


def test_checkpoint_decode_deadline_prevents_constructor(tmp_path, monkeypatch):
    path = tmp_path / "terminal.pt"
    path.write_bytes(b"verified fixture bytes")
    clock = [0.0]
    identity = {"fixture": True}

    def decode(*args, **kwargs):
        clock[0] = 2.0
        return {"identity": identity, "update": 49152, "schedule_position": 49152}

    monkeypatch.setattr(p.fit, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch, "load", decode)
    monkeypatch.setattr(
        p.clean,
        "PeriodicVorticityPCNO",
        lambda *a, **k: pytest.fail("constructed after deadline"),
    )
    record = {
        "model_construction_attempts": 0,
        "models_constructed": 0,
        "model_created": False,
    }
    with pytest.raises(TimeoutError):
        p.load_model(
            tmp_path,
            {"checkpoint_identity": identity},
            {"checkpoint": path.name, "checkpoint_sha256": p.clean._hash(path)},
            "cpu",
            record=record,
            deadline=1.0,
        )
    assert record["model_construction_attempts"] == 0


def test_failed_forward_is_counted_and_hook_removed(prepared, tmp_path, monkeypatch):
    paths, _ = prepared
    original = p.load_model
    returned = []

    def load(*args, **kwargs):
        model = original(*args, **kwargs)
        monkeypatch.setattr(
            model,
            "forward",
            lambda *a: (_ for _ in ()).throw(RuntimeError("forward failed")),
        )
        returned.append(model)
        return model

    monkeypatch.setattr(p, "load_model", load)
    result = p.run(paths, tmp_path / "failed_forward", phase="assay", device="cuda")
    assert result["status"] == "failed" and result["model_forward_calls"] == 1
    assert returned[0]._forward_pre_hooks == {}


@pytest.mark.parametrize("expired_stage", ["decode", "transfer", "forward"])
def test_sentinel_deadline_stops_next_expensive_stage(
    prepared, tmp_path, monkeypatch, expired_stage
):
    paths, _ = prepared
    clock, stages, loaded = [0.0], [], []
    with np.load(tmp_path / "clean8" / "teacher.npz", allow_pickle=False) as saved:
        archive_type = type(saved)
    original_getitem = archive_type.__getitem__
    original_to = torch.Tensor.to
    original_load = p.load_model

    def decode(saved, key):
        value = original_getitem(saved, key)
        if key == "sentinel_input":
            stages.append("decode")
            if expired_stage == "decode":
                clock[0] = p.WALL_SECONDS + 1
        return value

    def transfer(tensor, *args, **kwargs):
        value = original_to(tensor, *args, **kwargs)
        stages.append("transfer")
        if expired_stage == "transfer":
            clock[0] = p.WALL_SECONDS + 1
        return value

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        original_forward = model.forward

        def forward(inputs):
            result = original_forward(inputs)
            stages.append("forward")
            if expired_stage == "forward":
                clock[0] = p.WALL_SECONDS + 1
            return result

        monkeypatch.setattr(model, "forward", forward)
        loaded.append(model)
        return model

    monkeypatch.setattr(p, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(p.fit, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(archive_type, "__getitem__", decode)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(p, "load_model", load)
    output = tmp_path / "sentinel_expiry"
    result = p.run(paths, output, phase="assay", device="cuda")
    assert result["status"] == "incomplete_budget"
    assert result["model_created"] is True
    assert result["model_construction_attempts"] == result["models_constructed"] == 1
    expected_stages = ["decode", "transfer", "forward"]
    assert stages == expected_stages[: expected_stages.index(expired_stage) + 1]
    expected_calls = int(expired_stage == "forward")
    assert result["model_forward_calls"] == len(loaded[0].inputs) == expected_calls
    assert loaded[0]._forward_pre_hooks == {}
    assert result["cases"] == [] and result["checkpoint_replay"] == {}
    assert p.clean._read(output / "result.json")["status"] == "incomplete_budget"


@pytest.mark.parametrize(
    "case",
    ["exact", "within", "raw_mismatch", "next_mismatch", "nonfinite", "zero_expected"],
)
def test_sentinel_replay_equivalence_to_closed_helper(tmp_path, case):
    model = AffineMap(1.1, 0.0)
    inputs = np.random.default_rng(9).normal(size=(3, 16, 16)).astype(np.float32)
    inputs[0] = 0  # Includes the zero/zero denominator branch for next_state.
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs))
    raw = expected["raw_next"].numpy().copy()
    projected = expected["next_state"].numpy().copy()
    if case == "within":
        raw[1] += 1e-7
    elif case == "raw_mismatch":
        raw[1] += 1e-3
    elif case == "next_mismatch":
        projected[1] += 1e-3
    elif case == "nonfinite":
        raw[1, 0, 0] = np.nan
    elif case == "zero_expected":
        raw[0] = 0
    path = tmp_path / "sentinel.npz"
    np.savez(path, sentinel_input=inputs, sentinel_raw=raw, sentinel_next=projected)
    if case in ("exact", "within"):
        assert p.terminal_replay(model, path, float("inf")) == p.clean._terminal_replay(
            model, path
        )
    else:
        for replay in (
            lambda: p.terminal_replay(model, path, float("inf")),
            lambda: p.clean._terminal_replay(model, path),
        ):
            with pytest.raises(RuntimeError, match="terminal checkpoint sentinel"):
                replay()
