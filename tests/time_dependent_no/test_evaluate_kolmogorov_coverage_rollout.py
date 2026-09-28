from __future__ import annotations

import hashlib
import io
import pickle
import shutil
import zipfile
import zlib

import numpy as np
import pytest
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_coverage_rollout as p
from scripts.time_dependent_no import fit_kolmogorov_coverage as fit
from tests.time_dependent_no.test_fit_kolmogorov_coverage import (
    baseline_packet as baseline_packet,  # noqa: PLC0414
)
from tests.time_dependent_no.test_fit_kolmogorov_coverage import (
    coverage_inputs as coverage_inputs,  # noqa: PLC0414
)
from tests.time_dependent_no.test_fit_kolmogorov_coverage import (
    population as population,  # noqa: PLC0414
)
from tests.time_dependent_no.test_fit_kolmogorov_coverage import (
    run_fixture,
)


@pytest.fixture
def fitted(coverage_inputs, baseline_packet, tmp_path):
    packet = tmp_path / "coverage"
    result = run_fixture(coverage_inputs, baseline_packet, packet)
    assert result["status"] == "completed" and result["updates_completed"] == 12
    parent, parent_source, _, _, _ = coverage_inputs
    source = tmp_path / "coverage_source"
    for name in fit.SOURCE_PATHS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fit.REPO_ROOT / name, target)
    return parent, parent_source, packet, source


def invoke(fitted, output, *, phase="rollout"):
    return p.run(
        *fitted,
        output,
        "cpu" if phase == "rollout" else None,
        phase=phase,
        unit_fixture=True,
    )


def assert_final_packet(output, result):
    assert fit.clean._read(output / "result.json") == result
    manifest = fit.clean._read(output / "artifact_manifest.json")
    assert manifest["status"] == result["status"]
    assert set(manifest["artifacts"]) == {
        f.name for f in output.iterdir() if f.is_file()
    } - {"artifact_manifest.json"}
    assert all(
        fit.clean._hash(output / name) == digest
        for name, digest in manifest["artifacts"].items()
    )


def reseal(packet):
    """Reseal a synthetic packet to test semantic checks beyond hash integrity."""
    manifest = fit.clean._read(packet / "artifact_manifest.json")
    manifest["artifacts"] = {
        f.name: fit.clean._hash(f)
        for f in packet.iterdir()
        if f.name != "artifact_manifest.json"
    }
    fit.clean._write(packet / "artifact_manifest.json", manifest)


def teacher_inputs(fitted):
    parent, parent_source, packet, source = fitted
    previous, evidence = p.validate_coverage(packet, source, unit_fixture=True)
    pop = fit.clean.load_population(parent, parent_source, unit_fixture=True)
    return previous, evidence, pop


def test_scientific_pins_and_inherited_rollout_contract():
    assert p.COVERAGE_MANIFEST_SHA256 == (
        "437f826a273afc331ce5a1441ffd57f230b28bb27d94bc198080dca64f45fa76"
    )
    assert p.CHECKPOINT_SHA256 == (
        "6f8b106eb3758ac4cdb08ece4313969835bdc9c77afdaffab3f055810c9d0dbf"
    )
    assert p.TEACHER_SHA256 == (
        "b33aa703aafecc2e6f85889e5873dc65896dcf9aaa7c6a0e2b663c26bdf1a5a0"
    )
    assert p.WALL_SECONDS == 1800
    assert p.baseline.HORIZONS == (1, 32, 128, 512)
    assert p.baseline.SNAPSHOT_STEPS == (0, 1, 8, 32, 64, 128, 256, 512)
    assert p.baseline.MAXIMUM_RMS_OVER_TRAIN_SCALE == 1e6


def test_teacher_maps_original_and_validation_paths_not_added_paths(fitted):
    previous, evidence, pop = teacher_inputs(fitted)
    teacher, mapping = p.matched_teacher(fitted[2], evidence, pop, previous)
    expected = list(range(8)) + list(range(10, 14))
    assert mapping == [
        {
            "original_index": i,
            "coverage_index": j,
            "seed": seed,
            "role": role,
        }
        for i, (j, (seed, role)) in enumerate(zip(expected, fit.clean.SEED_ROLES))
    ]
    with np.load(fitted[2] / evidence["teacher_file"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            teacher, saved["learned_sse"].reshape(14, 4)[expected]
        )
    assert teacher.shape == (12, 4)


def test_validate_phase_has_no_model_or_checkpoint_deserialization(
    fitted, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fit.clean,
        "PeriodicVorticityPCNO",
        lambda *a, **kw: pytest.fail("validation created a model"),
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("validation unpickled a checkpoint")
    )
    output = tmp_path / "validated"
    result = invoke(fitted, output, phase="validate")
    assert result["status"] == "completed"
    assert result["model_created"] is False
    assert not result["cases"]
    assert result["optimization_steps"] == result["solver_calls"] == 0
    assert_final_packet(output, result)


def test_all_twelve_rollouts_use_frozen_checkpoint_without_input_mutation(
    fitted, tmp_path
):
    before = {
        str(root): {f.name: fit.clean._hash(f) for f in root.iterdir()}
        for root in (fitted[0], fitted[2])
    }
    output = tmp_path / "rollouts"
    result = invoke(fitted, output)
    assert result["status"] == "completed" and result["all_rollouts_completed"]
    assert len(result["cases"]) == 12
    assert [(c["seed"], c["role"]) for c in result["cases"]] == list(
        fit.clean.SEED_ROLES
    )
    assert all(c["completed_steps"] == 4 for c in result["cases"])
    assert result["checkpoint_replay"]["passed"]
    assert result["optimization_steps"] == result["solver_calls"] == 0
    assert not result["protected_access"] and not result["corrections_added"]
    assert not result["geometry_fitted"]
    assert before == {
        str(root): {f.name: fit.clean._hash(f) for f in root.iterdir()}
        for root in (fitted[0], fitted[2])
    }
    previous, evidence, pop = teacher_inputs(fitted)
    teacher, _ = p.matched_teacher(fitted[2], evidence, pop, previous)
    for i, case in enumerate(result["cases"]):
        np.testing.assert_array_equal(case["teacher_sse"], teacher[i])
        with np.load(output / case["snapshot_file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["truth"][0], pop.states[i, 0])
    assert_final_packet(output, result)
    with pytest.raises(FileExistsError):
        invoke(fitted, output)


@pytest.mark.parametrize("location", ["parent", "coverage"])
def test_overlap_is_rejected_before_creating_output(fitted, monkeypatch, location):
    output = fitted[0 if location == "parent" else 2] / "nested_output"
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("overlap unpickled a checkpoint")
    )
    with pytest.raises(ValueError, match="overlap|inside|input"):
        invoke(fitted, output)
    assert not output.exists()


def test_tampered_checkpoint_rejected_before_deserialization(
    fitted, tmp_path, monkeypatch
):
    packet, source = fitted[2:]
    previous = fit.clean._read(packet / "result.json")
    terminal = packet / previous["terminal_checkpoint"]["file"]
    terminal.write_bytes(b"not a verified checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("unverified checkpoint opened")
    )
    with pytest.raises(ValueError, match="hash"):
        p.validate_coverage(packet, source, unit_fixture=True)
    output = tmp_path / "rejected"
    result = invoke(fitted, output)
    assert result["status"] != "completed"
    assert not result["model_created"]
    assert_final_packet(output, result)


@pytest.mark.parametrize("source_index", [1, 3])
@pytest.mark.parametrize("direction", ["below", "above"])
def test_source_output_overlap_rejected_before_any_output(
    tmp_path, monkeypatch, source_index, direction
):
    inputs = (
        tmp_path / "packets" / "parent",
        tmp_path / "sources" / "parent",
        tmp_path / "packets" / "coverage",
        tmp_path / "sources" / "coverage",
    )
    source = inputs[source_index]
    output = source / "new_output" if direction == "below" else source.parent
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("overlap loaded checkpoint")
    )
    with pytest.raises(ValueError, match="source|input"):
        p.run(*inputs, output, "cpu", unit_fixture=True)
    assert not output.exists()


def test_checkpoint_changed_after_validation_is_not_deserialized(
    fitted, tmp_path, monkeypatch
):
    original_teacher = p.matched_teacher
    previous = fit.clean._read(fitted[2] / "result.json")
    terminal = fitted[2] / previous["terminal_checkpoint"]["file"]

    def change_after_validation(*args, **kwargs):
        teacher = original_teacher(*args, **kwargs)
        terminal.write_bytes(b"changed after initial hash validation")
        return teacher

    monkeypatch.setattr(p, "matched_teacher", change_after_validation)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("unverified bytes deserialized")
    )
    output = tmp_path / "changed_after_validation"
    result = invoke(fitted, output)
    assert result["status"] == "invalid_provenance"
    assert "captured checkpoint" in result["error"]
    assert not result["cases"]
    assert_final_packet(output, result)


def test_checkpoint_loader_receives_the_verified_byte_buffer(
    fitted, tmp_path, monkeypatch
):
    previous = fit.clean._read(fitted[2] / "result.json")
    expected = previous["terminal_checkpoint"]["sha256"]
    original_load = torch.load
    calls = []

    def load_buffer(stream, *args, **kwargs):
        assert isinstance(stream, io.BytesIO)
        assert hashlib.sha256(stream.getbuffer()).hexdigest() == expected
        calls.append(True)
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load_buffer)
    output = tmp_path / "captured_checkpoint"
    result = invoke(fitted, output)
    assert calls == [True]
    assert result["status"] == "completed"
    assert result["checkpoint_replay"]["passed"]
    assert_final_packet(output, result)


def test_scientific_mode_rejects_fixture_identity_and_cpu(fitted, tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        p.validate_coverage(fitted[2], fitted[3])
    output = tmp_path / "wrong_device"
    with pytest.raises(ValueError, match="CUDA|cuda"):
        p.run(*fitted, output, "cpu")
    assert not output.exists()


@pytest.mark.parametrize("field", ["target_sse", "input_sse", "persistence_sse"])
def test_teacher_norms_must_match_the_original_reference(fitted, field):
    previous, evidence, pop = teacher_inputs(fitted)
    teacher_path = fitted[2] / evidence["teacher_file"]
    with np.load(teacher_path, allow_pickle=False) as saved:
        rows = {name: saved[name].copy() for name in saved.files}
    rows[field][0] *= 2
    with teacher_path.open("wb") as stream:
        np.savez(stream, **rows)
    with pytest.raises(ValueError):
        p.matched_teacher(fitted[2], evidence, pop, previous)


@pytest.mark.parametrize("field", ["trajectory_index", "input_step"])
def test_teacher_rejects_malformed_pair_order(fitted, field):
    previous, evidence, pop = teacher_inputs(fitted)
    teacher_path = fitted[2] / evidence["teacher_file"]
    with np.load(teacher_path, allow_pickle=False) as saved:
        rows = {name: saved[name].copy() for name in saved.files}
    indices = [0, 4] if field == "trajectory_index" else [0, 1]
    rows[field][indices] = rows[field][indices[::-1]]
    with teacher_path.open("wb") as stream:
        np.savez(stream, **rows)
    with pytest.raises(ValueError):
        p.matched_teacher(fitted[2], evidence, pop, previous)


def test_teacher_rejects_population_role_substitution(fitted):
    previous, evidence, pop = teacher_inputs(fitted)
    previous["population_index"][10]["role"] = "train"
    with pytest.raises(ValueError, match="index|role"):
        p.matched_teacher(fitted[2], evidence, pop, previous)


@pytest.mark.parametrize("field", ["train_input_rms_float64", "state_store_sha256"])
def test_resealed_population_or_normalizer_mismatch_is_rejected(
    fitted, tmp_path, monkeypatch, field
):
    packet = fitted[2]
    previous = fit.clean._read(packet / "result.json")
    original = previous["data"]["original"]
    original[field] = (
        original[field] * 2 if field == "train_input_rms_float64" else "0" * 64
    )
    fit.clean._write(packet / "result.json", previous)
    reseal(packet)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **kw: pytest.fail("mismatched input reached checkpoint"),
    )
    output = tmp_path / "bad_population"
    result = invoke(fitted, output)
    assert result["status"] != "completed"
    assert not result["model_created"]
    assert_final_packet(output, result)


@pytest.mark.parametrize(
    "error_type", [zipfile.BadZipFile, EOFError, zlib.error, pickle.UnpicklingError]
)
def test_checkpoint_decoder_failure_leaves_a_failure_packet(
    fitted, tmp_path, monkeypatch, error_type
):
    def bad_decode(*args, **kwargs):
        raise error_type("synthetic checkpoint decoder failure")

    monkeypatch.setattr(torch, "load", bad_decode)
    output = tmp_path / "decode_failure"
    result = invoke(fitted, output)
    assert result["status"] != "completed"
    assert result["error_type"] == error_type.__name__
    assert not result["cases"]
    assert_final_packet(output, result)


def test_reference_npz_decode_failure_preserves_validated_evidence(
    fitted, tmp_path, monkeypatch
):
    def bad_decode(*args, **kwargs):
        raise zipfile.BadZipFile("synthetic reference archive decode failure")

    monkeypatch.setattr(np, "load", bad_decode)
    output = tmp_path / "reference_decode_failure"
    result = invoke(fitted, output, phase="validate")
    assert result["status"] == "invalid_input"
    assert result["error_type"] == "BadZipFile"
    assert result["input_evidence_checks"] == {"coverage": True, "original": True}
    assert not result["model_created"] and not result["cases"]
    assert_final_packet(output, result)


def test_budget_exhaustion_finalizes_without_deserializing(
    fitted, tmp_path, monkeypatch
):
    monkeypatch.setattr(p, "WALL_SECONDS", -1)
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("expired budget opened checkpoint")
    )
    output = tmp_path / "budget_failure"
    result = invoke(fitted, output)
    assert result["status"] == "incomplete_budget"
    assert not result["cases"]
    assert_final_packet(output, result)


def test_inference_failure_preserves_all_failed_paths_and_final_packet(
    fitted, tmp_path, monkeypatch
):
    original_replay = fit.clean._terminal_replay

    def replay_then_fail(model, *args, **kwargs):
        replay = original_replay(model, *args, **kwargs)

        def bad_forward(*args, **kwargs):
            raise RuntimeError("synthetic rollout inference failure")

        monkeypatch.setattr(model, "forward", bad_forward)
        return replay

    monkeypatch.setattr(fit.clean, "_terminal_replay", replay_then_fail)
    output = tmp_path / "inference_failure"
    result = invoke(fitted, output)
    assert result["status"] != "completed"
    assert len(result["cases"]) == 12
    assert all(c["status"] == "model_error" for c in result["cases"])
    assert all(c["completed_steps"] == 0 for c in result["cases"])
    assert not result["all_rollouts_completed"]
    assert_final_packet(output, result)


def test_checkpoint_normalizer_buffer_cannot_replace_the_frozen_scale(
    fitted, tmp_path, monkeypatch
):
    packet = fitted[2]
    previous = fit.clean._read(packet / "result.json")
    terminal = previous["terminal_checkpoint"]
    checkpoint_path = packet / terminal["file"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model"]["train_scale"] *= 2
    torch.save(checkpoint, checkpoint_path)
    terminal["sha256"] = fit.clean._hash(checkpoint_path)
    fit.clean._write(packet / "result.json", previous)
    reseal(packet)
    monkeypatch.setattr(
        fit.clean,
        "_terminal_replay",
        lambda *a, **kw: pytest.fail("mismatched checkpoint scale reached replay"),
    )
    output = tmp_path / "bad_checkpoint_scale"
    result = invoke(fitted, output)
    assert result["status"] != "completed"
    assert not result["cases"]
    assert_final_packet(output, result)


def test_snapshot_write_failure_retains_started_path_and_computed_metrics(
    fitted, tmp_path, monkeypatch
):
    def bad_write(*args, **kwargs):
        raise OSError("synthetic snapshot serialization failure")

    monkeypatch.setattr(np, "savez_compressed", bad_write)
    output = tmp_path / "snapshot_failure"
    result = invoke(fitted, output)
    first_seed, _ = fit.clean.SEED_ROLES[0]
    assert result["status"] == "failed"
    assert result["started_seeds"] == [first_seed]
    assert result["unstarted_seeds"] == [seed for seed, _ in fit.clean.SEED_ROLES[1:]]
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["seed"] == first_seed and case["completed_steps"] == 4
    assert len(case["rows"]) == 4
    assert "snapshot_file" not in case
    assert not result["all_rollouts_completed"]
    assert_final_packet(output, result)


def test_progress_write_failure_does_not_label_started_path_unstarted(
    fitted, tmp_path, monkeypatch
):
    original_rollout = p.baseline.rollout_one

    def fail_progress(*args, **kwargs):
        def bad_progress(row):
            raise OSError("synthetic progress serialization failure")

        kwargs["progress"] = bad_progress
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(p.baseline, "rollout_one", fail_progress)
    output = tmp_path / "progress_failure"
    result = invoke(fitted, output)
    assert result["status"] == "failed"
    assert result["started_seeds"] == [fit.clean.SEED_ROLES[0][0]]
    assert result["unstarted_seeds"] == [seed for seed, _ in fit.clean.SEED_ROLES[1:]]
    assert not result["cases"] and not result["all_rollouts_completed"]
    assert_final_packet(output, result)
