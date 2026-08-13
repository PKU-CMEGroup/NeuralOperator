from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from scripts.time_dependent_no import (
    capture_realm_ignithit_domain_link_failure as capture,
)
from scripts.time_dependent_no import (
    train_realm_ignithit_domain_linked_ffno as d089,
)
from utility.time_dependent_no.realm_benchmark import (
    RealmNormalizer,
    canonical_json_sha256,
)
from utility.time_dependent_no.realm_domain_link import (
    BoxCoxDomainLink,
    DomainLinkedMap,
)
from utility.time_dependent_no.realm_ignithit import sha256_file


def _eligible_validation() -> dict[str, Any]:
    return {
        "all_normalized_finite": True,
        "all_decoded_finite": True,
        "all_released_state_admissible": True,
        "all_bounded_10x_train_max": True,
        "realm_npe_mean": 1.25,
    }


def _parent_history() -> list[dict[str, Any]]:
    return [
        {
            "completed_step": 1,
            "validation": {"realm_npe_mean": capture.PARENT_BEST_SCORE},
        },
        {
            "completed_step": 50,
            "validation": {"realm_npe_mean": capture.PARENT_LAST_SCORE},
        },
    ]


def test_contract_freezes_identity_scope_and_knob_free_cli() -> None:
    assert capture.STABLE_ID == "D090"
    assert capture.PARENT_RUN_ID == d089.RUN_ID
    assert capture.PARENT_COMPLETED_STEP == 50
    assert capture.TARGET_COMPLETED_STEP == 100
    assert capture.REPLAY_STEPS == tuple(range(51, 101))

    contract = capture.diagnostic_contract()
    assert contract["run_id"] == capture.RUN_ID
    assert contract["replay_steps"] == list(range(51, 101))
    assert contract["validation_start_frame"] == 0
    assert contract["validation_horizon"] == 29
    assert contract["write_order"][0] == "validation_row.json"
    assert contract["stop_after_one_validation_regardless_of_outcome"] is True
    assert contract["step100_checkpoint_role"] == (
        "nonresumable diagnostic identity only"
    )
    assert contract["test_object_available"] is False
    assert contract["residual_arm_authorized"] is False
    assert contract["canonical_payload_sha256"] == (
        "a7a72e3a00fc801440cf8ab828be3dfd76db1d1582e09876d50cd65b7761f382"
    )
    digest = contract.pop("canonical_payload_sha256")
    assert digest == canonical_json_sha256(contract)

    parser = capture.build_parser()
    argument_names = {
        action.dest for action in parser._actions if action.dest != "help"
    }
    assert argument_names == {
        "manifest",
        "data_root",
        "normalizer_arrays",
        "parent_output_dir",
        "output_dir",
    }


def test_source_manifest_binds_full_executed_chain_and_frozen_parent() -> None:
    assert (
        d089.source_manifest()["canonical_payload_sha256"]
        == capture.PARENT_PROVENANCE["source_digest"]
    )
    assert (
        sha256_file(Path(d089.parent.__file__).resolve()) == d089.PARENT_SOURCE_SHA256
    )
    source = capture._source_manifest()
    assert [row["path"] for row in source["files"]] == [
        "utility/time_dependent_no/realm_benchmark.py",
        "utility/time_dependent_no/realm_ignithit.py",
        "utility/time_dependent_no/realm_ffno.py",
        "utility/time_dependent_no/realm_domain_link.py",
        "scripts/time_dependent_no/train_realm_ignithit_ffno.py",
        "scripts/time_dependent_no/train_realm_ignithit_domain_linked_ffno.py",
        "scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py",
        "scripts/time_dependent_no/capture_realm_ignithit_domain_link_failure.py",
    ]
    assert source["canonical_payload_sha256"] == canonical_json_sha256(
        {
            key: value
            for key, value in source.items()
            if key != "canonical_payload_sha256"
        }
    )
    assert source["canonical_payload_sha256"] == (
        "0ea9cbd4f7ed8bfc42d516c841fd9e4402903125302d1b16059341e17354400c"
    )


def test_parent_hash_guard_requires_exact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    payload = b"exact D089 parent"
    expected = hashlib.sha256(payload).hexdigest()
    (parent / "only.bin").write_bytes(payload)
    monkeypatch.setattr(capture, "PARENT_FILE_SHA256", {"only.bin": expected})

    assert capture.validate_parent_files(parent) == {"only.bin": expected}
    (parent / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory differs"):
        capture.validate_parent_files(parent)
    (parent / "unexpected.bin").unlink()
    (parent / "only.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        capture.validate_parent_files(parent)


def test_output_isolation_is_fail_closed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    parent = tmp_path / "parent"
    data.mkdir()
    parent.mkdir()
    output = tmp_path / "capture"

    capture.prepare_new_output_directory(
        output,
        data_root=data,
        parent_output_dir=parent,
    )
    assert output.is_dir()
    (output / "occupied.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        capture.prepare_new_output_directory(
            output,
            data_root=data,
            parent_output_dir=parent,
        )
    with pytest.raises(ValueError, match="parent output must not overlap"):
        capture.prepare_new_output_directory(
            parent / "child",
            data_root=data,
            parent_output_dir=parent,
        )
    with pytest.raises(ValueError, match="data root must not overlap"):
        capture.prepare_new_output_directory(
            data / "child",
            data_root=data,
            parent_output_dir=parent,
        )


def test_restored_parent_identity_binds_both_retained_rows() -> None:
    history = _parent_history()
    checkpoint = {
        "schema": d089.LAST_CHECKPOINT_SCHEMA,
        "run_id": capture.PARENT_RUN_ID,
        "run_signature": capture.PARENT_RUN_SIGNATURE,
        "provenance": capture.PARENT_PROVENANCE,
        "model_state_sha256": capture.PARENT_LAST_MODEL_STATE_SHA256,
    }
    capture.validate_restored_parent_identity(
        checkpoint,
        completed_step=50,
        best_step=1,
        best_score=capture.PARENT_BEST_SCORE,
        best_model_state_sha256=capture.PARENT_BEST_MODEL_STATE_SHA256,
        checkpoint_history=history,
        history_file={"rows": history},
    )

    drifted = _parent_history()
    drifted[-1]["validation"]["realm_npe_mean"] = 9.0
    with pytest.raises(ValueError, match="identity differs"):
        capture.validate_restored_parent_identity(
            checkpoint,
            completed_step=50,
            best_step=1,
            best_score=capture.PARENT_BEST_SCORE,
            best_model_state_sha256=capture.PARENT_BEST_MODEL_STATE_SHA256,
            checkpoint_history=drifted,
            history_file={"rows": drifted},
        )


def test_replay_wrapper_uses_only_the_maintained_exact_step_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    generator = random.Random(0)
    train = torch.zeros(1)
    coordinates = torch.zeros(1)
    observed: dict[str, Any] = {}

    def fake_replay(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "schema": "maintained",
            "rows": [{"completed_step": 100}],
            "optimizer_step_count": 50,
        }

    monkeypatch.setattr(
        capture.replay_support,
        "replay_registered_steps",
        fake_replay,
    )
    trace = capture.replay_registered_steps(
        model,
        optimizer,
        scheduler,
        generator,
        train,
        coordinates,
        device=torch.device("cpu"),
    )

    assert observed["args"] == (
        model,
        optimizer,
        scheduler,
        generator,
        train,
        coordinates,
    )
    assert observed["kwargs"] == {"device": torch.device("cpu")}
    assert trace["schema"] == "d090_ignithit_failure_capture_replay_trace_v1"
    assert trace["run_id"] == capture.RUN_ID
    assert trace["parent_run_id"] == capture.PARENT_RUN_ID
    assert trace["optimizer_step_count"] == 50


def test_validation_row_is_persisted_and_verified_before_failed_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_row.json"
    validation = _eligible_validation()
    validation["all_bounded_10x_train_max"] = False
    saw_persisted_row = False

    def check_after_write(values: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal saw_persisted_row
        persisted = json.loads(path.read_text(encoding="utf-8"))
        assert persisted["schema"] == capture.VALIDATION_ROW_SCHEMA
        assert persisted["validation"] == validation
        saw_persisted_row = True
        return d089.require_eligible_validation(values)

    result = capture.capture_validation_before_gate(
        path,
        {"completed_step": 100, "validation": validation},
        eligibility_check=check_after_write,
    )

    assert saw_persisted_row is True
    assert result["status"] == "captured_gate_failure"
    assert result["eligible"] is False
    assert result["failed_flags"] == ["all_bounded_10x_train_max"]
    assert result["validation_row_sha256"] == sha256_file(path)
    assert result["stopped_after_gate"] is True


def test_capture_stops_even_if_step100_is_unexpectedly_eligible(tmp_path: Path) -> None:
    path = tmp_path / "validation_row.json"
    result = capture.capture_validation_before_gate(
        path,
        {"completed_step": 100, "validation": _eligible_validation()},
    )

    assert result["status"] == "unexpected_eligible"
    assert result["eligible"] is True
    assert result["failed_flags"] == []
    assert result["stopped_after_gate"] is True
    assert path.is_file()


def test_capture_rejects_missing_nonboolean_or_nonjson_metrics(tmp_path: Path) -> None:
    missing = _eligible_validation()
    missing.pop("all_decoded_finite")
    with pytest.raises(ValueError, match="missing flags"):
        capture.capture_validation_before_gate(
            tmp_path / "missing.json",
            {"validation": missing},
        )

    non_boolean = _eligible_validation()
    non_boolean["all_decoded_finite"] = 1
    with pytest.raises(TypeError, match="must be booleans"):
        capture.capture_validation_before_gate(
            tmp_path / "non_boolean.json",
            {"validation": non_boolean},
        )

    nonjson = _eligible_validation()
    nonjson["realm_npe_mean"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        capture.capture_validation_before_gate(
            tmp_path / "nonjson.json",
            {"validation": nonjson},
        )
    assert not (tmp_path / "nonjson.json").exists()


class _IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        return state * self.scale.view(1, 1, 1, 1)


def _synthetic_validation_fixture() -> tuple[
    DomainLinkedMap,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    RealmNormalizer,
    torch.Tensor,
]:
    mean = torch.zeros(12, dtype=torch.float32)
    scale = torch.ones(12, dtype=torch.float32)
    transformed = tuple(range(8))
    normalizer = RealmNormalizer(
        mean=mean,
        scale=scale,
        transformed_channels=transformed,
        channel_axis=2,
        box_cox_lambda=0.1,
        box_cox_epsilon=1.0e-8,
        std_correction=1,
        scale_stabilizer=1.0e-10,
    )
    link = BoxCoxDomainLink(
        mean,
        scale,
        transformed_channels=transformed,
        box_cox_lambda=0.1,
        base_floor=d089.BASE_FLOOR,
        channel_axis=1,
    )
    model = DomainLinkedMap(_IdentityBackbone(), link)
    normalized = torch.zeros(1, 3, 12, 2, 2, dtype=torch.float32)
    normalized[:, :, 8:10] = 1.0
    native = normalizer.decode(normalized, inverse_domain_policy="nan")
    coordinates = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    train_max_abs = torch.full((12,), 2.0, dtype=torch.float32)
    return model, normalized, native, coordinates, normalizer, train_max_abs


def test_raw_capture_matches_parent_metrics_and_retains_stage_identities() -> None:
    model, normalized, native, coordinates, normalizer, train_max_abs = (
        _synthetic_validation_fixture()
    )
    model.train()

    validation, stages = capture.run_raw_validation_capture(
        model,
        normalized,
        native,
        coordinates,
        normalizer,
        train_max_abs,
        case_keys=("synthetic",),
        calls=2,
    )

    assert model.training is True
    assert validation["all_normalized_finite"] is True
    assert validation["all_decoded_finite"] is True
    assert validation["all_released_state_admissible"] is True
    assert validation["all_bounded_10x_train_max"] is True
    assert "domain_link" in validation
    assert stages["prediction_shape"] == [1, 2, 12, 2, 2]
    assert stages["calls"] == 2
    assert stages["recurrence"] == "current=domain_linked_proposal"
    assert stages["raw_output_repair"] is False
    digest = stages.pop("canonical_payload_sha256")
    assert digest == canonical_json_sha256(stages)


def test_raw_capture_rejects_nonregistered_h29_population() -> None:
    model, normalized, native, coordinates, normalizer, train_max_abs = (
        _synthetic_validation_fixture()
    )
    with pytest.raises(ValueError, match="frozen cases"):
        capture.run_raw_validation_capture(
            model,
            normalized,
            native,
            coordinates,
            normalizer,
            train_max_abs,
            case_keys=("synthetic",),
            calls=29,
        )


def test_capture_checkpoint_is_bound_and_nonresumable() -> None:
    model = _IdentityBackbone()
    normalizer_state = {
        "mean": torch.zeros(12),
        "scale": torch.ones(12),
        "source_sha256": "a" * 64,
    }
    checkpoint = capture.build_capture_checkpoint(
        model,
        normalizer_state=normalizer_state,
        diagnostic_run_signature="b" * 64,
        diagnostic_source_digest="c" * 64,
        runtime_digest="d" * 64,
        replay_trace_digest="e" * 64,
    )

    assert checkpoint["schema"] == capture.CAPTURE_CHECKPOINT_SCHEMA
    assert checkpoint["completed_step"] == 100
    assert checkpoint["diagnostic_only"] is True
    assert checkpoint["selection_eligible"] is False
    assert checkpoint["resume_supported"] is False
    assert checkpoint["test_object_opened"] is False
    assert "optimizer_state" not in checkpoint
    assert checkpoint["model_state_sha256"] == (
        capture.parent.structured_state_sha256(checkpoint["model_state"])
    )
    assert checkpoint["normalizer_state_sha256"] == (
        capture.parent.structured_state_sha256(checkpoint["normalizer_state"])
    )


def test_help_is_cpu_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert torch.cuda.is_initialized() is False
    with pytest.raises(SystemExit) as exc_info:
        capture.main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "missing D089 step-100 validation row" in output
    assert "--parent-output-dir" in output
    assert torch.cuda.is_initialized() is False
