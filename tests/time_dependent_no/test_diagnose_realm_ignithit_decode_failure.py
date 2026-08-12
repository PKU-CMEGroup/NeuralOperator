from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from scripts.time_dependent_no import train_realm_ignithit_ffno as parent_trainer
from scripts.time_dependent_no.diagnose_realm_ignithit_decode_failure import (
    MODEL_CHECKPOINT_SCHEMA,
    PARENT_BEST_SHA256,
    PARENT_COMPLETED_STEP,
    PARENT_LAST_SHA256,
    PARENT_PROVENANCE,
    PARENT_RUN_SIGNATURE,
    REPLAY_STEPS,
    RUN_ID,
    TARGET_COMPLETED_STEP,
    _diagnostic_source_manifest,
    build_parser,
    build_step100_model_checkpoint,
    diagnostic_contract,
    localize_decode_nonfiniteness,
    prepare_new_output_directory,
    replay_registered_steps,
    validate_file_hashes,
    validate_restored_parent_identity,
)
from scripts.time_dependent_no.diagnose_realm_ignithit_decode_failure import (
    main as diagnostic_main,
)
from utility.time_dependent_no.realm_benchmark import (
    RealmNormalizer,
    canonical_json_sha256,
)
from utility.time_dependent_no.realm_ffno import grouped_next_state_mse
from utility.time_dependent_no.realm_ignithit import TRAIN_GROUPS, sha256_file


def _normalizer(
    *,
    scale: torch.Tensor | None = None,
    transformed_channels: tuple[int, ...] = tuple(range(8)),
) -> RealmNormalizer:
    return RealmNormalizer(
        mean=torch.zeros(12, dtype=torch.float32),
        scale=torch.ones(12, dtype=torch.float32) if scale is None else scale,
        transformed_channels=transformed_channels,
        channel_axis=2,
        box_cox_lambda=0.1,
        box_cox_epsilon=1.0e-8,
        std_correction=1,
        scale_stabilizer=1.0e-10,
    )


def test_contract_fixes_parent_identity_step_range_and_knob_free_cli() -> None:
    assert PARENT_COMPLETED_STEP == 50
    assert TARGET_COMPLETED_STEP == 100
    assert REPLAY_STEPS == tuple(range(51, 101))
    assert len(REPLAY_STEPS) == 50
    assert len(PARENT_LAST_SHA256) == len(PARENT_BEST_SHA256) == 64
    assert len(PARENT_RUN_SIGNATURE) == 64

    contract = diagnostic_contract()
    assert contract["run_id"] == RUN_ID
    assert contract["replay_steps"] == list(range(51, 101))
    assert contract["stop_after_localization_regardless_of_reproduction"] is True
    assert contract["residual_arm_authorized"] is False

    parser = build_parser()
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


def test_parent_trainer_remains_byte_identical() -> None:
    assert sha256_file(Path(parent_trainer.__file__)) == (
        "a3fa0c0e831765ce24c6ae3aa11e3d2afcf67cc242de0a508381dc0b25b43f1e"
    )
    assert (
        parent_trainer.frozen_training_contract()["canonical_payload_sha256"]
        == PARENT_PROVENANCE["config_digest"]
    )
    assert (
        parent_trainer._source_manifest()["canonical_payload_sha256"]
        == (PARENT_PROVENANCE["source_digest"])
    )
    diagnostic_source = _diagnostic_source_manifest()
    assert [row["path"] for row in diagnostic_source["files"]] == [
        "utility/time_dependent_no/realm_benchmark.py",
        "utility/time_dependent_no/realm_ignithit.py",
        "utility/time_dependent_no/realm_ffno.py",
        "scripts/time_dependent_no/train_realm_ignithit_ffno.py",
        "scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py",
    ]


def test_parent_hash_guard_and_output_isolation(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    data = tmp_path / "data"
    parent.mkdir()
    data.mkdir()
    artifact = parent / "last.pt"
    artifact.write_bytes(b"exact parent")
    expected = {"last.pt": hashlib.sha256(b"exact parent").hexdigest()}
    assert validate_file_hashes(parent, expected) == expected
    artifact.write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_file_hashes(parent, expected)

    output = tmp_path / "diagnostic"
    prepare_new_output_directory(
        output,
        data_root=data,
        parent_output_dir=parent,
    )
    assert output.is_dir()
    (output / "occupied.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        prepare_new_output_directory(
            output,
            data_root=data,
            parent_output_dir=parent,
        )
    with pytest.raises(ValueError, match="parent output must not overlap"):
        prepare_new_output_directory(
            parent / "child",
            data_root=data,
            parent_output_dir=parent,
        )
    with pytest.raises(ValueError, match="data root must not overlap"):
        prepare_new_output_directory(
            data / "child",
            data_root=data,
            parent_output_dir=parent,
        )


def test_restored_parent_identity_rejects_wrong_step_and_provenance() -> None:
    checkpoint = {
        "run_id": parent_trainer.RUN_ID,
        "provenance": {
            "config_digest": (
                "9949335070d23ecd8719a94c67d327e0aa53ce421ea1dffa27892b7fba599fd7"
            ),
            "input_digest": (
                "08d80fc972ae5ace8f4c3c5392c4fd36b53187f2d2a7655fff820257a2ec0074"
            ),
            "source_digest": (
                "a6f71c1c906ebbe8ab36566a8dcc28dd09d1feb1ac0b887bb548ec66570725ee"
            ),
            "runtime_digest": (
                "35e3f3f4bf8098eb73fd6b51fc42828a7ba59cee2c3200b0fea28b0bbd4371aa"
            ),
        },
    }
    history = [{"completed_step": 50}]
    validate_restored_parent_identity(
        checkpoint,
        completed_step=50,
        best_step=50,
        checkpoint_history=history,
        history_file={"rows": history},
    )
    with pytest.raises(ValueError, match="step-50"):
        validate_restored_parent_identity(
            checkpoint,
            completed_step=49,
            best_step=50,
            checkpoint_history=history,
            history_file={"rows": history},
        )
    checkpoint["provenance"] = {"drift": "yes"}
    with pytest.raises(ValueError, match="provenance"):
        validate_restored_parent_identity(
            checkpoint,
            completed_step=50,
            best_step=50,
            checkpoint_history=history,
            history_file={"rows": history},
        )


def test_localization_uses_frozen_lexicographic_order_and_domain_label() -> None:
    prediction = torch.zeros(2, 3, 12, 2, 2)
    prediction[0, 1, 4, 0, 0] = -11.0
    prediction[0, 1, 4, 0, 1] = -11.0
    prediction[0, 1, 5, 0, 0] = -11.0
    prediction[0, 2, 0, 0, 0] = -11.0
    prediction[1, 0, 0, 0, 0] = -11.0
    result = localize_decode_nonfiniteness(
        prediction,
        _normalizer(),
        case_keys=("first-case", "second-case"),
    )

    assert result["reproduced_decoded_nonfinite"] is True
    assert result["total_decoded_nonfinite_points"] == 5
    first = result["first_failure"]
    assert first["case_key"] == "first-case"
    assert first["call"] == 2
    assert first["channel_index"] == 4
    assert (first["row"], first["column"]) == (0, 0)
    assert first["mechanism"] == "inverse_domain_violation"
    assert first["inverse_domain_margin"]["value"] == pytest.approx(-0.1)
    assert first["decoded_value"] == {"value": None, "class": "nan"}
    assert len(result["first_failure_trace"]) == 2
    assert [row["case_key"] for row in result["first_failure_per_case"]] == [
        "first-case",
        "second-case",
    ]
    assert result["affected_calls"] == [1, 2, 3]
    assert result["affected_channels"] == [
        {"channel_index": 0, "field": "H"},
        {"channel_index": 4, "field": "HO2"},
        {"channel_index": 5, "field": "O"},
    ]
    json.dumps(result, allow_nan=False)


def test_localization_distinguishes_inverse_power_and_transformed_overflow() -> None:
    inverse_power = torch.zeros(1, 1, 12, 1, 1)
    inverse_power[0, 0, 0, 0, 0] = 1.0e10
    result = localize_decode_nonfiniteness(
        inverse_power,
        _normalizer(),
        case_keys=("case",),
    )
    first = result["first_failure"]
    assert first["mechanism"] == "inverse_power_overflow"
    assert first["transformed_value"]["class"] == "finite"
    assert first["inverse_box_cox_base"]["class"] == "finite"
    assert first["decoded_value"]["class"] == "positive_infinity"

    scale = torch.ones(12)
    scale[0] = torch.finfo(torch.float32).max
    transformed_overflow = torch.zeros(1, 1, 12, 1, 1)
    transformed_overflow[0, 0, 0, 0, 0] = 2.0
    overflow_result = localize_decode_nonfiniteness(
        transformed_overflow,
        _normalizer(scale=scale),
        case_keys=("case",),
    )
    overflow_first = overflow_result["first_failure"]
    assert overflow_first["mechanism"] == "transformed_overflow"
    assert overflow_first["transformed_value"]["class"] == "positive_infinity"
    json.dumps(overflow_result, allow_nan=False)


def test_localization_distinguishes_linear_decode_overflow() -> None:
    scale = torch.ones(12)
    scale[8] = torch.finfo(torch.float32).max
    prediction = torch.zeros(1, 1, 12, 1, 1)
    prediction[0, 0, 8, 0, 0] = 2.0
    result = localize_decode_nonfiniteness(
        prediction,
        _normalizer(scale=scale),
        case_keys=("case",),
    )
    first = result["first_failure"]
    assert first["field"] == "T"
    assert first["mechanism"] == "linear_decode_overflow"
    assert first["inverse_box_cox_base"] is None
    assert first["decoded_value"]["class"] == "positive_infinity"


def test_localization_reports_nonreproduction_and_rejects_normalized_nonfinite() -> (
    None
):
    finite = torch.zeros(2, 2, 12, 1, 1)
    result = localize_decode_nonfiniteness(
        finite,
        _normalizer(),
        case_keys=("a", "b"),
    )
    assert result["reproduced_decoded_nonfinite"] is False
    assert result["first_failure"] is None
    assert result["first_failure_trace"] == []
    json.dumps(result, allow_nan=False)

    finite[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="finite normalized"):
        localize_decode_nonfiniteness(
            finite,
            _normalizer(),
            case_keys=("a", "b"),
        )


class _TinyDirectModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.8], dtype=torch.float32))

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        return state * self.scale.view(1, 1, 1, 1)


def _reference_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: random.Random,
    train: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    count: int,
) -> None:
    for _ in range(count):
        sample = parent_trainer.draw_step_sample(generator)
        optimizer.zero_grad(set_to_none=True)
        for case_index in sample.case_indices:
            prediction = model(
                train[case_index, sample.frame_start].unsqueeze(0),
                coordinates,
            )
            target = train[case_index, sample.frame_start + 1].unsqueeze(0)
            loss, _ = grouped_next_state_mse(prediction, target)
            parent_trainer.scaled_microbatch_loss(
                loss,
                effective_batch_size=parent_trainer.EFFECTIVE_BATCH_SIZE,
            ).backward()
        optimizer.step()
        scheduler.step()


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_registered_replay_matches_exact_restored_cpu_continuation() -> None:
    torch.manual_seed(19)
    train = torch.randn(len(TRAIN_GROUPS), 30, 12, 1, 1) * 0.1
    coordinates = torch.zeros(1, 2, 1, 1)
    parent_model = _TinyDirectModel()
    parent_optimizer, parent_scheduler = parent_trainer.build_optimizer_and_scheduler(
        parent_model
    )
    parent_generator = random.Random(parent_trainer.SEED)
    _reference_steps(
        parent_model,
        parent_optimizer,
        parent_scheduler,
        parent_generator,
        train,
        coordinates,
        count=PARENT_COMPLETED_STEP,
    )
    provenance = {"synthetic": "cpu"}
    signature = canonical_json_sha256(provenance)
    parent_digest = parent_trainer.structured_state_sha256(parent_model.state_dict())
    checkpoint = parent_trainer.build_last_checkpoint(
        parent_model,
        parent_optimizer,
        parent_scheduler,
        parent_generator,
        run_signature=signature,
        completed_step=PARENT_COMPLETED_STEP,
        best_step=PARENT_COMPLETED_STEP,
        best_score=1.0,
        best_model_state_sha256=parent_digest,
        history=(),
        provenance=provenance,
    )

    reference_model = _TinyDirectModel()
    reference_optimizer, reference_scheduler = (
        parent_trainer.build_optimizer_and_scheduler(reference_model)
    )
    reference_generator = random.Random(999)
    parent_trainer.restore_last_checkpoint(
        copy.deepcopy(checkpoint),
        reference_model,
        reference_optimizer,
        reference_scheduler,
        reference_generator,
        expected_run_signature=signature,
    )
    _reference_steps(
        reference_model,
        reference_optimizer,
        reference_scheduler,
        reference_generator,
        train,
        coordinates,
        count=len(REPLAY_STEPS),
    )
    replay_model = _TinyDirectModel()
    replay_optimizer, replay_scheduler = parent_trainer.build_optimizer_and_scheduler(
        replay_model
    )
    replay_generator = random.Random(123)
    parent_trainer.restore_last_checkpoint(
        copy.deepcopy(checkpoint),
        replay_model,
        replay_optimizer,
        replay_scheduler,
        replay_generator,
        expected_run_signature=signature,
    )
    trace = replay_registered_steps(
        replay_model,
        replay_optimizer,
        replay_scheduler,
        replay_generator,
        train,
        coordinates,
        device=torch.device("cpu"),
    )

    _assert_nested_equal(reference_model.state_dict(), replay_model.state_dict())
    _assert_nested_equal(
        reference_optimizer.state_dict(), replay_optimizer.state_dict()
    )
    _assert_nested_equal(
        reference_scheduler.state_dict(), replay_scheduler.state_dict()
    )
    assert parent_trainer.draw_step_sample(
        reference_generator
    ) == parent_trainer.draw_step_sample(replay_generator)
    assert trace["optimizer_step_count"] == 50
    assert trace["rows"][0]["completed_step"] == 51
    assert trace["rows"][-1]["completed_step"] == 100
    assert trace["final_model_state_sha256"] == (
        parent_trainer.structured_state_sha256(replay_model.state_dict())
    )


def test_step100_checkpoint_binds_parent_source_runtime_and_normalizer() -> None:
    model = _TinyDirectModel()
    normalizer_state = {
        "mean": torch.zeros(12),
        "scale": torch.ones(12),
        "source_sha256": "a" * 64,
    }
    checkpoint = build_step100_model_checkpoint(
        model,
        normalizer_state=normalizer_state,
        diagnostic_run_signature="b" * 64,
        diagnostic_source_digest="c" * 64,
        runtime_digest="d" * 64,
        replay_trace_digest="e" * 64,
    )

    assert checkpoint["schema"] == MODEL_CHECKPOINT_SCHEMA
    assert checkpoint["completed_step"] == 100
    assert checkpoint["parent_last_sha256"] == PARENT_LAST_SHA256
    assert checkpoint["inference_only"] is True
    assert checkpoint["resume_supported"] is False
    assert checkpoint["model_state_sha256"] == (
        parent_trainer.structured_state_sha256(checkpoint["model_state"])
    )
    assert checkpoint["normalizer_state_sha256"] == (
        parent_trainer.structured_state_sha256(checkpoint["normalizer_state"])
    )


def test_help_is_cpu_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert torch.cuda.is_initialized() is False
    with pytest.raises(SystemExit) as exc_info:
        diagnostic_main(["--help"])
    assert exc_info.value.code == 0
    assert "step-100 decode failure" in capsys.readouterr().out
    assert torch.cuda.is_initialized() is False
