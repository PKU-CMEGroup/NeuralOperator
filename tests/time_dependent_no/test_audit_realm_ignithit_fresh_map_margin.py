from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.time_dependent_no import (
    audit_realm_ignithit_fresh_map_margin as audit_module,
)
from scripts.time_dependent_no import (
    diagnose_realm_ignithit_decode_failure as localization,
)
from scripts.time_dependent_no import train_realm_ignithit_ffno as parent_trainer
from scripts.time_dependent_no.audit_realm_ignithit_fresh_map_margin import (
    FRAME_START,
    FRAME_TARGET,
    LOCALIZATION_FILE_SHA256,
    RUN_ID,
    STEP50_MODEL_STATE_SHA256,
    STEP100_MODEL_STATE_SHA256,
    _canonical_payload_sha256,
    _source_manifest,
    audit_contract,
    build_parser,
    infer_fresh_map_pair,
    prepare_new_output_directory,
    summarize_fresh_map_margin_audit,
    validate_known_step100_reproduction,
    validate_step50_checkpoint,
    validate_step100_checkpoint,
)
from utility.time_dependent_no.realm_benchmark import (
    RealmNormalizer,
    canonical_json_sha256,
)


def _normalizer() -> RealmNormalizer:
    return RealmNormalizer(
        mean=torch.zeros(12, dtype=torch.float32),
        scale=torch.ones(12, dtype=torch.float32),
        transformed_channels=tuple(range(8)),
        channel_axis=1,
        box_cox_lambda=0.1,
        box_cox_epsilon=1.0e-8,
        std_correction=1,
        scale_stabilizer=1.0e-10,
    )


def _synthetic_audit() -> dict[str, object]:
    truth = torch.zeros(2, 12, 3, 4)
    step50 = truth.clone()
    step100 = truth.clone()
    step100[0, 2, 0, 1] = -11.0
    step100[0, 2, 1, 2] = -12.0
    step100[1, 5, 2, 3] = -11.0
    return summarize_fresh_map_margin_audit(
        step50,
        step100,
        truth,
        _normalizer(),
        case_keys=("case-a", "case-b"),
    )


def test_contract_freezes_one_fresh_call_and_knob_free_cli() -> None:
    assert FRAME_START == 0
    assert FRAME_TARGET == 1
    assert len(STEP50_MODEL_STATE_SHA256) == len(STEP100_MODEL_STATE_SHA256) == 64
    assert LOCALIZATION_FILE_SHA256["step100_model.pt"] == (
        "44b09c34a0db4dce518e71caa0d82fc08056e368b4f33768003fe489c1c0c039"
    )
    contract = audit_contract()
    assert contract["run_id"] == RUN_ID
    assert contract["stable_result_id"] == "D088"
    assert contract["calls_per_checkpoint"] == 1
    assert contract["recurrence"] == "none"
    assert contract["training"] == "none"
    assert contract["output_repair"] == "none"
    assert contract["test_object_available"] is False
    assert contract["residual_arm_authorized"] is False
    assert contract["canonical_payload_sha256"] == _canonical_payload_sha256(contract)

    argument_names = {
        action.dest for action in build_parser()._actions if action.dest != "help"
    }
    assert argument_names == {
        "manifest",
        "data_root",
        "normalizer_arrays",
        "parent_output_dir",
        "localization_output_dir",
        "output_dir",
    }


def test_source_manifest_binds_every_executed_source() -> None:
    source = _source_manifest()
    assert [row["path"] for row in source["files"]] == [
        "utility/time_dependent_no/realm_benchmark.py",
        "utility/time_dependent_no/realm_ignithit.py",
        "utility/time_dependent_no/realm_ffno.py",
        "scripts/time_dependent_no/train_realm_ignithit_ffno.py",
        "scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py",
        "scripts/time_dependent_no/audit_realm_ignithit_fresh_map_margin.py",
    ]
    assert source["canonical_payload_sha256"] == _canonical_payload_sha256(source)
    assert Path(parent_trainer.__file__).is_file()
    assert Path(localization.__file__).is_file()


def test_output_isolation_rejects_each_protected_tree_and_occupied_output(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    parent = tmp_path / "parent"
    localization_output = tmp_path / "localization"
    for path in (data, parent, localization_output):
        path.mkdir()
    output = tmp_path / "audit"
    prepare_new_output_directory(
        output,
        data_root=data,
        parent_output_dir=parent,
        localization_output_dir=localization_output,
    )
    (output / "occupied.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        prepare_new_output_directory(
            output,
            data_root=data,
            parent_output_dir=parent,
            localization_output_dir=localization_output,
        )
    for protected, match in (
        (data, "data root"),
        (parent, "parent output"),
        (localization_output, "localization output"),
    ):
        with pytest.raises(ValueError, match=match):
            prepare_new_output_directory(
                protected / "child",
                data_root=data,
                parent_output_dir=parent,
                localization_output_dir=localization_output,
            )


def test_paired_margin_audit_localizes_new_step100_support_case_first() -> None:
    result = _synthetic_audit()
    assert result["schema"] == "d088_ignithit_fresh_map_margin_audit_v1"
    assert result["case_count"] == 2
    assert result["calls_per_checkpoint"] == 1
    assert result["checkpoints"]["step50"]["total_inverse_domain_violation_points"] == 0
    assert (
        result["checkpoints"]["step100"]["total_inverse_domain_violation_points"] == 3
    )
    assert result["truth"]["total_inverse_domain_violation_points"] == 0
    assert result["paired"]["classification"] == (
        "checkpoint_history_associated_margin_regression"
    )
    assert result["paired"]["checkpoint_history_associated_margin_regression"] is True
    assert result["paired"]["new_step100_inverse_domain_violation_points"] == 3
    assert result["paired"]["outcome_counts"] == {
        "new_step100_inverse_domain_violation": 2,
        "valid_at_both_checkpoints": 14,
    }

    first = result["checkpoints"]["step100"]["first_inverse_domain_violation"]
    assert first == {
        "case_index": 0,
        "case_key": "case-a",
        "channel_index": 2,
        "field": "H2O",
        "row": 0,
        "column": 1,
        "normalized_value": -11.0,
        "transformed_value": -11.0,
        "inverse_domain_margin": pytest.approx(-0.1),
    }
    h2o = next(
        row
        for row in result["checkpoints"]["step100"]["inverse_domain_margin"]
        if row["case_key"] == "case-a" and row["field"] == "H2O"
    )
    assert h2o["invalid_support"] == {
        "count": 2,
        "fraction": pytest.approx(2 / 12),
        "first": {"row": 0, "column": 1},
        "bbox_inclusive": {
            "row_min": 0,
            "row_max": 1,
            "column_min": 1,
            "column_max": 2,
        },
    }
    paired_h2o = next(
        row
        for row in result["paired"]["per_case_transformed_field"]
        if row["case_key"] == "case-a" and row["field"] == "H2O"
    )
    assert paired_h2o["status"] == "new_step100_inverse_domain_violation"
    assert paired_h2o["new_step100_support_count"] == 2
    assert paired_h2o["invalid_support_jaccard"] == pytest.approx(0.0)
    assert paired_h2o["minimum_margin_truth"] == pytest.approx(1.0)
    assert paired_h2o["minimum_margin_step50"] == pytest.approx(1.0)
    assert paired_h2o["minimum_margin_step100"] == pytest.approx(-0.2)
    json.dumps(result, allow_nan=False)


def test_truth_domain_failure_blocks_checkpoint_history_classification() -> None:
    truth = torch.zeros(1, 12, 2, 2)
    truth[0, 0, 0, 0] = -11.0
    step50 = truth.clone()
    step100 = truth.clone()
    step100[0, 1, 0, 1] = -12.0
    result = summarize_fresh_map_margin_audit(
        step50,
        step100,
        truth,
        _normalizer(),
        case_keys=("case",),
    )

    assert result["truth"]["total_inverse_domain_violation_points"] == 1
    assert result["paired"]["classification"] == ("truth_or_normalizer_domain_failure")
    assert result["paired"]["checkpoint_history_associated_margin_regression"] is False
    outcome_counts = result["paired"]["outcome_counts"]
    assert outcome_counts["truth_inverse_domain_violation"] == 1


def test_existing_invalid_support_and_empty_jaccard_are_reason_coded() -> None:
    truth = torch.zeros(1, 12, 2, 2)
    step50 = truth.clone()
    step100 = truth.clone()
    step50[0, 3, 0, 0] = -11.0
    step100[0, 3, 1, 1] = -11.0
    result = summarize_fresh_map_margin_audit(
        step50,
        step100,
        truth,
        _normalizer(),
        case_keys=("case",),
    )
    row = next(
        row
        for row in result["paired"]["per_case_transformed_field"]
        if row["field"] == "H2O2"
    )
    assert row["status"] == "invalid_at_both_checkpoints"
    assert row["invalid_support_intersection_count"] == 0
    assert row["invalid_support_union_count"] == 2
    assert row["invalid_support_jaccard"] == pytest.approx(0.0)
    valid_row = next(
        row
        for row in result["paired"]["per_case_transformed_field"]
        if row["field"] == "H"
    )
    assert valid_row["invalid_support_jaccard"] is None
    assert valid_row["invalid_support_jaccard_status"] == "empty_union"


class _RecordingMap(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        self.calls += 1
        self.inputs.append(state.detach().clone())
        return state + self.offset


class _MutatingMap(nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        state.add_(1.0)
        return state


def test_inference_pair_calls_each_map_once_on_the_same_immutable_input() -> None:
    initial = torch.zeros(2, 12, 2, 3)
    coordinates = torch.zeros(1, 2, 2, 3)
    initial_before = initial.clone()
    coordinates_before = coordinates.clone()
    step50_model = _RecordingMap(1.0)
    step100_model = _RecordingMap(2.0)
    step50_model.train()
    step100_model.eval()

    prediction50, prediction100 = infer_fresh_map_pair(
        step50_model,
        step100_model,
        initial,
        coordinates,
    )

    assert step50_model.calls == step100_model.calls == 1
    assert torch.equal(step50_model.inputs[0], initial_before)
    assert torch.equal(step100_model.inputs[0], initial_before)
    assert torch.equal(prediction50, torch.ones_like(initial))
    assert torch.equal(prediction100, torch.full_like(initial, 2.0))
    assert torch.equal(initial, initial_before)
    assert torch.equal(coordinates, coordinates_before)
    assert step50_model.training is True
    assert step100_model.training is False


def test_inference_pair_rejects_mutation_before_calling_second_map() -> None:
    initial = torch.zeros(1, 12, 2, 2)
    coordinates = torch.zeros(1, 2, 2, 2)
    step100_model = _RecordingMap(0.0)
    with pytest.raises(RuntimeError, match="step-50 fresh-map call mutated"):
        infer_fresh_map_pair(
            _MutatingMap(),
            step100_model,
            initial,
            coordinates,
        )
    assert step100_model.calls == 0


def test_input_guards_reject_shape_nonfinite_and_normalizer_drift() -> None:
    truth = torch.zeros(1, 12, 2, 2)
    with pytest.raises(ValueError, match="shapes must match"):
        summarize_fresh_map_margin_audit(
            truth[:, :, :, :1],
            truth,
            truth,
            _normalizer(),
            case_keys=("case",),
        )
    nonfinite = truth.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="finite normalized"):
        summarize_fresh_map_margin_audit(
            truth,
            nonfinite,
            truth,
            _normalizer(),
            case_keys=("case",),
        )
    wrong_axis = copy.copy(_normalizer())
    object.__setattr__(wrong_axis, "channel_axis", 2)
    with pytest.raises(ValueError, match="channel axis 1"):
        summarize_fresh_map_margin_audit(
            truth,
            truth,
            truth,
            wrong_axis,
            case_keys=("case",),
        )


def test_checkpoint_guards_bind_steps_links_states_and_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_digest = parent_trainer.structured_state_sha256
    step50_state = {"weight": torch.tensor([1.0])}
    step100_state = {"weight": torch.tensor([2.0])}
    normalizer_state = {"mean": torch.zeros(12), "scale": torch.ones(12)}
    normalizer_digest = original_digest(normalizer_state)

    def fake_digest(value: object) -> str:
        if value is step50_state:
            return STEP50_MODEL_STATE_SHA256
        if value is step100_state:
            return STEP100_MODEL_STATE_SHA256
        return original_digest(value)

    monkeypatch.setattr(parent_trainer, "structured_state_sha256", fake_digest)
    step50_checkpoint = {
        "schema": parent_trainer.LAST_CHECKPOINT_SCHEMA,
        "run_id": localization.PARENT_RUN_ID,
        "run_signature": localization.PARENT_RUN_SIGNATURE,
        "provenance": localization.PARENT_PROVENANCE,
        "completed_step": 50,
        "best_step": 50,
        "resume_supported": True,
        "model_state": step50_state,
        "model_state_sha256": STEP50_MODEL_STATE_SHA256,
    }
    assert validate_step50_checkpoint(step50_checkpoint) is step50_state
    damaged50 = dict(step50_checkpoint, completed_step=49)
    with pytest.raises(ValueError, match="step-50 checkpoint identity"):
        validate_step50_checkpoint(damaged50)

    source = {"canonical_payload_sha256": "a" * 64}
    runtime = {"canonical_payload_sha256": "b" * 64}
    replay = {"schema": "trace", "rows": []}
    summary = {"diagnostic_run_signature": "c" * 64}
    step100_checkpoint = {
        "schema": localization.MODEL_CHECKPOINT_SCHEMA,
        "run_id": localization.RUN_ID,
        "parent_run_id": localization.PARENT_RUN_ID,
        "parent_last_sha256": localization.PARENT_LAST_SHA256,
        "parent_run_signature": localization.PARENT_RUN_SIGNATURE,
        "diagnostic_run_signature": summary["diagnostic_run_signature"],
        "completed_step": 100,
        "model_state": step100_state,
        "model_state_sha256": STEP100_MODEL_STATE_SHA256,
        "normalizer_state": normalizer_state,
        "normalizer_state_sha256": normalizer_digest,
        "diagnostic_source_digest": source["canonical_payload_sha256"],
        "runtime_digest": runtime["canonical_payload_sha256"],
        "replay_trace_digest": canonical_json_sha256(replay),
        "inference_only": True,
        "resume_supported": False,
        "test_object_opened": False,
    }
    assert (
        validate_step100_checkpoint(
            step100_checkpoint,
            normalizer_state_sha256=normalizer_digest,
            localization_summary=summary,
            localization_source=source,
            localization_runtime=runtime,
            replay_trace=replay,
        )
        is step100_state
    )
    damaged100 = dict(step100_checkpoint, diagnostic_source_digest="d" * 64)
    with pytest.raises(ValueError, match="step-100 checkpoint identity"):
        validate_step100_checkpoint(
            damaged100,
            normalizer_state_sha256=normalizer_digest,
            localization_summary=summary,
            localization_source=source,
            localization_runtime=runtime,
            replay_trace=replay,
        )


def test_known_result_reproduction_requires_location_values_and_support_count() -> None:
    prediction50 = torch.zeros(1, 12, 9, 9)
    prediction100 = prediction50.clone()
    prediction100[0, 2].reshape(-1)[:66] = -11.0
    truth = prediction50.clone()
    audit = summarize_fresh_map_margin_audit(
        prediction50,
        prediction100,
        truth,
        _normalizer(),
        case_keys=("case",),
    )
    first = audit["checkpoints"]["step100"]["first_inverse_domain_violation"]
    retained = {
        "first_failure": {
            "call": 1,
            "case_key": first["case_key"],
            "channel_index": first["channel_index"],
            "field": first["field"],
            "row": first["row"],
            "column": first["column"],
            "normalized_value": {"value": first["normalized_value"]},
            "inverse_domain_margin": {"value": first["inverse_domain_margin"]},
        }
    }
    validate_known_step100_reproduction(audit, retained)
    damaged = copy.deepcopy(retained)
    damaged["first_failure"]["column"] += 1
    with pytest.raises(RuntimeError, match="location differs"):
        validate_known_step100_reproduction(audit, damaged)


def test_main_help_does_not_enter_scientific_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        audit_module,
        "run_audit",
        lambda args: pytest.fail(f"scientific runner invoked: {args}"),
    )
    with pytest.raises(SystemExit) as exc:
        audit_module.main(["--help"])
    assert exc.value.code == 0
    assert "--localization-output-dir" in capsys.readouterr().out
