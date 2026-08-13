from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from scripts.time_dependent_no import (
    diagnose_realm_ignithit_boundedness_attribution as diagnostic,
)
from utility.time_dependent_no.realm_benchmark import (
    MagnitudeEnvelope,
    canonical_json_sha256,
)
from utility.time_dependent_no.realm_boundedness_attribution import (
    attribute_free_first_events,
    channel_envelope_records,
    evaluate_normalized_modes,
    summarize_attribution_verdict,
    summarize_envelope_events,
    validate_recurrence_trace,
)


def _envelope(values: tuple[float, ...]) -> MagnitudeEnvelope:
    return MagnitudeEnvelope(
        max_abs=torch.tensor(values, dtype=torch.float64),
        quantile=1.0,
        channel_axis=0,
    )


def _records(
    states: torch.Tensor,
    *,
    checkpoint_id: str = "checkpoint",
    mode: str = "free_recurrence",
) -> list[dict[str, Any]]:
    return channel_envelope_records(
        states,
        _envelope((1.0, 0.0)),
        expansion_factor=10.0,
        case_keys=("case",),
        channel_names=("a", "b"),
        checkpoint_id=checkpoint_id,
        mode=mode,
    )


def test_contract_allocates_exact_scope_and_knob_free_cli() -> None:
    assert diagnostic.STABLE_ID == "D091"
    assert diagnostic.RUN_ID == (
        "d091_realm_ignithit_p1d_boundedness_attribution_20260813a"
    )
    contract = diagnostic.diagnostic_contract()
    assert contract["checkpoint_ids"] == [
        diagnostic.STEP50_ID,
        diagnostic.STEP100_ID,
    ]
    assert contract["modes"] == ["free_recurrence", "teacher_forced"]
    assert contract["calls"] == 29
    assert contract["boundedness_threshold_inclusive"] is True
    assert contract["spatial_support"] == (
        "strict_abs_gt_inclusive_limit_count_and_minimum_row_column_bbox"
    )
    assert contract["attribution_verdict"] == (
        "event_channel_counts_with_mixed_and_censored_outcomes_preserved"
    )
    assert contract["finite_unbounded_feedback"] == (
        "unchanged_deployed_proposal"
    )
    assert contract["selection"] == "none"
    assert contract["test_object_available"] is False
    digest = contract.pop("canonical_payload_sha256")
    assert digest == canonical_json_sha256(contract)
    assert digest == "d4969ab61d4391958e34a00b01e176a879f8132addf953d0bfffd819a908294c"

    argument_names = {
        action.dest
        for action in diagnostic.build_parser()._actions
        if action.dest != "help"
    }
    assert argument_names == {
        "manifest",
        "data_root",
        "normalizer_arrays",
        "d089_output_dir",
        "d090_output_dir",
        "output_dir",
        "expected_source_digest",
    }


def test_source_manifest_binds_model_adapter_metric_attribution_and_entry() -> None:
    source = diagnostic.source_manifest()
    paths = [row["path"] for row in source["files"]]
    assert paths == [
        "utility/time_dependent_no/realm_benchmark.py",
        "utility/time_dependent_no/realm_ignithit.py",
        "utility/time_dependent_no/realm_ffno.py",
        "utility/time_dependent_no/realm_domain_link.py",
        "utility/time_dependent_no/realm_boundedness_attribution.py",
        "scripts/time_dependent_no/train_realm_ignithit_ffno.py",
        "scripts/time_dependent_no/train_realm_ignithit_domain_linked_ffno.py",
        "scripts/time_dependent_no/diagnose_realm_ignithit_decode_failure.py",
        "scripts/time_dependent_no/capture_realm_ignithit_domain_link_failure.py",
        "scripts/time_dependent_no/diagnose_realm_ignithit_boundedness_attribution.py",
    ]
    digest = source.pop("canonical_payload_sha256")
    assert digest == canonical_json_sha256(source)
    assert digest == "f0543bce94c23a285dd125d4079ccf1142b8228d8a6d58b2e093202795e2facb"


def test_exact_inventory_and_output_isolation_fail_closed(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    payload = b"exact"
    expected = hashlib.sha256(payload).hexdigest()
    (parent / "file.bin").write_bytes(payload)
    assert diagnostic.validate_exact_inventory(
        parent, {"file.bin": expected}, name="synthetic"
    ) == {"file.bin": expected}

    (parent / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        diagnostic.validate_exact_inventory(
            parent, {"file.bin": expected}, name="synthetic"
        )
    (parent / "extra.bin").unlink()
    (parent / "file.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        diagnostic.validate_exact_inventory(
            parent, {"file.bin": expected}, name="synthetic"
        )

    data = tmp_path / "data"
    data.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    normalizer = tmp_path / "normalizer.npz"
    normalizer.write_bytes(b"normalizer")
    output = tmp_path / "output"
    diagnostic.prepare_new_output_directory(
        output,
        protected_dirs=(data, parent),
        protected_files=(manifest, normalizer),
    )
    assert output.is_dir()
    (output / "occupied").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        diagnostic.prepare_new_output_directory(
            output,
            protected_dirs=(data, parent),
            protected_files=(manifest, normalizer),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        diagnostic.prepare_new_output_directory(
            data / "child",
            protected_dirs=(data, parent),
            protected_files=(manifest, normalizer),
        )


def test_channel_records_use_inclusive_limit_zero_reason_codes_and_row_major_argmax() -> None:
    states = torch.zeros(1, 2, 2, 2, 3, dtype=torch.float64)
    states[0, 0, 0, 1, 0] = -10.0  # ratio exactly one: bounded
    states[0, 1, 0, 0, 2] = 10.0001  # first row-major maximum: unbounded
    states[0, 1, 0, 1, 1] = -10.0001
    states[0, 1, 1, 1, 2] = 0.5  # positive observation under zero limit

    records = _records(states)
    exact = next(
        row for row in records if row["call"] == 1 and row["channel"] == "a"
    )
    assert exact["ratio"] == 1.0
    assert exact["bounded"] is True
    assert exact["signed_value_at_argmax"] == -10.0
    assert (exact["argmax_row"], exact["argmax_column"]) == (1, 0)

    failed = next(
        row for row in records if row["call"] == 2 and row["channel"] == "a"
    )
    assert failed["ratio"] == pytest.approx(1.00001)
    assert failed["bounded"] is False
    assert (failed["argmax_row"], failed["argmax_column"]) == (0, 2)
    assert failed["exceedance_count"] == 2
    assert failed["exceedance_support"] == {
        "row_min": 0,
        "row_max": 1,
        "column_min": 1,
        "column_max": 2,
    }

    zero = next(
        row for row in records if row["call"] == 1 and row["channel"] == "b"
    )
    assert zero["ratio"] == 0.0
    assert zero["ratio_status"] == "zero_limit_zero_observed"
    positive = next(
        row for row in records if row["call"] == 2 and row["channel"] == "b"
    )
    assert positive["ratio"] is None
    assert positive["ratio_status"] == "positive_infinite_zero_limit"
    assert positive["bounded"] is False
    json.dumps(records, allow_nan=False)


def test_events_use_first_failure_prefix_and_preserve_raw_reentry() -> None:
    states = torch.zeros(1, 4, 2, 1, 1, dtype=torch.float64)
    states[0, :, 0, 0, 0] = torch.tensor(
        [9.0, 11.0, 8.0, 12.0], dtype=torch.float64
    )
    records = _records(states)
    summary = summarize_envelope_events(
        records,
        case_keys=("case",),
        channel_names=("a", "b"),
        checkpoint_id="checkpoint",
        mode="free_recurrence",
        horizon=4,
    )
    row = summary["per_case"][0]
    assert row["bounded_calls"] == 2
    assert row["boundedness"] == {
        "first_failure_call": 2,
        "accepted_prefix": 1,
        "censored": False,
        "censor_call": None,
    }
    assert row["raw_reentry_calls"] == [3]
    channel_b = next(
        item for item in row["channel_events"] if item["channel"] == "b"
    )
    assert channel_b["boundedness"] == {
        "first_failure_call": None,
        "accepted_prefix": 4,
        "censored": True,
        "censor_call": 4,
    }


def test_returned_nonfinite_fails_that_call_and_has_strict_json_records() -> None:
    states = torch.zeros(1, 2, 2, 1, 1, dtype=torch.float64)
    states[0, 1, 0, 0, 0] = torch.nan
    records = _records(states)
    event = summarize_envelope_events(
        records,
        case_keys=("case",),
        channel_names=("a", "b"),
        checkpoint_id="checkpoint",
        mode="free_recurrence",
        horizon=2,
    )["per_case"][0]
    assert event["decoded_finiteness"]["first_failure_call"] == 2
    assert event["decoded_finiteness"]["accepted_prefix"] == 1
    assert event["boundedness"]["first_failure_call"] == 2
    nonfinite = next(
        row for row in records if row["call"] == 2 and row["channel"] == "a"
    )
    assert nonfinite["ratio"] is None
    assert nonfinite["ratio_status"] == "native_nonfinite"
    assert nonfinite["bounded"] is False
    summary = summarize_envelope_events(
        records,
        case_keys=("case",),
        channel_names=("a", "b"),
        checkpoint_id="checkpoint",
        mode="free_recurrence",
        horizon=2,
    )
    assert summary["maximum_ratio"] is None
    assert summary["maximum_ratio_status"] == "unavailable_native_nonfinite"

    continued = torch.zeros(1, 3, 2, 1, 1, dtype=torch.float64)
    continued[0, 1, 0, 0, 0] = torch.nan
    independent = _records(
        continued,
        mode="teacher_forced",
    )
    assert any(record["call"] == 3 for record in independent)
    json.dumps(records, allow_nan=False)


class _ScaleMap(nn.Module):
    def __init__(self, scale: float, *, nonfinite_call: int | None = None) -> None:
        super().__init__()
        self.scale = scale
        self.nonfinite_call = nonfinite_call
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def forward(
        self, state: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        del coordinates
        self.inputs.append(state.detach().clone())
        self.calls += 1
        proposal = state * self.scale
        if self.nonfinite_call == self.calls:
            proposal = proposal.clone()
            proposal[0, 0, 0, 0] = torch.inf
        return proposal


def test_matched_modes_feed_finite_unbounded_free_state_but_stop_after_nonfinite() -> None:
    truth = torch.ones(1, 4, 1, 1, 1, dtype=torch.float64)
    coordinates = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    model = _ScaleMap(2.0)
    model.train()
    predictions = evaluate_normalized_modes(model, truth, coordinates, calls=3)
    assert model.training is True
    assert predictions.free_recurrence.flatten().tolist() == [2.0, 4.0, 8.0]
    assert predictions.teacher_forced.flatten().tolist() == [2.0, 2.0, 2.0]
    assert torch.equal(model.inputs[1], predictions.free_recurrence[:, 0])
    assert torch.equal(model.inputs[2], predictions.free_recurrence[:, 1])
    assert torch.equal(predictions.teacher_inputs, truth[:, :3])
    validate_recurrence_trace(
        predictions.free_inputs,
        predictions.free_recurrence,
    )
    drifted = predictions.free_inputs.clone()
    drifted[:, 1] += 1.0
    with pytest.raises(ValueError, match="exact prior deployed"):
        validate_recurrence_trace(drifted, predictions.free_recurrence)

    stopped = _ScaleMap(2.0, nonfinite_call=2)
    result = evaluate_normalized_modes(stopped, truth, coordinates, calls=3)
    assert result.free_recurrence.shape[1] == 2
    assert torch.isinf(result.free_recurrence[:, 1]).any()
    # Two free calls plus all three independent teacher-forced calls.
    assert stopped.calls == 5


def test_attribution_separates_fresh_sufficiency_and_propagated_requirement() -> None:
    free = torch.zeros(1, 2, 2, 1, 1, dtype=torch.float64)
    teacher = torch.zeros_like(free)
    free[0, 1, :, 0, 0] = 11.0
    teacher[0, 1, 0, 0, 0] = 11.0
    teacher[0, 1, 1, 0, 0] = 0.0
    envelope = _envelope((1.0, 1.0))
    free_records = channel_envelope_records(
        free,
        envelope,
        expansion_factor=10.0,
        case_keys=("case",),
        channel_names=("fresh", "propagated"),
        checkpoint_id="checkpoint",
        mode="free_recurrence",
    )
    teacher_records = channel_envelope_records(
        teacher,
        envelope,
        expansion_factor=10.0,
        case_keys=("case",),
        channel_names=("fresh", "propagated"),
        checkpoint_id="checkpoint",
        mode="teacher_forced",
    )
    event_channels = attribute_free_first_events(
        free_records,
        teacher_records,
        case_keys=("case",),
        channel_names=("fresh", "propagated"),
        checkpoint_id="checkpoint",
        horizon=2,
    )["cases"][0]["event_channels"]
    labels = {row["channel"]: row["label"] for row in event_channels}
    assert labels == {
        "fresh": "fresh_exact_input_sufficient_for_threshold_at_free_event",
        "propagated": "propagated_input_required_for_threshold_at_free_event",
    }

    verdict = summarize_attribution_verdict(
        {
            "checkpoint": attribute_free_first_events(
                free_records,
                teacher_records,
                case_keys=("case",),
                channel_names=("fresh", "propagated"),
                checkpoint_id="checkpoint",
                horizon=2,
            )
        }
    )
    assert verdict["verdict"] == "mixed_first_event_attribution"
    assert verdict["event_channel_counts"] == {
        "fresh_exact_input_sufficient_for_threshold_at_free_event": 1,
        "propagated_input_required_for_threshold_at_free_event": 1,
    }


def test_truth_records_are_independent_of_model_records() -> None:
    truth = torch.zeros(1, 1, 2, 1, 1, dtype=torch.float64)
    prediction = torch.full_like(truth, 11.0)
    truth_records = _records(
        truth, checkpoint_id="matching_truth", mode="matching_truth"
    )
    prediction_records = _records(prediction)
    assert all(record["bounded"] is True for record in truth_records)
    assert any(record["bounded"] is False for record in prediction_records)


def test_record_grid_rejects_duplicates_and_missing_cells() -> None:
    records = _records(torch.zeros(1, 1, 2, 1, 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_envelope_events(
            [*records, records[0]],
            case_keys=("case",),
            channel_names=("a", "b"),
            checkpoint_id="checkpoint",
            mode="free_recurrence",
            horizon=1,
        )
    with pytest.raises(ValueError, match="exact case/call/channel grid"):
        summarize_envelope_events(
            records[:-1],
            case_keys=("case",),
            channel_names=("a", "b"),
            checkpoint_id="checkpoint",
            mode="free_recurrence",
            horizon=1,
        )


def test_checkpoint_identity_guards_bind_roles_history_and_model_states() -> None:
    state = {"weight": torch.tensor([1.0])}
    digest = diagnostic.parent.structured_state_sha256(state)
    history = {"rows": [{"completed_step": 50}]}
    step50 = {
        "schema": diagnostic.d089.LAST_CHECKPOINT_SCHEMA,
        "run_id": diagnostic.d089.RUN_ID,
        "run_signature": diagnostic.d090.PARENT_RUN_SIGNATURE,
        "completed_step": 50,
        "model_state": state,
        "model_state_sha256": digest,
        "provenance": diagnostic.d090.PARENT_PROVENANCE,
        "resume_supported": True,
        "best_step": 1,
        "best_score": diagnostic.d090.PARENT_BEST_SCORE,
        "best_model_state_sha256": diagnostic.d090.PARENT_BEST_MODEL_STATE_SHA256,
        "history": history["rows"],
    }
    original = diagnostic.d090.PARENT_LAST_MODEL_STATE_SHA256
    try:
        diagnostic.d090.PARENT_LAST_MODEL_STATE_SHA256 = digest
        diagnostic.validate_step50_checkpoint(step50, history)
        step50["history"] = []
        with pytest.raises(ValueError, match="D089 history"):
            diagnostic.validate_step50_checkpoint(step50, history)
    finally:
        diagnostic.d090.PARENT_LAST_MODEL_STATE_SHA256 = original

    normalizer = {"mean": torch.zeros(1)}
    normalizer_digest = diagnostic.parent.structured_state_sha256(normalizer)
    step100 = {
        "schema": diagnostic.d090.CAPTURE_CHECKPOINT_SCHEMA,
        "run_id": diagnostic.d090.RUN_ID,
        "diagnostic_run_signature": diagnostic.D090_RUN_SIGNATURE,
        "model_state": state,
        "model_state_sha256": digest,
        "normalizer_state": normalizer,
        "normalizer_state_sha256": normalizer_digest,
        "diagnostic_only": True,
        "selection_eligible": False,
        "resume_supported": False,
        "test_object_opened": False,
    }
    original = diagnostic.D090_MODEL_STATE_SHA256
    try:
        diagnostic.D090_MODEL_STATE_SHA256 = digest
        diagnostic.validate_step100_checkpoint(step100, normalizer)
        step100["selection_eligible"] = True
        with pytest.raises(ValueError, match="D090 selection"):
            diagnostic.validate_step100_checkpoint(step100, normalizer)
    finally:
        diagnostic.D090_MODEL_STATE_SHA256 = original


def test_help_is_cpu_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert torch.cuda.is_initialized() is False
    with pytest.raises(SystemExit) as exc_info:
        diagnostic.main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "matched IgnitHIT boundedness attribution" in output
    assert "--d089-output-dir" in output
    assert "--d090-output-dir" in output
    assert torch.cuda.is_initialized() is False
