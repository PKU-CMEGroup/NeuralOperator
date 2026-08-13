"""Matched teacher-forced/free-recurrence boundedness diagnostics for REALM.

The functions in this module operate on already-decoded states or on an
already-constructed deployed map. They do not load data, checkpoints, or
artifacts. Event calls are one-based and accepted prefixes end immediately
before the first failed call.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from utility.time_dependent_no.realm_benchmark import (
    MagnitudeEnvelope,
    decoded_boundedness,
)


@dataclass(frozen=True)
class NormalizedModePredictions:
    """Normalized proposals returned by the two registered input policies."""

    free_inputs: torch.Tensor
    free_recurrence: torch.Tensor
    teacher_inputs: torch.Tensor
    teacher_forced: torch.Tensor


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _float_tensor(value: torch.Tensor, name: str, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")


def evaluate_normalized_modes(
    model: nn.Module,
    validation_normalized: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    calls: int,
) -> NormalizedModePredictions:
    """Run the same deployed map under free and exact-truth input policies.

    Free recurrence stops after returning its first native-nonfinite batch so
    that such a proposal is observable but is never fed into another call.
    The registered D091 scientific execution requires all 29 calls to return.
    """

    calls = _positive_int(calls, "calls")
    _float_tensor(validation_normalized, "validation_normalized", ndim=5)
    _float_tensor(coordinates, "coordinates", ndim=4)
    if validation_normalized.shape[1] <= calls:
        raise ValueError("validation truth does not cover all requested calls")
    if validation_normalized.shape[-2:] != coordinates.shape[-2:]:
        raise ValueError("state and coordinate spatial shapes differ")
    if coordinates.shape[0] not in (1, validation_normalized.shape[0]):
        raise ValueError("coordinates must have singleton or matching case axis")
    if not bool(torch.isfinite(validation_normalized).all()):
        raise ValueError("validation_normalized must be finite truth")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must be finite")
    if (
        validation_normalized.dtype != coordinates.dtype
        or validation_normalized.device != coordinates.device
    ):
        raise ValueError("state and coordinates must share dtype and device")

    was_training = model.training
    model.eval()
    free: list[torch.Tensor] = []
    free_inputs: list[torch.Tensor] = []
    teacher: list[torch.Tensor] = []
    teacher_inputs: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            current = validation_normalized[:, 0]
            for _ in range(calls):
                free_inputs.append(current.detach().clone())
                proposal = model(current, coordinates)
                if proposal.shape != current.shape:
                    raise ValueError(
                        "free proposal shape differs from the recurrent state"
                    )
                if proposal.dtype != current.dtype or proposal.device != current.device:
                    raise ValueError(
                        "free proposal dtype/device differs from the recurrent state"
                    )
                free.append(proposal)
                if not bool(torch.isfinite(proposal).all()):
                    break
                current = proposal

            for call_index in range(calls):
                truth_input = validation_normalized[:, call_index]
                teacher_inputs.append(truth_input.detach().clone())
                proposal = model(truth_input, coordinates)
                if proposal.shape != truth_input.shape:
                    raise ValueError(
                        "teacher-forced proposal shape differs from the truth input"
                    )
                if (
                    proposal.dtype != truth_input.dtype
                    or proposal.device != truth_input.device
                ):
                    raise ValueError(
                        "teacher-forced proposal dtype/device differs from truth"
                    )
                teacher.append(proposal)
    finally:
        model.train(was_training)

    if not free or not teacher:
        raise RuntimeError("both input policies must return at least one proposal")
    return NormalizedModePredictions(
        free_inputs=torch.stack(free_inputs, dim=1),
        free_recurrence=torch.stack(free, dim=1),
        teacher_inputs=torch.stack(teacher_inputs, dim=1),
        teacher_forced=torch.stack(teacher, dim=1),
    )


def channel_envelope_records(
    decoded_states: torch.Tensor,
    envelope: MagnitudeEnvelope,
    *,
    expansion_factor: float,
    case_keys: Sequence[str],
    channel_names: Sequence[str],
    checkpoint_id: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Return strict-JSON per-case/call/channel envelope and argmax records."""

    _float_tensor(decoded_states, "decoded_states", ndim=5)
    cases = _unique_names(case_keys, "case_keys")
    channels = _unique_names(channel_names, "channel_names")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("checkpoint_id must be a nonempty string")
    if not isinstance(mode, str) or not mode:
        raise ValueError("mode must be a nonempty string")
    if len(cases) != decoded_states.shape[0]:
        raise ValueError("case_keys do not match the decoded case axis")
    if len(channels) != decoded_states.shape[2]:
        raise ValueError("channel_names do not match the decoded channel axis")
    if decoded_states.shape[1] == 0 or decoded_states.shape[-2] == 0:
        raise ValueError("decoded states require nonempty call and spatial axes")
    if decoded_states.shape[-1] == 0:
        raise ValueError("decoded states require nonempty spatial axes")
    if not math.isfinite(expansion_factor) or expansion_factor <= 0.0:
        raise ValueError("expansion_factor must be finite and positive")
    if envelope.max_abs.numel() != len(channels):
        raise ValueError("envelope does not match the registered channels")

    limits = envelope.max_abs.to(
        device=decoded_states.device, dtype=decoded_states.dtype
    ) * expansion_factor
    records: list[dict[str, Any]] = []
    for case_index, case_key in enumerate(cases):
        for call_index in range(decoded_states.shape[1]):
            state = decoded_states[case_index, call_index]
            state_finite = bool(torch.isfinite(state).all())
            reused = decoded_boundedness(
                state,
                envelope,
                expansion_factor=expansion_factor,
                channel_axis=0,
            )
            channel_bounded: list[bool] = []
            for channel_index, channel_name in enumerate(channels):
                field = state[channel_index]
                channel_finite = bool(torch.isfinite(field).all())
                limit = float(limits[channel_index].item())
                if channel_finite:
                    flattened = field.abs().reshape(-1)
                    flat_index = int(torch.argmax(flattened).item())
                    column_count = field.shape[-1]
                    row, column = divmod(flat_index, column_count)
                    signed_value = float(field.reshape(-1)[flat_index].item())
                    observed = flattened[flat_index]
                    maximum = float(observed.item())
                    if limit > 0.0:
                        ratio: float | None = float(
                            (observed / limits[channel_index]).item()
                        )
                        ratio_status = "finite"
                    elif maximum == 0.0:
                        ratio = 0.0
                        ratio_status = "zero_limit_zero_observed"
                    else:
                        ratio = None
                        ratio_status = "positive_infinite_zero_limit"
                    bounded = bool(observed <= limits[channel_index])
                    exceedance = field.abs() > limits[channel_index]
                    exceedance_count = int(exceedance.sum().item())
                    if exceedance_count:
                        support = torch.nonzero(exceedance, as_tuple=False)
                        support_row_min = int(support[:, 0].min().item())
                        support_row_max = int(support[:, 0].max().item())
                        support_column_min = int(support[:, 1].min().item())
                        support_column_max = int(support[:, 1].max().item())
                    else:
                        support_row_min = None
                        support_row_max = None
                        support_column_min = None
                        support_column_max = None
                else:
                    row = None
                    column = None
                    signed_value = None
                    maximum = None
                    ratio = None
                    ratio_status = "native_nonfinite"
                    bounded = False
                    exceedance_count = None
                    support_row_min = None
                    support_row_max = None
                    support_column_min = None
                    support_column_max = None
                channel_bounded.append(bounded)
                records.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "mode": mode,
                        "case_index": case_index,
                        "case_key": case_key,
                        "call": call_index + 1,
                        "channel_index": channel_index,
                        "channel": channel_name,
                        "channel_finite": channel_finite,
                        "bounded": bounded,
                        "envelope_limit": limit,
                        "max_abs": maximum,
                        "ratio": ratio,
                        "ratio_status": ratio_status,
                        "argmax_row": row,
                        "argmax_column": column,
                        "signed_value_at_argmax": signed_value,
                        "exceedance_count": exceedance_count,
                        "exceedance_support": {
                            "row_min": support_row_min,
                            "row_max": support_row_max,
                            "column_min": support_column_min,
                            "column_max": support_column_max,
                        },
                    }
                )
            if reused.finite != state_finite:
                raise RuntimeError("reused boundedness finiteness disagrees")
            if reused.bounded != all(channel_bounded):
                raise RuntimeError("reused boundedness decision disagrees")

    json.dumps(records, allow_nan=False)
    return records


def _event_from_calls(failed_calls: Sequence[int], *, horizon: int) -> dict[str, Any]:
    horizon = _positive_int(horizon, "horizon")
    if failed_calls:
        event_call = min(failed_calls)
        if event_call < 1 or event_call > horizon:
            raise ValueError("event call lies outside the observed horizon")
        return {
            "first_failure_call": event_call,
            "accepted_prefix": event_call - 1,
            "censored": False,
            "censor_call": None,
        }
    return {
        "first_failure_call": None,
        "accepted_prefix": horizon,
        "censored": True,
        "censor_call": horizon,
    }


def validate_recurrence_trace(
    normalized_inputs: torch.Tensor,
    normalized_proposals: torch.Tensor,
) -> None:
    """Require each observed next free input to equal the prior deployment."""

    _float_tensor(normalized_inputs, "normalized_inputs", ndim=5)
    _float_tensor(normalized_proposals, "normalized_proposals", ndim=5)
    if normalized_inputs.shape != normalized_proposals.shape:
        raise ValueError("recurrence input/proposal trace shapes differ")
    if (
        normalized_inputs.dtype != normalized_proposals.dtype
        or normalized_inputs.device != normalized_proposals.device
    ):
        raise ValueError("recurrence input/proposal traces differ in dtype/device")
    if normalized_inputs.shape[1] == 0:
        raise ValueError("recurrence trace requires at least one observed call")
    if normalized_inputs.shape[1] > 1 and not torch.equal(
        normalized_inputs[:, 1:], normalized_proposals[:, :-1]
    ):
        raise ValueError("next free input is not the exact prior deployed proposal")


def _index_records(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
    mode: str,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("envelope records must be mappings")
        if record.get("checkpoint_id") != checkpoint_id or record.get("mode") != mode:
            raise ValueError("envelope record checkpoint/mode identity differs")
        case_key = record.get("case_key")
        call = record.get("call")
        channel = record.get("channel")
        if not isinstance(case_key, str) or not case_key:
            raise ValueError("envelope record case_key must be nonempty")
        if isinstance(call, bool) or not isinstance(call, int) or call <= 0:
            raise ValueError("envelope record call must be a positive integer")
        if not isinstance(channel, str) or not channel:
            raise ValueError("envelope record channel must be nonempty")
        if not isinstance(record.get("channel_finite"), bool) or not isinstance(
            record.get("bounded"), bool
        ):
            raise TypeError("envelope record predicates must be booleans")
        ratio = record.get("ratio")
        if ratio is not None:
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError("envelope record ratio must be numeric or null")
            if not math.isfinite(float(ratio)) or float(ratio) < 0.0:
                raise ValueError("finite envelope ratios must be nonnegative")
        key = (case_key, call, channel)
        if key in result:
            raise ValueError("duplicate envelope record")
        result[key] = record
    return result


def summarize_envelope_events(
    records: Sequence[Mapping[str, Any]],
    *,
    case_keys: Sequence[str],
    channel_names: Sequence[str],
    checkpoint_id: str,
    mode: str,
    horizon: int,
) -> dict[str, Any]:
    """Summarize first finite/bounded failures without mixed-prefix averaging."""

    cases = _unique_names(case_keys, "case_keys")
    channels = _unique_names(channel_names, "channel_names")
    horizon = _positive_int(horizon, "horizon")
    by_key = _index_records(
        records,
        checkpoint_id=checkpoint_id,
        mode=mode,
    )
    expected = {
        (case_key, call, channel)
        for case_key in cases
        for call in range(1, horizon + 1)
        for channel in channels
    }
    if set(by_key) != expected:
        raise ValueError("records do not form the exact case/call/channel grid")

    per_case: list[dict[str, Any]] = []
    all_ratios: list[float] = []
    has_positive_infinite_ratio = False
    has_native_nonfinite = False
    for case_key in cases:
        finite_failures: list[int] = []
        bounded_failures: list[int] = []
        bounded_calls = 0
        reentry_calls: list[int] = []
        any_bounded_by_call: dict[int, bool] = {}
        channel_events: list[dict[str, Any]] = []
        for call in range(1, horizon + 1):
            call_records = [by_key[(case_key, call, channel)] for channel in channels]
            call_finite = all(record["channel_finite"] is True for record in call_records)
            call_bounded = all(record["bounded"] is True for record in call_records)
            any_bounded_by_call[call] = call_bounded
            if not call_finite:
                finite_failures.append(call)
            if not call_bounded:
                bounded_failures.append(call)
            bounded_calls += int(call_bounded)
            for record in call_records:
                ratio = record["ratio"]
                if ratio is not None:
                    ratio_value = float(ratio)
                    if not math.isfinite(ratio_value) or ratio_value < 0.0:
                        raise ValueError("finite ratio records must be nonnegative")
                    all_ratios.append(ratio_value)
                elif record["ratio_status"] == "positive_infinite_zero_limit":
                    has_positive_infinite_ratio = True
                elif record["ratio_status"] == "native_nonfinite":
                    has_native_nonfinite = True

        bounded_event = _event_from_calls(bounded_failures, horizon=horizon)
        if bounded_event["first_failure_call"] is not None:
            reentry_calls = [
                call
                for call in range(bounded_event["first_failure_call"] + 1, horizon + 1)
                if any_bounded_by_call[call]
            ]
        for channel in channels:
            failed_calls = [
                call
                for call in range(1, horizon + 1)
                if by_key[(case_key, call, channel)]["bounded"] is False
            ]
            channel_events.append(
                {
                    "channel": channel,
                    "boundedness": _event_from_calls(failed_calls, horizon=horizon),
                }
            )
        per_case.append(
            {
                "case_key": case_key,
                "decoded_finiteness": _event_from_calls(
                    finite_failures, horizon=horizon
                ),
                "boundedness": bounded_event,
                "bounded_calls": bounded_calls,
                "raw_reentry_calls": reentry_calls,
                "channel_events": channel_events,
            }
        )

    if has_native_nonfinite:
        maximum_ratio = None
        maximum_ratio_status = "unavailable_native_nonfinite"
    elif has_positive_infinite_ratio:
        maximum_ratio = None
        maximum_ratio_status = "positive_infinite_zero_limit"
    elif all_ratios:
        maximum_ratio = max(all_ratios)
        maximum_ratio_status = "finite"
    else:
        maximum_ratio = None
        maximum_ratio_status = "unavailable_native_nonfinite"
    payload: dict[str, Any] = {
        "schema": "realm_boundedness_event_summary_v1",
        "checkpoint_id": checkpoint_id,
        "mode": mode,
        "horizon": horizon,
        "case_count": len(cases),
        "channel_count": len(channels),
        "bounded_case_calls": sum(row["bounded_calls"] for row in per_case),
        "total_case_calls": len(cases) * horizon,
        "maximum_ratio": maximum_ratio,
        "maximum_ratio_status": maximum_ratio_status,
        "per_case": per_case,
    }
    json.dumps(payload, allow_nan=False)
    return payload


def summarize_attribution_verdict(
    checkpoint_attributions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Count frozen first-event labels without collapsing mixed outcomes."""

    if not checkpoint_attributions:
        raise ValueError("attribution verdict requires at least one checkpoint")
    allowed = {
        "fresh_exact_input_sufficient_for_threshold_at_free_event",
        "propagated_input_required_for_threshold_at_free_event",
    }
    checkpoints: dict[str, Any] = {}
    total_counts = {label: 0 for label in sorted(allowed)}
    total_censored_cases = 0
    for checkpoint_id, attribution in checkpoint_attributions.items():
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint attribution keys must be nonempty strings")
        cases = attribution.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("checkpoint attribution requires nonempty cases")
        counts = {label: 0 for label in sorted(allowed)}
        censored_cases = 0
        observed_cases = 0
        for case in cases:
            if not isinstance(case, Mapping):
                raise TypeError("attribution cases must be mappings")
            status = case.get("status")
            if status == "free_event_censored":
                censored_cases += 1
                continue
            if status != "free_event_observed":
                raise ValueError("unknown attribution case status")
            observed_cases += 1
            event_channels = case.get("event_channels")
            if not isinstance(event_channels, list) or not event_channels:
                raise ValueError("observed free events require event channels")
            for event in event_channels:
                label = event.get("label") if isinstance(event, Mapping) else None
                if label not in allowed:
                    raise ValueError("unknown first-event attribution label")
                counts[label] += 1
                total_counts[label] += 1
        total_censored_cases += censored_cases
        checkpoints[checkpoint_id] = {
            "event_channel_counts": counts,
            "observed_free_event_cases": observed_cases,
            "censored_free_event_cases": censored_cases,
        }
    if all(count == 0 for count in total_counts.values()):
        verdict = "all_free_events_censored"
    elif all(count > 0 for count in total_counts.values()):
        verdict = "mixed_first_event_attribution"
    elif total_counts[
        "fresh_exact_input_sufficient_for_threshold_at_free_event"
    ]:
        verdict = "fresh_exact_input_sufficient_only_among_observed_events"
    else:
        verdict = "propagated_input_required_only_among_observed_events"
    payload = {
        "schema": "realm_boundedness_attribution_verdict_v1",
        "verdict": verdict,
        "event_channel_counts": total_counts,
        "censored_free_event_cases": total_censored_cases,
        "checkpoints": checkpoints,
        "claim_boundary": (
            "threshold attribution under exact truth-input reset; no training-factor "
            "or universal mechanism claim"
        ),
    }
    json.dumps(payload, allow_nan=False)
    return payload


def attribute_free_first_events(
    free_records: Sequence[Mapping[str, Any]],
    teacher_records: Sequence[Mapping[str, Any]],
    *,
    case_keys: Sequence[str],
    channel_names: Sequence[str],
    checkpoint_id: str,
    horizon: int,
) -> dict[str, Any]:
    """Classify exact free first events under the truth-input reset intervention."""

    cases = _unique_names(case_keys, "case_keys")
    channels = _unique_names(channel_names, "channel_names")
    horizon = _positive_int(horizon, "horizon")

    free = _index_records(
        free_records,
        checkpoint_id=checkpoint_id,
        mode="free_recurrence",
    )
    teacher = _index_records(
        teacher_records,
        checkpoint_id=checkpoint_id,
        mode="teacher_forced",
    )
    expected = {
        (case_key, call, channel)
        for case_key in cases
        for call in range(1, horizon + 1)
        for channel in channels
    }
    if set(free) != expected or set(teacher) != expected:
        raise ValueError("attribution records do not form matching exact grids")

    cases_payload: list[dict[str, Any]] = []
    for case_key in cases:
        free_event_call = next(
            (
                call
                for call in range(1, horizon + 1)
                if any(
                    free[(case_key, call, channel)]["bounded"] is False
                    for channel in channels
                )
            ),
            None,
        )
        if free_event_call is None:
            cases_payload.append(
                {
                    "case_key": case_key,
                    "free_first_failure_call": None,
                    "status": "free_event_censored",
                    "event_channels": [],
                }
            )
            continue
        event_channels: list[dict[str, Any]] = []
        for channel in channels:
            free_record = free[(case_key, free_event_call, channel)]
            if free_record["bounded"] is not False:
                continue
            teacher_record = teacher[(case_key, free_event_call, channel)]
            label = (
                "fresh_exact_input_sufficient_for_threshold_at_free_event"
                if teacher_record["bounded"] is False
                else "propagated_input_required_for_threshold_at_free_event"
            )
            event_channels.append(
                {
                    "channel": channel,
                    "label": label,
                    "free_ratio": free_record["ratio"],
                    "free_ratio_status": free_record["ratio_status"],
                    "teacher_forced_ratio": teacher_record["ratio"],
                    "teacher_forced_ratio_status": teacher_record["ratio_status"],
                }
            )
        cases_payload.append(
            {
                "case_key": case_key,
                "free_first_failure_call": free_event_call,
                "status": "free_event_observed",
                "event_channels": event_channels,
            }
        )

    payload = {
        "schema": "realm_boundedness_attribution_v1",
        "checkpoint_id": checkpoint_id,
        "horizon": horizon,
        "cases": cases_payload,
    }
    json.dumps(payload, allow_nan=False)
    return payload


__all__ = [
    "NormalizedModePredictions",
    "attribute_free_first_events",
    "channel_envelope_records",
    "evaluate_normalized_modes",
    "summarize_attribution_verdict",
    "summarize_envelope_events",
    "validate_recurrence_trace",
]
