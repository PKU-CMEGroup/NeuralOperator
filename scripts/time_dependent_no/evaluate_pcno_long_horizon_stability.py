"""Synthetic-contract kernel for W26-L1 long-horizon PCNO evaluation.

This A1 surface deliberately cannot load a checkpoint or dataset.  It freezes
representation-aware Euler diagnostics, event-prefix semantics, recurrence and
recovery attribution, and case-first survival aggregation before any A2 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import torch

from utility.time_dependent_no.pcno_euler2d import (
    conservative_to_primitive_torch,
    primitive_to_conservative_torch,
)
from utility.time_dependent_no.pcno_inadmissibility import stage_transition_label

SCHEMA = "pcno_long_horizon_stability_v1"
EVENT_NAMES = ("accurate", "admissible", "bounded", "finite")
STAGE_NAMES = ("current", "model_input", "model_proposal", "deployed")
FINAL_HASH_MANIFEST_SCHEMA = "pcno_long_horizon_final_hash_manifest_v1"
FINAL_HASH_MANIFEST_NAME = "final_hash_manifest.json"

StateRepresentation = Literal[
    "primitive_rho_v1_v2_p",
    "conservative_rho_m1_m2_E",
]
RecurrenceSource = Literal["model_proposal", "deployed"]
ComparisonKind = Literal[
    "descriptive_survival",
    "inference_map_diagnostic",
    "training_factor_causal_attribution",
]

COMPARISON_KINDS = (
    "descriptive_survival",
    "inference_map_diagnostic",
    "training_factor_causal_attribution",
)

PROVENANCE_FIELDS = (
    "system_id",
    "checkpoint_path_contract",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "training_source_digest",
    "executed_inference_source_digest",
    "native_map_equivalence_digest",
    "training_data_digest",
    "training_split_digest",
    "training_normalizer_digest",
    "evaluation_data_digest",
    "evaluation_population_digest",
    "inference_normalizer_digest",
    "runtime_manifest_digest",
    "returned_stage_manifest_digest",
    "training_precision",
    "inference_precision",
    "training_seed",
    "training_presentations",
    "optimizer_steps",
    "optimizer_history",
    "training_history",
    "recurrence",
    "boundary_policy",
    "evaluator_digest",
    "selection_rule",
    "state_representation",
)

DIGEST_FIELDS = (
    "checkpoint_sha256",
    "training_source_digest",
    "executed_inference_source_digest",
    "native_map_equivalence_digest",
    "training_data_digest",
    "training_split_digest",
    "training_normalizer_digest",
    "evaluation_data_digest",
    "evaluation_population_digest",
    "inference_normalizer_digest",
    "runtime_manifest_digest",
    "returned_stage_manifest_digest",
    "evaluator_digest",
)

TRAINING_SIDE_PROVENANCE_FIELDS = frozenset(
    {
        "training_source_digest",
        "training_data_digest",
        "training_split_digest",
        "training_normalizer_digest",
        "training_precision",
        "training_seed",
        "training_presentations",
        "optimizer_steps",
        "optimizer_history",
        "training_history",
        "selection_rule",
    }
)

DESCRIPTIVE_SURVIVAL_REQUIRED_FIELDS = frozenset(
    {
        "system_id",
        "checkpoint_path_contract",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "executed_inference_source_digest",
        "evaluation_data_digest",
        "evaluation_population_digest",
        "inference_normalizer_digest",
        "state_representation",
        "inference_precision",
        "recurrence",
        "boundary_policy",
        "evaluator_digest",
        "runtime_manifest_digest",
        "returned_stage_manifest_digest",
    }
)


@dataclass(frozen=True)
class StabilityThresholds:
    """Frozen pass thresholds in a common primitive analysis view."""

    accurate_relative_l2_max: float = 0.05
    amplitude_ratio_max: float = 100.0
    scaled_rms_ratio_max: float = 10.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "accurate_relative_l2_max": self.accurate_relative_l2_max,
            "amplitude_ratio_max": self.amplitude_ratio_max,
            "scaled_rms_ratio_max": self.scaled_rms_ratio_max,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_top_level_artifact_names(artifact_names: Sequence[str]) -> list[str]:
    names = list(artifact_names)
    if not names:
        raise ValueError("at least one closed artifact file is required")
    for name in names:
        if not isinstance(name, str):
            raise TypeError("artifact names must be strings")
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or len(path.parts) != 1
            or path.name != name
            or name in {".", ".."}
        ):
            raise ValueError("each artifact must use one top-level POSIX file name")
        if name == FINAL_HASH_MANIFEST_NAME:
            raise ValueError("the final hash manifest must not include itself")
    if len(set(names)) != len(names):
        raise ValueError("artifact names must be unique")
    return sorted(names)


def _require_exact_top_level_inventory(root: Path, names: Sequence[str]) -> None:
    expected = set(names)
    observed = {
        entry.name for entry in root.iterdir() if entry.is_file() or entry.is_symlink()
    }
    observed.discard(FINAL_HASH_MANIFEST_NAME)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "artifact root differs from the declared closed inventory; "
            f"missing files={missing}; unregistered top-level files={unexpected}"
        )


def build_final_hash_manifest(
    artifact_root: str | Path,
    artifact_names: Sequence[str],
) -> dict[str, Any]:
    """Hash an explicit closed flat artifact set without writing a manifest.

    Live transport logs and wrapper exit receipts must remain outside
    ``artifact_root``. Subdirectories such as immutable source staging are bound
    by their own manifests and are not silently added to this top-level result
    inventory.
    """

    root = Path(artifact_root)
    if not root.is_dir():
        raise ValueError("artifact_root must be an existing directory")
    root = root.resolve()
    names = _closed_top_level_artifact_names(artifact_names)
    _require_exact_top_level_inventory(root, names)

    files: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact must be a regular non-symlink file: {name}")
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        before_identity = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise ValueError(
                f"artifact changed while hashing; close its writer: {name}"
            )
        files.append(
            {
                "path": name,
                "size": after.st_size,
                "sha256": digest,
            }
        )

    _require_exact_top_level_inventory(root, names)
    return {
        "schema": FINAL_HASH_MANIFEST_SCHEMA,
        "files": files,
    }


def verify_final_hash_manifest(
    artifact_root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare a closed-file manifest after finalization."""

    if manifest.get("schema") != FINAL_HASH_MANIFEST_SCHEMA:
        raise ValueError("unsupported final hash manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(row, Mapping) for row in files
    ):
        raise TypeError("final hash manifest files must be a list of mappings")
    try:
        names = [str(row["path"]) for row in files]
    except KeyError as exc:
        raise ValueError("every final hash manifest row requires path") from exc
    rebuilt = build_final_hash_manifest(artifact_root, names)
    if rebuilt != dict(manifest):
        raise ValueError("artifact root does not match the closed-file manifest")
    return {
        "schema": FINAL_HASH_MANIFEST_SCHEMA,
        "verified_file_count": len(files),
    }


@dataclass(frozen=True)
class StepTrace:
    """The four physical stages of one autonomous model call.

    ``model_proposal_native`` is the model-returned, post-head physical state.
    For D019 this is the post-exponential primitive proposal, not an unavailable
    pre-exponential logit tensor.
    """

    call: int
    current_native: torch.Tensor
    model_input_native: torch.Tensor
    model_proposal_native: torch.Tensor
    deployed_native: torch.Tensor
    representation: StateRepresentation
    proposal_semantics: Literal["physical_post_head"] = "physical_post_head"
    recurrence_source: RecurrenceSource = "deployed"


def _check_representation(representation: str) -> None:
    if representation not in {
        "primitive_rho_v1_v2_p",
        "conservative_rho_m1_m2_E",
    }:
        raise ValueError(f"unsupported state representation: {representation}")


def _check_state(state: torch.Tensor) -> None:
    if state.ndim < 2 or state.shape[-1] != 4:
        raise ValueError("Euler state must have shape [..., N, 4]")


def _active_mask(state: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
    if node_mask is None:
        return torch.ones(state.shape[:-1], dtype=torch.bool, device=state.device)
    mask = node_mask.to(device=state.device)
    if mask.shape == (*state.shape[:-1], 1):
        mask = mask[..., 0]
    if mask.shape != state.shape[:-1]:
        raise ValueError("node_mask must match state shape without components")
    active = mask > 0
    if not bool(active.any()):
        raise ValueError("node_mask must select at least one active node")
    return active


def native_to_common_primitive(
    state: torch.Tensor,
    *,
    representation: StateRepresentation,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Map a native state to float64 ``[rho,v1,v2,p]`` for analysis only."""

    _check_state(state)
    _check_representation(representation)
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    state64 = state.to(dtype=torch.float64)
    if representation == "primitive_rho_v1_v2_p":
        return state64
    return conservative_to_primitive_torch(state64, gamma=gamma)


def common_primitive_to_native(
    primitive: torch.Tensor,
    *,
    representation: StateRepresentation,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Map common primitive coordinates to a native float64 state."""

    _check_state(primitive)
    _check_representation(representation)
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    primitive64 = primitive.to(dtype=torch.float64)
    if representation == "primitive_rho_v1_v2_p":
        return primitive64
    return primitive_to_conservative_torch(primitive64, gamma=gamma)


def _finite_minimum(value: torch.Tensor, active: torch.Tensor) -> float | None:
    selected = value[active]
    finite = selected[torch.isfinite(selected)]
    return None if finite.numel() == 0 else float(finite.min().cpu())


def diagnose_euler_state(
    state: torch.Tensor,
    *,
    representation: StateRepresentation,
    gamma: float = 1.4,
    node_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Diagnose native finiteness and common Euler admissibility without repair."""

    _check_state(state)
    _check_representation(representation)
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    native = state.detach().to(dtype=torch.float64)
    active = _active_mask(native, node_mask)
    rho = native[..., 0]

    if representation == "primitive_rho_v1_v2_p":
        v1, v2, pressure = native[..., 1], native[..., 2], native[..., 3]
        internal_energy = pressure / (gamma - 1.0)
        derived = torch.stack(
            (
                rho * v1,
                rho * v2,
                internal_energy + 0.5 * rho * (v1.square() + v2.square()),
                internal_energy,
            ),
            dim=-1,
        )
    else:
        momentum_x, momentum_y, energy = (
            native[..., 1],
            native[..., 2],
            native[..., 3],
        )
        v1 = momentum_x / rho
        v2 = momentum_y / rho
        internal_energy = (
            energy - 0.5 * (momentum_x.square() + momentum_y.square()) / rho
        )
        pressure = (gamma - 1.0) * internal_energy
        derived = torch.stack((v1, v2, internal_energy, pressure), dim=-1)

    active_components = active.unsqueeze(-1).expand_as(native)
    native_components_finite = bool(
        torch.isfinite(native)[active_components].all().item()
    )
    derived_fields_finite = bool(
        torch.isfinite(derived)[active_components].all().item()
    )
    rho_bad = active & torch.isfinite(rho) & (rho <= 0.0)
    internal_bad = active & torch.isfinite(internal_energy) & (internal_energy <= 0.0)
    pressure_bad = active & torch.isfinite(pressure) & (pressure <= 0.0)

    violation_fields: list[str] = []
    if not native_components_finite:
        violation_fields.append("native_components_nonfinite")
    if not derived_fields_finite:
        violation_fields.append("derived_fields_nonfinite")
    if bool(rho_bad.any()):
        violation_fields.append("nonpositive_density")
    if bool(internal_bad.any()):
        violation_fields.append("nonpositive_internal_energy")
    if bool(pressure_bad.any()):
        violation_fields.append("nonpositive_pressure")

    admissible = (
        native_components_finite
        and derived_fields_finite
        and not bool(rho_bad.any())
        and not bool(internal_bad.any())
        and not bool(pressure_bad.any())
    )
    if admissible:
        primary_failure = "admissible"
    elif not native_components_finite:
        primary_failure = "nonfinite_native_components"
    elif bool(rho_bad.any()):
        primary_failure = "nonpositive_density"
    elif not derived_fields_finite:
        primary_failure = "nonfinite_derived_fields"
    elif representation == "primitive_rho_v1_v2_p" and bool(pressure_bad.any()):
        primary_failure = "nonpositive_pressure"
    elif bool(internal_bad.any()):
        primary_failure = "nonpositive_internal_energy"
    else:
        primary_failure = "nonpositive_pressure"

    return {
        "state_representation": representation,
        "active_node_count": int(active.sum().item()),
        "native_components_finite": native_components_finite,
        "derived_fields_finite": derived_fields_finite,
        # T_finite concerns the stored/deployed native state, not derived divisions.
        "finite": native_components_finite,
        "thermodynamic_admissible": admissible,
        "admissible": admissible,
        "violation_fields": violation_fields,
        "primary_failure": primary_failure,
        "nonpositive_density_node_count": int(rho_bad.sum().item()),
        "nonpositive_internal_energy_node_count": int(internal_bad.sum().item()),
        "nonpositive_pressure_node_count": int(pressure_bad.sum().item()),
        "min_density": _finite_minimum(rho, active),
        "min_internal_energy": _finite_minimum(internal_energy, active),
        "min_pressure": _finite_minimum(pressure, active),
    }


def classify_call_events(
    *,
    state_diagnostics: Mapping[str, Any],
    truth_available: bool,
    common_relative_l2: float | None,
    common_amplitude_ratio: float | None,
    common_scaled_rms_ratio: float | None,
    thresholds: StabilityThresholds,
) -> dict[str, bool | None]:
    """Classify four independent deployed-state events for one model call."""

    finite = bool(state_diagnostics["finite"])
    admissible = bool(state_diagnostics["thermodynamic_admissible"])

    def checked_metric(
        name: str, value: float | None, *, required: bool
    ) -> float | None:
        if value is None:
            if required:
                raise ValueError(f"finite-state call lacks required {name}")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar or None")
        result = float(value)
        if math.isnan(result) or result < 0.0:
            raise ValueError(f"{name} must be nonnegative and not NaN")
        return result

    if not truth_available and common_relative_l2 is not None:
        raise ValueError("truth-free calls must not carry a relative-L2 accuracy value")

    if truth_available:
        relative_l2 = checked_metric(
            "common_relative_l2",
            common_relative_l2,
            required=finite,
        )
        accurate = bool(
            finite
            and relative_l2 is not None
            and relative_l2 <= thresholds.accurate_relative_l2_max
        )
    else:
        accurate = None

    amplitude_ratio = checked_metric(
        "common_amplitude_ratio",
        common_amplitude_ratio,
        required=finite,
    )
    scaled_rms_ratio = checked_metric(
        "common_scaled_rms_ratio",
        common_scaled_rms_ratio,
        required=finite,
    )
    bounded = bool(
        finite
        and amplitude_ratio is not None
        and scaled_rms_ratio is not None
        and amplitude_ratio <= thresholds.amplitude_ratio_max
        and scaled_rms_ratio <= thresholds.scaled_rms_ratio_max
    )
    return {
        "accurate": accurate,
        "admissible": admissible,
        "bounded": bounded,
        "finite": finite,
    }


def accepted_prefix_event(
    pass_values: Sequence[bool | None],
    *,
    requested_horizon: int,
    event_name: str,
) -> dict[str, Any]:
    """Return first-event and accepted-prefix semantics for one trajectory."""

    if event_name not in EVENT_NAMES:
        raise ValueError(f"unknown event: {event_name}")
    if requested_horizon < 1:
        raise ValueError("requested_horizon must be positive")
    values = list(pass_values)
    if len(values) > requested_horizon:
        raise ValueError("event sequence exceeds requested_horizon")
    if any(value is not None and not isinstance(value, bool) for value in values):
        raise TypeError("event values must be bool or None")
    unavailable = next((i for i, value in enumerate(values) if value is None), None)
    if unavailable is not None and any(
        value is not None for value in values[unavailable:]
    ):
        raise ValueError("unavailable event values must form a terminal suffix")
    observed = len(values) if unavailable is None else unavailable
    first_failure_offset = next(
        (i for i, value in enumerate(values[:observed]) if value is False), None
    )

    if first_failure_offset is not None:
        first_failure_call = first_failure_offset + 1
        return {
            "first_failure_call": first_failure_call,
            "accepted_prefix_calls": first_failure_call - 1,
            "right_censored": False,
            "censor_call": None,
            "censor_reason": None,
            "observed_call_count": observed,
            "recovered_after_failure": any(
                value is True for value in values[first_failure_offset + 1 : observed]
            ),
        }

    if unavailable is not None:
        censor_reason = (
            "truth_horizon" if event_name == "accurate" else "measurement_unavailable"
        )
    elif observed == requested_horizon:
        censor_reason = "requested_horizon"
    else:
        censor_reason = "execution_horizon"
    return {
        "first_failure_call": None,
        "accepted_prefix_calls": observed,
        "right_censored": True,
        "censor_call": observed,
        "censor_reason": censor_reason,
        "observed_call_count": observed,
        "recovered_after_failure": False,
    }


def summarize_stability_events(
    call_rows: Sequence[Mapping[str, Any]],
    *,
    requested_horizon: int,
    case_id: str | int,
) -> dict[str, Any]:
    """Summarize a one-indexed call trace without mixed-prefix averaging."""

    rows = list(call_rows)
    if not rows:
        raise ValueError("at least one call row is required")
    calls = [int(row["call"]) for row in rows]
    if calls != list(range(1, len(rows) + 1)):
        raise ValueError("call rows must be contiguous and one-indexed")
    if len(rows) > requested_horizon:
        raise ValueError("call rows exceed requested_horizon")
    events: dict[str, Any] = {}
    for event_name in EVENT_NAMES:
        values: list[bool | None] = []
        for row in rows:
            row_events = row.get("events")
            if not isinstance(row_events, Mapping) or event_name not in row_events:
                raise ValueError(f"call {row['call']} lacks event {event_name}")
            value = row_events[event_name]
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"event {event_name} must be bool or None")
            values.append(value)
        events[event_name] = accepted_prefix_event(
            values,
            requested_horizon=requested_horizon,
            event_name=event_name,
        )
    return {
        "schema": SCHEMA,
        "case_id": case_id,
        "requested_horizon": requested_horizon,
        "recorded_call_count": len(rows),
        "events": events,
    }


def event_survival_table(
    trajectory_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build case-first Kaplan-Meier rows for each registered event."""

    summaries = list(trajectory_summaries)
    if not summaries:
        raise ValueError("at least one trajectory summary is required")
    case_ids = [summary["case_id"] for summary in summaries]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("trajectory summaries must have unique case_id values")
    horizons = {int(summary["requested_horizon"]) for summary in summaries}
    if len(horizons) != 1:
        raise ValueError("all trajectories must share requested_horizon")
    horizons.pop()
    rows: list[dict[str, Any]] = []
    for event_name in EVENT_NAMES:
        survival = 1.0
        support_end = max(
            (
                int(event["first_failure_call"])
                if event["first_failure_call"] is not None
                else int(event["censor_call"])
            )
            for event in (summary["events"][event_name] for summary in summaries)
        )
        for call in range(1, support_end + 1):
            at_risk = 0
            failed = 0
            censored = 0
            for summary in summaries:
                event = summary["events"][event_name]
                first_failure = event["first_failure_call"]
                observed = int(event["observed_call_count"])
                if call <= observed and (
                    first_failure is None or call <= int(first_failure)
                ):
                    at_risk += 1
                    if first_failure == call:
                        failed += 1
                if bool(event["right_censored"]) and int(event["censor_call"]) == call:
                    censored += 1
            conditional = None if at_risk == 0 else (at_risk - failed) / at_risk
            if conditional is not None:
                survival *= conditional
            rows.append(
                {
                    "event": event_name,
                    "call": call,
                    "cohort_size": len(summaries),
                    "at_risk_count": at_risk,
                    "failure_count": failed,
                    "censored_after_call_count": censored,
                    "conditional_survival": conditional,
                    "kaplan_meier_survival": survival,
                }
            )
    return rows


def validate_step_trace(trace: StepTrace) -> dict[str, Any]:
    """Validate one stage trace without changing any tensor."""

    if trace.call < 1:
        raise ValueError("trace.call must be positive")
    _check_representation(trace.representation)
    if trace.proposal_semantics != "physical_post_head":
        raise ValueError("only physical post-head proposals are admissible")
    if trace.recurrence_source not in {"model_proposal", "deployed"}:
        raise ValueError("unsupported recurrence_source")
    shapes = {tuple(getattr(trace, f"{stage}_native").shape) for stage in STAGE_NAMES}
    if len(shapes) != 1:
        raise ValueError("all trace stages must have the same shape")
    dtypes = {getattr(trace, f"{stage}_native").dtype for stage in STAGE_NAMES}
    devices = {getattr(trace, f"{stage}_native").device for stage in STAGE_NAMES}
    if len(dtypes) != 1:
        raise ValueError("all trace stages must have the same dtype")
    if len(devices) != 1:
        raise ValueError("all trace stages must be on the same device")
    _check_state(trace.current_native)
    return {
        "call": trace.call,
        "shape": list(trace.current_native.shape),
        "representation": trace.representation,
        "proposal_semantics": trace.proposal_semantics,
        "recurrence_source": trace.recurrence_source,
    }


def validate_trace_sequence(
    traces: Sequence[StepTrace],
    *,
    node_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Require exact declared recurrence and stop after native nonfiniteness."""

    rows = list(traces)
    if not rows:
        raise ValueError("at least one trace is required")
    for offset, trace in enumerate(rows, start=1):
        validate_step_trace(trace)
        if trace.call != offset:
            raise ValueError("traces must be contiguous and one-indexed")
    representations = {trace.representation for trace in rows}
    proposal_semantics = {trace.proposal_semantics for trace in rows}
    recurrence_sources = {trace.recurrence_source for trace in rows}
    if len(representations) != 1:
        raise ValueError("state representation must remain fixed within a trajectory")
    if len(proposal_semantics) != 1:
        raise ValueError("proposal semantics must remain fixed within a trajectory")
    if len(recurrence_sources) != 1:
        raise ValueError("recurrence source must remain fixed within a trajectory")
    for previous, current in pairwise(rows):
        source = (
            previous.model_proposal_native
            if previous.recurrence_source == "model_proposal"
            else previous.deployed_native
        )
        if source.dtype != current.current_native.dtype:
            raise ValueError(
                f"recurrence dtype mismatch between calls {previous.call} and "
                f"{current.call}"
            )
        if source.device != current.current_native.device:
            raise ValueError(
                f"recurrence device mismatch between calls {previous.call} and "
                f"{current.call}"
            )
        if not torch.equal(source, current.current_native):
            raise ValueError(
                f"recurrence mismatch between calls {previous.call} and {current.call}"
            )
        active = _active_mask(source, node_mask)
        active_components = active.unsqueeze(-1).expand_as(source)
        if not bool(torch.isfinite(source)[active_components].all()):
            raise ValueError(
                "a native-nonfinite recurrence state must terminate the trace"
            )
    return {
        "call_count": len(rows),
        "recurrence_links_checked": max(0, len(rows) - 1),
        "exact_recurrence_closure": True,
    }


def recovery_policy_attribution(
    traces: Sequence[StepTrace],
    *,
    gamma: float = 1.4,
    node_mask: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Attribute invalidity transitions and record what state was fed back."""

    rows = list(traces)
    validate_trace_sequence(rows, node_mask=node_mask)
    output: list[dict[str, Any]] = []
    for index, trace in enumerate(rows):
        diagnostics = {
            stage: diagnose_euler_state(
                getattr(trace, f"{stage}_native"),
                representation=trace.representation,
                gamma=gamma,
                node_mask=node_mask,
            )
            for stage in STAGE_NAMES
        }
        next_current = (
            None if index + 1 == len(rows) else rows[index + 1].current_native
        )
        proposal_invalid = not diagnostics["model_proposal"]["admissible"]
        deployed_invalid = not diagnostics["deployed"]["admissible"]
        active = _active_mask(trace.model_proposal_native, node_mask)
        active_components = active.unsqueeze(-1).expand_as(trace.model_proposal_native)
        proposal_deployed_full_equal = torch.equal(
            trace.model_proposal_native,
            trace.deployed_native,
        )
        proposal_deployed_active_equal = torch.equal(
            trace.model_proposal_native[active_components],
            trace.deployed_native[active_components],
        )
        feedback_observed = next_current is not None
        output.append(
            {
                "call": trace.call,
                "input_transition": stage_transition_label(
                    diagnostics["current"]["primary_failure"],
                    diagnostics["model_input"]["primary_failure"],
                    stage="input_projection",
                ),
                "model_transition": stage_transition_label(
                    diagnostics["model_input"]["primary_failure"],
                    diagnostics["model_proposal"]["primary_failure"],
                    stage="model",
                ),
                "output_transition": stage_transition_label(
                    diagnostics["model_proposal"]["primary_failure"],
                    diagnostics["deployed"]["primary_failure"],
                    stage="output_projection",
                ),
                "recurrence_source": trace.recurrence_source,
                "feedback_observed": feedback_observed,
                "proposal_full_state_fed_back": (
                    None
                    if not feedback_observed
                    else torch.equal(trace.model_proposal_native, next_current)
                ),
                "proposal_invalid_active_nodes_fed_back": (
                    None
                    if not feedback_observed
                    else bool(
                        proposal_invalid
                        and torch.equal(
                            trace.model_proposal_native[active_components],
                            next_current[active_components],
                        )
                    )
                ),
                "deployed_invalid_fed_back": (
                    None
                    if not feedback_observed
                    else bool(
                        deployed_invalid
                        and torch.equal(trace.deployed_native, next_current)
                    )
                ),
                "proposal_deployed_full_equal": proposal_deployed_full_equal,
                "proposal_deployed_active_equal": proposal_deployed_active_equal,
                "proposal_deployed_padding_only_difference": bool(
                    not proposal_deployed_full_equal and proposal_deployed_active_equal
                ),
                "stage_failures": {
                    stage: diagnostics[stage]["primary_failure"]
                    for stage in STAGE_NAMES
                },
            }
        )
    return output


def decompose_common_error(
    *,
    prediction_from_rollout_input: torch.Tensor,
    prediction_from_truth_input: torch.Tensor,
    truth_next: torch.Tensor,
) -> dict[str, Any]:
    """Apply the exact fresh-defect/propagated-input identity in one view."""

    shapes = {
        tuple(prediction_from_rollout_input.shape),
        tuple(prediction_from_truth_input.shape),
        tuple(truth_next.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all common-coordinate tensors must have the same shape")
    if prediction_from_rollout_input.numel() == 0:
        raise ValueError("common-coordinate tensors must be nonempty")
    rollout = prediction_from_rollout_input.to(dtype=torch.float64)
    fresh_prediction = prediction_from_truth_input.to(dtype=torch.float64)
    target = truth_next.to(dtype=torch.float64)
    total_error = rollout - target
    propagated_input_error = rollout - fresh_prediction
    fresh_one_step_defect = fresh_prediction - target
    closure = total_error - propagated_input_error - fresh_one_step_defect
    denominator = max(
        float(torch.linalg.vector_norm(total_error).detach().cpu()),
        float(
            (
                torch.linalg.vector_norm(propagated_input_error)
                + torch.linalg.vector_norm(fresh_one_step_defect)
            )
            .detach()
            .cpu()
        ),
        torch.finfo(torch.float64).tiny,
    )
    return {
        "total_error": total_error,
        "propagated_input_error": propagated_input_error,
        "fresh_one_step_defect": fresh_one_step_defect,
        "closure_max_abs": float(closure.abs().max().detach().cpu()),
        "closure_relative_l2": float(torch.linalg.vector_norm(closure).detach().cpu())
        / denominator,
    }


def weighted_error_energy_closure(
    *,
    total_error: torch.Tensor,
    propagated_input_error: torch.Tensor,
    fresh_one_step_defect: torch.Tensor,
    proxy_weights: torch.Tensor,
    component_scales: torch.Tensor,
) -> dict[str, float]:
    """Audit ``total = propagated + fresh`` in one scaled weighted geometry."""

    errors = {
        "total_error": total_error,
        "propagated_input_error": propagated_input_error,
        "fresh_one_step_defect": fresh_one_step_defect,
    }
    if any(not isinstance(value, torch.Tensor) for value in errors.values()):
        raise TypeError("all error fields must be tensors")
    shapes = {tuple(value.shape) for value in errors.values()}
    if len(shapes) != 1:
        raise ValueError("all error tensors must have the same shape")
    if total_error.ndim < 2 or total_error.numel() == 0:
        raise ValueError("error tensors must be nonempty with shape [..., N, C]")

    expected_weight_shape = total_error.shape[:-1]
    if proxy_weights.shape == (*expected_weight_shape, 1):
        proxy_weights = proxy_weights[..., 0]
    if proxy_weights.shape != expected_weight_shape:
        raise ValueError("proxy_weights must match error shape without components")
    if component_scales.ndim != 1 or component_scales.shape[0] != total_error.shape[-1]:
        raise ValueError("component_scales must have shape [C]")

    device = total_error.device
    errors64 = {
        name: value.detach().to(device=device, dtype=torch.float64)
        for name, value in errors.items()
    }
    weights64 = proxy_weights.detach().to(device=device, dtype=torch.float64)
    scales64 = component_scales.detach().to(device=device, dtype=torch.float64)
    if any(not bool(torch.isfinite(value).all()) for value in errors64.values()):
        raise ValueError("error tensors must contain only finite values")
    if not bool(torch.isfinite(weights64).all()):
        raise ValueError("proxy_weights must contain only finite values")
    if not bool(torch.isfinite(scales64).all()) or not bool((scales64 > 0.0).all()):
        raise ValueError("component_scales must be finite and strictly positive")
    if bool((weights64 < 0.0).any()):
        raise ValueError("proxy_weights must be nonnegative")
    weight_sum = weights64.sum()
    if not bool(weight_sum > 0.0):
        raise ValueError("proxy_weights must have positive total weight")

    scale_shape = (1,) * (total_error.ndim - 1) + (total_error.shape[-1],)
    scales_view = scales64.reshape(scale_shape)
    total_scaled = errors64["total_error"] / scales_view
    propagated_scaled = errors64["propagated_input_error"] / scales_view
    fresh_scaled = errors64["fresh_one_step_defect"] / scales_view
    weights_view = weights64.unsqueeze(-1)

    def energy(value: torch.Tensor) -> torch.Tensor:
        return (weights_view * value.square()).sum() / weight_sum

    total_energy = energy(total_scaled)
    propagated_energy = energy(propagated_scaled)
    fresh_energy = energy(fresh_scaled)
    cross_term = (weights_view * propagated_scaled * fresh_scaled).sum() / weight_sum
    energy_rhs = propagated_energy + fresh_energy + 2.0 * cross_term
    energy_residual = total_energy - energy_rhs
    tiny = torch.finfo(torch.float64).tiny
    energy_denominator = max(
        float(total_energy.abs().detach().cpu()),
        float(
            (propagated_energy.abs() + fresh_energy.abs() + 2.0 * cross_term.abs())
            .detach()
            .cpu()
        ),
        tiny,
    )

    vector_residual = total_scaled - propagated_scaled - fresh_scaled
    vector_residual_l2 = torch.sqrt(energy(vector_residual))
    vector_denominator = max(
        float(torch.sqrt(total_energy).detach().cpu()),
        float(
            (torch.sqrt(propagated_energy) + torch.sqrt(fresh_energy)).detach().cpu()
        ),
        tiny,
    )
    return {
        "weighted_total_energy": float(total_energy.detach().cpu()),
        "weighted_propagated_energy": float(propagated_energy.detach().cpu()),
        "weighted_fresh_energy": float(fresh_energy.detach().cpu()),
        "weighted_propagated_fresh_cross_term": float(cross_term.detach().cpu()),
        "weighted_energy_identity_rhs": float(energy_rhs.detach().cpu()),
        "weighted_energy_closure_abs": float(energy_residual.abs().detach().cpu()),
        "weighted_energy_closure_denominator": energy_denominator,
        "weighted_energy_closure_relative": float(energy_residual.abs().detach().cpu())
        / energy_denominator,
        "weighted_vector_closure_l2": float(vector_residual_l2.detach().cpu()),
        "weighted_vector_closure_denominator": vector_denominator,
        "weighted_vector_closure_relative": float(vector_residual_l2.detach().cpu())
        / vector_denominator,
    }


def validate_provenance_row(
    row: Mapping[str, Any],
    *,
    comparison_kind: ComparisonKind,
) -> dict[str, Any]:
    """Validate a scientific result row; preflight artifacts use other schemas."""

    if comparison_kind not in COMPARISON_KINDS:
        raise ValueError(f"unsupported comparison_kind: {comparison_kind}")
    if not str(row.get("claim_boundary", "")).strip():
        raise ValueError(
            f"{comparison_kind} comparisons require a nonempty claim_boundary"
        )
    if comparison_kind == "training_factor_causal_attribution":
        raise ValueError(
            "training_factor_causal_attribution requires a separate registered "
            "intervention/validator; D087 registers no matched or randomized "
            "training intervention"
        )
    missing = [field for field in PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing provenance fields: {missing}")
    unresolved_value = row.get("unresolved_fields", [])
    if not isinstance(unresolved_value, Sequence) or isinstance(
        unresolved_value, (str, bytes)
    ):
        raise TypeError("unresolved_fields must be a sequence of field names")
    unresolved = set(unresolved_value)
    unknown = unresolved.difference(PROVENANCE_FIELDS)
    if unknown:
        raise ValueError(f"unknown unresolved fields: {sorted(unknown)}")
    null_fields = {field for field in PROVENANCE_FIELDS if row[field] is None}
    if null_fields != unresolved:
        raise ValueError(
            "None-valued fields and unresolved_fields must match exactly: "
            f"none={sorted(null_fields)}, declared={sorted(unresolved)}"
        )
    for field in DIGEST_FIELDS:
        digest = row[field]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    checkpoint_size = row["checkpoint_size_bytes"]
    if checkpoint_size is not None and (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size <= 0
    ):
        raise ValueError("checkpoint_size_bytes must be a positive integer")
    for field in (
        "system_id",
        "checkpoint_path_contract",
        "training_precision",
        "inference_precision",
        "state_representation",
    ):
        value = row[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field} must be a nonempty string when resolved")
    if row["state_representation"] is not None:
        _check_representation(str(row["state_representation"]))

    def contract_is_nonempty(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return bool(value)
        return False

    for field in (
        "optimizer_history",
        "training_history",
        "recurrence",
        "boundary_policy",
        "selection_rule",
    ):
        value = row[field]
        if value is not None and not contract_is_nonempty(value):
            raise ValueError(f"{field} must be a nonempty contract when resolved")
    training_seed = row["training_seed"]
    if training_seed is not None and (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ValueError("training_seed must be a nonnegative integer when resolved")
    for field in ("training_presentations", "optimizer_steps"):
        value = row[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{field} must be a positive integer when resolved")
    if comparison_kind == "descriptive_survival":
        missing_survival_identity = unresolved.intersection(
            DESCRIPTIVE_SURVIVAL_REQUIRED_FIELDS
        )
        if missing_survival_identity:
            raise ValueError(
                "descriptive_survival requires executable survival provenance; "
                f"unresolved={sorted(missing_survival_identity)}"
            )
    if comparison_kind == "inference_map_diagnostic":
        unresolved_inference = unresolved.difference(TRAINING_SIDE_PROVENANCE_FIELDS)
        if unresolved_inference:
            raise ValueError(
                "inference_map_diagnostic requires all inference-side provenance; "
                f"unresolved={sorted(unresolved_inference)}"
            )
    return {
        "system_id": row["system_id"],
        "comparison_kind": comparison_kind,
        "unresolved_fields": sorted(unresolved),
        "provenance_complete": not unresolved,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen synthetic-only A1 contract and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.dry_run:
        _parser().error("A1 exposes only --dry-run; checkpoint/data execution is A2")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "scope": "synthetic_contract_only",
                "events": list(EVENT_NAMES),
                "stages": list(STAGE_NAMES),
                "thresholds": StabilityThresholds().as_dict(),
                "comparison_kinds": list(COMPARISON_KINDS),
                "required_provenance_fields": list(PROVENANCE_FIELDS),
                "descriptive_survival_required_fields": sorted(
                    DESCRIPTIVE_SURVIVAL_REQUIRED_FIELDS
                ),
                "inference_map_permitted_unresolved_fields": sorted(
                    TRAINING_SIDE_PROVENANCE_FIELDS
                ),
                "provenance_validator_scope": "scientific_result_rows_only",
                "training_factor_causal_attribution_supported": False,
                "final_hash_manifest_schema": FINAL_HASH_MANIFEST_SCHEMA,
                "artifact_finalization_contract": {
                    "scientific_root": "explicit_closed_top_level_files_only",
                    "live_transport_log": "outside_scientific_root_until_closed",
                    "wrapper_exit_receipt": "outside_scientific_root_until_closed",
                    "manifest_self_hash": "excluded",
                    "post_write_reverification": "required",
                },
                "checkpoint_loading_supported": False,
                "dataset_loading_supported": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
