"""Manifest-only provenance checks for the W26-L5 A46-A2 preflight.

This module intentionally uses only the Python standard library.  It does not
import the PCNO model stack and never opens checkpoint tensors or state arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REFERENCE_CHECKPOINT_SHA256 = (
    "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9"
)
POPULATION_SCHEMA = "pcno_same_state_refresh_population_v1"
PREFLIGHT_SCHEMA = "pcno_same_state_refresh_metadata_preflight_v1"
SELECTOR_WORKING_ID = "W26-L5-P6-RFB19-A31-SP19-BINARY-POSITION-PHASE-ROLLOUT"
EXPECTED_INPUT_CALLS = tuple(range(30))
EXPECTED_OUTPUT_CALLS = tuple(range(1, 31))
EXPECTED_NATIVE_RESOLUTION = (250, 100)
EXPECTED_FINE_RESOLUTION = (500, 200)
EXPECTED_MODEL_DELTA_T = 0.02
EXPECTED_SELECTOR_THRESHOLD = 0.8125
EXPECTED_REFERENCE_NODE_TYPES = "physical dynamic-FV semantics regenerated on each grid"
MODEL_CONFIG_KEYS = (
    "model",
    "in_dim",
    "out_dim",
    "k_max",
    "domain_lengths",
    "layers",
    "fc_dim",
    "nmeasures",
)
NORMALIZATION_KEYS = (
    "gamma",
    "mach_mean",
    "mach_scale",
    "mach_scale_floor",
    "residual_scale",
    "state_mean",
    "state_scale",
    "weight_provenance",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_PATTERN = re.compile(r"^sv_[A-Za-z0-9_-]+$")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Read one JSON metadata object, rejecting array/checkpoint paths."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError(f"metadata input must be JSON: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metadata input must contain a JSON object: {source}")
    return value


def canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON payload after removing any self-referential digest."""

    payload = dict(value)
    payload.pop("payload_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def with_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    digest = _string(value, name=name).lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _canonical_sequence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_sequence(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_canonical_sequence(item) for item in value)
    return value


def _selected_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: _canonical_sequence(value.get(key)) for key in keys}


def _normalization_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = _selected_mapping(value, NORMALIZATION_KEYS)
    for key in ("state_mean", "state_scale", "residual_scale"):
        sequence = contract[key]
        if not isinstance(sequence, tuple) or len(sequence) != 4:
            raise ValueError(f"normalization {key} must contain four values")
        for index, item in enumerate(sequence):
            _finite_number(item, name=f"normalization.{key}[{index}]")
    for key in ("gamma", "mach_mean", "mach_scale", "mach_scale_floor"):
        _finite_number(contract[key], name=f"normalization.{key}")
    return contract


def _effective_node_type_policy(
    summary: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> str:
    control = summary.get("node_type_channel_control")
    args = run_contract.get("args")
    input_policy = (
        args.get("model_node_type_input") if isinstance(args, Mapping) else None
    )
    if control == "no_type_channels":
        return "omitted"
    if input_policy == "all_normal":
        return "four_literal_zero_channels"
    if input_policy == "physical":
        return EXPECTED_REFERENCE_NODE_TYPES
    return "unresolved"


def build_reference_contract(
    *,
    source_manifest: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    resolution_run_contract: Mapping[str, Any],
    normalization: Mapping[str, Any],
    normalization_file_sha256: str,
) -> dict[str, Any]:
    """Normalize and authenticate the frozen A28--A45 checkpoint contract."""

    base = _mapping(
        source_manifest.get("base_runtime_source_manifest"),
        name="source_manifest.base_runtime_source_manifest",
    )
    checkpoint_contract = _mapping(
        base.get("checkpoint_contract"), name="base.checkpoint_contract"
    )
    external = _mapping(base.get("external_sha256"), name="base.external_sha256")
    model_config = _mapping(
        checkpoint_contract.get("model_config"), name="checkpoint.model_config"
    )
    transfer = _mapping(
        base.get("resolution_contract"), name="base.resolution_contract"
    )
    resolution = _mapping(
        resolution_run_contract.get("resolution_contract"),
        name="resolution_run_contract.resolution_contract",
    )
    registered_checkpoint = _mapping(
        resolution_run_contract.get("checkpoint"),
        name="resolution_run_contract.checkpoint",
    )
    training_data = _mapping(
        training_summary.get("data_contract"), name="training_summary.data_contract"
    )
    checkpoint_sha = _sha256(external.get("checkpoint"), name="reference checkpoint")
    normalization_sha = _sha256(
        normalization_file_sha256, name="reference normalization file"
    )
    normalized = _normalization_contract(normalization)

    checks = {
        "checkpoint_sha_registered": (
            checkpoint_sha == REFERENCE_CHECKPOINT_SHA256
            and registered_checkpoint.get("sha256") == checkpoint_sha
        ),
        "checkpoint_contract_matches_training_summary": (
            _selected_mapping(model_config, MODEL_CONFIG_KEYS)
            == _selected_mapping(
                _mapping(training_summary.get("model_config"), name="training model"),
                MODEL_CONFIG_KEYS,
            )
            and checkpoint_contract.get("boundary_mode")
            == training_summary.get("boundary_mode")
            and checkpoint_contract.get("raw_recurrence")
            == training_summary.get("raw_recurrence")
            and checkpoint_contract.get("step_stride")
            == training_summary.get("step_stride")
        ),
        "normalization_digest_bound": (
            checkpoint_contract.get("normalization_digest")
            == training_summary.get("normalization_digest")
            == registered_checkpoint.get("normalization_digest")
        ),
        "normalization_file_bound": (
            external.get("normalization_file") == normalization_sha
        ),
        "physical_fourier_periods_bound": (
            _canonical_sequence(model_config.get("domain_lengths"))
            == _canonical_sequence(resolution.get("fourier_periods"))
        ),
        "physical_node_types_bound": (
            resolution.get("node_types") == EXPECTED_REFERENCE_NODE_TYPES
        ),
        "boundary_policy_bound": (
            resolution.get("boundary_policy") == "model_all_nodes raw recurrence"
            and checkpoint_contract.get("boundary_mode") == "model_all_nodes"
            and checkpoint_contract.get("raw_recurrence") is True
        ),
        "native_fine_geometry_bound": (
            _canonical_sequence(transfer.get("native")) == EXPECTED_NATIVE_RESOLUTION
            and _canonical_sequence(transfer.get("fine")) == EXPECTED_FINE_RESOLUTION
            and transfer.get("fine_input")
            == "float64 piecewise-constant prolongation of native physical state"
            and transfer.get("model_boundary")
            == "one explicit FP32 rounding per actual grid"
        ),
        "physical_volume_component_scale_bound": (
            normalized.get("weight_provenance")
            == "validated_physical_cell_volume_normalized"
            and all(float(value) > 0.0 for value in normalized["state_scale"])
        ),
    }
    data_manifest_digest = checkpoint_contract.get("data_manifest_digest")
    if data_manifest_digest != training_summary.get("data_manifest_digest"):
        checks["checkpoint_contract_matches_training_summary"] = False

    return {
        "checks": checks,
        "valid": all(checks.values()),
        "checkpoint_sha256": checkpoint_sha,
        "model_config": _selected_mapping(model_config, MODEL_CONFIG_KEYS),
        "boundary_mode": checkpoint_contract.get("boundary_mode"),
        "raw_recurrence": checkpoint_contract.get("raw_recurrence"),
        "step_stride": checkpoint_contract.get("step_stride"),
        "normalization_digest": checkpoint_contract.get("normalization_digest"),
        "normalization": normalized,
        "component_scale": normalized["state_scale"],
        "node_type_policy": EXPECTED_REFERENCE_NODE_TYPES,
        "boundary_field_mode": "none",
        "boundary_field_names": (),
        "data_manifest_digest": data_manifest_digest,
        "source_family_id": training_data.get("source_family_id"),
        "source_family_manifest_digest": training_data.get(
            "source_family_manifest_digest"
        ),
        "geometry_contract_digest": training_data.get("geometry_contract_digest"),
        "resolution_contract_digest": training_data.get("resolution_contract_digest"),
        "state_convention": training_data.get("state_convention"),
        "time_contract": _canonical_sequence(training_data.get("time_contract")),
        "native_resolution": EXPECTED_NATIVE_RESOLUTION,
        "fine_resolution": EXPECTED_FINE_RESOLUTION,
        "model_boundary": transfer.get("model_boundary"),
        "fine_input": transfer.get("fine_input"),
    }


def build_candidate_contract(
    *,
    candidate_id: str,
    summary: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    normalization: Mapping[str, Any],
    checkpoint_file_sha256: str | None,
    checkpoint_file_bytes: int | None,
) -> dict[str, Any]:
    """Normalize one independently trained checkpoint's metadata."""

    artifact_sha = _mapping(
        summary.get("artifact_sha256"), name=f"{candidate_id}.artifact_sha256"
    )
    checkpoint_sha = _sha256(
        artifact_sha.get("best_checkpoint"),
        name=f"{candidate_id}.best_checkpoint",
    )
    model_config = _mapping(
        summary.get("model_config"), name=f"{candidate_id}.model_config"
    )
    data_contract = _mapping(
        run_contract.get("data_contract") or summary.get("data_contract"),
        name=f"{candidate_id}.data_contract",
    )
    normalized = _normalization_contract(normalization)
    args = run_contract.get("args")
    args_present = isinstance(args, Mapping)
    args = args if args_present else {}
    boundary_field_mode = summary.get(
        "boundary_field_mode", args.get("boundary_field_mode", "none")
    )
    boundary_field_names = summary.get("boundary_field_names", ())
    boundary_field_names = _canonical_sequence(boundary_field_names)
    if boundary_field_names is None:
        boundary_field_names = ()
    file_sha = checkpoint_file_sha256.lower() if checkpoint_file_sha256 else None

    return {
        "candidate_id": _string(candidate_id, name="candidate_id"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_file_sha256": file_sha,
        "checkpoint_file_bytes": checkpoint_file_bytes,
        "checkpoint_file_present": file_sha is not None,
        "model_config": _selected_mapping(model_config, MODEL_CONFIG_KEYS),
        "boundary_mode": summary.get("boundary_mode"),
        "raw_recurrence": summary.get("raw_recurrence"),
        "step_stride": summary.get("step_stride"),
        "normalization_digest": summary.get("normalization_digest"),
        "normalization": normalized,
        "component_scale": normalized["state_scale"],
        "node_type_policy": _effective_node_type_policy(summary, run_contract),
        "node_type_channel_control": summary.get("node_type_channel_control"),
        "boundary_field_mode": boundary_field_mode,
        "boundary_field_names": boundary_field_names,
        "data_manifest_digest": summary.get("data_manifest_digest"),
        "source_family_id": data_contract.get("source_family_id"),
        "source_family_manifest_digest": data_contract.get(
            "source_family_manifest_digest"
        ),
        "geometry_contract_digest": data_contract.get("geometry_contract_digest"),
        "resolution_contract_digest": data_contract.get("resolution_contract_digest"),
        "state_convention": data_contract.get("state_convention"),
        "time_contract": _canonical_sequence(data_contract.get("time_contract")),
        "independent_initialization_attested": (
            "parent_checkpoint" in summary
            and summary.get("parent_checkpoint") is None
            and args_present
            and "init_checkpoint" in args
            and "resume_checkpoint" in args
            and args.get("init_checkpoint") in (None, "")
            and args.get("resume_checkpoint") in (None, "")
        ),
    }


def evaluate_candidate_contract(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare one candidate against the immutable A46 compatibility cell."""

    checks = {
        "checkpoint_sha_distinct": (
            candidate.get("checkpoint_sha256") != reference.get("checkpoint_sha256")
        ),
        "checkpoint_bytes_rehashed": bool(candidate.get("checkpoint_file_present")),
        "checkpoint_bytes_match_declared_sha": (
            candidate.get("checkpoint_file_sha256")
            == candidate.get("checkpoint_sha256")
        ),
        "checkpoint_nonempty": (
            isinstance(candidate.get("checkpoint_file_bytes"), int)
            and not isinstance(candidate.get("checkpoint_file_bytes"), bool)
            and int(candidate["checkpoint_file_bytes"]) > 0
        ),
        "independent_initialization_attested": (
            candidate.get("independent_initialization_attested") is True
        ),
        "model_config_exact": (
            candidate.get("model_config") == reference.get("model_config")
        ),
        "physical_fourier_periods_exact": (
            candidate.get("model_config", {}).get("domain_lengths")
            == reference.get("model_config", {}).get("domain_lengths")
        ),
        "boundary_policy_exact": (
            candidate.get("boundary_mode") == reference.get("boundary_mode")
            and candidate.get("raw_recurrence") is True
            and candidate.get("step_stride") == reference.get("step_stride")
        ),
        "normalization_digest_exact": (
            candidate.get("normalization_digest")
            == reference.get("normalization_digest")
        ),
        "normalization_values_exact": (
            candidate.get("normalization") == reference.get("normalization")
        ),
        "component_scale_exact": (
            candidate.get("component_scale") == reference.get("component_scale")
        ),
        "physical_volume_provenance_exact": (
            candidate.get("normalization", {}).get("weight_provenance")
            == reference.get("normalization", {}).get("weight_provenance")
        ),
        "physical_node_type_semantics_exact": (
            candidate.get("node_type_policy") == reference.get("node_type_policy")
        ),
        "no_added_boundary_fields": (
            candidate.get("boundary_field_mode") in (None, "none")
            and candidate.get("boundary_field_names") in (None, (), [])
        ),
        "training_data_manifest_exact": (
            candidate.get("data_manifest_digest")
            == reference.get("data_manifest_digest")
        ),
        "source_family_exact": (
            candidate.get("source_family_id") == reference.get("source_family_id")
            and candidate.get("source_family_manifest_digest")
            == reference.get("source_family_manifest_digest")
        ),
        "native_geometry_exact": (
            candidate.get("geometry_contract_digest")
            == reference.get("geometry_contract_digest")
            and candidate.get("resolution_contract_digest")
            == reference.get("resolution_contract_digest")
        ),
        "state_and_time_contract_exact": (
            candidate.get("state_convention") == reference.get("state_convention")
            and candidate.get("time_contract") == reference.get("time_contract")
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return {
        **candidate,
        "checks": checks,
        "failed_checks": failed,
        "eligible": all(checks.values()),
    }


def collect_registered_case_ids(*values: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect case IDs only from explicit source-manifest inventory fields."""

    keys = {
        "case_ids",
        "calibration_cases",
        "evaluation_cases",
        "expected_active_cases",
    }
    result: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if (
                    key in keys
                    and isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes, bytearray))
                ):
                    for case_id in item:
                        if isinstance(case_id, str) and CASE_ID_PATTERN.fullmatch(
                            case_id
                        ):
                            result.add(case_id)
                else:
                    visit(item)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                visit(item)

    for item in values:
        visit(item)
    return tuple(sorted(result))


def _call_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    calls: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers")
        calls.append(item)
    return tuple(calls)


def _string_inventory(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        result.append(item)
    return tuple(result)


def validate_population_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    opened_case_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate the unopened new-case contract without reading any array."""

    if manifest is None:
        return {
            "present": False,
            "valid": False,
            "checks": {"population_manifest_present": False},
            "failed_checks": ("population_manifest_present",),
            "branch_active_case_ids": (),
            "recurrence_active_case_ids": (),
        }

    selector = manifest.get("selector")
    selector = selector if isinstance(selector, Mapping) else {}
    generation = manifest.get("generation")
    generation = generation if isinstance(generation, Mapping) else {}
    calls = manifest.get("calls")
    calls = calls if isinstance(calls, Mapping) else {}
    cases_value = manifest.get("cases")
    cases = (
        list(cases_value)
        if isinstance(cases_value, Sequence)
        and not isinstance(cases_value, (str, bytes, bytearray))
        else []
    )

    case_ids: list[str] = []
    branch_active: list[str] = []
    recurrence_active: list[str] = []
    active_by_population_group: dict[str, dict[str, list[str]]] = {
        "branch_label": defaultdict(list),
        "recurrence": defaultdict(list),
    }
    rows_well_formed = True
    all_unopened = True
    all_truth_metadata = True
    all_positive_volumes = True
    inactive_controls_retained = True
    no_test_members = True
    physical_times_match = True
    solver_sources_match = True
    declared_physical_times = calls.get("physical_times_sha256")
    declared_solver_source = generation.get("solver_source_sha256")
    for index, row_value in enumerate(cases):
        if not isinstance(row_value, Mapping):
            rows_well_formed = False
            continue
        try:
            case_id = _string(row_value.get("case_id"), name=f"cases[{index}].case_id")
            group = _string(
                row_value.get("strength_group"),
                name=f"cases[{index}].strength_group",
            )
            _string(row_value.get("position_id"), name=f"cases[{index}].position_id")
        except (TypeError, ValueError):
            rows_well_formed = False
            continue
        population = row_value.get("population")
        active = row_value.get("selector_active")
        if (
            CASE_ID_PATTERN.fullmatch(case_id) is None
            or population not in active_by_population_group
            or not isinstance(active, bool)
            or row_value.get("generated") is not True
        ):
            rows_well_formed = False
        case_ids.append(case_id)
        if active is True and population in active_by_population_group:
            active_by_population_group[population][group].append(case_id)
            if population == "branch_label":
                branch_active.append(case_id)
            else:
                recurrence_active.append(case_id)
        if row_value.get("outcome_opened") is not False:
            all_unopened = False
        if row_value.get("project_test_member") is not False:
            no_test_members = False
        try:
            volume_min = _finite_number(
                row_value.get("physical_volume_min"),
                name=f"cases[{index}].physical_volume_min",
            )
            all_positive_volumes = all_positive_volumes and volume_min > 0.0
            for key in (
                "truth_metadata_manifest_sha256",
                "truth_array_sha256",
                "solver_source_sha256",
                "configuration_sha256",
                "physical_times_sha256",
            ):
                _sha256(row_value.get(key), name=f"cases[{index}].{key}")
            all_truth_metadata = all_truth_metadata and (
                row_value.get("retained_evolved_native_truth") is True
            )
            physical_times_match = physical_times_match and (
                row_value.get("physical_times_sha256") == declared_physical_times
            )
            solver_sources_match = solver_sources_match and (
                row_value.get("solver_source_sha256") == declared_solver_source
            )
        except (TypeError, ValueError):
            all_truth_metadata = False
            all_positive_volumes = False
            physical_times_match = False
            solver_sources_match = False
        if active is False and row_value.get("inactive_control_retained") is not True:
            inactive_controls_retained = False

    branch_groups = active_by_population_group["branch_label"]
    recurrence_groups = active_by_population_group["recurrence"]
    opened = set(opened_case_ids)
    input_calls: tuple[int, ...] = ()
    output_calls: tuple[int, ...] = ()
    try:
        input_calls = _call_tuple(calls.get("input_calls"), name="calls.input_calls")
        output_calls = _call_tuple(calls.get("output_calls"), name="calls.output_calls")
    except (TypeError, ValueError):
        pass

    branch_fit_declared = _string_inventory(manifest.get("branch_fit_case_ids"))
    recurrence_declared = _string_inventory(manifest.get("recurrence_case_ids"))
    checks = {
        "population_manifest_present": True,
        "schema_exact": manifest.get("schema") == POPULATION_SCHEMA,
        "frozen_before_outcomes": (
            manifest.get("frozen_before_outcomes") is True
            and manifest.get("reference_arrays_opened") is False
            and manifest.get("truth_arrays_loaded") is False
        ),
        "dynamic_fv_only": manifest.get("family") == "dynamic_shock_vortex_fv",
        "selector_exact_and_truth_free": (
            selector.get("working_id") == SELECTOR_WORKING_ID
            and selector.get("truth_free") is True
            and selector.get("feature_clock") == "call_zero_native_physical_state"
            and selector.get("normalized_wall_distance_max")
            == EXPECTED_SELECTOR_THRESHOLD
            and SHA256_PATTERN.fullmatch(str(selector.get("source_sha256", "")))
            is not None
        ),
        "generation_manifest_complete": (
            generation.get("inventory_complete") is True
            and generation.get("newly_generated") is True
            and generation.get("project_test_split_excluded") is True
            and generation.get("solver") == "WENO5-HLLC-SSPRK3"
            and all(
                SHA256_PATTERN.fullmatch(str(generation.get(key, ""))) is not None
                for key in (
                    "solver_source_sha256",
                    "configuration_sha256",
                    "family_manifest_sha256",
                )
            )
        ),
        "time_alignment_exact": (
            input_calls == EXPECTED_INPUT_CALLS
            and output_calls == EXPECTED_OUTPUT_CALLS
            and calls.get("model_delta_t") == EXPECTED_MODEL_DELTA_T
            and _canonical_sequence(calls.get("native_resolution"))
            == EXPECTED_NATIVE_RESOLUTION
            and _canonical_sequence(calls.get("fine_resolution"))
            == EXPECTED_FINE_RESOLUTION
            and SHA256_PATTERN.fullmatch(str(declared_physical_times or "")) is not None
            and physical_times_match
            and solver_sources_match
        ),
        "case_rows_well_formed": bool(cases) and rows_well_formed,
        "case_ids_unique": len(case_ids) == len(set(case_ids)),
        "outside_opened_populations": not set(case_ids).intersection(opened),
        "branch_group_cardinality": (
            len(branch_groups) >= 4
            and all(len(ids) >= 2 for ids in branch_groups.values())
        ),
        "recurrence_group_cardinality": (
            len(recurrence_groups) >= 2
            and all(len(ids) >= 2 for ids in recurrence_groups.values())
        ),
        "branch_recurrence_groups_disjoint": not set(branch_groups).intersection(
            recurrence_groups
        ),
        "branch_fit_inventory_exact": (
            branch_fit_declared is not None
            and tuple(sorted(branch_fit_declared)) == tuple(sorted(branch_active))
        ),
        "recurrence_inventory_exact": (
            recurrence_declared is not None
            and tuple(sorted(recurrence_declared)) == tuple(sorted(recurrence_active))
        ),
        "all_cases_unopened": all_unopened,
        "project_test_sealed": no_test_members,
        "retained_truth_metadata_complete": all_truth_metadata,
        "physical_volumes_positive": all_positive_volumes,
        "inactive_controls_retained": inactive_controls_retained,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return {
        "present": True,
        "valid": all(checks.values()),
        "checks": checks,
        "failed_checks": failed,
        "case_count": len(cases),
        "branch_strength_groups": tuple(sorted(branch_groups)),
        "recurrence_strength_groups": tuple(sorted(recurrence_groups)),
        "branch_active_case_ids": tuple(sorted(branch_active)),
        "recurrence_active_case_ids": tuple(sorted(recurrence_active)),
        "opened_case_overlap": tuple(sorted(set(case_ids).intersection(opened))),
        "duplicate_case_ids": tuple(
            sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
        ),
    }


def build_metadata_preflight(
    *,
    reference: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any] | None,
    opened_case_ids: Sequence[str],
    selected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the fail-closed A46-A2 metadata decision."""

    candidate_reports = tuple(
        evaluate_candidate_contract(reference, candidate) for candidate in candidates
    )
    eligible = tuple(row for row in candidate_reports if row["eligible"])
    selected = None
    if selected_checkpoint_sha256 is not None:
        selected_matches = tuple(
            row
            for row in eligible
            if row.get("checkpoint_sha256") == selected_checkpoint_sha256
        )
        if len(selected_matches) == 1:
            selected = selected_matches[0]
    elif len(eligible) == 1:
        selected = eligible[0]

    population_report = validate_population_manifest(
        population, opened_case_ids=opened_case_ids
    )
    checks = {
        "reference_contract_valid": reference.get("valid") is True,
        "candidate_inventory_nonempty": bool(candidate_reports),
        "one_compatible_checkpoint_selected": selected is not None,
        "population_contract_valid": population_report["valid"] is True,
        "metadata_only_activity": True,
    }
    status = "ready" if all(checks.values()) else "blocked"
    blockers: list[str] = []
    if not checks["reference_contract_valid"]:
        blockers.append("reference_contract_invalid")
    if not candidate_reports:
        blockers.append("candidate_inventory_empty")
    elif not eligible:
        blockers.append("no_compatible_independent_checkpoint")
    elif selected is None:
        blockers.append("compatible_checkpoint_selection_ambiguous")
    if not population_report["present"]:
        blockers.append("new_case_population_manifest_missing")
    elif not population_report["valid"]:
        blockers.append("new_case_population_manifest_invalid")

    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "ready_for_branch_label_execution": status == "ready",
        "checks": checks,
        "blockers": tuple(blockers),
        "reference_contract": reference,
        "candidate_reports": candidate_reports,
        "eligible_checkpoint_sha256": tuple(
            row["checkpoint_sha256"] for row in eligible
        ),
        "selected_checkpoint_sha256": (
            selected.get("checkpoint_sha256") if selected is not None else None
        ),
        "opened_case_inventory": {
            "case_count": len(set(opened_case_ids)),
            "case_ids": tuple(sorted(set(opened_case_ids))),
        },
        "population_report": population_report,
        "activity": {
            "checkpoint_tensor_loads": 0,
            "model_builds": 0,
            "model_predictions": 0,
            "truth_array_loads": 0,
            "state_array_loads": 0,
            "population_outcomes_opened": 0,
            "remote_commands": 0,
            "controllers_run": 0,
        },
        "claim_limit": (
            "Metadata compatibility and population readiness only; no branch utility, "
            "coefficient transfer, recurrent benefit, conservation, or deployment claim."
        ),
    }
