from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from scripts.time_dependent_no import (
    preflight_pcno_same_state_refresh_metadata as metadata_preflight,
)
from scripts.time_dependent_no.preflight_pcno_same_state_refresh_metadata import (
    run_preflight,
)
from utility.time_dependent_no.pcno_same_state_refresh_metadata import (
    EXPECTED_REFERENCE_NODE_TYPES,
    POPULATION_SCHEMA,
    REFERENCE_CHECKPOINT_SHA256,
    SELECTOR_WORKING_ID,
    build_candidate_contract,
    build_metadata_preflight,
    build_reference_contract,
    canonical_payload_sha256,
    collect_registered_case_ids,
    evaluate_candidate_contract,
    load_json_object,
    sha256_file,
    validate_population_manifest,
    with_payload_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
NORMALIZATION_DIGEST = "normalization-contract-v1"


def _normalization() -> dict[str, object]:
    return {
        "gamma": 1.4,
        "mach_mean": 1.1,
        "mach_scale": 0.1,
        "mach_scale_floor": 0.1,
        "residual_scale": [0.01, 0.02, 0.03, 0.04],
        "state_mean": [1.0, 2.0, 3.0, 4.0],
        "state_scale": [0.1, 0.2, 0.3, 0.4],
        "weight_provenance": "validated_physical_cell_volume_normalized",
    }


def _model_config() -> dict[str, object]:
    return {
        "model": "PCNOEuler2DResidual",
        "in_dim": 12,
        "out_dim": 4,
        "k_max": 8,
        "domain_lengths": [2.0, 1.0],
        "layers": [128, 128, 128, 128, 128],
        "fc_dim": 128,
        "nmeasures": 1,
    }


def _data_contract() -> dict[str, object]:
    return {
        "source_family_id": "shock_vortex_eps_y_new_v1",
        "source_family_manifest_digest": SHA_A,
        "geometry_contract_digest": SHA_B,
        "resolution_contract_digest": SHA_C,
        "state_convention": "conservative_[rho,rho_u,rho_v,E]",
        "time_contract": {
            "delta_t": 0.01,
            "final_time": 0.6,
            "saved_calls": 60,
            "uniform_delta_t": True,
        },
    }


def _reference_inputs() -> tuple[dict, dict, dict, dict]:
    model = _model_config()
    data = _data_contract()
    training_summary = {
        "artifact_sha256": {"best_checkpoint": REFERENCE_CHECKPOINT_SHA256},
        "model_config": model,
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "step_stride": 2,
        "normalization_digest": NORMALIZATION_DIGEST,
        "data_manifest_digest": SHA_D,
        "data_contract": data,
    }
    source_manifest = {
        "base_runtime_source_manifest": {
            "checkpoint_contract": {
                "model_config": model,
                "boundary_mode": "model_all_nodes",
                "raw_recurrence": True,
                "step_stride": 2,
                "normalization_digest": NORMALIZATION_DIGEST,
                "data_manifest_digest": SHA_D,
            },
            "external_sha256": {
                "checkpoint": REFERENCE_CHECKPOINT_SHA256,
                "normalization_file": SHA_A,
            },
            "resolution_contract": {
                "native": [250, 100],
                "fine": [500, 200],
                "fine_input": (
                    "float64 piecewise-constant prolongation of native physical state"
                ),
                "model_boundary": "one explicit FP32 rounding per actual grid",
            },
        }
    }
    resolution_contract = {
        "checkpoint": {
            "sha256": REFERENCE_CHECKPOINT_SHA256,
            "normalization_digest": NORMALIZATION_DIGEST,
        },
        "resolution_contract": {
            "fourier_periods": [2.0, 1.0],
            "node_types": EXPECTED_REFERENCE_NODE_TYPES,
            "boundary_policy": "model_all_nodes raw recurrence",
        },
    }
    return source_manifest, training_summary, resolution_contract, _normalization()


def _reference() -> dict:
    source, summary, resolution, normalization = _reference_inputs()
    return build_reference_contract(
        source_manifest=source,
        training_summary=summary,
        resolution_run_contract=resolution,
        normalization=normalization,
        normalization_file_sha256=SHA_A,
    )


def _candidate(reference: dict | None = None) -> dict:
    reference = _reference() if reference is None else reference
    return {
        "candidate_id": "independent_seed",
        "checkpoint_sha256": SHA_B,
        "checkpoint_file_sha256": SHA_B,
        "checkpoint_file_bytes": 1024,
        "checkpoint_file_present": True,
        "model_config": copy.deepcopy(reference["model_config"]),
        "boundary_mode": reference["boundary_mode"],
        "raw_recurrence": True,
        "step_stride": reference["step_stride"],
        "normalization_digest": reference["normalization_digest"],
        "normalization": copy.deepcopy(reference["normalization"]),
        "component_scale": reference["component_scale"],
        "node_type_policy": reference["node_type_policy"],
        "node_type_channel_control": "physical_type_channels",
        "boundary_field_mode": "none",
        "boundary_field_names": (),
        "data_manifest_digest": reference["data_manifest_digest"],
        "source_family_id": reference["source_family_id"],
        "source_family_manifest_digest": reference["source_family_manifest_digest"],
        "geometry_contract_digest": reference["geometry_contract_digest"],
        "resolution_contract_digest": reference["resolution_contract_digest"],
        "state_convention": reference["state_convention"],
        "time_contract": copy.deepcopy(reference["time_contract"]),
        "independent_initialization_attested": True,
    }


def _population(opened_case: str = "sv_open_e00_y00") -> dict:
    rows: list[dict[str, object]] = []
    branch_active: list[str] = []
    recurrence_active: list[str] = []
    for population, group_count in (("branch_label", 4), ("recurrence", 2)):
        prefix = "b" if population == "branch_label" else "r"
        for group_index in range(group_count):
            group = f"{prefix}{group_index}"
            for position_index in range(3):
                case_id = f"sv_new_{group}_p{position_index}"
                active = position_index < 2
                row = {
                    "case_id": case_id,
                    "strength_group": group,
                    "position_id": f"p{position_index}",
                    "population": population,
                    "selector_active": active,
                    "generated": True,
                    "outcome_opened": False,
                    "project_test_member": False,
                    "retained_evolved_native_truth": True,
                    "truth_metadata_manifest_sha256": SHA_A,
                    "truth_array_sha256": SHA_B,
                    "solver_source_sha256": SHA_C,
                    "configuration_sha256": SHA_D,
                    "physical_times_sha256": SHA_A,
                    "physical_volume_min": 1.0e-5,
                    "inactive_control_retained": not active,
                }
                rows.append(row)
                if active and population == "branch_label":
                    branch_active.append(case_id)
                elif active:
                    recurrence_active.append(case_id)
    return {
        "schema": POPULATION_SCHEMA,
        "family": "dynamic_shock_vortex_fv",
        "frozen_before_outcomes": True,
        "reference_arrays_opened": False,
        "truth_arrays_loaded": False,
        "selector": {
            "working_id": SELECTOR_WORKING_ID,
            "truth_free": True,
            "feature_clock": "call_zero_native_physical_state",
            "normalized_wall_distance_max": 0.8125,
            "source_sha256": SHA_A,
        },
        "generation": {
            "inventory_complete": True,
            "newly_generated": True,
            "project_test_split_excluded": True,
            "solver": "WENO5-HLLC-SSPRK3",
            "solver_source_sha256": SHA_C,
            "configuration_sha256": SHA_D,
            "family_manifest_sha256": SHA_A,
        },
        "calls": {
            "input_calls": list(range(30)),
            "output_calls": list(range(1, 31)),
            "model_delta_t": 0.02,
            "native_resolution": [250, 100],
            "fine_resolution": [500, 200],
            "physical_times_sha256": SHA_A,
        },
        "cases": rows,
        "branch_fit_case_ids": branch_active,
        "recurrence_case_ids": recurrence_active,
        "opened_control": opened_case,
    }


def test_reference_contract_closes() -> None:
    reference = _reference()
    assert reference["valid"] is True
    assert all(reference["checks"].values())
    assert reference["component_scale"] == (0.1, 0.2, 0.3, 0.4)


def test_exact_independent_candidate_is_eligible() -> None:
    report = evaluate_candidate_contract(_reference(), _candidate())
    assert report["eligible"] is True
    assert report["failed_checks"] == ()


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("checkpoint_sha256", REFERENCE_CHECKPOINT_SHA256, "checkpoint_sha_distinct"),
        ("checkpoint_file_sha256", SHA_C, "checkpoint_bytes_match_declared_sha"),
        (
            "node_type_policy",
            "four_literal_zero_channels",
            "physical_node_type_semantics_exact",
        ),
        ("boundary_field_mode", "semantic_collar", "no_added_boundary_fields"),
        ("data_manifest_digest", SHA_A, "training_data_manifest_exact"),
        ("geometry_contract_digest", SHA_A, "native_geometry_exact"),
    ],
)
def test_candidate_mismatches_fail_closed(
    field: str, value: object, failed_check: str
) -> None:
    candidate = _candidate()
    candidate[field] = value
    report = evaluate_candidate_contract(_reference(), candidate)
    assert report["eligible"] is False
    assert failed_check in report["failed_checks"]


def test_model_and_normalization_mismatches_fail_closed() -> None:
    candidate = _candidate()
    candidate["model_config"] = {**candidate["model_config"], "in_dim": 8}
    candidate["normalization"] = {
        **candidate["normalization"],
        "state_scale": (0.2, 0.2, 0.3, 0.4),
    }
    report = evaluate_candidate_contract(_reference(), candidate)
    assert report["eligible"] is False
    assert "model_config_exact" in report["failed_checks"]
    assert "normalization_values_exact" in report["failed_checks"]


def test_missing_checkpoint_bytes_do_not_qualify() -> None:
    candidate = _candidate()
    candidate.update(
        {
            "checkpoint_file_present": False,
            "checkpoint_file_sha256": None,
            "checkpoint_file_bytes": None,
        }
    )
    report = evaluate_candidate_contract(_reference(), candidate)
    assert report["eligible"] is False
    assert "checkpoint_bytes_rehashed" in report["failed_checks"]
    assert "checkpoint_nonempty" in report["failed_checks"]


def test_candidate_normalizer_extracts_physical_and_omitted_policies() -> None:
    reference = _reference()
    summary = {
        "artifact_sha256": {"best_checkpoint": SHA_B},
        "model_config": _model_config(),
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "step_stride": 2,
        "normalization_digest": NORMALIZATION_DIGEST,
        "data_manifest_digest": SHA_D,
        "data_contract": _data_contract(),
        "parent_checkpoint": None,
    }
    run_contract = {
        "args": {
            "model_node_type_input": "physical",
            "init_checkpoint": None,
            "resume_checkpoint": None,
        },
        "data_contract": _data_contract(),
    }
    candidate = build_candidate_contract(
        candidate_id="physical",
        summary=summary,
        run_contract=run_contract,
        normalization=_normalization(),
        checkpoint_file_sha256=SHA_B,
        checkpoint_file_bytes=1024,
    )
    assert candidate["node_type_policy"] == reference["node_type_policy"]
    summary["node_type_channel_control"] = "no_type_channels"
    omitted = build_candidate_contract(
        candidate_id="omitted",
        summary=summary,
        run_contract=run_contract,
        normalization=_normalization(),
        checkpoint_file_sha256=SHA_B,
        checkpoint_file_bytes=1024,
    )
    assert omitted["node_type_policy"] == "omitted"


def test_valid_population_contract_closes() -> None:
    report = validate_population_manifest(
        _population(), opened_case_ids=("sv_open_e00_y00",)
    )
    assert report["valid"] is True
    assert report["branch_strength_groups"] == ("b0", "b1", "b2", "b3")
    assert report["recurrence_strength_groups"] == ("r0", "r1")


@pytest.mark.parametrize(
    "mutation",
    [
        "opened_overlap",
        "duplicate_case",
        "outcome_opened",
        "test_member",
        "nonpositive_volume",
        "bad_call_clock",
        "missing_branch_group",
        "shared_strength_group",
        "inactive_dropped",
        "physical_time_mismatch",
        "malformed_fit_inventory",
    ],
)
def test_population_failures_are_closed(mutation: str) -> None:
    manifest = _population()
    rows = manifest["cases"]
    if mutation == "opened_overlap":
        rows[0]["case_id"] = "sv_open_e00_y00"
        manifest["branch_fit_case_ids"][0] = "sv_open_e00_y00"
    elif mutation == "duplicate_case":
        rows[1]["case_id"] = rows[0]["case_id"]
        manifest["branch_fit_case_ids"][1] = rows[0]["case_id"]
    elif mutation == "outcome_opened":
        rows[0]["outcome_opened"] = True
    elif mutation == "test_member":
        rows[0]["project_test_member"] = True
    elif mutation == "nonpositive_volume":
        rows[0]["physical_volume_min"] = 0.0
    elif mutation == "bad_call_clock":
        manifest["calls"]["input_calls"] = list(range(1, 31))
    elif mutation == "missing_branch_group":
        rows[:] = [row for row in rows if row["strength_group"] != "b3"]
        manifest["branch_fit_case_ids"] = [
            case_id
            for case_id in manifest["branch_fit_case_ids"]
            if "_b3_" not in case_id
        ]
    elif mutation == "shared_strength_group":
        for row in rows:
            if row["strength_group"] == "r0":
                row["strength_group"] = "b0"
    elif mutation == "inactive_dropped":
        inactive = next(row for row in rows if not row["selector_active"])
        inactive["inactive_control_retained"] = False
    elif mutation == "physical_time_mismatch":
        rows[0]["physical_times_sha256"] = SHA_B
    elif mutation == "malformed_fit_inventory":
        manifest["branch_fit_case_ids"][0] = {"not": "a case id"}
    report = validate_population_manifest(
        manifest, opened_case_ids=("sv_open_e00_y00",)
    )
    assert report["valid"] is False
    assert report["failed_checks"]


def test_metadata_preflight_ready_and_blocked_paths() -> None:
    ready = build_metadata_preflight(
        reference=_reference(),
        candidates=(_candidate(),),
        population=_population(),
        opened_case_ids=("sv_open_e00_y00",),
    )
    assert ready["status"] == "ready"
    assert ready["selected_checkpoint_sha256"] == SHA_B
    blocked = build_metadata_preflight(
        reference=_reference(),
        candidates=(),
        population=None,
        opened_case_ids=("sv_open_e00_y00",),
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == (
        "candidate_inventory_empty",
        "new_case_population_manifest_missing",
    )
    assert not any(blocked["activity"].values())


def test_case_inventory_reads_only_explicit_source_fields() -> None:
    manifest = {
        "population": {"case_ids": ["sv_e12_y01"]},
        "nested": {
            "calibration_cases": ["sv_e01_y00"],
            "case_scores": [{"case_id": "sv_outcome_only"}],
        },
    }
    assert collect_registered_case_ids(manifest) == ("sv_e01_y00", "sv_e12_y01")


def test_json_loader_and_payload_hash_are_fail_closed(tmp_path: Path) -> None:
    bad_suffix = tmp_path / "metadata.npy"
    bad_suffix.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be JSON"):
        load_json_object(bad_suffix)
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_json_object(array_json)
    payload = with_payload_sha256({"b": 2, "a": 1})
    assert payload["payload_sha256"] == canonical_payload_sha256(payload)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_thin_preflight_writes_authenticated_ready_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        metadata_preflight,
        "OWNED_SOURCE_PATHS",
        ("tests/time_dependent_no/test_pcno_same_state_refresh_metadata.py",),
    )
    source, reference_summary, resolution, normalization = _reference_inputs()
    reference_dir = tmp_path / "reference"
    reference_normalization = reference_dir / "normalization.json"
    _write_json(reference_normalization, normalization)
    normalization_sha = sha256_file(reference_normalization)
    source["base_runtime_source_manifest"]["external_sha256"]["normalization_file"] = (
        normalization_sha
    )
    source_path = reference_dir / "source_manifest.json"
    summary_path = reference_dir / "summary.json"
    resolution_path = reference_dir / "resolution_contract.json"
    _write_json(source_path, source)
    _write_json(summary_path, reference_summary)
    _write_json(resolution_path, resolution)

    opened_path = tmp_path / "opened.json"
    _write_json(opened_path, {"population": {"case_ids": ["sv_open_e00_y00"]}})
    population_path = tmp_path / "population.json"
    _write_json(population_path, _population())

    candidate_dir = tmp_path / "candidate"
    checkpoint_path = candidate_dir / "best.pt"
    candidate_dir.mkdir(parents=True)
    checkpoint_path.write_bytes(b"metadata-only-checkpoint-bytes")
    checkpoint_sha = sha256_file(checkpoint_path)
    candidate_summary = {
        "artifacts": {"best_checkpoint": str(checkpoint_path)},
        "artifact_sha256": {"best_checkpoint": checkpoint_sha},
        "model_config": _model_config(),
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "step_stride": 2,
        "normalization_digest": NORMALIZATION_DIGEST,
        "data_manifest_digest": SHA_D,
        "data_contract": _data_contract(),
        "parent_checkpoint": None,
    }
    candidate_contract = {
        "args": {
            "model_node_type_input": "physical",
            "init_checkpoint": None,
            "resume_checkpoint": None,
        },
        "data_contract": _data_contract(),
    }
    candidate_summary_path = candidate_dir / "summary.json"
    _write_json(candidate_summary_path, candidate_summary)
    _write_json(candidate_dir / "run_contract.json", candidate_contract)
    _write_json(candidate_dir / "normalization.json", normalization)

    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        reference_source_manifest=source_path,
        reference_training_summary=summary_path,
        reference_resolution_contract=resolution_path,
        reference_normalization=reference_normalization,
        opened_source_manifest=[opened_path],
        candidate_summary=[candidate_summary_path],
        population_manifest=population_path,
        selected_checkpoint_sha256=None,
        output_dir=output_dir,
        require_ready=True,
    )
    report, report_path = run_preflight(args)
    assert report["status"] == "ready"
    assert report["selected_checkpoint_sha256"] == checkpoint_sha
    assert report_path.is_file()
    source_manifest = load_json_object(output_dir / "source_manifest.json")
    assert source_manifest["report_sha256"] == sha256_file(report_path)
    assert not any(source_manifest["activity"].values())
