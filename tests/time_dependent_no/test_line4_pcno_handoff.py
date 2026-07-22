from __future__ import annotations

import pytest

from scripts.time_dependent_no.freeze_line4_pcno_handoff import (
    build_handoff,
    canonical_json_digest,
)


def _reports() -> tuple[dict, dict, dict]:
    source_mapping = {"case-a": "reference-a"}
    data_contract = {
        "source_family_id": "family",
        "source_family_manifest_digest": "family-digest",
        "source_artifact_set_digest": canonical_json_digest(source_mapping),
        "data_manifest_digest": "manifest-digest",
        "grouped_split_digest": "split-digest",
        "geometry_contract_digest": "geometry-digest",
        "resolution_contract": {"stored_grid": [4, 2]},
        "resolution_contract_digest": "resolution-digest",
        "time_contract": {"delta_t": 0.01, "saved_calls": 60},
        "time_contract_digest": "time-digest",
        "normalization_digest": "normalization-digest",
        "state_convention": "conservative",
        "weight_provenance": "validated_physical_cell_volume_normalized",
    }
    training = {
        "mode": "pcno_euler2d_conservative_residual",
        "data_contract": data_contract,
        "normalization_digest": "normalization-digest",
        "config_digest": "config-digest",
        "artifact_sha256": {"best_checkpoint": "checkpoint-digest"},
    }
    evaluation = {
        "schema": "evaluation",
        "data_contract": data_contract,
        "checkpoint": {
            "sha256": "checkpoint-digest",
            "config_digest": "config-digest",
            "normalization_digest": "normalization-digest",
            "model_config": {"k_max": 8},
            "parameter_count": 10,
        },
        "evaluation": {
            "split": "validation",
            "num_steps": 60,
            "physical_horizon": 0.6,
            "boundary_mode": "model_all_nodes",
            "raw_recurrence": True,
            "inference_interventions": {"clipping": False},
        },
        "aggregates": {"pcno_baseline": {"completion_rate": 1.0}},
        "endpoint_aggregates": [],
        "grouped_parameter_ood": [],
        "cost": {},
        "reference_contract": {},
    }
    diagnostic = {
        "schema": "diagnostic",
        "checkpoint": {
            "sha256": "checkpoint-digest",
            "config_digest": "config-digest",
        },
        "mechanism_screen": {
            "classification": "unresolved",
            "supported_screens": [],
        },
        "line2_d014_interface": {"status": "consumed"},
    }
    return training, evaluation, diagnostic


def _build(**overrides):
    training, evaluation, diagnostic = _reports()
    arguments = {
        "training_summary": training,
        "checkpoint_name": "best.pt",
        "checkpoint_sha256": "checkpoint-digest",
        "evaluation": evaluation,
        "evaluation_sha256": "evaluation-digest",
        "diagnostic": diagnostic,
        "diagnostic_sha256": "diagnostic-digest",
        "benchmark_audit": {
            "schema": "benchmark",
            "status": "benchmark_contract_closed",
        },
        "benchmark_audit_name": "benchmark.json",
        "benchmark_audit_sha256": "benchmark-digest",
        "family_audit": {
            "schema": "family",
            "status": "passed",
            "family_id": "family",
            "manifest_digest_sha256": "family-digest",
            "requested_case_ids": ["case-a"],
            "passed_case_count": 1,
            "rows": [
                {
                    "case_id": "case-a",
                    "status": "passed",
                    "reference_artifact_sha256": "reference-a",
                }
            ],
            "artifact_set_digest_sha256": canonical_json_digest(
                {
                    "manifest_digest_sha256": "family-digest",
                    "artifacts": [
                        {
                            "case_id": "case-a",
                            "reference_artifact_sha256": "reference-a",
                        }
                    ],
                }
            ),
        },
        "family_audit_name": "family.json",
        "family_audit_sha256": "family-audit-digest",
        "training_truth_authorized": True,
        "training_truth_reason": "independent state agreement and family audit passed",
        "front_candidate_available": False,
        "front_candidate_reason": "D013 did not select a validated front route",
        "front_candidate": None,
    }
    arguments.update(overrides)
    return build_handoff(**arguments)


def test_handoff_records_explicit_flags_without_unblocking_transition() -> None:
    handoff = _build()
    assert handoff["line4_training_truth_authorized"] is True
    assert handoff["line4_front_candidate_available"] is False
    assert handoff["line4_transition_training_authorized"] is False
    assert handoff["training_truth"]["grouped_split_digest"] == "split-digest"
    assert (
        handoff["training_truth"]["family_audit"]["source_mapping_digest"]
        == handoff["training_truth"]["reference_artifact_set_digest"]
    )
    assert handoff["physical_baseline"]["checkpoint_sha256"] == "checkpoint-digest"


def test_front_availability_requires_complete_candidate_contract() -> None:
    with pytest.raises(ValueError, match="candidate summary"):
        _build(front_candidate_available=True)
    with pytest.raises(ValueError, match="incomplete"):
        _build(front_candidate_available=True, front_candidate={"fixed_variables": []})


def test_handoff_rejects_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="training summary"):
        _build(checkpoint_sha256="different")


def test_handoff_rejects_family_source_mapping_mismatch() -> None:
    training, evaluation, diagnostic = _reports()
    evaluation = {**evaluation, "data_contract": dict(evaluation["data_contract"])}
    training = {**training, "data_contract": dict(training["data_contract"])}
    training["data_contract"]["source_artifact_set_digest"] = "different"
    evaluation["data_contract"]["source_artifact_set_digest"] = "different"
    with pytest.raises(ValueError, match="source artifact mappings"):
        _build(training_summary=training, evaluation=evaluation, diagnostic=diagnostic)


def test_handoff_rejects_training_evaluation_data_contract_mismatch() -> None:
    training, evaluation, diagnostic = _reports()
    evaluation = {**evaluation, "data_contract": dict(evaluation["data_contract"])}
    evaluation["data_contract"]["grouped_split_digest"] = "different"
    with pytest.raises(ValueError, match="complete data contract"):
        _build(training_summary=training, evaluation=evaluation, diagnostic=diagnostic)


def test_handoff_rejects_tampered_family_audit_digest() -> None:
    with pytest.raises(ValueError, match="internally inconsistent"):
        _build(
            family_audit={
                "schema": "family",
                "status": "passed",
                "family_id": "family",
                "manifest_digest_sha256": "family-digest",
                "requested_case_ids": ["case-a"],
                "rows": [
                    {
                        "case_id": "case-a",
                        "status": "passed",
                        "reference_artifact_sha256": "reference-a",
                    }
                ],
                "artifact_set_digest_sha256": "tampered",
            }
        )
