#!/usr/bin/env python3
"""Freeze the explicit Line-3 to Line-4 physical-baseline handoff.

This script performs no model call or training.  It binds already-frozen
reference, checkpoint, rollout, and D013 reports and requires both Line-4 flags
to be chosen explicitly.  A front candidate cannot be declared without its
full remapping and intervention contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "line3_to_line4_physical_baseline_handoff_v1"
FRONT_REQUIRED_FIELDS = {
    "fixed_variables",
    "learned_variables",
    "front_topology_convention",
    "conservative_remap",
    "geometry_resolution_contract",
    "encoder_digest",
    "decoder_digest",
    "valid_lengths",
    "inference_interventions",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument("--benchmark-audit", type=Path, required=True)
    parser.add_argument("--family-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--line4-training-truth-authorized", type=parse_bool, required=True
    )
    parser.add_argument("--training-truth-reason", required=True)
    parser.add_argument(
        "--line4-front-candidate-available", type=parse_bool, required=True
    )
    parser.add_argument("--front-candidate-reason", required=True)
    parser.add_argument("--front-candidate-summary", type=Path)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_equal(name: str, values: Sequence[Any]) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"handoff reports disagree on {name}")
    return first


def _audit_passed(audit: Mapping[str, Any], *, benchmark: bool) -> bool:
    status = str(audit.get("status"))
    allowed = (
        {"benchmark_contract_closed"}
        if benchmark
        else {"passed", "complete", "complete_and_valid"}
    )
    if status not in allowed:
        return False
    checks = audit.get("checks", audit.get("contract_checks"))
    return not isinstance(checks, Mapping) or all(
        bool(value) for value in checks.values()
    )


def _validate_family_audit_linkage(
    data_contract: Mapping[str, Any], family_audit: Mapping[str, Any]
) -> dict[str, str]:
    if family_audit.get("family_id") != data_contract.get("source_family_id"):
        raise ValueError("family audit and data contract family ids differ")
    manifest_digest = family_audit.get("manifest_digest_sha256")
    if manifest_digest != data_contract.get("source_family_manifest_digest"):
        raise ValueError("family audit and data contract manifest digests differ")

    requested = family_audit.get("requested_case_ids")
    rows = family_audit.get("rows")
    if not isinstance(requested, list) or not isinstance(rows, list):
        raise ValueError("family audit lacks its ordered case and row contracts")
    if len(requested) != len(set(map(str, requested))) or len(rows) != len(requested):
        raise ValueError("family audit case rows are missing or duplicated")
    row_ids = [str(row.get("case_id")) for row in rows if isinstance(row, Mapping)]
    if row_ids != list(map(str, requested)):
        raise ValueError("family audit row order differs from its requested cases")
    if any(row.get("status") != "passed" for row in rows):
        raise ValueError("family audit contains a non-passing case row")

    source_mapping = {
        str(row["case_id"]): row.get("reference_artifact_sha256") for row in rows
    }
    if any(
        not isinstance(value, str) or not value for value in source_mapping.values()
    ):
        raise ValueError("family audit lacks a source artifact digest")
    source_mapping_digest = canonical_json_digest(source_mapping)
    if source_mapping_digest != data_contract.get("source_artifact_set_digest"):
        raise ValueError(
            "family audit and data contract source artifact mappings differ"
        )

    audit_artifact_set_digest = family_audit.get(
        "artifact_set_digest_sha256", family_audit.get("artifact_set_digest")
    )
    expected_audit_digest = canonical_json_digest(
        {
            "manifest_digest_sha256": manifest_digest,
            "artifacts": [
                {
                    "case_id": str(row["case_id"]),
                    "reference_artifact_sha256": row["reference_artifact_sha256"],
                }
                for row in rows
            ],
        }
    )
    if audit_artifact_set_digest != expected_audit_digest:
        raise ValueError("family audit artifact-set digest is internally inconsistent")
    return {
        "source_mapping_digest": source_mapping_digest,
        "artifact_set_digest": expected_audit_digest,
    }


def build_handoff(
    *,
    training_summary: Mapping[str, Any],
    checkpoint_name: str,
    checkpoint_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    diagnostic: Mapping[str, Any],
    diagnostic_sha256: str,
    benchmark_audit: Mapping[str, Any],
    benchmark_audit_name: str,
    benchmark_audit_sha256: str,
    family_audit: Mapping[str, Any],
    family_audit_name: str,
    family_audit_sha256: str,
    training_truth_authorized: bool,
    training_truth_reason: str,
    front_candidate_available: bool,
    front_candidate_reason: str,
    front_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data_contracts = [
        training_summary.get("data_contract"),
        evaluation.get("data_contract"),
    ]
    if any(not isinstance(value, Mapping) for value in data_contracts):
        raise ValueError("training and evaluation reports require data contracts")
    data_contract = dict(data_contracts[0])
    _require_equal(
        "complete data contract",
        [canonical_json_digest(dict(value)) for value in data_contracts],
    )
    _require_equal(
        "data manifest digest",
        [value.get("data_manifest_digest") for value in data_contracts],
    )
    _require_equal(
        "normalization digest",
        [
            training_summary.get("normalization_digest"),
            evaluation.get("checkpoint", {}).get("normalization_digest"),
            data_contract.get("normalization_digest"),
        ],
    )
    family_linkage = _validate_family_audit_linkage(data_contract, family_audit)
    training_checkpoint_sha = training_summary.get("artifact_sha256", {}).get(
        "best_checkpoint"
    )
    if training_checkpoint_sha != checkpoint_sha256:
        raise ValueError("selected checkpoint does not match the training summary")
    evaluation_checkpoint = evaluation.get("checkpoint", {})
    diagnostic_checkpoint = diagnostic.get("checkpoint", {})
    _require_equal(
        "selected checkpoint digest",
        [
            checkpoint_sha256,
            evaluation_checkpoint.get("sha256"),
            diagnostic_checkpoint.get("sha256"),
        ],
    )
    _require_equal(
        "selected configuration digest",
        [
            training_summary.get("config_digest"),
            evaluation_checkpoint.get("config_digest"),
            diagnostic_checkpoint.get("config_digest"),
        ],
    )
    if training_truth_authorized and not (
        _audit_passed(benchmark_audit, benchmark=True)
        and _audit_passed(family_audit, benchmark=False)
    ):
        raise ValueError("training truth cannot be authorized from failing audits")
    if front_candidate_available:
        if front_candidate is None:
            raise ValueError(
                "front candidate availability requires a candidate summary"
            )
        missing = sorted(FRONT_REQUIRED_FIELDS - set(front_candidate))
        if missing:
            raise ValueError(f"front candidate summary is incomplete: {missing}")
    elif front_candidate is not None:
        raise ValueError("front candidate summary supplied while availability is false")
    if not training_truth_reason.strip() or not front_candidate_reason.strip():
        raise ValueError("both authorization reasons must be nonempty")

    mechanism = diagnostic.get("mechanism_screen", {})
    evaluation_contract = evaluation.get("evaluation", {})
    return {
        "schema": SCHEMA,
        "status": "frozen",
        "line4_training_truth_authorized": bool(training_truth_authorized),
        "line4_front_candidate_available": bool(front_candidate_available),
        "line4_transition_training_authorized": False,
        "training_truth": {
            "reason": training_truth_reason,
            "reference_family_id": data_contract.get("source_family_id"),
            "reference_family_manifest_digest": data_contract.get(
                "source_family_manifest_digest"
            ),
            "reference_artifact_set_digest": data_contract.get(
                "source_artifact_set_digest"
            ),
            "data_manifest_digest": data_contract.get("data_manifest_digest"),
            "grouped_split_digest": data_contract.get("grouped_split_digest"),
            "geometry_contract_digest": data_contract.get("geometry_contract_digest"),
            "resolution_contract": data_contract.get("resolution_contract"),
            "resolution_contract_digest": data_contract.get(
                "resolution_contract_digest"
            ),
            "time_contract": data_contract.get("time_contract"),
            "time_contract_digest": data_contract.get("time_contract_digest"),
            "normalization_digest": data_contract.get("normalization_digest"),
            "state_convention": data_contract.get("state_convention"),
            "weight_provenance": data_contract.get("weight_provenance"),
            "benchmark_audit": {
                "path_name": benchmark_audit_name,
                "sha256": benchmark_audit_sha256,
                "schema": benchmark_audit.get("schema"),
                "status": benchmark_audit.get("status"),
            },
            "family_audit": {
                "path_name": family_audit_name,
                "sha256": family_audit_sha256,
                "schema": family_audit.get("schema"),
                "status": family_audit.get("status"),
                "artifact_set_digest": family_linkage["artifact_set_digest"],
                "source_mapping_digest": family_linkage["source_mapping_digest"],
            },
        },
        "physical_baseline": {
            "checkpoint_name": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "config_digest": evaluation_checkpoint.get("config_digest"),
            "model_config": evaluation_checkpoint.get("model_config"),
            "parameter_count": evaluation_checkpoint.get("parameter_count"),
            "checkpoint_epoch": evaluation_checkpoint.get("epoch"),
            "selection_rule_state": evaluation_checkpoint.get("selection_rule_state"),
            "evaluation_report_sha256": evaluation_sha256,
            "split": evaluation_contract.get("split"),
            "calls": evaluation_contract.get("num_steps"),
            "physical_horizon": evaluation_contract.get("physical_horizon"),
            "boundary_mode": evaluation_contract.get("boundary_mode"),
            "raw_recurrence": evaluation_contract.get("raw_recurrence"),
            "inference_interventions": evaluation_contract.get(
                "inference_interventions"
            ),
            "aggregates": evaluation.get("aggregates"),
            "front_and_smooth_hierarchy": evaluation.get("endpoint_aggregates"),
            "grouped_parameter_ood": evaluation.get("grouped_parameter_ood"),
            "cost": evaluation.get("cost"),
            "reference_contract": evaluation.get("reference_contract"),
        },
        "d013": {
            "report_sha256": diagnostic_sha256,
            "classification": mechanism.get("classification"),
            "supported_screens": mechanism.get("supported_screens"),
            "evidence": mechanism.get("evidence"),
            "selected_first_branch": mechanism.get("selected_first_branch"),
            "falsification": mechanism.get("falsification"),
            "line2_d014_interface": diagnostic.get("line2_d014_interface"),
        },
        "front_candidate": (
            {"reason": front_candidate_reason, **dict(front_candidate)}
            if front_candidate_available
            else {"reason": front_candidate_reason, "available": False}
        ),
        "dependencies": {
            "line4_reconstruction_and_closure_tests_may_reopen": bool(
                training_truth_authorized
            ),
            "line4_transition_training_still_requires_representation_gates": True,
            "line4_assimilation_remains_blocked_by_raw_open_loop_gate": True,
        },
        "source_reports": {
            "training_summary_schema": training_summary.get("mode"),
            "evaluation_schema": evaluation.get("schema"),
            "diagnostic_schema": diagnostic.get("schema"),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen handoff: {args.output}")
    paths = (
        args.training_summary,
        args.checkpoint,
        args.evaluation_summary,
        args.diagnostic_summary,
        args.benchmark_audit,
        args.family_audit,
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more required frozen inputs are absent")
    front_candidate = (
        None
        if args.front_candidate_summary is None
        else load_json(args.front_candidate_summary)
    )
    handoff = build_handoff(
        training_summary=load_json(args.training_summary),
        checkpoint_name=args.checkpoint.name,
        checkpoint_sha256=sha256_file(args.checkpoint),
        evaluation=load_json(args.evaluation_summary),
        evaluation_sha256=sha256_file(args.evaluation_summary),
        diagnostic=load_json(args.diagnostic_summary),
        diagnostic_sha256=sha256_file(args.diagnostic_summary),
        benchmark_audit=load_json(args.benchmark_audit),
        benchmark_audit_name=args.benchmark_audit.name,
        benchmark_audit_sha256=sha256_file(args.benchmark_audit),
        family_audit=load_json(args.family_audit),
        family_audit_name=args.family_audit.name,
        family_audit_sha256=sha256_file(args.family_audit),
        training_truth_authorized=args.line4_training_truth_authorized,
        training_truth_reason=args.training_truth_reason,
        front_candidate_available=args.line4_front_candidate_available,
        front_candidate_reason=args.front_candidate_reason,
        front_candidate=front_candidate,
    )
    atomic_write_json(args.output, handoff)
    print(
        json.dumps(
            {
                "handoff": str(args.output),
                "flags": {
                    "line4_training_truth_authorized": handoff[
                        "line4_training_truth_authorized"
                    ],
                    "line4_front_candidate_available": handoff[
                        "line4_front_candidate_available"
                    ],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
