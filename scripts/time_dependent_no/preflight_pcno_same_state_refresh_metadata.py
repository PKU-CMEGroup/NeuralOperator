#!/usr/bin/env python3
"""Run the manifest-only W26-L5 A46-A2 compatibility preflight."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_same_state_refresh_metadata import (
    atomic_write_json,
    build_candidate_contract,
    build_metadata_preflight,
    build_reference_contract,
    collect_registered_case_ids,
    load_json_object,
    sha256_file,
    with_payload_sha256,
)

WORKING_ID = "W26-L5-P6-RFB19-A46-A2-METADATA-PREFLIGHT"
SOURCE_MANIFEST_SCHEMA = "pcno_same_state_refresh_metadata_source_manifest_v1"
OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_SAME_STATE_REFRESH_COUNTERFACTUAL_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_same_state_refresh_metadata.py",
    "scripts/time_dependent_no/preflight_pcno_same_state_refresh_metadata.py",
    "tests/time_dependent_no/test_pcno_same_state_refresh_metadata.py",
)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _checkpoint_file(summary: dict[str, Any], summary_path: Path) -> Path | None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    raw_path = artifacts.get("best_checkpoint")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    declared = Path(raw_path)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend((ROOT / declared, summary_path.parent / declared.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _candidate_from_summary(
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    summary = load_json_object(summary_path)
    run_contract_path = summary_path.parent / "run_contract.json"
    normalization_path = summary_path.parent / "normalization.json"
    run_contract = load_json_object(run_contract_path)
    normalization = load_json_object(normalization_path)
    checkpoint_path = _checkpoint_file(summary, summary_path)
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path else None
    checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path else None
    candidate = build_candidate_contract(
        candidate_id=summary_path.parent.name,
        summary=summary,
        run_contract=run_contract,
        normalization=normalization,
        checkpoint_file_sha256=checkpoint_sha,
        checkpoint_file_bytes=checkpoint_bytes,
    )
    input_hashes = {
        _relative(summary_path): sha256_file(summary_path),
        _relative(run_contract_path): sha256_file(run_contract_path),
        _relative(normalization_path): sha256_file(normalization_path),
    }
    if checkpoint_path is not None:
        input_hashes[_relative(checkpoint_path)] = checkpoint_sha or ""
    return candidate, input_hashes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source-manifest", type=Path, required=True)
    parser.add_argument("--reference-training-summary", type=Path, required=True)
    parser.add_argument("--reference-resolution-contract", type=Path, required=True)
    parser.add_argument("--reference-normalization", type=Path, required=True)
    parser.add_argument(
        "--opened-source-manifest",
        action="append",
        type=Path,
        required=True,
        help="Source manifest containing only already-open case inventories.",
    )
    parser.add_argument(
        "--candidate-summary",
        action="append",
        type=Path,
        required=True,
        help="Training summary; sibling run_contract.json and normalization.json are used.",
    )
    parser.add_argument("--population-manifest", type=Path)
    parser.add_argument("--selected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return status 2 unless every metadata gate passes.",
    )
    return parser.parse_args(argv)


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    reference_source = load_json_object(args.reference_source_manifest)
    reference_summary = load_json_object(args.reference_training_summary)
    reference_resolution = load_json_object(args.reference_resolution_contract)
    reference_normalization = load_json_object(args.reference_normalization)
    reference = build_reference_contract(
        source_manifest=reference_source,
        training_summary=reference_summary,
        resolution_run_contract=reference_resolution,
        normalization=reference_normalization,
        normalization_file_sha256=sha256_file(args.reference_normalization),
    )

    opened_manifests = [load_json_object(path) for path in args.opened_source_manifest]
    opened_case_ids = collect_registered_case_ids(*opened_manifests)
    candidates: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    reference_paths = (
        args.reference_source_manifest,
        args.reference_training_summary,
        args.reference_resolution_contract,
        args.reference_normalization,
        *args.opened_source_manifest,
    )
    input_hashes.update(
        {_relative(path): sha256_file(path) for path in reference_paths}
    )
    for summary_path in args.candidate_summary:
        candidate, hashes = _candidate_from_summary(summary_path)
        candidates.append(candidate)
        input_hashes.update(hashes)

    population = None
    if args.population_manifest is not None:
        population = load_json_object(args.population_manifest)
        input_hashes[_relative(args.population_manifest)] = sha256_file(
            args.population_manifest
        )

    report = build_metadata_preflight(
        reference=reference,
        candidates=candidates,
        population=population,
        opened_case_ids=opened_case_ids,
        selected_checkpoint_sha256=args.selected_checkpoint_sha256,
    )
    report = with_payload_sha256(
        {
            **report,
            "working_id": WORKING_ID,
            "input_sha256": dict(sorted(input_hashes.items())),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "stdlib_only": True,
            },
        }
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metadata_preflight.json"
    atomic_write_json(report_path, report)
    source_hashes = {path: sha256_file(ROOT / path) for path in OWNED_SOURCE_PATHS}
    source_manifest = with_payload_sha256(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "working_id": WORKING_ID,
            "status": report["status"],
            "report_sha256": sha256_file(report_path),
            "report_payload_sha256": report["payload_sha256"],
            "source_sha256": source_hashes,
            "input_sha256": dict(sorted(input_hashes.items())),
            "activity": report["activity"],
        }
    )
    atomic_write_json(output_dir / "source_manifest.json", source_manifest)
    return report, report_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, report_path = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "blockers": report["blockers"],
                "eligible_checkpoint_sha256": report["eligible_checkpoint_sha256"],
                "report": _relative(report_path),
                "payload_sha256": report["payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_ready and report["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
