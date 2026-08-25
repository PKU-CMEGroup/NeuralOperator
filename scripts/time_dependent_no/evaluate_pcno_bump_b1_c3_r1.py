#!/usr/bin/env python3
"""Evaluate the D094 B1-C3-R1 seed-0 exact-exposure replay.

Each cold PCNO/PCFNO/count cell is evaluated at optimizer step ``64 * n``.
The scopes and numerics match the closed B1-C3 audit: fixed seen and validation
one-step error, the frozen selection cohort, the seed-0 outside-development
cohort, the common-nine view, and structure diagnostics.  This evaluator never
resumes training, reselects a checkpoint, or opens the historical test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_b1_c2 import (
    ARCHITECTURES,
    EXPECTED_COMMON_OUTSIDE_KEY_DIGEST,
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_SPLIT_PARTITION_DIGEST,
    EXPECTED_VALIDATION_KEY_DIGEST,
    SPLIT_SCHEMA,
    TRAJECTORY_COUNTS,
    _ordered_key_digest,
    _rollout_subset_summary,
    _summary_row,
)
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c3 import (
    CHECKPOINT_ROLE,
    EXPECTED_B1_C2_AUDIT_SUMMARY_SHA256,
    PRESENTATIONS_PER_TRAJECTORY,
    _evaluate_recomputed_scopes,
    _sentinel_descriptor,
    _validate_prior_audit,
    matched_exposure_step,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    ROLLOUT_CHECKPOINTS,
    _build_boundary_policies,
    _evaluate_cell,
    _load_json,
    outside_selection_keys,
)
from scripts.time_dependent_no.train_pcno_bump_scaling import (
    fixed_evaluation_pairs,
    presentation_stream_sha256,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_rollout import (
    FINITE_ONLY_ROLLOUT_POLICY,
    load_bump_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import select_device

SCHEMA = "d094_b1_c3_r1_exact_exposure_audit_v1"
ARTIFACT_SCHEMA = "d094_b1_c3_r1_exact_exposure_artifacts_v1"
SEED = 20_260_718
ATTEMPT = "d094_b1_c3_r1_seed0_replay_20260825a"
EXPECTED_SOURCE_COMMIT = "0da7ad5a66be7bf9ff940ffca777c355bc5341db"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "7198ff6c865bb312ca53cf30d54e0c986759ff6151adf3e000d07b52e565e04b"
)
EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST = (
    "8c2f0930e139834722d039f9ae19aef19896e3761cb64d551a098358870e9664"
)
EXPECTED_CLOSEOUT_MATRIX_RECEIPT_SHA256 = (
    "cc935ce0495d9c153152ca280bbd650d47b4589cd7ddaf3c1afd6d03e05756b9"
)
EXPECTED_CLOSEOUT_ARTIFACT_MANIFEST_SHA256 = (
    "e74281466fba618bd0c472a5261eba3a34b8df260bb063467eebeb42144951360"
)
CLOSEOUT_RECEIPT_SCHEMA = "d094_b1_c3_r1_seed0_replay_matrix_receipt_v2"
CLOSEOUT_ARTIFACT_SCHEMA = "d094_b1_c3_r1_closeout_artifacts_v1"
REGISTERED_STAGE = "b1_c3_r1_seed0_replay_20480"
SCHEDULE = "b1_c3_r1_seed0_replay_stretched"

EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/closeout_pcno_bump_b1_c3_r1.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c2.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c3.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c3_r1.py",
    "scripts/time_dependent_no/train_pcno_bump_scaling.py",
    "utility/time_dependent_no/pcno_differential_branch.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--closeout-root", type=Path, required=True)
    parser.add_argument("--b1-c2-audit-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=(
            REPO_ROOT / "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), required=True)
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    return parser


def _validate_closeout(closeout_root: Path) -> dict[str, Any]:
    if closeout_root.is_symlink() or not closeout_root.is_dir():
        raise ValueError("B1-C3-R1 closeout root must be a regular directory")
    receipt_path = closeout_root / "matrix_receipt.json"
    manifest_path = closeout_root / "artifact_manifest.json"
    if (
        sha256_file(receipt_path) != EXPECTED_CLOSEOUT_MATRIX_RECEIPT_SHA256
        or sha256_file(manifest_path)
        != EXPECTED_CLOSEOUT_ARTIFACT_MANIFEST_SHA256
    ):
        raise ValueError("B1-C3-R1 closeout differs from the registered correction")
    receipt = _load_json(receipt_path)
    manifest = _load_json(manifest_path)
    if (
        receipt.get("schema") != CLOSEOUT_RECEIPT_SCHEMA
        or receipt.get("status") != "complete_after_closeout_correction"
        or receipt.get("attempt") != ATTEMPT
        or receipt.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or receipt.get("source_archive_sha256")
        != EXPECTED_SOURCE_ARCHIVE_SHA256
        or receipt.get("historical_test_population_accessed") is not False
        or receipt.get("checkpoint_reselection_performed") is not False
        or receipt.get("seed") != SEED
        or receipt.get("trajectory_counts") != list(TRAJECTORY_COUNTS)
        or receipt.get("architectures") != list(ARCHITECTURES)
        or len(receipt.get("cells", [])) != 12
    ):
        raise ValueError("B1-C3-R1 closeout receipt contract changed")
    if manifest.get("schema") != CLOSEOUT_ARTIFACT_SCHEMA:
        raise ValueError("B1-C3-R1 closeout artifact schema changed")
    record = manifest.get("files", {}).get("matrix_receipt.json")
    if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(
        receipt_path
    ):
        raise ValueError("closeout artifact manifest does not bind its receipt")
    correction = receipt.get("closeout_correction")
    if (
        not isinstance(correction, Mapping)
        or correction.get("original_attempt_outputs_modified") is not False
        or correction.get("recorded_initial_state_exactly_reconstructed_for_every_count")
        is not True
        or correction.get("parameter_only_hash_cardinality_across_counts") != 1
        or correction.get(
            "nondifferential_parameter_only_hash_cardinality_across_counts"
        )
        != 1
    ):
        raise ValueError("B1-C3-R1 initialization correction changed")
    return receipt


def _receipt_file_records(
    receipt: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    expected_cells = {
        (
            count,
            architecture,
            (
                f"b1_c3_r1_s{SEED}_n{count}_{architecture}_"
                "stretched_20480_20260825a"
            ),
        )
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    }
    observed_cells = set()
    for cell in receipt["cells"]:
        count = int(cell["trajectory_count"])
        architecture = str(cell["architecture"])
        name = str(cell["cell"])
        observed_cells.add((count, architecture, name))
        files = cell.get("files")
        if not isinstance(files, Mapping):
            raise TypeError(f"closeout file inventory is malformed: {name}")
        for relative, record in files.items():
            key = f"results/{name}/{relative}"
            if key in records or not isinstance(record, Mapping):
                raise ValueError(f"duplicate or malformed closeout file: {key}")
            records[key] = record
    if observed_cells != expected_cells:
        raise ValueError("B1-C3-R1 closeout cell inventory changed")
    return records


def discover_sentinels(
    attempt_root: Path, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("B1-C3-R1 attempt root must be a regular directory")
    records = _receipt_file_records(receipt)
    cells = {
        (int(cell["trajectory_count"]), str(cell["architecture"])): cell
        for cell in receipt["cells"]
    }
    descriptors = []
    for count in TRAJECTORY_COUNTS:
        for architecture in ARCHITECTURES:
            cell = cells[(count, architecture)]
            run_dir = attempt_root / "results" / str(cell["cell"])
            summary = _load_json(run_dir / "summary.json")
            contract = _load_json(run_dir / "bump_scaling_contract.json")
            split = _load_json(run_dir / "split.json")
            expected_mode = "full" if architecture == "pcno" else "no_gradient"
            if (
                contract.get("registered_stage") != REGISTERED_STAGE
                or contract.get("initialization_seed") != SEED
                or contract.get("trajectory_count") != count
                or contract.get("differential_branch_mode") != expected_mode
                or contract.get("actual_optimizer_steps") != 20_480
                or cell.get("sentinel_step") != matched_exposure_step(count)
                or cell.get("config_digest") != summary.get("config_digest")
                or cell.get("normalization_digest")
                != summary.get("normalization_digest")
            ):
                raise ValueError(
                    f"B1-C3-R1 cell contract changed: n={count} {architecture}"
                )
            base = {
                "seed": SEED,
                "trajectory_count": count,
                "architecture": architecture,
                "run": str(cell["cell"]),
                "run_dir": run_dir,
                "schedule": SCHEDULE,
                "split": split,
                "summary": summary,
                "contract": contract,
            }
            descriptors.append(
                _sentinel_descriptor(
                    base,
                    replication_root=attempt_root,
                    file_records=records,
                    expected_source_set_digest=(
                        EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
                    ),
                )
            )
    if len(descriptors) != 12:
        raise RuntimeError("B1-C3-R1 must expose exactly 12 exact sentinels")
    return descriptors


def _seed0_cohorts(
    split_manifest: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    prior_audit: Mapping[str, Any],
) -> dict[str, Any]:
    validation, selection, outside = outside_selection_keys(
        split_manifest, [descriptor["split"] for descriptor in descriptors]
    )
    if _ordered_key_digest(validation) != EXPECTED_VALIDATION_KEY_DIGEST:
        raise ValueError("registered validation cohort changed")
    seed_key = str(SEED)
    expected_selection = [
        str(key) for key in prior_audit["selection_keys_by_seed"][seed_key]
    ]
    expected_outside = [
        str(key) for key in prior_audit["outside_selection_keys_by_seed"][seed_key]
    ]
    common = [str(key) for key in prior_audit["common_outside_selection_keys"]]
    if selection != expected_selection or outside != expected_outside:
        raise ValueError("seed-0 selection/outside-development cohort changed")
    if (
        len(common) != 9
        or _ordered_key_digest(common) != EXPECTED_COMMON_OUTSIDE_KEY_DIGEST
        or not set(common) <= set(outside)
    ):
        raise ValueError("registered common-nine cohort changed")
    return {
        "validation_keys": validation,
        "selection_keys": selection,
        "outside_selection_keys": outside,
        "common_outside_selection_keys": common,
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C3-R1 output directory must not already exist")
    split_manifest = _load_json(args.split_manifest)
    if (
        split_manifest.get("schema") != SPLIT_SCHEMA
        or split_manifest.get("partition_digest")
        != EXPECTED_SPLIT_PARTITION_DIGEST
    ):
        raise ValueError("D094 split manifest changed")

    receipt = _validate_closeout(args.closeout_root)
    prior_audit = _validate_prior_audit(args.b1_c2_audit_root)
    descriptors = discover_sentinels(args.attempt_root, receipt)
    cohorts = _seed0_cohorts(split_manifest, descriptors, prior_audit)
    validation_keys = cohorts["validation_keys"]
    selection_keys = cohorts["selection_keys"]
    outside_keys = cohorts["outside_selection_keys"]
    common_keys = cohorts["common_outside_selection_keys"]

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if store.manifest_digest != EXPECTED_DATA_MANIFEST_DIGEST:
            raise ValueError("B1-C3-R1 evaluation data manifest changed")
        if not set(validation_keys) <= set(store.keys):
            raise ValueError("open-validation keys are absent from the shard store")
        reference = load_bump_checkpoint(Path(descriptors[0]["checkpoint"]))
        policies, policy_metadata = _build_boundary_policies(
            store, validation_keys, reference, device
        )
        step_stride = int(reference["step_stride"])
        del reference
        fixed_validation_pairs = fixed_evaluation_pairs(
            store,
            validation_keys,
            step_stride=step_stride,
            windows_per_trajectory=4,
        )
        fixed_seen_pairs_by_count = {
            count: fixed_evaluation_pairs(
                store,
                split_manifest["nested_exposure"]["subsets"][str(count)],
                step_stride=step_stride,
                windows_per_trajectory=4,
            )
            for count in TRAJECTORY_COUNTS
        }

        results = []
        for descriptor in descriptors:
            count = int(descriptor["trajectory_count"])
            result = _evaluate_cell(
                descriptor,
                store=store,
                outside_keys=outside_keys,
                policies=policies,
                policy_metadata=policy_metadata,
                expected_checkpoint_source_set_digest=(
                    EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
                ),
                device=device,
                amp=args.amp,
                shock_quantile=args.shock_quantile,
            )
            result.pop("selected_training_metrics")
            result.update(
                _evaluate_recomputed_scopes(
                    descriptor,
                    store=store,
                    fixed_seen_pairs=fixed_seen_pairs_by_count[count],
                    fixed_validation_pairs=fixed_validation_pairs,
                    selection_keys=selection_keys,
                    policies=policies,
                    device=device,
                    amp=args.amp,
                )
            )
            for name in (
                "seed",
                "checkpoint_role",
                "optimizer_step",
                "presentations_per_training_trajectory",
                "config_digest",
                "normalization_digest",
            ):
                result[name] = descriptor[name]
            result["checkpoint_source_set_digest"] = (
                EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
            )
            result["audit_cohort"] = {
                "scope": "seed0_outside_checkpoint_selection",
                "seed": SEED,
                "selection_count": len(selection_keys),
                "outside_selection_count": len(outside_keys),
                "outside_selection_keys": outside_keys,
            }
            result["common_outside_selection_rollout"] = _rollout_subset_summary(
                result["outside_selection_rollout"], common_keys
            )
            results.append(result)
            filename = (
                f"s{SEED}_n{count:03d}_{descriptor['architecture']}_"
                f"{CHECKPOINT_ROLE}.json"
            )
            atomic_write_json(args.output_dir / filename, result)
            print(
                json.dumps(
                    {
                        "amp": args.amp,
                        "architecture": descriptor["architecture"],
                        "completed_checkpoints": len(results),
                        "elapsed_seconds": result["elapsed_seconds"],
                        "seed": SEED,
                        "trajectory_count": count,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
    finally:
        store.close()

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": "seed0_exact_64_presentations_per_trajectory_sentinel_audit",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "training_or_resume_performed": False,
        "matrix_receipt_sha256": EXPECTED_CLOSEOUT_MATRIX_RECEIPT_SHA256,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "prior_b1_c2_audit_summary_sha256": (
            EXPECTED_B1_C2_AUDIT_SUMMARY_SHA256
        ),
        "checkpoint_source_set_digest": EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        "evaluator_source_snapshot": source_snapshot,
        "seeds": [SEED],
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "architectures": list(ARCHITECTURES),
        "checkpoint_role": CHECKPOINT_ROLE,
        "optimizer_step_by_trajectory_count": {
            str(count): matched_exposure_step(count)
            for count in TRAJECTORY_COUNTS
        },
        "presentations_per_training_trajectory": PRESENTATIONS_PER_TRAJECTORY,
        "fixed_evaluation_windows_per_trajectory": 4,
        "fixed_validation_pair_bank_sha256": presentation_stream_sha256(
            fixed_validation_pairs
        ),
        "fixed_seen_pair_bank_sha256_by_count": {
            str(count): presentation_stream_sha256(pairs)
            for count, pairs in fixed_seen_pairs_by_count.items()
        },
        "validation_count": len(validation_keys),
        "selection_keys_by_seed": {str(SEED): selection_keys},
        "outside_selection_keys_by_seed": {str(SEED): outside_keys},
        "common_outside_selection_keys": common_keys,
        "common_outside_selection_count": len(common_keys),
        "rollout_horizon": 79,
        "rollout_checkpoints": list(ROLLOUT_CHECKPOINTS),
        "rollout_failure_policy": FINITE_ONLY_ROLLOUT_POLICY,
        "shock_quantile": args.shock_quantile,
        "evaluation_numerics": {
            "amp": args.amp,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "boundary_policy_digests": {
            key: record["policy_digest"] for key, record in policy_metadata.items()
        },
        "cells": results,
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "three-seed conclusions without the separately bound combined analysis",
            "fixed optimizer updates, compute, or learning-rate phase",
            "checkpoint selection on outside-development trajectories",
            "independent test performance",
            "causal attribution to data, compute, optimizer, capacity, or gradient",
            "a paper-faithful FFNO comparison",
            "physical conservation from reconstructed proxy weights",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_csv(
        args.output_dir / "checkpoint_summary.csv",
        [_summary_row(result) for result in results],
    )
    cell_files = [
        f"s{SEED}_n{count:03d}_{architecture}_{CHECKPOINT_ROLE}.json"
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    ]
    artifact_files = ["summary.json", "checkpoint_summary.csv", *cell_files]
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": int((args.output_dir / name).stat().st_size),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_files
        },
    }
    atomic_write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(argv)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cell_count": len(summary["cells"]),
                "amp": summary["evaluation_numerics"]["amp"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
