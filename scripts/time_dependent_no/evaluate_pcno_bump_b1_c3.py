#!/usr/bin/env python3
"""Evaluate the frozen D094 B1-C2 sentinels at matched data exposure.

For both B1-C2 replication seeds, each PCNO/PCFNO/count cell is evaluated at
optimizer step ``64 * n``.  The balanced training stream makes this exactly 64
presentations per training trajectory.  The evaluator uses the same seed-local
outside-selection cohorts and fixed common-nine view as the closed B1-C2 audit.
It never resumes training or selects a checkpoint.
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
    ARTIFACT_SCHEMA as B1_C2_ARTIFACT_SCHEMA,
    EXPECTED_COMMON_OUTSIDE_KEY_DIGEST,
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
    EXPECTED_SPLIT_PARTITION_DIGEST,
    EXPECTED_VALIDATION_KEY_DIGEST,
    REPLICATION_SEEDS,
    SCHEMA as B1_C2_SCHEMA,
    SPLIT_SCHEMA,
    TRAJECTORY_COUNTS,
    _ordered_key_digest,
    _replication_descriptors,
    _rollout_subset_summary,
    _summary_row,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    ROLLOUT_CHECKPOINTS,
    _build_boundary_policies,
    _evaluate_cell,
    _load_json,
    _load_jsonl,
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
    build_bump_checkpoint_model,
    evaluate_pairs,
    evaluate_rollouts,
    load_bump_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import select_device

SCHEMA = "d094_b1_c3_exact_exposure_audit_v1"
ARTIFACT_SCHEMA = "d094_b1_c3_exact_exposure_artifacts_v1"
CHECKPOINT_ROLE = "matched_exposure"
SENTINEL_SCHEMA = "d094_model_only_sentinel_v1"
STEPS_PER_EPOCH = 256
PRESENTATIONS_PER_TRAJECTORY = 64

EXPECTED_B1_C2_AUDIT_SUMMARY_SHA256 = (
    "1a9601907ebd9d84c49db61cf1717fd26c309aeed80dc1fd984d463a506f033e"
)
EXPECTED_B1_C2_AUDIT_ARTIFACT_MANIFEST_SHA256 = (
    "0645155a3d09e4eac642568f6491a542a51089252a7ffb0f37408855c96eef90"
)

EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c2.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c3.py",
    "scripts/time_dependent_no/train_pcno_bump_scaling.py",
    "utility/time_dependent_no/pcno_differential_branch.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", type=Path, required=True)
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


def matched_exposure_step(trajectory_count: int) -> int:
    if trajectory_count not in TRAJECTORY_COUNTS:
        raise ValueError(f"unregistered trajectory count: {trajectory_count}")
    return PRESENTATIONS_PER_TRAJECTORY * trajectory_count


def _validate_prior_audit(audit_root: Path) -> dict[str, Any]:
    if audit_root.is_symlink() or not audit_root.is_dir():
        raise ValueError("B1-C2 audit root must be a regular directory")
    summary_path = audit_root / "summary.json"
    manifest_path = audit_root / "artifact_manifest.json"
    if (
        sha256_file(summary_path) != EXPECTED_B1_C2_AUDIT_SUMMARY_SHA256
        or sha256_file(manifest_path)
        != EXPECTED_B1_C2_AUDIT_ARTIFACT_MANIFEST_SHA256
    ):
        raise ValueError("B1-C2 audit root differs from the registered closeout")
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if (
        manifest.get("schema") != B1_C2_ARTIFACT_SCHEMA
        or summary.get("schema") != B1_C2_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("historical_test_population_accessed") is not False
        or summary.get("checkpoint_reselection_on_outside_cases") is not False
        or summary.get("seeds")
        != [20_260_718, *REPLICATION_SEEDS]
        or summary.get("trajectory_counts") != list(TRAJECTORY_COUNTS)
        or summary.get("architectures") != list(ARCHITECTURES)
    ):
        raise ValueError("B1-C2 audit contract changed")
    summary_record = manifest.get("files", {}).get("summary.json")
    if not isinstance(summary_record, Mapping) or summary_record.get(
        "sha256"
    ) != sha256_file(summary_path):
        raise ValueError("B1-C2 artifact manifest does not bind its summary")
    return summary


def _audit_cohorts(
    split_manifest: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    prior_audit: Mapping[str, Any],
) -> dict[str, Any]:
    validation_keys = [
        str(key) for key in split_manifest["split"]["open_validation_keys"]
    ]
    if _ordered_key_digest(validation_keys) != EXPECTED_VALIDATION_KEY_DIGEST:
        raise ValueError("registered validation cohort changed")

    selection_by_seed: dict[str, list[str]] = {}
    outside_by_seed: dict[str, list[str]] = {}
    for seed in REPLICATION_SEEDS:
        splits = [
            descriptor["split"]
            for descriptor in descriptors
            if int(descriptor["seed"]) == seed
        ]
        validation, selection, outside = outside_selection_keys(split_manifest, splits)
        if validation != validation_keys:
            raise ValueError(f"validation order changed for seed={seed}")
        seed_key = str(seed)
        expected_selection = [
            str(key) for key in prior_audit["selection_keys_by_seed"][seed_key]
        ]
        expected_outside = [
            str(key)
            for key in prior_audit["outside_selection_keys_by_seed"][seed_key]
        ]
        if selection != expected_selection or outside != expected_outside:
            raise ValueError(f"B1-C2 audit cohort changed for seed={seed}")
        selection_by_seed[seed_key] = selection
        outside_by_seed[seed_key] = outside

    common_outside = [
        str(key) for key in prior_audit["common_outside_selection_keys"]
    ]
    if (
        len(common_outside) != 9
        or _ordered_key_digest(common_outside)
        != EXPECTED_COMMON_OUTSIDE_KEY_DIGEST
        or any(
            not set(common_outside) <= set(outside_by_seed[str(seed)])
            for seed in REPLICATION_SEEDS
        )
    ):
        raise ValueError("registered common-nine cohort changed")
    return {
        "validation_keys": validation_keys,
        "selection_keys_by_seed": selection_by_seed,
        "outside_selection_keys_by_seed": outside_by_seed,
        "common_outside_selection_keys": common_outside,
    }


def _matrix_file_records(replication_root: Path) -> dict[str, Mapping[str, Any]]:
    receipt = _load_json(replication_root / "matrix_receipt.json")
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("B1-C2 matrix receipt lacks its file inventory")
    records: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise TypeError("B1-C2 matrix file inventory is malformed")
        name = str(row["path"])
        if name in records:
            raise ValueError(f"duplicate B1-C2 matrix file: {name}")
        records[name] = row
    return records


def _metric_at_step(run_dir: Path, optimizer_step: int) -> dict[str, Any]:
    matches = []
    for row in _load_jsonl(run_dir / "metrics.jsonl"):
        train = row.get("train")
        if isinstance(train, Mapping) and int(
            train.get("completed_optimizer_steps", -1)
        ) == optimizer_step:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"{run_dir.name} does not have one exact metric row at step "
            f"{optimizer_step}"
        )
    row = matches[0]
    train = row.get("train")
    validation = row.get("validation")
    if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
        raise TypeError("exact sentinel history row lacks train or validation metrics")
    result = {
        "epoch": int(row["epoch"]),
        "optimizer_step": int(train["completed_optimizer_steps"]),
        "online_train_one_step_relative_l2": float(train["relative_l2"]),
        "stored_fixed_validation_one_step_relative_l2": float(
            validation["relative_l2"]
        ),
    }
    if result["optimizer_step"] != optimizer_step:
        raise ValueError("exact sentinel history row changed optimizer step")
    return result


def _evaluate_recomputed_scopes(
    descriptor: Mapping[str, Any],
    *,
    store: PCNOEuler2DShardStore,
    fixed_seen_pairs: Sequence[tuple[str, int]],
    fixed_validation_pairs: Sequence[tuple[str, int]],
    selection_keys: Sequence[str],
    policies: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    """Recompute scopes omitted by the sentinel epoch's logging cadence."""

    seen_digest = presentation_stream_sha256(fixed_seen_pairs)
    validation_digest = presentation_stream_sha256(fixed_validation_pairs)
    contract = descriptor["contract"]
    if (
        seen_digest != contract["fixed_seen_pair_bank_sha256"]
        or validation_digest != contract["fixed_validation_pair_bank_sha256"]
    ):
        raise ValueError("fixed one-step pair bank changed")

    checkpoint = load_bump_checkpoint(Path(descriptor["checkpoint"]))
    model = build_bump_checkpoint_model(checkpoint, device)
    step_stride = int(checkpoint["step_stride"])
    train_keys = [str(key) for key in descriptor["contract"]["train_keys"]]
    train_policies, train_policy_metadata = _build_boundary_policies(
        store, train_keys, checkpoint, device
    )
    cell_policies = dict(policies)
    if set(cell_policies) & set(train_policies):
        raise ValueError("fixed-seen and validation policy populations overlap")
    cell_policies.update(train_policies)
    try:
        fixed_seen = evaluate_pairs(
            model,
            store,
            fixed_seen_pairs,
            step_stride=step_stride,
            batch_size=1,
            device=device,
            amp=amp,
            boundary_policies=cell_policies,
        )
        fixed_seen["pair_bank_sha256"] = seen_digest
        fixed_validation = evaluate_pairs(
            model,
            store,
            fixed_validation_pairs,
            step_stride=step_stride,
            batch_size=1,
            device=device,
            amp=amp,
            boundary_policies=cell_policies,
        )
        fixed_validation["pair_bank_sha256"] = validation_digest
        selection_rollout = evaluate_rollouts(
            model,
            store,
            selection_keys,
            step_stride=step_stride,
            start_frame=0,
            num_steps=79,
            device=device,
            amp=amp,
            boundary_policies=cell_policies,
            rollout_checkpoints=ROLLOUT_CHECKPOINTS,
            failure_policy=FINITE_ONLY_ROLLOUT_POLICY,
        )
    finally:
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    endpoints = selection_rollout["mean_endpoint_relative_l2"]
    history = descriptor["selected_training_metrics"]
    metrics = {
        "epoch": int(history["epoch"]),
        "optimizer_step": int(history["optimizer_step"]),
        "online_train_one_step_relative_l2": float(
            history["online_train_one_step_relative_l2"]
        ),
        "fixed_seen_train_one_step_relative_l2": float(fixed_seen["relative_l2"]),
        "fixed_validation_one_step_relative_l2": float(
            fixed_validation["relative_l2"]
        ),
        "stored_fixed_validation_one_step_relative_l2": float(
            history["stored_fixed_validation_one_step_relative_l2"]
        ),
        "rollout_all_call_mean_relative_l2": float(
            selection_rollout["mean_full_horizon_relative_l2"]
        ),
        "rollout_h79_relative_l2": float(endpoints["79"]),
        "rollout_completion_rate": float(selection_rollout["completion_rate"]),
        "rollout_hard_failure_count": int(
            selection_rollout.get("hard_failure_count", 0)
        ),
        "physical_admissibility_rate": float(
            selection_rollout["physical_admissibility_rate"]
        ),
    }
    return {
        "checkpoint_training_metrics": metrics,
        "fixed_seen_one_step": fixed_seen,
        "fixed_validation_one_step": fixed_validation,
        "selection_rollout": selection_rollout,
        "fixed_seen_boundary_policy_digests": {
            key: record["policy_digest"]
            for key, record in train_policy_metadata.items()
        },
    }


def _sentinel_descriptor(
    base: Mapping[str, Any],
    *,
    replication_root: Path,
    file_records: Mapping[str, Mapping[str, Any]],
    expected_source_set_digest: str = EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
) -> dict[str, Any]:
    seed = int(base["seed"])
    count = int(base["trajectory_count"])
    architecture = str(base["architecture"])
    step = matched_exposure_step(count)
    run_dir = Path(base["run_dir"])
    checkpoint = run_dir / "sentinels" / f"step_{step:09d}.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    relative = checkpoint.relative_to(replication_root).as_posix()
    record = file_records.get(relative)
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        not isinstance(record, Mapping)
        or int(record.get("bytes", -1)) != checkpoint.stat().st_size
        or record.get("sha256") != checkpoint_sha256
    ):
        raise ValueError(f"sentinel differs from matrix receipt: {relative}")

    payload = load_bump_checkpoint(checkpoint)
    expected_epoch = step // STEPS_PER_EPOCH - 1
    sentinel_contract = payload.get("sentinel_contract")
    if (
        int(payload.get("epoch", -1)) != expected_epoch
        or payload.get("checkpoint_role") != "model_only_sentinel"
        or payload.get("resume_supported") is not False
        or "optimizer_state" in payload
        or "scheduler_state" in payload
        or not isinstance(sentinel_contract, Mapping)
        or sentinel_contract.get("schema") != SENTINEL_SCHEMA
        or sentinel_contract.get("evaluation_initialization_supported") is not True
        or sentinel_contract.get("exact_training_resume_supported") is not False
        or payload.get("data_manifest_digest") != EXPECTED_DATA_MANIFEST_DIGEST
        or payload.get("source_snapshot", {}).get("source_set_digest")
        != expected_source_set_digest
        or payload.get("config_digest") != base["summary"]["config_digest"]
        or payload.get("normalization_digest")
        != base["summary"]["normalization_digest"]
    ):
        raise ValueError(
            f"matched-exposure sentinel payload changed for seed={seed} "
            f"n={count} {architecture}"
        )
    metrics = _metric_at_step(run_dir, step)
    if int(metrics["epoch"]) != expected_epoch:
        raise ValueError("sentinel epoch and exact metric row disagree")

    descriptor = dict(base)
    descriptor.update(
        {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_role": CHECKPOINT_ROLE,
            "optimizer_step": step,
            "presentations_per_training_trajectory": (
                PRESENTATIONS_PER_TRAJECTORY
            ),
            "config_digest": str(payload["config_digest"]),
            "normalization_digest": str(payload["normalization_digest"]),
            "selected_training_metrics": metrics,
        }
    )
    del payload
    return descriptor


def discover_sentinels(
    replication_root: Path, split_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    endpoints = _replication_descriptors(replication_root, split_manifest)
    bases = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
        ): row
        for row in endpoints
        if row["checkpoint_role"] == "selected"
    }
    expected = {
        (seed, count, architecture)
        for seed in REPLICATION_SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    }
    if set(bases) != expected:
        raise ValueError("B1-C2 replication endpoints changed")
    records = _matrix_file_records(replication_root)
    descriptors = [
        _sentinel_descriptor(
            bases[(seed, count, architecture)],
            replication_root=replication_root,
            file_records=records,
        )
        for seed in REPLICATION_SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    ]
    if len(descriptors) != 24:
        raise ValueError("B1-C3 must expose exactly 24 sentinels")
    return descriptors


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C3 output directory must not already exist")
    split_manifest = _load_json(args.split_manifest)
    if (
        split_manifest.get("schema") != SPLIT_SCHEMA
        or split_manifest.get("partition_digest")
        != EXPECTED_SPLIT_PARTITION_DIGEST
    ):
        raise ValueError("D094 split manifest changed")

    prior_audit = _validate_prior_audit(args.b1_c2_audit_root)
    descriptors = discover_sentinels(args.replication_root, split_manifest)
    cohorts = _audit_cohorts(split_manifest, descriptors, prior_audit)
    validation_keys = cohorts["validation_keys"]
    outside_by_seed = cohorts["outside_selection_keys_by_seed"]
    common_outside_keys = cohorts["common_outside_selection_keys"]

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if store.manifest_digest != EXPECTED_DATA_MANIFEST_DIGEST:
            raise ValueError("B1-C3 evaluation data manifest changed")
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
            seed = int(descriptor["seed"])
            count = int(descriptor["trajectory_count"])
            outside_keys = outside_by_seed[str(seed)]
            result = _evaluate_cell(
                descriptor,
                store=store,
                outside_keys=outside_keys,
                policies=policies,
                policy_metadata=policy_metadata,
                expected_checkpoint_source_set_digest=(
                    EXPECTED_REPLICATION_SOURCE_SET_DIGEST
                ),
                device=device,
                amp=args.amp,
                shock_quantile=args.shock_quantile,
            )
            result.pop("selected_training_metrics")
            recomputed = _evaluate_recomputed_scopes(
                descriptor,
                store=store,
                fixed_seen_pairs=fixed_seen_pairs_by_count[count],
                fixed_validation_pairs=fixed_validation_pairs,
                selection_keys=cohorts["selection_keys_by_seed"][str(seed)],
                policies=policies,
                device=device,
                amp=args.amp,
            )
            result.update(recomputed)
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
                EXPECTED_REPLICATION_SOURCE_SET_DIGEST
            )
            result["audit_cohort"] = {
                "scope": "seed_specific_outside_checkpoint_selection",
                "seed": seed,
                "selection_count": len(
                    cohorts["selection_keys_by_seed"][str(seed)]
                ),
                "outside_selection_count": len(outside_keys),
                "outside_selection_keys": outside_keys,
            }
            result["common_outside_selection_rollout"] = _rollout_subset_summary(
                result["outside_selection_rollout"], common_outside_keys
            )
            results.append(result)
            atomic_write_json(
                args.output_dir
                / (
                    f"s{seed}_n{int(descriptor['trajectory_count']):03d}_"
                    f"{descriptor['architecture']}_{CHECKPOINT_ROLE}.json"
                ),
                result,
            )
            print(
                json.dumps(
                    {
                        "amp": args.amp,
                        "architecture": descriptor["architecture"],
                        "completed_checkpoints": len(results),
                        "elapsed_seconds": result["elapsed_seconds"],
                        "seed": seed,
                        "trajectory_count": descriptor["trajectory_count"],
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
        "scientific_scope": (
            "two_seed_exact_64_presentations_per_trajectory_sentinel_audit"
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "training_or_resume_performed": False,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "prior_b1_c2_audit_summary_sha256": (
            EXPECTED_B1_C2_AUDIT_SUMMARY_SHA256
        ),
        "evaluator_source_snapshot": source_snapshot,
        "seeds": list(REPLICATION_SEEDS),
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
        "selection_keys_by_seed": cohorts["selection_keys_by_seed"],
        "outside_selection_keys_by_seed": outside_by_seed,
        "common_outside_selection_keys": common_outside_keys,
        "common_outside_selection_count": len(common_outside_keys),
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
            "three-seed fixed-exposure uncertainty",
            "fixed optimizer updates, compute, or learning-rate phase",
            "checkpoint selection on outside-selection trajectories",
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
        f"s{seed}_n{count:03d}_{architecture}_{CHECKPOINT_ROLE}.json"
        for seed in REPLICATION_SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    ]
    artifact_files = ["summary.json", "checkpoint_summary.csv", *cell_files]
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
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
