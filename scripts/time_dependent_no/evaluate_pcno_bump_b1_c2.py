#!/usr/bin/env python3
"""Audit the frozen three-seed D094 B1-C2 checkpoints outside selection.

The audit evaluates selected and terminal checkpoints for the retained seed-0
ladder and the two B1-C2 replication seeds.  Each of the 72 fixed checkpoints
uses its seed's 28 trajectories excluded from checkpoint selection.  A fixed
nine-trajectory subset excluded for all three seeds supplies the common-cohort
view.  The evaluator never reselects a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    ROLLOUT_CHECKPOINTS,
    _build_boundary_policies,
    _evaluate_cell,
    _load_json,
    _load_jsonl,
    outside_selection_keys,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_ladder import (
    B1A_SOURCE_SET_DIGEST,
    FRESH_SOURCE_SET_DIGEST,
    discover_scaling_cells,
    training_metric_scopes,
    verify_retained_source_snapshot,
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

SCHEMA = "d094_b1_c2_three_seed_outside_selection_audit_v2"
ARTIFACT_SCHEMA = "d094_b1_c2_three_seed_outside_selection_artifacts_v2"
MATRIX_RECEIPT_SCHEMA = "d094_b1_c2_matrix_receipt_v1"
CELL_RECEIPT_SCHEMA = "d094_b1_c2_cell_receipt_v1"
SPLIT_SCHEMA = "d094_bump_trajectory_scaling_split_v1"
TRAINING_CONTRACT_SCHEMA = "d094_bump_scaling_training_v1"

EXPECTED_SPLIT_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
EXPECTED_DATA_MANIFEST_DIGEST = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
EXPECTED_REPLICATION_SOURCE_SET_DIGEST = (
    "3dc03431bf2af1eb48b9b933d28ccbe60a25f78fe9cd64396fd4bcb2d0b91e1e"
)
EXPECTED_REPLICATION_SOURCE_COMMIT = "2372b8b34cb8ddd21e4776faaa04c92ce3e60c00"
EXPECTED_REPLICATION_ARCHIVE_SHA256 = (
    "ff029e6321686b1c2733f6d66dca237c45b323353eab4c2a38555d216ddd355b"
)
EXPECTED_VALIDATION_KEY_DIGEST = (
    "7bed09ff30a07b7440a0edf95ac2f9227d8f8f426716c97bc40f4e144a7d1db9"
)
EXPECTED_SELECTION_KEY_DIGESTS = {
    20_260_718: "d976e8bb8474db7f9b9736825e736ad8d72df417f55f944b1152ff4df0f017f2",
    20_260_812: "2c169b57e9e1448976a948c80fd916d08129ed57613480550e0e4398a6b3dfa8",
    20_260_813: "2e68bad2fc75050d20d923ec726bccccf7738ee8f7107e463fb42887115c476f",
}
EXPECTED_COMMON_OUTSIDE_KEY_DIGEST = (
    "4c9cb532143d457c8d57928a8054a14c08a002b60ca02e7ace157c9d30bc6f9c"
)

SEEDS = (20_260_718, 20_260_812, 20_260_813)
REPLICATION_SEEDS = SEEDS[1:]
TRAJECTORY_COUNTS = (8, 16, 32, 64, 128, 256)
ARCHITECTURES = ("pcno", "pcfno")
CHECKPOINT_ROLES = ("selected", "terminal")
EXPECTED_OPTIMIZER_STEPS = 20_480
STEPS_PER_EPOCH = 256

EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c2.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-ladder-root", type=Path, required=True)
    parser.add_argument("--seed0-n256-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
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
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    return parser


def _validate_matrix_receipt(receipt: Mapping[str, Any]) -> None:
    expected = {
        "schema": MATRIX_RECEIPT_SCHEMA,
        "status": "complete",
        "attempt": "d094_b1_c2_seed_replication_20260824b",
        "cell_count": 24,
        "seeds": list(REPLICATION_SEEDS),
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "architectures": list(ARCHITECTURES),
        "source_commit": EXPECTED_REPLICATION_SOURCE_COMMIT,
        "source_archive_sha256": EXPECTED_REPLICATION_ARCHIVE_SHA256,
        "source_set_digest": EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
        "split_partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
        "historical_test_population_accessed": False,
    }
    drifted = [name for name, value in expected.items() if receipt.get(name) != value]
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        drifted.append("files")
    if drifted:
        raise ValueError(f"B1-C2 matrix receipt changed: {drifted}")


def _validate_replication_contract(
    contract: Mapping[str, Any], *, seed: int, trajectory_count: int, architecture: str
) -> None:
    branch = "full" if architecture == "pcno" else "no_gradient"
    expected = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "status": "completed_replication_cell",
        "registered_stage": "b1_c2_seed_replication_20480",
        "initialization_seed": seed,
        "trajectory_count": trajectory_count,
        "differential_branch_mode": branch,
        "schedule_arm": "b1_c2_replication_stretched",
        "requested_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "actual_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "requested_epochs": 80,
        "completed_epochs": 80,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": FINITE_ONLY_ROLLOUT_POLICY,
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "historical_test_population_accessed": False,
        "automatic_continuation_authorized": False,
        "source_manifest_sha256": EXPECTED_DATA_MANIFEST_DIGEST,
        "split_partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
        "selected_terminal_checkpoint_payload": "model_only_evaluation",
        "sentinel_steps": [64 * trajectory_count],
    }
    drifted = [name for name, value in expected.items() if contract.get(name) != value]
    if drifted:
        raise ValueError(
            f"B1-C2 seed={seed} n={trajectory_count} {architecture} contract "
            f"changed: {drifted}"
        )


def _checkpoint_descriptor(
    descriptor: Mapping[str, Any], *, role: str, seed: int
) -> dict[str, Any]:
    if role not in CHECKPOINT_ROLES:
        raise ValueError(f"unknown checkpoint role: {role}")
    run_dir = Path(descriptor["run_dir"])
    summary = descriptor["summary"]
    scopes = descriptor["training_metric_scopes"]
    metrics = scopes[f"{role}_checkpoint"]
    filename = "best.pt" if role == "selected" else "last.pt"
    digest_name = "best_checkpoint" if role == "selected" else "last_checkpoint"
    path = run_dir / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = sha256_file(path)
    if checkpoint_sha256 != str(summary["artifact_sha256"][digest_name]):
        raise ValueError(f"{role} checkpoint digest changed for {run_dir.name}")
    checkpoint = load_bump_checkpoint(path)
    expected_source_digest = str(descriptor["expected_checkpoint_source_set_digest"])
    if (
        int(checkpoint.get("epoch", -1)) != int(metrics["epoch"])
        or checkpoint.get("data_manifest_digest") != EXPECTED_DATA_MANIFEST_DIGEST
        or checkpoint.get("source_snapshot", {}).get("source_set_digest")
        != expected_source_digest
    ):
        raise ValueError(f"{role} checkpoint payload changed for {run_dir.name}")
    expected_checkpoint_role = (
        f"evaluation_only_{role}" if seed in REPLICATION_SEEDS else "training"
    )
    if checkpoint.get("checkpoint_role") != expected_checkpoint_role:
        raise ValueError(f"{role} checkpoint role changed for {run_dir.name}")
    result = dict(descriptor)
    result.update(
        {
            "seed": seed,
            "checkpoint_role": role,
            "optimizer_step": int(metrics["optimizer_step"]),
            "checkpoint": path,
            "checkpoint_sha256": checkpoint_sha256,
            "config_digest": str(checkpoint["config_digest"]),
            "normalization_digest": str(checkpoint["normalization_digest"]),
            "selected_training_metrics": dict(metrics),
        }
    )
    del checkpoint
    return result


def _seed0_descriptors(ladder_root: Path, n256_root: Path) -> list[dict[str, Any]]:
    cells = discover_scaling_cells(ladder_root, n256_root)
    descriptors: list[dict[str, Any]] = []
    for identity in sorted(cells):
        base = dict(cells[identity])
        base["expected_checkpoint_source_set_digest"] = (
            FRESH_SOURCE_SET_DIGEST if identity[0] < 256 else B1A_SOURCE_SET_DIGEST
        )
        for role in CHECKPOINT_ROLES:
            descriptors.append(_checkpoint_descriptor(base, role=role, seed=SEEDS[0]))
    return descriptors


def _validate_cell_receipt(
    receipt: Mapping[str, Any],
    *,
    run_name: str,
    seed: int,
    trajectory_count: int,
    architecture: str,
    scopes: Mapping[str, Any],
) -> None:
    expected = {
        "schema": CELL_RECEIPT_SCHEMA,
        "status": "pass",
        "cell": run_name,
        "seed": seed,
        "trajectory_count": trajectory_count,
        "architecture": architecture,
        "branch": "full" if architecture == "pcno" else "no_gradient",
        "source_set_digest": EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
        "split_partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
        "historical_test_population_accessed": False,
        "selected_checkpoint": scopes["selected_checkpoint"],
        "terminal_checkpoint": scopes["terminal_checkpoint"],
        "sentinel_step": 64 * trajectory_count,
    }
    drifted = [name for name, value in expected.items() if receipt.get(name) != value]
    if drifted:
        raise ValueError(f"B1-C2 cell receipt changed for {run_name}: {drifted}")


def _replication_descriptors(
    replication_root: Path, split_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if replication_root.is_symlink() or not replication_root.is_dir():
        raise ValueError("B1-C2 replication root must be a regular directory")
    if (
        not (replication_root / "matrix.completed").is_file()
        or not (replication_root / "matrix.exit").is_file()
        or (replication_root / "matrix.exit").read_text(encoding="utf-8").strip() != "0"
    ):
        raise ValueError("B1-C2 replication root lacks clean completion receipts")
    _validate_matrix_receipt(_load_json(replication_root / "matrix_receipt.json"))

    outputs = replication_root / "outputs"
    receipts = replication_root / "receipts"
    run_dirs = sorted(path for path in outputs.iterdir() if path.is_dir())
    if len(run_dirs) != 24:
        raise ValueError("B1-C2 replication root must contain exactly 24 cells")

    expected_identities = {
        (seed, count, architecture)
        for seed in REPLICATION_SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
    }
    observed: dict[tuple[int, int, str], dict[str, Any]] = {}
    descriptors: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _load_json(run_dir / "summary.json")
        contract = _load_json(run_dir / "bump_scaling_contract.json")
        seed = int(contract["initialization_seed"])
        trajectory_count = int(contract["trajectory_count"])
        branch = str(contract["differential_branch_mode"])
        architecture = "pcno" if branch == "full" else "pcfno"
        identity = (seed, trajectory_count, architecture)
        if identity not in expected_identities or identity in observed:
            raise ValueError(f"unexpected or duplicate B1-C2 cell: {identity}")
        _validate_replication_contract(
            contract,
            seed=seed,
            trajectory_count=trajectory_count,
            architecture=architecture,
        )
        if (
            int(summary.get("completed_epochs", -1)) != 80
            or int(summary.get("actual_optimizer_steps", -1))
            != EXPECTED_OPTIMIZER_STEPS
            or summary.get("wall_time_stop_reason") is not None
            or summary.get("data_manifest_digest") != EXPECTED_DATA_MANIFEST_DIGEST
        ):
            raise ValueError(f"incomplete B1-C2 cell: {run_dir.name}")
        verify_retained_source_snapshot(
            run_dir,
            summary,
            expected_source_set_digest=EXPECTED_REPLICATION_SOURCE_SET_DIGEST,
        )
        split = _load_json(run_dir / "split.json")
        if split.get("test_keys") != []:
            raise ValueError("B1-C2 cell exposes a sealed/test population")
        expected_train_keys = [
            str(key)
            for key in split_manifest["nested_exposure"]["subsets"][
                str(trajectory_count)
            ]
        ]
        if [str(key) for key in contract["train_keys"]] != expected_train_keys:
            raise ValueError(f"B1-C2 train subset changed for {identity}")
        rows = _load_jsonl(run_dir / "metrics.jsonl")
        if len(rows) != 80 or any(
            int(row["epoch"]) != index for index, row in enumerate(rows)
        ):
            raise ValueError(f"B1-C2 metric history changed for {identity}")
        scopes = training_metric_scopes(run_dir, summary, require_receipt=True)
        cell_receipt = _load_json(receipts / f"{run_dir.name}.json")
        _validate_cell_receipt(
            cell_receipt,
            run_name=run_dir.name,
            seed=seed,
            trajectory_count=trajectory_count,
            architecture=architecture,
            scopes=scopes,
        )
        base = {
            "run_dir": run_dir,
            "run": run_dir.name,
            "architecture": architecture,
            "schedule": "b1_c2_replication_stretched",
            "trajectory_count": trajectory_count,
            "summary": summary,
            "contract": contract,
            "training_metric_scopes": scopes,
            "split": split,
            "expected_checkpoint_source_set_digest": (
                EXPECTED_REPLICATION_SOURCE_SET_DIGEST
            ),
        }
        observed[identity] = base
        for role in CHECKPOINT_ROLES:
            descriptors.append(_checkpoint_descriptor(base, role=role, seed=seed))

    if set(observed) != expected_identities or len(descriptors) != 48:
        raise ValueError("B1-C2 replication matrix identity set changed")
    for seed in REPLICATION_SEEDS:
        for count in TRAJECTORY_COUNTS:
            pcno = observed[(seed, count, "pcno")]
            pcfno = observed[(seed, count, "pcfno")]
            if (
                pcno["summary"]["normalization_digest"]
                != pcfno["summary"]["normalization_digest"]
                or pcno["summary"]["initialization_control"]["differential_branch"][
                    "initial_full_state_sha256"
                ]
                != pcfno["summary"]["initialization_control"]["differential_branch"][
                    "initial_full_state_sha256"
                ]
                or pcno["summary"]["initialization_control"]["differential_branch"][
                    "initial_nondifferential_state_sha256"
                ]
                != pcfno["summary"]["initialization_control"]["differential_branch"][
                    "initial_nondifferential_state_sha256"
                ]
            ):
                raise ValueError(
                    f"paired initialization changed for seed={seed} n={count}"
                )
    return descriptors


def discover_checkpoints(
    seed0_ladder_root: Path,
    seed0_n256_root: Path,
    replication_root: Path,
    split_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    descriptors = _seed0_descriptors(seed0_ladder_root, seed0_n256_root)
    descriptors.extend(_replication_descriptors(replication_root, split_manifest))
    identities = {
        (
            int(row["seed"]),
            int(row["trajectory_count"]),
            str(row["architecture"]),
            str(row["checkpoint_role"]),
        )
        for row in descriptors
    }
    expected = {
        (seed, count, architecture, role)
        for seed in SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
        for role in CHECKPOINT_ROLES
    }
    if identities != expected or len(descriptors) != 72:
        raise ValueError("three-seed audit does not expose the exact 72 checkpoints")
    return sorted(
        descriptors,
        key=lambda row: (
            int(row["seed"]),
            int(row["trajectory_count"]),
            ARCHITECTURES.index(str(row["architecture"])),
            CHECKPOINT_ROLES.index(str(row["checkpoint_role"])),
        ),
    )


def _ordered_key_digest(keys: Sequence[str]) -> str:
    payload = json.dumps(
        [str(key) for key in keys], separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_cohorts(
    split_manifest: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve seed-specific audit cohorts and their common held-out subset."""
    selection_by_seed: dict[str, list[str]] = {}
    outside_by_seed: dict[str, list[str]] = {}
    validation_keys: list[str] | None = None
    for seed in SEEDS:
        splits = [
            descriptor["split"]
            for descriptor in descriptors
            if int(descriptor["seed"]) == seed
        ]
        if not splits:
            raise ValueError(f"audit lacks checkpoint splits for seed={seed}")
        validation, selection, outside = outside_selection_keys(split_manifest, splits)
        if validation_keys is None:
            validation_keys = validation
        elif validation != validation_keys:
            raise ValueError("audit seeds do not share the registered validation order")
        observed_digest = _ordered_key_digest(selection)
        if observed_digest != EXPECTED_SELECTION_KEY_DIGESTS[seed]:
            raise ValueError(f"selection cohort changed for seed={seed}")
        selection_by_seed[str(seed)] = selection
        outside_by_seed[str(seed)] = outside

    if validation_keys is None:
        raise ValueError("audit has no validation cohort")
    if _ordered_key_digest(validation_keys) != EXPECTED_VALIDATION_KEY_DIGEST:
        raise ValueError("registered validation cohort changed")
    selection_union = {key for keys in selection_by_seed.values() for key in keys}
    common_outside = [key for key in validation_keys if key not in selection_union]
    if (
        len(common_outside) != 9
        or _ordered_key_digest(common_outside) != EXPECTED_COMMON_OUTSIDE_KEY_DIGEST
    ):
        raise ValueError("common outside-selection cohort changed")
    return {
        "validation_keys": validation_keys,
        "selection_keys_by_seed": selection_by_seed,
        "outside_selection_keys_by_seed": outside_by_seed,
        "common_outside_selection_keys": common_outside,
    }


def _rollout_subset_summary(
    rollout: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    rows = rollout.get("trajectories")
    if not isinstance(rows, Sequence):
        raise TypeError("rollout lacks trajectory rows")
    rows_by_key = {str(row["trajectory"]): row for row in rows}
    ordered_keys = [str(key) for key in keys]
    if (
        not ordered_keys
        or len(rows_by_key) != len(rows)
        or len(set(ordered_keys)) != len(ordered_keys)
        or not set(ordered_keys) <= set(rows_by_key)
    ):
        raise ValueError("rollout does not contain the exact requested subset")
    selected = [rows_by_key[key] for key in ordered_keys]
    completed = [row for row in selected if bool(row["completed"])]
    full_horizon_values = [
        float(row["mean_prefix_relative_l2"])
        for row in selected
        if bool(row["completed"]) and row.get("mean_prefix_relative_l2") is not None
    ]
    endpoint_means: dict[str, float | None] = {}
    endpoint_counts: dict[str, int] = {}
    for checkpoint in ROLLOUT_CHECKPOINTS:
        values = [
            float(row["endpoint_relative_l2"][str(checkpoint)])
            for row in selected
            if str(checkpoint) in row["endpoint_relative_l2"]
        ]
        endpoint_counts[str(checkpoint)] = len(values)
        endpoint_means[str(checkpoint)] = (
            None if not values else sum(values) / len(values)
        )
    return {
        "keys": ordered_keys,
        "num_trajectories": len(selected),
        "completed": len(completed),
        "completion_rate": len(completed) / len(selected),
        "hard_failure_count": sum(
            row.get("hard_failure_cause") is not None for row in selected
        ),
        "physical_admissibility_rate": sum(
            bool(row["physically_admissible"]) for row in selected
        )
        / len(selected),
        "mean_full_horizon_relative_l2": (
            sum(full_horizon_values) / len(full_horizon_values)
            if len(full_horizon_values) == len(selected)
            else None
        ),
        "mean_endpoint_relative_l2": endpoint_means,
        "endpoint_population_count": endpoint_counts,
        "rollout_failure_policy": rollout["rollout_failure_policy"],
    }


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    rollout = result["outside_selection_rollout"]
    common = result["common_outside_selection_rollout"]
    metrics = result["checkpoint_training_metrics"]
    return {
        "seed": result["seed"],
        "trajectory_count": result["trajectory_count"],
        "architecture": result["architecture"],
        "checkpoint_role": result["checkpoint_role"],
        "optimizer_step": result["optimizer_step"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "online_train_one_step_relative_l2": metrics[
            "online_train_one_step_relative_l2"
        ],
        "fixed_seen_train_one_step_relative_l2": metrics[
            "fixed_seen_train_one_step_relative_l2"
        ],
        "fixed_validation_one_step_relative_l2": metrics[
            "fixed_validation_one_step_relative_l2"
        ],
        "internal_rollout_all_call_mean_relative_l2": metrics[
            "rollout_all_call_mean_relative_l2"
        ],
        "internal_rollout_h79_relative_l2": metrics["rollout_h79_relative_l2"],
        "seed_specific_outside_rollout_all_call_mean_relative_l2": rollout[
            "mean_full_horizon_relative_l2"
        ],
        "seed_specific_outside_rollout_h79_relative_l2": rollout[
            "mean_endpoint_relative_l2"
        ]["79"],
        "seed_specific_outside_rollout_completion_rate": rollout["completion_rate"],
        "seed_specific_outside_rollout_hard_failure_count": rollout[
            "hard_failure_count"
        ],
        "seed_specific_outside_rollout_physical_admissibility_rate": rollout[
            "physical_admissibility_rate"
        ],
        "common_outside_rollout_all_call_mean_relative_l2": common[
            "mean_full_horizon_relative_l2"
        ],
        "common_outside_rollout_h79_relative_l2": common["mean_endpoint_relative_l2"][
            "79"
        ],
        "common_outside_rollout_completion_rate": common["completion_rate"],
        "common_outside_rollout_hard_failure_count": common["hard_failure_count"],
        "common_outside_rollout_physical_admissibility_rate": common[
            "physical_admissibility_rate"
        ],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C2 audit output directory must not already exist")
    split_manifest = _load_json(args.split_manifest)
    if (
        split_manifest.get("schema") != SPLIT_SCHEMA
        or split_manifest.get("partition_digest") != EXPECTED_SPLIT_PARTITION_DIGEST
    ):
        raise ValueError("D094 split manifest changed")
    descriptors = discover_checkpoints(
        args.seed0_ladder_root,
        args.seed0_n256_root,
        args.replication_root,
        split_manifest,
    )
    cohorts = audit_cohorts(split_manifest, descriptors)
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
            raise ValueError("B1-C2 evaluation data manifest changed")
        if not set(validation_keys) <= set(store.keys):
            raise ValueError("open-validation keys are absent from the shard store")
        reference = load_bump_checkpoint(Path(descriptors[0]["checkpoint"]))
        policies, policy_metadata = _build_boundary_policies(
            store, validation_keys, reference, device
        )
        del reference

        results = []
        for descriptor in descriptors:
            outside_keys = outside_by_seed[str(int(descriptor["seed"]))]
            result = _evaluate_cell(
                descriptor,
                store=store,
                outside_keys=outside_keys,
                policies=policies,
                policy_metadata=policy_metadata,
                expected_checkpoint_source_set_digest=str(
                    descriptor["expected_checkpoint_source_set_digest"]
                ),
                device=device,
                amp=args.amp,
                shock_quantile=args.shock_quantile,
            )
            result["checkpoint_training_metrics"] = result.pop(
                "selected_training_metrics"
            )
            for name in (
                "seed",
                "checkpoint_role",
                "optimizer_step",
                "config_digest",
                "normalization_digest",
            ):
                result[name] = descriptor[name]
            result["checkpoint_source_set_digest"] = descriptor[
                "expected_checkpoint_source_set_digest"
            ]
            result["audit_cohort"] = {
                "scope": "seed_specific_outside_checkpoint_selection",
                "seed": int(descriptor["seed"]),
                "selection_count": len(
                    cohorts["selection_keys_by_seed"][str(int(descriptor["seed"]))]
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
                    f"s{int(descriptor['seed'])}_"
                    f"n{int(descriptor['trajectory_count']):03d}_"
                    f"{descriptor['architecture']}_"
                    f"{descriptor['checkpoint_role']}.json"
                ),
                result,
            )
            print(
                json.dumps(
                    {
                        "architecture": descriptor["architecture"],
                        "checkpoint_role": descriptor["checkpoint_role"],
                        "completed_checkpoints": len(results),
                        "elapsed_seconds": result["elapsed_seconds"],
                        "seed": descriptor["seed"],
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
            "three_seed_selected_terminal_bump_audit_with_seed_specific_and_common_"
            "outside_selection_cohorts"
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "evaluator_source_snapshot": source_snapshot,
        "seeds": list(SEEDS),
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "architectures": list(ARCHITECTURES),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "validation_count": len(validation_keys),
        "selection_rollout_count_by_seed": {
            seed: len(keys) for seed, keys in cohorts["selection_keys_by_seed"].items()
        },
        "outside_selection_count_by_seed": {
            seed: len(keys) for seed, keys in outside_by_seed.items()
        },
        "selection_keys_by_seed": cohorts["selection_keys_by_seed"],
        "outside_selection_keys_by_seed": outside_by_seed,
        "common_outside_selection_count": len(common_outside_keys),
        "common_outside_selection_keys": common_outside_keys,
        "same_audit_cohort_across_all_seeds": False,
        "cross_seed_uncertainty_scope": (
            "initialization_and_seed_specific_selection_cohort_combined; the common-"
            "nine view holds the trajectory cohort fixed"
        ),
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
            "checkpoint reselection on any outside-selection trajectory",
            "independent test performance",
            "a universal data-scaling law from three seeds and one PDE family",
            "a pure initialization-seed effect from seed-specific 28-case metrics",
            "causal attribution of optimizer, representation, data, or capacity",
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
        f"s{seed}_n{count:03d}_{architecture}_{role}.json"
        for seed in SEEDS
        for count in TRAJECTORY_COUNTS
        for architecture in ARCHITECTURES
        for role in CHECKPOINT_ROLES
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
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
