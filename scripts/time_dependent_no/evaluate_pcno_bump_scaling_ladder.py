#!/usr/bin/env python3
"""Evaluate the selected D094 scaling checkpoints outside their selection set.

The fresh stretched-schedule seed-0 cells at n={8,16,32,64,128} are combined
with the preregistered stretched n=256 B1-A endpoints.  Every already-selected
checkpoint is evaluated on the same 28 open-validation trajectories excluded
from checkpoint selection.  This script never reselects a checkpoint and has
no historical-test input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST as B1A_SOURCE_SET_DIGEST,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    ROLLOUT_CHECKPOINTS,
    _build_boundary_policies,
    _cell_identity,
    _discover_cells,
    _evaluate_cell,
    _load_json,
    _load_jsonl,
    outside_selection_keys,
)
from utility.time_dependent_no.pcno_artifacts import (
    PCNO_SOURCE_SNAPSHOT_SCHEMA,
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

SCHEMA = "d094_bump_scaling_ladder_outside_selection_audit_v1"
ARTIFACT_SCHEMA = "d094_bump_scaling_ladder_outside_selection_artifacts_v1"
METRIC_RECEIPT_SCHEMA = "d094_bump_scaling_metric_receipt_v1"
TRAINING_CONTRACT_SCHEMA = "d094_bump_scaling_training_v1"
FRESH_COUNTS = (8, 16, 32, 64, 128)
ALL_COUNTS = (*FRESH_COUNTS, 256)
ARCHITECTURES = ("pcno", "pcfno")
SCHEDULE = "stretched"
SENTINEL_STEPS = (256, 1_280, 2_560, 3_840, 5_120, 7_680, 10_240, 15_360, 20_480)
FRESH_SOURCE_SET_DIGEST = (
    "c9ecfd93f75f61a69a1333f2f25b0778f4436a33660b40bd3dbc38fc862a8e61"
)
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-root", type=Path, required=True)
    parser.add_argument("--n256-gate-root", type=Path, required=True)
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


def checkpoint_metric_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    train = row.get("train")
    validation = row.get("validation")
    rollout = row.get("rollout")
    if not all(isinstance(value, Mapping) for value in (train, validation, rollout)):
        raise TypeError("checkpoint metric row lacks train, validation, or rollout")
    seen = train.get("comparable_seen")
    endpoints = rollout.get("mean_endpoint_relative_l2")
    if not isinstance(seen, Mapping) or not isinstance(endpoints, Mapping):
        raise TypeError("checkpoint metric row lacks fixed seen or endpoint metrics")
    snapshot = {
        "epoch": int(row["epoch"]),
        "optimizer_step": int(train["completed_optimizer_steps"]),
        "online_train_one_step_relative_l2": float(train["relative_l2"]),
        "fixed_seen_train_one_step_relative_l2": float(seen["relative_l2"]),
        "fixed_validation_one_step_relative_l2": float(validation["relative_l2"]),
        "rollout_all_call_mean_relative_l2": float(
            rollout["mean_full_horizon_relative_l2"]
        ),
        "rollout_h79_relative_l2": float(endpoints["79"]),
        "rollout_completion_rate": float(rollout["completion_rate"]),
        "rollout_hard_failure_count": int(rollout.get("hard_failure_count", 0)),
        "physical_admissibility_rate": float(rollout["physical_admissibility_rate"]),
    }
    finite_names = set(snapshot) - {
        "epoch",
        "optimizer_step",
        "rollout_hard_failure_count",
    }
    if not all(math.isfinite(float(snapshot[name])) for name in finite_names):
        raise ValueError("checkpoint metric snapshot contains a nonfinite scalar")
    return snapshot


def training_metric_scopes(
    run_dir: Path,
    summary: Mapping[str, Any],
    *,
    require_receipt: bool,
) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    best_epoch = int(summary["best_epoch"])
    selected_rows = [row for row in rows if int(row["epoch"]) == best_epoch]
    if len(selected_rows) != 1:
        raise ValueError("best epoch does not identify exactly one metric row")
    selected = checkpoint_metric_snapshot(selected_rows[0])
    terminal = checkpoint_metric_snapshot(rows[-1])
    receipt_path = run_dir / "d094_metric_receipt.json"
    if require_receipt and not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        if receipt.get("schema") != METRIC_RECEIPT_SCHEMA:
            raise ValueError("unsupported D094 metric receipt")
        if (
            receipt.get("selected_checkpoint") != selected
            or receipt.get("terminal_checkpoint") != terminal
            or int(receipt.get("best_epoch", -1)) != best_epoch
            or receipt.get("historical_test_population_accessed") is not False
        ):
            raise ValueError("D094 metric receipt differs from metric history")
    return {"selected_checkpoint": selected, "terminal_checkpoint": terminal}


def _mapping_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_retained_source_snapshot(
    run_dir: Path,
    summary: Mapping[str, Any],
    *,
    expected_source_set_digest: str,
) -> str:
    snapshot_root = run_dir / "source_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    retained = _load_json(manifest_path)
    if retained != summary.get("source_snapshot"):
        raise ValueError("retained source manifest differs from the training summary")
    if retained.get("schema") != PCNO_SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("D094 run lacks a v6 source snapshot")
    digest_fields = {
        "files": "source_set_digest",
        "provenance_files": "provenance_set_digest",
    }
    resolved_root = snapshot_root.resolve()
    for category, digest_field in digest_fields.items():
        records = retained.get(category)
        if not isinstance(records, Mapping) or not records:
            raise ValueError(f"retained source manifest lacks {category}")
        for raw_name, raw_record in records.items():
            if not isinstance(raw_name, str) or not isinstance(raw_record, Mapping):
                raise TypeError("retained source record is malformed")
            relative = Path(raw_name)
            if relative.is_absolute():
                raise ValueError("retained source path must be relative")
            candidate = snapshot_root / relative
            try:
                candidate.resolve().relative_to(resolved_root)
            except ValueError as error:
                raise ValueError("retained source path leaves its snapshot") from error
            if candidate.is_symlink() or not candidate.is_file():
                raise FileNotFoundError(candidate)
            if candidate.stat().st_size != int(raw_record["bytes"]) or sha256_file(
                candidate
            ) != str(raw_record["sha256"]):
                raise ValueError(f"retained source file changed: {raw_name}")
        if retained.get(digest_field) != _mapping_digest(records):
            raise ValueError(f"retained {category} digest changed")
    if retained.get("source_set_digest") != expected_source_set_digest:
        raise ValueError("retained source-set digest differs from the registered run")
    return sha256_file(manifest_path)


def _descriptor(
    run_dir: Path,
    *,
    expected_source_set_digest: str,
    require_metric_receipt: bool,
) -> dict[str, Any]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("D094 run directory must be a regular directory")
    summary = _load_json(run_dir / "summary.json")
    contract = _load_json(run_dir / "bump_scaling_contract.json")
    architecture, schedule = _cell_identity(summary, contract)
    trajectory_count = int(contract["trajectory_count"])
    if schedule != SCHEDULE:
        raise ValueError("ladder endpoint must use the stretched schedule")
    if (
        int(summary["completed_epochs"]) != 80
        or int(summary["actual_optimizer_steps"]) != 20_480
        or summary.get("wall_time_stop_reason") is not None
        or int(contract["actual_optimizer_steps"]) != 20_480
    ):
        raise ValueError("D094 ladder endpoint is incomplete")
    if contract.get("historical_test_population_accessed") is not False:
        raise ValueError("D094 endpoint accessed the historical test population")
    source_snapshot_manifest_sha256 = verify_retained_source_snapshot(
        run_dir,
        summary,
        expected_source_set_digest=expected_source_set_digest,
    )
    checkpoint = run_dir / "best.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != str(summary["artifact_sha256"]["best_checkpoint"]):
        raise ValueError("D094 best-checkpoint digest changed")
    scopes = training_metric_scopes(
        run_dir, summary, require_receipt=require_metric_receipt
    )
    selected = scopes["selected_checkpoint"]
    return {
        "run_dir": run_dir,
        "run": run_dir.name,
        "architecture": architecture,
        "schedule": schedule,
        "trajectory_count": trajectory_count,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "training_source_snapshot_manifest_sha256": source_snapshot_manifest_sha256,
        "expected_checkpoint_source_set_digest": expected_source_set_digest,
        "summary": summary,
        "contract": contract,
        "selected_training_metrics": {
            "epoch": selected["epoch"],
            "optimizer_step": selected["optimizer_step"],
            "online_train_one_step_relative_l2": selected[
                "online_train_one_step_relative_l2"
            ],
            "fixed_seen_train_one_step_relative_l2": selected[
                "fixed_seen_train_one_step_relative_l2"
            ],
            "fixed_validation_one_step_relative_l2": selected[
                "fixed_validation_one_step_relative_l2"
            ],
            "selection_rollout_all_call_mean_relative_l2": selected[
                "rollout_all_call_mean_relative_l2"
            ],
            "selection_rollout_h79_relative_l2": selected["rollout_h79_relative_l2"],
            "selection_rollout_physical_admissibility_rate": selected[
                "physical_admissibility_rate"
            ],
        },
        "training_metric_scopes": scopes,
        "split": _load_json(run_dir / "split.json"),
    }


def discover_scaling_cells(
    ladder_root: Path, n256_gate_root: Path
) -> dict[tuple[int, str], dict[str, Any]]:
    if ladder_root.is_symlink() or not ladder_root.is_dir():
        raise ValueError("ladder root must be a regular directory")
    if not (ladder_root / "ladder.completed").is_file():
        raise ValueError("ladder lacks its completion receipt")
    if (
        not (ladder_root / "ladder.exit").is_file()
        or (ladder_root / "ladder.exit").read_text(encoding="utf-8").strip() != "0"
    ):
        raise ValueError("ladder lacks a zero exit receipt")
    outputs = ladder_root / "outputs"
    run_dirs = sorted(path for path in outputs.iterdir() if path.is_dir())
    cells: dict[tuple[int, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        descriptor = _descriptor(
            run_dir,
            expected_source_set_digest=FRESH_SOURCE_SET_DIGEST,
            require_metric_receipt=True,
        )
        identity = (
            int(descriptor["trajectory_count"]),
            str(descriptor["architecture"]),
        )
        if identity in cells:
            raise ValueError(f"duplicate fresh ladder cell: {identity}")
        cells[identity] = descriptor
    expected_fresh = {
        (trajectory_count, architecture)
        for trajectory_count in FRESH_COUNTS
        for architecture in ARCHITECTURES
    }
    if set(cells) != expected_fresh:
        raise ValueError("fresh ladder does not contain the exact ten-cell matrix")

    gate_cells = _discover_cells(n256_gate_root)
    for architecture in ARCHITECTURES:
        gate = gate_cells[(architecture, SCHEDULE)]
        descriptor = _descriptor(
            Path(gate["run_dir"]),
            expected_source_set_digest=B1A_SOURCE_SET_DIGEST,
            require_metric_receipt=False,
        )
        if int(descriptor["trajectory_count"]) != 256:
            raise ValueError("B1-A endpoint does not use 256 trajectories")
        cells[(256, architecture)] = descriptor
    expected_all = {
        (trajectory_count, architecture)
        for trajectory_count in ALL_COUNTS
        for architecture in ARCHITECTURES
    }
    if set(cells) != expected_all:
        raise ValueError("combined ladder does not contain the exact 12-cell matrix")
    return cells


def validate_nested_training_subsets(
    cells: Mapping[tuple[int, str], Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
) -> None:
    subsets = split_manifest.get("nested_exposure", {}).get("subsets")
    if not isinstance(subsets, Mapping):
        raise TypeError("D094 split lacks nested exposure subsets")
    for trajectory_count in ALL_COUNTS:
        expected = [str(key) for key in subsets[str(trajectory_count)]]
        for architecture in ARCHITECTURES:
            descriptor = cells[(trajectory_count, architecture)]
            actual = [str(key) for key in descriptor["contract"]["train_keys"]]
            if actual != expected:
                raise ValueError(
                    f"D094 train subset changed for n={trajectory_count} "
                    f"architecture={architecture}"
                )


def validate_registered_cell_contract(
    descriptor: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    trajectory_count: int,
) -> None:
    contract = descriptor["contract"]
    split = descriptor["split"]
    expected_contract = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "status": "completed_unreplicated_pilot",
        "engineering_smoke": False,
        "science_result_eligible": False,
        "requested_epochs": 80,
        "completed_epochs": 80,
        "optimizer_steps_per_epoch": 256,
        "requested_optimizer_steps": 20_480,
        "actual_optimizer_steps": 20_480,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": FINITE_ONLY_ROLLOUT_POLICY,
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "fixed_evaluation_windows_per_trajectory": 4,
        "sentinel_every_epochs": 0,
        "sentinel_payload": "model_only",
        "within_update_duplicate_pairs": 0,
        "historical_test_population_accessed": False,
    }
    drifted = [
        name
        for name, expected in expected_contract.items()
        if contract.get(name) != expected
    ]
    if tuple(contract.get("sentinel_steps", ())) != SENTINEL_STEPS:
        drifted.append("sentinel_steps")
    if drifted:
        raise ValueError(f"D094 registered training contract changed: {drifted}")
    expected_bindings = {
        "split_schema": split_manifest["schema"],
        "split_partition_digest": split_manifest["partition_digest"],
        "split_canonical_payload_sha256": split_manifest["canonical_payload_sha256"],
        "source_manifest_sha256": split_manifest["source_manifest_sha256"],
    }
    if any(
        contract.get(name) != expected for name, expected in expected_bindings.items()
    ):
        raise ValueError("D094 training contract differs from the registered split")
    expected_keys = [
        str(key)
        for key in split_manifest["nested_exposure"]["subsets"][str(trajectory_count)]
    ]
    if (
        int(contract.get("trajectory_count", -1)) != trajectory_count
        or [str(key) for key in contract.get("train_keys", ())] != expected_keys
        or [str(key) for key in split.get("train_keys", ())] != expected_keys
    ):
        raise ValueError("D094 cell training population changed")
    if split.get("test_keys") != []:
        raise ValueError("D094 cell exposes a sealed/test population")
    expected_manifest = split_manifest["source_manifest_sha256"]
    if (
        split.get("data_manifest_digest") != expected_manifest
        or descriptor["summary"].get("data_manifest_digest") != expected_manifest
    ):
        raise ValueError("D094 cell data-manifest binding changed")


def _summary_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    selected = cell["training_metric_scopes"]["selected_checkpoint"]
    terminal = cell["training_metric_scopes"]["terminal_checkpoint"]
    rollout = cell["outside_selection_rollout"]
    structure = cell["outside_selection_structure"]
    h79 = cell["outside_selection_h79_structure_means"]
    return {
        "run": cell["run"],
        "trajectory_count": cell["trajectory_count"],
        "architecture": cell["architecture"],
        "schedule": cell["schedule"],
        **{f"selected_{name}": value for name, value in selected.items()},
        **{f"terminal_{name}": value for name, value in terminal.items()},
        "audit_rollout_all_call_mean_relative_l2": rollout[
            "mean_full_horizon_relative_l2"
        ],
        "audit_rollout_h79_relative_l2": rollout["mean_endpoint_relative_l2"]["79"],
        "audit_rollout_final_normal_relative_l2": rollout[
            "mean_final_normal_relative_l2"
        ],
        "audit_rollout_final_boundary_relative_l2": rollout[
            "mean_final_boundary_relative_l2"
        ],
        "audit_rollout_completion_rate": rollout["completion_rate"],
        "audit_rollout_hard_failure_count": rollout["hard_failure_count"],
        "audit_rollout_physical_admissibility_rate": rollout[
            "physical_admissibility_rate"
        ],
        "audit_structure_h79_population_count": structure["endpoints"]["79"][
            "population_count"
        ],
        **{f"audit_structure_h79_{field}": h79[field] for field in h79},
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("ladder audit output directory must not already exist")
    cells = discover_scaling_cells(args.ladder_root, args.n256_gate_root)
    split_manifest = _load_json(args.split_manifest)
    ordered_identities = [
        (trajectory_count, architecture)
        for trajectory_count in ALL_COUNTS
        for architecture in ARCHITECTURES
    ]
    for trajectory_count, architecture in ordered_identities:
        validate_registered_cell_contract(
            cells[(trajectory_count, architecture)],
            split_manifest,
            trajectory_count=trajectory_count,
        )
    validate_nested_training_subsets(cells, split_manifest)
    validation_keys, selection_keys, outside_keys = outside_selection_keys(
        split_manifest, [cells[identity]["split"] for identity in ordered_identities]
    )

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if not set(outside_keys) <= set(store.keys):
            raise ValueError("outside-selection keys are absent from the shard store")
        reference = cells[(FRESH_COUNTS[0], ARCHITECTURES[0])]
        reference_checkpoint = load_bump_checkpoint(Path(reference["checkpoint"]))
        policies, policy_metadata = _build_boundary_policies(
            store, outside_keys, reference_checkpoint, device
        )
        del reference_checkpoint
        results = []
        for identity in ordered_identities:
            descriptor = cells[identity]
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
            result["training_metric_scopes"] = descriptor["training_metric_scopes"]
            result["checkpoint_source_set_digest"] = descriptor[
                "expected_checkpoint_source_set_digest"
            ]
            result["training_source_snapshot_manifest_sha256"] = descriptor[
                "training_source_snapshot_manifest_sha256"
            ]
            results.append(result)
            atomic_write_json(
                args.output_dir
                / (f"n{identity[0]:03d}_{identity[1]}_{descriptor['schedule']}.json"),
                result,
            )
    finally:
        store.close()

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": (
            "single_seed_nested_n8_to_n256_outside_selection_scaling_audit"
        ),
        "historical_test_population_accessed": False,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "checkpoint_source_set_digests": {
            "fresh_n8_to_n128": FRESH_SOURCE_SET_DIGEST,
            "reused_b1a_n256": B1A_SOURCE_SET_DIGEST,
        },
        "evaluator_source_snapshot": source_snapshot,
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "trajectory_counts": list(ALL_COUNTS),
        "architectures": list(ARCHITECTURES),
        "schedule": SCHEDULE,
        "validation_count": len(validation_keys),
        "selection_rollout_count": len(selection_keys),
        "outside_selection_count": len(outside_keys),
        "selection_keys": selection_keys,
        "outside_selection_keys": outside_keys,
        "rollout_horizon": 79,
        "rollout_checkpoints": list(ROLLOUT_CHECKPOINTS),
        "rollout_failure_policy": FINITE_ONLY_ROLLOUT_POLICY,
        "shock_quantile": args.shock_quantile,
        "boundary_policy_digests": {
            key: record["policy_digest"] for key, record in policy_metadata.items()
        },
        "cells": results,
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "checkpoint reselection on the 28 audit trajectories",
            "independent test performance",
            "multi-seed data-scaling or architecture conclusions",
            "causal attribution of any architecture-by-exposure interaction",
            "physical conservation from reconstructed proxy weights",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_csv(
        args.output_dir / "cell_summary.csv", [_summary_row(row) for row in results]
    )
    cell_files = [
        f"n{trajectory_count:03d}_{architecture}_{SCHEDULE}.json"
        for trajectory_count in ALL_COUNTS
        for architecture in ARCHITECTURES
    ]
    artifact_files = ["summary.json", "cell_summary.csv", *cell_files]
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
                "trajectory_counts": summary["trajectory_counts"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
