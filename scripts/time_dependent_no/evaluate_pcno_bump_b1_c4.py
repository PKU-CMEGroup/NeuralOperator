#!/usr/bin/env python3
"""Audit the frozen D094 B1-C4 PCNO checkpoints outside selection.

The audit evaluates four already-created checkpoints for each of n={128,256}
on the same 28 open-validation trajectories excluded from checkpoint selection.
It never reselects a checkpoint.  After all eight checkpoints are scored, the
two internally selected checkpoints export two visualization-only trajectory
bundles: the first registered outside-selection case and the largest paired
H79 disagreement among the remaining cases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.analyze_pcno_bump_scaling import _metric_snapshot
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    ROLLOUT_CHECKPOINTS,
    _build_boundary_policies,
    _evaluate_cell,
    _load_json,
    _load_jsonl,
    outside_selection_keys,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_ladder import (
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
    build_bump_checkpoint_model,
    contract_forward_sample,
    load_bump_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import autocast_context, select_device

SCHEMA = "d094_b1_c4_outside_selection_audit_v1"
ARTIFACT_SCHEMA = "d094_b1_c4_outside_selection_artifacts_v1"
BUNDLE_SCHEMA = "d094_b1_c4_visualization_bundle_v1"
SPLIT_SCHEMA = "d094_bump_trajectory_scaling_split_v1"
TRAINING_CONTRACT_SCHEMA = "d094_bump_scaling_training_v1"
EXPECTED_SPLIT_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST = (
    "f447f13f3a9fc682e14a454bfe25b48a0a4090308ea4ae18da30e2ac2e1dd774"
)
EXPECTED_DATA_MANIFEST_DIGEST = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
TRAJECTORY_COUNTS = (128, 256)
EXPECTED_SENTINELS = {128: (8_192, 20_480), 256: (16_384, 20_480)}
EXPECTED_SELECTED_STEP = 38_400
EXPECTED_TERMINAL_STEP = 40_960
STEPS_PER_EPOCH = 256
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c4.py",
    "scripts/time_dependent_no/visualize_pcno_bump_b1_c4.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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


def _checkpoint_metric_anchor(
    rows: Sequence[Mapping[str, Any]], optimizer_step: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if int(row["train"]["completed_optimizer_steps"]) == optimizer_step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"optimizer step {optimizer_step} does not identify one metric row"
        )
    return _metric_snapshot(matches[0])


def _validate_training_contract(
    contract: Mapping[str, Any], *, trajectory_count: int
) -> None:
    expected = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "status": "completed_unreplicated_pilot",
        "registered_stage": "b1_c4_pcno_40960",
        "trajectory_count": trajectory_count,
        "differential_branch_mode": "full",
        "schedule_arm": "b1_c4_cold_stretched",
        "requested_optimizer_steps": EXPECTED_TERMINAL_STEP,
        "actual_optimizer_steps": EXPECTED_TERMINAL_STEP,
        "requested_epochs": 160,
        "completed_epochs": 160,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "checkpoint_selection_mode": "full_horizon_error_first",
        "rollout_failure_policy": FINITE_ONLY_ROLLOUT_POLICY,
        "rollout_selection_trajectory_count": 16,
        "rollout_steps": 79,
        "historical_test_population_accessed": False,
        "automatic_continuation_authorized": False,
        "source_manifest_sha256": EXPECTED_DATA_MANIFEST_DIGEST,
        "split_partition_digest": EXPECTED_SPLIT_PARTITION_DIGEST,
    }
    drifted = [name for name, value in expected.items() if contract.get(name) != value]
    if (
        tuple(int(step) for step in contract.get("sentinel_steps", ()))
        != (EXPECTED_SENTINELS[trajectory_count])
    ):
        drifted.append("sentinel_steps")
    if drifted:
        raise ValueError(
            f"B1-C4 n={trajectory_count} training contract changed: {drifted}"
        )


def _validate_run_root(run_root: Path) -> None:
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("B1-C4 run root must be a regular directory")
    exit_path = run_root / "runner.exit"
    if (
        not exit_path.is_file()
        or exit_path.read_text(encoding="utf-8").strip() != "0"
        or not (run_root / "runner.completed").is_file()
    ):
        raise ValueError("B1-C4 run root lacks clean completion receipts")


def _checkpoint_plan(
    run_dir: Path,
    *,
    trajectory_count: int,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    best_epoch = int(summary["best_epoch"])
    selected_step = int(rows[best_epoch]["train"]["completed_optimizer_steps"])
    if selected_step != EXPECTED_SELECTED_STEP:
        raise ValueError("B1-C4 selected optimizer step changed")
    plan = [
        ("early_sentinel", EXPECTED_SENTINELS[trajectory_count][0], None),
        ("matched_20480", 20_480, None),
        ("selected", selected_step, "best.pt"),
        ("terminal", EXPECTED_TERMINAL_STEP, "last.pt"),
    ]
    descriptors: list[dict[str, Any]] = []
    for role, optimizer_step, fixed_name in plan:
        path = (
            run_dir / fixed_name
            if fixed_name is not None
            else run_dir / "sentinels" / f"step_{optimizer_step:09d}.pt"
        )
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_sha256 = sha256_file(path)
        if role == "selected" and checkpoint_sha256 != str(
            summary["artifact_sha256"]["best_checkpoint"]
        ):
            raise ValueError("B1-C4 best checkpoint digest changed")
        if role == "terminal" and checkpoint_sha256 != str(
            summary["artifact_sha256"]["last_checkpoint"]
        ):
            raise ValueError("B1-C4 terminal checkpoint digest changed")
        checkpoint = load_bump_checkpoint(path)
        expected_epoch = optimizer_step // STEPS_PER_EPOCH - 1
        if (
            int(checkpoint.get("epoch", -1)) != expected_epoch
            or int(checkpoint.get("step_stride", -1)) != 1
            or checkpoint.get("data_manifest_digest") != EXPECTED_DATA_MANIFEST_DIGEST
            or checkpoint.get("source_snapshot", {}).get("source_set_digest")
            != EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
        ):
            raise ValueError(f"checkpoint payload changed at step {optimizer_step}")
        expected_role = "model_only_sentinel" if fixed_name is None else "training"
        if checkpoint.get("checkpoint_role") != expected_role:
            raise ValueError(f"checkpoint role changed at step {optimizer_step}")
        config_digest = str(checkpoint["config_digest"])
        normalization_digest = str(checkpoint["normalization_digest"])
        del checkpoint
        descriptors.append(
            {
                "run_dir": run_dir,
                "run": run_dir.name,
                "architecture": "pcno",
                "schedule": "b1_c4_cold_stretched",
                "trajectory_count": trajectory_count,
                "checkpoint_role": role,
                "optimizer_step": optimizer_step,
                "checkpoint": path,
                "checkpoint_sha256": checkpoint_sha256,
                "config_digest": config_digest,
                "normalization_digest": normalization_digest,
                "selected_training_metrics": _checkpoint_metric_anchor(
                    rows, optimizer_step
                ),
            }
        )
    return descriptors


def discover_checkpoints(run_root: Path) -> list[dict[str, Any]]:
    _validate_run_root(run_root)
    outputs = run_root / "outputs"
    if outputs.is_symlink() or not outputs.is_dir():
        raise ValueError("B1-C4 outputs must be a regular directory")
    descriptors: list[dict[str, Any]] = []
    observed_counts: set[int] = set()
    for run_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        if run_dir.is_symlink():
            raise ValueError("B1-C4 cell directory must not be a symlink")
        summary = _load_json(run_dir / "summary.json")
        contract = _load_json(run_dir / "bump_scaling_contract.json")
        trajectory_count = int(contract["trajectory_count"])
        if (
            trajectory_count not in TRAJECTORY_COUNTS
            or trajectory_count in observed_counts
        ):
            raise ValueError(
                "B1-C4 root does not contain one cell per trajectory count"
            )
        observed_counts.add(trajectory_count)
        _validate_training_contract(contract, trajectory_count=trajectory_count)
        if (
            int(summary.get("completed_epochs", -1)) != 160
            or int(summary.get("actual_optimizer_steps", -1)) != EXPECTED_TERMINAL_STEP
            or summary.get("wall_time_stop_reason") is not None
            or summary.get("data_manifest_digest") != EXPECTED_DATA_MANIFEST_DIGEST
        ):
            raise ValueError(f"B1-C4 n={trajectory_count} run is incomplete")
        verify_retained_source_snapshot(
            run_dir,
            summary,
            expected_source_set_digest=EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        )
        split = _load_json(run_dir / "split.json")
        if split.get("test_keys") != []:
            raise ValueError("B1-C4 run exposes a sealed/test population")
        rows = _load_jsonl(run_dir / "metrics.jsonl")
        if len(rows) != 160 or any(
            int(row["epoch"]) != index for index, row in enumerate(rows)
        ):
            raise ValueError("B1-C4 metric history is incomplete or out of order")
        for descriptor in _checkpoint_plan(
            run_dir,
            trajectory_count=trajectory_count,
            summary=summary,
            rows=rows,
        ):
            descriptor["summary"] = summary
            descriptor["contract"] = contract
            descriptor["split"] = split
            descriptors.append(descriptor)
    if observed_counts != set(TRAJECTORY_COUNTS) or len(descriptors) != 8:
        raise ValueError("B1-C4 root must expose the exact two-by-four audit matrix")
    return sorted(
        descriptors,
        key=lambda row: (int(row["trajectory_count"]), int(row["optimizer_step"])),
    )


def select_diagnostic_keys(
    outside_keys: Sequence[str], selected_results: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Choose visualization cases without changing any reported model metric."""

    keys = [str(key) for key in outside_keys]
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("diagnostic selection needs at least two ordered unique keys")
    if set(selected_results) != set(TRAJECTORY_COUNTS):
        raise ValueError("diagnostic selection requires both selected checkpoints")

    errors: dict[int, dict[str, float]] = {}
    for count in TRAJECTORY_COUNTS:
        rows = selected_results[count]["outside_selection_rollout"]["trajectories"]
        mapping = {
            str(row["trajectory"]): float(row["final_relative_l2"]) for row in rows
        }
        if set(mapping) != set(keys) or any(
            not math.isfinite(value) for value in mapping.values()
        ):
            raise ValueError("selected-checkpoint H79 rows are incomplete or nonfinite")
        errors[count] = mapping

    fixed_key = keys[0]
    disagreement_key = max(
        keys[1:],
        key=lambda key: abs(errors[128][key] - errors[256][key]),
    )

    def record(key: str, criterion: str) -> dict[str, Any]:
        return {
            "trajectory": key,
            "criterion": criterion,
            "n128_h79_relative_l2": errors[128][key],
            "n256_h79_relative_l2": errors[256][key],
            "absolute_h79_difference": abs(errors[128][key] - errors[256][key]),
        }

    return {
        "selection_use": "visualization_only_after_frozen_outside_audit",
        "checkpoint_reselection": False,
        "records": [
            record(fixed_key, "first_registered_outside_selection_key"),
            record(
                disagreement_key,
                "largest_selected_checkpoint_h79_disagreement_excluding_fixed_key",
            ),
        ],
    }


def _write_npz_compressed_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def capture_visualization_bundle(
    models: Mapping[int, torch.nn.Module],
    checkpoints: Mapping[int, Mapping[str, Any]],
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    policy: Mapping[str, Any],
    device: torch.device,
    amp: str,
) -> dict[str, np.ndarray]:
    """Capture free states and exact-input next-state proposals for one case."""

    if set(models) != set(TRAJECTORY_COUNTS) or set(checkpoints) != set(
        TRAJECTORY_COUNTS
    ):
        raise ValueError("visualization capture requires selected n=128 and n=256")
    reference = np.array(store.states(key)[:80], dtype=np.float32, copy=True)
    if reference.shape[0] != 80 or reference.ndim != 3 or reference.shape[-1] != 4:
        raise ValueError(f"trajectory {key} does not expose the exact H79 reference")
    sample = store.tensor_sample(key, 0, step_stride=1, device=device)
    result: dict[str, np.ndarray] = {
        "schema": np.asarray(BUNDLE_SCHEMA),
        "trajectory": np.asarray(str(key)),
        "state_component_names_json": np.asarray(
            json.dumps(["density", "momentum_x", "momentum_y", "energy"])
        ),
        "positions": np.asarray(store.array(key, "nodes"), dtype=np.float32),
        "node_type": np.asarray(store.array(key, "node_type"), dtype=np.int64),
        "node_weights": np.asarray(store.array(key, "node_weights"), dtype=np.float32),
        "reference_states_conservative": reference,
        "physical_times": np.arange(80, dtype=np.float64) * float(store.manifest["dt"]),
        "free_rollout_definition": np.asarray(
            "autoregressive recurrence from reference frame zero"
        ),
        "teacher_forced_definition": np.asarray(
            "each next-state proposal receives the exact reference U(t)"
        ),
        "one_step_residual_definition": np.asarray(
            "predicted residual = proposed U(t+1) - exact reference U(t); "
            "true residual = exact U(t+1) - exact U(t)"
        ),
    }
    for count in TRAJECTORY_COUNTS:
        model = models[count]
        model.eval()
        free_current = sample["current"]
        free_states = [reference[0]]
        teacher_next = []
        for call in range(1, 80):
            exact_current = torch.as_tensor(
                reference[call - 1], dtype=torch.float32, device=device
            ).unsqueeze(0)
            with autocast_context(device, amp):
                free_proposal, _, _ = contract_forward_sample(
                    model,
                    sample,
                    free_current,
                    boundary_policy=policy,
                )
                teacher_proposal, _, _ = contract_forward_sample(
                    model,
                    sample,
                    exact_current,
                    boundary_policy=policy,
                )
            free_np = free_proposal[0].float().cpu().numpy()
            teacher_np = teacher_proposal[0].float().cpu().numpy()
            if not np.isfinite(free_np).all() or not np.isfinite(teacher_np).all():
                raise ValueError(
                    f"n={count} trajectory {key} produced a nonfinite visualization state"
                )
            free_states.append(free_np)
            teacher_next.append(teacher_np)
            free_current = free_proposal
        result[f"n{count}_free_states_conservative"] = np.asarray(
            free_states, dtype=np.float32
        )
        result[f"n{count}_teacher_forced_next_conservative"] = np.asarray(
            teacher_next, dtype=np.float32
        )
        result[f"n{count}_checkpoint_sha256"] = np.asarray(
            str(checkpoints[count]["checkpoint_sha256"])
        )
        result[f"n{count}_state_scale"] = np.asarray(
            model.state_scale.detach().float().cpu().numpy(), dtype=np.float32
        )
    return result


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    rollout = result["outside_selection_rollout"]
    metrics = result["checkpoint_training_metrics"]
    return {
        "trajectory_count": result["trajectory_count"],
        "checkpoint_role": result["checkpoint_role"],
        "optimizer_step": result["optimizer_step"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "config_digest": result["config_digest"],
        "normalization_digest": result["normalization_digest"],
        "online_train_one_step_relative_l2": metrics[
            "online_train_one_step_relative_l2"
        ],
        "fixed_seen_train_one_step_relative_l2": metrics[
            "fixed_seen_train_one_step_relative_l2"
        ],
        "fixed_validation_one_step_relative_l2": metrics[
            "fixed_validation_one_step_relative_l2"
        ],
        "internal_rollout_h79_relative_l2": metrics["rollout_h79_relative_l2"],
        "outside_rollout_all_call_mean_relative_l2": rollout[
            "mean_full_horizon_relative_l2"
        ],
        "outside_rollout_h79_relative_l2": rollout["mean_endpoint_relative_l2"]["79"],
        "outside_rollout_completion_rate": rollout["completion_rate"],
        "outside_rollout_hard_failure_count": rollout["hard_failure_count"],
        "outside_rollout_physical_admissibility_rate": rollout[
            "physical_admissibility_rate"
        ],
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C4 audit output directory must not already exist")
    descriptors = discover_checkpoints(args.run_root)
    split_manifest = _load_json(args.split_manifest)
    if (
        split_manifest.get("schema") != SPLIT_SCHEMA
        or split_manifest.get("partition_digest") != EXPECTED_SPLIT_PARTITION_DIGEST
    ):
        raise ValueError("D094 split manifest changed")
    validation_keys, selection_keys, outside_keys = outside_selection_keys(
        split_manifest, [descriptor["split"] for descriptor in descriptors]
    )

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    store = PCNOEuler2DShardStore(args.data_dir)
    bundle_records: list[dict[str, Any]] = []
    try:
        if store.manifest_digest != EXPECTED_DATA_MANIFEST_DIGEST:
            raise ValueError("B1-C4 evaluation data manifest changed")
        if not set(outside_keys) <= set(store.keys):
            raise ValueError("outside-selection keys are absent from the shard store")
        reference_checkpoint = load_bump_checkpoint(Path(descriptors[0]["checkpoint"]))
        policies, policy_metadata = _build_boundary_policies(
            store, outside_keys, reference_checkpoint, device
        )
        del reference_checkpoint

        results = []
        for descriptor in descriptors:
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
            result["checkpoint_training_metrics"] = result.pop(
                "selected_training_metrics"
            )
            result["checkpoint_role"] = descriptor["checkpoint_role"]
            result["optimizer_step"] = int(descriptor["optimizer_step"])
            result["checkpoint_source_set_digest"] = (
                EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
            )
            result["config_digest"] = descriptor["config_digest"]
            result["normalization_digest"] = descriptor["normalization_digest"]
            results.append(result)
            atomic_write_json(
                args.output_dir
                / (
                    f"n{int(descriptor['trajectory_count']):03d}_"
                    f"step_{int(descriptor['optimizer_step']):06d}.json"
                ),
                result,
            )

        selected_results = {
            int(result["trajectory_count"]): result
            for result in results
            if result["checkpoint_role"] == "selected"
        }
        diagnostic_selection = select_diagnostic_keys(outside_keys, selected_results)
        selected_descriptors = {
            int(descriptor["trajectory_count"]): descriptor
            for descriptor in descriptors
            if descriptor["checkpoint_role"] == "selected"
        }
        selected_checkpoints = {
            count: load_bump_checkpoint(Path(descriptor["checkpoint"]))
            for count, descriptor in selected_descriptors.items()
        }
        models = {
            count: build_bump_checkpoint_model(checkpoint, device)
            for count, checkpoint in selected_checkpoints.items()
        }
        try:
            for record in diagnostic_selection["records"]:
                key = str(record["trajectory"])
                arrays = capture_visualization_bundle(
                    models,
                    selected_descriptors,
                    store,
                    key,
                    policy=policies[key],
                    device=device,
                    amp=args.amp,
                )
                relative = f"visualization_bundles/trajectory_{key}.npz"
                path = args.output_dir / relative
                _write_npz_compressed_atomic(path, arrays)
                bundle_records.append(
                    {
                        **record,
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        finally:
            del models, selected_checkpoints
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": (
            "single_seed_n128_n256_budget_audit_outside_checkpoint_selection"
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_on_outside_cases": False,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "checkpoint_source_set_digest": EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        "evaluator_source_snapshot": source_snapshot,
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "checkpoint_steps": {
            str(count): [
                int(descriptor["optimizer_step"])
                for descriptor in descriptors
                if int(descriptor["trajectory_count"]) == count
            ]
            for count in TRAJECTORY_COUNTS
        },
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
        "visualization_case_selection": {
            **diagnostic_selection,
            "records": bundle_records,
        },
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "checkpoint reselection on the 28 outside-selection trajectories",
            "independent test performance",
            "multi-seed data-scaling conclusions",
            "causal attribution of budget, schedule, data, or architecture effects",
            "physical conservation from reconstructed proxy weights",
            "scientific inference from post-hoc visualization-case selection",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_csv(
        args.output_dir / "checkpoint_summary.csv",
        [_summary_row(result) for result in results],
    )
    cell_files = [
        f"n{int(descriptor['trajectory_count']):03d}_"
        f"step_{int(descriptor['optimizer_step']):06d}.json"
        for descriptor in descriptors
    ]
    artifact_files = [
        "summary.json",
        "checkpoint_summary.csv",
        *cell_files,
        *[record["path"] for record in bundle_records],
    ]
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
                "visualization_cases": [
                    row["trajectory"]
                    for row in summary["visualization_case_selection"]["records"]
                ],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
