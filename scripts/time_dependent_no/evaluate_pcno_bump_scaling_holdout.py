#!/usr/bin/env python3
"""Audit D094 schedule-gate checkpoints on the 28 outside-selection cases.

This is a schedule audit, not a second checkpoint-selection pass.  The four
already-selected n=256 checkpoints are evaluated on the exact open-validation
trajectories excluded from rollout selection.  Schedule rank prioritizes
finite numerical completion and rollout error.  Physical admissibility,
boundary, shock/front, high-frequency, and reconstructed-weight proxy-total
diagnostics remain separate reported outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    build_graph_causal_boundary_policy,
)
from utility.time_dependent_no.pcno_rollout import (
    FINITE_ONLY_ROLLOUT_POLICY,
    STRUCTURE_ERROR_FIELDS,
    build_bump_checkpoint_model,
    evaluate_rollouts,
    load_bump_checkpoint,
    rollout_structure_diagnostics,
)
from utility.time_dependent_no.pcno_runtime import select_device

SCHEMA = "d094_bump_schedule_outside_selection_audit_v1"
SPLIT_SCHEMA = "d094_bump_trajectory_scaling_split_v1"
EXPECTED_SPLIT_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST = (
    "6c510fbdca8f50d2bfacd40239574e7ac4496bdb0ba575c5ce69bb744568fca5"
)
ARCHITECTURE_BY_BRANCH_MODE = {"full": "pcno", "no_gradient": "pcfno"}
SCHEDULE_BY_DECAY_STEPS = {5_120: "prefix_tail", 20_480: "stretched"}
CELL_ORDER = (
    ("pcno", "prefix_tail"),
    ("pcno", "stretched"),
    ("pcfno", "prefix_tail"),
    ("pcfno", "stretched"),
)
ROLLOUT_CHECKPOINTS = (20, 40, 60, 79)
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-root", type=Path, required=True)
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"metric history must contain object rows: {path}")
    return rows


def _cell_identity(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[str, str]:
    branch_mode = str(contract.get("differential_branch_mode"))
    if branch_mode not in ARCHITECTURE_BY_BRANCH_MODE:
        raise ValueError(f"unsupported differential branch mode: {branch_mode}")
    scheduler = summary.get("optimizer", {}).get("scheduler_contract", {})
    decay_steps = int(scheduler.get("decay_optimizer_steps", -1))
    if decay_steps not in SCHEDULE_BY_DECAY_STEPS:
        raise ValueError(f"unsupported schedule decay length: {decay_steps}")
    return ARCHITECTURE_BY_BRANCH_MODE[branch_mode], SCHEDULE_BY_DECAY_STEPS[
        decay_steps
    ]


def _selected_training_metrics(
    run_dir: Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    best_epoch = int(summary["best_epoch"])
    selected = [
        row
        for row in _load_jsonl(run_dir / "metrics.jsonl")
        if int(row["epoch"]) == best_epoch
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("rollout"), Mapping):
        raise ValueError("best epoch does not identify one rollout-evaluation row")
    row = selected[0]
    train = row["train"]
    seen = train.get("comparable_seen")
    validation = row["validation"]
    rollout = row["rollout"]
    if not isinstance(seen, Mapping):
        raise TypeError("selected checkpoint lacks fixed seen-train metrics")
    return {
        "epoch": best_epoch,
        "optimizer_step": int(train["completed_optimizer_steps"]),
        "online_train_one_step_relative_l2": float(train["relative_l2"]),
        "fixed_seen_train_one_step_relative_l2": float(seen["relative_l2"]),
        "fixed_validation_one_step_relative_l2": float(validation["relative_l2"]),
        "selection_rollout_all_call_mean_relative_l2": float(
            rollout["mean_full_horizon_relative_l2"]
        ),
        "selection_rollout_h79_relative_l2": float(
            rollout["mean_endpoint_relative_l2"]["79"]
        ),
        "selection_rollout_physical_admissibility_rate": float(
            rollout["physical_admissibility_rate"]
        ),
    }


def _discover_cells(gate_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if gate_root.is_symlink() or not gate_root.is_dir():
        raise ValueError("gate root must be a regular directory")
    if (
        not (gate_root / "gate.exit").is_file()
        or (gate_root / "gate.exit").read_text(encoding="utf-8").strip() != "0"
    ):
        raise ValueError("schedule gate lacks a zero exit receipt")
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in sorted((gate_root / "outputs").glob("gate_n256_*")):
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("gate run directory must be regular")
        summary = _load_json(run_dir / "summary.json")
        contract = _load_json(run_dir / "bump_scaling_contract.json")
        identity = _cell_identity(summary, contract)
        if identity in cells:
            raise ValueError(f"duplicate schedule-gate cell: {identity}")
        checkpoint = run_dir / "best.pt"
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected_checkpoint_sha256 = str(summary["artifact_sha256"]["best_checkpoint"])
        observed_checkpoint_sha256 = sha256_file(checkpoint)
        if observed_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(f"best-checkpoint digest changed: {run_dir.name}")
        cells[identity] = {
            "run_dir": run_dir,
            "run": run_dir.name,
            "architecture": identity[0],
            "schedule": identity[1],
            "checkpoint": checkpoint,
            "checkpoint_sha256": observed_checkpoint_sha256,
            "summary": summary,
            "contract": contract,
            "selected_training_metrics": _selected_training_metrics(run_dir, summary),
            "split": _load_json(run_dir / "split.json"),
        }
    if set(cells) != set(CELL_ORDER):
        raise ValueError("gate root does not contain the exact four-cell cross-product")
    return cells


def outside_selection_keys(
    split_manifest: Mapping[str, Any], cell_splits: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    if split_manifest.get("schema") != SPLIT_SCHEMA:
        raise ValueError("unsupported D094 split manifest")
    if (
        split_manifest.get("state_arrays_opened") is not False
        or split_manifest.get("historical_test_population_opened") is not False
    ):
        raise ValueError("D094 split is not field-blind and test-closed")
    if split_manifest.get("partition_digest") != EXPECTED_SPLIT_PARTITION_DIGEST:
        raise ValueError("D094 partition digest changed")
    split = split_manifest.get("split")
    if not isinstance(split, Mapping):
        raise TypeError("D094 manifest lacks its split object")
    registered_validation = [str(key) for key in split["open_validation_keys"]]
    if len(registered_validation) != 44 or len(set(registered_validation)) != 44:
        raise ValueError("D094 open-validation population must contain 44 unique keys")
    if not cell_splits:
        raise ValueError("at least one cell split is required")
    first = cell_splits[0]
    validation = [str(key) for key in first["val_keys"]]
    selection = [str(key) for key in first["rollout_keys"]]
    if validation != registered_validation:
        raise ValueError("gate validation order differs from the registered split")
    if len(selection) != 16 or len(set(selection)) != 16:
        raise ValueError("rollout-selection cohort must contain 16 unique keys")
    if not set(selection) <= set(validation):
        raise ValueError("rollout-selection keys leave open validation")
    for candidate in cell_splits[1:]:
        if [str(key) for key in candidate["val_keys"]] != validation or [
            str(key) for key in candidate["rollout_keys"]
        ] != selection:
            raise ValueError("schedule-gate cells do not share one validation cohort")
    outside = [key for key in validation if key not in set(selection)]
    if len(outside) != 28 or len(set(outside)) != 28:
        raise ValueError("outside-selection cohort must contain 28 unique keys")
    return validation, selection, outside


def _build_boundary_policies(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    if checkpoint.get("boundary_mode") != "causal_nodal_physical":
        raise ValueError("D094 holdout requires the trained causal boundary closure")
    contract = checkpoint.get("boundary_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("checkpoint lacks its causal boundary contract")
    expected_digests = contract.get("policy_digests")
    if not isinstance(expected_digests, Mapping):
        raise TypeError("checkpoint lacks boundary-policy digests")
    policies = {}
    metadata = {}
    for key in keys:
        policy, record = build_graph_causal_boundary_policy(
            store,
            key,
            device=device,
            max_source_hops=int(contract["max_source_hops"]),
            rho_inf=float(contract["rho_inf"]),
            p_inf=float(contract["p_inf"]),
        )
        if str(record["policy_digest"]) != str(expected_digests.get(key)):
            raise ValueError(f"boundary-policy digest changed for trajectory {key}")
        policies[key] = policy
        metadata[key] = record
    return policies, metadata


def _endpoint_means(structure: Mapping[str, Any], checkpoint: int) -> dict[str, Any]:
    endpoint = structure["endpoints"][str(checkpoint)]
    return {
        field: endpoint[field]["mean"]
        for field in (*STRUCTURE_ERROR_FIELDS, "front_iou", "front_symmetric_chamfer")
    }


def _evaluate_cell(
    descriptor: Mapping[str, Any],
    *,
    store: PCNOEuler2DShardStore,
    outside_keys: Sequence[str],
    policies: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    amp: str,
    shock_quantile: float,
) -> dict[str, Any]:
    checkpoint = load_bump_checkpoint(Path(descriptor["checkpoint"]))
    if checkpoint.get("data_manifest_digest") != store.manifest_digest:
        raise ValueError("checkpoint and holdout data manifests differ")
    source_snapshot = checkpoint.get("source_snapshot")
    if (
        not isinstance(source_snapshot, Mapping)
        or source_snapshot.get("source_set_digest")
        != EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
    ):
        raise ValueError("checkpoint source-set digest changed")
    if [str(key) for key in checkpoint["val_keys"]] != [
        str(key) for key in descriptor["split"]["val_keys"]
    ]:
        raise ValueError("checkpoint validation population differs from its run split")
    model = build_bump_checkpoint_model(checkpoint, device)
    start = perf_counter()
    rollout = evaluate_rollouts(
        model,
        store,
        outside_keys,
        step_stride=int(checkpoint["step_stride"]),
        start_frame=0,
        num_steps=79,
        device=device,
        amp=amp,
        boundary_policies=policies,
        rollout_checkpoints=ROLLOUT_CHECKPOINTS,
        failure_policy=FINITE_ONLY_ROLLOUT_POLICY,
    )
    structure = rollout_structure_diagnostics(
        model,
        store,
        outside_keys,
        policies,
        step_stride=int(checkpoint["step_stride"]),
        start_frame=0,
        num_steps=79,
        rollout_checkpoints=ROLLOUT_CHECKPOINTS,
        shock_quantile=shock_quantile,
        device=device,
        amp=amp,
        failure_policy=FINITE_ONLY_ROLLOUT_POLICY,
    )
    elapsed = perf_counter() - start
    result = {
        "run": descriptor["run"],
        "architecture": descriptor["architecture"],
        "schedule": descriptor["schedule"],
        "checkpoint_sha256": descriptor["checkpoint_sha256"],
        "selected_training_metrics": descriptor["selected_training_metrics"],
        "outside_selection_rollout": rollout,
        "outside_selection_structure": structure,
        "outside_selection_h79_structure_means": _endpoint_means(structure, 79),
        "elapsed_seconds": elapsed,
    }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def schedule_decision(
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    scores = {}
    for schedule in ("prefix_tail", "stretched"):
        scheduled = [
            cells[(architecture, schedule)] for architecture in ("pcno", "pcfno")
        ]
        complete = all(
            float(cell["outside_selection_rollout"]["completion_rate"]) == 1.0
            and int(cell["outside_selection_rollout"]["hard_failure_count"]) == 0
            for cell in scheduled
        )
        full_errors = [
            float(cell["outside_selection_rollout"]["mean_full_horizon_relative_l2"])
            for cell in scheduled
        ]
        h79_errors = [
            float(cell["outside_selection_rollout"]["mean_endpoint_relative_l2"]["79"])
            for cell in scheduled
        ]
        if any(
            value <= 0.0 or not math.isfinite(value)
            for value in (*full_errors, *h79_errors)
        ):
            raise ValueError(
                "schedule decision requires positive finite rollout errors"
            )
        scores[schedule] = {
            "numerically_complete_for_both_architectures": complete,
            "geometric_mean_all_call_relative_l2": float(
                np.exp(np.mean(np.log(full_errors)))
            ),
            "geometric_mean_h79_relative_l2": float(
                np.exp(np.mean(np.log(h79_errors)))
            ),
        }
    winner = min(
        scores,
        key=lambda schedule: (
            not scores[schedule]["numerically_complete_for_both_architectures"],
            scores[schedule]["geometric_mean_all_call_relative_l2"],
            scores[schedule]["geometric_mean_h79_relative_l2"],
        ),
    )
    ratios = {}
    for architecture in ("pcno", "pcfno"):
        stretched = cells[(architecture, "stretched")]["outside_selection_rollout"]
        prefix = cells[(architecture, "prefix_tail")]["outside_selection_rollout"]
        ratios[architecture] = {
            "all_call_mean_stretched_over_prefix_tail": float(
                stretched["mean_full_horizon_relative_l2"]
            )
            / float(prefix["mean_full_horizon_relative_l2"]),
            "h79_stretched_over_prefix_tail": float(
                stretched["mean_endpoint_relative_l2"]["79"]
            )
            / float(prefix["mean_endpoint_relative_l2"]["79"]),
        }
    return {
        "ranking_contract": (
            "numerical completion for both architectures, then geometric mean of "
            "PCNO/PCFNO all-call relative L2, then geometric mean H79; finite "
            "physical violations and structure diagnostics are reported, not ranked"
        ),
        "scores": scores,
        "winner": winner,
        "paired_stretched_over_prefix_tail_ratios": ratios,
        "authorizes_seed0_ladder_schedule": bool(
            scores[winner]["numerically_complete_for_both_architectures"]
        ),
    }


def _summary_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    rollout = cell["outside_selection_rollout"]
    structure = cell["outside_selection_structure"]
    return {
        "run": cell["run"],
        "architecture": cell["architecture"],
        "schedule": cell["schedule"],
        "selected_optimizer_step": cell["selected_training_metrics"]["optimizer_step"],
        "selection_rollout_all_call_mean_relative_l2": cell[
            "selected_training_metrics"
        ]["selection_rollout_all_call_mean_relative_l2"],
        "selection_rollout_h79_relative_l2": cell["selected_training_metrics"][
            "selection_rollout_h79_relative_l2"
        ],
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
        **{
            f"audit_structure_h79_{field}": value
            for field, value in cell["outside_selection_h79_structure_means"].items()
        },
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("holdout output directory must not already exist")
    cells = _discover_cells(args.gate_root)
    split_manifest = _load_json(args.split_manifest)
    validation_keys, selection_keys, outside_keys = outside_selection_keys(
        split_manifest, [cells[key]["split"] for key in CELL_ORDER]
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
        reference_checkpoint = load_bump_checkpoint(cells[CELL_ORDER[0]]["checkpoint"])
        policies, policy_metadata = _build_boundary_policies(
            store, outside_keys, reference_checkpoint, device
        )
        results = []
        for identity in CELL_ORDER:
            result = _evaluate_cell(
                cells[identity],
                store=store,
                outside_keys=outside_keys,
                policies=policies,
                device=device,
                amp=args.amp,
                shock_quantile=args.shock_quantile,
            )
            results.append(result)
            atomic_write_json(
                args.output_dir / f"{identity[0]}_{identity[1]}.json", result
            )
    finally:
        store.close()
    result_map = {(row["architecture"], row["schedule"]): row for row in results}
    decision = schedule_decision(result_map)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": "single_seed_n256_outside_selection_schedule_audit",
        "historical_test_population_accessed": False,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "checkpoint_source_set_digest": EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        "evaluator_source_snapshot": source_snapshot,
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
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
        "schedule_decision": decision,
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "checkpoint reselection on the 28 audit trajectories",
            "independent test performance",
            "multi-seed architecture interaction",
            "physical conservation from reconstructed proxy weights",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_csv(
        args.output_dir / "cell_summary.csv", [_summary_row(row) for row in results]
    )
    artifact_files = [
        "summary.json",
        "cell_summary.csv",
        *(f"{architecture}_{schedule}.json" for architecture, schedule in CELL_ORDER),
    ]
    artifact_manifest = {
        "schema": "d094_bump_schedule_outside_selection_artifacts_v1",
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
                "schedule_decision": summary["schedule_decision"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
