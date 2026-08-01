#!/usr/bin/env python3
"""Compare native, causal, and minimum-change boundary recurrences.

This is a validation-only protocol-selection evaluator. It never selects or
opens the manifest test split. Final test reporting remains the responsibility
of the official rollout evaluator after a protocol is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    sha256_file,
    write_json,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    balanced_presentations,
    build_graph_causal_boundary_policy,
    build_graph_minimum_change_boundary_policy,
)
from utility.time_dependent_no.pcno_rollout import (
    CAUSAL_BOUNDARY_MODE,
    LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    MINIMUM_CHANGE_BOUNDARY_MODE,
    NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    RAW_ALL_NODES_PRIMARY_OBJECTIVE,
    STRUCTURE_ERROR_FIELDS,
    build_bump_checkpoint_model as build_model,
    evaluate_pairs,
    evaluate_rollouts,
    load_bump_checkpoint as load_checkpoint,
    projection_decomposition,
    rollout_structure_diagnostics,
)
from utility.time_dependent_no.pcno_runtime import select_device

SCHEMA = "pcno_euler2d_boundary_protocol_validation_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--one-step-presentations", type=int, default=128)
    parser.add_argument("--presentation-seed", type=int, default=20260729)
    parser.add_argument("--rollout-count", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--rollout-checkpoints", type=int, nargs="+", default=(1, 5, 10, 20)
    )
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--boundary-rho-inf", type=float, default=1.4)
    parser.add_argument("--boundary-p-inf", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.one_step_presentations < 1 or args.rollout_count < 1:
        raise ValueError("presentation and rollout counts must be positive")
    if args.rollout_steps < 1 or args.start_frame < 0:
        raise ValueError("rollout steps must be positive and start frame nonnegative")
    checkpoints = sorted({int(value) for value in args.rollout_checkpoints})
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > args.rollout_steps:
        raise ValueError("rollout checkpoints must lie inside the requested horizon")
    args.rollout_checkpoints = checkpoints
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie inside (0, 1)")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")


def _policy_set(
    mode: str,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    device: torch.device,
    rho_inf: float,
    p_inf: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if mode == "model_all_nodes":
        return {}, {}
    policies: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    contract = checkpoint.get("boundary_contract", {})
    for key in keys:
        if mode == CAUSAL_BOUNDARY_MODE:
            policy, record = build_graph_causal_boundary_policy(
                store,
                key,
                device=device,
                max_source_hops=int(contract.get("max_source_hops", 3)),
                rho_inf=float(contract.get("rho_inf", rho_inf)),
                p_inf=float(contract.get("p_inf", p_inf)),
            )
        elif mode == MINIMUM_CHANGE_BOUNDARY_MODE:
            policy, record = build_graph_minimum_change_boundary_policy(
                store,
                key,
                device=device,
                rho_inf=float(contract.get("rho_inf", rho_inf)),
                p_inf=float(contract.get("p_inf", p_inf)),
            )
        else:
            raise ValueError(f"unsupported boundary mode: {mode}")
        expected = contract.get("policy_digests", {}).get(key)
        if expected is not None and record["policy_digest"] != expected:
            raise ValueError(f"trajectory {key} native boundary policy digest changed")
        policies[key] = policy
        metadata[key] = record
    return policies, metadata


def _paired_structure_ratios(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    ratio_name: str,
    rollout_checkpoints: Sequence[int],
) -> dict[str, Any]:
    baseline_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row
        for row in baseline["rows"]
    }
    candidate_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row
        for row in candidate["rows"]
    }
    result: dict[str, Any] = {}
    for checkpoint in rollout_checkpoints:
        endpoint: dict[str, Any] = {}
        for field in STRUCTURE_ERROR_FIELDS:
            ratios = []
            for key, baseline_row in baseline_rows.items():
                if key[1] != checkpoint or key not in candidate_rows:
                    continue
                baseline_value = baseline_row.get(field)
                candidate_value = candidate_rows[key].get(field)
                if (
                    baseline_value is not None
                    and candidate_value is not None
                    and float(baseline_value) > 0.0
                ):
                    ratios.append(float(candidate_value) / float(baseline_value))
            endpoint[field] = {
                "count": len(ratios),
                "mean_ratio": None if not ratios else float(np.mean(ratios)),
                "median_ratio": None if not ratios else float(np.median(ratios)),
            }
        result[str(checkpoint)] = endpoint
    return {
        "ratio": ratio_name,
        "direction": "less_than_or_equal_to_one_is_nonworse",
        "paired_by": ["trajectory", "call_index"],
        "endpoints": result,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    validate_args(args, device)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = load_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir)
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and data manifest differ")
    if (
        args.expected_manifest_sha256
        and args.expected_manifest_sha256 != store.manifest_digest
    ):
        raise ValueError("manifest SHA-256 mismatch")
    validation_keys = [str(key) for key in checkpoint["val_keys"]]
    if not validation_keys or any(key not in store.keys for key in validation_keys):
        raise ValueError(
            "checkpoint validation split is unavailable in the shard store"
        )
    if args.rollout_count > len(validation_keys):
        raise ValueError("rollout count exceeds the checkpoint validation split")
    model = build_model(checkpoint, device)
    step_stride = int(checkpoint["step_stride"])
    pairs = balanced_presentations(
        store,
        validation_keys,
        step_stride=step_stride,
        count=args.one_step_presentations,
        rng=np.random.default_rng(args.presentation_seed),
    )
    rollout_keys = validation_keys[: args.rollout_count]

    native_mode = str(checkpoint["boundary_mode"])
    native_policies, native_metadata = _policy_set(
        native_mode,
        checkpoint,
        store,
        validation_keys,
        device=device,
        rho_inf=args.boundary_rho_inf,
        p_inf=args.boundary_p_inf,
    )
    causal_policies, causal_metadata = _policy_set(
        CAUSAL_BOUNDARY_MODE,
        {},
        store,
        validation_keys,
        device=device,
        rho_inf=args.boundary_rho_inf,
        p_inf=args.boundary_p_inf,
    )
    minimum_policies, minimum_metadata = _policy_set(
        MINIMUM_CHANGE_BOUNDARY_MODE,
        {},
        store,
        validation_keys,
        device=device,
        rho_inf=args.boundary_rho_inf,
        p_inf=args.boundary_p_inf,
    )
    saved_primary = str(
        checkpoint.get("training_args", {}).get(
            "primary_objective", NORMAL_CLOSED_PRIMARY_OBJECTIVE
        )
    )
    if saved_primary not in {
        NORMAL_CLOSED_PRIMARY_OBJECTIVE,
        RAW_ALL_NODES_PRIMARY_OBJECTIVE,
        LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    }:
        raise ValueError(f"unsupported checkpoint primary objective: {saved_primary}")
    native_one_step = evaluate_pairs(
        model,
        store,
        pairs,
        step_stride=step_stride,
        batch_size=1,
        device=device,
        amp=args.amp,
        primary_objective=saved_primary,
        boundary_policies=native_policies,
    )
    causal_one_step = evaluate_pairs(
        model,
        store,
        pairs,
        step_stride=step_stride,
        batch_size=1,
        device=device,
        amp=args.amp,
        primary_objective=NORMAL_CLOSED_PRIMARY_OBJECTIVE,
        boundary_policies=causal_policies,
    )
    minimum_one_step = evaluate_pairs(
        model,
        store,
        pairs,
        step_stride=step_stride,
        batch_size=1,
        device=device,
        amp=args.amp,
        primary_objective=LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
        boundary_policies=minimum_policies,
    )
    native_rollout = evaluate_rollouts(
        model,
        store,
        rollout_keys,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
        boundary_policies=native_policies,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    causal_rollout = evaluate_rollouts(
        model,
        store,
        rollout_keys,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
        boundary_policies=causal_policies,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    minimum_rollout = evaluate_rollouts(
        model,
        store,
        rollout_keys,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        device=device,
        amp=args.amp,
        boundary_policies=minimum_policies,
        rollout_checkpoints=args.rollout_checkpoints,
    )
    native_structure = rollout_structure_diagnostics(
        model,
        store,
        rollout_keys,
        native_policies,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        rollout_checkpoints=args.rollout_checkpoints,
        shock_quantile=args.shock_quantile,
        device=device,
        amp=args.amp,
    )
    causal_structure = rollout_structure_diagnostics(
        model,
        store,
        rollout_keys,
        causal_policies,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        rollout_checkpoints=args.rollout_checkpoints,
        shock_quantile=args.shock_quantile,
        device=device,
        amp=args.amp,
    )
    minimum_structure = rollout_structure_diagnostics(
        model,
        store,
        rollout_keys,
        minimum_policies,
        step_stride=step_stride,
        start_frame=args.start_frame,
        num_steps=args.rollout_steps,
        rollout_checkpoints=args.rollout_checkpoints,
        shock_quantile=args.shock_quantile,
        device=device,
        amp=args.amp,
    )
    structure_ratios = _paired_structure_ratios(
        native_structure,
        minimum_structure,
        ratio_name="minimum_change_over_native",
        rollout_checkpoints=args.rollout_checkpoints,
    )
    causal_structure_ratios = _paired_structure_ratios(
        native_structure,
        causal_structure,
        ratio_name="causal_over_native",
        rollout_checkpoints=args.rollout_checkpoints,
    )
    causal_decomposition = projection_decomposition(
        model,
        store,
        pairs,
        causal_policies,
        step_stride=step_stride,
        device=device,
        amp=args.amp,
    )
    minimum_decomposition = projection_decomposition(
        model,
        store,
        pairs,
        minimum_policies,
        step_stride=step_stride,
        device=device,
        amp=args.amp,
    )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "utility/time_dependent_no/pcno_artifacts.py",
        ROOT / "utility/time_dependent_no/cpg_mesh_contract.py",
        ROOT / "utility/time_dependent_no/euler2d_metrics.py",
        ROOT / "utility/time_dependent_no/pcno_euler2d.py",
        ROOT / "utility/time_dependent_no/pcno_ripple_diagnostics.py",
        ROOT / "utility/time_dependent_no/pcno_rollout.py",
        ROOT / "utility/time_dependent_no/pcno_runtime.py",
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "selection_population": "checkpoint_validation_split_only",
        "test_split_opened": False,
        "checkpoint": {
            "path_name": args.checkpoint.name,
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "boundary_mode": native_mode,
            "primary_objective": saved_primary,
            "data_manifest_digest": checkpoint["data_manifest_digest"],
        },
        "evaluation": {
            "validation_keys": validation_keys,
            "one_step_pairs": [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in pairs
            ],
            "presentation_seed": args.presentation_seed,
            "rollout_keys": rollout_keys,
            "start_frame": args.start_frame,
            "rollout_steps": args.rollout_steps,
            "rollout_checkpoints": args.rollout_checkpoints,
            "shock_quantile": args.shock_quantile,
            "step_stride": step_stride,
            "device": str(device),
            "amp": args.amp,
        },
        "native": {
            "one_step": native_one_step,
            "rollout": native_rollout,
            "structure": native_structure,
            "policy_metadata": native_metadata,
        },
        "causal": {
            "one_step": causal_one_step,
            "rollout": causal_rollout,
            "structure": causal_structure,
            "projection_decomposition": causal_decomposition,
            "policy_metadata": causal_metadata,
        },
        "minimum_change": {
            "one_step": minimum_one_step,
            "rollout": minimum_rollout,
            "structure": minimum_structure,
            "projection_decomposition": minimum_decomposition,
            "policy_metadata": minimum_metadata,
        },
        "paired_structure_ratios": structure_ratios,
        "paired_causal_structure_ratios": causal_structure_ratios,
        "claim_boundary": {
            "verified": "frozen-checkpoint validation behavior under three matched recurrences",
            "inference": "whether either hard-boundary recurrence improves the frozen checkpoint",
            "not_supported": [
                "exact DG boundary replay",
                "physical conservation",
                "test or distribution-shift performance",
                "causal attribution to training before a matched checkpoint exists",
            ],
        },
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "native_horizon": native_rollout["mean_endpoint_relative_l2"],
                "causal_horizon": causal_rollout["mean_endpoint_relative_l2"],
                "minimum_change_horizon": minimum_rollout["mean_endpoint_relative_l2"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    store.close()


if __name__ == "__main__":
    main()
