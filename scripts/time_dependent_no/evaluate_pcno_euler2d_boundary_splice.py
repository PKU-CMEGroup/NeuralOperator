#!/usr/bin/env python3
"""Localize a checkpoint difference to boundary or normal-node output.

Both frozen PCNOs receive the same minimum-change-projected recurrent state.
One hybrid uses the parent raw proposal on normal nodes and the teacher raw
proposal on boundary nodes; the reverse hybrid swaps those roles. The shared
hard projection is applied after splicing, so prescribed boundary coordinates
remain prescribed and only legal free boundary state can differ.

This is a validation-only, no-training mechanism diagnostic. Each hybrid costs
two learned calls per step and is not itself a deployable one-call model.
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

from scripts.time_dependent_no.evaluate_pcno_euler2d_boundary_protocol import (  # noqa: E402
    STRUCTURE_ERROR_FIELDS,
    projection_decomposition,
    rollout_structure_diagnostics,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (  # noqa: E402
    build_model,
    load_checkpoint,
    sha256_file,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (  # noqa: E402
    LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    evaluate_pairs,
    evaluate_rollouts,
    select_device,
    write_json,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    balanced_presentations,
    build_graph_minimum_change_boundary_policy,
    digest_mapping,
    normal_node_mask,
)

SCHEMA = "pcno_euler2d_boundary_output_splice_validation_v2"
BOUNDARY_FROM_TEACHER = "parent_normal_teacher_boundary"
NORMAL_FROM_TEACHER = "teacher_normal_parent_boundary"
VARIANTS = (BOUNDARY_FROM_TEACHER, NORMAL_FROM_TEACHER)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-sha256")
    parser.add_argument("--expected-teacher-checkpoint-sha256")
    parser.add_argument("--expected-reference-summary-sha256")
    parser.add_argument("--expected-teacher-summary-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--one-step-presentations", type=int, default=128)
    parser.add_argument("--presentation-seed", type=int, default=20260729)
    parser.add_argument("--rollout-count", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--rollout-checkpoints", type=int, nargs="+", default=(1, 5, 10, 20)
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
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
    for path in (
        args.data_dir,
        args.parent_checkpoint,
        args.teacher_checkpoint,
        args.reference_summary,
        args.teacher_summary,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.one_step_presentations < 1 or args.rollout_count < 1:
        raise ValueError("presentation and rollout counts must be positive")
    if args.rollout_steps < 1:
        raise ValueError("rollout steps must be positive")
    if args.start_frame != 0:
        raise ValueError(
            "boundary-protocol reference summaries are bound to start frame 0"
        )
    checkpoints = sorted({int(value) for value in args.rollout_checkpoints})
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > args.rollout_steps:
        raise ValueError("rollout checkpoints must lie inside the requested horizon")
    args.rollout_checkpoints = checkpoints
    args.variants = list(dict.fromkeys(args.variants))
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie inside (0, 1)")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")


def splice_raw_predictions(
    parent_raw: torch.Tensor,
    teacher_raw: torch.Tensor,
    node_type: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    """Splice whole nodal states before applying the shared hard projection."""

    if parent_raw.shape != teacher_raw.shape:
        raise ValueError("parent and teacher proposal shapes differ")
    if parent_raw.ndim != 3 or parent_raw.shape[-1] != 4:
        raise ValueError("proposals must have shape [B,N,4]")
    if node_mask.shape != (*parent_raw.shape[:2], 1):
        raise ValueError("node mask does not align with proposals")
    normal = normal_node_mask(node_type, node_mask)
    boundary = node_mask - normal
    if variant == BOUNDARY_FROM_TEACHER:
        result = parent_raw * normal + teacher_raw * boundary
    elif variant == NORMAL_FROM_TEACHER:
        result = teacher_raw * normal + parent_raw * boundary
    else:
        raise ValueError(f"unsupported splice variant: {variant}")
    return result * node_mask


class BoundarySplicePCNO(torch.nn.Module):
    """Two frozen PCNO calls presented through the ordinary evaluator API."""

    def __init__(
        self,
        parent: PCNOEuler2DResidual,
        teacher: PCNOEuler2DResidual,
        *,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unsupported splice variant: {variant}")
        if parent.gamma != teacher.gamma:
            raise ValueError("parent and teacher gamma differ")
        for name in (
            "state_mean",
            "state_scale",
            "residual_scale",
            "mach_mean",
            "mach_scale",
        ):
            if not torch.equal(getattr(parent, name), getattr(teacher, name)):
                raise ValueError(f"parent and teacher {name} buffers differ")
        self.parent = parent
        self.teacher = teacher
        self.variant = variant
        self.requires_grad_(False)
        self.eval()

    @property
    def gamma(self) -> float:
        return float(self.parent.gamma)

    @property
    def state_scale(self) -> torch.Tensor:
        return self.parent.state_scale

    def forward(
        self,
        current: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        nodes: torch.Tensor,
        node_weights: torch.Tensor,
        node_rhos: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        node_type: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        arguments = {
            "node_mask": node_mask,
            "nodes": nodes,
            "node_weights": node_weights,
            "node_rhos": node_rhos,
            "directed_edges": directed_edges,
            "edge_gradient_weights": edge_gradient_weights,
            "node_type": node_type,
            "mach": mach,
        }
        parent_raw = self.parent(current, **arguments)
        teacher_raw = self.teacher(current, **arguments)
        return splice_raw_predictions(
            parent_raw,
            teacher_raw,
            node_type,
            node_mask,
            variant=self.variant,
        )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def require_digest(path: Path, expected: str | None, *, name: str) -> str:
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return observed


def verified_normalization_digest(checkpoint: Mapping[str, Any], *, name: str) -> str:
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError(f"{name} lacks normalization metadata")
    observed = digest_mapping(normalization)
    declared = checkpoint.get("normalization_digest")
    if declared is not None and declared != observed:
        raise ValueError(f"{name} normalization digest does not match its payload")
    return observed


def gain_fraction(parent: float, teacher: float, hybrid: float) -> float | None:
    improvement = float(parent) - float(teacher)
    if improvement <= 0.0:
        return None
    return (float(parent) - float(hybrid)) / improvement


def paired_structure_ratios(
    reference: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    reference_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row
        for row in reference["rows"]
    }
    hybrid_rows = {
        (str(row["trajectory"]), int(row["call_index"])): row for row in hybrid["rows"]
    }
    endpoints: dict[str, Any] = {}
    for checkpoint in checkpoints:
        endpoint: dict[str, Any] = {}
        for field in STRUCTURE_ERROR_FIELDS:
            ratios = []
            wins = 0
            for key, baseline_row in reference_rows.items():
                if key[1] != checkpoint or key not in hybrid_rows:
                    continue
                baseline = baseline_row.get(field)
                value = hybrid_rows[key].get(field)
                if baseline is None or value is None or float(baseline) <= 0.0:
                    continue
                ratios.append(float(value) / float(baseline))
                wins += int(float(value) < float(baseline))
            endpoint[field] = {
                "count": len(ratios),
                "wins": wins,
                "mean_ratio": None if not ratios else float(np.mean(ratios)),
                "median_ratio": None if not ratios else float(np.median(ratios)),
            }
        endpoints[str(checkpoint)] = endpoint
    return {
        "ratio": "hybrid_over_frozen_parent",
        "direction": "less_than_or_equal_to_one_is_nonworse",
        "paired_by": ["trajectory", "call_index"],
        "endpoints": endpoints,
    }


def compare_variant(
    one_step: Mapping[str, Any],
    rollout: Mapping[str, Any],
    structure: Mapping[str, Any],
    reference: Mapping[str, Any],
    teacher: Mapping[str, Any],
    *,
    checkpoints: Sequence[int],
    final_checkpoint: int,
) -> dict[str, Any]:
    endpoint = str(final_checkpoint)
    reference_rollout = reference["rollout"]
    teacher_rollout = teacher["rollout"]

    def endpoint_record(region: str) -> dict[str, float | None]:
        field = (
            "mean_endpoint_relative_l2"
            if region == "all"
            else "mean_endpoint_normal_relative_l2"
        )
        parent_value = float(reference_rollout[field][endpoint])
        teacher_value = float(teacher_rollout[field][endpoint])
        hybrid_value = float(rollout[field][endpoint])
        return {
            "parent": parent_value,
            "teacher": teacher_value,
            "hybrid": hybrid_value,
            "hybrid_over_parent": hybrid_value / parent_value,
            "teacher_parent_gain_fraction_retained": gain_fraction(
                parent_value, teacher_value, hybrid_value
            ),
        }

    return {
        "final_checkpoint": final_checkpoint,
        "one_step_ratio_to_parent": {
            region: (
                float(one_step[f"{region}_relative_l2"])
                / float(reference["one_step"][f"{region}_relative_l2"])
            )
            for region in ("all", "normal", "boundary")
        },
        "endpoint": {
            "all": endpoint_record("all"),
            "normal": endpoint_record("normal"),
        },
        "structure_ratios": paired_structure_ratios(
            reference["structure"], structure, checkpoints
        ),
    }


def boundary_h20_advance_gate(
    comparison: Mapping[str, Any], rollout: Mapping[str, Any]
) -> dict[str, Any]:
    if int(comparison["final_checkpoint"]) != 20:
        raise ValueError("the adapter-routing gate is defined only at H20")
    endpoint = comparison["structure_ratios"]["endpoints"]["20"]
    all_fraction = comparison["endpoint"]["all"][
        "teacher_parent_gain_fraction_retained"
    ]
    normal_fraction = comparison["endpoint"]["normal"][
        "teacher_parent_gain_fraction_retained"
    ]
    outflow_values = [
        float(row["minimum_outflow_normal_mach"])
        for row in rollout["trajectories"]
        if row["minimum_outflow_normal_mach"] is not None
    ]
    checks = {
        "completion_30_of_30": (
            int(rollout["num_trajectories"]) == 30 and int(rollout["completed"]) == 30
        ),
        "outflow_remains_supersonic": (
            len(outflow_values) == int(rollout["num_trajectories"])
            and min(outflow_values) > 1.0
        ),
        "all_state_gain_fraction_at_least_half": (
            all_fraction is not None and float(all_fraction) >= 0.5
        ),
        "normal_state_gain_fraction_at_least_half": (
            normal_fraction is not None and float(normal_fraction) >= 0.5
        ),
        **{
            f"{field}_paired_count_30": int(endpoint[field]["count"]) == 30
            for field in STRUCTURE_ERROR_FIELDS
        },
        **{
            f"{field}_median_ratio_at_most_1p05": (
                endpoint[field]["median_ratio"] is not None
                and float(endpoint[field]["median_ratio"]) <= 1.05
            )
            for field in STRUCTURE_ERROR_FIELDS
        },
    }
    return {
        "schema": "pcno_euler2d_boundary_splice_h20_gate_v2",
        "checks": checks,
        "passed": all(checks.values()),
        "if_passed": "evaluate only the boundary-from-teacher hybrid at H79",
        "if_failed": "do not train a boundary adapter from this checkpoint pair",
    }


def validate_summary_contract(
    summary: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    validation_keys: Sequence[str],
    rollout_keys: Sequence[str],
    pairs: Sequence[tuple[str, int]],
    step_stride: int,
    args: argparse.Namespace,
) -> None:
    if summary.get("schema") != "pcno_euler2d_boundary_protocol_validation_v1":
        raise ValueError("unsupported reference boundary-summary schema")
    if summary.get("status") != "complete" or summary.get("test_split_opened"):
        raise ValueError("reference summary is incomplete or opened the test split")
    if summary["checkpoint"]["sha256"] != checkpoint_sha256:
        raise ValueError("reference summary binds a different checkpoint")
    evaluation = summary["evaluation"]
    if [str(value) for value in evaluation["validation_keys"]] != list(validation_keys):
        raise ValueError("reference summary validation split differs")
    observed_rollout = [
        str(value) for value in evaluation["rollout_keys"][: args.rollout_count]
    ]
    if observed_rollout != list(rollout_keys):
        raise ValueError("reference summary rollout population differs")
    expected_pairs = [
        {"trajectory": key, "time_index": int(time_index)} for key, time_index in pairs
    ]
    if evaluation["one_step_pairs"] != expected_pairs:
        raise ValueError("reference summary one-step pair stream differs")
    if int(evaluation["step_stride"]) != step_stride:
        raise ValueError("reference summary step stride differs")
    if int(evaluation["rollout_steps"]) < args.rollout_steps:
        raise ValueError("reference summary does not reach the requested horizon")
    if not set(args.rollout_checkpoints).issubset(evaluation["rollout_checkpoints"]):
        raise ValueError("reference summary lacks requested checkpoints")
    if float(evaluation["shock_quantile"]) != args.shock_quantile:
        raise ValueError("reference summary shock quantile differs")
    if str(evaluation["amp"]) != args.amp:
        raise ValueError("reference summary precision differs")


def validate_policy_digests(
    summary: Mapping[str, Any],
    observed: Mapping[str, str],
    *,
    name: str,
) -> None:
    metadata = summary.get("minimum_change", {}).get("policy_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{name} lacks minimum-change policy metadata")
    expected = {
        str(key): str(record["policy_digest"])
        for key, record in metadata.items()
        if isinstance(record, Mapping) and "policy_digest" in record
    }
    if expected.keys() != observed.keys():
        raise ValueError(f"{name} minimum-change policy population differs")
    for key, digest in observed.items():
        if expected[key] != digest:
            raise ValueError(
                f"{name} minimum-change policy digest differs for trajectory {key}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    validate_args(args, device)
    parent_sha256 = require_digest(
        args.parent_checkpoint,
        args.expected_parent_checkpoint_sha256,
        name="parent checkpoint",
    )
    teacher_sha256 = require_digest(
        args.teacher_checkpoint,
        args.expected_teacher_checkpoint_sha256,
        name="teacher checkpoint",
    )
    reference_summary_sha256 = require_digest(
        args.reference_summary,
        args.expected_reference_summary_sha256,
        name="reference summary",
    )
    teacher_summary_sha256 = require_digest(
        args.teacher_summary,
        args.expected_teacher_summary_sha256,
        name="teacher summary",
    )
    parent_checkpoint = load_checkpoint(args.parent_checkpoint)
    teacher_checkpoint = load_checkpoint(args.teacher_checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir)
    if args.expected_manifest_sha256 and (
        store.manifest_digest != args.expected_manifest_sha256
    ):
        raise ValueError("data manifest SHA-256 mismatch")
    for checkpoint in (parent_checkpoint, teacher_checkpoint):
        if checkpoint["data_manifest_digest"] != store.manifest_digest:
            raise ValueError("checkpoint and data manifest differ")
    validation_keys = [str(key) for key in parent_checkpoint["val_keys"]]
    if validation_keys != [str(key) for key in teacher_checkpoint["val_keys"]]:
        raise ValueError("parent and teacher validation splits differ")
    if int(parent_checkpoint["step_stride"]) != int(teacher_checkpoint["step_stride"]):
        raise ValueError("parent and teacher step strides differ")
    parent_normalization_digest = verified_normalization_digest(
        parent_checkpoint, name="parent checkpoint"
    )
    teacher_normalization_digest = verified_normalization_digest(
        teacher_checkpoint, name="teacher checkpoint"
    )
    if parent_normalization_digest != teacher_normalization_digest:
        raise ValueError("parent and teacher normalization digests differ")
    if args.rollout_count > len(validation_keys):
        raise ValueError("rollout count exceeds the validation split")
    rollout_keys = validation_keys[: args.rollout_count]
    step_stride = int(parent_checkpoint["step_stride"])
    pairs = balanced_presentations(
        store,
        validation_keys,
        step_stride=step_stride,
        count=args.one_step_presentations,
        rng=np.random.default_rng(args.presentation_seed),
    )
    reference_summary = read_json(args.reference_summary)
    teacher_summary = read_json(args.teacher_summary)
    validate_summary_contract(
        reference_summary,
        checkpoint_sha256=parent_sha256,
        validation_keys=validation_keys,
        rollout_keys=rollout_keys,
        pairs=pairs,
        step_stride=step_stride,
        args=args,
    )
    validate_summary_contract(
        teacher_summary,
        checkpoint_sha256=teacher_sha256,
        validation_keys=validation_keys,
        rollout_keys=rollout_keys,
        pairs=pairs,
        step_stride=step_stride,
        args=args,
    )
    parent = build_model(parent_checkpoint, device)
    teacher = build_model(teacher_checkpoint, device)
    parent_parameters = sum(parameter.numel() for parameter in parent.parameters())
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    if parent_parameters != teacher_parameters:
        raise ValueError("parent and teacher parameter counts differ")
    policies: dict[str, dict[str, Any]] = {}
    policy_digests: dict[str, str] = {}
    for key in validation_keys:
        policy, record = build_graph_minimum_change_boundary_policy(
            store,
            key,
            device=device,
            rho_inf=args.boundary_rho_inf,
            p_inf=args.boundary_p_inf,
        )
        policies[key] = policy
        policy_digests[key] = record["policy_digest"]
    validate_policy_digests(
        reference_summary,
        policy_digests,
        name="parent reference summary",
    )
    validate_policy_digests(
        teacher_summary,
        policy_digests,
        name="teacher reference summary",
    )
    reference = reference_summary["minimum_change"]
    teacher_reference = teacher_summary["minimum_change"]
    variants: dict[str, Any] = {}
    for variant in args.variants:
        hybrid = BoundarySplicePCNO(parent, teacher, variant=variant).to(device)
        one_step = evaluate_pairs(
            hybrid,
            store,
            pairs,
            step_stride=step_stride,
            batch_size=1,
            device=device,
            amp=args.amp,
            primary_objective=LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
            boundary_policies=policies,
        )
        rollout = evaluate_rollouts(
            hybrid,
            store,
            rollout_keys,
            step_stride=step_stride,
            start_frame=args.start_frame,
            num_steps=args.rollout_steps,
            device=device,
            amp=args.amp,
            boundary_policies=policies,
            rollout_checkpoints=args.rollout_checkpoints,
        )
        structure = rollout_structure_diagnostics(
            hybrid,
            store,
            rollout_keys,
            policies,
            step_stride=step_stride,
            start_frame=args.start_frame,
            num_steps=args.rollout_steps,
            rollout_checkpoints=args.rollout_checkpoints,
            shock_quantile=args.shock_quantile,
            device=device,
            amp=args.amp,
        )
        decomposition = projection_decomposition(
            hybrid,
            store,
            pairs,
            policies,
            step_stride=step_stride,
            device=device,
            amp=args.amp,
        )
        comparison = compare_variant(
            one_step,
            rollout,
            structure,
            reference,
            teacher_reference,
            checkpoints=args.rollout_checkpoints,
            final_checkpoint=args.rollout_steps,
        )
        state_rollout_calls = 2 * sum(
            int(row["valid_length"]) for row in rollout["trajectories"]
        )
        structure_calls = 2 * sum(
            int(row["valid_length"]) for row in structure["trajectories"]
        )
        variants[variant] = {
            "one_step": one_step,
            "rollout": rollout,
            "structure": structure,
            "projection_decomposition": decomposition,
            "comparison": comparison,
            "learned_call_accounting": {
                "one_step": 2 * len(pairs),
                "state_rollout": state_rollout_calls,
                "structure_rollout": structure_calls,
                "projection_decomposition": 2 * len(pairs),
                "total": (4 * len(pairs) + state_rollout_calls + structure_calls),
            },
        }
    h20_gate = None
    if args.rollout_steps == 20 and BOUNDARY_FROM_TEACHER in variants:
        h20_gate = boundary_h20_advance_gate(
            variants[BOUNDARY_FROM_TEACHER]["comparison"],
            variants[BOUNDARY_FROM_TEACHER]["rollout"],
        )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_boundary_protocol.py",
        ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
        ROOT / "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
        ROOT / "utility/time_dependent_no/pcno_euler2d.py",
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "selection_population": "checkpoint_validation_split_only",
        "test_split_opened": False,
        "training_performed": False,
        "checkpoints": {
            "parent": {
                "path_name": args.parent_checkpoint.name,
                "sha256": parent_sha256,
                "parameter_count": parent_parameters,
            },
            "teacher": {
                "path_name": args.teacher_checkpoint.name,
                "sha256": teacher_sha256,
                "parameter_count": teacher_parameters,
            },
        },
        "reference_summaries": {
            "parent_sha256": reference_summary_sha256,
            "teacher_sha256": teacher_summary_sha256,
            "branch": "minimum_change",
        },
        "splice_contract": {
            "hard_projection": "minimum_change_nodal_physical",
            "model_input": "same_projected_recurrent_state_for_both_models",
            "boundary_from_teacher": (
                "parent raw whole-state proposal on normal nodes and teacher "
                "raw whole-state proposal on boundary nodes"
            ),
            "normal_from_teacher": (
                "teacher raw whole-state proposal on normal nodes and parent "
                "raw whole-state proposal on boundary nodes"
            ),
            "hard_projection_after_splice": True,
            "learned_calls_per_hybrid_step": 2,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "physical_conservation_claim": False,
            "exact_dg_boundary_replay": False,
        },
        "evaluation": {
            "validation_keys": validation_keys,
            "rollout_keys": rollout_keys,
            "one_step_pairs": [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in pairs
            ],
            "presentation_seed": args.presentation_seed,
            "start_frame": args.start_frame,
            "rollout_steps": args.rollout_steps,
            "rollout_checkpoints": args.rollout_checkpoints,
            "shock_quantile": args.shock_quantile,
            "step_stride": step_stride,
            "device": str(device),
            "amp": args.amp,
            "variants": args.variants,
        },
        "data_manifest_digest": store.manifest_digest,
        "normalization_digest": parent_normalization_digest,
        "policy_digests": policy_digests,
        "variants": variants,
        "boundary_h20_advance_gate": h20_gate,
        "claim_boundary": {
            "verified": (
                "which frozen checkpoint output population carries validation "
                "state and structure differences under a shared hard recurrence"
            ),
            "not_supported": [
                "a deployable one-call adapter",
                "parameter efficiency",
                "exact DG replay",
                "physical conservation",
                "test or family transfer",
            ],
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "boundary_h20_advance": (
                    None if h20_gate is None else h20_gate["passed"]
                ),
                "variant_endpoints": {
                    name: value["rollout"]["mean_endpoint_relative_l2"]
                    for name, value in variants.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    store.close()


if __name__ == "__main__":
    main()
