#!/usr/bin/env python3
"""Train and rollout-select a conservative-residual PCNO on 2D Euler shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.cpg_mesh_contract import NORMAL_NODE
from utility.time_dependent_no.pcno_artifacts import (
    atomic_torch_save,
    digest_array,
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    verify_source_snapshot,
    write_json,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_euler2d import (
    BOUNDARY_FIELD_MODES,
    BOUNDARY_FIELD_NONE,
    BOUNDARY_FIELD_SEMANTIC_COLLAR,
    BOUNDARY_RESIDUAL_MODES,
    BOUNDARY_RESIDUAL_NONE,
    BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE,
    NODE_TYPE_FEATURE_CONSTANT_ZERO,
    NODE_TYPE_FEATURE_OMITTED,
    NODE_TYPE_FEATURE_ONE_HOT,
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_admissible_primitive_noise,
    balanced_presentations,
    boundary_band_normal_node_mask,
    build_graph_causal_boundary_policy,
    build_graph_minimum_change_boundary_policy,
    conservative_admissibility,
    conservative_to_primitive_torch,
    copy_no_boundary_initialization_to_boundary_field_model,
    copy_no_boundary_initialization_to_boundary_residual_model,
    copy_no_type_initialization_to_zero_channels,
    digest_mapping,
    fit_normalization,
    fixed_tiny_presentations,
    full_coverage_presentations,
    homogeneous_optimizer_step_count,
    homogeneous_presentation_batches,
    normal_node_mask,
    parameter_count,
    stratified_train_val_split,
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.pcno_rollout import (
    CAUSAL_BOUNDARY_MODE,
    LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
    MINIMUM_CHANGE_BOUNDARY_MODE,
    NORMAL_CLOSED_PRIMARY_OBJECTIVE,
    RAW_ALL_NODES_PRIMARY_OBJECTIVE,
    boundary_outflow_normal_mach,
    close_boundary,
    contract_forward_sample,
    evaluate_pairs,
    evaluate_rollouts,
    pair_metrics,
    primary_training_metrics,
    resolve_boundary_policy,
)
from utility.time_dependent_no.pcno_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    autocast_context,
    load_checkpoint,
    select_device,
)

RESIDUAL_TARGET_KIND = "conservative_variable_residual"
NO_BOUNDARY_AUXILIARY = "none"
PROJECTED_BOUNDARY_AUXILIARY = "projected_target"
NEAR_BOUNDARY_AUXILIARY = "near_boundary_band"
RAW_BOUNDARY_REFERENCE_AUXILIARY = "raw_boundary_reference"
NO_INIT_BOUNDARY_TRANSITION = "none"
RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION = "model_all_nodes_to_causal_nodal_physical"
RAW_TO_MINIMUM_CHANGE_INIT_BOUNDARY_TRANSITION = (
    "model_all_nodes_to_minimum_change_nodal_physical"
)
HISTORICAL_SELECTION_MODE = "historical_all_node"
INTERIOR_SELECTION_MODE = "interior_rollout"
FIXED_HORIZON_BLOCKS_PRESENTATION_MODE = "fixed_horizon_blocks"
ATTACHED_PREDICTION_MULTISTEP_INPUT = "attached_prediction"
PROJECTED_TEACHER_MULTISTEP_INPUT = "projected_teacher"
STANDARD_NODE_TYPE_CHANNEL_CONTROL = "standard"
NO_TYPE_CHANNEL_CONTROL = "no_type_channels"
ZERO_CHANNEL_ORDINARY_CONTROL = "four_zero_channels_ordinary"
ZERO_CHANNEL_MATCHED_CONTROL = "four_zero_channels_matched"
NODE_TYPE_CHANNEL_CONTROLS = (
    STANDARD_NODE_TYPE_CHANNEL_CONTROL,
    NO_TYPE_CHANNEL_CONTROL,
    ZERO_CHANNEL_ORDINARY_CONTROL,
    ZERO_CHANNEL_MATCHED_CONTROL,
)
RESUME_MUTABLE_ARGS = {
    "checkpoint_every",
    "device",
    "init_checkpoint",
    "init_boundary_mode_transition",
    "max_wall_hours",
    "output_dir",
    "resume_checkpoint",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--split-seed", type=int, default=20260718)
    parser.add_argument(
        "--split-mode",
        choices=("stratified", "manifest"),
        default="stratified",
        help=(
            "Use the legacy deterministic train/validation split or require the "
            "exact train/validation/test partition declared by the shard manifest."
        ),
    )
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--presentation-mode",
        choices=(
            "balanced",
            "full_coverage",
            FIXED_HORIZON_BLOCKS_PRESENTATION_MODE,
        ),
        default="balanced",
        help=(
            "Use sampled trajectory-balanced presentations or visit every "
            "eligible transition exactly once per epoch. Fixed horizon blocks "
            "construct one deterministic same-trajectory multistep screen."
        ),
    )
    parser.add_argument("--presentations-per-epoch", type=int, default=4096)
    parser.add_argument("--val-presentations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Accumulate this many trajectory-homogeneous microbatches before "
            "one optimizer/scheduler step."
        ),
    )
    parser.add_argument(
        "--tiny-pairs",
        type=int,
        default=0,
        help=(
            "Repeat one immutable pair bank of this size; zero uses balanced sampling."
        ),
    )
    parser.add_argument("--tiny-fit-rel-l2", type=float, default=0.01)
    parser.add_argument("--tiny-fit-loss-ratio", type=float, default=0.01)
    parser.add_argument(
        "--stop-on-tiny-fit",
        action="store_true",
        help="Stop after the immutable tiny bank first passes both gates.",
    )
    parser.add_argument("--stats-time-stride", type=int, default=1)
    parser.add_argument(
        "--mach-scale-floor",
        type=float,
        default=0.1,
        help="Lower bound for the training-set Mach standard deviation.",
    )
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--domain-lengths", type=float, nargs=2, default=(6.0, 2.0))
    parser.add_argument(
        "--model-node-type-input",
        choices=("physical", "all_normal"),
        default="physical",
        help=(
            "Feed regenerated physical type codes or zeros to the unchanged four-slot "
            "one-hot input; physical codes remain available to metrics and masks."
        ),
    )
    parser.add_argument(
        "--node-type-channel-control",
        choices=NODE_TYPE_CHANNEL_CONTROLS,
        default=STANDARD_NODE_TYPE_CHANNEL_CONTROL,
        help=(
            "Training-only controlled comparison: retain the standard one-hot "
            "layout, omit all four type columns, use four literal-zero columns "
            "with ordinary initialization, or copy the no-type initialization "
            "into the four-zero-column model exactly."
        ),
    )
    parser.add_argument(
        "--boundary-field-mode",
        choices=BOUNDARY_FIELD_MODES,
        default=BOUNDARY_FIELD_NONE,
        help=(
            "Use no continuous boundary input, the union geometry collar, or "
            "the manifest-declared semantic collars. Active field modes require "
            "--node-type-channel-control no_type_channels."
        ),
    )
    parser.add_argument(
        "--boundary-residual-mode",
        choices=BOUNDARY_RESIDUAL_MODES,
        default=BOUNDARY_RESIDUAL_NONE,
        help=(
            "Keep the shared PCNO input unchanged and route manifest-declared "
            "semantic collars through a separately gated pointwise residual path."
        ),
    )
    parser.add_argument("--boundary-residual-width", type=int, default=64)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=(128, 128, 128, 128, 128)
    )
    parser.add_argument("--fc-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler",
        choices=("constant", "onecycle", "warmup_cosine"),
        default="constant",
        help="Use the strong residual-FNO recipe by default; OneCycle is explicit.",
    )
    parser.add_argument("--lr-div-factor", type=float, default=10.0)
    parser.add_argument("--lr-final-div-factor", type=float, default=1000.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--generated-state-exposure-weight",
        type=float,
        default=0.0,
        help=(
            "Mix this fraction of detached two-call generated-state loss with "
            "the clean one-step anchor; zero disables exposure."
        ),
    )
    parser.add_argument(
        "--multistep-loss-steps",
        type=int,
        default=1,
        help=(
            "Number of differentiable recurrent calls in the training objective; "
            "one preserves the one-step baseline."
        ),
    )
    parser.add_argument(
        "--multistep-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Total auxiliary weight for normal-node losses at recurrent calls "
            "2..K; zero is required when K=1."
        ),
    )
    parser.add_argument(
        "--multistep-recurrent-input",
        choices=(
            ATTACHED_PREDICTION_MULTISTEP_INPUT,
            PROJECTED_TEACHER_MULTISTEP_INPUT,
        ),
        default=ATTACHED_PREDICTION_MULTISTEP_INPUT,
        help=(
            "Feed the deployed prediction through the differentiable recurrence, "
            "or execute the call-matched projected-teacher control."
        ),
    )
    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--rollout-val-count", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=(HISTORICAL_SELECTION_MODE, INTERIOR_SELECTION_MODE),
        default=HISTORICAL_SELECTION_MODE,
        help=(
            "Select checkpoints by the historical all-node rollout hierarchy or "
            "by completion followed by normal-node long/short-horizon error."
        ),
    )
    parser.add_argument(
        "--selection-short-horizon",
        type=int,
        default=20,
        help="Short normal-node endpoint used by interior-rollout selection.",
    )
    parser.add_argument(
        "--rollout-checkpoints",
        type=int,
        nargs="*",
        default=(),
        help="Calls retained for checkpoint selection and horizon reporting.",
    )
    parser.add_argument(
        "--parity-rollout-keys",
        nargs="*",
        default=(),
        help="Validation-only keys used for an exact historical parity gate.",
    )
    parser.add_argument("--parity-rollout-horizon", type=int, default=20)
    parser.add_argument("--parity-max-rollout-relative-l2", type=float, default=None)
    parser.add_argument("--parity-max-one-step-relative-l2", type=float, default=None)
    parser.add_argument("--rollout-start-frame", type=int, default=0)
    parser.add_argument(
        "--boundary-mode",
        choices=(
            "model_all_nodes",
            CAUSAL_BOUNDARY_MODE,
            MINIMUM_CHANGE_BOUNDARY_MODE,
        ),
        default="model_all_nodes",
    )
    parser.add_argument("--boundary-max-source-hops", type=int, default=3)
    parser.add_argument("--boundary-rho-inf", type=float, default=1.4)
    parser.add_argument("--boundary-p-inf", type=float, default=1.0)
    parser.add_argument(
        "--primary-objective",
        choices=(
            NORMAL_CLOSED_PRIMARY_OBJECTIVE,
            RAW_ALL_NODES_PRIMARY_OBJECTIVE,
            LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE,
        ),
        default=NORMAL_CLOSED_PRIMARY_OBJECTIVE,
        help=(
            "Optimize the deployed proposal on normal nodes, directly supervise "
            "the raw proposal on every valid node, or supervise every deployed "
            "free degree of freedom through the minimum-change projection."
        ),
    )
    parser.add_argument(
        "--boundary-auxiliary",
        choices=(
            NO_BOUNDARY_AUXILIARY,
            PROJECTED_BOUNDARY_AUXILIARY,
            NEAR_BOUNDARY_AUXILIARY,
            RAW_BOUNDARY_REFERENCE_AUXILIARY,
        ),
        default=NO_BOUNDARY_AUXILIARY,
        help=(
            "Add one boundary training signal while preserving the "
            "normal-node next-state objective and inference closure."
        ),
    )
    parser.add_argument("--boundary-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--near-boundary-hops", type=int, default=2)
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        default=0.0,
        help="Gracefully stop before another epoch would exceed this wall budget.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model/split/scales and start a fresh matched continuation.",
    )
    checkpoint_group.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, epoch, and best-selection state.",
    )
    parser.add_argument(
        "--init-boundary-mode-transition",
        choices=(
            NO_INIT_BOUNDARY_TRANSITION,
            RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION,
            RAW_TO_MINIMUM_CHANGE_INIT_BOUNDARY_TRANSITION,
        ),
        default=NO_INIT_BOUNDARY_TRANSITION,
        help=(
            "Explicitly authorize a supported initialization-time boundary "
            "contract transition; never applies to resume."
        ),
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if (
        args.boundary_field_mode != BOUNDARY_FIELD_NONE
        and args.node_type_channel_control != NO_TYPE_CHANNEL_CONTROL
    ):
        raise ValueError(
            "continuous boundary fields require "
            "--node-type-channel-control no_type_channels"
        )
    if args.boundary_residual_mode != BOUNDARY_RESIDUAL_NONE:
        if args.boundary_field_mode != BOUNDARY_FIELD_NONE:
            raise ValueError(
                "lifted boundary fields and the boundary-residual side path are "
                "separate representation studies"
            )
        if args.node_type_channel_control != NO_TYPE_CHANNEL_CONTROL:
            raise ValueError(
                "the boundary-residual side path requires "
                "--node-type-channel-control no_type_channels"
            )
    if args.node_type_channel_control != STANDARD_NODE_TYPE_CHANNEL_CONTROL:
        if args.model_node_type_input != "physical":
            raise ValueError(
                "controlled zero-channel runs retain physical type tensors for "
                "metrics and require --model-node-type-input physical"
            )
        if (
            args.node_type_channel_control != NO_TYPE_CHANNEL_CONTROL
            and args.boundary_mode != "model_all_nodes"
        ):
            raise ValueError(
                "constant-zero-channel controls freeze "
                "--boundary-mode model_all_nodes"
            )
        if args.init_checkpoint is not None:
            raise ValueError(
                "controlled zero-channel runs require fresh initialization"
            )
    positive_ints = {
        "epochs": args.epochs,
        "presentations_per_epoch": args.presentations_per_epoch,
        "val_presentations": args.val_presentations,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "step_stride": args.step_stride,
        "stats_time_stride": args.stats_time_stride,
        "boundary_residual_width": args.boundary_residual_width,
        "rollout_every": args.rollout_every,
        "rollout_val_count": args.rollout_val_count,
        "rollout_steps": args.rollout_steps,
        "checkpoint_every": args.checkpoint_every,
        "multistep_loss_steps": args.multistep_loss_steps,
        "selection_short_horizon": args.selection_short_horizon,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tiny_pairs < 0:
        raise ValueError("--tiny-pairs must be nonnegative")
    if args.stop_on_tiny_fit and args.tiny_pairs == 0:
        raise ValueError("--stop-on-tiny-fit requires --tiny-pairs")
    if args.rollout_start_frame < 0:
        raise ValueError("--rollout-start-frame must be nonnegative")
    if args.tiny_fit_rel_l2 <= 0.0 or args.tiny_fit_loss_ratio <= 0.0:
        raise ValueError("tiny-fit thresholds must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer configuration")
    if args.gradient_clip < 0.0:
        raise ValueError("--gradient-clip must be nonnegative")
    if args.lr_div_factor <= 0.0 or args.lr_final_div_factor <= 0.0:
        raise ValueError("OneCycle division factors must be positive")
    if not 0.0 < args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must lie in (0,1)")
    if not 0.0 < args.warmup_start_factor <= 1.0:
        raise ValueError("--warmup-start-factor must lie in (0,1]")
    if args.scheduler == "warmup_cosine" and (
        not math.isfinite(args.min_learning_rate)
        or args.min_learning_rate <= 0.0
        or args.min_learning_rate >= args.learning_rate
    ):
        raise ValueError("--min-learning-rate must lie in (0, learning-rate)")
    if args.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be nonnegative")
    if (
        not math.isfinite(args.generated_state_exposure_weight)
        or not 0.0 <= args.generated_state_exposure_weight < 1.0
    ):
        raise ValueError("--generated-state-exposure-weight must lie in [0,1)")
    if args.generated_state_exposure_weight > 0.0 and args.input_noise_std > 0.0:
        raise ValueError(
            "generated-state exposure and input noise are separate branches"
        )
    if args.generated_state_exposure_weight > 0.0 and args.tiny_pairs > 0:
        raise ValueError("generated-state exposure is not a tiny-fit mode")
    if args.multistep_loss_steps < 1:
        raise ValueError("--multistep-loss-steps must be positive")
    if (
        not math.isfinite(args.multistep_loss_weight)
        or args.multistep_loss_weight < 0.0
    ):
        raise ValueError("--multistep-loss-weight must be finite and nonnegative")
    if args.multistep_loss_steps == 1 and args.multistep_loss_weight != 0.0:
        raise ValueError("K=1 requires --multistep-loss-weight 0")
    if (
        args.multistep_loss_steps == 1
        and args.multistep_recurrent_input != ATTACHED_PREDICTION_MULTISTEP_INPUT
    ):
        raise ValueError("K=1 has no multistep recurrent input")
    if args.multistep_loss_steps > 1 and args.multistep_loss_weight <= 0.0:
        raise ValueError("K>1 requires a positive --multistep-loss-weight")
    if args.multistep_loss_steps > 1:
        if args.presentation_mode not in (
            "full_coverage",
            FIXED_HORIZON_BLOCKS_PRESENTATION_MODE,
        ):
            raise ValueError(
                "multistep loss requires full coverage or fixed horizon blocks"
            )
        if args.tiny_pairs > 0:
            raise ValueError("differentiable multistep loss is not a tiny-fit mode")
        if args.input_noise_std > 0.0:
            raise ValueError(
                "differentiable multistep loss and input noise are separate studies"
            )
        if args.generated_state_exposure_weight > 0.0:
            raise ValueError(
                "differentiable multistep loss and detached exposure are "
                "separate studies"
            )
    if args.presentation_mode == FIXED_HORIZON_BLOCKS_PRESENTATION_MODE:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        if args.multistep_loss_steps < 2:
            raise ValueError("fixed horizon blocks require K>1")
        if args.epochs != 1:
            raise ValueError("fixed horizon blocks are a one-epoch screen")
        if args.presentations_per_epoch % effective_batch_size != 0:
            raise ValueError(
                "fixed horizon presentations must divide into complete effective batches"
            )
    if args.presentation_mode == "full_coverage" and args.tiny_pairs > 0:
        raise ValueError("full coverage and tiny-fit sampling are separate modes")
    if (
        args.presentation_mode == "full_coverage"
        and args.generated_state_exposure_weight > 0.0
    ):
        raise ValueError("the registered full-coverage baseline has no exposure loss")
    rollout_checkpoints = sorted({int(value) for value in args.rollout_checkpoints})
    if any(value < 1 or value > args.rollout_steps for value in rollout_checkpoints):
        raise ValueError("rollout checkpoints must lie inside the requested horizon")
    args.rollout_checkpoints = rollout_checkpoints
    parity_keys = [str(key) for key in args.parity_rollout_keys]
    if len(set(parity_keys)) != len(parity_keys):
        raise ValueError("parity rollout keys must be unique")
    args.parity_rollout_keys = parity_keys
    parity_thresholds = (
        args.parity_max_rollout_relative_l2,
        args.parity_max_one_step_relative_l2,
    )
    if parity_keys:
        if args.parity_rollout_horizon not in rollout_checkpoints:
            raise ValueError(
                "parity rollout horizon must be one of --rollout-checkpoints"
            )
        if any(
            value is None or not math.isfinite(value) or value <= 0.0
            for value in parity_thresholds
        ):
            raise ValueError("enabled parity thresholds must be positive and finite")
    elif any(value is not None for value in parity_thresholds):
        raise ValueError("parity thresholds require --parity-rollout-keys")
    if args.selection_mode == INTERIOR_SELECTION_MODE:
        if args.parity_rollout_keys:
            raise ValueError(
                "interior-rollout selection and the historical parity gate are "
                "separate contracts"
            )
        if args.selection_short_horizon > args.rollout_steps:
            raise ValueError("selection short horizon exceeds the rollout horizon")
        if args.selection_short_horizon not in args.rollout_checkpoints:
            raise ValueError(
                "interior-rollout selection short horizon must be retained"
            )
    if args.boundary_max_source_hops < 1:
        raise ValueError("--boundary-max-source-hops must be positive")
    if args.near_boundary_hops < 1:
        raise ValueError("--near-boundary-hops must be positive")
    if (
        not math.isfinite(args.boundary_auxiliary_weight)
        or args.boundary_auxiliary_weight < 0.0
    ):
        raise ValueError("--boundary-auxiliary-weight must be finite and nonnegative")
    if args.boundary_auxiliary == NO_BOUNDARY_AUXILIARY:
        if args.boundary_auxiliary_weight != 0.0:
            raise ValueError("a boundary auxiliary weight requires an auxiliary kind")
    else:
        if args.boundary_mode != CAUSAL_BOUNDARY_MODE:
            raise ValueError("boundary auxiliaries require causal boundary mode")
        if args.boundary_auxiliary_weight <= 0.0:
            raise ValueError("an enabled boundary auxiliary requires positive weight")
        if args.generated_state_exposure_weight > 0.0:
            raise ValueError(
                "boundary auxiliaries and generated-state exposure are separate studies"
            )
    if args.primary_objective == RAW_ALL_NODES_PRIMARY_OBJECTIVE:
        if args.boundary_mode != CAUSAL_BOUNDARY_MODE:
            raise ValueError("raw all-node supervision requires causal boundary mode")
        if args.boundary_auxiliary != NO_BOUNDARY_AUXILIARY:
            raise ValueError(
                "raw all-node supervision and boundary auxiliaries are separate studies"
            )
        if args.generated_state_exposure_weight > 0.0:
            raise ValueError(
                "raw all-node supervision and generated-state exposure are "
                "separate studies"
            )
        if args.multistep_loss_steps > 1:
            raise ValueError(
                "raw all-node supervision and deployed multistep loss are "
                "separate studies"
            )
    if args.primary_objective == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE:
        if args.boundary_mode != MINIMUM_CHANGE_BOUNDARY_MODE:
            raise ValueError(
                "learned-DOF supervision requires minimum-change boundary mode"
            )
        if args.boundary_auxiliary != NO_BOUNDARY_AUXILIARY:
            raise ValueError(
                "learned-DOF supervision already includes deployed boundary DOFs"
            )
        if args.generated_state_exposure_weight > 0.0:
            raise ValueError(
                "learned-DOF supervision and generated-state exposure are "
                "separate studies"
            )
    if (
        not math.isfinite(args.boundary_rho_inf)
        or not math.isfinite(args.boundary_p_inf)
        or args.boundary_rho_inf <= 0.0
        or args.boundary_p_inf <= 0.0
    ):
        raise ValueError("boundary freestream density and pressure must be positive")
    if not math.isfinite(args.max_wall_hours) or args.max_wall_hours < 0.0:
        raise ValueError("--max-wall-hours must be finite and nonnegative")
    if args.init_boundary_mode_transition != NO_INIT_BOUNDARY_TRANSITION:
        if args.init_checkpoint is None:
            raise ValueError(
                "--init-boundary-mode-transition requires --init-checkpoint"
            )
        if args.resume_checkpoint is not None:
            raise ValueError(
                "initialization boundary transitions never apply to resume"
            )
    if not math.isfinite(args.mach_scale_floor) or args.mach_scale_floor <= 0.0:
        raise ValueError("--mach-scale-floor must be positive and finite")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA; use --amp none on CPU")
    if (
        args.amp == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bfloat16")


def assert_resume_training_args(
    checkpoint: Mapping[str, Any], args: argparse.Namespace
) -> None:
    """Allow operational resume changes but freeze scientific/numerical choices."""

    saved_payload = checkpoint.get("training_args")
    if not isinstance(saved_payload, Mapping):
        raise ValueError("resume checkpoint lacks its training arguments")
    saved = dict(saved_payload)
    saved.setdefault(
        "node_type_channel_control", STANDARD_NODE_TYPE_CHANNEL_CONTROL
    )
    saved.setdefault("boundary_field_mode", BOUNDARY_FIELD_NONE)
    saved.setdefault("boundary_field_names", [])
    saved.setdefault("boundary_residual_mode", BOUNDARY_RESIDUAL_NONE)
    saved.setdefault("boundary_residual_names", [])
    saved.setdefault("boundary_residual_width", 64)
    current = jsonable_args(args)
    missing = sorted(set(current) - set(saved) - RESUME_MUTABLE_ARGS)
    differences = {
        name: {"saved": saved[name], "current": current[name]}
        for name in sorted((set(current) & set(saved)) - RESUME_MUTABLE_ARGS)
        if saved[name] != current[name]
    }
    if missing or differences:
        raise ValueError(
            "resume would change the frozen training contract: "
            f"missing={missing}, differences={differences}"
        )


def warmup_cosine_factor(
    step_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    start_factor: float,
    minimum_factor: float,
) -> float:
    """Return an optimizer-step-indexed warmup plus cosine multiplier."""

    if total_steps < 2 or not 1 <= warmup_steps < total_steps:
        raise ValueError("warmup schedule requires 1 <= warmup_steps < total_steps")
    step = min(max(int(step_index), 0), total_steps - 1)
    if step < warmup_steps:
        if warmup_steps == 1:
            return 1.0
        progress = step / (warmup_steps - 1)
        return start_factor + progress * (1.0 - start_factor)
    decay_steps = total_steps - warmup_steps
    if decay_steps == 1:
        return minimum_factor
    progress = (step - warmup_steps) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_factor + (1.0 - minimum_factor) * cosine


def build_data_contract_summary(
    store: PCNOEuler2DShardStore,
    *,
    train_keys: Sequence[str],
    val_keys: Sequence[str],
    test_keys: Sequence[str],
    normalization: Euler2DNormalization,
) -> dict[str, Any]:
    """Bind the training data needed by evaluation and the later Line-4 handoff."""

    entries = {key: store.entry(key) for key in store.keys}
    split_contract = {
        "train": list(train_keys),
        "validation": list(val_keys),
        "test": list(test_keys),
        "split_groups": {key: entries[key].get("split_group_id") for key in store.keys},
    }
    geometry_by_trajectory = {
        key: entries[key].get("geometry_digest") for key in store.keys
    }
    source_artifacts = {
        key: entries[key].get("source_reference_sha256") for key in store.keys
    }
    node_counts = sorted({int(entry["num_nodes"]) for entry in entries.values()})
    edge_counts = sorted(
        {int(entry["num_directed_edges"]) for entry in entries.values()}
    )
    resolution_contract: dict[str, Any] = {
        "node_counts": node_counts,
        "directed_edge_counts": edge_counts,
        "coordinate_convention": store.manifest.get("coordinate_convention"),
    }
    time_contract: dict[str, Any] | None = None
    if store.manifest.get("dataset") == "shock_vortex_fv_family":
        first_key = store.keys[0]
        nodes = np.asarray(store.array(first_key, "nodes"), dtype=np.float64)
        nx = int(np.unique(nodes[:, 0]).size)
        ny = int(np.unique(nodes[:, 1]).size)
        if nx * ny != nodes.shape[0]:
            raise ValueError("shock-vortex PCNO nodes do not form the declared grid")
        resolution_contract["stored_grid"] = [nx, ny]
        physical_times = np.asarray(
            store.array(first_key, "physical_times"), dtype=np.float64
        )
        if (
            physical_times.ndim != 1
            or physical_times.size < 2
            or not np.all(np.isfinite(physical_times))
            or not np.all(np.diff(physical_times) > 0.0)
        ):
            raise ValueError("physical time coordinates are invalid")
        deltas = np.diff(physical_times)
        time_contract = {
            "physical_times_digest": digest_array(physical_times),
            "start_time": float(physical_times[0]),
            "final_time": float(physical_times[-1]),
            "saved_calls": int(physical_times.size - 1),
            "delta_t": float(deltas[0]),
            "uniform_delta_t": bool(
                np.allclose(deltas, deltas[0], rtol=0.0, atol=1.0e-14)
            ),
        }
        if not time_contract["uniform_delta_t"]:
            raise ValueError("shock-vortex saved times are not uniformly spaced")

    normalization_payload = normalization.to_dict()
    boundary_field_contract = store.boundary_field_contract
    return {
        "schema": "pcno_euler2d_data_contract_v1",
        "dataset": store.manifest.get("dataset"),
        "data_manifest_digest": store.manifest_digest,
        "source_family_id": store.manifest.get("source_family_id"),
        "source_family_manifest_digest": store.manifest.get(
            "source_family_manifest_digest"
        ),
        "requested_splits": store.manifest.get("requested_splits"),
        "source_artifact_set_digest": digest_mapping(source_artifacts),
        "grouped_split_digest": digest_mapping(split_contract),
        "geometry_contract_digest": digest_mapping(geometry_by_trajectory),
        "unique_geometry_digests": sorted(set(geometry_by_trajectory.values())),
        "resolution_contract": resolution_contract,
        "resolution_contract_digest": digest_mapping(resolution_contract),
        "time_contract": time_contract,
        "time_contract_digest": (
            None if time_contract is None else digest_mapping(time_contract)
        ),
        "state_convention": store.manifest.get("state_convention"),
        "weight_provenance": normalization.weight_provenance,
        "normalization_digest": digest_mapping(normalization_payload),
        "mesh_to_graph_map": store.manifest.get("mesh_to_graph_map"),
        "boundary_field_contract": boundary_field_contract,
        "boundary_field_contract_digest": (
            None
            if boundary_field_contract is None
            else digest_mapping(boundary_field_contract)
        ),
        "line4_handoff": {
            "status": "prerequisites_only_baseline_not_frozen",
            "line4_training_truth_authorized": None,
            "line4_front_candidate_available": None,
        },
    }


def manifest_train_val_test_split(
    store: PCNOEuler2DShardStore,
) -> tuple[list[str], list[str], list[str]]:
    """Return an exact, complete manifest partition or fail closed."""

    split_names = ("train", "validation", "test")
    raw_requested_splits = store.manifest.get("requested_splits")
    if raw_requested_splits is None:
        requested_splits = split_names
    else:
        if not isinstance(raw_requested_splits, list) or not all(
            isinstance(name, str) for name in raw_requested_splits
        ):
            raise ValueError("manifest requested_splits must be a list of names")
        requested_splits = tuple(raw_requested_splits)
        canonical_requested = tuple(
            name for name in split_names if name in set(requested_splits)
        )
        if (
            requested_splits != canonical_requested
            or not {"train", "validation"}.issubset(requested_splits)
        ):
            raise ValueError(
                "manifest requested_splits must contain train and validation "
                "once each in canonical order"
            )
    raw_splits = store.manifest.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(split_names):
        raise ValueError(
            "manifest split mode requires exactly train, validation, and test splits"
        )
    splits: dict[str, list[str]] = {}
    for split_name in split_names:
        raw_keys = raw_splits[split_name]
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) for key in raw_keys
        ):
            raise ValueError(f"manifest split {split_name!r} must be a list of keys")
        keys = list(raw_keys)
        if split_name in requested_splits and not keys:
            raise ValueError(f"manifest split {split_name!r} must not be empty")
        if split_name not in requested_splits and keys:
            raise ValueError(
                f"manifest split {split_name!r} is outside requested_splits"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(f"manifest split {split_name!r} contains duplicate keys")
        splits[split_name] = keys

    partition = [key for split_name in split_names for key in splits[split_name]]
    if len(set(partition)) != len(partition):
        raise ValueError("manifest train/validation/test splits overlap")
    store_keys = set(store.keys)
    partition_keys = set(partition)
    if partition_keys != store_keys:
        missing = sorted(store_keys - partition_keys)
        unknown = sorted(partition_keys - store_keys)
        raise ValueError(
            "manifest splits do not exactly partition the shard store: "
            f"missing={missing}, unknown={unknown}"
        )

    actual_counts = {name: len(splits[name]) for name in split_names}
    raw_prepared_counts = store.manifest.get("prepared_split_counts")
    if not isinstance(raw_prepared_counts, Mapping):
        raise ValueError("manifest split mode requires prepared_split_counts")
    prepared_counts = {
        name: int(raw_prepared_counts.get(name, -1)) for name in split_names
    }
    if prepared_counts != actual_counts:
        raise ValueError(
            "prepared_split_counts does not match the complete manifest partition: "
            f"{prepared_counts} != {actual_counts}"
        )

    raw_declared_counts = store.manifest.get("declared_split_counts")
    if not isinstance(raw_declared_counts, Mapping):
        raise ValueError("manifest split mode requires declared_split_counts")
    declared_counts = {
        name: int(raw_declared_counts.get(name, -1)) for name in split_names
    }
    invalid_declared = {
        name: (declared_counts[name], actual_counts[name])
        for name in split_names
        if declared_counts[name] < 0
        or (
            name in requested_splits
            and declared_counts[name] != actual_counts[name]
        )
    }
    if invalid_declared:
        raise ValueError(
            "declared_split_counts disagrees with requested populations: "
            f"{invalid_declared}"
        )

    for split_name in split_names:
        for key in splits[split_name]:
            if store.entry(key).get("split") != split_name:
                raise ValueError(
                    f"trajectory {key!r} entry disagrees with split {split_name!r}"
                )
    return splits["train"], splits["validation"], splits["test"]


def resolve_initialization_transition(
    checkpoint: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Resolve an explicitly allowed fresh-initialization boundary change."""

    source_mode = str(checkpoint.get("boundary_mode", "model_all_nodes"))
    target_mode = str(args.boundary_mode)
    requested = str(args.init_boundary_mode_transition)
    if source_mode == target_mode:
        if requested != NO_INIT_BOUNDARY_TRANSITION:
            raise ValueError(
                "an initialization boundary transition was requested but the "
                "checkpoint and target boundary modes already match"
            )
        return None
    supported = {
        ("model_all_nodes", CAUSAL_BOUNDARY_MODE): (
            RAW_TO_CAUSAL_INIT_BOUNDARY_TRANSITION
        ),
        ("model_all_nodes", MINIMUM_CHANGE_BOUNDARY_MODE): (
            RAW_TO_MINIMUM_CHANGE_INIT_BOUNDARY_TRANSITION
        ),
    }
    expected = supported.get((source_mode, target_mode))
    if args.init_checkpoint is None or expected is None or requested != expected:
        raise ValueError(
            "checkpoint and requested boundary modes differ without the exact "
            "supported initialization transition"
        )
    source_contract = checkpoint.get("boundary_contract")
    return {
        "schema": "pcno_euler2d_initialization_transition_v1",
        "kind": expected,
        "source_boundary_mode": source_mode,
        "target_boundary_mode": target_mode,
        "parent_checkpoint_sha256": sha256_file(args.init_checkpoint),
        "source_boundary_contract_digest": (
            digest_mapping(source_contract)
            if isinstance(source_contract, Mapping)
            else None
        ),
        "target_boundary_contract_digest": None,
        "model_state_loaded_strictly": True,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "future_reference_boundary_values": False,
    }


def assert_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    store: PCNOEuler2DShardStore,
    args: argparse.Namespace,
    initialization_transition: Mapping[str, Any] | None = None,
) -> None:
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and current shard manifest digests differ")
    if int(checkpoint["step_stride"]) != args.step_stride:
        raise ValueError("checkpoint and requested step stride differ")
    saved_boundary_mode = str(checkpoint.get("boundary_mode", "model_all_nodes"))
    if saved_boundary_mode != args.boundary_mode:
        if initialization_transition is None:
            raise ValueError("checkpoint and requested boundary modes differ")
        if (
            initialization_transition.get("source_boundary_mode") != saved_boundary_mode
            or initialization_transition.get("target_boundary_mode")
            != args.boundary_mode
        ):
            raise ValueError("initialization boundary transition metadata mismatch")
    saved_training_args = checkpoint.get("training_args", {})
    saved_model_node_type_input = str(
        checkpoint.get(
            "model_node_type_input",
            saved_training_args.get("model_node_type_input", "physical"),
        )
    )
    if saved_model_node_type_input != args.model_node_type_input:
        raise ValueError("checkpoint and requested model node-type inputs differ")
    saved_channel_control = str(
        saved_training_args.get(
            "node_type_channel_control", STANDARD_NODE_TYPE_CHANNEL_CONTROL
        )
    )
    if saved_channel_control != args.node_type_channel_control:
        raise ValueError("checkpoint and requested node-type channel controls differ")
    saved_split_mode = str(saved_training_args.get("split_mode", "stratified"))
    if saved_split_mode != args.split_mode:
        raise ValueError("checkpoint and requested split modes differ")
    if args.split_mode == "manifest":
        train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
        checkpoint_partition = (
            [str(key) for key in checkpoint["train_keys"]],
            [str(key) for key in checkpoint["val_keys"]],
            [str(key) for key in checkpoint.get("test_keys", [])],
        )
        if checkpoint_partition != (train_keys, val_keys, test_keys):
            raise ValueError("checkpoint keys differ from the manifest partition")
    saved_mach_floor = float(checkpoint["normalization"]["mach_scale_floor"])
    if not math.isclose(
        saved_mach_floor, args.mach_scale_floor, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("checkpoint and requested Mach scale floor differ")
    actual = checkpoint["model_config"]
    saved_target_kind = str(
        saved_training_args.get(
            "target_kind",
            actual.get("target_kind", RESIDUAL_TARGET_KIND),
        )
    )
    if saved_target_kind != RESIDUAL_TARGET_KIND:
        raise ValueError("checkpoint target is not the conservative residual")
    expected = {
        "k_max": int(args.k_max),
        "domain_lengths": [float(value) for value in args.domain_lengths],
        "layers": [int(value) for value in args.layers],
        "fc_dim": int(args.fc_dim),
        "node_type_feature_mode": node_type_feature_mode_for_control(
            args.node_type_channel_control
        ),
        "boundary_field_mode": getattr(
            args, "boundary_field_mode", BOUNDARY_FIELD_NONE
        ),
        "boundary_field_names": list(
            getattr(args, "boundary_field_names", ())
        ),
        "boundary_residual_mode": getattr(
            args, "boundary_residual_mode", BOUNDARY_RESIDUAL_NONE
        ),
        "boundary_residual_names": list(
            getattr(args, "boundary_residual_names", ())
        ),
        "boundary_residual_width": int(
            getattr(args, "boundary_residual_width", 64)
        ),
    }
    for key, value in expected.items():
        if key == "node_type_feature_mode":
            actual_value = actual.get(key, NODE_TYPE_FEATURE_ONE_HOT)
        elif key == "boundary_field_mode":
            actual_value = actual.get(key, BOUNDARY_FIELD_NONE)
        elif key == "boundary_field_names":
            actual_value = actual.get(key, [])
        elif key == "boundary_residual_mode":
            actual_value = actual.get(key, BOUNDARY_RESIDUAL_NONE)
        elif key == "boundary_residual_names":
            actual_value = actual.get(key, [])
        elif key == "boundary_residual_width":
            actual_value = actual.get(key, 64)
        else:
            actual_value = actual[key]
        if actual_value != value:
            raise ValueError(
                f"checkpoint model config mismatch for {key}: "
                f"{actual_value} != {value}"
            )


def node_type_feature_mode_for_control(control: str) -> str:
    if control == STANDARD_NODE_TYPE_CHANNEL_CONTROL:
        return NODE_TYPE_FEATURE_ONE_HOT
    if control == NO_TYPE_CHANNEL_CONTROL:
        return NODE_TYPE_FEATURE_OMITTED
    if control in {ZERO_CHANNEL_ORDINARY_CONTROL, ZERO_CHANNEL_MATCHED_CONTROL}:
        return NODE_TYPE_FEATURE_CONSTANT_ZERO
    raise ValueError(f"unsupported node-type channel control: {control}")


def boundary_field_names_for_store(
    store: PCNOEuler2DShardStore, mode: str
) -> tuple[str, ...]:
    """Resolve model field order only from the verified shard manifest."""

    if mode == BOUNDARY_FIELD_NONE:
        return ()
    names = store.boundary_field_names
    if not names:
        raise ValueError(
            f"boundary-field mode {mode!r} requires a shard boundary-field contract"
        )
    return names


def build_model(
    args: argparse.Namespace,
    normalization: Euler2DNormalization,
    *,
    zero_initialize: bool,
) -> PCNOEuler2DResidual:
    """Construct the conservative-residual baseline."""

    common = {
        "normalization": normalization,
        "k_max": args.k_max,
        "domain_lengths": args.domain_lengths,
        "layers": args.layers,
        "fc_dim": args.fc_dim,
        "zero_initialize": zero_initialize,
    }
    control = args.node_type_channel_control
    boundary_field_mode = getattr(
        args, "boundary_field_mode", BOUNDARY_FIELD_NONE
    )
    boundary_field_names = tuple(getattr(args, "boundary_field_names", ()))
    boundary_residual_mode = getattr(
        args, "boundary_residual_mode", BOUNDARY_RESIDUAL_NONE
    )
    boundary_residual_names = tuple(getattr(args, "boundary_residual_names", ()))
    boundary_residual_width = int(getattr(args, "boundary_residual_width", 64))
    if boundary_residual_mode != BOUNDARY_RESIDUAL_NONE:
        if boundary_field_mode != BOUNDARY_FIELD_NONE:
            raise ValueError(
                "lifted boundary fields and the boundary-residual side path are "
                "separate representation studies"
            )
        if control != NO_TYPE_CHANNEL_CONTROL:
            raise ValueError(
                "the boundary-residual side path requires the no-type-channel control"
            )
        no_boundary_model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
        )
        no_boundary_rng_state = torch.get_rng_state().clone()
        model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
            boundary_residual_mode=boundary_residual_mode,
            boundary_residual_names=boundary_residual_names,
            boundary_residual_width=boundary_residual_width,
        )
        initialization_control = (
            copy_no_boundary_initialization_to_boundary_residual_model(
                no_boundary_model, model
            )
        )
        torch.set_rng_state(no_boundary_rng_state)
        initialization_control["cpu_rng_state_matches_no_boundary_arm"] = True
    elif boundary_field_mode != BOUNDARY_FIELD_NONE:
        if control != NO_TYPE_CHANNEL_CONTROL:
            raise ValueError(
                "continuous boundary fields require the no-type-channel control"
            )
        no_boundary_model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
        )
        no_boundary_rng_state = torch.get_rng_state().clone()
        model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
            boundary_field_mode=boundary_field_mode,
            boundary_field_names=boundary_field_names,
        )
        initialization_control = (
            copy_no_boundary_initialization_to_boundary_field_model(
                no_boundary_model, model
            )
        )
        torch.set_rng_state(no_boundary_rng_state)
        initialization_control["cpu_rng_state_matches_no_boundary_arm"] = True
    elif control == ZERO_CHANNEL_MATCHED_CONTROL:
        no_type_model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_OMITTED,
        )
        no_type_rng_state = torch.get_rng_state().clone()
        model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=NODE_TYPE_FEATURE_CONSTANT_ZERO,
        )
        initialization_control = copy_no_type_initialization_to_zero_channels(
            no_type_model, model
        )
        torch.set_rng_state(no_type_rng_state)
        initialization_control["cpu_rng_state_matches_no_type_arm"] = True
    else:
        feature_mode = node_type_feature_mode_for_control(control)
        model = PCNOEuler2DResidual(
            **common,
            node_type_feature_mode=feature_mode,
        )
        initialization_control = {
            "schema": "pcno_zero_channel_initialization_v1",
            "kind": control,
            "feature_mode": feature_mode,
            "ordinary_initialization": control
            in {
                STANDARD_NODE_TYPE_CHANNEL_CONTROL,
                NO_TYPE_CHANNEL_CONTROL,
                ZERO_CHANNEL_ORDINARY_CONTROL,
            },
            "mathematical_initial_function_match": None,
        }
    initialization_control["training_control"] = control
    initialization_control["boundary_field_mode"] = boundary_field_mode
    initialization_control["boundary_field_names"] = list(boundary_field_names)
    initialization_control["boundary_residual_mode"] = boundary_residual_mode
    initialization_control["boundary_residual_names"] = list(boundary_residual_names)
    initialization_control["boundary_residual_width"] = boundary_residual_width
    model.initialization_control = initialization_control
    model.model_node_type_input = args.model_node_type_input
    return model


def target_contract() -> dict[str, Any]:
    """Describe the conservative-residual supervision contract."""

    return {
        "target_kind": RESIDUAL_TARGET_KIND,
        "supervision": "decoded_next_state_loss",
        "coordinate_convention": "conservative_[rho,rho_u,rho_v,E]",
        "emits_physical_face_exchange": False,
        "reference_impulse_supervision": False,
        "conserved_totals_are_outcome_metrics_not_guaranteed_structure": True,
    }


def checkpoint_selection_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Describe checkpoint ranking without conflating boundary trace and dynamics."""

    if args.selection_mode == INTERIOR_SELECTION_MODE:
        ranking = [
            "all_validation_rollouts_complete",
            "completion_rate",
            "mean_survival_fraction",
            "final_normal_node_relative_l2",
            f"H{args.selection_short_horizon}_normal_node_relative_l2",
            "fixed_pair_normal_node_relative_l2",
            "final_all_node_relative_l2_tiebreak",
            "fixed_pair_all_node_relative_l2_tiebreak",
        ]
    else:
        ranking = [
            "historical_parity_eligibility_if_configured",
            "all_validation_rollouts_complete",
            "completion_rate",
            "mean_survival_fraction",
            "final_all_node_relative_l2",
            "H20_all_node_relative_l2",
            "fixed_pair_primary_relative_l2",
        ]
    return {
        "mode": args.selection_mode,
        "ranking": ranking,
        "rollout_population": "frozen_validation_rollout_keys",
        "primary_horizon": args.rollout_steps,
        "short_horizon": (
            args.selection_short_horizon
            if args.selection_mode == INTERIOR_SELECTION_MODE
            else 20
        ),
        "boundary_reference_error_is_primary": (
            False if args.selection_mode == INTERIOR_SELECTION_MODE else True
        ),
        "front_and_anti_smearing_metrics": "mandatory_post_selection_no_harm_audit",
    }


def boundary_training_objective_contract(args: argparse.Namespace) -> dict[str, Any]:
    kind = str(args.boundary_auxiliary)
    primary_kind = str(args.primary_objective)
    if primary_kind == RAW_ALL_NODES_PRIMARY_OBJECTIVE:
        primary = "raw_all_node_proxy_weighted_scaled_conservative_next_state_mse"
        primary_prediction = "raw_model_proposal_before_causal_closure"
        primary_node_population = "all_valid_nodes"
    elif primary_kind == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE:
        primary = "projected_all_node_proxy_weighted_scaled_conservative_next_state_mse"
        primary_prediction = "minimum_change_closed_deployed_proposal"
        primary_node_population = (
            "all_valid_nodes_with_fixed_and_constrained_dofs_zeroed_by_projection"
        )
    elif args.boundary_mode == "model_all_nodes":
        primary = "raw_all_node_proxy_weighted_scaled_conservative_next_state_mse"
        primary_prediction = "raw_model_proposal_equal_to_deployed_proposal"
        primary_node_population = "all_valid_nodes"
    else:
        primary = "normal_node_proxy_weighted_scaled_conservative_next_state_mse"
        primary_prediction = "causally_closed_deployed_proposal"
        primary_node_population = "normal_nodes_only"
    return {
        "primary_kind": primary_kind,
        "primary": primary,
        "primary_prediction": primary_prediction,
        "primary_node_population": primary_node_population,
        "reference_boundary_targets": (
            "training_and_validation_only"
            if primary_kind == RAW_ALL_NODES_PRIMARY_OBJECTIVE
            or primary_kind == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE
            or kind == RAW_BOUNDARY_REFERENCE_AUXILIARY
            else None
        ),
        "auxiliary": kind,
        "auxiliary_weight": float(args.boundary_auxiliary_weight),
        "auxiliary_prediction": (
            "raw_model_proposal_before_causal_closure"
            if kind == RAW_BOUNDARY_REFERENCE_AUXILIARY
            else (
                "causally_closed_deployed_proposal"
                if kind != NO_BOUNDARY_AUXILIARY
                else None
            )
        ),
        "auxiliary_node_population": (
            "valid_non_normal_nodes"
            if kind == RAW_BOUNDARY_REFERENCE_AUXILIARY
            else (
                "wall_and_outflow"
                if kind == PROJECTED_BOUNDARY_AUXILIARY
                else (
                    "near_boundary_normal_band"
                    if kind == NEAR_BOUNDARY_AUXILIARY
                    else None
                )
            )
        ),
        "near_boundary_hops": (
            int(args.near_boundary_hops) if kind == NEAR_BOUNDARY_AUXILIARY else None
        ),
        "projected_target_boundary_nodes": (
            "wall_and_outflow" if kind == PROJECTED_BOUNDARY_AUXILIARY else None
        ),
        "fixed_inflow_auxiliary_gradient": (
            "excluded" if kind == PROJECTED_BOUNDARY_AUXILIARY else None
        ),
        "uses_future_reference_at_inference": False,
        "changes_recurrence_state": False,
        "fixed_inflow_target_gradient": (
            "zero_through_projection"
            if primary_kind == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE
            else None
        ),
        "slip_wall_normal_target_gradient": (
            "zero_through_projection"
            if primary_kind == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE
            else None
        ),
        "wall_tangential_density_pressure_and_outflow_supervision": (
            "enabled" if primary_kind == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE else None
        ),
        "clipping_smoothing_or_limiter": False,
    }


def compare_initialized_boundary_contracts(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allow_legacy_metadata: bool,
) -> dict[str, Any]:
    """Compare deployed boundary semantics without rewriting parent metadata."""

    result = {
        "schema": "pcno_euler2d_boundary_contract_compatibility_v1",
        "compatible": False,
        "comparison": "strict",
        "saved_digest": digest_mapping(saved),
        "current_digest": digest_mapping(current),
        "accepted_differences": [],
    }
    if saved == current:
        result["compatible"] = True
        result["comparison"] = "exact"
        return result
    if not allow_legacy_metadata:
        return result

    saved_reduced = dict(saved)
    current_reduced = dict(current)
    accepted: list[dict[str, Any]] = []
    additive_fields = {
        "outflow_characteristic_treatment": "current-interior nodal extrapolation",
        "physical_conservation_claim": False,
        "projection_metric": None,
    }
    for field, expected in additive_fields.items():
        if field not in saved_reduced:
            if current_reduced.get(field) != expected:
                return result
            current_reduced.pop(field)
            accepted.append(
                {
                    "field": field,
                    "kind": "additive_descriptive_metadata",
                    "current_value": expected,
                }
            )

    if "training_objective" not in saved_reduced:
        objective = current_reduced.get("training_objective")
        legacy_normal_objective = (
            isinstance(objective, Mapping)
            and objective.get("primary_kind") == NORMAL_CLOSED_PRIMARY_OBJECTIVE
            and objective.get("auxiliary") == NO_BOUNDARY_AUXILIARY
            and saved_reduced.get("state_loss_mask") == "normal_nodes_only"
            and saved_reduced.get("teacher_input_closure") is True
            and saved_reduced.get("proposal_closure") is True
            and saved_reduced.get("recurrence_closure") is True
            and saved_reduced.get("future_reference_boundary_values") is False
        )
        if not legacy_normal_objective:
            return result
        current_reduced.pop("training_objective")
        accepted.append(
            {
                "field": "training_objective",
                "kind": (
                    "legacy_normal_closed_objective_recovered_from_operational_fields"
                ),
            }
        )

    rms_field = "reference_endpoint_boundary_primitive_rms_by_component"
    saved_rms = saved_reduced.get(rms_field)
    current_rms = current_reduced.get(rms_field)
    if (
        isinstance(saved_rms, Sequence)
        and not isinstance(saved_rms, (str, bytes))
        and isinstance(current_rms, Sequence)
        and not isinstance(current_rms, (str, bytes))
        and len(saved_rms) == len(current_rms)
    ):
        absolute_differences = [
            abs(float(saved_value) - float(current_value))
            for saved_value, current_value in zip(saved_rms, current_rms)
        ]
        maximum_difference = max(absolute_differences, default=0.0)
        if maximum_difference > 0.0:
            if maximum_difference > 2.0e-9:
                return result
            current_reduced[rms_field] = saved_rms
            accepted.append(
                {
                    "field": rms_field,
                    "kind": "bounded_diagnostic_roundoff",
                    "maximum_absolute_difference": maximum_difference,
                    "absolute_tolerance": 2.0e-9,
                }
            )

    if saved_reduced != current_reduced:
        return result
    result["compatible"] = True
    result["comparison"] = "legacy_metadata_compatible"
    result["accepted_differences"] = accepted
    return result


def build_boundary_contract(
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build and audit every graph policy before training starts."""

    if args.boundary_mode == "model_all_nodes":
        return {}, {
            "schema": "pcno_euler2d_boundary_contract_v1",
            "mode": "model_all_nodes",
            "trajectory_count": len(set(keys)),
            "policy_set_digest": None,
            "training_objective": boundary_training_objective_contract(args),
            "future_reference_boundary_values": False,
        }
    policies: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    reference_outflow_min = math.inf
    reference_density_min = math.inf
    reference_pressure_min = math.inf
    boundary_primitive_squared_sum = np.zeros(4, dtype=np.float64)
    boundary_primitive_max_abs = np.zeros(4, dtype=np.float64)
    boundary_primitive_value_count = 0
    for key in dict.fromkeys(str(value) for value in keys):
        if args.boundary_mode == MINIMUM_CHANGE_BOUNDARY_MODE:
            policy, record = build_graph_minimum_change_boundary_policy(
                store,
                key,
                device=device,
                rho_inf=args.boundary_rho_inf,
                p_inf=args.boundary_p_inf,
            )
        else:
            policy, record = build_graph_causal_boundary_policy(
                store,
                key,
                device=device,
                max_source_hops=args.boundary_max_source_hops,
                rho_inf=args.boundary_rho_inf,
                p_inf=args.boundary_p_inf,
            )
        policies[key] = policy
        metadata[key] = record
        states = store.states(key)
        audit_indices = sorted({0, states.shape[0] - 1})
        reference = torch.as_tensor(
            np.array(states[audit_indices], copy=True),
            dtype=torch.float32,
            device=device,
        )
        closed = close_boundary(
            reference, policy, gamma=float(store.manifest.get("gamma", 1.4))
        )
        node_type = torch.as_tensor(
            np.array(store.array(key, "node_type"), copy=True),
            dtype=torch.int64,
            device=device,
        ).reshape(-1)
        normal = node_type == NORMAL_NODE
        if not torch.equal(closed[:, normal], reference[:, normal]):
            raise RuntimeError(
                f"trajectory {key} boundary closure changed an interior node"
            )
        reference_primitive = conservative_to_primitive_torch(
            reference, gamma=float(store.manifest.get("gamma", 1.4))
        )
        closed_primitive = conservative_to_primitive_torch(
            closed, gamma=float(store.manifest.get("gamma", 1.4))
        )
        boundary_difference = (closed_primitive - reference_primitive)[:, ~normal]
        boundary_primitive_squared_sum += (
            boundary_difference.double().square().sum(dim=(0, 1)).cpu().numpy()
        )
        boundary_primitive_max_abs = np.maximum(
            boundary_primitive_max_abs,
            boundary_difference.double().abs().amax(dim=(0, 1)).cpu().numpy(),
        )
        boundary_primitive_value_count += int(
            boundary_difference.shape[0] * boundary_difference.shape[1]
        )
        admissibility = conservative_admissibility(
            closed.float(), gamma=float(store.manifest.get("gamma", 1.4))
        )
        if not bool(admissibility["admissible"].all()):
            raise ValueError(
                f"trajectory {key} boundary-closed reference endpoints are invalid"
            )
        outflow = boundary_outflow_normal_mach(closed.float(), policy)
        if outflow is None or not bool(torch.isfinite(outflow).all()):
            raise ValueError(f"trajectory {key} has invalid causal outflow Mach")
        reference_outflow_min = min(reference_outflow_min, float(outflow.min().cpu()))
        reference_density_min = min(
            reference_density_min,
            float(admissibility["density"].min().cpu()),
        )
        reference_pressure_min = min(
            reference_pressure_min,
            float(admissibility["pressure"].min().cpu()),
        )
    if reference_outflow_min <= 1.0:
        raise ValueError("boundary-closed reference endpoints are not supersonic")
    policy_digests = {key: record["policy_digest"] for key, record in metadata.items()}
    summary = {
        "schema": "pcno_euler2d_boundary_contract_v1",
        "mode": args.boundary_mode,
        "trajectory_count": len(metadata),
        "max_source_hops": (
            args.boundary_max_source_hops
            if args.boundary_mode == CAUSAL_BOUNDARY_MODE
            else None
        ),
        "rho_inf": args.boundary_rho_inf,
        "p_inf": args.boundary_p_inf,
        "fallback_target_count": 0,
        "projection_metric": (
            "euclidean_primitive_variables"
            if args.boundary_mode == MINIMUM_CHANGE_BOUNDARY_MODE
            else None
        ),
        "outflow_characteristic_treatment": (
            "preserve all outgoing modes while outward normal Mach exceeds one"
            if args.boundary_mode == MINIMUM_CHANGE_BOUNDARY_MODE
            else "current-interior nodal extrapolation"
        ),
        "policy_digests": policy_digests,
        "policy_set_digest": digest_mapping(policy_digests),
        "reference_endpoint_density_min": reference_density_min,
        "reference_endpoint_pressure_min": reference_pressure_min,
        "reference_endpoint_outflow_normal_mach_min": reference_outflow_min,
        "reference_endpoint_boundary_primitive_rms_by_component": (
            np.sqrt(
                boundary_primitive_squared_sum / boundary_primitive_value_count
            ).tolist()
        ),
        "reference_endpoint_boundary_primitive_max_abs_by_component": (
            boundary_primitive_max_abs.tolist()
        ),
        "teacher_input_closure": True,
        "proposal_closure": True,
        "recurrence_closure": True,
        "state_loss_mask": (
            "all_valid_nodes"
            if args.primary_objective == RAW_ALL_NODES_PRIMARY_OBJECTIVE
            else (
                "all_valid_nodes_after_minimum_change_projection"
                if args.primary_objective == LEARNED_DOFS_CLOSED_PRIMARY_OBJECTIVE
                else "normal_nodes_only"
            )
        ),
        "training_objective": boundary_training_objective_contract(args),
        "future_reference_boundary_values": False,
        "exact_dg_boundary_replay": False,
        "physical_conservation_claim": False,
        "clipping": False,
        "primitive_floors": False,
        "limiter": False,
        "smoothing": False,
    }
    return policies, summary


def multistep_future_comparison_count(
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    step_stride: int,
    rollout_steps: int,
) -> int:
    """Count legal post-anchor targets without dropping one-step presentations."""

    if step_stride < 1 or rollout_steps < 1:
        raise ValueError("step stride and rollout steps must be positive")
    comparisons = 0
    for key, time_index in pairs:
        num_frames = int(store.entry(str(key))["num_steps"])
        available_calls = (num_frames - 1 - int(time_index)) // step_stride
        if available_calls < 1:
            raise ValueError(
                f"presentation {key}:{time_index} has no stride-{step_stride} target"
            )
        comparisons += max(0, min(rollout_steps, available_calls) - 1)
    return comparisons


def differentiable_multistep_normal_terms(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    *,
    key: str,
    time_indices: Sequence[int],
    first_prediction: torch.Tensor,
    step_stride: int,
    rollout_steps: int,
    device: torch.device,
    boundary_policy: Mapping[str, Any] | None,
    recurrent_input: str,
) -> dict[str, torch.Tensor | int]:
    """Execute matched future calls and score their deployed normal-node states."""

    if rollout_steps < 2:
        raise ValueError("differentiable multistep terms require at least two calls")
    if first_prediction.ndim != 3 or first_prediction.shape[0] != len(time_indices):
        raise ValueError("first prediction and presentation batch do not align")
    if recurrent_input not in (
        ATTACHED_PREDICTION_MULTISTEP_INPUT,
        PROJECTED_TEACHER_MULTISTEP_INPUT,
    ):
        raise ValueError(f"unsupported multistep recurrent input: {recurrent_input}")

    start_indices = [int(value) for value in time_indices]
    recurrent_prediction = first_prediction
    loss_sum = first_prediction.new_zeros(())
    relative_sum = first_prediction.new_zeros(())
    comparison_count = 0
    admissible_count = 0
    num_frames = int(store.entry(str(key))["num_steps"])

    for call_index in range(2, rollout_steps + 1):
        keep = [
            index
            for index, start in enumerate(start_indices)
            if start + call_index * step_stride < num_frames
        ]
        if not keep:
            break
        if len(keep) != len(start_indices):
            recurrent_prediction = recurrent_prediction[keep]
            start_indices = [start_indices[index] for index in keep]

        teacher_indices = [
            start + (call_index - 1) * step_stride for start in start_indices
        ]
        horizon_sample = store.tensor_batch(
            key,
            teacher_indices,
            step_stride=step_stride,
            device=device,
        )
        recurrent_current = (
            recurrent_prediction
            if recurrent_input == ATTACHED_PREDICTION_MULTISTEP_INPUT
            else horizon_sample["current"]
        )
        recurrent_prediction, _, _ = contract_forward_sample(
            model,
            horizon_sample,
            recurrent_current,
            boundary_policy=boundary_policy,
        )
        if not bool(torch.isfinite(recurrent_prediction).all()):
            raise FloatingPointError(
                f"nonfinite differentiable rollout at call {call_index} "
                f"for trajectory {key}"
            )
        normal_mask = normal_node_mask(
            horizon_sample["node_type"], horizon_sample["node_mask"]
        )
        horizon_loss = weighted_scaled_mse(
            recurrent_prediction,
            horizon_sample["target"],
            horizon_sample["node_weights"],
            normal_mask,
            model.state_scale,
        )
        horizon_relative_l2 = weighted_scaled_relative_l2(
            recurrent_prediction,
            horizon_sample["target"],
            horizon_sample["node_weights"],
            normal_mask,
            model.state_scale,
        )
        horizon_count = len(start_indices)
        loss_sum = loss_sum + horizon_count * horizon_loss
        relative_sum = relative_sum + horizon_count * horizon_relative_l2
        comparison_count += horizon_count
        with torch.no_grad():
            admissible = conservative_admissibility(
                recurrent_prediction.float(), gamma=model.gamma
            )["admissible"].all(dim=1)
            admissible_count += int(admissible.sum().cpu())

    return {
        "loss_sum": loss_sum,
        "relative_l2_sum": relative_sum,
        "comparisons": comparison_count,
        "admissible_comparisons": admissible_count,
    }


def boundary_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    model: PCNOEuler2DResidual,
    *,
    boundary_policy: Mapping[str, Any] | None,
    kind: str,
    near_boundary_hops: int,
    raw_prediction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one legal boundary auxiliary without changing the recurrent state."""

    if kind == NO_BOUNDARY_AUXILIARY:
        return prediction.new_zeros(())
    if boundary_policy is None:
        raise ValueError("boundary auxiliaries require a causal boundary policy")
    if kind == PROJECTED_BOUNDARY_AUXILIARY:
        objective_prediction = prediction
        projected_target = close_boundary(
            target,
            boundary_policy,
            gamma=model.gamma,
        )
        loss_mask = torch.zeros_like(sample["node_mask"])
        target_nodes = boundary_policy["target_nodes"]
        loss_mask[:, target_nodes] = sample["node_mask"][:, target_nodes]
    elif kind == NEAR_BOUNDARY_AUXILIARY:
        objective_prediction = prediction
        loss_mask = boundary_band_normal_node_mask(
            sample["node_type"],
            sample["node_mask"],
            sample["directed_edges"],
            max_hops=near_boundary_hops,
        )
        projected_target = target
    elif kind == RAW_BOUNDARY_REFERENCE_AUXILIARY:
        if raw_prediction is None:
            raise ValueError("raw boundary reference auxiliary requires raw prediction")
        objective_prediction = raw_prediction
        loss_mask = sample["node_mask"] - normal_node_mask(
            sample["node_type"], sample["node_mask"]
        )
        projected_target = target
    else:
        raise ValueError(f"unsupported boundary auxiliary: {kind}")
    return weighted_scaled_mse(
        objective_prediction,
        projected_target,
        sample["node_weights"],
        loss_mask,
        model.state_scale,
    )


def selection_tuple(
    rollout: Mapping[str, Any],
    one_step: Mapping[str, float],
    *,
    mode: str = HISTORICAL_SELECTION_MODE,
    short_horizon: int = 20,
    parity_max_rollout_relative_l2: float | None = None,
    parity_max_one_step_relative_l2: float | None = None,
) -> tuple[float, ...]:
    def finite_error(value: Any) -> float:
        resolved = float(value) if value is not None else float("nan")
        return resolved if math.isfinite(resolved) else float(np.finfo(np.float64).max)

    raw_error = rollout["mean_selection_relative_l2"]
    error = finite_error(raw_error)
    endpoint = rollout.get("mean_endpoint_relative_l2", {})
    h20 = endpoint.get("20")
    h20_error = finite_error(h20) if h20 is not None else error
    if mode == INTERIOR_SELECTION_MODE:
        if rollout.get("parity") is not None:
            raise ValueError(
                "interior-rollout selection cannot use the historical parity gate"
            )
        normal_error = finite_error(rollout.get("mean_final_normal_relative_l2"))
        normal_endpoints = rollout.get("mean_endpoint_normal_relative_l2", {})
        short_error = finite_error(normal_endpoints.get(str(short_horizon)))
        one_step_normal = finite_error(one_step.get("normal_relative_l2"))
        one_step_all = finite_error(one_step.get("all_relative_l2"))
        return (
            float(rollout["completion_rate"] == 1.0),
            float(rollout["completion_rate"]),
            float(rollout["mean_survival_fraction"]),
            -normal_error,
            -short_error,
            -one_step_normal,
            -error,
            -one_step_all,
        )
    if mode != HISTORICAL_SELECTION_MODE:
        raise ValueError(f"unsupported checkpoint selection mode: {mode}")
    base = (
        float(rollout["completion_rate"] == 1.0),
        float(rollout["completion_rate"]),
        float(rollout["mean_survival_fraction"]),
        -error,
        -h20_error,
        -float(one_step["relative_l2"]),
    )
    parity = rollout.get("parity")
    if parity is None:
        return base
    if (
        parity_max_rollout_relative_l2 is None
        or parity_max_one_step_relative_l2 is None
    ):
        raise ValueError("parity metrics require both registered thresholds")
    parity_error = parity["mean_relative_l2"]
    if "all_relative_l2" not in one_step:
        raise ValueError("parity selection requires all-node one-step relative L2")
    parity_one_step = float(one_step["all_relative_l2"])
    parity_passed = (
        float(parity["completion_rate"]) == 1.0
        and parity_error is not None
        and float(parity_error) <= parity_max_rollout_relative_l2
        and parity_one_step <= parity_max_one_step_relative_l2
    )
    all_completed = rollout["completion_rate"] == 1.0
    eligible = all_completed and parity_passed
    return (
        float(eligible),
        float(all_completed),
        float(parity_passed),
        *base[1:-1],
        -parity_one_step,
        base[-1],
    )


def tiny_fit_snapshot(
    initial: Mapping[str, float],
    current: Mapping[str, float],
    *,
    relative_l2_threshold: float,
    loss_ratio_threshold: float,
) -> dict[str, Any]:
    initial_loss = float(initial["loss"])
    current_loss = float(current["loss"])
    loss_ratio = (
        current_loss / initial_loss
        if initial_loss > 0.0
        else (0.0 if current_loss == 0.0 else float(np.finfo(np.float64).max))
    )
    relative_metric = (
        "normal_relative_l2"
        if current.get("normal_relative_l2") is not None
        else "relative_l2"
    )
    relative_l2 = float(current[relative_metric])
    admissible_fraction = float(current["admissible_fraction"])
    return {
        "metrics": dict(current),
        "loss_ratio": loss_ratio,
        "relative_metric": relative_metric,
        "selection_relative_l2": relative_l2,
        "relative_l2_threshold": float(relative_l2_threshold),
        "loss_ratio_threshold": float(loss_ratio_threshold),
        "passed": (
            relative_l2 <= relative_l2_threshold
            and loss_ratio <= loss_ratio_threshold
            and admissible_fraction == 1.0
        ),
    }


def tiny_fit_selection(snapshot: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = snapshot["metrics"]
    relative_ratio = float(snapshot["selection_relative_l2"]) / float(
        snapshot["relative_l2_threshold"]
    )
    loss_ratio = float(snapshot["loss_ratio"]) / float(snapshot["loss_ratio_threshold"])
    maximum_ratio = max(relative_ratio, loss_ratio)
    return (
        float(bool(snapshot["passed"])),
        float(metrics["admissible_fraction"]),
        -maximum_ratio,
        -relative_ratio,
        -loss_ratio,
    )


def clone_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def checkpoint_payload(
    *,
    epoch: int,
    model: PCNOEuler2DResidual,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalization: Euler2DNormalization,
    train_keys: Sequence[str],
    val_keys: Sequence[str],
    test_keys: Sequence[str],
    data_contract: Mapping[str, Any],
    boundary_contract: Mapping[str, Any],
    exposure_contract: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    resolved_target_contract: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    args: argparse.Namespace,
    best_selection: Sequence[float] | None,
    best_epoch: int | None,
    parent_checkpoint: Mapping[str, Any] | None,
    initialization_transition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "train_keys": list(train_keys),
        "val_keys": list(val_keys),
        "test_keys": list(test_keys),
        "split_mode": args.split_mode,
        "data_contract": dict(data_contract),
        "boundary_contract": dict(boundary_contract),
        "exposure_contract": dict(exposure_contract),
        "source_snapshot": dict(source_snapshot),
        "target_contract": dict(resolved_target_contract),
        "target_kind": RESIDUAL_TARGET_KIND,
        "normalization_digest": data_contract["normalization_digest"],
        "data_manifest_digest": store.manifest_digest,
        "step_stride": int(args.step_stride),
        "model_node_type_input": args.model_node_type_input,
        "node_type_channel_control": args.node_type_channel_control,
        "initialization_control": dict(model.initialization_control),
        "training_args": jsonable_args(args),
        "config_digest": digest_mapping(jsonable_args(args)),
        "best_selection": None if best_selection is None else list(best_selection),
        "best_epoch": best_epoch,
        "checkpoint_selection_contract": checkpoint_selection_contract(args),
        "parent_checkpoint": parent_checkpoint,
        "initialization_transition": initialization_transition,
        "git": git_state(),
        "boundary_mode": args.boundary_mode,
        "raw_recurrence": args.boundary_mode == "model_all_nodes",
        "autonomous_recurrence": True,
        "inference_interventions": {
            "future_reference_boundary_values": False,
            "clipping": False,
            "primitive_floors": False,
            "limiter": False,
            "smoothing": False,
            "boundary_decode_reencode_closure": (
                args.boundary_mode != "model_all_nodes"
            ),
            "decode_reencode_projection": (args.boundary_mode != "model_all_nodes"),
        },
        "weight_provenance": normalization.weight_provenance,
        "generated_state_exposure": {
            "weight": float(args.generated_state_exposure_weight),
            "depth": 1 if args.generated_state_exposure_weight > 0.0 else 0,
            "first_call_gradient": "detached",
            "clean_one_step_anchor": (
                1.0 - float(args.generated_state_exposure_weight)
            ),
            "raw_generated_state": True,
        },
        "differentiable_multistep": dict(exposure_contract["differentiable_multistep"]),
        "checkpoint_role": "training",
        "resume_supported": True,
    }


def train_epoch(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step_stride: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    batch_rng: np.random.Generator,
    device: torch.device,
    amp: str,
    input_noise_std: float,
    generated_state_exposure_weight: float,
    multistep_loss_steps: int,
    multistep_loss_weight: float,
    multistep_recurrent_input: str,
    primary_objective: str,
    boundary_auxiliary: str,
    boundary_auxiliary_weight: float,
    near_boundary_hops: int,
    gradient_clip: float,
    noise_generator: torch.Generator,
    boundary_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    model.train()
    training_start = perf_counter()
    loss_sum = 0.0
    relative_error_sum = 0.0
    clean_loss_sum = 0.0
    clean_relative_error_sum = 0.0
    boundary_auxiliary_loss_sum = 0.0
    generated_loss_sum = 0.0
    generated_relative_error_sum = 0.0
    generated_input_error_sum = 0.0
    multistep_loss_sum = 0.0
    multistep_relative_error_sum = 0.0
    multistep_comparisons = 0
    multistep_admissible_comparisons = 0
    gradient_norm_sum = torch.zeros((), dtype=torch.float32, device=device)
    gradient_norm_max = torch.zeros((), dtype=torch.float32, device=device)
    clipped_steps = torch.zeros((), dtype=torch.int64, device=device)
    effective_batch_size = batch_size * gradient_accumulation_steps
    requested_future_slots = len(pairs) * max(0, multistep_loss_steps - 1)
    expected_future_comparisons = (
        multistep_future_comparison_count(
            store,
            pairs,
            step_stride=step_stride,
            rollout_steps=multistep_loss_steps,
        )
        if multistep_loss_steps > 1
        else 0
    )
    if multistep_loss_steps > 1 and expected_future_comparisons == 0:
        raise ValueError("no future target is available for multistep training")
    multistep_coverage_fraction = (
        expected_future_comparisons / requested_future_slots
        if requested_future_slots
        else 0.0
    )
    multistep_objective_normalizer = (
        1.0 + multistep_loss_weight * multistep_coverage_fraction
    )
    optimizer_batches = homogeneous_presentation_batches(
        pairs,
        batch_size=effective_batch_size,
        rng=batch_rng,
    )
    for key, effective_time_indices in optimizer_batches:
        optimizer.zero_grad(set_to_none=True)
        effective_sample_count = len(effective_time_indices)
        for microbatch_start in range(0, effective_sample_count, batch_size):
            time_indices = effective_time_indices[
                microbatch_start : microbatch_start + batch_size
            ]
            sample = store.tensor_batch(
                key,
                time_indices,
                step_stride=step_stride,
                device=device,
            )
            boundary_policy = resolve_boundary_policy(boundary_policies, key)
            batch_loss_mask = (
                normal_node_mask(sample["node_type"], sample["node_mask"])
                if boundary_policy is not None
                and primary_objective == NORMAL_CLOSED_PRIMARY_OBJECTIVE
                else sample["node_mask"]
            )
            if input_noise_std > 0.0:
                current = apply_admissible_primitive_noise(
                    sample["current"],
                    input_noise_std,
                    gamma=model.gamma,
                    generator=noise_generator,
                )
            else:
                current = sample["current"]
            generated_current = None
            generated_input_relative_l2 = None
            multistep_terms = None
            if generated_state_exposure_weight > 0.0:
                previous_sample = store.tensor_batch(
                    key,
                    [index - step_stride for index in time_indices],
                    step_stride=step_stride,
                    device=device,
                )
                with torch.no_grad(), autocast_context(device, amp):
                    generated_current, _, _ = contract_forward_sample(
                        model,
                        previous_sample,
                        previous_sample["current"],
                        boundary_policy=boundary_policy,
                    )
                    generated_current = generated_current.detach()
                    _, generated_input_relative_l2 = pair_metrics(
                        generated_current.float(),
                        sample["current"],
                        sample,
                        model,
                        loss_node_mask=batch_loss_mask,
                    )
                generated_admissibility = conservative_admissibility(
                    generated_current.float(),
                    gamma=model.gamma,
                )
                if not bool(generated_admissibility["admissible"].all()):
                    raise FloatingPointError(
                        "raw generated-state exposure produced an inadmissible "
                        f"input for trajectory {key} frames {time_indices}"
                    )
            with autocast_context(device, amp):
                prediction, raw_prediction, _ = contract_forward_sample(
                    model,
                    sample,
                    current,
                    boundary_policy=boundary_policy,
                )
                clean_loss, clean_relative_l2 = primary_training_metrics(
                    prediction,
                    raw_prediction,
                    sample["target"],
                    sample,
                    model,
                    boundary_policy=boundary_policy,
                    primary_objective=primary_objective,
                )
                auxiliary_loss = boundary_auxiliary_loss(
                    prediction,
                    sample["target"],
                    sample,
                    model,
                    boundary_policy=boundary_policy,
                    kind=boundary_auxiliary,
                    near_boundary_hops=near_boundary_hops,
                    raw_prediction=raw_prediction,
                )
                clean_objective = (
                    clean_loss + boundary_auxiliary_weight * auxiliary_loss
                )
                if generated_current is None:
                    generated_loss = None
                    generated_relative_l2 = None
                    loss = clean_objective
                    relative_l2 = clean_relative_l2
                else:
                    (
                        generated_prediction,
                        raw_generated_prediction,
                        _,
                    ) = contract_forward_sample(
                        model,
                        sample,
                        generated_current,
                        boundary_policy=boundary_policy,
                    )
                    generated_loss, generated_relative_l2 = primary_training_metrics(
                        generated_prediction,
                        raw_generated_prediction,
                        sample["target"],
                        sample,
                        model,
                        boundary_policy=boundary_policy,
                        primary_objective=primary_objective,
                    )
                    loss = (
                        1.0 - generated_state_exposure_weight
                    ) * clean_loss + generated_state_exposure_weight * generated_loss
                    relative_l2 = (
                        (1.0 - generated_state_exposure_weight) * clean_relative_l2
                        + generated_state_exposure_weight * generated_relative_l2
                    )
                if multistep_loss_steps > 1:
                    if generated_current is not None:
                        raise RuntimeError(
                            "attached multistep loss cannot share detached exposure"
                        )
                    multistep_terms = differentiable_multistep_normal_terms(
                        model,
                        store,
                        key=key,
                        time_indices=time_indices,
                        first_prediction=prediction,
                        step_stride=step_stride,
                        rollout_steps=multistep_loss_steps,
                        device=device,
                        boundary_policy=boundary_policy,
                        recurrent_input=multistep_recurrent_input,
                    )
                    requested_batch_slots = len(time_indices) * (
                        multistep_loss_steps - 1
                    )
                    padded_future_loss = (
                        multistep_terms["loss_sum"] / requested_batch_slots
                    )
                    padded_future_relative_l2 = (
                        multistep_terms["relative_l2_sum"] / requested_batch_slots
                    )
                    loss = (
                        clean_objective + multistep_loss_weight * padded_future_loss
                    ) / multistep_objective_normalizer
                    relative_l2 = (
                        clean_relative_l2
                        + multistep_loss_weight * padded_future_relative_l2
                    ) / multistep_objective_normalizer
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"nonfinite training loss for trajectory {key} frames "
                    f"{time_indices}"
                )
            current_batch_size = len(time_indices)
            microbatch_weight = current_batch_size / effective_sample_count
            scaler.scale(loss * microbatch_weight).backward()
            loss_sum += float(loss.detach().cpu()) * current_batch_size
            relative_error_sum += float(relative_l2.detach().cpu()) * current_batch_size
            clean_loss_sum += float(clean_loss.detach().cpu()) * current_batch_size
            clean_relative_error_sum += (
                float(clean_relative_l2.detach().cpu()) * current_batch_size
            )
            boundary_auxiliary_loss_sum += (
                float(auxiliary_loss.detach().cpu()) * current_batch_size
            )
            if generated_loss is not None:
                generated_loss_sum += (
                    float(generated_loss.detach().cpu()) * current_batch_size
                )
                generated_relative_error_sum += (
                    float(generated_relative_l2.detach().cpu()) * current_batch_size
                )
                generated_input_error_sum += (
                    float(generated_input_relative_l2.detach().cpu())
                    * current_batch_size
                )
            if multistep_terms is not None:
                multistep_loss_sum += float(multistep_terms["loss_sum"].detach().cpu())
                multistep_relative_error_sum += float(
                    multistep_terms["relative_l2_sum"].detach().cpu()
                )
                multistep_comparisons += int(multistep_terms["comparisons"])
                multistep_admissible_comparisons += int(
                    multistep_terms["admissible_comparisons"]
                )
        scaler.unscale_(optimizer)
        maximum_norm = gradient_clip if gradient_clip > 0.0 else math.inf
        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                maximum_norm,
                error_if_nonfinite=True,
            )
            .detach()
            .float()
        )
        gradient_norm_sum += gradient_norm
        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm)
        if gradient_clip > 0.0:
            clipped_steps += (gradient_norm > gradient_clip).to(torch.int64)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    optimizer_steps = len(optimizer_batches)
    training_seconds = perf_counter() - training_start
    if multistep_comparisons != expected_future_comparisons:
        raise RuntimeError(
            "multistep target accounting changed during the training epoch: "
            f"expected {expected_future_comparisons}, got {multistep_comparisons}"
        )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    if not parameters_finite:
        raise FloatingPointError("model parameters became nonfinite")
    return {
        "loss": loss_sum / len(pairs),
        "relative_l2": relative_error_sum / len(pairs),
        "clean_loss": clean_loss_sum / len(pairs),
        "clean_relative_l2": clean_relative_error_sum / len(pairs),
        "boundary_auxiliary_loss": (
            boundary_auxiliary_loss_sum / len(pairs)
            if boundary_auxiliary != NO_BOUNDARY_AUXILIARY
            else None
        ),
        "boundary_auxiliary": boundary_auxiliary,
        "boundary_auxiliary_weight": boundary_auxiliary_weight,
        "primary_objective": primary_objective,
        "near_boundary_hops": near_boundary_hops,
        "generated_loss": (
            generated_loss_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_relative_l2": (
            generated_relative_error_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_input_relative_l2": (
            generated_input_error_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_state_exposure_weight": generated_state_exposure_weight,
        "multistep_loss": (
            multistep_loss_sum / multistep_comparisons
            if multistep_comparisons
            else None
        ),
        "multistep_relative_l2": (
            multistep_relative_error_sum / multistep_comparisons
            if multistep_comparisons
            else None
        ),
        "multistep_loss_steps": multistep_loss_steps,
        "multistep_loss_weight": multistep_loss_weight,
        "multistep_first_call_gradient": (
            (
                "attached"
                if multistep_recurrent_input == ATTACHED_PREDICTION_MULTISTEP_INPUT
                else "not_connected_projected_teacher_control"
            )
            if multistep_loss_steps > 1
            else None
        ),
        "multistep_recurrent_input": (
            multistep_recurrent_input if multistep_loss_steps > 1 else None
        ),
        "multistep_future_node_population": (
            "normal_nodes_only" if multistep_loss_steps > 1 else None
        ),
        "multistep_future_comparisons": multistep_comparisons,
        "multistep_requested_future_slots": requested_future_slots,
        "multistep_future_coverage_fraction": multistep_coverage_fraction,
        "multistep_objective_normalizer": multistep_objective_normalizer,
        "multistep_recurrent_admissible_fraction": (
            multistep_admissible_comparisons / multistep_comparisons
            if multistep_comparisons
            else None
        ),
        "presentations": len(pairs),
        "model_calls": len(pairs) + multistep_comparisons,
        "optimizer_steps": optimizer_steps,
        "microbatch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "gradient_norm_mean": float((gradient_norm_sum / optimizer_steps).cpu()),
        "gradient_norm_max": float(gradient_norm_max.cpu()),
        "gradient_clip_fraction": float((clipped_steps / optimizer_steps).cpu()),
        "parameters_finite": parameters_finite,
        "training_seconds": training_seconds,
        "presentations_per_second": len(pairs) / training_seconds,
        "model_calls_per_second": (len(pairs) + multistep_comparisons)
        / training_seconds,
        "optimizer_steps_per_second": optimizer_steps / training_seconds,
    }


def fixed_horizon_block_presentations(
    store: PCNOEuler2DShardStore,
    train_keys: Sequence[str],
    *,
    step_stride: int,
    rollout_steps: int,
    count: int,
    block_size: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Build one deterministic bank with exactly one optimizer block per trajectory."""

    if step_stride < 1 or rollout_steps < 2:
        raise ValueError("fixed horizon blocks require a positive stride and K>1")
    if count < 1 or block_size < 1 or count % block_size != 0:
        raise ValueError("fixed horizon count must be a positive block multiple")
    block_count = count // block_size
    eligible: list[tuple[str, int]] = []
    for raw_key in sorted(str(value) for value in train_keys):
        num_frames = int(store.entry(raw_key)["num_steps"])
        legal_start_count = num_frames - rollout_steps * step_stride
        if legal_start_count >= block_size:
            eligible.append((raw_key, legal_start_count))
    if block_count > len(eligible):
        raise ValueError(
            "fixed horizon blocks require one eligible trajectory per effective batch: "
            f"requested {block_count}, available {len(eligible)}"
        )

    rng = np.random.default_rng(seed)
    selected = [
        eligible[index] for index in rng.permutation(len(eligible))[:block_count]
    ]
    pairs: list[tuple[str, int]] = []
    for key, legal_start_count in selected:
        starts = rng.choice(legal_start_count, size=block_size, replace=False)
        pairs.extend((key, int(start)) for start in starts)
    return pairs


def presentation_stream_sha256(pairs: Sequence[tuple[str, int]]) -> str:
    """Hash the ordered trajectory/time presentation stream."""

    payload = json.dumps(
        [[str(key), int(time_index)] for key, time_index in pairs],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def epoch_presentations(
    store: PCNOEuler2DShardStore,
    train_keys: Sequence[str],
    *,
    args: argparse.Namespace,
    epoch: int,
    tiny_bank: Sequence[tuple[str, int]] | None,
) -> list[tuple[str, int]]:
    """Resolve one epoch's immutable presentation semantics."""

    if tiny_bank is not None:
        repetitions = math.ceil(args.presentations_per_epoch / len(tiny_bank))
        return list((list(tiny_bank) * repetitions)[: args.presentations_per_epoch])
    if args.presentation_mode == FIXED_HORIZON_BLOCKS_PRESENTATION_MODE:
        return fixed_horizon_block_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            rollout_steps=args.multistep_loss_steps,
            count=args.presentations_per_epoch,
            block_size=args.batch_size * args.gradient_accumulation_steps,
            seed=args.seed,
        )
    if args.presentation_mode == "full_coverage":
        return full_coverage_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            rng=np.random.default_rng(args.seed + epoch),
        )
    return balanced_presentations(
        store,
        train_keys,
        step_stride=args.step_stride,
        count=args.presentations_per_epoch,
        rng=np.random.default_rng(args.seed + epoch),
        minimum_time_index=(
            args.step_stride if args.generated_state_exposure_weight > 0.0 else 0
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    process_start = perf_counter()
    args = parse_args(argv)
    device = select_device(args.device)
    validate_args(args, device)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    store = PCNOEuler2DShardStore(args.data_dir)
    args.boundary_field_names = list(
        boundary_field_names_for_store(store, args.boundary_field_mode)
    )
    args.boundary_residual_names = list(
        boundary_field_names_for_store(
            store,
            (
                BOUNDARY_FIELD_SEMANTIC_COLLAR
                if args.boundary_residual_mode
                == BOUNDARY_RESIDUAL_SEMANTIC_COLLAR_POINTWISE
                else BOUNDARY_FIELD_NONE
            ),
        )
    )
    checkpoint_path = args.resume_checkpoint or args.init_checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if args.resume_checkpoint is not None:
        assert_resume_training_args(checkpoint, args)

    initialization_transition: dict[str, Any] | None = None
    fresh_boundary_transition = False
    if checkpoint is not None:
        if args.resume_checkpoint is not None:
            saved_transition = checkpoint.get("initialization_transition")
            if saved_transition is not None:
                if not isinstance(saved_transition, Mapping):
                    raise ValueError(
                        "checkpoint initialization transition is malformed"
                    )
                initialization_transition = dict(saved_transition)
        else:
            initialization_transition = resolve_initialization_transition(
                checkpoint,
                args=args,
            )
            fresh_boundary_transition = initialization_transition is not None
        assert_checkpoint_contract(
            checkpoint,
            store=store,
            args=args,
            initialization_transition=(
                initialization_transition if fresh_boundary_transition else None
            ),
        )
        train_keys = [str(key) for key in checkpoint["train_keys"]]
        val_keys = [str(key) for key in checkpoint["val_keys"]]
        test_keys = [str(key) for key in checkpoint.get("test_keys", [])]
        normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    else:
        if args.split_mode == "manifest":
            train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
        else:
            train_keys, val_keys = stratified_train_val_split(
                store,
                val_count=args.val_count,
                seed=args.split_seed,
            )
            test_keys = []
        normalization = fit_normalization(
            store,
            train_keys,
            step_stride=args.step_stride,
            time_stride=args.stats_time_stride,
            gamma=float(store.manifest.get("gamma", 1.4)),
            mach_scale_floor=args.mach_scale_floor,
        )
    data_contract = build_data_contract_summary(
        store,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
        normalization=normalization,
    )
    boundary_policies, boundary_contract = build_boundary_contract(
        store,
        [*train_keys, *val_keys],
        args=args,
        device=device,
    )
    boundary_contract_compatibility: dict[str, Any] | None = None
    if fresh_boundary_transition:
        if initialization_transition is None:
            raise RuntimeError("fresh boundary transition metadata was not resolved")
        initialization_transition["target_boundary_contract_digest"] = digest_mapping(
            boundary_contract
        )
    if checkpoint is not None and checkpoint.get("data_contract") not in (
        None,
        data_contract,
    ):
        raise ValueError("checkpoint and current resolved data contracts differ")
    if checkpoint is not None and not fresh_boundary_transition:
        saved_boundary_contract = checkpoint.get("boundary_contract")
        if saved_boundary_contract is not None:
            if not isinstance(saved_boundary_contract, Mapping):
                raise ValueError("checkpoint boundary contract is malformed")
            boundary_contract_compatibility = compare_initialized_boundary_contracts(
                saved_boundary_contract,
                boundary_contract,
                allow_legacy_metadata=args.init_checkpoint is not None,
            )
            if not boundary_contract_compatibility["compatible"]:
                raise ValueError("checkpoint and current boundary contracts differ")
    if args.rollout_val_count > len(val_keys):
        raise ValueError("--rollout-val-count exceeds the validation split")
    rollout_selection_seed = args.seed + 1991
    if args.rollout_val_count == len(val_keys):
        rollout_keys = list(val_keys)
    else:
        _, rollout_keys = stratified_train_val_split(
            store,
            val_count=args.rollout_val_count,
            seed=rollout_selection_seed,
            keys=val_keys,
        )
    if args.parity_rollout_keys:
        outside_validation = sorted(set(args.parity_rollout_keys) - set(val_keys))
        outside_rollout = sorted(set(args.parity_rollout_keys) - set(rollout_keys))
        if outside_validation or outside_rollout:
            raise ValueError(
                "parity keys must be validation rollout keys: "
                f"outside_validation={outside_validation}, "
                f"outside_rollout={outside_rollout}"
            )
    model = build_model(
        args,
        normalization,
        zero_initialize=checkpoint is None,
    ).to(device)
    tiny_bank = None
    if args.tiny_pairs:
        tiny_bank = fixed_tiny_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            count=args.tiny_pairs,
            seed=args.seed + 17,
        )
    accounting_pairs = epoch_presentations(
        store,
        train_keys,
        args=args,
        epoch=0,
        tiny_bank=tiny_bank,
    )
    accounting_presentation_stream_sha256 = presentation_stream_sha256(accounting_pairs)
    resolved_presentations_per_epoch = len(accounting_pairs)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    optimizer_steps_per_epoch = homogeneous_optimizer_step_count(
        accounting_pairs,
        batch_size=effective_batch_size,
    )
    total_optimizer_steps = args.epochs * optimizer_steps_per_epoch
    multistep_future_comparisons_per_epoch = (
        multistep_future_comparison_count(
            store,
            accounting_pairs,
            step_stride=args.step_stride,
            rollout_steps=args.multistep_loss_steps,
        )
        if args.multistep_loss_steps > 1
        else 0
    )
    multistep_requested_future_slots_per_epoch = len(accounting_pairs) * max(
        0, args.multistep_loss_steps - 1
    )
    multistep_future_coverage_fraction = (
        multistep_future_comparisons_per_epoch
        / multistep_requested_future_slots_per_epoch
        if multistep_requested_future_slots_per_epoch
        else 0.0
    )
    multistep_objective_normalizer = (
        1.0 + args.multistep_loss_weight * multistep_future_coverage_fraction
    )
    differentiable_multistep_contract = {
        "enabled": args.multistep_loss_steps > 1,
        "rollout_steps": args.multistep_loss_steps,
        "future_loss_weight": args.multistep_loss_weight,
        "recurrent_input": (
            args.multistep_recurrent_input if args.multistep_loss_steps > 1 else None
        ),
        "future_node_population": (
            "normal_nodes_only" if args.multistep_loss_steps > 1 else None
        ),
        "first_call_gradient": (
            (
                "attached"
                if args.multistep_recurrent_input == ATTACHED_PREDICTION_MULTISTEP_INPUT
                else "not_connected_projected_teacher_control"
            )
            if args.multistep_loss_steps > 1
            else None
        ),
        "recurrent_state": (
            (
                "deployed_post_boundary_projection"
                if args.multistep_recurrent_input == ATTACHED_PREDICTION_MULTISTEP_INPUT
                else "projected_reference_current_state_training_control"
            )
            if args.multistep_loss_steps > 1
            else None
        ),
        "future_reference_as_model_input": (
            args.multistep_loss_steps > 1
            and args.multistep_recurrent_input == PROJECTED_TEACHER_MULTISTEP_INPUT
        ),
        "future_reference_use": (
            (
                "loss_targets_only"
                if args.multistep_recurrent_input == ATTACHED_PREDICTION_MULTISTEP_INPUT
                else "projected_training_input_and_loss_target_only"
            )
            if args.multistep_loss_steps > 1
            else None
        ),
        "one_step_presentations_retained": len(accounting_pairs),
        "future_comparisons_per_epoch": multistep_future_comparisons_per_epoch,
        "requested_future_slots_per_epoch": (
            multistep_requested_future_slots_per_epoch
        ),
        "future_coverage_fraction": multistep_future_coverage_fraction,
        "objective_normalizer": multistep_objective_normalizer,
        "clean_one_step_coefficient": 1.0 / multistep_objective_normalizer,
        "realized_future_coefficient": (
            args.multistep_loss_weight
            * multistep_future_coverage_fraction
            / multistep_objective_normalizer
        ),
        "model_calls_per_epoch": (
            len(accounting_pairs) + multistep_future_comparisons_per_epoch
        ),
        "clipping_smoothing_or_limiter": False,
    }
    exposure_contract = {
        "schema": "pcno_euler2d_exposure_contract_v1",
        "presentation_mode": args.presentation_mode,
        "coverage_passes_requested": (
            args.epochs if args.presentation_mode == "full_coverage" else None
        ),
        "unique_eligible_train_transitions": (
            resolved_presentations_per_epoch
            if args.presentation_mode == "full_coverage"
            else None
        ),
        "presentations_per_epoch": resolved_presentations_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "requested_presentations": args.epochs * resolved_presentations_per_epoch,
        "requested_optimizer_steps": total_optimizer_steps,
        "presentation_stream_sha256": accounting_presentation_stream_sha256,
        "batch_size": args.batch_size,
        "microbatch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "trajectory_homogeneous_batches": True,
        "fixed_horizon_block_size": (
            effective_batch_size
            if args.presentation_mode == FIXED_HORIZON_BLOCKS_PRESENTATION_MODE
            else None
        ),
        "fixed_horizon_block_count": (
            optimizer_steps_per_epoch
            if args.presentation_mode == FIXED_HORIZON_BLOCKS_PRESENTATION_MODE
            else None
        ),
        "target_comparisons_per_epoch": (
            len(accounting_pairs) + multistep_future_comparisons_per_epoch
        ),
        "differentiable_multistep": differentiable_multistep_contract,
    }
    resolved_target_contract = target_contract()
    if checkpoint is not None:
        saved_target_contract = checkpoint.get("target_contract")
        if saved_target_contract not in (None, resolved_target_contract):
            raise ValueError("checkpoint and current target contracts differ")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        scheduler_contract = {
            "name": "constant",
            "total_optimizer_steps": total_optimizer_steps,
        }
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=total_optimizer_steps,
            div_factor=args.lr_div_factor,
            final_div_factor=args.lr_final_div_factor,
        )
        scheduler_contract = {
            "name": "onecycle",
            "total_optimizer_steps": total_optimizer_steps,
            "div_factor": args.lr_div_factor,
            "final_div_factor": args.lr_final_div_factor,
        }
    else:
        if total_optimizer_steps < 2:
            raise ValueError("warmup-cosine requires at least two optimizer steps")
        warmup_steps = min(
            total_optimizer_steps - 1,
            max(1, round(args.warmup_fraction * total_optimizer_steps)),
        )
        minimum_factor = args.min_learning_rate / args.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(
                step,
                total_steps=total_optimizer_steps,
                warmup_steps=warmup_steps,
                start_factor=args.warmup_start_factor,
                minimum_factor=minimum_factor,
            ),
        )
        scheduler_contract = {
            "name": "warmup_cosine",
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "warmup_fraction_resolved": warmup_steps / total_optimizer_steps,
            "warmup_start_factor": args.warmup_start_factor,
            "peak_learning_rate": args.learning_rate,
            "minimum_learning_rate": args.min_learning_rate,
        }
    start_epoch = 0
    best_selection: tuple[float, ...] | None = None
    best_epoch: int | None = None
    if args.resume_checkpoint is not None:
        if not bool(checkpoint.get("resume_supported", True)):
            raise ValueError("the selected checkpoint supports initialization only")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_selection = checkpoint.get("best_selection")
        best_selection = (
            None
            if saved_selection is None
            else tuple(float(x) for x in saved_selection)
        )
        best_epoch = checkpoint.get("best_epoch")
        if start_epoch >= args.epochs:
            raise ValueError("resume checkpoint has already reached --epochs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    if log_path.exists() and args.resume_checkpoint is None:
        raise FileExistsError(f"refusing to append to an existing run log: {log_path}")
    if args.resume_checkpoint is None:
        source_snapshot = write_source_snapshot(
            args.output_dir,
            extra_source_files=getattr(args, "source_snapshot_extra_files", ()),
        )
    else:
        snapshot_manifest = args.output_dir / "source_snapshot" / "manifest.json"
        if not snapshot_manifest.is_file():
            raise FileNotFoundError("resume output lacks its source snapshot")
        source_snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        verify_source_snapshot(source_snapshot)
    write_json(args.output_dir / "normalization.json", normalization.to_dict())
    write_json(args.output_dir / "boundary_contract.json", boundary_contract)
    write_json(args.output_dir / "exposure_contract.json", exposure_contract)
    validation_pair_seed = args.seed + 991
    validation_pairs = balanced_presentations(
        store,
        val_keys,
        step_stride=args.step_stride,
        count=args.val_presentations,
        rng=np.random.default_rng(validation_pair_seed),
    )
    split_payload = {
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
        "rollout_selection_seed": rollout_selection_seed,
        "rollout_keys": rollout_keys,
        "parity_rollout_keys": args.parity_rollout_keys,
        "parity_rollout_horizon": args.parity_rollout_horizon,
        "parity_max_rollout_relative_l2": args.parity_max_rollout_relative_l2,
        "parity_max_one_step_relative_l2": args.parity_max_one_step_relative_l2,
        "validation_pair_seed": validation_pair_seed,
        "validation_pairs": [
            {"trajectory": key, "time_index": int(time_index)}
            for key, time_index in validation_pairs
        ],
        "tiny_bank": (
            None
            if tiny_bank is None
            else [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in tiny_bank
            ]
        ),
        "data_manifest_digest": store.manifest_digest,
    }
    write_json(args.output_dir / "split.json", split_payload)
    run_contract = {
        "schema": "pcno_euler2d_serious_training_contract_v1",
        "args": jsonable_args(args),
        "config_digest": digest_mapping(jsonable_args(args)),
        "data_contract": data_contract,
        "boundary_contract": boundary_contract,
        "exposure_contract": exposure_contract,
        "scheduler_contract": scheduler_contract,
        "target_contract": resolved_target_contract,
        "source_snapshot": source_snapshot,
        "initialization_transition": initialization_transition,
        "initialization_control": dict(model.initialization_control),
        "boundary_contract_compatibility": boundary_contract_compatibility,
        "environment": runtime_environment(device),
        "git": git_state(),
    }
    write_json(args.output_dir / "run_contract.json", run_contract)

    if tiny_bank is not None:
        initial_train_metrics = evaluate_pairs(
            model,
            store,
            tiny_bank,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
            primary_objective=args.primary_objective,
            boundary_policies=boundary_policies,
        )
    else:
        initial_train_metrics = None

    parent_checkpoint = None
    if args.init_checkpoint is not None:
        parent_checkpoint = {
            "path": str(args.init_checkpoint),
            "sha256": sha256_file(args.init_checkpoint),
            "epoch": int(checkpoint["epoch"]),
            "source_boundary_mode": str(
                checkpoint.get("boundary_mode", "model_all_nodes")
            ),
            "target_boundary_mode": args.boundary_mode,
        }
    elif args.resume_checkpoint is not None:
        saved_parent = checkpoint.get("parent_checkpoint")
        if saved_parent is not None:
            if not isinstance(saved_parent, Mapping):
                raise ValueError("resume checkpoint parent metadata is malformed")
            parent_checkpoint = dict(saved_parent)
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(args.seed + 101)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp == "fp16"))
    run_start = perf_counter()
    last_train_metrics: dict[str, float | None] | None = None
    last_validation_metrics: dict[str, float] | None = None
    last_rollout_metrics: dict[str, Any] | None = None
    best_tiny_snapshot: dict[str, Any] | None = None
    best_tiny_selection: tuple[float, ...] | None = None
    best_tiny_epoch: int | None = None
    best_tiny_state: dict[str, torch.Tensor] | None = None
    wall_time_stop_reason: str | None = None
    completed_epoch_durations: list[float] = []
    gpu_peak_allocated_bytes = 0
    gpu_peak_reserved_bytes = 0

    for epoch in range(start_epoch, args.epochs):
        if args.max_wall_hours > 0.0 and completed_epoch_durations:
            wall_limit_seconds = args.max_wall_hours * 3600.0
            elapsed_process = perf_counter() - process_start
            conservative_next_epoch = 1.1 * float(
                np.mean(completed_epoch_durations[-3:])
            )
            if elapsed_process + conservative_next_epoch > wall_limit_seconds:
                wall_time_stop_reason = (
                    "stopped_before_epoch_to_preserve_max_wall_hours"
                )
                break
        epoch_start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pairs = epoch_presentations(
            store,
            train_keys,
            args=args,
            epoch=epoch,
            tiny_bank=tiny_bank,
        )
        if len(pairs) != resolved_presentations_per_epoch:
            raise RuntimeError("resolved presentations changed across epochs")
        if (
            args.presentation_mode == FIXED_HORIZON_BLOCKS_PRESENTATION_MODE
            and presentation_stream_sha256(pairs)
            != accounting_presentation_stream_sha256
        ):
            raise RuntimeError("fixed horizon presentation stream changed")
        if (
            homogeneous_optimizer_step_count(
                pairs,
                batch_size=effective_batch_size,
            )
            != optimizer_steps_per_epoch
        ):
            raise RuntimeError("resolved optimizer steps changed across epochs")
        last_train_metrics = train_epoch(
            model,
            store,
            pairs,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_rng=np.random.default_rng(args.seed + 100_000 + epoch),
            device=device,
            amp=args.amp,
            input_noise_std=args.input_noise_std,
            generated_state_exposure_weight=(args.generated_state_exposure_weight),
            multistep_loss_steps=args.multistep_loss_steps,
            multistep_loss_weight=args.multistep_loss_weight,
            multistep_recurrent_input=args.multistep_recurrent_input,
            primary_objective=args.primary_objective,
            boundary_auxiliary=args.boundary_auxiliary,
            boundary_auxiliary_weight=args.boundary_auxiliary_weight,
            near_boundary_hops=args.near_boundary_hops,
            gradient_clip=args.gradient_clip,
            noise_generator=noise_generator,
            boundary_policies=boundary_policies,
        )
        last_validation_metrics = evaluate_pairs(
            model,
            store,
            validation_pairs,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
            primary_objective=args.primary_objective,
            boundary_policies=boundary_policies,
        )
        current_tiny_snapshot = None
        if tiny_bank is not None:
            current_tiny_metrics = evaluate_pairs(
                model,
                store,
                tiny_bank,
                step_stride=args.step_stride,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
                primary_objective=args.primary_objective,
                boundary_policies=boundary_policies,
            )
            current_tiny_snapshot = tiny_fit_snapshot(
                initial_train_metrics,
                current_tiny_metrics,
                relative_l2_threshold=args.tiny_fit_rel_l2,
                loss_ratio_threshold=args.tiny_fit_loss_ratio,
            )
            current_tiny_selection = tiny_fit_selection(current_tiny_snapshot)
            if (
                best_tiny_selection is None
                or current_tiny_selection > best_tiny_selection
            ):
                best_tiny_snapshot = current_tiny_snapshot
                best_tiny_selection = current_tiny_selection
                best_tiny_epoch = epoch
                if bool(current_tiny_snapshot["passed"]):
                    best_tiny_state = clone_state_dict_cpu(model)
        stop_for_tiny_fit = bool(
            args.stop_on_tiny_fit
            and current_tiny_snapshot is not None
            and current_tiny_snapshot["passed"]
        )
        should_rollout = (
            (epoch + 1) % args.rollout_every == 0
            or epoch == start_epoch
            or epoch + 1 == args.epochs
        )
        if should_rollout:
            last_rollout_metrics = evaluate_rollouts(
                model,
                store,
                rollout_keys,
                step_stride=args.step_stride,
                start_frame=args.rollout_start_frame,
                num_steps=args.rollout_steps,
                device=device,
                amp=args.amp,
                boundary_policies=boundary_policies,
                rollout_checkpoints=args.rollout_checkpoints,
                parity_keys=args.parity_rollout_keys,
                parity_horizon=args.parity_rollout_horizon,
            )
            candidate = selection_tuple(
                last_rollout_metrics,
                last_validation_metrics,
                mode=args.selection_mode,
                short_horizon=args.selection_short_horizon,
                parity_max_rollout_relative_l2=(args.parity_max_rollout_relative_l2),
                parity_max_one_step_relative_l2=(args.parity_max_one_step_relative_l2),
            )
            if best_selection is None or candidate > best_selection:
                best_selection = candidate
                best_epoch = epoch
                payload = checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    normalization=normalization,
                    train_keys=train_keys,
                    val_keys=val_keys,
                    test_keys=test_keys,
                    data_contract=data_contract,
                    boundary_contract=boundary_contract,
                    exposure_contract=exposure_contract,
                    source_snapshot=source_snapshot,
                    resolved_target_contract=resolved_target_contract,
                    store=store,
                    args=args,
                    best_selection=best_selection,
                    best_epoch=best_epoch,
                    parent_checkpoint=parent_checkpoint,
                    initialization_transition=initialization_transition,
                )
                atomic_torch_save(payload, args.output_dir / "best.pt")
        epoch_peak_allocated_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        epoch_peak_reserved_bytes = (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        )
        gpu_peak_allocated_bytes = max(
            gpu_peak_allocated_bytes,
            epoch_peak_allocated_bytes,
        )
        gpu_peak_reserved_bytes = max(
            gpu_peak_reserved_bytes,
            epoch_peak_reserved_bytes,
        )
        record = {
            "epoch": epoch,
            "elapsed_seconds": perf_counter() - epoch_start,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": last_train_metrics,
            "validation": last_validation_metrics,
            "rollout": last_rollout_metrics if should_rollout else None,
            "tiny_fit": current_tiny_snapshot,
            "best_epoch": best_epoch,
            "best_selection": best_selection,
            "gpu_max_memory_bytes": epoch_peak_allocated_bytes,
            "gpu_max_reserved_memory_bytes": epoch_peak_reserved_bytes,
        }
        completed_epoch_durations.append(float(record["elapsed_seconds"]))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if (
            (epoch + 1) % args.checkpoint_every == 0
            or epoch + 1 == args.epochs
            or stop_for_tiny_fit
        ):
            payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalization=normalization,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                data_contract=data_contract,
                boundary_contract=boundary_contract,
                exposure_contract=exposure_contract,
                source_snapshot=source_snapshot,
                resolved_target_contract=resolved_target_contract,
                store=store,
                args=args,
                best_selection=best_selection,
                best_epoch=best_epoch,
                parent_checkpoint=parent_checkpoint,
                initialization_transition=initialization_transition,
            )
            atomic_torch_save(payload, args.output_dir / "last.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_rel_l2": last_train_metrics["relative_l2"],
                    "val_rel_l2": last_validation_metrics["relative_l2"],
                    "val_all_rel_l2": last_validation_metrics["all_relative_l2"],
                    "rollout_completion": (
                        None
                        if not should_rollout
                        else last_rollout_metrics["completion_rate"]
                    ),
                    "best_epoch": best_epoch,
                    "tiny_fit_passed": (
                        None
                        if current_tiny_snapshot is None
                        else current_tiny_snapshot["passed"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stop_for_tiny_fit:
            break

    tiny_fit = None
    if tiny_bank is not None:
        final_tiny = evaluate_pairs(
            model,
            store,
            tiny_bank,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
            primary_objective=args.primary_objective,
            boundary_policies=boundary_policies,
        )
        final_tiny_snapshot = tiny_fit_snapshot(
            initial_train_metrics,
            final_tiny,
            relative_l2_threshold=args.tiny_fit_rel_l2,
            loss_ratio_threshold=args.tiny_fit_loss_ratio,
        )
        final_tiny_selection = tiny_fit_selection(final_tiny_snapshot)
        if best_tiny_selection is None or final_tiny_selection > best_tiny_selection:
            best_tiny_snapshot = final_tiny_snapshot
            best_tiny_selection = final_tiny_selection
            best_tiny_epoch = epoch
            if bool(final_tiny_snapshot["passed"]):
                best_tiny_state = clone_state_dict_cpu(model)
        if best_tiny_snapshot is None or best_tiny_epoch is None:
            raise RuntimeError("tiny-fit evaluation did not produce a selection")
        tiny_fit = {
            "initial": initial_train_metrics,
            "final": final_tiny,
            "final_loss_ratio": final_tiny_snapshot["loss_ratio"],
            "final_relative_metric": final_tiny_snapshot["relative_metric"],
            "final_selection_relative_l2": final_tiny_snapshot["selection_relative_l2"],
            "final_passed": final_tiny_snapshot["passed"],
            "best": best_tiny_snapshot["metrics"],
            "best_loss_ratio": best_tiny_snapshot["loss_ratio"],
            "best_relative_metric": best_tiny_snapshot["relative_metric"],
            "best_selection_relative_l2": best_tiny_snapshot["selection_relative_l2"],
            "best_epoch": best_tiny_epoch,
            "relative_l2_threshold": args.tiny_fit_rel_l2,
            "loss_ratio_threshold": args.tiny_fit_loss_ratio,
            "passed": best_tiny_snapshot["passed"],
        }
        if best_tiny_state is not None:
            tiny_payload = checkpoint_payload(
                epoch=best_tiny_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalization=normalization,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                data_contract=data_contract,
                boundary_contract=boundary_contract,
                exposure_contract=exposure_contract,
                source_snapshot=source_snapshot,
                resolved_target_contract=resolved_target_contract,
                store=store,
                args=args,
                best_selection=best_selection,
                best_epoch=best_epoch,
                parent_checkpoint=parent_checkpoint,
                initialization_transition=initialization_transition,
            )
            tiny_payload["model_state"] = best_tiny_state
            tiny_payload.pop("optimizer_state")
            tiny_payload.pop("scheduler_state")
            tiny_payload["checkpoint_role"] = "tiny_fit_selection"
            tiny_payload["resume_supported"] = False
            tiny_payload["tiny_fit"] = tiny_fit
            atomic_torch_save(tiny_payload, args.output_dir / "tiny_best.pt")

    summary = {
        "mode": f"pcno_euler2d_{RESIDUAL_TARGET_KIND}",
        "target_kind": RESIDUAL_TARGET_KIND,
        "target_contract": resolved_target_contract,
        "device": str(device),
        "amp": args.amp,
        "parameter_count": parameter_count(model),
        "model_config": model.model_config(),
        "node_type_channel_control": args.node_type_channel_control,
        "boundary_field_mode": args.boundary_field_mode,
        "boundary_field_names": list(args.boundary_field_names),
        "boundary_residual_mode": args.boundary_residual_mode,
        "boundary_residual_names": list(args.boundary_residual_names),
        "boundary_residual_width": args.boundary_residual_width,
        "initialization_control": dict(model.initialization_control),
        "normalization": normalization.to_dict(),
        "normalization_digest": data_contract["normalization_digest"],
        "data_contract": data_contract,
        "boundary_contract": boundary_contract,
        "exposure_contract": exposure_contract,
        "split_mode": args.split_mode,
        "train_trajectories": len(train_keys),
        "validation_trajectories": len(val_keys),
        "test_trajectories": len(test_keys),
        "step_stride": args.step_stride,
        "batch_size": args.batch_size,
        "microbatch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "boundary_mode": args.boundary_mode,
        "raw_recurrence": args.boundary_mode == "model_all_nodes",
        "autonomous_recurrence": True,
        "inference_interventions": {
            "future_reference_boundary_values": False,
            "clipping": False,
            "primitive_floors": False,
            "limiter": False,
            "smoothing": False,
            "boundary_decode_reencode_closure": (
                args.boundary_mode != "model_all_nodes"
            ),
            "decode_reencode_projection": (args.boundary_mode != "model_all_nodes"),
        },
        "input_noise_std": args.input_noise_std,
        "generated_state_exposure": {
            "weight": args.generated_state_exposure_weight,
            "depth": 1 if args.generated_state_exposure_weight > 0.0 else 0,
            "first_call_gradient": "detached",
            "clean_one_step_anchor": (1.0 - args.generated_state_exposure_weight),
            "raw_generated_state": True,
        },
        "differentiable_multistep": differentiable_multistep_contract,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "scheduler": args.scheduler,
            "scheduler_contract": scheduler_contract,
        },
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "checkpoint_selection_contract": checkpoint_selection_contract(args),
        "parity_selection_contract": {
            "keys": args.parity_rollout_keys,
            "horizon": args.parity_rollout_horizon,
            "max_rollout_relative_l2": args.parity_max_rollout_relative_l2,
            "max_one_step_relative_l2": args.parity_max_one_step_relative_l2,
            "one_step_metric": "fixed_128_presentation_all_node_relative_l2",
            "validation_pair_seed": args.seed + 991,
            "eligibility_requires_all_rollout_trajectories_complete": True,
        },
        "last_train": last_train_metrics,
        "last_validation": last_validation_metrics,
        "last_rollout": last_rollout_metrics,
        "tiny_fit": tiny_fit,
        "elapsed_seconds": perf_counter() - run_start,
        "process_elapsed_seconds": perf_counter() - process_start,
        "requested_epochs": args.epochs,
        "completed_epochs": len(completed_epoch_durations),
        "wall_time_stop_reason": wall_time_stop_reason,
        "actual_presentations": (
            len(completed_epoch_durations) * resolved_presentations_per_epoch
        ),
        "actual_optimizer_steps": (
            len(completed_epoch_durations) * optimizer_steps_per_epoch
        ),
        "gpu_max_memory_bytes": gpu_peak_allocated_bytes,
        "gpu_max_reserved_memory_bytes": gpu_peak_reserved_bytes,
        "data_manifest_digest": store.manifest_digest,
        "config_digest": digest_mapping(jsonable_args(args)),
        "code_sha256": {
            "trainer": sha256_file(Path(__file__)),
            "pcno_euler2d": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_euler2d.py"
            ),
            "pcno_core": sha256_file(ROOT / "pcno/pcno.py"),
            "cpg_mesh_contract": sha256_file(
                ROOT / "utility/time_dependent_no/cpg_mesh_contract.py"
            ),
            "pcno_artifacts": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_artifacts.py"
            ),
            "euler2d_metrics": sha256_file(
                ROOT / "utility/time_dependent_no/euler2d_metrics.py"
            ),
            "pcno_runtime": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_runtime.py"
            ),
            "pcno_rollout": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_rollout.py"
            ),
            "pcno_ripple_diagnostics": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_ripple_diagnostics.py"
            ),
            "evaluator": sha256_file(
                ROOT / "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py"
            ),
        },
        "source_snapshot": source_snapshot,
        "environment": runtime_environment(device),
        "parent_checkpoint": parent_checkpoint,
        "initialization_transition": initialization_transition,
        "boundary_contract_compatibility": boundary_contract_compatibility,
        "git": git_state(),
        "artifacts": {
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "last_checkpoint": str(args.output_dir / "last.pt"),
            "tiny_best_checkpoint": (
                str(args.output_dir / "tiny_best.pt")
                if best_tiny_state is not None
                else None
            ),
            "metrics": str(log_path),
            "run_contract": str(args.output_dir / "run_contract.json"),
            "boundary_contract": str(args.output_dir / "boundary_contract.json"),
            "exposure_contract": str(args.output_dir / "exposure_contract.json"),
            "source_snapshot": str(args.output_dir / "source_snapshot"),
        },
        "artifact_sha256": {
            "best_checkpoint": sha256_file(args.output_dir / "best.pt"),
            "last_checkpoint": sha256_file(args.output_dir / "last.pt"),
            "tiny_best_checkpoint": (
                sha256_file(args.output_dir / "tiny_best.pt")
                if best_tiny_state is not None
                else None
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {"summary": str(args.output_dir / "summary.json"), "tiny_fit": tiny_fit},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
